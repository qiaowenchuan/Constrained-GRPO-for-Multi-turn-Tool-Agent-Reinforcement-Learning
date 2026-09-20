import json

import torch
import torch.nn.functional as F

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.tokenizer.chat_template import apply_chat_template


class DecisionSFTDataset(MultiTurnSFTDataset):
    """
    Decision-level SFT dataset for tool-using agents.

    Each original multi-turn trajectory is expanded into one training
    sample per assistant decision.

    For each assistant turn:

        history
            -> apply_chat_template(add_generation_prompt=True)

        history + current assistant target
            -> apply_chat_template(add_generation_prompt=False)

    The generation prompt is treated as context and receives loss_mask=0.
    Only the current assistant target receives loss_mask=1.

    This matches the actual autoregressive context used during agent
    rollout, including Qwen3 <think></think> generation-prompt tokens.
    """

    def _read_files_and_process(self):
        # Reuse all normal parquet loading, sampling and metadata handling.
        super()._read_files_and_process()

        # messages/tools are stored as JSON strings in the v3 parquet files
        # so that Arrow cannot merge heterogeneous nested schemas.
        self.messages = [
            json.loads(x) if isinstance(x, str) else x
            for x in self.messages
        ]

        if self.tools is not None:
            self.tools = [
                json.loads(x) if isinstance(x, str) else x
                for x in self.tools
            ]

    def _build_messages(self, example):
        # The dataframe itself still contains the JSON string, so decode it
        # before delegating placeholder/image/video processing to the parent.
        example = dict(example)

        raw_messages = example[self.messages_key]
        if isinstance(raw_messages, str):
            example[self.messages_key] = json.loads(raw_messages)

        return super()._build_messages(example)

    def __init__(
        self,
        parquet_files,
        tokenizer,
        config,
        processor=None,
        max_samples=-1,
    ):
        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

        if self.processor is not None:
            raise NotImplementedError(
                "DecisionSFTDataset currently supports text-only SFT."
            )

        # One entry corresponds to one actual assistant generation decision.
        #
        # Example:
        #   original row
        #       assistant query_order
        #       assistant refund_order
        #       assistant final answer
        #
        # becomes:
        #       (row, assistant_idx_1)
        #       (row, assistant_idx_2)
        #       (row, assistant_idx_3)
        self.decision_index = []

        for row_idx, messages in enumerate(self.messages):
            for message_idx, message in enumerate(messages):
                if (
                    isinstance(message, dict)
                    and message.get("role") == "assistant"
                ):
                    self.decision_index.append(
                        (row_idx, message_idx)
                    )

        print(
            "decision-level dataset len:",
            len(self.decision_index),
        )

    def __len__(self):
        return len(self.decision_index)

    def __getitem__(self, item):
        row_idx, target_message_idx = self.decision_index[item]

        row_dict = self.dataframe.iloc[row_idx].to_dict()
        messages = self._build_messages(row_dict)

        target_message = messages[target_message_idx]

        if target_message.get("role") != "assistant":
            raise RuntimeError(
                "Decision index does not point to an assistant message."
            )

        tools = (
            self.tools[row_idx]
            if self.tools is not None
            else None
        )

        enable_thinking = (
            self.enable_thinking[row_idx]
            if self.enable_thinking is not None
            else self.enable_thinking_default
        )

        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        apply_kwargs = {
            **self.apply_chat_template_kwargs
        }

        if enable_thinking is not None:
            apply_kwargs["enable_thinking"] = enable_thinking

        processor = self.tokenizer

        # --------------------------------------------------------------
        # 1. Build the exact inference-time prefix.
        #
        # This includes Qwen3's generation prompt, e.g.
        #
        # <|im_start|>assistant
        # <think>
        #
        # </think>
        #
        # --------------------------------------------------------------

        history_messages = messages[:target_message_idx]

        prompt_inputs = apply_chat_template(
            processor,
            messages=history_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_kwargs,
        )

        prompt_ids = prompt_inputs["input_ids"][0]

        # --------------------------------------------------------------
        # 2. Serialize the same prefix plus the current assistant target.
        # --------------------------------------------------------------

        target_messages = messages[: target_message_idx + 1]

        full_inputs = apply_chat_template(
            processor,
            messages=target_messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_kwargs,
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]

        # --------------------------------------------------------------
        # 3. The inference prefix MUST be an exact token prefix of the
        #    supervised full sequence.
        # --------------------------------------------------------------

        prompt_len = prompt_ids.shape[0]

        if input_ids.shape[0] <= prompt_len:
            raise AssertionError(
                "Assistant target produced no target tokens."
            )

        if not torch.equal(
            input_ids[:prompt_len],
            prompt_ids,
        ):
            raise AssertionError(
                "Decision SFT prefix mismatch. "
                "The inference-time generation prefix is not an "
                "exact prefix of the supervised sequence."
            )

        # --------------------------------------------------------------
        # 4. Supervise only the current assistant decision.
        #
        # prompt / history / generation prompt -> 0
        # current assistant target             -> 1
        # --------------------------------------------------------------

        loss_mask = torch.zeros_like(attention_mask)
        loss_mask[prompt_len:] = 1

        if int(loss_mask.sum()) == 0:
            raise AssertionError(
                "Decision SFT sample has zero supervised tokens."
            )

        assert (
            input_ids.shape
            == attention_mask.shape
            == loss_mask.shape
        )

        # --------------------------------------------------------------
        # 5. Position ids.
        # --------------------------------------------------------------

        position_ids = torch.arange(
            input_ids.shape[0],
            dtype=torch.long,
        )

        # --------------------------------------------------------------
        # 6. Preserve verl's existing padding/truncation behavior.
        # --------------------------------------------------------------

        sequence_length = input_ids.shape[0]

        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                pad_token_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else 0
                )

                pad_len = self.max_length - sequence_length

                padded_input_ids = torch.full(
                    (pad_len,),
                    pad_token_id,
                    dtype=input_ids.dtype,
                )

                padded_attention_mask = torch.zeros(
                    (pad_len,),
                    dtype=attention_mask.dtype,
                )

                padded_loss_mask = torch.zeros(
                    (pad_len,),
                    dtype=loss_mask.dtype,
                )

                input_ids = torch.cat(
                    (
                        input_ids,
                        padded_input_ids,
                    )
                )

                attention_mask = torch.cat(
                    (
                        attention_mask,
                        padded_attention_mask,
                    )
                )

                loss_mask = torch.cat(
                    (
                        loss_mask,
                        padded_loss_mask,
                    )
                )

                position_ids = F.pad(
                    position_ids,
                    (0, pad_len),
                    value=0,
                )

            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length:]
                    attention_mask = attention_mask[-self.max_length:]
                    loss_mask = loss_mask[-self.max_length:]
                    position_ids = position_ids[-self.max_length:]

                elif self.truncation == "right":
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    loss_mask = loss_mask[:self.max_length]
                    position_ids = position_ids[:self.max_length]

                elif self.truncation == "error":
                    raise ValueError(
                        f"{sequence_length=} is larger than "
                        f"{self.max_length=}"
                    )

                else:
                    raise ValueError(
                        f"Unknown truncation method "
                        f"{self.truncation}"
                    )

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }

        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if (
                sequence_length > self.max_length
                and self.truncation == "error"
            ):
                raise ValueError(
                    f"{sequence_length=} is larger than "
                    f"{self.max_length=}"
                )

            if sequence_length > self.max_length:
                # Keep the same behavior as the current verl
                # MultiTurnSFTDataset no-padding path.
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
                position_ids = position_ids[:self.max_length]

            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }

        else:
            raise ValueError(
                f"Unknown pad mode {self.pad_mode}"
            )
