#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${VERL_ROOT:?Set VERL_ROOT to the local verl checkout}"
ROOT="$(cd "$VERL_ROOT" && pwd)"

PROJECT_DIR="$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-$ROOT/checkpoints/after_sales_agent/tool_use_sft_v4_lr2e6_sweep/global_step_4/huggingface}"

TRAIN_FILE="$PROJECT_DIR/data_v2_constraint_ablation_formal/train.parquet"
VAL_FILE="$PROJECT_DIR/data_v2_constraint_ablation_formal/validation.parquet"
TOOL_CONFIG="$PROJECT_DIR/tool_config_ablation.json"

REWARD_PATH="$PROJECT_DIR/reward_stateful_v2.py"

export PYTHONPATH="$RELEASE_ROOT:${PYTHONPATH:-}"


cd "$ROOT"

source .venv/bin/activate


# SGLang subprocess needs CUDA 13 runtime.
export LD_LIBRARY_PATH=$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}


# Clean old Ray processes.
ray stop --force >/dev/null 2>&1 || true


python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.seed=123 \
    actor_rollout_ref.actor.fsdp_config.seed=123 \
    actor_rollout_ref.actor.data_loader_seed=123 \
    actor_rollout_ref.rollout.seed=123 \
    actor_rollout_ref.actor.fsdp_config.full_determinism=true \
    actor_rollout_ref.rollout.full_determinism=true \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.enable_deterministic_inference=true \
    \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    \
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers=1 \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.1 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen \
    \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.resume_mode=disable \
    trainer.val_only=True \
    trainer.log_val_generations=2 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=1 \
    "trainer.logger=['console']" \
    trainer.project_name=after_sales_agent \
    trainer.experiment_name=v4_ablation_baseline_val
