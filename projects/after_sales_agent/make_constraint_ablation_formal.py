from pathlib import Path
import random

from datasets import Dataset


OUT_DIR = Path(
    "projects/after_sales_agent/"
    "data_v2_constraint_ablation_formal"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """You are an after-sales service agent.

You have four tools:
1. query_order
2. refund_order
3. exchange_order
4. modify_order

Rules:
1. Always query the order before performing a business action.
2. Pending orders may be refunded or modified.
3. Shipped orders may be refunded or exchanged.
4. Follow the user's requested operation.
5. Only reply to the user after the requested operation succeeds."""


TEMPLATES = [
    "Change order {order_id}'s shipping address to new_address.",
    "Please edit the address for {order_id} and use new_address.",
]


def make_row(i, split, template_index):
    if split == "train":
        order_id = f"V2ABTR{i:04d}"
    else:
        order_id = f"V2ABVA{i:04d}"

    template = TEMPLATES[template_index]

    order = {
        "order_id": order_id,
        "status": "pending",
        "amount": float(100 + (i % 80)),
        "refunded": False,
        "exchanged": False,
        "address": "old_address",
        "quantity": 1,
    }

    create_kwargs = {
        "orders": [order]
    }

    extra_info = {
        "split": f"constraint_ablation_{split}",
        "index": i,
        "order_id": order_id,
        "task_type": "modify_address",
        "template_index": template_index,
        "target_action": "modify_order",
        "target_field": "address",
        "target_value": "new_address",
        "need_tools_kwargs": True,
        "tools_kwargs": {
            name: {
                "create_kwargs": create_kwargs,
            }
            for name in [
                "query_order",
                "refund_order",
                "exchange_order",
                "modify_order",
            ]
        },
    }

    return {
        "data_source": "after_sales_agent_v2",
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": template.format(
                    order_id=order_id
                ),
            },
        ],
        "agent_name": "tool_agent",
        "ability": "tool_use",
        "reward_model": {
            "ground_truth": order_id,
        },
        "extra_info": extra_info,
    }


def make_split(split):
    rows = []
    index = 0

    for template_index in range(2):
        for _ in range(20):
            rows.append(
                make_row(
                    index,
                    split,
                    template_index,
                )
            )
            index += 1

    seed = (
        20260920
        if split == "train"
        else 20260921
    )

    rng = random.Random(seed)
    rng.shuffle(rows)

    return rows


train_rows = make_split("train")
val_rows = make_split("validation")


Dataset.from_list(
    train_rows
).to_parquet(
    str(OUT_DIR / "train.parquet")
)

Dataset.from_list(
    val_rows
).to_parquet(
    str(OUT_DIR / "validation.parquet")
)


def summarize(name, rows):
    counts = {}

    for row in rows:
        k = row["extra_info"]["template_index"]
        counts[k] = counts.get(k, 0) + 1

    print(f"\n===== {name} =====")
    print("rows =", len(rows))
    print("template counts =", counts)

    for row in rows[:5]:
        print(
            row["extra_info"]["order_id"],
            "| template",
            row["extra_info"]["template_index"],
            "|",
            row["prompt"][1]["content"],
        )


summarize("train", train_rows)
summarize("validation", val_rows)


train_ids = {
    row["extra_info"]["order_id"]
    for row in train_rows
}

val_ids = {
    row["extra_info"]["order_id"]
    for row in val_rows
}

print(
    "\ntrain/validation overlap =",
    len(train_ids & val_ids),
)
