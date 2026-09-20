import json
from pathlib import Path

from datasets import Dataset


OUT_DIR = Path("projects/after_sales_agent/data_sft_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """
You are an after-sales service agent.

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
5. Only reply to the user after the requested operation succeeds.
""".strip()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "Query the current information and status of an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to query.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "Refund an order. The order must be queried first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to refund.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exchange_order",
            "description": (
                "Exchange a shipped order. "
                "The order must be queried first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to exchange.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_order",
            "description": (
                "Modify a pending order. "
                "The order must be queried first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to modify.",
                    },
                    "field": {
                        "type": "string",
                        "enum": ["address", "quantity"],
                        "description": "The order field to modify.",
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "The new value. "
                            "Quantity should be a positive integer."
                        ),
                    },
                },
                "required": [
                    "order_id",
                    "field",
                    "value",
                ],
            },
        },
    },
]


TASKS = [
    {
        "task_type": "refund_pending",
        "status": "pending",
        "action": "refund_order",
        "user_template": "Please refund order {order_id}.",
    },
    {
        "task_type": "refund_shipped",
        "status": "shipped",
        "action": "refund_order",
        "user_template": "I want a refund for order {order_id}.",
    },
    {
        "task_type": "modify_address",
        "status": "pending",
        "action": "modify_order",
        "field": "address",
        "value": "new_address",
        "user_template": (
            "Please change the delivery address "
            "of order {order_id} to new_address."
        ),
    },
    {
        "task_type": "modify_quantity",
        "status": "pending",
        "action": "modify_order",
        "field": "quantity",
        "value": "2",
        "user_template": (
            "Please change the quantity of "
            "order {order_id} to 2."
        ),
    },
    {
        "task_type": "exchange_shipped",
        "status": "shipped",
        "action": "exchange_order",
        "user_template": "Please exchange order {order_id}.",
    },
]


def assistant_tool_call(name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        ],
    }


def tool_message(payload):
    return {
        "role": "tool",
        "content": json.dumps(
            payload,
            ensure_ascii=False,
        ),
    }


def make_query_response(order):
    status = order["status"]

    if status == "pending":
        available_actions = [
            "refund_order",
            "modify_order",
        ]
    elif status == "shipped":
        available_actions = [
            "refund_order",
            "exchange_order",
        ]
    else:
        available_actions = []

    return {
        "success": True,
        "order_id": order["order_id"],
        **order,
        "available_actions": available_actions,
        "message": (
            "Order query succeeded. "
            "Choose the next tool according to the user request "
            "and the available actions."
        ),
    }


def make_business_response(task, order):
    after = dict(order)

    if task["action"] == "refund_order":
        after["status"] = "refunded"
        after["refunded"] = True

    elif task["action"] == "exchange_order":
        after["status"] = "exchanged"
        after["exchanged"] = True

    elif task["action"] == "modify_order":
        field = task["field"]

        if field == "quantity":
            after[field] = int(task["value"])
        else:
            after[field] = task["value"]

    return {
        "success": True,
        "order_id": order["order_id"],
        **after,
    }


def make_action_arguments(task, order_id):
    if task["action"] == "modify_order":
        return {
            "order_id": order_id,
            "field": task["field"],
            "value": task["value"],
        }

    return {
        "order_id": order_id,
    }


def make_final_answer(task, order_id):
    if task["action"] == "refund_order":
        return f"Order {order_id} has been successfully refunded."

    if task["action"] == "exchange_order":
        return f"Order {order_id} has been successfully exchanged."

    if task["field"] == "address":
        return (
            f"The delivery address for order {order_id} "
            "has been successfully updated to new_address."
        )

    return (
        f"The quantity for order {order_id} "
        "has been successfully updated to 2."
    )


def make_row(i):
    task = TASKS[i % len(TASKS)]
    order_id = f"SFTV2{i:04d}"

    order = {
        "order_id": order_id,
        "status": task["status"],
        "amount": float(100 + (i % 50)),
        "refunded": False,
        "exchanged": False,
        "address": "old_address",
        "quantity": 1,
    }

    query_response = make_query_response(order)

    action_arguments = make_action_arguments(
        task,
        order_id,
    )

    business_response = make_business_response(
        task,
        order,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": task["user_template"].format(
                order_id=order_id
            ),
        },

        assistant_tool_call(
            "query_order",
            {"order_id": order_id},
        ),

        tool_message(query_response),

        assistant_tool_call(
            task["action"],
            action_arguments,
        ),

        tool_message(business_response),

        {
            "role": "assistant",
            "content": make_final_answer(
                task,
                order_id,
            ),
        },
    ]

    return {
        "messages": json.dumps(
            messages,
            ensure_ascii=False,
        ),
        "tools": json.dumps(
            TOOLS,
            ensure_ascii=False,
        ),
        "enable_thinking": False,
        "task_type": task["task_type"],
    }


rows = [
    make_row(i)
    for i in range(200)
]

dataset = Dataset.from_list(rows)

train_ds = dataset.select(
    range(160)
)

val_ds = dataset.select(
    range(160, 200)
)

train_path = OUT_DIR / "train.parquet"
val_path = OUT_DIR / "val.parquet"

train_ds.to_parquet(
    str(train_path)
)

val_ds.to_parquet(
    str(val_path)
)

print("train =", train_path)
print("train rows =", len(train_ds))
print("val =", val_path)
print("val rows =", len(val_ds))

print("\ntrain task distribution")
for task in TASKS:
    name = task["task_type"]
    count = sum(
        row["task_type"] == name
        for row in rows[:160]
    )
    print(name, count)

print("\nval task distribution")
for task in TASKS:
    name = task["task_type"]
    count = sum(
        row["task_type"] == name
        for row in rows[160:]
    )
    print(name, count)
