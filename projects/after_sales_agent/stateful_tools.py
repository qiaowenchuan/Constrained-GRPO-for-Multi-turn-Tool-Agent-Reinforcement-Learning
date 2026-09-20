import json
from copy import deepcopy
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


ENV_KEY = "after_sales_env"


class OrderToolBase(BaseTool):

    def __init__(self, config: dict, tool_schema=None):
        super().__init__(config, tool_schema)
        self._create_payloads: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id=None, **kwargs):
        instance_id, response = await super().create(
            instance_id=instance_id,
            **kwargs,
        )

        self._create_payloads[instance_id] = deepcopy(
            kwargs.get("create_kwargs", {})
        )

        return instance_id, response

    def _get_env(self, instance_id: str, agent_data):
        if ENV_KEY not in agent_data.extra_fields:
            payload = self._create_payloads.get(
                instance_id,
                {},
            )

            order_list = deepcopy(
                payload.get("orders", [])
            )

            orders = {}

            for item in order_list:
                if not item:
                    continue

                item = deepcopy(item)
                order_id = item.pop("order_id")

                item.setdefault("refunded", False)
                item.setdefault("exchanged", False)
                item.setdefault("address", "default_address")
                item.setdefault("quantity", 1)

                orders[order_id] = item

            agent_data.extra_fields[ENV_KEY] = {
                "orders": orders,
                "events": [],
                "queried_orders": [],
                "tool_call_count": 0,
                "violation_count": 0,
            }

            print(
                f"[ENV_CREATED] request_id={agent_data.request_id}",
                flush=True,
            )

        return agent_data.extra_fields[ENV_KEY]

    def _print_env(self, env):
        print(
            "[ENV_STATE] "
            + json.dumps(
                env,
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _record_violation(
        self,
        env,
        action,
        order_id,
        violation_type,
        message,
    ):
        env["violation_count"] += 1

        event = {
            "action": action,
            "order_id": order_id,
            "success": False,
            "violation": True,
            "violation_type": violation_type,
            "message": message,
        }

        env["events"].append(event)

        return {
            "success": False,
            "order_id": order_id,
            "violation": True,
            "violation_type": violation_type,
            "message": message,
        }

    def _record_success(
        self,
        env,
        action,
        order_id,
        before=None,
        after=None,
        extra=None,
    ):
        event = {
            "action": action,
            "order_id": order_id,
            "success": True,
            "violation": False,
        }

        if before is not None:
            event["before"] = before

        if after is not None:
            event["after"] = after

        if extra:
            event.update(extra)

        env["events"].append(event)

    def _metadata(
        self,
        env,
        action,
        success,
    ):
        return {
            "tool_success": int(success),
            "action": action,
            "tool_call_count": env["tool_call_count"],
            "violation_count": env["violation_count"],
        }

    async def release(self, instance_id: str, **kwargs):
        self._create_payloads.pop(
            instance_id,
            None,
        )


class QueryOrderTool(OrderToolBase):

    def get_openai_tool_schema(self):
        schema = {
            "type": "function",
            "function": {
                "name": "query_order",
                "description": (
                    "Query the current information and status of an order."
                ),
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
        }

        return OpenAIFunctionToolSchema.model_validate(schema)

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ):
        agent_data = kwargs.get("agent_data")

        env = self._get_env(
            instance_id,
            agent_data,
        )

        env["tool_call_count"] += 1

        order_id = parameters.get("order_id")
        order = env["orders"].get(order_id)

        if order is None:
            result = {
                "success": False,
                "order_id": order_id,
                "message": "Order not found",
            }

            env["events"].append(
                {
                    "action": "query_order",
                    "order_id": order_id,
                    "success": False,
                    "violation": False,
                    "message": "Order not found",
                }
            )

        else:
            if order_id not in env["queried_orders"]:
                env["queried_orders"].append(order_id)

            status = order.get("status")

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

            result = {
                "success": True,
                "order_id": order_id,
                **deepcopy(order),
                "available_actions": available_actions,
                "message": (
                    "Order query succeeded. "
                    "Choose the next tool according to the user request "
                    "and the available actions."
                ),
            }

            self._record_success(
                env,
                "query_order",
                order_id,
            )

        self._print_env(env)

        return (
            ToolResponse(
                text=json.dumps(
                    result,
                    ensure_ascii=False,
                )
            ),
            0.0,
            self._metadata(
                env,
                "query_order",
                result["success"],
            ),
        )


class RefundOrderTool(OrderToolBase):

    def get_openai_tool_schema(self):
        schema = {
            "type": "function",
            "function": {
                "name": "refund_order",
                "description": (
                    "Refund an order. "
                    "The order must be queried first."
                ),
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
        }

        return OpenAIFunctionToolSchema.model_validate(schema)

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ):
        agent_data = kwargs.get("agent_data")

        env = self._get_env(
            instance_id,
            agent_data,
        )

        env["tool_call_count"] += 1

        order_id = parameters.get("order_id")
        order = env["orders"].get(order_id)

        if order is None:
            result = self._record_violation(
                env,
                "refund_order",
                order_id,
                "invalid_order_id",
                "Order not found.",
            )

        elif order_id not in env["queried_orders"]:
            result = self._record_violation(
                env,
                "refund_order",
                order_id,
                "refund_before_query",
                "You must query the order before refunding it.",
            )

        elif order.get("status") not in {
            "pending",
            "shipped",
        }:
            status = order.get("status")

            result = self._record_violation(
                env,
                "refund_order",
                order_id,
                f"refund_not_allowed_for_status_{status}",
                f"Refund is not allowed when order status is {status}.",
            )

        else:
            before = deepcopy(order)

            order["status"] = "refunded"
            order["refunded"] = True

            after = deepcopy(order)

            result = {
                "success": True,
                "order_id": order_id,
                **after,
            }

            self._record_success(
                env,
                "refund_order",
                order_id,
                before=before,
                after=after,
            )

        self._print_env(env)

        return (
            ToolResponse(
                text=json.dumps(
                    result,
                    ensure_ascii=False,
                )
            ),
            0.0,
            self._metadata(
                env,
                "refund_order",
                result["success"],
            ),
        )


class ExchangeOrderTool(OrderToolBase):

    def get_openai_tool_schema(self):
        schema = {
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
        }

        return OpenAIFunctionToolSchema.model_validate(schema)

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ):
        agent_data = kwargs.get("agent_data")

        env = self._get_env(
            instance_id,
            agent_data,
        )

        env["tool_call_count"] += 1

        order_id = parameters.get("order_id")
        order = env["orders"].get(order_id)

        if order is None:
            result = self._record_violation(
                env,
                "exchange_order",
                order_id,
                "invalid_order_id",
                "Order not found.",
            )

        elif order_id not in env["queried_orders"]:
            result = self._record_violation(
                env,
                "exchange_order",
                order_id,
                "exchange_before_query",
                "You must query the order before exchanging it.",
            )

        elif order.get("status") != "shipped":
            status = order.get("status")

            result = self._record_violation(
                env,
                "exchange_order",
                order_id,
                f"exchange_not_allowed_for_status_{status}",
                f"Exchange is not allowed when order status is {status}.",
            )

        else:
            before = deepcopy(order)

            order["status"] = "exchanged"
            order["exchanged"] = True

            after = deepcopy(order)

            result = {
                "success": True,
                "order_id": order_id,
                **after,
            }

            self._record_success(
                env,
                "exchange_order",
                order_id,
                before=before,
                after=after,
            )

        self._print_env(env)

        return (
            ToolResponse(
                text=json.dumps(
                    result,
                    ensure_ascii=False,
                )
            ),
            0.0,
            self._metadata(
                env,
                "exchange_order",
                result["success"],
            ),
        )


class ModifyOrderTool(OrderToolBase):

    def get_openai_tool_schema(self):
        schema = {
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
                            "enum": [
                                "address",
                                "quantity",
                            ],
                            "description": (
                                "The order field to modify."
                            ),
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
        }

        return OpenAIFunctionToolSchema.model_validate(schema)

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ):
        agent_data = kwargs.get("agent_data")

        env = self._get_env(
            instance_id,
            agent_data,
        )

        env["tool_call_count"] += 1

        order_id = parameters.get("order_id")
        field = parameters.get("field")
        value = parameters.get("value")

        order = env["orders"].get(order_id)

        if order is None:
            result = self._record_violation(
                env,
                "modify_order",
                order_id,
                "invalid_order_id",
                "Order not found.",
            )

        elif order_id not in env["queried_orders"]:
            result = self._record_violation(
                env,
                "modify_order",
                order_id,
                "modify_before_query",
                "You must query the order before modifying it.",
            )

        elif order.get("status") != "pending":
            status = order.get("status")

            result = self._record_violation(
                env,
                "modify_order",
                order_id,
                f"modify_not_allowed_for_status_{status}",
                f"Modification is not allowed when order status is {status}.",
            )

        elif field not in {
            "address",
            "quantity",
        }:
            result = self._record_violation(
                env,
                "modify_order",
                order_id,
                "invalid_modify_field",
                "Only address and quantity can be modified.",
            )

        else:
            normalized_value = value

            if field == "quantity":
                try:
                    normalized_value = int(value)
                except (TypeError, ValueError):
                    normalized_value = None

                if (
                    normalized_value is None
                    or normalized_value <= 0
                ):
                    result = self._record_violation(
                        env,
                        "modify_order",
                        order_id,
                        "invalid_quantity",
                        "Quantity must be a positive integer.",
                    )

                    self._print_env(env)

                    return (
                        ToolResponse(
                            text=json.dumps(
                                result,
                                ensure_ascii=False,
                            )
                        ),
                        0.0,
                        self._metadata(
                            env,
                            "modify_order",
                            False,
                        ),
                    )

            if (
                field == "address"
                and (
                    value is None
                    or not str(value).strip()
                )
            ):
                result = self._record_violation(
                    env,
                    "modify_order",
                    order_id,
                    "invalid_address",
                    "Address must not be empty.",
                )

            else:
                before = deepcopy(order)

                order[field] = normalized_value

                after = deepcopy(order)

                result = {
                    "success": True,
                    "order_id": order_id,
                    **after,
                }

                self._record_success(
                    env,
                    "modify_order",
                    order_id,
                    before=before,
                    after=after,
                    extra={
                        "field": field,
                        "value": normalized_value,
                    },
                )

        self._print_env(env)

        return (
            ToolResponse(
                text=json.dumps(
                    result,
                    ensure_ascii=False,
                )
            ),
            0.0,
            self._metadata(
                env,
                "modify_order",
                result["success"],
            ),
        )
