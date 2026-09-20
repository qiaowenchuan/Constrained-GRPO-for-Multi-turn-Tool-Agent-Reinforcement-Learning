def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    extra_info = extra_info or {}

    order_id = str(
        extra_info.get("order_id")
        or ground_truth
        or ""
    )

    target_action = extra_info.get("target_action")
    target_field = extra_info.get("target_field")
    target_value = extra_info.get("target_value")

    env = extra_info.get("after_sales_env", {})

    orders = env.get("orders", {})
    events = env.get("events", [])

    order = orders.get(order_id, {})

    queried = any(
        e.get("action") == "query_order"
        and e.get("order_id") == order_id
        and e.get("success") is True
        for e in events
    )

    action_success = any(
        e.get("action") == target_action
        and e.get("order_id") == order_id
        and e.get("success") is True
        for e in events
    )

    task_success = False

    if target_action == "refund_order":
        task_success = (
            action_success
            and order.get("status") == "refunded"
            and order.get("refunded") is True
        )

    elif target_action == "exchange_order":
        task_success = (
            action_success
            and order.get("status") == "exchanged"
            and order.get("exchanged") is True
        )

    elif target_action == "modify_order":
        final_value = order.get(target_field)

        expected_value = target_value

        if target_field == "quantity":
            try:
                expected_value = int(target_value)
            except (TypeError, ValueError):
                expected_value = target_value

        task_success = (
            action_success
            and final_value == expected_value
        )

    violation_count = int(
        env.get("violation_count", 0)
    )

    tool_call_count = int(
        env.get("tool_call_count", 0)
    )

    violation_types = [
        e.get("violation_type")
        for e in events
        if e.get("violation") is True
    ]

    clean_success = (
        task_success
        and violation_count == 0
    )

    # Important:
    # score is ONLY task reward.
    # Constraint costs are returned separately.
    score = float(task_success)

    return {
        "score": score,

        # Reward signal
        "task_success": float(task_success),

        # Useful diagnostics
        "clean_success": float(clean_success),
        "queried": float(queried),
        "target_action_success": float(action_success),

        # Constraint / cost signals
        "violation_cost": float(violation_count),
        "tool_call_cost": float(tool_call_count),

        # Extra diagnostics
        "has_violation": float(
            violation_count > 0
        ),
    }
