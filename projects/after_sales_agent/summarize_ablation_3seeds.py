import re
from pathlib import Path
import numpy as np


ROOT = Path("projects/after_sales_agent")

runs = {
    ("vanilla", 123): ROOT / "v4_ablation_vanilla_permissive_5step.log",
    ("vanilla", 456): ROOT / "v4_ablation_vanilla_seed456_5step.log",
    ("vanilla", 789): ROOT / "v4_ablation_vanilla_seed789_5step.log",
    ("constrained", 123): ROOT / "v4_ablation_constrained_permissive_5step.log",
    ("constrained", 456): ROOT / "v4_ablation_constrained_seed456_5step.log",
    ("constrained", 789): ROOT / "v4_ablation_constrained_seed789_5step.log",
}

metrics = {
    "task_success":
        "val-aux/after_sales_agent_v2/task_success/mean@1",
    "clean_success":
        "val-aux/after_sales_agent_v2/clean_success/mean@1",
    "violation_cost":
        "val-aux/after_sales_agent_v2/violation_cost/mean@1",
    "tool_call_cost":
        "val-aux/after_sales_agent_v2/tool_call_cost/mean@1",
}


def extract_final(path):
    text = path.read_text(errors="ignore")

    lines = [
        line
        for line in text.splitlines()
        if "val-aux/after_sales_agent_v2/task_success/mean@1:" in line
    ]

    if not lines:
        raise RuntimeError(f"No validation line found in {path}")

    line = lines[-1]
    result = {}

    for name, key in metrics.items():
        m = re.search(
            re.escape(key)
            + r":(?:np\.float64\()?([-+0-9.eE]+)",
            line,
        )

        if not m:
            raise RuntimeError(
                f"{name} missing in {path}"
            )

        result[name] = float(m.group(1))

    return result


all_results = {}

for (method, seed), path in runs.items():
    result = extract_final(path)
    all_results[(method, seed)] = result

    print(
        method,
        seed,
        result,
    )


print("\n===== 3-SEED SUMMARY =====")

for method in ["vanilla", "constrained"]:
    print(f"\n{method}")

    for metric in metrics:
        values = np.array([
            all_results[(method, seed)][metric]
            for seed in [123, 456, 789]
        ])

        print(
            f"{metric:18s} "
            f"{values.mean():.4f} ± "
            f"{values.std(ddof=1):.4f}"
        )
