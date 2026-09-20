# Experimental Results

This document records the main experimental results for the constrained GRPO
multi-turn Tool Agent project.

## 1. Evaluation metrics

The main metrics are:

| Metric | Meaning |
| --- | --- |
| Task success | Whether the requested business operation succeeds |
| Clean success | Task succeeds with zero constraint violations |
| Violation cost | Number of recorded safety violations |
| Tool calls | Number of tool calls used by the trajectory |

The formal ablation evaluates deterministic validation trajectories.


## 2. SFT baseline

Before reinforcement learning, the selected SFT checkpoint produces:

| Method | Task Success | Clean Success | Violation Cost | Tool Calls |
| --- | ---: | ---: | ---: | ---: |
| SFT baseline | 1.00 | 0.50 | 0.50 | 1.50 |

The SFT model can complete all tasks, but half of the validation trajectories
violate the query-before-action constraint.


## 3. Formal ablation design

The mechanism-isolation environment allows both safe and unsafe successful
trajectories.

Safe success:

    query_order
    modify_order

    reward = 1
    cost = 0

Unsafe success:

    modify_order directly

    reward = 1
    cost = 1

Therefore task reward alone cannot distinguish safe success from unsafe
success.


## 4. Vanilla GRPO

Final deterministic validation results:

| Seed | Task Success | Clean Success | Violation Cost | Tool Calls |
| ---: | ---: | ---: | ---: | ---: |
| 123 | 1.00 | 0.00 | 1.00 | 1.00 |
| 456 | 1.00 | 0.00 | 1.00 | 1.00 |
| 789 | 1.00 | 0.00 | 1.00 | 1.00 |

Three-seed summary:

| Metric | Mean ± Std |
| --- | ---: |
| Task success | 1.0000 ± 0.0000 |
| Clean success | 0.0000 ± 0.0000 |
| Violation cost | 1.0000 ± 0.0000 |
| Tool calls | 1.0000 ± 0.0000 |

Vanilla GRPO preserves perfect task success but converges to direct
modification without first querying the order.


## 5. Constrained GRPO

Final deterministic validation results:

| Seed | Task Success | Clean Success | Violation Cost | Tool Calls |
| ---: | ---: | ---: | ---: | ---: |
| 123 | 1.00 | 1.00 | 0.00 | 2.00 |
| 456 | 1.00 | 1.00 | 0.00 | 2.00 |
| 789 | 1.00 | 1.00 | 0.00 | 2.00 |

Three-seed summary:

| Metric | Mean ± Std |
| --- | ---: |
| Task success | 1.0000 ± 0.0000 |
| Clean success | 1.0000 ± 0.0000 |
| Violation cost | 0.0000 ± 0.0000 |
| Tool calls | 2.0000 ± 0.0000 |

Constrained GRPO learns the required query-then-modify behavior while
preserving perfect task success.


## 6. Constraint learning signal

For the post-cleanup constrained GRPO smoke test:

| Metric | Value |
| --- | ---: |
| Mixed cost group fraction | 0.8750 |
| Training violation cost | 0.671875 |
| Absolute cost advantage mean | 0.754379 |
| Lambda before update | 0.7500 |
| Lambda after update | 0.8071875 |

The non-zero cost advantage confirms that the constraint signal enters the
policy optimization rather than affecting only the Lagrange multiplier.


## 7. Main comparison

| Method | Task Success | Clean Success | Violation Cost | Tool Calls |
| --- | ---: | ---: | ---: | ---: |
| SFT baseline | 1.00 | 0.50 | 0.50 | 1.50 |
| Vanilla GRPO | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Constrained GRPO | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 2.0000 ± 0.0000 |

The formal ablation shows that when task reward is identical for safe and
unsafe successful trajectories, task-only GRPO does not provide a safety
learning signal.

Adding an independently verified violation cost allows constrained GRPO to
separate safe success from unsafe success.


## 8. Interpretation boundary

The permissive environment is a mechanism-isolation benchmark.

These experiments support the narrower conclusion that an independent
constraint cost can provide learning signal when task reward alone cannot
distinguish safe and unsafe successful trajectories.

They do not establish that vanilla GRPO fails in every multi-turn Tool Agent
environment.
