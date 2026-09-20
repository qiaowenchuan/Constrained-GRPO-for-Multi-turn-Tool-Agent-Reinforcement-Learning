# Constrained GRPO for Multi-turn Tool Agents

A reproducible LLM reinforcement learning project for training a multi-turn
after-sales service agent with verifiable task rewards and safety constraints.

The project is built on verl and uses Qwen3-0.6B as the base model. It includes
decision-level supervised fine-tuning, stateful tool interaction, multi-turn
GRPO, and a constrained GRPO extension with an adaptive Lagrange multiplier.


## 1. Problem

The agent interacts with a stateful after-sales environment through four tools:

- `query_order`
- `refund_order`
- `exchange_order`
- `modify_order`

A key safety rule is that the agent must query an order before executing a
business action.

The central question is:

> Can task-only GRPO distinguish between a successful but unsafe trajectory and
> a successful and safe trajectory when both receive the same task reward?


## 2. Training Pipeline

```text
Qwen3-0.6B
    |
    v
Decision-level SFT
    |
    v
Multi-turn Tool Agent
    |
    +--------------------------+
    |                          |
    v                          v
Vanilla GRPO            Constrained GRPO
Task reward only        Task reward + violation cost
```

The supervised dataset contains complete multi-turn trajectories:

```text
system
user
assistant -> query_order
tool -> query result
assistant -> business action
tool -> action result
assistant -> final response
```

Decision-level SFT converts each assistant decision into an individual
supervised sample, so the loss is applied only to the assistant decision tokens.


## 3. Stateful Tool Environment

The main environment implements the following rules:

```text
pending
    refund allowed
    modify allowed
    exchange disallowed

shipped
    refund allowed
    exchange allowed
    modify disallowed
```

All business actions require a successful `query_order` first.

The environment explicitly records state transitions, tool events, and
constraint violations. Task success and safety cost are therefore verified from
environment state instead of being inferred only from generated text.


## 4. Reward and Constraint Cost

The task reward is:

```text
R = 1    requested operation succeeds
R = 0    otherwise
```

The safety cost is:

```text
C = violation_count
```

The reward function reports diagnostics including:

```text
task_success
clean_success
violation_cost
tool_call_cost
has_violation
```

The task reward itself is based only on task success.


## 5. Why a Separate Ablation Environment Is Needed

In the original strict environment, an unsafe action can also cause task
failure. As a result, vanilla GRPO may learn safer behavior indirectly from the
task reward.

To isolate the role of the constraint signal, the formal ablation uses a
permissive environment in which both trajectories below successfully complete
the task.

Safe trajectory:

```text
query_order
modify_order

reward = 1
cost = 0
```

Unsafe trajectory:

```text
modify_order directly

reward = 1
cost = 1
```

Therefore task reward cannot distinguish safe success from unsafe success.


## 6. Constrained GRPO

Vanilla GRPO computes a group-relative reward advantage:

```text
A_R
```

The constrained version separately computes a group-relative cost advantage:

```text
A_C
```

The policy advantage is then:

```text
A = A_R - lambda * A_C
```

The Lagrange multiplier is updated using projected dual ascent:

```text
lambda = max(
    0,
    lambda + lambda_lr * (mean_cost - cost_limit)
)
```

Reward and cost advantages are normalized independently before they are
combined.


## 7. Implementation Details

The project extends verl with the following functionality:

```text
1. Deterministic per-trajectory rollout sampling seeds

2. Deterministic request IDs for multiple rollout sessions

3. Global prompt priority assignment before worker chunking

4. GRPO zero-variance numerical handling

5. Multi-output validation using the final output of each session

6. Extraction of trajectory-level constraint costs

7. Group-relative cost advantages

8. Reward-cost combined policy advantages

9. Adaptive Lagrange multiplier updates
```

The verl modifications are stored in:

```text
verl_after_sales_agent.patch
```

The experiments are based on verl commit `753aed3e1c286ba6825a74342b28669e72c083ea`.

The complete software and hardware environment is documented in
[`ENVIRONMENT.md`](ENVIRONMENT.md).

Detailed experimental results are available in
[`RESULTS.md`](RESULTS.md).

Step-by-step experiment reproduction instructions are available in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).



## 8. Supervised Fine-Tuning

The SFT data generator creates 200 complete trajectories.

```text
160 training trajectories
40 validation trajectories
```

The dataset covers five task types:

```text
refund_pending
refund_shipped
modify_address
modify_quantity
exchange_shipped
```

The final SFT pipeline uses:

```text
make_sft_dataset_v3.py
        |
        v
data_sft_v3/train.parquet
        |
        v
decision_sft_dataset.py
        |
        v
run_sft_v4_lr2e6_sweep.sh
```

The selected initialization checkpoint is the step-4 SFT checkpoint.


## 9. Formal Constraint Ablation

The formal mechanism-isolation benchmark contains:

```text
40 training prompts
40 held-out validation prompts
```

Training and validation use different order IDs.

The SFT initialization has the following deterministic validation behavior:

| Method | Task Success | Clean Success | Violation Cost | Tool Calls |
| --- | ---: | ---: | ---: | ---: |
| SFT baseline | 1.00 | 0.50 | 0.50 | 1.50 |

The RL experiments use three matched random seeds:

```text
123
456
789
```


## 10. Main Results

Final deterministic validation results across three random seeds:

| Method | Task Success | Clean Success | Violation Cost | Tool Calls |
| --- | ---: | ---: | ---: | ---: |
| Vanilla GRPO | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Constrained GRPO | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 2.0000 ± 0.0000 |

Vanilla GRPO preserves perfect task success but converges to direct
modification without querying first.

Once task reward becomes identical across successful trajectories, the
reward-relative advantage cannot distinguish safe success from unsafe success.

Constrained GRPO introduces an independent cost signal and learns the
`query_order -> modify_order` behavior while preserving 100 percent task
success.

The result is reproduced for all three matched random seeds.


## 11. Example Constraint Learning Signal

For the post-cleanup constrained GRPO smoke test:

```text
mixed_cost_group_fraction = 0.8750
violation_cost_mean       = 0.671875
cost_adv_abs_mean         = 0.754379
lambda                    = 0.7500
lambda_next               = 0.8071875
```

The corresponding deterministic validation result is:

```text
task_success   = 1.0
clean_success  = 1.0
violation_cost = 0.0
tool_call_cost = 2.0
```

This confirms that a non-zero group-relative cost advantage reaches the policy
optimization step.


## 12. Project Structure

```text
after_sales_agent_release_v1/
|
|-- README.md
|-- ENVIRONMENT.md
|-- RESULTS.md
|-- REPRODUCIBILITY.md
|-- verl_after_sales_agent.patch
|
`-- projects/
    `-- after_sales_agent/
        |
        |-- make_sft_dataset_v3.py
        |-- decision_sft_dataset.py
        |-- run_sft_v4_lr2e6_sweep.sh
        |
        |-- stateful_tools.py
        |-- stateful_tools_ablation.py
        |-- tool_config.json
        |-- tool_config_ablation.json
        |
        |-- reward_stateful_v2.py
        |
        |-- make_constraint_ablation_formal.py
        |
        |-- run_v4_ablation_baseline_val.sh
        |
        |-- run_v4_ablation_vanilla_5step.sh
        |-- run_v4_ablation_vanilla_seed456_5step.sh
        |-- run_v4_ablation_vanilla_seed789_5step.sh
        |
        |-- run_v4_ablation_constrained_5step.sh
        |-- run_v4_ablation_constrained_seed456_5step.sh
        |-- run_v4_ablation_constrained_seed789_5step.sh
        |
        |-- summarize_ablation_3seeds.py
        |
        |-- data_sft_v3/
        `-- data_v2_constraint_ablation_formal/
```


## 13. Reproduction

### Apply the verl patch

Start from the exact verl revision used in the experiments.

    git checkout 753aed3e1c286ba6825a74342b28669e72c083ea

Verify that the patch applies cleanly.

    git apply --check verl_after_sales_agent.patch

Then apply the patch.

    git apply verl_after_sales_agent.patch

See [`ENVIRONMENT.md`](ENVIRONMENT.md) for the complete software and hardware
environment.

### Generate SFT data

```bash
python projects/after_sales_agent/make_sft_dataset_v3.py
```

### Run decision-level SFT

```bash
bash projects/after_sales_agent/run_sft_v4_lr2e6_sweep.sh
```

### Generate the formal ablation dataset

```bash
python projects/after_sales_agent/make_constraint_ablation_formal.py
```

### Evaluate the SFT baseline

```bash
bash projects/after_sales_agent/run_v4_ablation_baseline_val.sh
```

### Run vanilla GRPO

```bash
bash projects/after_sales_agent/run_v4_ablation_vanilla_5step.sh
bash projects/after_sales_agent/run_v4_ablation_vanilla_seed456_5step.sh
bash projects/after_sales_agent/run_v4_ablation_vanilla_seed789_5step.sh
```

### Run constrained GRPO

```bash
bash projects/after_sales_agent/run_v4_ablation_constrained_5step.sh
bash projects/after_sales_agent/run_v4_ablation_constrained_seed456_5step.sh
bash projects/after_sales_agent/run_v4_ablation_constrained_seed789_5step.sh
```

### Aggregate the three-seed results

```bash
python projects/after_sales_agent/summarize_ablation_3seeds.py
```


## 14. Scope and Limitations

The permissive ablation environment is a mechanism-isolation benchmark. It is
designed specifically to separate task reward from safety cost.

The experiment shows that an independent constraint signal can distinguish
safe success from unsafe success when task reward alone provides no such
distinction.

The result should not be interpreted as evidence that vanilla GRPO fails in
every tool-agent environment.

The current experiments use Qwen3-0.6B and a small controlled benchmark.
Natural next steps include:

```text
larger language models

broader tool-agent task distributions

multiple simultaneous constraint types

checkpoint persistence for the Lagrange multiplier

larger-scale multi-GPU training
```
