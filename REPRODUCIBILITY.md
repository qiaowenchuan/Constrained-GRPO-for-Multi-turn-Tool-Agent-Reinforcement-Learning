# Reproducibility Guide

This document describes the experiment reproduction workflow for the
Constrained GRPO multi-turn Tool Agent project.

## 1. Base verl revision

The reported experiments use the following verl revision

    753aed3e1c286ba6825a74342b28669e72c083ea

Start from this revision before applying the project patch

    git checkout 753aed3e1c286ba6825a74342b28669e72c083ea

Verify the patch

    git apply --check verl_after_sales_agent.patch

Apply the patch

    git apply verl_after_sales_agent.patch


## 2. Environment

The exact experiment environment is recorded in

    ENVIRONMENT.md

The main runtime components are

    Python          3.12.3
    PyTorch         2.11.0+cu130
    Transformers    5.9.0
    Ray             2.55.1
    SGLang          0.5.12
    FlashAttention  2.8.3
    verl            0.10.0.dev0
    CUDA            13.0

The reported experiments used SGLang as the rollout backend.


## 3. Base model

The experiments use

    Qwen3-0.6B

The original local path was

    /root/autodl-tmp/models/Qwen3-0.6B

This path is machine-specific. Update MODEL paths in the shell scripts when
running on another machine.


## 4. Generate SFT data

Run

    python projects/after_sales_agent/make_sft_dataset_v3.py

This produces

    projects/after_sales_agent/data_sft_v3/train.parquet
    projects/after_sales_agent/data_sft_v3/val.parquet

The generated dataset contains

    160 training trajectories
    40 validation trajectories


## 5. Run decision-level SFT

Run

    bash projects/after_sales_agent/run_sft_v4_lr2e6_sweep.sh

The reported experiments use the step-4 SFT checkpoint as the RL
initialization.


## 6. Generate the formal constraint ablation dataset

Run

    python projects/after_sales_agent/make_constraint_ablation_formal.py

This produces

    projects/after_sales_agent/data_v2_constraint_ablation_formal/train.parquet
    projects/after_sales_agent/data_v2_constraint_ablation_formal/validation.parquet

The formal benchmark contains

    40 training prompts
    40 validation prompts

Training and validation use different order IDs.


## 7. Evaluate the SFT baseline

Run

    bash projects/after_sales_agent/run_v4_ablation_baseline_val.sh

Expected deterministic validation behavior

    task_success     1.00
    clean_success    0.50
    violation_cost   0.50
    tool_call_cost   1.50


## 8. Run vanilla GRPO

Seed 123

    bash projects/after_sales_agent/run_v4_ablation_vanilla_5step.sh

Seed 456

    bash projects/after_sales_agent/run_v4_ablation_vanilla_seed456_5step.sh

Seed 789

    bash projects/after_sales_agent/run_v4_ablation_vanilla_seed789_5step.sh

Expected final deterministic validation behavior for all three seeds

    task_success     1.00
    clean_success    0.00
    violation_cost   1.00
    tool_call_cost   1.00


## 9. Run constrained GRPO

Seed 123

    bash projects/after_sales_agent/run_v4_ablation_constrained_5step.sh

Seed 456

    bash projects/after_sales_agent/run_v4_ablation_constrained_seed456_5step.sh

Seed 789

    bash projects/after_sales_agent/run_v4_ablation_constrained_seed789_5step.sh

Expected final deterministic validation behavior for all three seeds

    task_success     1.00
    clean_success    1.00
    violation_cost   0.00
    tool_call_cost   2.00


## 10. Aggregate results

Run

    python projects/after_sales_agent/summarize_ablation_3seeds.py

Expected three-seed summary

Vanilla GRPO

    task_success       1.0000 ± 0.0000
    clean_success      0.0000 ± 0.0000
    violation_cost     1.0000 ± 0.0000
    tool_call_cost     1.0000 ± 0.0000

Constrained GRPO

    task_success       1.0000 ± 0.0000
    clean_success      1.0000 ± 0.0000
    violation_cost     0.0000 ± 0.0000
    tool_call_cost     2.0000 ± 0.0000


## 11. Post-cleanup smoke test

The cleaned constrained GRPO implementation was verified with

    bash projects/after_sales_agent/run_v4_ablation_constrained_cleanup_smoke_1step.sh

The verified run produced a non-zero group-relative cost advantage and the
following deterministic validation result

    task_success     1.00
    clean_success    1.00
    violation_cost   0.00
    tool_call_cost   2.00


## 12. Important path note

The experiment shell scripts were originally executed under

    /root/autodl-tmp/verl

Some scripts therefore contain machine-specific absolute paths.

When reproducing the project on another machine, update

    ROOT
    MODEL
    checkpoint paths

before running the scripts.


## 13. Result interpretation

The formal permissive environment is designed to isolate task reward from
constraint cost.

The experiment demonstrates that an independently verified violation cost can
provide a learning signal when task reward alone cannot distinguish safe and
unsafe successful trajectories.

The result does not imply that vanilla GRPO fails in every Tool Agent
environment.
