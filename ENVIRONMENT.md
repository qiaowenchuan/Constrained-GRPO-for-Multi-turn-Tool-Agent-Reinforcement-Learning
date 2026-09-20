# Environment

This file records the environment used for the reported experiments.

## verl revision

Commit:

753aed3e1c286ba6825a74342b28669e72c083ea

verl version:

0.10.0.dev0

The project-specific verl modifications are stored in:

verl_after_sales_agent.patch

The patch was verified against the clean verl revision above using:

    git apply --check verl_after_sales_agent.patch


## Python

Python 3.12.3


## Core packages

| Package | Version |
| --- | --- |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.9.0 |
| Ray | 2.55.1 |
| SGLang | 0.5.12 |
| FlashAttention | 2.8.3 |
| verl | 0.10.0.dev0 |

vLLM was not installed in the experiment environment.

The reported Agent RL experiments used SGLang as the rollout backend.


## CUDA and GPU

| Item | Value |
| --- | --- |
| PyTorch CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 4090 D |
| Compute capability | 8.9 |
| GPU memory | 24564 MiB |
| NVIDIA driver | 580.105.08 |


## Base model

Qwen3-0.6B

The local experiment path was:

    /root/autodl-tmp/models/Qwen3-0.6B

This path is machine-specific and does not need to be reproduced exactly.


## Notes

The versions above describe the environment used for the reported experiments.
They are not intended to define independently verified minimum version
requirements.

The environment was managed with uv.

The constrained GRPO experiments used SGLang rollout rather than vLLM.
