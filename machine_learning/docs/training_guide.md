# Training guide

One training loop (`training/train_qlora.py`) runs both the CPU tiny-model smoke test and the real
8B 4-bit QLoRA GPU run — only the config differs. It is dependency-light plain PyTorch with lazy
torch imports.

## Configs

`configs/training/`:

| config | base_model (default) | quantization | data view |
|---|---|---|---|
| `tiny_smoke_test` | distilgpt2 (82M) | none | dataset `casino` |
| `8b_actor_choice` | Qwen/Qwen2.5-7B | 4bit | `actor_choice_v1` |
| `8b_social_interaction` | Qwen/Qwen2.5-7B | 4bit | `social_interaction_v1` |
| `8b_long_horizon` | Qwen/Qwen2.5-7B | 4bit | `long_horizon_behavior_v1` |
| `8b_population_response` | Qwen/Qwen2.5-7B | 4bit | `population_response_v1` |
| `8b_causal_intervention` | Qwen/Qwen2.5-7B | 4bit | `causal_intervention_v1` |
| `8b_unified_multitask` | Qwen/Qwen2.5-7B | 4bit | `unified_behavior_multitask_v1` |

## Configurable base model

The base model is **configurable per config** in `model.base_model` — it is not hard-coded. The
default is `Qwen/Qwen2.5-7B` (Apache-2.0, ~7.6B). Swap in `meta-llama/Llama-3.1-8B` or another
open-weight model by editing `model.base_model` / `model.tokenizer`; the run manifest records exactly
which base + revision was used.

## QLoRA vs CPU-LoRA

The 8B configs use **4-bit QLoRA** (bitsandbytes, CUDA required). On a machine with no CUDA, the model
loader transparently downgrades to full-precision **plain LoRA** — this is the path the CPU smoke test
uses (distilgpt2), needing **no bitsandbytes and no GPU**. The LoRA / target-only-loss / checkpoint /
resume / export code path is byte-for-byte identical between the two; only the LoRA target-module name
and quantization differ.

## Target-only loss

The SFT formatter (`examples/formatters/sft.py`) renders leakage-safe prompt text ending in
`TARGET:\n`, and returns the exact character offset where the target begins. The collator masks every
token before that offset (`IGNORE_INDEX`), so **loss is applied only to the target section** — the
model is never rewarded for reproducing the prompt.

## Checkpoint / resume + run manifest

Checkpoints are atomic and periodic (`save_steps`); `resume=True` (default) picks up the latest
checkpoint in the run dir and restores model/optimizer/scheduler/step. Deterministic seeds
(`train.seed`) are set for random/numpy/torch/cuda. On completion the adapter is exported to
`<run_dir>/adapter`. A full `run_manifest.json` records the base + tokenizer revision, effective
quantization, data-view + data-manifest hash, code commit, seed, planned steps, trainable-parameter
report, package versions, and device — everything needed to reproduce the run.

## Build a manifest first (with human approval)

Training data must come from an approved view. Before launching:

```bash
# 1. Approve datasets in registry/training_approvals.yaml (see licensing_guide.md).
# 2. Build the view manifest (fails-open exclusions are listed with reasons):
python -m machine_learning.cli manifests build actor_choice_v1
python -m machine_learning.cli manifests build actor_choice_v1 --preview   # ignores approval; gated, never used for a real run
```

A view selects training-eligible datasets, restricts to the view's split + tasks, applies
anti-dominance weights, and writes a full record list (working storage) + a committed summary with a
`manifest_hash`. Under `--preview` the record list is marked `pending_approval` and never used for a
real run.

## Launching

```bash
# Dry-run: loads + validates the config, prints base_model/quantization/data, refuses to launch:
python -m machine_learning.cli train run 8b_actor_choice

# Real launch (needs a CUDA GPU + bitsandbytes for the 8B configs):
python -m machine_learning.cli train run 8b_actor_choice --launch
```

Data-source precedence in `train()`: explicit records > the config's `data.view` manifest >
`data.dataset_id` split. Always run the smoke test (see [`../README.md`](../README.md) and
`smoke run`) before a GPU launch. GPU sizing is in [`gpu_setup_guide.md`](gpu_setup_guide.md).
