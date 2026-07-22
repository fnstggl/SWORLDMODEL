# SWORLDMODEL behaviour-ML

A self-contained, auditable pipeline for **acquiring, normalizing, validating, splitting,
sampling, and (QLoRA) fine-tuning** a specialized human-behaviour model on real human
choices, actions, messages, belief changes, delays, reactions, trajectories, population
responses, and treatment effects.

This subsystem is **fully isolated** from the production `swm/` runtime: it never imports
`swm`, `swm` never imports it, it has its own `pyproject.toml` + tests, and its bulk data
lives outside the repo under `$SWM_DATA_ROOT`. Nothing here changes existing SWORLDMODEL
behaviour.

> The behaviour model does **not** replace SWORLDMODEL. The intended architecture is:
> a general model researches/constructs a scenario → the SWORLDMODEL runtime enforces
> time / information / authority / feasibility / ordering constraints → **this fine-tuned
> model predicts what an actor or population does next** → a calibration/evaluation layer
> corrects probabilities, timing, frequencies, and distributions. See
> `docs/integration_contract.md`.

## Non-negotiable principles (enforced in code)

- **Never invent missing facts.** Absent source fields become `null` / `[]` / `{}` and are
  listed in `data_quality.missing_fields`. No LLM-filled fields ever enter default
  manifests (a separate `weak_label_fields` namespace exists and must be empty).
- **Protect chronology.** An example's input contains only pre-cutoff information; the
  target and anything after it live only in `payload.target`. Enforced by the canonical
  schema (`cutoff.future_hidden`) + the chronology validator.
- **Preserve source meaning.** A click, a belief rating, a negotiation message, a donation,
  an experimental choice, a population click-rate, a treatment effect, and a discussion-tree
  continuation are *different targets* — one shared outer schema, 16 task-specific payloads.
- **Preserve raw data.** `raw source → canonical records → task examples → split manifests
  → training views`; every example traces back to an exact raw record (`provenance show`).
- **Do not train on all benchmarks.** Every dataset gets an explicit role
  (`TRAIN_CANDIDATE` / `*_EVAL_ONLY` / `INFRASTRUCTURE_ONLY` / `ACCESS_BLOCKED` / …). Some
  are held out for genuine cross-dataset transfer testing.

## Layout

```
machine_learning/
  registry/        datasets.yaml, licenses.yaml, task_taxonomy.yaml, field_mappings.yaml,
                   training_approvals.yaml   (the machine-readable spine)
  schemas/         canonical_behavior_event.schema.json + task_payloads/ + source_manifests/
  acquisition/     download.py, verify.py, source_adapters/ (hf | git | http, resumable)
  normalization/   base.py, pipeline.py, common/, converters/<dataset>.py
  examples/        formatters/sft.py  (target-only-loss SFT rendering)
  splitting/       policies.py, leakage_checks.py  (leakage-safe splits)
  sampling/        balanced_sampler.py, manifests.py  (anti-dominance training views)
  validation/      schema, chronology, deduplication, leakage, distributions,
                   provenance, licensing, orchestrator (critical-failure gating)
  training/        train_qlora.py, collators.py, loss_masking.py, checkpointing.py,
                   resume.py, model_registry.py, smoke.py
  evaluation/      metrics.py, baselines.py, per-family metrics, model_eval.py, reports.py
  cli/             python -m machine_learning.cli ...
  tests/           unit/ + integration/ + fixtures/  (small committed samples)
  reports/         acquisition/ normalization/ leakage/ licenses/ audit/ readiness/ …
  docs/            guides + the SWORLDMODEL integration contract
  data/            working storage (gitignored)  — or point $SWM_DATA_ROOT elsewhere
```

## Quickstart

```bash
export SWM_DATA_ROOT=/path/to/large/volume        # bulk working storage (NOT the repo)
export HF_HOME=/path/to/hf_home
export HF_TOKEN=...                                # for large / gated HF pulls
pip install -r machine_learning/requirements/base.txt -r machine_learning/requirements/data.txt

python -m machine_learning.cli registry verify
python -m machine_learning.cli datasets list
python -m machine_learning.cli datasets acquire casino
python -m machine_learning.cli datasets normalize casino
python -m machine_learning.cli datasets split casino
python -m machine_learning.cli datasets validate casino
python -m machine_learning.cli datasets audit casino
python -m machine_learning.cli datasets prepare-all           # resumable; failures recorded, not fatal
python -m machine_learning.cli eval baselines
python -m machine_learning.cli manifests build social_interaction_v1 --preview
python -m machine_learning.cli readiness check
python -m machine_learning.cli provenance show <record_id>

# Pre-GPU smoke test (CPU, no bitsandbytes):
pip install -r machine_learning/requirements/training.txt
python -m machine_learning.cli smoke run

# Launch an 8B QLoRA fine-tune (needs a CUDA GPU):
python -m machine_learning.cli train run 8b_actor_choice           # dry-run (prints the plan)
python -m machine_learning.cli train run 8b_actor_choice --launch  # real launch
```

## Datasets (23)

12 `TRAIN_CANDIDATE`, 4 held-out `CROSS_DATASET_EVAL_ONLY`, 2 `LICENSE_RESTRICTED_EVAL_ONLY`,
4 `ACCESS_BLOCKED`, 1 `INFRASTRUCTURE_ONLY`. License + access were verified from official
dataset cards / repos / papers on the `last_verified_at` date recorded per entry. See
`registry/datasets.yaml` and `reports/readiness/license_matrix.csv`.

## Status

Run `python -m machine_learning.cli readiness check` and read
`reports/readiness/final_readiness_report.md`. Training data enters a manifest only when
the registry role + license **and** a human approval in `registry/training_approvals.yaml`
both permit it — nothing is approved by default.

See `docs/` for the acquisition, normalization, schema, leakage-prevention, licensing,
training, evaluation, GPU-setup, Colab, persistent-server, troubleshooting, reproduction,
and integration-contract guides.
