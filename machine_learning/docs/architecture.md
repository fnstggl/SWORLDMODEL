# Architecture

How a raw dataset becomes a leakage-safe training/eval corpus and, eventually, a fine-tuned
behaviour adapter. Every stage is auditable and re-derivable from committed code + public
source data; bulk artifacts live under `$SWM_DATA_ROOT`, never in the repo.

## The pipeline

```
raw source ──► canonical records ──► split table ──► training/eval views ──► SFT ──► eval
 (acquisition)   (normalization)     (splitting)      (sampling)          (training) (evaluation)
```

1. **Acquisition** (`acquisition/download.py`) — dispatch a source adapter (`hf`/`git`/`http`),
   resumable, checksummed, storage-guarded. Writes a source manifest + a redacted committed copy.
2. **Normalization** (`normalization/pipeline.py`) — a per-dataset `Converter` turns raw rows into
   canonical behaviour-event records, validated against the schema, written as sharded Parquet.
   Malformed records are quarantined with a reason, never dropped silently.
3. **Splitting** (`splitting/policies.py`) — assign each record to a split by a deterministic hash
   of an *isolation unit*. Result is a separate immutable split table (`record_id -> split`), not a
   rewrite of the shards. `splitting/leakage_checks.py` then proves isolation.
4. **Sampling** (`sampling/manifests.py`) — build a named training *view* (e.g. `actor_choice_v1`):
   select training-eligible datasets, restrict to the view's split + tasks, apply anti-dominance
   weights. Emits a full record list (working storage) + a committed summary with a `manifest_hash`.
5. **Training** (`training/train_qlora.py`) — one training loop runs both the CPU LoRA smoke test
   and the 8B 4-bit QLoRA GPU run; only the config differs. Target-only loss, checkpoint/resume,
   adapter export, full `run_manifest.json`.
6. **Evaluation** (`evaluation/`) — non-learned baselines run now on CPU; the model-eval harness
   (built, not launched) scores base vs adapter on held-out test splits with per-task metrics.

The `cli/main.py` entry point (`python -m machine_learning.cli <group> <cmd>`) drives every stage;
imports are lazy per-command so data commands never pull torch.

## The 12-section canonical schema

`schemas/canonical_behavior_event.schema.json` is a thin outer envelope shared by every dataset;
the task-specific label lives in `payload.target` and is validated separately against one of 16
`schemas/task_payloads/<task_type>.schema.json`. Required top-level sections:

| section | role |
|---|---|
| `schema_version` | const `"1.0.0"` |
| `record_id` | deterministic `<dataset>:<task_type>:<hash>` |
| `source` | dataset_id, license_class, converter_version, citation |
| `task_type` | one of the 16 tasks |
| `episode` | interaction/session id — the primary leakage-isolation candidate |
| `cutoff` | `cutoff_time` / `cutoff_sequence_index` + `future_hidden` (must be true) |
| `decision_unit` | actor/population making the decision |
| `context` | pre-cutoff actor_profile, private_state_before, known_history, current_observation, available_actions |
| `payload` | `{input, target}` — nothing outside `payload.target` may reveal the label |
| `causal_metadata` | assignment mechanism / arms / propensity (experimental sets only) |
| `data_quality` | missing/inferred/weak_label fields, chronology_verified, possible_leakage |
| `provenance` | converter, code_commit, transformation_steps, raw_record_locator, content_hash |
| `split_metadata` | split + the isolation keys that decided it (populated by splitting) |

`record_id` and `content_hash` are deterministic and timestamp-independent (see
`reproduction_guide.md`): re-running a converter on the same raw input reproduces the same ids.

## The registry is the spine

`registry/*.yaml` (`datasets.yaml`, `licenses.yaml`, `task_taxonomy.yaml`, `field_mappings.yaml`,
`training_approvals.yaml`) declare, per dataset: source, license, acquire spec, supported tasks,
leakage-isolation `split_unit`, and a **role** (`TRAIN_CANDIDATE` / `*_EVAL_ONLY` /
`INFRASTRUCTURE_ONLY` / `ACCESS_BLOCKED` / …). Nothing downstream treats a dataset as training data
unless the registry role, the license class, **and** a human approval all permit it.

## SWORLDMODEL-adjacent role

This subsystem is fully isolated from the production `swm/` runtime — it never imports `swm` and
`swm` never imports it. The behaviour model does not replace SWORLDMODEL: a general model constructs
a scenario, the SWORLDMODEL runtime enforces constraints, this fine-tuned model predicts what an
actor or population does next, and a calibration layer corrects the distribution. That interface is
a contract only and is **not** wired into the simulator in this task — see
[`integration_contract.md`](integration_contract.md).
