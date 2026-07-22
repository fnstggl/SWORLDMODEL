# Normalization guide

Normalization turns one dataset's raw rows into canonical behaviour-event records: schema-validated,
deduped, and written as sharded Parquet. The unit of work is a **converter** — one Python class per
dataset under `normalization/converters/<dataset>.py`.

## Writing a converter

Subclass `normalization/base.py:Converter`, set the identity fields, declare a `DOC`, and implement
`iter_records`. Build every record with `self.make(...)` (which wires provenance + determinism via
`canonical.make_record`). Register the class in `registry/datasets.yaml` under the dataset's
`converter:` key.

```python
from ..base import Converter as BaseConverter

class Converter(BaseConverter):
    DATASET_ID = "mydataset"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "mydataset"      # optional: tests/fixtures/<name> for the fixture self-test
    DOC = {                            # validated vs schemas/source_manifests/converter_doc.schema.json
        "dataset_id": "mydataset",
        "original_fields": [...],      # what each raw field means
        "canonical_mapping": [...],    # source_field -> canonical_path
        "tasks_produced": ["PREDICT_NEXT_ACTION"],
        "unavailable_fields": [...],   # things the source does NOT provide
        "chronology_rules": "for a decision at step k, expose only steps 0..k-1",
        "split_key": "conversation (episode_id)",
        "leakage_risks": [...],
        "license_implications": "...",
    }

    def iter_records(self, raw_dir):
        for i, row in enumerate(_load(raw_dir)):
            yield self.make(
                task_type="PREDICT_NEXT_ACTION",
                payload={"input": {...}, "target": {"action_type": ...}},
                episode_id=f"mydataset-{i}",
                sequence_index=k, cutoff_sequence_index=k,
                actor_id=self.pseudonym("participant", raw_actor_id),
                context={"known_history": hist, "current_observation": obs,
                         "available_actions": None},
                raw_locator={"files": ["data/train.parquet"], "indices": [i], "ids": [...]},
                transformation_steps=["load row", f"cutoff before step {k}", "extract target"],
                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True,
                              "inferred_fields": ["sequence_index"]},
            )
```

See `normalization/converters/casino.py` for a complete, real example (CaSiNo negotiation dialogues
producing `PREDICT_NEXT_MESSAGE` / `PREDICT_NEXT_ACTION` / `PREDICT_FINAL_OUTCOME`).

## The non-negotiable rules

- **No fabrication.** If the source did not record a field, leave it `null`/`[]`/`{}` and list its
  name in `data_quality.missing_fields`. Never invent a value. Model-generated fields would go in the
  separate `weak_label_fields` namespace, which **must be empty** in default training manifests.
- **Protect chronology.** For a decision at step `k`, `context.known_history` /
  `current_observation` may contain only events strictly before `k`; the label lives only in
  `payload.target`. Set `cutoff_sequence_index` (and/or `cutoff_time`). `future_hidden` is always
  true — a record with it false is a leakage bug, not data.
- **Preserve source meaning.** Pick the task type that matches what actually happened; don't coerce a
  message into a choice. `casino.py` deliberately does *not* emit `PREDICT_PRIVATE_STATE_UPDATE`
  because CaSiNo only measures satisfaction post-hoc — there's no pre-measurement to predict a change
  from, so inventing one would break the no-fabrication rule.
- **Preserve lineage.** Every record's `raw_locator` must point back to the exact raw record(s).

## Quarantine + dedup

`normalize()` validates every emitted record. Ones that fail schema validation are **quarantined
with a reason** to `$SWM_DATA_ROOT/state/<id>/quarantine.jsonl` and counted — never silently dropped.
Exact duplicates (same deterministic `record_id`) are collapsed and counted. If the quarantine rate
exceeds 20% of emitted records, the report raises a **converter-bug warning**: investigate before
trusting the output.

## Storage model

Normalized data is **sharded Parquet** (`$SWM_DATA_ROOT/normalized/<id>/`), each record stored as a
JSON blob plus flat index columns, written 5000 records/shard and resumable. A small human-review
sample (`reports/human_review/<id>.sample.jsonl`) and a most-suspicious set
(`<id>.suspicious.jsonl`) are committed for the audit. JSONL is used only for those small samples —
never as the bulk store.

## Running it

```bash
python -m machine_learning.cli datasets normalize casino
python -m machine_learning.cli datasets normalize omnibehavior --limit 5000   # sample a huge set
```

The report prints emitted/valid/quarantined/duplicate counts, shard count, task + missing-field
distributions, and any warnings. A normalization manifest is written under `state/<id>/` and a copy
under `reports/normalization/<id>.json`.
