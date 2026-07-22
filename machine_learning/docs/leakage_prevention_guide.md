# Leakage prevention guide

Two independent leakage classes are defended against: **temporal** (a record's input revealing its
own future) and **cross-split** (the same unit or content appearing in both train and test). Both are
enforced in code and gated by validation.

## Chronology / cutoff

Converters expose only pre-cutoff information: for a decision at step `k`, `context.known_history` and
`current_observation` contain only events strictly before `k`, and the label lives only in
`payload.target`. `cutoff.future_hidden` must be true and `cutoff_sequence_index` / `cutoff_time` mark
the boundary. `validation/chronology.py` re-checks this on the normalized records; any violation is a
**critical** failure.

## Isolation units → splits

Splits are assigned by a deterministic hash of an *isolation unit* so the same unit always lands in
the same split and no unit straddles the train/eval boundary (`splitting/policies.py`). Each dataset's
`split_unit` (registry) picks the primary unit: participant, platform_user, conversation, session,
group, experiment/study, topic, event, item/product, organization, or time_period. Assignment is
first-match-wins, most to least restrictive:

1. whole-dataset eval-only → `test_cross_dataset` (roles `CROSS_DATASET_EVAL_ONLY` /
   `LICENSE_RESTRICTED_EVAL_ONLY`);
2. future-time holdout → `test_future_time` (latest time fraction, when `cutoff_time` exists);
3. unseen-secondary holdout → `test_unseen_{people|groups|topics|experiments|conditions}`;
4. primary split → `train` / `validation` / `test_in_domain` (default 0.8 / 0.1 / 0.1 by hash bucket).

The result is a separate immutable **split table** (`record_id -> split` + the exact isolation-key
values that decided it), not a rewrite of the normalized shards. Re-running is reproducible.

### The 10 split names

`train`, `validation`, `test_in_domain`, `test_unseen_people`, `test_unseen_groups`,
`test_unseen_topics`, `test_unseen_experiments`, `test_unseen_conditions`, `test_future_time`,
`test_cross_dataset`.

## The leakage checks

`splitting/leakage_checks.py` (run as part of `datasets split` and `datasets validate`) proves the
assignment is trustworthy, not merely assigned:

- **episode isolation** — every record of one episode is in exactly one split;
- **isolation-unit isolation** — no participant/group/topic/… value straddles splits (from the
  recorded `isolation_keys`);
- **cross-split exact-dup** — the same `content_hash` must not appear in a train split and a test
  split.

Any violation sets the report `ok=false` and **blocks** the dataset from a training manifest.
Near-duplicate detection (`validation/deduplication.py`) is advisory: a near-dup rate over 20% is
surfaced as a warning, not a hard block.

## Cross-dataset held-outs

Some whole datasets are held out as `CROSS_DATASET_EVAL_ONLY` — they never enter training and route
entirely to `test_cross_dataset`, giving a genuine transfer test (does the model generalize to a
dataset it never saw?). Every task family has at least one such held-out set. `test_future_time` is
the temporal analogue: predict a period the model was never trained on.

## Running it

```bash
python -m machine_learning.cli datasets split casino      # assign + run leakage checks (exit 1 on leak)
python -m machine_learning.cli datasets validate casino   # full gate incl. chronology + leakage
```

`validate` reports `critical_ok` over schema, chronology, split-isolation leakage, provenance, and
licensing. A `critical_ok=false` dataset cannot be approved for training. Reports land under
`reports/leakage/<id>.json` and `reports/normalization/<id>.validation.json`.
