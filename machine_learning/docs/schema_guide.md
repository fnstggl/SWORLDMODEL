# Schema guide

The canonical behaviour-event is a thin outer **envelope** shared by every dataset, plus a
task-specific **payload** validated separately. One decision unit = an actor (or population) about to
produce an observed behaviour at a cutoff, together with everything known *before* it.

## The envelope

`schemas/canonical_behavior_event.schema.json` requires all 12 top-level sections
(`additionalProperties: false` — no extra keys). See [`architecture.md`](architecture.md) for the
full section table. Key invariants:

- `schema_version` is the const `"1.0.0"`.
- `record_id` is `"<dataset_id>:<task_type>:<stable-hash>"`, deterministic and reproducible.
- `cutoff.future_hidden` **must be true**. Everything in `context` / `known_history` must be at or
  before `cutoff_time` (and strictly before `cutoff_sequence_index`).
- `payload` is `{input, target}`; nothing outside `payload.target` may reveal the label.
- `context.available_actions` is `null` when the choice set is genuinely unknown — `null != []`.
  Empty list means "no actions available"; null means "we don't know the option set".
- `causal_metadata` is populated only for interventional/experimental datasets, `{}` otherwise.

## The 16 task payloads

Each `task_type` selects one `schemas/task_payloads/<task_type>.schema.json`. The taxonomy lives in
`registry/task_taxonomy.yaml` and is mirrored by `machine_learning/tasks.py` (a test asserts they
agree). The required primary key inside `payload.target` per task:

| task | family | `payload.target` primary key |
|---|---|---|
| `PREDICT_NEXT_CHOICE` | individual_choice | `choice` |
| `PREDICT_NEXT_ACTION` | individual_choice | `action_type` |
| `PREDICT_RESPONSE_OR_NONRESPONSE` | individual_choice | `responded` |
| `PREDICT_NEXT_MESSAGE` | social_conversation | `message_text` |
| `PREDICT_NEXT_SPEAKER` | social_conversation | `speaker_id` |
| `PREDICT_BELIEF_CHANGE` | social_conversation | `belief_after` |
| `PREDICT_PRIVATE_STATE_UPDATE` | social_conversation | `private_state_after` |
| `PREDICT_DISCUSSION_TREE` | social_conversation | `tree` |
| `PREDICT_TIME_TO_ACTION` | long_horizon | `acted` |
| `PREDICT_TRAJECTORY_CONTINUATION` | long_horizon | `continuation` |
| `PREDICT_FINAL_OUTCOME` | long_horizon | `outcome` |
| `PREDICT_POPULATION_RESPONSE` | population_response | `aggregate_metrics` |
| `PREDICT_POPULATION_TIME_SERIES` | population_response | `time_series` |
| `PREDICT_INTERVENTION_EFFECT` | intervention_effect | `treated_outcome` |
| `RANK_CANDIDATE_ACTIONS` | intervention_effect | `chosen_id` |
| `PREDICT_POLICY_VALUE` | intervention_effect | `reward` |

Every task family must have at least one held-out cross-dataset test (see
[`leakage_prevention_guide.md`](leakage_prevention_guide.md)).

## Determinism

`canonical.make_record` guarantees `record_id` and `content_hash` are functions of the *semantic*
content only — they never depend on wall-clock time or run-specific values. `content_hash` excludes
`provenance`, `split_metadata`, and `source.normalization_timestamp`; timestamps live only in
`source.normalization_timestamp` and `provenance.conversion_timestamp`. Re-running a converter on the
same raw input reproduces identical ids. `content_hash` doubles as `split_metadata.dedup_hash`.

## Special tokens

Inactivity, waiting, censoring, and unknown action spaces are first-class outcomes (from
`machine_learning/tasks.py`), never silently dropped:

`<NO_ACTION>` · `<NO_RESPONSE>` · `<WAIT>` · `<UNKNOWN_ACTION_SPACE>` · `<MISSING_TIMESTAMP>` ·
`<EPISODE_END>`

## data_quality flags

Required keys: `missing_fields`, `chronology_verified`, `target_verified`, `possible_leakage`,
`license_verified`. Optional: `inferred_fields` (deterministically derived, e.g. `sequence_index`
computed from ordering — **not** model-generated), `weak_label_fields` (model-generated; must be
empty in default manifests), `warnings`, `confidence` (`high`/`medium`/`low`/`""`). These flags drive
the validation gate and the human-review sampling — a record with `possible_leakage=true` or non-empty
`warnings` is routed into the suspicious set for review.
