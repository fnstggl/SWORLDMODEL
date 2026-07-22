# Audit — Open Bandit Dataset (ZOZO)

- **id**: `open_bandit`  |  **role**: TRAIN_CANDIDATE  |  **status**: PENDING
- **official source**: https://research.zozo.com/data.html
- **paper**: https://arxiv.org/abs/2008.07146
- **license**: Apache-2.0 (repo + in-repo sample). Full data: citation-required 'research' terms (commercial not clearly granted). (`permissive_commercial`) — commercial=yes, derivatives=yes
- **acquisition**: acquired (14 raw files, 24288588 bytes)

## Normalized data

- examples: **120000**  |  quarantined: 0  |  episodes: 60000  |  actors: 0
- task counts: `{'PREDICT_POLICY_VALUE': 60000, 'PREDICT_NEXT_ACTION': 60000}`
- split sizes: `{'train': 96230, 'validation': 11822, 'test_in_domain': 11948}`

## Distributions

- inactivity: `{'n_inactive': 0, 'n_action_or_response': 10000, 'inactivity_rate': 0.0}`
- action types: `{'recommend_item': 10000}`
- outcomes: `{}`
- response-time (s): `{}`
- context length (chars): `{'min': 476.0, 'p25': 476.0, 'median': 477.0, 'p75': 477.0, 'p95': 477.0, 'max': 477.0, 'mean': 476.5, 'n': 4000}`
- missing fields: `{'available_action_set': 10000}`

## Leakage

- result: `{'dataset_id': 'open_bandit', 'ok': True, 'n_records': 120000, 'episode_violations': 0, 'unit_violations': 0, 'cross_split_dupes': 0, 'details': {'dataset_id': 'open_bandit', 'ok': True, 'episode_violations': [], 'unit_violations': [], 'cross_split_dupes': [], 'n_records': 120000, 'notes': []}}`

## Converter assumptions

- dir layout {policy}/{campaign}/{campaign}.csv

## Known limitations

- in-repo sample is 10k rows/policy/campaign; full set ~26M rows on research.zozo.com

## Unavailable fields (stored null, never fabricated)

- matched treatment/control outcomes
- user identity across sessions

## Recommendations

- **training**: Eligible for training pending human approval (training_approvals.yaml).
- **evaluation**: Usable for in-domain evaluation on its own test split.

## 50 rendered examples (human review)

### Example 1 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:11b55821a8c533d9`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 2 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:55320cec73853e22`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "79", "position": 2}, "action_type": "recommend_item"}
```

### Example 3 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:7c883acdc8e565c9`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 4 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:849855212eac76a3`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "14", "position": 1}, "action_type": "recommend_item"}
```

### Example 5 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:ca8c1226c2e40189`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 6 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:2f068f821a6349f1`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "18", "position": 2}, "action_type": "recommend_item"}
```

### Example 7 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:6331b962a056f5aa`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 8 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:34f0d1ca95792bab`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "28", "position": 1}, "action_type": "recommend_item"}
```

### Example 9 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:e1bfed14e1f7c81e`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 10 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:fd53ca6b9d783638`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "65", "position": 2}, "action_type": "recommend_item"}
```

### Example 11 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:98346c44feea1fac`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "05b76f5e97e51128862059ac7df9e42a"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 12 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:06df5a42bcd47a04`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "05b76f5e97e51128862059ac7df9e42a"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "63", "position": 2}, "action_type": "recommend_item"}
```

### Example 13 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:5da51e5ad58456a5`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 14 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:f95fb4102762ac87`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "61", "position": 2}, "action_type": "recommend_item"}
```

### Example 15 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:5e42ebb871c41176`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 16 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:29bec939f533bf90`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "52", "position": 2}, "action_type": "recommend_item"}
```

### Example 17 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:4325c071f1b19c17`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "719dab53a7560218a9d1f96b25d6fa32", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 18 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:c2b544ddc05e961f`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "719dab53a7560218a9d1f96b25d6fa32", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "51", "position": 3}, "action_type": "recommend_item"}
```

### Example 19 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:c60a305b96de221f`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 20 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:0087b9db745f8a0b`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "27", "position": 3}, "action_type": "recommend_item"}
```

### Example 21 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:12eb27a841ec841c`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 22 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:43f7c63738b76008`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "61", "position": 3}, "action_type": "recommend_item"}
```

### Example 23 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:02efdbb851fda596`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "9b2d331c329ceb74d3dcfb48d8798c78", "user_feature_3": "f97571b9c14a786aab269f0b427d2a85"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 24 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:461a85d9b4362b0a`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "9b2d331c329ceb74d3dcfb48d8798c78", "user_feature_3": "f97571b9c14a786aab269f0b427d2a85"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "71", "position": 2}, "action_type": "recommend_item"}
```

### Example 25 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:bb98f3b3b32aadfb`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 26 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:4ed6fa5a17980f81`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "79", "position": 3}, "action_type": "recommend_item"}
```

### Example 27 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:7769891d79ffcf02`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 28 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:91655194dc133855`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "7bc94a2da491829b777c49c4b5e480f2", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "60", "position": 3}, "action_type": "recommend_item"}
```

### Example 29 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:8832c8b6325cb145`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "6ff54aa8ff7a9dde75161c20a3ee4231", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 30 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:5ae3cb5df155e939`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "6ff54aa8ff7a9dde75161c20a3ee4231", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "27", "position": 1}, "action_type": "recommend_item"}
```

### Example 31 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:d52c56192cb7698b`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "9b2d331c329ceb74d3dcfb48d8798c78", "user_feature_3": "f97571b9c14a786aab269f0b427d2a85"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 32 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:03c8fdf8ed0a7df3`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "9b2d331c329ceb74d3dcfb48d8798c78", "user_feature_3": "f97571b9c14a786aab269f0b427d2a85"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "24", "position": 1}, "action_type": "recommend_item"}
```

### Example 33 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:3c3bad8444dd699a`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "9b2d331c329ceb74d3dcfb48d8798c78", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 34 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:517c4463a99b52e4`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "9b2d331c329ceb74d3dcfb48d8798c78", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "66", "position": 3}, "action_type": "recommend_item"}
```

### Example 35 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:3ba95d7cab045e6c`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "cef3390ed299c09874189c387777674a", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "06128286bcc64b6a4b0fb7bc0328fe17"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 36 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:4c48c51bdd30c5ef`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "cef3390ed299c09874189c387777674a", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "06128286bcc64b6a4b0fb7bc0328fe17"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "68", "position": 2}, "action_type": "recommend_item"}
```

### Example 37 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:58b77f697075895d`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "719dab53a7560218a9d1f96b25d6fa32", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 38 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:95675565a8c6f327`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "719dab53a7560218a9d1f96b25d6fa32", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "78", "position": 1}, "action_type": "recommend_item"}
```

### Example 39 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:f9dc8289dfe9e178`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "cef3390ed299c09874189c387777674a", "user_feature_1": "f1c2d6a32ec39249160cf784b63f4c6f", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "270b3e1c052b4f2e9c90bf0ebeb84f34"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 40 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:75da7c333da997b2`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "cef3390ed299c09874189c387777674a", "user_feature_1": "f1c2d6a32ec39249160cf784b63f4c6f", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "270b3e1c052b4f2e9c90bf0ebeb84f34"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "61", "position": 3}, "action_type": "recommend_item"}
```

### Example 41 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:c74950589efbb200`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 42 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:0e8a084205b31c66`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 1}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "28", "position": 1}, "action_type": "recommend_item"}
```

### Example 43 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:91c176d37abe758f`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "9b2d331c329ceb74d3dcfb48d8798c78", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 44 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:a73b576e19c0c983`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "9b2d331c329ceb74d3dcfb48d8798c78", "user_feature_3": "9bde591ffaab8d54c457448e4dca6f53"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "59", "position": 3}, "action_type": "recommend_item"}
```

### Example 45 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:cfa8b931aaf388b2`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "719dab53a7560218a9d1f96b25d6fa32", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 46 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:c3f99493762f9cd8`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "2d03db5543b14483e52d761760686b64", "user_feature_2": "719dab53a7560218a9d1f96b25d6fa32", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "41", "position": 3}, "action_type": "recommend_item"}
```

### Example 47 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:4c106167e3b0f0c1`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "05b76f5e97e51128862059ac7df9e42a"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 48 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:12c888fde52857e7`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "c2e4f76cdbabecd33b8c762aeef386b3", "user_feature_3": "05b76f5e97e51128862059ac7df9e42a"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 2}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "52", "position": 2}, "action_type": "recommend_item"}
```

### Example 49 — PREDICT_POLICY_VALUE — `open_bandit:PREDICT_POLICY_VALUE:1254063dc4a81bfa`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 50 — PREDICT_NEXT_ACTION — `open_bandit:PREDICT_NEXT_ACTION:db0ef3cc411512d2`
```
TASK: PREDICT_NEXT_ACTION

ACTOR:
role=population
{"user_feature_0": "81ce123cbb5bd8ce818f60fb3586bba5", "user_feature_1": "03a5648a76832f83c859d46bc06cb64a", "user_feature_2": "2723d2eb8bba04e0362098011fa3997b", "user_feature_3": "c39b0c7dd5d4eb9a18e7db6ba2f258f8"}

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"campaign": "all", "position": 3}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"item_id": "60", "position": 3}, "action_type": "recommend_item"}
```

## 25 most-suspicious examples (warnings / possible leakage)

- `open_bandit:PREDICT_NEXT_ACTION:55320cec73853e22` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:849855212eac76a3` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:2f068f821a6349f1` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:34f0d1ca95792bab` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:fd53ca6b9d783638` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:06df5a42bcd47a04` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:f95fb4102762ac87` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:29bec939f533bf90` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:c2b544ddc05e961f` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:0087b9db745f8a0b` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:43f7c63738b76008` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:461a85d9b4362b0a` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:4ed6fa5a17980f81` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:91655194dc133855` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:5ae3cb5df155e939` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:03c8fdf8ed0a7df3` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:517c4463a99b52e4` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:4c48c51bdd30c5ef` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:95675565a8c6f327` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:75da7c333da997b2` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:0e8a084205b31c66` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:a73b576e19c0c983` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:c3f99493762f9cd8` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:12c888fde52857e7` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
- `open_bandit:PREDICT_NEXT_ACTION:db0ef3cc411512d2` (PREDICT_NEXT_ACTION): warnings=['under bts the action is policy-biased (see causal_metadata)'] possible_leakage=False
