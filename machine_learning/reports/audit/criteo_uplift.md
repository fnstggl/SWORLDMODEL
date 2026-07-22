# Audit — Criteo Uplift Modeling Dataset (v2.1)

- **id**: `criteo_uplift`  |  **role**: CROSS_DATASET_EVAL_ONLY  |  **status**: PENDING
- **official source**: https://huggingface.co/datasets/criteo/criteo-uplift
- **paper**: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- **license**: CC-BY-NC-SA-4.0 (non-commercial) (`cc_by_nc`) — commercial=no, derivatives=yes
- **acquisition**: partial (1 raw files, 35312 bytes)

## Normalized data

- examples: **4004**  |  quarantined: 0  |  episodes: 2004  |  actors: 0
- task counts: `{'PREDICT_INTERVENTION_EFFECT': 2000, 'PREDICT_POLICY_VALUE': 2000, 'PREDICT_POPULATION_RESPONSE': 4}`
- split sizes: `{'test_cross_dataset': 4004}`

## Distributions

- inactivity: `{'n_inactive': 0, 'n_action_or_response': 0, 'inactivity_rate': 0.0}`
- action types: `{}`
- outcomes: `{}`
- response-time (s): `{}`
- context length (chars): `{'min': 221.0, 'p25': 549.0, 'median': 552.0, 'p75': 556.0, 'p95': 556.0, 'max': 559.0, 'mean': 551.53, 'n': 801}`
- missing fields: `{'counterfactual_outcome': 2000, 'propensity': 2000}`

## Leakage

- result: `{'dataset_id': 'criteo_uplift', 'ok': True, 'n_records': 4004, 'episode_violations': 0, 'unit_violations': 0, 'cross_split_dupes': 0, 'details': {'dataset_id': 'criteo_uplift', 'ok': True, 'episode_violations': [], 'unit_violations': [], 'cross_split_dupes': [], 'n_records': 4004, 'notes': []}}`

## Converter assumptions

- column names f0..f11, treatment, exposure, visit, conversion (verified from HF stream)
- treatment=1 is the treated arm, treatment=0 the control/holdout arm

## Known limitations

- streamed sample from offset 0 is entirely treatment=1 (dataset is ~85% treated); control rows appear later in the full ~14M-row file
- features are anonymized doubles with no semantic meaning
- aggregation block boundaries are a normalization choice (BLOCK_SIZE=500), not a source field

## Unavailable fields (stored null, never fabricated)

- counterfactual outcome (only the realized arm is observed; never imputed)
- per-row logging propensity (assignment is randomized; no logged probability)
- timestamps / user identity (rows are independent anonymized impressions)

## Recommendations

- **training**: NOT for training (role=CROSS_DATASET_EVAL_ONLY).
- **evaluation**: Reserved as held-out EVALUATION data (never in training manifests).

## 50 rendered examples (human review)

### Example 1 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:5ce5ebbae402a9dd`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.976428838331028, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 2 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:99a6f8f12db67d1f`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.976428838331028, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 3 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:002e71a5e6d864d9`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.002689495197341, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 4 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:b272897fab3c2df8`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.002689495197341, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 5 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:0e88e459d1c8cced`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.964775175534072, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 6 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:d89e249cfcc4d6db`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.964775175534072, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 7 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:9240d4c81f3dc774`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.002800861653911, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 8 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:a694c4a4db9cf9b2`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.002800861653911, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 9 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:43f9d484cd316bc1`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.037998700075557, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 10 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:cf1e8aa3acd926a2`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.037998700075557, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 11 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:21159a53e21627ca`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.904507024922992, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 12 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:0da5f25440b4d9d3`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.904507024922992, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 13 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:3aaa9fa301d29488`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.783340294205589, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 14 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:176178491420f0a9`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.783340294205589, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 15 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:82509a94d9cfbbb1`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.964527838098572, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 16 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:c141cb15ea535247`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.964527838098572, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 17 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:f0dc47239a3957d0`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.03780876622985, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 18 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:ee5b60c880868adb`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.03780876622985, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 19 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:c299e110038209fe`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.015127634347312, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 20 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:db48f930b4e98120`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.015127634347312, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 21 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:ab41d6405b072091`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.80779063191969, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 22 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:289678000ef33b48`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.80779063191969, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 23 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:d7f61ed300b8bf79`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.808535235071965, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 24 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:83494c3e79079fae`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.808535235071965, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 25 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:e52b471a5e6d53f3`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.82996693973104, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 26 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:1acf3356e554a300`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.82996693973104, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 27 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:2e609b8d026228bd`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.866943550843212, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 28 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:0ec52a846721975d`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.866943550843212, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 29 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:a7e39c872cf0896d`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.002699150409835, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 30 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:13893edfff6f541f`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.002699150409835, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 31 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:7a85c32676673f80`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.991078985379634, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 32 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:0a2d05214decbe06`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.991078985379634, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 33 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:840716f5457d116a`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.964672023629491, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 34 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:41cd24f8789d30de`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.964672023629491, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 35 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:fa4dc1bb6404e6fe`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.784377728348007, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 36 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:43ba9a332a099a0a`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.784377728348007, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 37 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:1bb9466323d05d72`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.048240678342502, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 38 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:790db5ea025a9617`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.048240678342502, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 39 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:e57e36cc6ce20076`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.83046331569276, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 40 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:852d84be80429d84`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.83046331569276, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 41 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:c06ce75eac4503d2`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.027074118890749, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 42 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:cd17bad386fdb3fe`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.027074118890749, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 43 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:7ae48f7b0b1e9fa7`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.037865471385109, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 44 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:c41af1ca8aaf6046`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.037865471385109, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 45 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:602fbcad37cba4d2`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.002886597695381, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 46 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:0d764f59336d86b2`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.002886597695381, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 47 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:20f127f898fa74c6`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.048287654237775, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 48 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:b92859043124ab7e`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 9.048287654237775, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

### Example 49 — PREDICT_INTERVENTION_EFFECT — `criteo_uplift:PREDICT_INTERVENTION_EFFECT:b6000c552ea62f37`
```
TASK: PREDICT_INTERVENTION_EFFECT

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.990539505649195, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"control_outcome": {"arm": "control", "conversion": null, "exposure": null, "observed": false, "visit": null}, "estimated_effect": {"identified": false, "note": "per-unit effect not identifiable from a single realized arm"}, "treated_outcome": {"arm": "treatment", "conversion": 0, "exposure": 0, "observed": true, "visit": 0}}
```

### Example 50 — PREDICT_POLICY_VALUE — `criteo_uplift:PREDICT_POLICY_VALUE:b49c7c7e30f87cf8`
```
TASK: PREDICT_POLICY_VALUE

ACTOR:
role=population

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {"features": {"f0": 12.616364906986496, "f1": 10.059654474774549, "f10": 5.300374864042156, "f11": -0.1686792210005612, "f2": 8.990539505649195, "f3": 4.679881620097284, "f4": 10.280525225748212, "f5": 4.115453421277861, "f6": 0.294442711255606, "f7": 4.833814577796811, "f8": 3.9553959684262416, "f9": 13.190055934673358}}}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"reward": 0, "value": null}
```

## 25 most-suspicious examples (warnings / possible leakage)

- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:5ce5ebbae402a9dd` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:99a6f8f12db67d1f` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:002e71a5e6d864d9` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:b272897fab3c2df8` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:0e88e459d1c8cced` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:d89e249cfcc4d6db` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:9240d4c81f3dc774` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:a694c4a4db9cf9b2` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:43f9d484cd316bc1` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:cf1e8aa3acd926a2` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:21159a53e21627ca` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:0da5f25440b4d9d3` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:3aaa9fa301d29488` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:176178491420f0a9` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:82509a94d9cfbbb1` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:c141cb15ea535247` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:f0dc47239a3957d0` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:ee5b60c880868adb` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:c299e110038209fe` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:db48f930b4e98120` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:ab41d6405b072091` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:289678000ef33b48` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:d7f61ed300b8bf79` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
- `criteo_uplift:PREDICT_POLICY_VALUE:83494c3e79079fae` (PREDICT_POLICY_VALUE): warnings=['assignment randomized; no per-row logged propensity (marginal rate is a population quantity)'] possible_leakage=False
- `criteo_uplift:PREDICT_INTERVENTION_EFFECT:e52b471a5e6d53f3` (PREDICT_INTERVENTION_EFFECT): warnings=['only the realized arm is observed; counterfactual is null, not imputed'] possible_leakage=False
