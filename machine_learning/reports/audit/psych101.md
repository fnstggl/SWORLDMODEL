# Audit — Psych-101 (Centaur)

- **id**: `psych101`  |  **role**: TRAIN_CANDIDATE  |  **status**: CONVERTER_READY_STORAGE_BLOCKED
- **official source**: https://huggingface.co/datasets/marcelbinz/Psych-101
- **paper**: https://arxiv.org/abs/2410.20268
- **license**: Apache-2.0 (`permissive_commercial`) — commercial=yes, derivatives=yes
- **acquisition**: partial (1 raw files, 335530 bytes)

## Normalized data

- examples: **10200**  |  quarantined: 0  |  episodes: 200  |  actors: 122
- task counts: `{'PREDICT_NEXT_CHOICE': 10000, 'PREDICT_TRAJECTORY_CONTINUATION': 200}`
- split sizes: `{'train': 8364, 'test_in_domain': 969, 'validation': 867}`

## Distributions

- inactivity: `{'n_inactive': 0, 'n_action_or_response': 0, 'inactivity_rate': 0.0}`
- action types: `{}`
- outcomes: `{}`
- response-time (s): `{}`
- context length (chars): `{'min': 538.0, 'p25': 1443.0, 'median': 1970.0, 'p75': 2464.0, 'p95': 2570.0, 'max': 2583.0, 'mean': 1883.43, 'n': 2040}`
- missing fields: `{'available_actions': 10200, 'actor_profile': 10200, 'timestamps': 10200}`

## Leakage

- result: `{'dataset_id': 'psych101', 'ok': True, 'n_records': 10200, 'episode_violations': 0, 'unit_violations': 0, 'cross_split_dupes': 0, 'details': {'dataset_id': 'psych101', 'ok': True, 'episode_violations': [], 'unit_violations': [], 'cross_split_dupes': [], 'n_records': 10200, 'notes': []}}`

## Converter assumptions

- each <<...>> span is exactly one human choice/keypress; empty spans (<<>>) are ignored
- everything before a marker is context the participant had already seen

## Known limitations

- at most CAP (~50) PREDICT_NEXT_CHOICE examples emitted per transcript (transcripts can hold 100s of markers); surfaced via data_quality.warning when capped
- PREDICT_TRAJECTORY_CONTINUATION continuation capped to CAP events and may not reach the transcript end
- transcripts with zero <<>> markers are skipped and counted
- option set / available_actions not recovered (varies per experiment; not guessed)

## Unavailable fields (stored null, never fabricated)

- available_actions / option set (not reliably parseable from heterogeneous free text)
- actor_profile / demographics
- per-trial timestamps
- private beliefs/goals
- PREDICT_FINAL_OUTCOME (no single downstream outcome; a transcript is a trial sequence)

## Recommendations

- **training**: Eligible for training pending human approval (training_approvals.yaml).
- **evaluation**: Usable for in-domain evaluation on its own test split.

## 50 rendered examples (human review)

### Example 1 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:69dcafd08fef9a39`
```
TASK: PREDICT_NEXT_CHOICE

ACTOR:
role=participant, id=psych101-participant-fede82fda297

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 2 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:e0a4ad7a176fb924`
```
TASK: PREDICT_NEXT_CHOICE

ACTOR:
role=participant, id=psych101-participant-fede82fda297

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[psych101-participant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 3 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:1671dc754916e657`
```
participant, id=psych101-participant-fede82fda297

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[psych101-participant-fede82fda297] K
[psych101-participant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 4 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:7bd7f7beecc094b1`
```
h101-participant-fede82fda297] K
[psych101-participant-fede82fda297] K
[psych101-participant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 5 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:2dda6b807f64ebc7`
```

[psych101-participant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 6 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:3187a372a11c2fb8`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 7 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:a77bb50881ed693d`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 8 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:66f1708a6a245457`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 9 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:f9d7ba3faef08e83`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 10 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:0c25a74a1b93b7c8`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 11 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:4c8bd651650a029e`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 12 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:01570a00e0a63322`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 13 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:ac6b8f1a43633712`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 14 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:5c58e7aca9ecaf76`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 15 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:d1bff60421f10178`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 16 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:82d141864cc8b7db`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 17 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:1517a5ad2dd2ee93`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 18 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:f38db554f4ad5199`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 19 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:bf1a185a123e3300`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 20 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:1c1f5c974e9dd52b`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 21 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:385919bd440dc14d`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 22 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:7dcd5021aa6d7b5d`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 23 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:d650aa6d6da1b814`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 24 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:66a6811b3f6dc34a`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 25 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:7a6a3576ea1a19f9`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 26 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:14db3258592bc595`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 27 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:261cc5168350ed52`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 28 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:c38340a52f9dd910`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 29 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:df9e6703c7b19ba9`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 30 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:b20320530ad47e2e`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 31 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:4dc802dd90374744`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 32 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:89c50571cf89601e`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 33 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:4b8e938747b83c29`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 34 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:f312a80588493c05`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 35 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:d6d4fa1fa7a73d01`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 36 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:cbe0d60c13003c11`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 37 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:5114f3bb69255565`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 38 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:924f4c7580632aa3`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 39 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:8270621749e1f773`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 40 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:793b8a9656803ad3`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 41 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:72d53859d02a2942`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 42 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:7e8d7506068dcd3a`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 43 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:12f97113b6cea8f7`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 44 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:ab6a7f69e68a490e`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 45 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:ce5ee847c911c622`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 46 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:62741eff11f56404`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 47 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:e853c6f8def9fc0e`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 48 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:95d61b066e0043ef`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "E", "choice_index": null}
```

### Example 49 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:9f882bf4464441a4`
```
icipant-fede82fda297] E

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

### Example 50 — PREDICT_NEXT_CHOICE — `psych101:PREDICT_NEXT_CHOICE:23bae3ac9b32bf5b`
```
icipant-fede82fda297] K

CURRENT OBSERVATION:
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.
You will encounter four different problems with different rules.

You encounter a new problem with a new rule determining which objects belong to each category:
You see a big black square. You press K. The correct category is K.
You see a small black triangle. You press K. The correct category is E.
You see a big white triangle. You press E. The correct category is K.
You see a small white triangle. You press E. The correct category is E.
You see a small white square. You press E. The correct…

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"acted": true, "choice": "K", "choice_index": null}
```

## 25 most-suspicious examples (warnings / possible leakage)

- `psych101:PREDICT_NEXT_CHOICE:69dcafd08fef9a39` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:e0a4ad7a176fb924` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:1671dc754916e657` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:7bd7f7beecc094b1` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:2dda6b807f64ebc7` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:3187a372a11c2fb8` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:a77bb50881ed693d` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:66f1708a6a245457` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:f9d7ba3faef08e83` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:0c25a74a1b93b7c8` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:4c8bd651650a029e` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:01570a00e0a63322` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:ac6b8f1a43633712` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:5c58e7aca9ecaf76` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:d1bff60421f10178` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:82d141864cc8b7db` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:1517a5ad2dd2ee93` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:f38db554f4ad5199` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:bf1a185a123e3300` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:1c1f5c974e9dd52b` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:385919bd440dc14d` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:7dcd5021aa6d7b5d` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:d650aa6d6da1b814` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:66a6811b3f6dc34a` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
- `psych101:PREDICT_NEXT_CHOICE:7a6a3576ea1a19f9` (PREDICT_NEXT_CHOICE): warnings=['transcript has 384 markers; capped to 50 choice-examples'] possible_leakage=False
