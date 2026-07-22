# Audit — Persuasion For Good

- **id**: `persuasionforgood`  |  **role**: TRAIN_CANDIDATE  |  **status**: PENDING
- **official source**: https://gitlab.com/ucdavisnlp/persuasionforgood
- **paper**: https://arxiv.org/abs/1906.06725
- **license**: Apache-2.0 (`permissive_commercial`) — commercial=yes, derivatives=yes
- **acquisition**: acquired (60 raw files, 359587454 bytes)

## Normalized data

- examples: **52464**  |  quarantined: 0  |  episodes: 1017  |  actors: 1285
- task counts: `{'PREDICT_NEXT_MESSAGE': 20932, 'PREDICT_RESPONSE_OR_NONRESPONSE': 10600, 'PREDICT_TRAJECTORY_CONTINUATION': 19915, 'PREDICT_FINAL_OUTCOME': 1017}`
- split sizes: `{'validation': 5126, 'train': 40019, 'test_in_domain': 4319}`

## Distributions

- inactivity: `{'n_inactive': 91, 'n_action_or_response': 4039, 'inactivity_rate': 0.0225}`
- action types: `{}`
- outcomes: `{'None': 388}`
- response-time (s): `{}`
- context length (chars): `{'min': 253.0, 'p25': 1182.0, 'median': 1940.0, 'p75': 2754.0, 'p95': 3914.0, 'max': 7332.0, 'mean': 2029.87, 'n': 4000}`
- missing fields: `{'utterance_strategy_annotation': 7991, 'dialogue_act': 7991, 'timestamps': 20000, 'latency_seconds': 4039}`

## Leakage

- result: `{'dataset_id': 'persuasionforgood', 'ok': True, 'n_records': 49464, 'episode_violations': 0, 'unit_violations': 0, 'cross_split_dupes': 0, 'details': {'dataset_id': 'persuasionforgood', 'ok': True, 'episode_violations': [], 'unit_violations': [], 'cross_split_dupes': [], 'n_records': 49464, 'notes': []}}`

## Converter assumptions

- role code 0=persuader, 1=persuadee (per FullData/readme.md)
- charity is 'Save the Children' (dataset task design)

## Known limitations

- FullData carries no utterance-level persuasion-strategy labels
- donation B6 is a single private post-task self-report; a null value means the participant did not report a donation and is flagged, never imputed

## Unavailable fields (stored null, never fabricated)

- per-utterance strategy annotation (FullData has none; only the 300-dialogue AnnotatedData subset)
- intended donation B5 (annotated only in AnnotatedData)
- per-utterance timestamps / reply latency
- RANK_CANDIDATE_ACTIONS: no candidate-action set exists in the source

## Recommendations

- **training**: Eligible for training pending human approval (training_approvals.yaml).
- **evaluation**: Usable for in-domain evaluation on its own test split.

## 50 rendered examples (human review)

### Example 1 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:498b6de9e7f18d82`
```
ics": {"age.x": "34.0", "edu.x": "Less than four-year college", "employment.x": "Employed for wages", "ideology.x": "Liberal", "income.x": "5.0", "marital.x": "Unmarried", "race.x": "White", "religion.x": "Other religion", "sex.x": "Male"}, "moral_foundations": {"authority.x": 4.4, "care.x": 3.4, "fairness.x": 5.0, "loyalty.x": 4.333333333, "purity.x": 4.0}, "schwartz_values": {"achievement.x": 4.0, "benevolence.x": 4.0, "conform.x": 4.0, "freedom.x": 5.0, "hedonism.x": 4.0, "power.x": 4.0, "security.x": 4.0, "self_direction.x": 4.0, "stimulation.x": 4.0, "tradition.x": 4.0, "universalism.x": 4.0}}

PRIVATE STATE BEFORE:
{"charity": "Save the Children", "goal": "persuade the persuadee to donate to Save the Children", "role": "persuader"}

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {}, "text": null}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Good morning. How are you doing today?
```

### Example 2 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:a68692d84aedb737`
```
ional.x": 3.5}, "demographics": {"age.x": "50.0", "edu.x": "Less than four-year college", "employment.x": "Employed for wages", "ideology.x": "Conservative", "income.x": "10.0", "marital.x": "Married", "race.x": "White", "religion.x": "Protestant", "sex.x": "Female"}, "moral_foundations": {"authority.x": 4.4, "care.x": 4.0, "fairness.x": 3.0, "loyalty.x": 2.333333333, "purity.x": 4.666666667}, "schwartz_values": {"achievement.x": 2.0, "benevolence.x": 6.0, "conform.x": 6.0, "freedom.x": 1.0, "hedonism.x": 3.0, "power.x": 2.0, "security.x": 6.0, "self_direction.x": 2.0, "stimulation.x": 2.0, "tradition.x": 6.0, "universalism.x": 6.0}}

PRIVATE STATE BEFORE:
{"role": "persuadee"}

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?

CURRENT OBSERVATION:
Good morning. How are you doing today?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Hi. I am doing good. How about you?
```

### Example 3 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:51ee02b45c50ec6f`
```
"5.0", "marital.x": "Unmarried", "race.x": "White", "religion.x": "Other religion", "sex.x": "Male"}, "moral_foundations": {"authority.x": 4.4, "care.x": 3.4, "fairness.x": 5.0, "loyalty.x": 4.333333333, "purity.x": 4.0}, "schwartz_values": {"achievement.x": 4.0, "benevolence.x": 4.0, "conform.x": 4.0, "freedom.x": 5.0, "hedonism.x": 4.0, "power.x": 4.0, "security.x": 4.0, "self_direction.x": 4.0, "stimulation.x": 4.0, "tradition.x": 4.0, "universalism.x": 4.0}}

PRIVATE STATE BEFORE:
{"charity": "Save the Children", "goal": "persuade the persuadee to donate to Save the Children", "role": "persuader"}

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?

CURRENT OBSERVATION:
Hi. I am doing good. How about you?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I'm doing pretty good for a Tuesday morning. 
```

### Example 4 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:b67c22c1592876f6`
```
marital.x": "Married", "race.x": "White", "religion.x": "Protestant", "sex.x": "Female"}, "moral_foundations": {"authority.x": 4.4, "care.x": 4.0, "fairness.x": 3.0, "loyalty.x": 2.333333333, "purity.x": 4.666666667}, "schwartz_values": {"achievement.x": 2.0, "benevolence.x": 6.0, "conform.x": 6.0, "freedom.x": 1.0, "hedonism.x": 3.0, "power.x": 2.0, "security.x": 6.0, "self_direction.x": 2.0, "stimulation.x": 2.0, "tradition.x": 6.0, "universalism.x": 6.0}}

PRIVATE STATE BEFORE:
{"role": "persuadee"}

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 

CURRENT OBSERVATION:
I'm doing pretty good for a Tuesday morning. 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Haha. Same here, but it really feels like a Monday.
```

### Example 5 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:8c9a7240d6d264bc`
```
"purity.x": 4.0}, "schwartz_values": {"achievement.x": 4.0, "benevolence.x": 4.0, "conform.x": 4.0, "freedom.x": 5.0, "hedonism.x": 4.0, "power.x": 4.0, "security.x": 4.0, "self_direction.x": 4.0, "stimulation.x": 4.0, "tradition.x": 4.0, "universalism.x": 4.0}}

PRIVATE STATE BEFORE:
{"charity": "Save the Children", "goal": "persuade the persuadee to donate to Save the Children", "role": "persuader"}

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.

CURRENT OBSERVATION:
Haha. Same here, but it really feels like a Monday.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Ugh yes it does!
```

### Example 6 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:a98b9d3f38050702`
```
, "care.x": 4.0, "fairness.x": 3.0, "loyalty.x": 2.333333333, "purity.x": 4.666666667}, "schwartz_values": {"achievement.x": 2.0, "benevolence.x": 6.0, "conform.x": 6.0, "freedom.x": 1.0, "hedonism.x": 3.0, "power.x": 2.0, "security.x": 6.0, "self_direction.x": 2.0, "stimulation.x": 2.0, "tradition.x": 6.0, "universalism.x": 6.0}}

PRIVATE STATE BEFORE:
{"role": "persuadee"}

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!

CURRENT OBSERVATION:
Ugh yes it does!

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I can not believe how warm it is already.
```

### Example 7 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:56b56acc861eab47`
```
ower.x": 4.0, "security.x": 4.0, "self_direction.x": 4.0, "stimulation.x": 4.0, "tradition.x": 4.0, "universalism.x": 4.0}}

PRIVATE STATE BEFORE:
{"charity": "Save the Children", "goal": "persuade the persuadee to donate to Save the Children", "role": "persuader"}

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.

CURRENT OBSERVATION:
I can not believe how warm it is already.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Where are you from? 
```

### Example 8 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:2dc163dc63eb3ab2`
```
s": {"achievement.x": 2.0, "benevolence.x": 6.0, "conform.x": 6.0, "freedom.x": 1.0, "hedonism.x": 3.0, "power.x": 2.0, "security.x": 6.0, "self_direction.x": 2.0, "stimulation.x": 2.0, "tradition.x": 6.0, "universalism.x": 6.0}}

PRIVATE STATE BEFORE:
{"role": "persuadee"}

KNOWN HISTORY:
... (1 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 

CURRENT OBSERVATION:
Where are you from? 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I am from the Midwest. What about you?
```

### Example 9 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:c49f2f103130899f`
```
, "security.x": 4.0, "self_direction.x": 4.0, "stimulation.x": 4.0, "tradition.x": 4.0, "universalism.x": 4.0}}

PRIVATE STATE BEFORE:
{"charity": "Save the Children", "goal": "persuade the persuadee to donate to Save the Children", "role": "persuader"}

KNOWN HISTORY:
... (2 earlier events elided)
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?

CURRENT OBSERVATION:
I am from the Midwest. What about you?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I'm from the South East. It's always warm here. 
```

### Example 10 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:819c3efd751378da`
```
lence.x": 6.0, "conform.x": 6.0, "freedom.x": 1.0, "hedonism.x": 3.0, "power.x": 2.0, "security.x": 6.0, "self_direction.x": 2.0, "stimulation.x": 2.0, "tradition.x": 6.0, "universalism.x": 6.0}}

PRIVATE STATE BEFORE:
{"role": "persuadee"}

KNOWN HISTORY:
... (3 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 

CURRENT OBSERVATION:
I'm from the South East. It's always warm here. 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
```

### Example 11 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:dc99f67dfe72a069`
```
versalism.x": 4.0}}

PRIVATE STATE BEFORE:
{"charity": "Save the Children", "goal": "persuade the persuadee to donate to Save the Children", "role": "persuader"}

KNOWN HISTORY:
... (4 earlier events elided)
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.

CURRENT OBSERVATION:
Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
We're about to get hit by a tropical storm.
```

### Example 12 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:4ace1269f8b41c7d`
```
.x": 3.0, "power.x": 2.0, "security.x": 6.0, "self_direction.x": 2.0, "stimulation.x": 2.0, "tradition.x": 6.0, "universalism.x": 6.0}}

PRIVATE STATE BEFORE:
{"role": "persuadee"}

KNOWN HISTORY:
... (5 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.

CURRENT OBSERVATION:
We're about to get hit by a tropical storm.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I heard that some bad weather was going to be coming. I hope it is not too severe.
```

### Example 13 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:2dc59567a510c767`
```
the Children", "goal": "persuade the persuadee to donate to Save the Children", "role": "persuader"}

KNOWN HISTORY:
... (6 earlier events elided)
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.

CURRENT OBSERVATION:
I heard that some bad weather was going to be coming. I hope it is not too severe.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Me too. It's just part of living on the Gulf. You have to be prepared for it.
```

### Example 14 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:760a2d699cd8fc03`
```
0}}

PRIVATE STATE BEFORE:
{"role": "persuadee"}

KNOWN HISTORY:
... (7 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.

CURRENT OBSERVATION:
Me too. It's just part of living on the Gulf. You have to be prepared for it.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yes, I am sure you get a lot of storms.
```

### Example 15 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:50fbe3488a376370`
```
"goal": "persuade the persuadee to donate to Save the Children", "role": "persuader"}

KNOWN HISTORY:
... (8 earlier events elided)
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.

CURRENT OBSERVATION:
Yes, I am sure you get a lot of storms.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
```

### Example 16 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:5aa31bb0cd6c6d24`
```
icipant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?

CURRENT OBSERVATION:
We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I have heard about them. What do you like about them?
```

### Example 17 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:334c7a68a15dcc13`
```
e Children", "role": "persuader"}

KNOWN HISTORY:
... (10 earlier events elided)
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?

CURRENT OBSERVATION:
I have heard about them. What do you like about them?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 
```

### Example 18 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:9b5ce7a720b1f02b`
```
ipant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?
[persuasionforgood-participant-1f2414366e30] I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 

CURRENT OBSERVATION:
I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yes, I also like what they do. They are a great organization.
```

### Example 19 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:f5ab9ee9d47df370`
```
nt-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?
[persuasionforgood-participant-1f2414366e30] I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 
[persuasionforgood-participant-74a7c7dccf8d] Yes, I also like what they do. They are a great organization.

CURRENT OBSERVATION:
Yes, I also like what they do. They are a great organization.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I'm planning on donating most of my earnings today. Would you like to donate as well?
```

### Example 20 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:c69e17b586ffa0cd`
```
you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?
[persuasionforgood-participant-1f2414366e30] I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 
[persuasionforgood-participant-74a7c7dccf8d] Yes, I also like what they do. They are a great organization.
[persuasionforgood-participant-1f2414366e30] I'm planning on donating most of my earnings today. Would you like to donate as well?

CURRENT OBSERVATION:
I'm planning on donating most of my earnings today. Would you like to donate as well?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I would like to dotate $0.20. Would that help?
```

### Example 21 — PREDICT_NEXT_MESSAGE — `persuasionforgood:PREDICT_NEXT_MESSAGE:3ade70416b04c70d`
```
nt-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?
[persuasionforgood-participant-1f2414366e30] I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 
[persuasionforgood-participant-74a7c7dccf8d] Yes, I also like what they do. They are a great organization.
[persuasionforgood-participant-1f2414366e30] I'm planning on donating most of my earnings today. Would you like to donate as well?
[persuasionforgood-participant-74a7c7dccf8d] I would like to dotate $0.20. Would that help?

CURRENT OBSERVATION:
I would like to dotate $0.20. Would that help?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yes it would. Any little bit helps. Thank you for your donation!
```

### Example 22 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:e8bbeaf256651b9e`
```
TASK: PREDICT_RESPONSE_OR_NONRESPONSE

ACTOR:
role=persuadee, id=persuasionforgood-participant-74a7c7dccf8d

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?

CURRENT OBSERVATION:
Good morning. How are you doing today?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 23 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:6e8757abebd20cdb`
```
TASK: PREDICT_RESPONSE_OR_NONRESPONSE

ACTOR:
role=persuadee, id=persuasionforgood-participant-74a7c7dccf8d

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 

CURRENT OBSERVATION:
I'm doing pretty good for a Tuesday morning. 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 24 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:aa3a16f87aa887ac`
```
TASK: PREDICT_RESPONSE_OR_NONRESPONSE

ACTOR:
role=persuadee, id=persuasionforgood-participant-74a7c7dccf8d

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!

CURRENT OBSERVATION:
Ugh yes it does!

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 25 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:c42f8dc8b5b29fd1`
```
TASK: PREDICT_RESPONSE_OR_NONRESPONSE

ACTOR:
role=persuadee, id=persuasionforgood-participant-74a7c7dccf8d

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (1 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 

CURRENT OBSERVATION:
Where are you from? 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 26 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:c847444394dca2f5`
```
TASK: PREDICT_RESPONSE_OR_NONRESPONSE

ACTOR:
role=persuadee, id=persuasionforgood-participant-74a7c7dccf8d

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (3 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 

CURRENT OBSERVATION:
I'm from the South East. It's always warm here. 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 27 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:025dd0b01ad8e71b`
```
TASK: PREDICT_RESPONSE_OR_NONRESPONSE

ACTOR:
role=persuadee, id=persuasionforgood-participant-74a7c7dccf8d

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (5 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.

CURRENT OBSERVATION:
We're about to get hit by a tropical storm.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 28 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:543c4de864c1ce7a`
```
7c7dccf8d

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (7 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.

CURRENT OBSERVATION:
Me too. It's just part of living on the Gulf. You have to be prepared for it.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 29 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:2c4e53a92131c950`
```
icipant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?

CURRENT OBSERVATION:
We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 30 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:a0215ecfd1455c41`
```
ipant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?
[persuasionforgood-participant-1f2414366e30] I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 

CURRENT OBSERVATION:
I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 31 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:194cb6a18c5887af`
```
you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?
[persuasionforgood-participant-1f2414366e30] I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 
[persuasionforgood-participant-74a7c7dccf8d] Yes, I also like what they do. They are a great organization.
[persuasionforgood-participant-1f2414366e30] I'm planning on donating most of my earnings today. Would you like to donate as well?

CURRENT OBSERVATION:
I'm planning on donating most of my earnings today. Would you like to donate as well?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": true}
```

### Example 32 — PREDICT_RESPONSE_OR_NONRESPONSE — `persuasionforgood:PREDICT_RESPONSE_OR_NONRESPONSE:75c5894bc4e46bfc`
```
ts elided)
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?
[persuasionforgood-participant-1f2414366e30] I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 
[persuasionforgood-participant-74a7c7dccf8d] Yes, I also like what they do. They are a great organization.
[persuasionforgood-participant-1f2414366e30] I'm planning on donating most of my earnings today. Would you like to donate as well?
[persuasionforgood-participant-74a7c7dccf8d] I would like to dotate $0.20. Would that help?
[persuasionforgood-participant-1f2414366e30] Yes it would. Any little bit helps. Thank you for your donation!

CURRENT OBSERVATION:
Yes it would. Any little bit helps. Thank you for your donation!

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"latency_seconds": null, "responded": false}
```

### Example 33 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:9ca7166230d1ff7c`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 1, "kind": "message", "meta": {"role": "persuadee", "turn": 0.0}, "t": "<MISSING_TIMESTAMP>", "text": "Hi. I am doing good. How about you?"}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 2, "kind": "messag
```

### Example 34 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:2194b34c9949c068`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 2, "kind": "message", "meta": {"role": "persuader", "turn": 1.0}, "t": "<MISSING_TIMESTAMP>", "text": "I'm doing pretty good for a Tuesday morning. "}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 3, "kind
```

### Example 35 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:53caaf0336fa2681`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 3, "kind": "message", "meta": {"role": "persuadee", "turn": 1.0}, "t": "<MISSING_TIMESTAMP>", "text": "Haha. Same here, but it really feels like a Monday."}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 4,
```

### Example 36 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:f4e045b2160977b3`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 4, "kind": "message", "meta": {"role": "persuader", "turn": 2.0}, "t": "<MISSING_TIMESTAMP>", "text": "Ugh yes it does!"}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 5, "kind": "message", "meta": {"role"
```

### Example 37 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:d8f10ce529d27f4d`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 5, "kind": "message", "meta": {"role": "persuadee", "turn": 2.0}, "t": "<MISSING_TIMESTAMP>", "text": "I can not believe how warm it is already."}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 6, "kind": "
```

### Example 38 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:4fbdbfdc2bc34bc6`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[persuasionforgood-participant-1f2414366e30] Good morning. How are you doing today?
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 6, "kind": "message", "meta": {"role": "persuader", "turn": 3.0}, "t": "<MISSING_TIMESTAMP>", "text": "Where are you from? "}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 7, "kind": "message", "meta": {"r
```

### Example 39 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:034e522e97cc57c2`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (1 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] Hi. I am doing good. How about you?
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 7, "kind": "message", "meta": {"role": "persuadee", "turn": 3.0}, "t": "<MISSING_TIMESTAMP>", "text": "I am from the Midwest. What about you?"}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 8, "kind": "mes
```

### Example 40 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:071662c94cafdb86`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (2 earlier events elided)
[persuasionforgood-participant-1f2414366e30] I'm doing pretty good for a Tuesday morning. 
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 8, "kind": "message", "meta": {"role": "persuader", "turn": 4.0}, "t": "<MISSING_TIMESTAMP>", "text": "I'm from the South East. It's always warm here. "}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 9, "k
```

### Example 41 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:ed0341331834977c`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (3 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] Haha. Same here, but it really feels like a Monday.
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 9, "kind": "message", "meta": {"role": "persuadee", "turn": 4.0}, "t": "<MISSING_TIMESTAMP>", "text": "Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned."}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-
```

### Example 42 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:25de4d6c8439b998`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (4 earlier events elided)
[persuasionforgood-participant-1f2414366e30] Ugh yes it does!
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 10, "kind": "message", "meta": {"role": "persuader", "turn": 5.0}, "t": "<MISSING_TIMESTAMP>", "text": "We're about to get hit by a tropical storm."}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 11, "kind
```

### Example 43 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:4f683faffe042631`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (5 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] I can not believe how warm it is already.
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 11, "kind": "message", "meta": {"role": "persuadee", "turn": 5.0}, "t": "<MISSING_TIMESTAMP>", "text": "I heard that some bad weather was going to be coming. I hope it is not too severe."}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-partic
```

### Example 44 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:a38aa4b5e478fda3`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (6 earlier events elided)
[persuasionforgood-participant-1f2414366e30] Where are you from? 
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 12, "kind": "message", "meta": {"role": "persuader", "turn": 6.0}, "t": "<MISSING_TIMESTAMP>", "text": "Me too. It's just part of living on the Gulf. You have to be prepared for it."}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant
```

### Example 45 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:c45e02dff24bd92d`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (7 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] I am from the Midwest. What about you?
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 13, "kind": "message", "meta": {"role": "persuadee", "turn": 6.0}, "t": "<MISSING_TIMESTAMP>", "text": "Yes, I am sure you get a lot of storms."}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 14, "kind": "
```

### Example 46 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:3b0e82d2410c2921`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (8 earlier events elided)
[persuasionforgood-participant-1f2414366e30] I'm from the South East. It's always warm here. 
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 14, "kind": "message", "meta": {"role": "persuader", "turn": 7.0}, "t": "<MISSING_TIMESTAMP>", "text": "We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?"}, {"action_content": {}, "action_type": null
```

### Example 47 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:4d28b5913c00d935`
```
RIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (9 earlier events elided)
[persuasionforgood-participant-74a7c7dccf8d] Oh, yep. You are definitely in for warm weather, which is great as far as I am concerned.
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 15, "kind": "message", "meta": {"role": "persuadee", "turn": 7.0}, "t": "<MISSING_TIMESTAMP>", "text": "I have heard about them. What do you like about them?"}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index":
```

### Example 48 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:3d609683d2965c67`
```
NTINUATION

ACTOR:
role=dialogue

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (10 earlier events elided)
[persuasionforgood-participant-1f2414366e30] We're about to get hit by a tropical storm.
[persuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 16, "kind": "message", "meta": {"role": "persuader", "turn": 8.0}, "t": "<MISSING_TIMESTAMP>", "text": "I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. "}
```

### Example 49 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:6273a0d9bbc36ac1`
```
rsuasionforgood-participant-74a7c7dccf8d] I heard that some bad weather was going to be coming. I hope it is not too severe.
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?
[persuasionforgood-participant-1f2414366e30] I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-74a7c7dccf8d", "index": 17, "kind": "message", "meta": {"role": "persuadee", "turn": 8.0}, "t": "<MISSING_TIMESTAMP>", "text": "Yes, I also like what they do. They are a great organization."}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", 
```

### Example 50 — PREDICT_TRAJECTORY_CONTINUATION — `persuasionforgood:PREDICT_TRAJECTORY_CONTINUATION:b4591a7792b7cf91`
```
er events elided)
[persuasionforgood-participant-1f2414366e30] Me too. It's just part of living on the Gulf. You have to be prepared for it.
[persuasionforgood-participant-74a7c7dccf8d] Yes, I am sure you get a lot of storms.
[persuasionforgood-participant-1f2414366e30] We do. I guess I should get into what this chat is supposed to be about. Have you heard of the Charity Save The Children?
[persuasionforgood-participant-74a7c7dccf8d] I have heard about them. What do you like about them?
[persuasionforgood-participant-1f2414366e30] I like that they're committed to helping children in need. They're very transparent in their work and do great things to help children in underprivileged countries. 
[persuasionforgood-participant-74a7c7dccf8d] Yes, I also like what they do. They are a great organization.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-participant-1f2414366e30", "index": 18, "kind": "message", "meta": {"role": "persuader", "turn": 9.0}, "t": "<MISSING_TIMESTAMP>", "text": "I'm planning on donating most of my earnings today. Would you like to donate as well?"}, {"action_content": {}, "action_type": null, "actor_id": "persuasionforgood-par
```

## 25 most-suspicious examples (warnings / possible leakage)

- `persuasionforgood:PREDICT_FINAL_OUTCOME:4682917a7297aa2c` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:5decee8f111989be` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:5ee46000b26e542b` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:e6e17105c081c8f5` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:8a98df93d68fee6e` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:ef4d24e87e1a5111` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:6c7e27c414038540` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:106d35b98262dce6` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:9769baa02b9af7d6` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:0a90221afcf563cc` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:b2745212d50cfed5` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:a3dba4fc1a3490a1` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:29f241141c43d2ec` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:3263adeecaee0b65` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:429fb21b0fb53f58` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:7e89ee0b358fca78` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:aafba5adceeca022` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:a9f8b01774d4c4f8` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:fdd96df6aa291ffe` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:43dfcd0760d2cb42` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:00b4a5de249b04a4` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:620f1d05cc2177ec` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:b480aa35e522cabf` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:0f6511b2151a944b` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
- `persuasionforgood:PREDICT_FINAL_OUTCOME:4a0b0b24f89971a6` (PREDICT_FINAL_OUTCOME): warnings=['donation is a private post-task self-report (never appears in dialogue)'] possible_leakage=False
