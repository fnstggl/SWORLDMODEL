# Audit — CraigslistBargain

- **id**: `craigslistbargain`  |  **role**: CROSS_DATASET_EVAL_ONLY  |  **status**: PENDING
- **official source**: https://huggingface.co/datasets/stanfordnlp/craigslist_bargains
- **paper**: https://arxiv.org/abs/1808.09637
- **license**: Dataset: no explicit license (HF card 'More Information Needed'). Code (cocoa): MIT. (`unknown_unstated`) — commercial=unknown, derivatives=unknown
- **acquisition**: acquired (2 raw files, 22435777 bytes)

## Normalized data

- examples: **106922**  |  quarantined: 0  |  episodes: 5844  |  actors: 11688
- task counts: `{'PREDICT_NEXT_MESSAGE': 43104, 'PREDICT_NEXT_ACTION': 10357, 'PREDICT_TRAJECTORY_CONTINUATION': 47617, 'PREDICT_FINAL_OUTCOME': 5844}`
- split sizes: `{'test_cross_dataset': 103922}`

## Distributions

- inactivity: `{'n_inactive': 0, 'n_action_or_response': 1972, 'inactivity_rate': 0.0}`
- action types: `{'offer': 937, 'accept': 788, 'quit': 161, 'reject': 86}`
- outcomes: `{'None': 1068}`
- response-time (s): `{}`
- context length (chars): `{'min': 242.0, 'p25': 500.0, 'median': 808.0, 'p75': 1148.0, 'p95': 1709.0, 'max': 3952.0, 'mean': 879.46, 'n': 4000}`
- missing fields: `{'bottomline': 8033, 'final_price': 150, 'offer_price': 3}`

## Leakage

- result: `{'dataset_id': 'craigslistbargain', 'ok': True, 'n_records': 103922, 'episode_violations': 0, 'unit_violations': 0, 'cross_split_dupes': 0, 'details': {'dataset_id': 'craigslistbargain', 'ok': True, 'episode_violations': [], 'unit_violations': [], 'cross_split_dupes': [], 'n_records': 103922, 'notes': []}}`

## Converter assumptions

- kbs[a] corresponds to events with agent==a
- an 'accept' event denotes a reached agreement

## Known limitations

- no stated dataset license — cross-dataset EVAL ONLY (see license_implications)
- reward is the source's scalar payoff; agreement is derived from the presence of an accept event

## Unavailable fields (stored null, never fabricated)

- stable cross-conversation agent id (agents are only indexed 0/1 per dialogue)
- Bottomline is null in the source for essentially all agents

## Recommendations

- **training**: NOT for training (role=CROSS_DATASET_EVAL_ONLY).
- **evaluation**: Reserved as held-out EVALUATION data (never in training manifests).

## 50 rendered examples (human review)

### Example 1 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:51e39556adfc8c02`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=buyer, id=craigslistbargain-participant-7e1b9380862e
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 7.0}

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {}, "text": null}

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
```

### Example 2 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:eb291f1a1b925ce4`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=seller, id=craigslistbargain-participant-ac1721e4734c
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 10.0}

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?

CURRENT OBSERVATION:
Hi, not sure if the charger would work for my car. Can you sell it to me for $5?

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
It will work, i have never seen a car without a cigarette lighter port.\
```

### Example 3 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:68237e9b1e151e64`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=buyer, id=craigslistbargain-participant-7e1b9380862e
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 7.0}

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\

CURRENT OBSERVATION:
It will work, i have never seen a car without a cigarette lighter port.\

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Still, can I buy it for $5? I'm on a tight budge
```

### Example 4 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:209e02a62863fc5f`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=seller, id=craigslistbargain-participant-ac1721e4734c
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 10.0}

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge

CURRENT OBSERVATION:
Still, can I buy it for $5? I'm on a tight budge

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
I think the lowest I would want to go is 8. 
```

### Example 5 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:cc66768f03ff58f1`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=buyer, id=craigslistbargain-participant-7e1b9380862e
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 7.0}

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 

CURRENT OBSERVATION:
I think the lowest I would want to go is 8. 

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
How about $6 and I pick it up myself? It'll save you shipping to me.
```

### Example 6 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:728e07c3b9df6dad`
```
CTOR:
role=seller, id=craigslistbargain-participant-ac1721e4734c
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 10.0}

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 
[craigslistbargain-participant-7e1b9380862e] How about $6 and I pick it up myself? It'll save you shipping to me.

CURRENT OBSERVATION:
How about $6 and I pick it up myself? It'll save you shipping to me.

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
7, and we have a deal.
```

### Example 7 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:b53a9e35fde37297`
```
d=craigslistbargain-participant-7e1b9380862e
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 7.0}

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 
[craigslistbargain-participant-7e1b9380862e] How about $6 and I pick it up myself? It'll save you shipping to me.
[craigslistbargain-participant-ac1721e4734c] 7, and we have a deal.

CURRENT OBSERVATION:
7, and we have a deal.

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Eh, fine. $7.
```

### Example 8 — PREDICT_NEXT_ACTION — `craigslistbargain:PREDICT_NEXT_ACTION:010a3b9349499977`
```
ASK: PREDICT_NEXT_ACTION

ACTOR:
role=buyer, id=craigslistbargain-participant-7e1b9380862e
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 7.0}

KNOWN HISTORY:
... (1 earlier events elided)
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 
[craigslistbargain-participant-7e1b9380862e] How about $6 and I pick it up myself? It'll save you shipping to me.
[craigslistbargain-participant-ac1721e4734c] 7, and we have a deal.
[craigslistbargain-participant-7e1b9380862e] Eh, fine. $7.

CURRENT OBSERVATION:
Eh, fine. $7.

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"price": 7.0, "sides": ""}, "action_type": "offer"}
```

### Example 9 — PREDICT_NEXT_ACTION — `craigslistbargain:PREDICT_NEXT_ACTION:5bf36f68675a6306`
```
rgain-participant-ac1721e4734c
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 10.0}

KNOWN HISTORY:
... (2 earlier events elided)
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 
[craigslistbargain-participant-7e1b9380862e] How about $6 and I pick it up myself? It'll save you shipping to me.
[craigslistbargain-participant-ac1721e4734c] 7, and we have a deal.
[craigslistbargain-participant-7e1b9380862e] Eh, fine. $7.
[craigslistbargain-participant-7e1b9380862e] ACTION offer: {"price": 7.0, "sides": ""}

CURRENT OBSERVATION:
{"kind": "action", "meta": {"from_actor": "craigslistbargain-participant-7e1b9380862e"}, "text": null}

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"price": null, "sides": null}, "action_type": "accept"}
```

### Example 10 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:bd751879f46dbbc8`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-ac1721e4734c", "index": 1, "kind": "message", "meta": {"agent": 1, "intent": "unknown", "role": "seller"}, "t": "1496341307.92", "text": "It will work, i have never seen a car without a cigarette lighter port.\\"}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-parti
```

### Example 11 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:1be5c9c349663149`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-7e1b9380862e", "index": 2, "kind": "message", "meta": {"agent": 0, "intent": "insist", "role": "buyer"}, "t": "1496341329.15", "text": "Still, can I buy it for $5? I'm on a tight budge"}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-ac1721e4734c", "inde
```

### Example 12 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:89e9dcec217cfac7`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-ac1721e4734c", "index": 3, "kind": "message", "meta": {"agent": 1, "intent": "counter-price", "role": "seller"}, "t": "1496341345.99", "text": "I think the lowest I would want to go is 8. "}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-7e1b9380862e", "
```

### Example 13 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:b75e343265465d8f`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-7e1b9380862e", "index": 4, "kind": "message", "meta": {"agent": 0, "intent": "counter-price", "role": "buyer"}, "t": "1496341376.38", "text": "How about $6 and I pick it up myself? It'll save you shipping to me."}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-parti
```

### Example 14 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:07f1310b4f7fb5e6`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 
[craigslistbargain-participant-7e1b9380862e] How about $6 and I pick it up myself? It'll save you shipping to me.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-ac1721e4734c", "index": 5, "kind": "message", "meta": {"agent": 1, "intent": "counter-price", "role": "seller"}, "t": "1496341391.82", "text": "7, and we have a deal."}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-7e1b9380862e", "index": 6, "kind": "me
```

### Example 15 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:86b0abe577a1b317`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-7e1b9380862e] Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 
[craigslistbargain-participant-7e1b9380862e] How about $6 and I pick it up myself? It'll save you shipping to me.
[craigslistbargain-participant-ac1721e4734c] 7, and we have a deal.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-7e1b9380862e", "index": 6, "kind": "message", "meta": {"agent": 0, "intent": "agree", "role": "buyer"}, "t": "1496341400.98", "text": "Eh, fine. $7."}, {"action_content": {"price": 7.0, "sides": ""}, "action_type": "offer", "actor_id": "craigslistbargain-participant-7e1b9380862e", "index": 7, "
```

### Example 16 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:e32c4873272779f8`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (1 earlier events elided)
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 
[craigslistbargain-participant-7e1b9380862e] How about $6 and I pick it up myself? It'll save you shipping to me.
[craigslistbargain-participant-ac1721e4734c] 7, and we have a deal.
[craigslistbargain-participant-7e1b9380862e] Eh, fine. $7.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {"price": 7.0, "sides": ""}, "action_type": "offer", "actor_id": "craigslistbargain-participant-7e1b9380862e", "index": 7, "kind": "action", "meta": {"agent": 0, "intent": "offer", "role": "buyer"}, "t": "1496341417.71", "text": null}, {"action_content": {"price": null, "sides": null}, "action_type": "accept", "actor_id": "craigslistbargain-participant-ac1721e4
```

### Example 17 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:5d3e427ca2ed61e0`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (2 earlier events elided)
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 
[craigslistbargain-participant-7e1b9380862e] How about $6 and I pick it up myself? It'll save you shipping to me.
[craigslistbargain-participant-ac1721e4734c] 7, and we have a deal.
[craigslistbargain-participant-7e1b9380862e] Eh, fine. $7.
[craigslistbargain-participant-7e1b9380862e] ACTION offer: {"price": 7.0, "sides": ""}

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {"price": null, "sides": null}, "action_type": "accept", "actor_id": "craigslistbargain-participant-ac1721e4734c", "index": 8, "kind": "action", "meta": {"agent": 1, "intent": "accept", "role": "seller"}, "t": "1496341426.16", "text": null}]}
```

### Example 18 — PREDICT_FINAL_OUTCOME — `craigslistbargain:PREDICT_FINAL_OUTCOME:c40413c09a896d77`
```
TASK: PREDICT_FINAL_OUTCOME

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (1 earlier events elided)
[craigslistbargain-participant-ac1721e4734c] It will work, i have never seen a car without a cigarette lighter port.\
[craigslistbargain-participant-7e1b9380862e] Still, can I buy it for $5? I'm on a tight budge
[craigslistbargain-participant-ac1721e4734c] I think the lowest I would want to go is 8. 
[craigslistbargain-participant-7e1b9380862e] How about $6 and I pick it up myself? It'll save you shipping to me.
[craigslistbargain-participant-ac1721e4734c] 7, and we have a deal.
[craigslistbargain-participant-7e1b9380862e] Eh, fine. $7.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"outcome": {"agreement": true, "final_price": 7.0, "list_price": 10.0, "reward": 1}, "outcome_type": "negotiation_result"}
```

### Example 19 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:6f32fe6fce852937`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=buyer, id=craigslistbargain-participant-806f343e7e38
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 120.0}

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {}, "text": null}

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Hi,I am interested in your board!
```

### Example 20 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:79e1cd7097db41ab`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=seller, id=craigslistbargain-participant-2715786ce68c
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 200.0}

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!

CURRENT OBSERVATION:
Hi,I am interested in your board!

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Hi, do you have any questions?
```

### Example 21 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:4556cc9f8a0c9d61`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=buyer, id=craigslistbargain-participant-806f343e7e38
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 120.0}

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?

CURRENT OBSERVATION:
Hi, do you have any questions?

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Do you know what wood it is made out of?
```

### Example 22 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:c7c0c89843414644`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=seller, id=craigslistbargain-participant-2715786ce68c
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 200.0}

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?

CURRENT OBSERVATION:
Do you know what wood it is made out of?

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
I do not know specifically but the brand is a gravity skateboard. 
```

### Example 23 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:b245bf574ed2f042`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=buyer, id=craigslistbargain-participant-806f343e7e38
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 120.0}

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 

CURRENT OBSERVATION:
I do not know specifically but the brand is a gravity skateboard. 

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
```

### Example 24 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:55aad6ec6341037d`
```
EXT_MESSAGE

ACTOR:
role=seller, id=craigslistbargain-participant-2715786ce68c
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 200.0}

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?

CURRENT OBSERVATION:
The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
```

### Example 25 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:56f2a29d57352f0e`
```
argain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.

CURRENT OBSERVATION:
No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
If I picked it up at your location would you accpet 120?
```

### Example 26 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:28ce293f4a0a34bb`
```
OWN HISTORY:
... (1 earlier events elided)
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?

CURRENT OBSERVATION:
If I picked it up at your location would you accpet 120?

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Make it 150 and we got a deal. 
```

### Example 27 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:1b9d155eb4e23328`
```
arget_price": 120.0}

KNOWN HISTORY:
... (2 earlier events elided)
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 

CURRENT OBSERVATION:
Make it 150 and we got a deal. 

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 
```

### Example 28 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:e3d11c4397e54150`
```
know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 
[craigslistbargain-participant-806f343e7e38] I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 

CURRENT OBSERVATION:
I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Deal
```

### Example 29 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:710f2e493d29ac3d`
```
mline": null, "role": "buyer", "target_price": 120.0}

KNOWN HISTORY:
... (4 earlier events elided)
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 
[craigslistbargain-participant-806f343e7e38] I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 
[craigslistbargain-participant-2715786ce68c] Deal

CURRENT OBSERVATION:
Deal

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Great!
```

### Example 30 — PREDICT_NEXT_ACTION — `craigslistbargain:PREDICT_NEXT_ACTION:db5c9c14000df067`
```
istbargain-participant-806f343e7e38
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 120.0}

KNOWN HISTORY:
... (5 earlier events elided)
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 
[craigslistbargain-participant-806f343e7e38] I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 
[craigslistbargain-participant-2715786ce68c] Deal
[craigslistbargain-participant-806f343e7e38] Great!

CURRENT OBSERVATION:
Great!

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"price": 145.0, "sides": ""}, "action_type": "offer"}
```

### Example 31 — PREDICT_NEXT_ACTION — `craigslistbargain:PREDICT_NEXT_ACTION:2c04d3b456917047`
```
gain-participant-2715786ce68c
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 200.0}

KNOWN HISTORY:
... (6 earlier events elided)
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 
[craigslistbargain-participant-806f343e7e38] I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 
[craigslistbargain-participant-2715786ce68c] Deal
[craigslistbargain-participant-806f343e7e38] Great!
[craigslistbargain-participant-806f343e7e38] ACTION offer: {"price": 145.0, "sides": ""}

CURRENT OBSERVATION:
{"kind": "action", "meta": {"from_actor": "craigslistbargain-participant-806f343e7e38"}, "text": null}

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"price": null, "sides": null}, "action_type": "accept"}
```

### Example 32 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:e01a2ae9e7ed1b63`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-2715786ce68c", "index": 1, "kind": "message", "meta": {"agent": 1, "intent": "unknown", "role": "seller"}, "t": "1496341323.47", "text": "Hi, do you have any questions?"}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 2, "kind": "
```

### Example 33 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:7f0876ab6cc70766`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 2, "kind": "message", "meta": {"agent": 0, "intent": "inquiry", "role": "buyer"}, "t": "1496341349.68", "text": "Do you know what wood it is made out of?"}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-2715786ce68c", "index": 3, 
```

### Example 34 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:72e69eda4430e47b`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-2715786ce68c", "index": 3, "kind": "message", "meta": {"agent": 1, "intent": "disagree", "role": "seller"}, "t": "1496341385.18", "text": "I do not know specifically but the brand is a gravity skateboard. "}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant
```

### Example 35 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:d1ba14581ae68cc7`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 4, "kind": "message", "meta": {"agent": 0, "intent": "init-price", "role": "buyer"}, "t": "1496341420.57", "text": "The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?"}, {"action_content": {}, "action_type": null, "actor_id
```

### Example 36 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:dc654b6511449c1f`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-2715786ce68c", "index": 5, "kind": "message", "meta": {"agent": 1, "intent": "vague-price", "role": "seller"}, "t": "1496341476.72", "text": "No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new."}, {"action_conten
```

### Example 37 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:c4a4c8e6d67a800f`
```
CT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[craigslistbargain-participant-806f343e7e38] Hi,I am interested in your board!
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 6, "kind": "message", "meta": {"agent": 0, "intent": "counter-price", "role": "buyer"}, "t": "1496341503.98", "text": "If I picked it up at your location would you accpet 120?"}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-27157
```

### Example 38 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:e271f371aa6b628f`
```
PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (1 earlier events elided)
[craigslistbargain-participant-2715786ce68c] Hi, do you have any questions?
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-2715786ce68c", "index": 7, "kind": "message", "meta": {"agent": 1, "intent": "counter-price", "role": "seller"}, "t": "1496341555.05", "text": "Make it 150 and we got a deal. "}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 8, "k
```

### Example 39 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:f0717a5feb47a904`
```
RIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (2 earlier events elided)
[craigslistbargain-participant-806f343e7e38] Do you know what wood it is made out of?
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 8, "kind": "message", "meta": {"agent": 0, "intent": "counter-price", "role": "buyer"}, "t": "1496341586.55", "text": "I could do 145, if you can gurantee the painting on the front of the board is not scratched up. "}, {"action_content": {}, "action_type": null, "actor_i
```

### Example 40 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:1b26e16a625cc086`
```
 (3 earlier events elided)
[craigslistbargain-participant-2715786ce68c] I do not know specifically but the brand is a gravity skateboard. 
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 
[craigslistbargain-participant-806f343e7e38] I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-2715786ce68c", "index": 9, "kind": "message", "meta": {"agent": 1, "intent": "agree", "role": "seller"}, "t": "1496341610.13", "text": "Deal"}, {"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 10, "kind": "message", "meta": {"agent":
```

### Example 41 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:f65d88c4f2d98e91`
```
ion

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (4 earlier events elided)
[craigslistbargain-participant-806f343e7e38] The wheels seem nice on it, but they could be a beter quality. Would you accept 100 for the board?
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 
[craigslistbargain-participant-806f343e7e38] I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 
[craigslistbargain-participant-2715786ce68c] Deal

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {}, "action_type": null, "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 10, "kind": "message", "meta": {"agent": 0, "intent": "agree", "role": "buyer"}, "t": "1496341631.69", "text": "Great!"}, {"action_content": {"price": 145.0, "sides": ""}, "action_type": "offer", "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 11, "kin
```

### Example 42 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:4062de0656a06faf`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (5 earlier events elided)
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 
[craigslistbargain-participant-806f343e7e38] I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 
[craigslistbargain-participant-2715786ce68c] Deal
[craigslistbargain-participant-806f343e7e38] Great!

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {"price": 145.0, "sides": ""}, "action_type": "offer", "actor_id": "craigslistbargain-participant-806f343e7e38", "index": 11, "kind": "action", "meta": {"agent": 0, "intent": "offer", "role": "buyer"}, "t": "1496341634.34", "text": null}, {"action_content": {"price": null, "sides": null}, "action_type": "accept", "actor_id": "craigslistbargain-participant-27157
```

### Example 43 — PREDICT_TRAJECTORY_CONTINUATION — `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:a5b086f221bec7f0`
```
TASK: PREDICT_TRAJECTORY_CONTINUATION

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (6 earlier events elided)
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 
[craigslistbargain-participant-806f343e7e38] I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 
[craigslistbargain-participant-2715786ce68c] Deal
[craigslistbargain-participant-806f343e7e38] Great!
[craigslistbargain-participant-806f343e7e38] ACTION offer: {"price": 145.0, "sides": ""}

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"continuation": [{"action_content": {"price": null, "sides": null}, "action_type": "accept", "actor_id": "craigslistbargain-participant-2715786ce68c", "index": 12, "kind": "action", "meta": {"agent": 1, "intent": "accept", "role": "seller"}, "t": "1496341677.66", "text": null}]}
```

### Example 44 — PREDICT_FINAL_OUTCOME — `craigslistbargain:PREDICT_FINAL_OUTCOME:cd0c198000105d09`
```
TASK: PREDICT_FINAL_OUTCOME

ACTOR:
role=negotiation

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (5 earlier events elided)
[craigslistbargain-participant-2715786ce68c] No, that offer is too low. The board is pretty much brand new as it's been ridden only 4 or 5 times.  The bone bearings are brand new.
[craigslistbargain-participant-806f343e7e38] If I picked it up at your location would you accpet 120?
[craigslistbargain-participant-2715786ce68c] Make it 150 and we got a deal. 
[craigslistbargain-participant-806f343e7e38] I could do 145, if you can gurantee the painting on the front of the board is not scratched up. 
[craigslistbargain-participant-2715786ce68c] Deal
[craigslistbargain-participant-806f343e7e38] Great!

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
{"outcome": {"agreement": true, "final_price": 145.0, "list_price": 200.0, "reward": 1}, "outcome_type": "negotiation_result"}
```

### Example 45 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:13664b6121c87834`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=buyer, id=craigslistbargain-participant-34ec5d4a8f96
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 1920.0}

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {}, "text": null}

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
i would love to buy 
```

### Example 46 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:23ef8cfac4035ad0`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=seller, id=craigslistbargain-participant-54b95a1659f5
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 3200.0}

KNOWN HISTORY:
[craigslistbargain-participant-34ec5d4a8f96] i would love to buy 

CURRENT OBSERVATION:
i would love to buy 

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Sure! What's your price?
```

### Example 47 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:4088a56942967a36`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=buyer, id=craigslistbargain-participant-34ec5d4a8f96
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 1920.0}

KNOWN HISTORY:
[craigslistbargain-participant-34ec5d4a8f96] i would love to buy 
[craigslistbargain-participant-54b95a1659f5] Sure! What's your price?

CURRENT OBSERVATION:
Sure! What's your price?

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
im on a budget so i could do 1850
```

### Example 48 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:c0a105b1b7bea1de`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=seller, id=craigslistbargain-participant-54b95a1659f5
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 3200.0}

KNOWN HISTORY:
[craigslistbargain-participant-34ec5d4a8f96] i would love to buy 
[craigslistbargain-participant-54b95a1659f5] Sure! What's your price?
[craigslistbargain-participant-34ec5d4a8f96] im on a budget so i could do 1850

CURRENT OBSERVATION:
im on a budget so i could do 1850

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
How about $1900 and I'll wave the deposit.
```

### Example 49 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:e18bc233b68cf745`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=buyer, id=craigslistbargain-participant-34ec5d4a8f96
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "buyer", "target_price": 1920.0}

KNOWN HISTORY:
[craigslistbargain-participant-34ec5d4a8f96] i would love to buy 
[craigslistbargain-participant-54b95a1659f5] Sure! What's your price?
[craigslistbargain-participant-34ec5d4a8f96] im on a budget so i could do 1850
[craigslistbargain-participant-54b95a1659f5] How about $1900 and I'll wave the deposit.

CURRENT OBSERVATION:
How about $1900 and I'll wave the deposit.

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
i will take it
```

### Example 50 — PREDICT_NEXT_MESSAGE — `craigslistbargain:PREDICT_NEXT_MESSAGE:2628ad6a771f7af6`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=seller, id=craigslistbargain-participant-54b95a1659f5
{"agent_type": "human"}

PRIVATE STATE BEFORE:
{"bottomline": null, "role": "seller", "target_price": 3200.0}

KNOWN HISTORY:
[craigslistbargain-participant-34ec5d4a8f96] i would love to buy 
[craigslistbargain-participant-54b95a1659f5] Sure! What's your price?
[craigslistbargain-participant-34ec5d4a8f96] im on a budget so i could do 1850
[craigslistbargain-participant-54b95a1659f5] How about $1900 and I'll wave the deposit.
[craigslistbargain-participant-34ec5d4a8f96] i will take it

CURRENT OBSERVATION:
i will take it

AVAILABLE ACTIONS:
["message", "offer", "accept", "reject", "quit"]

TARGET:

--- TARGET ---
Great!
```

## 25 most-suspicious examples (warnings / possible leakage)

- `craigslistbargain:PREDICT_NEXT_MESSAGE:51e39556adfc8c02` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:eb291f1a1b925ce4` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:68237e9b1e151e64` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:209e02a62863fc5f` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:cc66768f03ff58f1` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:728e07c3b9df6dad` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:b53a9e35fde37297` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_ACTION:010a3b9349499977` (PREDICT_NEXT_ACTION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_ACTION:5bf36f68675a6306` (PREDICT_NEXT_ACTION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:bd751879f46dbbc8` (PREDICT_TRAJECTORY_CONTINUATION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:1be5c9c349663149` (PREDICT_TRAJECTORY_CONTINUATION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:89e9dcec217cfac7` (PREDICT_TRAJECTORY_CONTINUATION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:b75e343265465d8f` (PREDICT_TRAJECTORY_CONTINUATION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:07f1310b4f7fb5e6` (PREDICT_TRAJECTORY_CONTINUATION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:86b0abe577a1b317` (PREDICT_TRAJECTORY_CONTINUATION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:e32c4873272779f8` (PREDICT_TRAJECTORY_CONTINUATION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_TRAJECTORY_CONTINUATION:5d3e427ca2ed61e0` (PREDICT_TRAJECTORY_CONTINUATION): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_FINAL_OUTCOME:c40413c09a896d77` (PREDICT_FINAL_OUTCOME): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:6f32fe6fce852937` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:79e1cd7097db41ab` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:4556cc9f8a0c9d61` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:c7c0c89843414644` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:b245bf574ed2f042` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:55aad6ec6341037d` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
- `craigslistbargain:PREDICT_NEXT_MESSAGE:56f2a29d57352f0e` (PREDICT_NEXT_MESSAGE): warnings=['dataset license unstated — eval-only'] possible_leakage=False
