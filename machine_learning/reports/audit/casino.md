# Audit — CaSiNo: A Corpus of Campsite Negotiation Dialogues

- **id**: `casino`  |  **role**: TRAIN_CANDIDATE  |  **status**: PENDING
- **official source**: https://huggingface.co/datasets/kchawla123/casino
- **paper**: https://aclanthology.org/2021.naacl-main.254/
- **license**: CC-BY-4.0 (`cc_by`) — commercial=yes, derivatives=yes
- **acquisition**: acquired (3 raw files, 1260547 bytes)

## Normalized data

- examples: **15327**  |  quarantined: 0  |  episodes: 1030  |  actors: 2060
- task counts: `{'PREDICT_NEXT_MESSAGE': 11919, 'PREDICT_NEXT_ACTION': 2378, 'PREDICT_FINAL_OUTCOME': 1030}`
- split sizes: `{'train': 12330, 'test_in_domain': 1477, 'validation': 1520}`

## Distributions

- inactivity: `{'n_inactive': 0, 'n_action_or_response': 2378, 'inactivity_rate': 0.0}`
- action types: `{'Submit-Deal': 1181, 'Accept-Deal': 1005, 'Reject-Deal': 167, 'Walk-Away': 25}`
- outcomes: `{'None': 1030}`
- response-time (s): `{}`
- context length (chars): `{'min': 738.0, 'p25': 1250.0, 'median': 1680.0, 'p75': 2201.0, 'p95': 3147.0, 'max': 8510.0, 'mean': 1812.22, 'n': 3066}`
- missing fields: `{'timestamps': 15327, 'strategy_annotation': 7304}`

## Leakage

- result: `{'dataset_id': 'casino', 'ok': True, 'n_records': 15327, 'episode_violations': 0, 'unit_violations': 0, 'cross_split_dupes': 0, 'details': {'dataset_id': 'casino', 'ok': True, 'episode_violations': [], 'unit_violations': [], 'cross_split_dupes': [], 'n_records': 15327, 'notes': []}}`

## Converter assumptions

- deal actions are exactly {Submit-Deal, Accept-Deal, Reject-Deal, Walk-Away}

## Known limitations

- strategy labels only on ~396/1030 dialogues
- satisfaction/liking are single post-hoc self-reports

## Unavailable fields (stored null, never fabricated)

- per-turn timestamps
- stable cross-dialogue worker id
- pre-negotiation satisfaction (only post measured)

## Recommendations

- **training**: Eligible for training pending human approval (training_approvals.yaml).
- **evaluation**: Usable for in-domain evaluation on its own test split.

## 50 rendered examples (human review)

### Example 1 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:6f575a8457cbcae4`
```
: {"big-five": {"agreeableness": 6.0, "conscientiousness": 6.0, "emotional-stability": 5.0, "extraversion": 5.0, "openness-to-experiences": 5.5}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Firewood", "Low": "Water", "Medium": "Food"}, "reasons": {"High": "We have a larger group than normal and therefore require extra firewood to keep everyone warm, and we also have no lanterns or other light source available.", "Low": "Our group has sufficient water from our complement basic supplies, particularly if we ration the water sparingly among ourselves.", "Medium": "Extra food will be needed to feed our larger than normal-sized group or else we'll go hungry which will certainly impact our trip in a negative way."}}

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {}, "text": null}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Hello! 🙂 Let's work together on a deal for these packages, shall we? What are you most interested in?
```

### Example 2 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:c398bf179883e1fc`
```
: {"age": 22, "education": "some 4 year college, bachelor's degree", "ethnicity": "asian american", "gender": "female"}, "personality": {"big-five": {"agreeableness": 6.0, "conscientiousness": 5.5, "emotional-stability": 3.0, "extraversion": 4.0, "openness-to-experiences": 7.0}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Firewood", "Low": "Food", "Medium": "Water"}, "reasons": {"High": "my dog has fleas, the fire repels them.", "Low": "i'm on a diet, trying to lose weight.", "Medium": "i'm dehydrated, and i need to drink constantly."}}

KNOWN HISTORY:
[casino-participant-190084be3400] Hello! 🙂 Let's work together on a deal for these packages, shall we? What are you most interested in?

CURRENT OBSERVATION:
Hello! 🙂 Let's work together on a deal for these packages, shall we? What are you most interested in?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Hey! I'd like some more firewood to keep my doggo warm. What do you need?
```

### Example 3 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:acfef3ed72c7d727`
```
"Food"}, "reasons": {"High": "We have a larger group than normal and therefore require extra firewood to keep everyone warm, and we also have no lanterns or other light source available.", "Low": "Our group has sufficient water from our complement basic supplies, particularly if we ration the water sparingly among ourselves.", "Medium": "Extra food will be needed to feed our larger than normal-sized group or else we'll go hungry which will certainly impact our trip in a negative way."}}

KNOWN HISTORY:
[casino-participant-190084be3400] Hello! 🙂 Let's work together on a deal for these packages, shall we? What are you most interested in?
[casino-participant-e2ff424ab61a] Hey! I'd like some more firewood to keep my doggo warm. What do you need?

CURRENT OBSERVATION:
Hey! I'd like some more firewood to keep my doggo warm. What do you need?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I need firewood as well. We have a large group consisting of mostly senior citizens, including my grandma, so we'd like the firewood to keep everyone warm.
```

### Example 4 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:2254cb96025e51b1`
```
"Firewood", "Low": "Food", "Medium": "Water"}, "reasons": {"High": "my dog has fleas, the fire repels them.", "Low": "i'm on a diet, trying to lose weight.", "Medium": "i'm dehydrated, and i need to drink constantly."}}

KNOWN HISTORY:
[casino-participant-190084be3400] Hello! 🙂 Let's work together on a deal for these packages, shall we? What are you most interested in?
[casino-participant-e2ff424ab61a] Hey! I'd like some more firewood to keep my doggo warm. What do you need?
[casino-participant-190084be3400] I need firewood as well. We have a large group consisting of mostly senior citizens, including my grandma, so we'd like the firewood to keep everyone warm.

CURRENT OBSERVATION:
I need firewood as well. We have a large group consisting of mostly senior citizens, including my grandma, so we'd like the firewood to keep everyone warm.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I see. 😮 What are you least interested in?
```

### Example 5 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:c887667122e9c607`
```
 complement basic supplies, particularly if we ration the water sparingly among ourselves.", "Medium": "Extra food will be needed to feed our larger than normal-sized group or else we'll go hungry which will certainly impact our trip in a negative way."}}

KNOWN HISTORY:
[casino-participant-190084be3400] Hello! 🙂 Let's work together on a deal for these packages, shall we? What are you most interested in?
[casino-participant-e2ff424ab61a] Hey! I'd like some more firewood to keep my doggo warm. What do you need?
[casino-participant-190084be3400] I need firewood as well. We have a large group consisting of mostly senior citizens, including my grandma, so we'd like the firewood to keep everyone warm.
[casino-participant-e2ff424ab61a] I see. 😮 What are you least interested in?

CURRENT OBSERVATION:
I see. 😮 What are you least interested in?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
We can make do without extra water. Can we trade two waters for an extra firewood package and an extra food package?
```

### Example 6 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:dd9295d43c8d6b19`
```
i need to drink constantly."}}

KNOWN HISTORY:
[casino-participant-190084be3400] Hello! 🙂 Let's work together on a deal for these packages, shall we? What are you most interested in?
[casino-participant-e2ff424ab61a] Hey! I'd like some more firewood to keep my doggo warm. What do you need?
[casino-participant-190084be3400] I need firewood as well. We have a large group consisting of mostly senior citizens, including my grandma, so we'd like the firewood to keep everyone warm.
[casino-participant-e2ff424ab61a] I see. 😮 What are you least interested in?
[casino-participant-190084be3400] We can make do without extra water. Can we trade two waters for an extra firewood package and an extra food package?

CURRENT OBSERVATION:
We can make do without extra water. Can we trade two waters for an extra firewood package and an extra food package?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
We could do without the water as well. I'm willing to trade you 3 firewood for 3 food and 2 waters
```

### Example 7 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:a75c531e7e94abbd`
```
deal for these packages, shall we? What are you most interested in?
[casino-participant-e2ff424ab61a] Hey! I'd like some more firewood to keep my doggo warm. What do you need?
[casino-participant-190084be3400] I need firewood as well. We have a large group consisting of mostly senior citizens, including my grandma, so we'd like the firewood to keep everyone warm.
[casino-participant-e2ff424ab61a] I see. 😮 What are you least interested in?
[casino-participant-190084be3400] We can make do without extra water. Can we trade two waters for an extra firewood package and an extra food package?
[casino-participant-e2ff424ab61a] We could do without the water as well. I'm willing to trade you 3 firewood for 3 food and 2 waters

CURRENT OBSERVATION:
We could do without the water as well. I'm willing to trade you 3 firewood for 3 food and 2 waters

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
We need some firewood too, though! ☹️ Let's try to make a deal that benefits us both! 🙂 Could I have 1 firewood, 3 food, and 3 waters?
```

### Example 8 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:683bb038157f8eab`
```
400] I need firewood as well. We have a large group consisting of mostly senior citizens, including my grandma, so we'd like the firewood to keep everyone warm.
[casino-participant-e2ff424ab61a] I see. 😮 What are you least interested in?
[casino-participant-190084be3400] We can make do without extra water. Can we trade two waters for an extra firewood package and an extra food package?
[casino-participant-e2ff424ab61a] We could do without the water as well. I'm willing to trade you 3 firewood for 3 food and 2 waters
[casino-participant-190084be3400] We need some firewood too, though! ☹️ Let's try to make a deal that benefits us both! 🙂 Could I have 1 firewood, 3 food, and 3 waters?

CURRENT OBSERVATION:
We need some firewood too, though! ☹️ Let's try to make a deal that benefits us both! 🙂 Could I have 1 firewood, 3 food, and 3 waters?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I meant I would give you my firewood, what would you trade in return?
```

### Example 9 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:bc072cff7c408f69`
```
 large group consisting of mostly senior citizens, including my grandma, so we'd like the firewood to keep everyone warm.
[casino-participant-e2ff424ab61a] I see. 😮 What are you least interested in?
[casino-participant-190084be3400] We can make do without extra water. Can we trade two waters for an extra firewood package and an extra food package?
[casino-participant-e2ff424ab61a] We could do without the water as well. I'm willing to trade you 3 firewood for 3 food and 2 waters
[casino-participant-190084be3400] We need some firewood too, though! ☹️ Let's try to make a deal that benefits us both! 🙂 Could I have 1 firewood, 3 food, and 3 waters?
[casino-participant-e2ff424ab61a] I meant I would give you my firewood, what would you trade in return?

CURRENT OBSERVATION:
I meant I would give you my firewood, what would you trade in return?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Oh sorry for the confusion. 😮 In that case, thank you for the generosity! 🙂 How about if I have 3 firewood, 1 food, and 1 water?
```

### Example 10 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:4c1d747e0024f9ca`
```
084be3400] We can make do without extra water. Can we trade two waters for an extra firewood package and an extra food package?
[casino-participant-e2ff424ab61a] We could do without the water as well. I'm willing to trade you 3 firewood for 3 food and 2 waters
[casino-participant-190084be3400] We need some firewood too, though! ☹️ Let's try to make a deal that benefits us both! 🙂 Could I have 1 firewood, 3 food, and 3 waters?
[casino-participant-e2ff424ab61a] I meant I would give you my firewood, what would you trade in return?
[casino-participant-190084be3400] Oh sorry for the confusion. 😮 In that case, thank you for the generosity! 🙂 How about if I have 3 firewood, 1 food, and 1 water?

CURRENT OBSERVATION:
Oh sorry for the confusion. 😮 In that case, thank you for the generosity! 🙂 How about if I have 3 firewood, 1 food, and 1 water?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I can do that for 3 food and 3 water?
```

### Example 11 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:ac0f06fff74544e6`
```
ino-participant-190084be3400] We can make do without extra water. Can we trade two waters for an extra firewood package and an extra food package?
[casino-participant-e2ff424ab61a] We could do without the water as well. I'm willing to trade you 3 firewood for 3 food and 2 waters
[casino-participant-190084be3400] We need some firewood too, though! ☹️ Let's try to make a deal that benefits us both! 🙂 Could I have 1 firewood, 3 food, and 3 waters?
[casino-participant-e2ff424ab61a] I meant I would give you my firewood, what would you trade in return?
[casino-participant-190084be3400] Oh sorry for the confusion. 😮 In that case, thank you for the generosity! 🙂 How about if I have 3 firewood, 1 food, and 1 water?
[casino-participant-e2ff424ab61a] I can do that for 3 food and 3 water?

CURRENT OBSERVATION:
I can do that for 3 food and 3 water?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I think we're close. Let's shake hands on this: 3 firewood, 1 food for me, and then 2 food and 3 water for you. I'll make sure your food packages have extra doggy treats! 🙂 Is that a deal?
```

### Example 12 — PREDICT_NEXT_ACTION — `casino:PREDICT_NEXT_ACTION:62ecd6557a53f9c5`
```
 and 3 waters?
[casino-participant-e2ff424ab61a] I meant I would give you my firewood, what would you trade in return?
[casino-participant-190084be3400] Oh sorry for the confusion. 😮 In that case, thank you for the generosity! 🙂 How about if I have 3 firewood, 1 food, and 1 water?
[casino-participant-e2ff424ab61a] I can do that for 3 food and 3 water?
[casino-participant-190084be3400] I think we're close. Let's shake hands on this: 3 firewood, 1 food for me, and then 2 food and 3 water for you. I'll make sure your food packages have extra doggy treats! 🙂 Is that a deal?

CURRENT OBSERVATION:
I think we're close. Let's shake hands on this: 3 firewood, 1 food for me, and then 2 food and 3 water for you. I'll make sure your food packages have extra doggy treats! 🙂 Is that a deal?

AVAILABLE ACTIONS:
["Accept-Deal", "Reject-Deal", "Submit-Deal", "Walk-Away", "continue-negotiating"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"proposer_gets": {"Firewood": "0", "Food": "2", "Water": "3"}, "responder_gets": {"Firewood": "3", "Food": "1", "Water": "0"}}, "action_type": "Submit-Deal"}
```

### Example 13 — PREDICT_NEXT_ACTION — `casino:PREDICT_NEXT_ACTION:a1308d2cb5a31e46`
```
, what would you trade in return?
[casino-participant-190084be3400] Oh sorry for the confusion. 😮 In that case, thank you for the generosity! 🙂 How about if I have 3 firewood, 1 food, and 1 water?
[casino-participant-e2ff424ab61a] I can do that for 3 food and 3 water?
[casino-participant-190084be3400] I think we're close. Let's shake hands on this: 3 firewood, 1 food for me, and then 2 food and 3 water for you. I'll make sure your food packages have extra doggy treats! 🙂 Is that a deal?
[casino-participant-e2ff424ab61a] ACTION Submit-Deal: {"proposer_gets": {"Firewood": "0", "Water": "3", "Food": "2"}, "responder_gets": {"Firewood": "3", "Water": "0", "Food": "1"}}

CURRENT OBSERVATION:
{"kind": "action", "meta": {"from_actor": "casino-participant-e2ff424ab61a"}, "text": null}

AVAILABLE ACTIONS:
["Accept-Deal", "Reject-Deal", "Submit-Deal", "Walk-Away", "continue-negotiating"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"proposer_gets": {"Firewood": "", "Food": "", "Water": ""}, "responder_gets": {"Firewood": "", "Food": "", "Water": ""}}, "action_type": "Accept-Deal"}
```

### Example 14 — PREDICT_FINAL_OUTCOME — `casino:PREDICT_FINAL_OUTCOME:5aec8f1eb19925ca`
```
the water as well. I'm willing to trade you 3 firewood for 3 food and 2 waters
[casino-participant-190084be3400] We need some firewood too, though! ☹️ Let's try to make a deal that benefits us both! 🙂 Could I have 1 firewood, 3 food, and 3 waters?
[casino-participant-e2ff424ab61a] I meant I would give you my firewood, what would you trade in return?
[casino-participant-190084be3400] Oh sorry for the confusion. 😮 In that case, thank you for the generosity! 🙂 How about if I have 3 firewood, 1 food, and 1 water?
[casino-participant-e2ff424ab61a] I can do that for 3 food and 3 water?
[casino-participant-190084be3400] I think we're close. Let's shake hands on this: 3 firewood, 1 food for me, and then 2 food and 3 water for you. I'll make sure your food packages have extra doggy treats! 🙂 Is that a deal?

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"outcome": {"deal_reached": true, "opponent_likeness": {"mturk_agent_1": "Slightly like", "mturk_agent_2": "Extremely like"}, "points": {"mturk_agent_1": 19, "mturk_agent_2": 18}, "satisfaction": {"mturk_agent_1": "Slightly satisfied", "mturk_agent_2": "Extremely satisfied"}}, "outcome_type": "negotiation_result"}
```

### Example 15 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:a338fa4a183485fd`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=negotiator, id=casino-participant-e6f678bfe5fa
{"demographics": {"age": 43, "education": "high school graduate / ged", "ethnicity": "white american", "gender": "female"}, "personality": {"big-five": {"agreeableness": 7.0, "conscientiousness": 5.0, "emotional-stability": 7.0, "extraversion": 1.0, "openness-to-experiences": 6.0}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Food", "Low": "Water", "Medium": "Firewood"}, "reasons": {"High": "when I get nervous or scared I eat, so I'd like more food", "Low": "I can do without water, but will take it.", "Medium": "I don't like the dark, so this would help keep it light."}}

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {}, "text": null}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Hello. How are you?
```

### Example 16 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:a1f6e9d04ee277f3`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=negotiator, id=casino-participant-8f201656f4a4
{"demographics": {"age": 24, "education": "some 4 year college, bachelor's degree", "ethnicity": "black or african american", "gender": "female"}, "personality": {"big-five": {"agreeableness": 4.5, "conscientiousness": 6.0, "emotional-stability": 6.5, "extraversion": 3.5, "openness-to-experiences": 6.5}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Food", "Low": "Water", "Medium": "Firewood"}, "reasons": {"High": "I am camping with two children, and need food to keep them comfortable", "Low": "The stream nearby is very clean", "Medium": "It eliminates the heavy misquito coverage at night"}}

KNOWN HISTORY:
[casino-participant-e6f678bfe5fa] Hello. How are you?

CURRENT OBSERVATION:
Hello. How are you?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I am good. I am pretty excited for the trip this weekend. what about you?
```

### Example 17 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:fc31d1641547c822`
```
ucation": "high school graduate / ged", "ethnicity": "white american", "gender": "female"}, "personality": {"big-five": {"agreeableness": 7.0, "conscientiousness": 5.0, "emotional-stability": 7.0, "extraversion": 1.0, "openness-to-experiences": 6.0}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Food", "Low": "Water", "Medium": "Firewood"}, "reasons": {"High": "when I get nervous or scared I eat, so I'd like more food", "Low": "I can do without water, but will take it.", "Medium": "I don't like the dark, so this would help keep it light."}}

KNOWN HISTORY:
[casino-participant-e6f678bfe5fa] Hello. How are you?
[casino-participant-8f201656f4a4] I am good. I am pretty excited for the trip this weekend. what about you?

CURRENT OBSERVATION:
I am good. I am pretty excited for the trip this weekend. what about you?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Very excited. It will be fun.
```

### Example 18 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:96148f27d6a2e9ef`
```
's degree", "ethnicity": "black or african american", "gender": "female"}, "personality": {"big-five": {"agreeableness": 4.5, "conscientiousness": 6.0, "emotional-stability": 6.5, "extraversion": 3.5, "openness-to-experiences": 6.5}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Food", "Low": "Water", "Medium": "Firewood"}, "reasons": {"High": "I am camping with two children, and need food to keep them comfortable", "Low": "The stream nearby is very clean", "Medium": "It eliminates the heavy misquito coverage at night"}}

KNOWN HISTORY:
[casino-participant-e6f678bfe5fa] Hello. How are you?
[casino-participant-8f201656f4a4] I am good. I am pretty excited for the trip this weekend. what about you?
[casino-participant-e6f678bfe5fa] Very excited. It will be fun.

CURRENT OBSERVATION:
Very excited. It will be fun.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yes, Hopefully the weather holds up. So I was thinking, i am bringning my two kids along with me, and they will need food, so I was thinking i could take the three foods, and either one of the firewood or water, up to you.
```

### Example 19 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:627fbd3c0a73156d`
```
thout water, but will take it.", "Medium": "I don't like the dark, so this would help keep it light."}}

KNOWN HISTORY:
[casino-participant-e6f678bfe5fa] Hello. How are you?
[casino-participant-8f201656f4a4] I am good. I am pretty excited for the trip this weekend. what about you?
[casino-participant-e6f678bfe5fa] Very excited. It will be fun.
[casino-participant-8f201656f4a4] Yes, Hopefully the weather holds up. So I was thinking, i am bringning my two kids along with me, and they will need food, so I was thinking i could take the three foods, and either one of the firewood or water, up to you.

CURRENT OBSERVATION:
Yes, Hopefully the weather holds up. So I was thinking, i am bringning my two kids along with me, and they will need food, so I was thinking i could take the three foods, and either one of the firewood or water, up to you.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I would also like a little extra food for my kids. Maybe we can split it somehow?
```

### Example 20 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:041d186501900072`
```
ep them comfortable", "Low": "The stream nearby is very clean", "Medium": "It eliminates the heavy misquito coverage at night"}}

KNOWN HISTORY:
[casino-participant-e6f678bfe5fa] Hello. How are you?
[casino-participant-8f201656f4a4] I am good. I am pretty excited for the trip this weekend. what about you?
[casino-participant-e6f678bfe5fa] Very excited. It will be fun.
[casino-participant-8f201656f4a4] Yes, Hopefully the weather holds up. So I was thinking, i am bringning my two kids along with me, and they will need food, so I was thinking i could take the three foods, and either one of the firewood or water, up to you.
[casino-participant-e6f678bfe5fa] I would also like a little extra food for my kids. Maybe we can split it somehow?

CURRENT OBSERVATION:
I would also like a little extra food for my kids. Maybe we can split it somehow?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Ok, I am willing to give you one food, in exchange for two firewoods, that would mean you get 3 waters, 1 food and 1 firewood. you get 5 items, while i get 4.
```

### Example 21 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:de3f2e30a7ea81b2`
```
e trip this weekend. what about you?
[casino-participant-e6f678bfe5fa] Very excited. It will be fun.
[casino-participant-8f201656f4a4] Yes, Hopefully the weather holds up. So I was thinking, i am bringning my two kids along with me, and they will need food, so I was thinking i could take the three foods, and either one of the firewood or water, up to you.
[casino-participant-e6f678bfe5fa] I would also like a little extra food for my kids. Maybe we can split it somehow?
[casino-participant-8f201656f4a4] Ok, I am willing to give you one food, in exchange for two firewoods, that would mean you get 3 waters, 1 food and 1 firewood. you get 5 items, while i get 4.

CURRENT OBSERVATION:
Ok, I am willing to give you one food, in exchange for two firewoods, that would mean you get 3 waters, 1 food and 1 firewood. you get 5 items, while i get 4.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
If I only get 1 food, than I would like 2 firewood. So you get 2 food, 1 firewood, and 1 water.
```

### Example 22 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:32174d6b365b9102`
```
fa] Very excited. It will be fun.
[casino-participant-8f201656f4a4] Yes, Hopefully the weather holds up. So I was thinking, i am bringning my two kids along with me, and they will need food, so I was thinking i could take the three foods, and either one of the firewood or water, up to you.
[casino-participant-e6f678bfe5fa] I would also like a little extra food for my kids. Maybe we can split it somehow?
[casino-participant-8f201656f4a4] Ok, I am willing to give you one food, in exchange for two firewoods, that would mean you get 3 waters, 1 food and 1 firewood. you get 5 items, while i get 4.
[casino-participant-e6f678bfe5fa] If I only get 1 food, than I would like 2 firewood. So you get 2 food, 1 firewood, and 1 water.

CURRENT OBSERVATION:
If I only get 1 food, than I would like 2 firewood. So you get 2 food, 1 firewood, and 1 water.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
No i do not need water, as i am camping near a potable stream. I would like 2 firewood. It is only fair that i get two firewood, since you are getting more supplies (5), i think i should be able to choose what 4 i should get.
```

### Example 23 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:ae180b9c64f22bcf`
```
plit it somehow?
[casino-participant-8f201656f4a4] Ok, I am willing to give you one food, in exchange for two firewoods, that would mean you get 3 waters, 1 food and 1 firewood. you get 5 items, while i get 4.
[casino-participant-e6f678bfe5fa] If I only get 1 food, than I would like 2 firewood. So you get 2 food, 1 firewood, and 1 water.
[casino-participant-8f201656f4a4] No i do not need water, as i am camping near a potable stream. I would like 2 firewood. It is only fair that i get two firewood, since you are getting more supplies (5), i think i should be able to choose what 4 i should get.

CURRENT OBSERVATION:
No i do not need water, as i am camping near a potable stream. I would like 2 firewood. It is only fair that i get two firewood, since you are getting more supplies (5), i think i should be able to choose what 4 i should get.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
No, if you get 2 food, it's only fair that I get 2 firewood. 
```

### Example 24 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:8eb75c2a98c22b42`
```
a] I would also like a little extra food for my kids. Maybe we can split it somehow?
[casino-participant-8f201656f4a4] Ok, I am willing to give you one food, in exchange for two firewoods, that would mean you get 3 waters, 1 food and 1 firewood. you get 5 items, while i get 4.
[casino-participant-e6f678bfe5fa] If I only get 1 food, than I would like 2 firewood. So you get 2 food, 1 firewood, and 1 water.
[casino-participant-8f201656f4a4] No i do not need water, as i am camping near a potable stream. I would like 2 firewood. It is only fair that i get two firewood, since you are getting more supplies (5), i think i should be able to choose what 4 i should get.
[casino-participant-e6f678bfe5fa] No, if you get 2 food, it's only fair that I get 2 firewood. 

CURRENT OBSERVATION:
No, if you get 2 food, it's only fair that I get 2 firewood. 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Then i will take 3 waters, and 2 food.
```

### Example 25 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:3510985d2dda3d1f`
```
ds. Maybe we can split it somehow?
[casino-participant-8f201656f4a4] Ok, I am willing to give you one food, in exchange for two firewoods, that would mean you get 3 waters, 1 food and 1 firewood. you get 5 items, while i get 4.
[casino-participant-e6f678bfe5fa] If I only get 1 food, than I would like 2 firewood. So you get 2 food, 1 firewood, and 1 water.
[casino-participant-8f201656f4a4] No i do not need water, as i am camping near a potable stream. I would like 2 firewood. It is only fair that i get two firewood, since you are getting more supplies (5), i think i should be able to choose what 4 i should get.
[casino-participant-e6f678bfe5fa] No, if you get 2 food, it's only fair that I get 2 firewood. 
[casino-participant-8f201656f4a4] Then i will take 3 waters, and 2 food.

CURRENT OBSERVATION:
Then i will take 3 waters, and 2 food.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Okay I will take all the firewood and 1 food.
```

### Example 26 — PREDICT_NEXT_ACTION — `casino:PREDICT_NEXT_ACTION:10bf39e638dbc225`
```
ld mean you get 3 waters, 1 food and 1 firewood. you get 5 items, while i get 4.
[casino-participant-e6f678bfe5fa] If I only get 1 food, than I would like 2 firewood. So you get 2 food, 1 firewood, and 1 water.
[casino-participant-8f201656f4a4] No i do not need water, as i am camping near a potable stream. I would like 2 firewood. It is only fair that i get two firewood, since you are getting more supplies (5), i think i should be able to choose what 4 i should get.
[casino-participant-e6f678bfe5fa] No, if you get 2 food, it's only fair that I get 2 firewood. 
[casino-participant-8f201656f4a4] Then i will take 3 waters, and 2 food.
[casino-participant-e6f678bfe5fa] Okay I will take all the firewood and 1 food.

CURRENT OBSERVATION:
Okay I will take all the firewood and 1 food.

AVAILABLE ACTIONS:
["Accept-Deal", "Reject-Deal", "Submit-Deal", "Walk-Away", "continue-negotiating"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"proposer_gets": {"Firewood": "0", "Food": "2", "Water": "3"}, "responder_gets": {"Firewood": "3", "Food": "1", "Water": "0"}}, "action_type": "Submit-Deal"}
```

### Example 27 — PREDICT_NEXT_ACTION — `casino:PREDICT_NEXT_ACTION:1e7788efad1f3ff5`
```
nt-8f201656f4a4] No i do not need water, as i am camping near a potable stream. I would like 2 firewood. It is only fair that i get two firewood, since you are getting more supplies (5), i think i should be able to choose what 4 i should get.
[casino-participant-e6f678bfe5fa] No, if you get 2 food, it's only fair that I get 2 firewood. 
[casino-participant-8f201656f4a4] Then i will take 3 waters, and 2 food.
[casino-participant-e6f678bfe5fa] Okay I will take all the firewood and 1 food.
[casino-participant-8f201656f4a4] ACTION Submit-Deal: {"proposer_gets": {"Firewood": "0", "Water": "3", "Food": "2"}, "responder_gets": {"Firewood": "3", "Water": "0", "Food": "1"}}

CURRENT OBSERVATION:
{"kind": "action", "meta": {"from_actor": "casino-participant-8f201656f4a4"}, "text": null}

AVAILABLE ACTIONS:
["Accept-Deal", "Reject-Deal", "Submit-Deal", "Walk-Away", "continue-negotiating"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"proposer_gets": {"Firewood": "", "Food": "", "Water": ""}, "responder_gets": {"Firewood": "", "Food": "", "Water": ""}}, "action_type": "Accept-Deal"}
```

### Example 28 — PREDICT_FINAL_OUTCOME — `casino:PREDICT_FINAL_OUTCOME:dcb305e58f18dc5e`
```
201656f4a4] Ok, I am willing to give you one food, in exchange for two firewoods, that would mean you get 3 waters, 1 food and 1 firewood. you get 5 items, while i get 4.
[casino-participant-e6f678bfe5fa] If I only get 1 food, than I would like 2 firewood. So you get 2 food, 1 firewood, and 1 water.
[casino-participant-8f201656f4a4] No i do not need water, as i am camping near a potable stream. I would like 2 firewood. It is only fair that i get two firewood, since you are getting more supplies (5), i think i should be able to choose what 4 i should get.
[casino-participant-e6f678bfe5fa] No, if you get 2 food, it's only fair that I get 2 firewood. 
[casino-participant-8f201656f4a4] Then i will take 3 waters, and 2 food.
[casino-participant-e6f678bfe5fa] Okay I will take all the firewood and 1 food.

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"outcome": {"deal_reached": true, "opponent_likeness": {"mturk_agent_1": "Slightly dislike", "mturk_agent_2": "Slightly like"}, "points": {"mturk_agent_1": 19, "mturk_agent_2": 17}, "satisfaction": {"mturk_agent_1": "Extremely satisfied", "mturk_agent_2": "Slightly satisfied"}}, "outcome_type": "negotiation_result"}
```

### Example 29 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:483e1473b50edb9e`
```
ipant-1dd0263ed683
{"demographics": {"age": 26, "education": "some 4 year college, no degree", "ethnicity": "white american", "gender": "male"}, "personality": {"big-five": {"agreeableness": 1.5, "conscientiousness": 5.0, "emotional-stability": 4.5, "extraversion": 1.0, "openness-to-experiences": 6.5}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Water", "Low": "Food", "Medium": "Firewood"}, "reasons": {"High": "Water is key for keeping everyone hydrated. It's not easy to find water in some campgrounds.", "Low": "People can survive about 2 weeks without food. By that time if you're not rescued you're toast.", "Medium": "Pre cut firewood can be used in an emergency if people get hurt and can't walk very far."}}

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {}, "text": null}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Hey there!
```

### Example 30 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:f58da1d2effffee9`
```
"ethnicity": "white american", "gender": "female"}, "personality": {"big-five": {"agreeableness": 7.0, "conscientiousness": 7.0, "emotional-stability": 5.0, "extraversion": 5.0, "openness-to-experiences": 7.0}, "svo": "prosocial"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Water", "Low": "Firewood", "Medium": "Food"}, "reasons": {"High": "It's been very dry in the area and I plan to stay a couple extra days with my family, so we need to be fully prepared.", "Low": "Due it it being so dry, there is an abundance of dry wood that can be easy to scavenge for.", "Medium": "Due to the are being so dry there is no fruit or small animals in the area to hunt for.  The streams with water have also dried considerably leaving not many fish."}}

KNOWN HISTORY:
[casino-participant-1dd0263ed683] Hey there!

CURRENT OBSERVATION:
Hey there!

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Hi! How are you?! You excited for your camping trip??! I sure am ready to go on mine!
```

### Example 31 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:2cb74dc2be615903`
```
: 1.5, "conscientiousness": 5.0, "emotional-stability": 4.5, "extraversion": 1.0, "openness-to-experiences": 6.5}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Water", "Low": "Food", "Medium": "Firewood"}, "reasons": {"High": "Water is key for keeping everyone hydrated. It's not easy to find water in some campgrounds.", "Low": "People can survive about 2 weeks without food. By that time if you're not rescued you're toast.", "Medium": "Pre cut firewood can be used in an emergency if people get hurt and can't walk very far."}}

KNOWN HISTORY:
[casino-participant-1dd0263ed683] Hey there!
[casino-participant-c59c8ab0a149] Hi! How are you?! You excited for your camping trip??! I sure am ready to go on mine!

CURRENT OBSERVATION:
Hi! How are you?! You excited for your camping trip??! I sure am ready to go on mine!

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I am very excited, I'm actually going camping in a week. I drink a lot of water so it's important that I bring a lot. What about you?
```

### Example 32 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:5703582640adaf75`
```
 extra days with my family, so we need to be fully prepared.", "Low": "Due it it being so dry, there is an abundance of dry wood that can be easy to scavenge for.", "Medium": "Due to the are being so dry there is no fruit or small animals in the area to hunt for.  The streams with water have also dried considerably leaving not many fish."}}

KNOWN HISTORY:
[casino-participant-1dd0263ed683] Hey there!
[casino-participant-c59c8ab0a149] Hi! How are you?! You excited for your camping trip??! I sure am ready to go on mine!
[casino-participant-1dd0263ed683] I am very excited, I'm actually going camping in a week. I drink a lot of water so it's important that I bring a lot. What about you?

CURRENT OBSERVATION:
I am very excited, I'm actually going camping in a week. I drink a lot of water so it's important that I bring a lot. What about you?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I really am! I have been watching the weather and updates about the area I will be traveling to.  They are experiencing a severe drought, so I will be in need of some extra water as well! I planned on staying an extra couple days as well.  There is a stream nearby I believe, but I'm not sure how much it has dried up.
```

### Example 33 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:a4405937eda1f926`
```
ipant-1dd0263ed683] I am very excited, I'm actually going camping in a week. I drink a lot of water so it's important that I bring a lot. What about you?
[casino-participant-c59c8ab0a149] I really am! I have been watching the weather and updates about the area I will be traveling to.  They are experiencing a severe drought, so I will be in need of some extra water as well! I planned on staying an extra couple days as well.  There is a stream nearby I believe, but I'm not sure how much it has dried up.

CURRENT OBSERVATION:
I really am! I have been watching the weather and updates about the area I will be traveling to.  They are experiencing a severe drought, so I will be in need of some extra water as well! I planned on staying an extra couple days as well.  There is a stream nearby I believe, but I'm not sure how much it has dried up.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
That's good to hear, you always have to be careful with wild water though, it has all kind of germs. Sounds like you'll have plenty of dry firewood though. 
```

### Example 34 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:e500ee6085e9b4ca`
```
 excited, I'm actually going camping in a week. I drink a lot of water so it's important that I bring a lot. What about you?
[casino-participant-c59c8ab0a149] I really am! I have been watching the weather and updates about the area I will be traveling to.  They are experiencing a severe drought, so I will be in need of some extra water as well! I planned on staying an extra couple days as well.  There is a stream nearby I believe, but I'm not sure how much it has dried up.
[casino-participant-1dd0263ed683] That's good to hear, you always have to be careful with wild water though, it has all kind of germs. Sounds like you'll have plenty of dry firewood though. 

CURRENT OBSERVATION:
That's good to hear, you always have to be careful with wild water though, it has all kind of germs. Sounds like you'll have plenty of dry firewood though. 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yes, I would be willing to let you have all the firewood for 2 of the cases of water.  How much food were you needing?
```

### Example 35 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:ed2aa5d14b1ca48e`
```
bout you?
[casino-participant-c59c8ab0a149] I really am! I have been watching the weather and updates about the area I will be traveling to.  They are experiencing a severe drought, so I will be in need of some extra water as well! I planned on staying an extra couple days as well.  There is a stream nearby I believe, but I'm not sure how much it has dried up.
[casino-participant-1dd0263ed683] That's good to hear, you always have to be careful with wild water though, it has all kind of germs. Sounds like you'll have plenty of dry firewood though. 
[casino-participant-c59c8ab0a149] Yes, I would be willing to let you have all the firewood for 2 of the cases of water.  How much food were you needing?

CURRENT OBSERVATION:
Yes, I would be willing to let you have all the firewood for 2 of the cases of water.  How much food were you needing?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I like your water and firewood arrangement. Can I have 2 of the food since you're getting 2 of the water? 
```

### Example 36 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:cea2ffd8b1b55257`
```
aveling to.  They are experiencing a severe drought, so I will be in need of some extra water as well! I planned on staying an extra couple days as well.  There is a stream nearby I believe, but I'm not sure how much it has dried up.
[casino-participant-1dd0263ed683] That's good to hear, you always have to be careful with wild water though, it has all kind of germs. Sounds like you'll have plenty of dry firewood though. 
[casino-participant-c59c8ab0a149] Yes, I would be willing to let you have all the firewood for 2 of the cases of water.  How much food were you needing?
[casino-participant-1dd0263ed683] I like your water and firewood arrangement. Can I have 2 of the food since you're getting 2 of the water? 

CURRENT OBSERVATION:
I like your water and firewood arrangement. Can I have 2 of the food since you're getting 2 of the water? 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Let me think on that. I know there won't be much for berries or small animals to hunt for.  Is where you are going have much available for you to hunt and scavenge for?
```

### Example 37 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:3cc216e265be49af`
```
3] That's good to hear, you always have to be careful with wild water though, it has all kind of germs. Sounds like you'll have plenty of dry firewood though. 
[casino-participant-c59c8ab0a149] Yes, I would be willing to let you have all the firewood for 2 of the cases of water.  How much food were you needing?
[casino-participant-1dd0263ed683] I like your water and firewood arrangement. Can I have 2 of the food since you're getting 2 of the water? 
[casino-participant-c59c8ab0a149] Let me think on that. I know there won't be much for berries or small animals to hunt for.  Is where you are going have much available for you to hunt and scavenge for?

CURRENT OBSERVATION:
Let me think on that. I know there won't be much for berries or small animals to hunt for.  Is where you are going have much available for you to hunt and scavenge for?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
It might but with the summer season hunting isn't exactly legal at the moment. I tend to camp in remote areas so going to the store for more food isn't really an option. I've already let you have extra water so I don't think it's too far of a stretch to let me have some extra food.
```

### Example 38 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:fc0068fab454956c`
```
tting 2 of the water? 
[casino-participant-c59c8ab0a149] Let me think on that. I know there won't be much for berries or small animals to hunt for.  Is where you are going have much available for you to hunt and scavenge for?
[casino-participant-1dd0263ed683] It might but with the summer season hunting isn't exactly legal at the moment. I tend to camp in remote areas so going to the store for more food isn't really an option. I've already let you have extra water so I don't think it's too far of a stretch to let me have some extra food.

CURRENT OBSERVATION:
It might but with the summer season hunting isn't exactly legal at the moment. I tend to camp in remote areas so going to the store for more food isn't really an option. I've already let you have extra water so I don't think it's too far of a stretch to let me have some extra food.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Okay, I can let you have 2 of the food then.  I do appreciate your generosity on the extra water.  
```

### Example 39 — PREDICT_NEXT_ACTION — `casino:PREDICT_NEXT_ACTION:bad23e9fcfde2bb0`
```
the water? 
[casino-participant-c59c8ab0a149] Let me think on that. I know there won't be much for berries or small animals to hunt for.  Is where you are going have much available for you to hunt and scavenge for?
[casino-participant-1dd0263ed683] It might but with the summer season hunting isn't exactly legal at the moment. I tend to camp in remote areas so going to the store for more food isn't really an option. I've already let you have extra water so I don't think it's too far of a stretch to let me have some extra food.
[casino-participant-c59c8ab0a149] Okay, I can let you have 2 of the food then.  I do appreciate your generosity on the extra water.  

CURRENT OBSERVATION:
Okay, I can let you have 2 of the food then.  I do appreciate your generosity on the extra water.  

AVAILABLE ACTIONS:
["Accept-Deal", "Reject-Deal", "Submit-Deal", "Walk-Away", "continue-negotiating"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"proposer_gets": {"Firewood": "3", "Food": "2", "Water": "1"}, "responder_gets": {"Firewood": "0", "Food": "1", "Water": "2"}}, "action_type": "Submit-Deal"}
```

### Example 40 — PREDICT_NEXT_ACTION — `casino:PREDICT_NEXT_ACTION:cc1037cfdb509d54`
```
ilable for you to hunt and scavenge for?
[casino-participant-1dd0263ed683] It might but with the summer season hunting isn't exactly legal at the moment. I tend to camp in remote areas so going to the store for more food isn't really an option. I've already let you have extra water so I don't think it's too far of a stretch to let me have some extra food.
[casino-participant-c59c8ab0a149] Okay, I can let you have 2 of the food then.  I do appreciate your generosity on the extra water.  
[casino-participant-1dd0263ed683] ACTION Submit-Deal: {"proposer_gets": {"Firewood": "3", "Water": "1", "Food": "2"}, "responder_gets": {"Firewood": "0", "Water": "2", "Food": "1"}}

CURRENT OBSERVATION:
{"kind": "action", "meta": {"from_actor": "casino-participant-1dd0263ed683"}, "text": null}

AVAILABLE ACTIONS:
["Accept-Deal", "Reject-Deal", "Submit-Deal", "Walk-Away", "continue-negotiating"]

TARGET:

--- TARGET ---
{"acted": true, "action_content": {"proposer_gets": {"Firewood": "", "Food": "", "Water": ""}, "responder_gets": {"Firewood": "", "Food": "", "Water": ""}}, "action_type": "Accept-Deal"}
```

### Example 41 — PREDICT_FINAL_OUTCOME — `casino:PREDICT_FINAL_OUTCOME:9ac799ee4cb1450d`
```
e you needing?
[casino-participant-1dd0263ed683] I like your water and firewood arrangement. Can I have 2 of the food since you're getting 2 of the water? 
[casino-participant-c59c8ab0a149] Let me think on that. I know there won't be much for berries or small animals to hunt for.  Is where you are going have much available for you to hunt and scavenge for?
[casino-participant-1dd0263ed683] It might but with the summer season hunting isn't exactly legal at the moment. I tend to camp in remote areas so going to the store for more food isn't really an option. I've already let you have extra water so I don't think it's too far of a stretch to let me have some extra food.
[casino-participant-c59c8ab0a149] Okay, I can let you have 2 of the food then.  I do appreciate your generosity on the extra water.  

CURRENT OBSERVATION:
(none recorded)

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
{"outcome": {"deal_reached": true, "opponent_likeness": {"mturk_agent_1": "Extremely like", "mturk_agent_2": "Extremely like"}, "points": {"mturk_agent_1": 14, "mturk_agent_2": 23}, "satisfaction": {"mturk_agent_1": "Slightly satisfied", "mturk_agent_2": "Extremely satisfied"}}, "outcome_type": "negotiation_result"}
```

### Example 42 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:df4876172fa90b83`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=negotiator, id=casino-participant-07141d6d4473
{"demographics": {"age": 48, "education": "master's degree", "ethnicity": "white american", "gender": "female"}, "personality": {"big-five": {"agreeableness": 5.0, "conscientiousness": 6.5, "emotional-stability": 5.5, "extraversion": 2.0, "openness-to-experiences": 2.5}, "svo": "prosocial"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Food", "Low": "Firewood", "Medium": "Water"}, "reasons": {"High": "I have diabetes and my blood sugar get drop.", "Low": "I forgot to bring a thick coat and will get cold.", "Medium": "I get migraines if I don't drink enough water."}}

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {}, "text": null}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Hi, I'd like 3 packages of food. I have diabetes and my blood sugar could drop.
```

### Example 43 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:a631132512a0897d`
```
tor, id=casino-participant-85b963150c47
{"demographics": {"age": 33, "education": "some 4 year college, bachelor's degree", "ethnicity": "white american", "gender": "female"}, "personality": {"big-five": {"agreeableness": 3.5, "conscientiousness": 6.0, "emotional-stability": 4.5, "extraversion": 2.5, "openness-to-experiences": 5.0}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Firewood", "Low": "Food", "Medium": "Water"}, "reasons": {"High": "I have hypothyroidism which makes me cold.", "Low": "I could stand to lose some weight.", "Medium": "I enjoy hiking and staying hydrated."}}

KNOWN HISTORY:
[casino-participant-07141d6d4473] Hi, I'd like 3 packages of food. I have diabetes and my blood sugar could drop.

CURRENT OBSERVATION:
Hi, I'd like 3 packages of food. I have diabetes and my blood sugar could drop.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
oh dear, I am sorry to hear that my son is type one, I am okay with giving you all the food if you could give me all the firewood. I have hypothyroidism and it makes me get cold.
```

### Example 44 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:89bc67c05c7eb048`
```
 "prosocial"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Food", "Low": "Firewood", "Medium": "Water"}, "reasons": {"High": "I have diabetes and my blood sugar get drop.", "Low": "I forgot to bring a thick coat and will get cold.", "Medium": "I get migraines if I don't drink enough water."}}

KNOWN HISTORY:
[casino-participant-07141d6d4473] Hi, I'd like 3 packages of food. I have diabetes and my blood sugar could drop.
[casino-participant-85b963150c47] oh dear, I am sorry to hear that my son is type one, I am okay with giving you all the food if you could give me all the firewood. I have hypothyroidism and it makes me get cold.

CURRENT OBSERVATION:
oh dear, I am sorry to hear that my son is type one, I am okay with giving you all the food if you could give me all the firewood. I have hypothyroidism and it makes me get cold.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
That's a deal. 
```

### Example 45 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:fe631c7d6d3641ba`
```
ve": {"agreeableness": 3.5, "conscientiousness": 6.0, "emotional-stability": 4.5, "extraversion": 2.5, "openness-to-experiences": 5.0}, "svo": "proself"}}

PRIVATE STATE BEFORE:
{"preference_order": {"High": "Firewood", "Low": "Food", "Medium": "Water"}, "reasons": {"High": "I have hypothyroidism which makes me cold.", "Low": "I could stand to lose some weight.", "Medium": "I enjoy hiking and staying hydrated."}}

KNOWN HISTORY:
[casino-participant-07141d6d4473] Hi, I'd like 3 packages of food. I have diabetes and my blood sugar could drop.
[casino-participant-85b963150c47] oh dear, I am sorry to hear that my son is type one, I am okay with giving you all the food if you could give me all the firewood. I have hypothyroidism and it makes me get cold.
[casino-participant-07141d6d4473] That's a deal. 

CURRENT OBSERVATION:
That's a deal. 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
now how to split the water, wish it wasn't an odd number
```

### Example 46 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:822d1e4955837930`
```
VATE STATE BEFORE:
{"preference_order": {"High": "Food", "Low": "Firewood", "Medium": "Water"}, "reasons": {"High": "I have diabetes and my blood sugar get drop.", "Low": "I forgot to bring a thick coat and will get cold.", "Medium": "I get migraines if I don't drink enough water."}}

KNOWN HISTORY:
[casino-participant-07141d6d4473] Hi, I'd like 3 packages of food. I have diabetes and my blood sugar could drop.
[casino-participant-85b963150c47] oh dear, I am sorry to hear that my son is type one, I am okay with giving you all the food if you could give me all the firewood. I have hypothyroidism and it makes me get cold.
[casino-participant-07141d6d4473] That's a deal. 
[casino-participant-85b963150c47] now how to split the water, wish it wasn't an odd number

CURRENT OBSERVATION:
now how to split the water, wish it wasn't an odd number

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I'll take 1 and you can have 2 since you gave me all the food.
```

### Example 47 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:1c425f1e372cca35`
```
"Medium": "Water"}, "reasons": {"High": "I have hypothyroidism which makes me cold.", "Low": "I could stand to lose some weight.", "Medium": "I enjoy hiking and staying hydrated."}}

KNOWN HISTORY:
[casino-participant-07141d6d4473] Hi, I'd like 3 packages of food. I have diabetes and my blood sugar could drop.
[casino-participant-85b963150c47] oh dear, I am sorry to hear that my son is type one, I am okay with giving you all the food if you could give me all the firewood. I have hypothyroidism and it makes me get cold.
[casino-participant-07141d6d4473] That's a deal. 
[casino-participant-85b963150c47] now how to split the water, wish it wasn't an odd number
[casino-participant-07141d6d4473] I'll take 1 and you can have 2 since you gave me all the food.

CURRENT OBSERVATION:
I'll take 1 and you can have 2 since you gave me all the food.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Oh that is nice of you! I appreciate that very much! We do enjoy a good hike, so water will be good!
```

### Example 48 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:3ea78dce485233b5`
```
ater."}}

KNOWN HISTORY:
[casino-participant-07141d6d4473] Hi, I'd like 3 packages of food. I have diabetes and my blood sugar could drop.
[casino-participant-85b963150c47] oh dear, I am sorry to hear that my son is type one, I am okay with giving you all the food if you could give me all the firewood. I have hypothyroidism and it makes me get cold.
[casino-participant-07141d6d4473] That's a deal. 
[casino-participant-85b963150c47] now how to split the water, wish it wasn't an odd number
[casino-participant-07141d6d4473] I'll take 1 and you can have 2 since you gave me all the food.
[casino-participant-85b963150c47] Oh that is nice of you! I appreciate that very much! We do enjoy a good hike, so water will be good!

CURRENT OBSERVATION:
Oh that is nice of you! I appreciate that very much! We do enjoy a good hike, so water will be good!

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Sounds like we are set.
```

### Example 49 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:118011e5316be05e`
```
akes me cold.", "Low": "I could stand to lose some weight.", "Medium": "I enjoy hiking and staying hydrated."}}

KNOWN HISTORY:
... (1 earlier events elided)
[casino-participant-85b963150c47] oh dear, I am sorry to hear that my son is type one, I am okay with giving you all the food if you could give me all the firewood. I have hypothyroidism and it makes me get cold.
[casino-participant-07141d6d4473] That's a deal. 
[casino-participant-85b963150c47] now how to split the water, wish it wasn't an odd number
[casino-participant-07141d6d4473] I'll take 1 and you can have 2 since you gave me all the food.
[casino-participant-85b963150c47] Oh that is nice of you! I appreciate that very much! We do enjoy a good hike, so water will be good!
[casino-participant-07141d6d4473] Sounds like we are set.

CURRENT OBSERVATION:
Sounds like we are set.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
yeah, I hope you have a nice time camping! weather should be nice!
```

### Example 50 — PREDICT_NEXT_MESSAGE — `casino:PREDICT_NEXT_MESSAGE:786fe8d8441fee91`
```
": {"High": "I have diabetes and my blood sugar get drop.", "Low": "I forgot to bring a thick coat and will get cold.", "Medium": "I get migraines if I don't drink enough water."}}

KNOWN HISTORY:
... (2 earlier events elided)
[casino-participant-07141d6d4473] That's a deal. 
[casino-participant-85b963150c47] now how to split the water, wish it wasn't an odd number
[casino-participant-07141d6d4473] I'll take 1 and you can have 2 since you gave me all the food.
[casino-participant-85b963150c47] Oh that is nice of you! I appreciate that very much! We do enjoy a good hike, so water will be good!
[casino-participant-07141d6d4473] Sounds like we are set.
[casino-participant-85b963150c47] yeah, I hope you have a nice time camping! weather should be nice!

CURRENT OBSERVATION:
yeah, I hope you have a nice time camping! weather should be nice!

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I hope it will be because I don't love camping! Hope you have fun too!
```

## 25 most-suspicious examples (warnings / possible leakage)

- `casino:PREDICT_FINAL_OUTCOME:5aec8f1eb19925ca` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:dcb305e58f18dc5e` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:9ac799ee4cb1450d` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:5a47b1804eacb52b` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:c6566b8fbe0f5a2a` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:e5074523444ccc96` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:3560ee5e52656d3d` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:cb2cb4df5ed28994` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:547aed9d6358e514` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:fb6bed97d6f085b2` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:08120d52b90bfb12` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:fd6e63981a22690d` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:f8268d2b5bc4c969` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:2513893843c15c99` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:43ff5e82708315af` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:4b7b4f516dd9ab26` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:ccced8ae124faf5d` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:2fa9f21e8a1925a0` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:14a63d2b12eedc92` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:b2a2fecb7bd6b6c5` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:992327af56faed31` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:b785ac8b2584a0d9` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:b2713b30865039ca` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:b19db96687236cb7` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
- `casino:PREDICT_FINAL_OUTCOME:280ec58a43c61096` (PREDICT_FINAL_OUTCOME): warnings=['outcome (points/satisfaction) is post-negotiation ground truth'] possible_leakage=False
