# Audit — Werewolf Among Us

- **id**: `werewolf`  |  **role**: TRAIN_CANDIDATE  |  **status**: PENDING
- **official source**: https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us
- **paper**: https://aclanthology.org/2023.findings-acl.411/
- **license**: Apache-2.0 (text/annotations). Ego4D video subset under separate Ego4D license. (`permissive_commercial`) — commercial=yes, derivatives=yes
- **acquisition**: acquired (340 raw files, 5627482 bytes)

## Normalized data

- examples: **54632**  |  quarantined: 0  |  episodes: 199  |  actors: 933
- task counts: `{'PREDICT_NEXT_MESSAGE': 26911, 'PREDICT_TRAJECTORY_CONTINUATION': 26712, 'PREDICT_NEXT_ACTION': 818, 'PREDICT_FINAL_OUTCOME': 191}`
- split sizes: `{}`

## Distributions

- inactivity: `{'n_inactive': 0, 'n_action_or_response': 143, 'inactivity_rate': 0.0}`
- action types: `{'vote': 143}`
- outcomes: `{'None': 68}`
- response-time (s): `{}`
- context length (chars): `{'min': 235.0, 'p25': 2359.0, 'median': 2787.0, 'p75': 3057.0, 'p95': 3524.0, 'max': 4333.0, 'mean': 2550.44, 'n': 4000}`
- missing fields: `{'private_goal_hidden_role': 10046, 'video_features': 20000, 'explicit_winner_label': 68, 'strategy_annotation': 7}`

## Leakage

- result: `{'ok': None, 'n_records': 0}`

## Converter assumptions

- votingOutcome[i] is the index into playerNames of the player that player i voted for
- a game with no startRoles is an Avalon game

## Known limitations

- Avalon games (Ego4D/split/avalon.json) carry only dialogue — no players/votes/roles — so they yield only NEXT_MESSAGE + TRAJECTORY_CONTINUATION
- timestamps are video mm:ss strings (kept as event.t), not absolute wall-clock times

## Unavailable fields (stored null, never fabricated)

- PREDICT_BELIEF_CHANGE: private beliefs are not recorded (no before/after belief measurement); only observed votes and revealed roles exist — a belief-change label would have to be fabricated, so it is NOT emitted
- explicit game winner label (only revealed roles + votes are recorded; a winner is derivable via game rules but not provided, so it is not fabricated)
- a player's own start role in INPUT (deliberately withheld: roles are deception ground truth, exposed only in the FINAL_OUTCOME target)
- N/A votes (voted center / unrecorded) are not emitted as discrete vote actions
- video / audio / MViT numpy feature files (skipped)

## Recommendations

- **training**: Eligible for training pending human approval (training_approvals.yaml).
- **evaluation**: Usable for in-domain evaluation on its own test split.

## 50 rendered examples (human review)

### Example 1 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:cbdb45ac6b140a98`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
(no prior events)

CURRENT OBSERVATION:
{"kind": "state", "meta": {}, "text": null}

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Ready for the rising action.
```

### Example 2 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:692a44f429f6b4b6`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[werewolf-actor-a9751be3b249] Ready for the rising action.

CURRENT OBSERVATION:
Ready for the rising action.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yeah. Everyone ready?
```

### Example 3 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:05b44221a98c2e03`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[werewolf-actor-a9751be3b249] Ready for the rising action.
[werewolf-actor-963896794b09] Yeah. Everyone ready?

CURRENT OBSERVATION:
Yeah. Everyone ready?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yeah. Close your eyes.
```

### Example 4 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:cd3dc165612f90c0`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[werewolf-actor-a9751be3b249] Ready for the rising action.
[werewolf-actor-963896794b09] Yeah. Everyone ready?
[werewolf-actor-e8c809c922a6] Yeah. Close your eyes.

CURRENT OBSERVATION:
Yeah. Close your eyes.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Close your eyes. So now, Minions of Mordred will open their eyes and look for the other Minion. Minions of Mordred close your eyes. Put your thumbs up if you are a Minion. Now Merlin open your eyes. Minions of Mordred put your thumbs down. Merlin close your eyes. And now everyone open their eyes.
```

### Example 5 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:9c4cd41205f135ab`
```
RE:
(none recorded)

KNOWN HISTORY:
[werewolf-actor-a9751be3b249] Ready for the rising action.
[werewolf-actor-963896794b09] Yeah. Everyone ready?
[werewolf-actor-e8c809c922a6] Yeah. Close your eyes.
[werewolf-actor-963896794b09] Close your eyes. So now, Minions of Mordred will open their eyes and look for the other Minion. Minions of Mordred close your eyes. Put your thumbs up if you are a Minion. Now Merlin open your eyes. Minions of Mordred put your thumbs down. Merlin close your eyes. And now everyone open their eyes.

CURRENT OBSERVATION:
Close your eyes. So now, Minions of Mordred will open their eyes and look for the other Minion. Minions of Mordred close your eyes. Put your thumbs up if you are a Minion. Now Merlin open your eyes. Minions of Mordred put your thumbs down. Merlin close your eyes. And now everyone open their eyes.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Cool.
```

### Example 6 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:064793f7df44a67e`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[werewolf-actor-a9751be3b249] Ready for the rising action.
[werewolf-actor-963896794b09] Yeah. Everyone ready?
[werewolf-actor-e8c809c922a6] Yeah. Close your eyes.
[werewolf-actor-963896794b09] Close your eyes. So now, Minions of Mordred will open their eyes and look for the other Minion. Minions of Mordred close your eyes. Put your thumbs up if you are a Minion. Now Merlin open your eyes. Minions of Mordred put your thumbs down. Merlin close your eyes. And now everyone open their eyes.
[werewolf-actor-a1e6045d2a02] Cool.

CURRENT OBSERVATION:
Cool.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
All right.
```

### Example 7 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:7561ffe7bd870917`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
[werewolf-actor-a9751be3b249] Ready for the rising action.
[werewolf-actor-963896794b09] Yeah. Everyone ready?
[werewolf-actor-e8c809c922a6] Yeah. Close your eyes.
[werewolf-actor-963896794b09] Close your eyes. So now, Minions of Mordred will open their eyes and look for the other Minion. Minions of Mordred close your eyes. Put your thumbs up if you are a Minion. Now Merlin open your eyes. Minions of Mordred put your thumbs down. Merlin close your eyes. And now everyone open their eyes.
[werewolf-actor-a1e6045d2a02] Cool.
[werewolf-actor-e8c809c922a6] All right.

CURRENT OBSERVATION:
All right.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Well you're ready?
```

### Example 8 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:8e023badf5831f10`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (1 earlier events elided)
[werewolf-actor-963896794b09] Yeah. Everyone ready?
[werewolf-actor-e8c809c922a6] Yeah. Close your eyes.
[werewolf-actor-963896794b09] Close your eyes. So now, Minions of Mordred will open their eyes and look for the other Minion. Minions of Mordred close your eyes. Put your thumbs up if you are a Minion. Now Merlin open your eyes. Minions of Mordred put your thumbs down. Merlin close your eyes. And now everyone open their eyes.
[werewolf-actor-a1e6045d2a02] Cool.
[werewolf-actor-e8c809c922a6] All right.
[werewolf-actor-a9751be3b249] Well you're ready?

CURRENT OBSERVATION:
Well you're ready?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
We got it?
```

### Example 9 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:2aae8b88fc9356e7`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (2 earlier events elided)
[werewolf-actor-e8c809c922a6] Yeah. Close your eyes.
[werewolf-actor-963896794b09] Close your eyes. So now, Minions of Mordred will open their eyes and look for the other Minion. Minions of Mordred close your eyes. Put your thumbs up if you are a Minion. Now Merlin open your eyes. Minions of Mordred put your thumbs down. Merlin close your eyes. And now everyone open their eyes.
[werewolf-actor-a1e6045d2a02] Cool.
[werewolf-actor-e8c809c922a6] All right.
[werewolf-actor-a9751be3b249] Well you're ready?
[werewolf-actor-963896794b09] We got it?

CURRENT OBSERVATION:
We got it?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yep. Think so.
```

### Example 10 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:d9be90c829d0eebe`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (3 earlier events elided)
[werewolf-actor-963896794b09] Close your eyes. So now, Minions of Mordred will open their eyes and look for the other Minion. Minions of Mordred close your eyes. Put your thumbs up if you are a Minion. Now Merlin open your eyes. Minions of Mordred put your thumbs down. Merlin close your eyes. And now everyone open their eyes.
[werewolf-actor-a1e6045d2a02] Cool.
[werewolf-actor-e8c809c922a6] All right.
[werewolf-actor-a9751be3b249] Well you're ready?
[werewolf-actor-963896794b09] We got it?
[werewolf-actor-e8c809c922a6] Yep. Think so.

CURRENT OBSERVATION:
Yep. Think so.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Okay.
```

### Example 11 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:9545719622638198`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (4 earlier events elided)
[werewolf-actor-a1e6045d2a02] Cool.
[werewolf-actor-e8c809c922a6] All right.
[werewolf-actor-a9751be3b249] Well you're ready?
[werewolf-actor-963896794b09] We got it?
[werewolf-actor-e8c809c922a6] Yep. Think so.
[werewolf-actor-963896794b09] Okay.

CURRENT OBSERVATION:
Okay.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Part of this too.
```

### Example 12 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:0bf5cff1d2b93c98`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (5 earlier events elided)
[werewolf-actor-e8c809c922a6] All right.
[werewolf-actor-a9751be3b249] Well you're ready?
[werewolf-actor-963896794b09] We got it?
[werewolf-actor-e8c809c922a6] Yep. Think so.
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] Part of this too.

CURRENT OBSERVATION:
Part of this too.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Let's just put these below. So that way...
```

### Example 13 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:fda07b01a61560ef`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (6 earlier events elided)
[werewolf-actor-a9751be3b249] Well you're ready?
[werewolf-actor-963896794b09] We got it?
[werewolf-actor-e8c809c922a6] Yep. Think so.
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] Part of this too.
[werewolf-actor-e8c809c922a6] Let's just put these below. So that way...

CURRENT OBSERVATION:
Let's just put these below. So that way...

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
So who wants to start as the team leader. I guess. You.
```

### Example 14 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:676edab9bbbcefa0`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (7 earlier events elided)
[werewolf-actor-963896794b09] We got it?
[werewolf-actor-e8c809c922a6] Yep. Think so.
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] Part of this too.
[werewolf-actor-e8c809c922a6] Let's just put these below. So that way...
[werewolf-actor-963896794b09] So who wants to start as the team leader. I guess. You.

CURRENT OBSERVATION:
So who wants to start as the team leader. I guess. You.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Why don't you start?
```

### Example 15 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:38f740f6fdd3ba5b`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (8 earlier events elided)
[werewolf-actor-e8c809c922a6] Yep. Think so.
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] Part of this too.
[werewolf-actor-e8c809c922a6] Let's just put these below. So that way...
[werewolf-actor-963896794b09] So who wants to start as the team leader. I guess. You.
[werewolf-actor-a9751be3b249] Why don't you start?

CURRENT OBSERVATION:
Why don't you start?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Okay.
```

### Example 16 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:f18aff06cc9a9a9d`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (9 earlier events elided)
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] Part of this too.
[werewolf-actor-e8c809c922a6] Let's just put these below. So that way...
[werewolf-actor-963896794b09] So who wants to start as the team leader. I guess. You.
[werewolf-actor-a9751be3b249] Why don't you start?
[werewolf-actor-963896794b09] Okay.

CURRENT OBSERVATION:
Okay.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
You played it once before so. You've got the best experience.
```

### Example 17 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:942c88ce2428e6b9`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (10 earlier events elided)
[werewolf-actor-a9751be3b249] Part of this too.
[werewolf-actor-e8c809c922a6] Let's just put these below. So that way...
[werewolf-actor-963896794b09] So who wants to start as the team leader. I guess. You.
[werewolf-actor-a9751be3b249] Why don't you start?
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] You played it once before so. You've got the best experience.

CURRENT OBSERVATION:
You played it once before so. You've got the best experience.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
All right, so. I don't know exactly what the team leader is supposed to be.
```

### Example 18 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:c1576bd816382508`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (11 earlier events elided)
[werewolf-actor-e8c809c922a6] Let's just put these below. So that way...
[werewolf-actor-963896794b09] So who wants to start as the team leader. I guess. You.
[werewolf-actor-a9751be3b249] Why don't you start?
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] You played it once before so. You've got the best experience.
[werewolf-actor-963896794b09] All right, so. I don't know exactly what the team leader is supposed to be.

CURRENT OBSERVATION:
All right, so. I don't know exactly what the team leader is supposed to be.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I think 
```

### Example 19 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:452b5a1fc9dcba6b`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (12 earlier events elided)
[werewolf-actor-963896794b09] So who wants to start as the team leader. I guess. You.
[werewolf-actor-a9751be3b249] Why don't you start?
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] You played it once before so. You've got the best experience.
[werewolf-actor-963896794b09] All right, so. I don't know exactly what the team leader is supposed to be.
[werewolf-actor-e8c809c922a6] I think 

CURRENT OBSERVATION:
I think 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
These are supposed to be?
```

### Example 20 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:30d2d7ab80606e17`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (13 earlier events elided)
[werewolf-actor-a9751be3b249] Why don't you start?
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] You played it once before so. You've got the best experience.
[werewolf-actor-963896794b09] All right, so. I don't know exactly what the team leader is supposed to be.
[werewolf-actor-e8c809c922a6] I think 
[werewolf-actor-963896794b09] These are supposed to be?

CURRENT OBSERVATION:
These are supposed to be?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Success and fails. Are those team tokens?
```

### Example 21 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:49b8e367bf54371b`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (14 earlier events elided)
[werewolf-actor-963896794b09] Okay.
[werewolf-actor-a9751be3b249] You played it once before so. You've got the best experience.
[werewolf-actor-963896794b09] All right, so. I don't know exactly what the team leader is supposed to be.
[werewolf-actor-e8c809c922a6] I think 
[werewolf-actor-963896794b09] These are supposed to be?
[werewolf-actor-a9751be3b249] Success and fails. Are those team tokens?

CURRENT OBSERVATION:
Success and fails. Are those team tokens?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
No think that's, yeah.
```

### Example 22 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:d65851beabb9a1e5`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (15 earlier events elided)
[werewolf-actor-a9751be3b249] You played it once before so. You've got the best experience.
[werewolf-actor-963896794b09] All right, so. I don't know exactly what the team leader is supposed to be.
[werewolf-actor-e8c809c922a6] I think 
[werewolf-actor-963896794b09] These are supposed to be?
[werewolf-actor-a9751be3b249] Success and fails. Are those team tokens?
[werewolf-actor-e8c809c922a6] No think that's, yeah.

CURRENT OBSERVATION:
No think that's, yeah.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
They look like team tokens.
```

### Example 23 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:181cec28ec4ebf09`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a1e6045d2a02

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (16 earlier events elided)
[werewolf-actor-963896794b09] All right, so. I don't know exactly what the team leader is supposed to be.
[werewolf-actor-e8c809c922a6] I think 
[werewolf-actor-963896794b09] These are supposed to be?
[werewolf-actor-a9751be3b249] Success and fails. Are those team tokens?
[werewolf-actor-e8c809c922a6] No think that's, yeah.
[werewolf-actor-963896794b09] They look like team tokens.

CURRENT OBSERVATION:
They look like team tokens.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Okay.
```

### Example 24 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:9512a7c2237d23d2`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (17 earlier events elided)
[werewolf-actor-e8c809c922a6] I think 
[werewolf-actor-963896794b09] These are supposed to be?
[werewolf-actor-a9751be3b249] Success and fails. Are those team tokens?
[werewolf-actor-e8c809c922a6] No think that's, yeah.
[werewolf-actor-963896794b09] They look like team tokens.
[werewolf-actor-a1e6045d2a02] Okay.

CURRENT OBSERVATION:
Okay.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Oh that's probably what we play on the board.
```

### Example 25 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:871655836c8f0fd1`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (18 earlier events elided)
[werewolf-actor-963896794b09] These are supposed to be?
[werewolf-actor-a9751be3b249] Success and fails. Are those team tokens?
[werewolf-actor-e8c809c922a6] No think that's, yeah.
[werewolf-actor-963896794b09] They look like team tokens.
[werewolf-actor-a1e6045d2a02] Okay.
[werewolf-actor-a9751be3b249] Oh that's probably what we play on the board.

CURRENT OBSERVATION:
Oh that's probably what we play on the board.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Are these it?
```

### Example 26 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:f157e44871becc46`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (19 earlier events elided)
[werewolf-actor-a9751be3b249] Success and fails. Are those team tokens?
[werewolf-actor-e8c809c922a6] No think that's, yeah.
[werewolf-actor-963896794b09] They look like team tokens.
[werewolf-actor-a1e6045d2a02] Okay.
[werewolf-actor-a9751be3b249] Oh that's probably what we play on the board.
[werewolf-actor-e8c809c922a6] Are these it?

CURRENT OBSERVATION:
Are these it?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Those are score markers.
```

### Example 27 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:0afd02f24df6e98a`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (20 earlier events elided)
[werewolf-actor-e8c809c922a6] No think that's, yeah.
[werewolf-actor-963896794b09] They look like team tokens.
[werewolf-actor-a1e6045d2a02] Okay.
[werewolf-actor-a9751be3b249] Oh that's probably what we play on the board.
[werewolf-actor-e8c809c922a6] Are these it?
[werewolf-actor-963896794b09] Those are score markers.

CURRENT OBSERVATION:
Those are score markers.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
These?
```

### Example 28 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:8cc8e2f77365f87c`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (21 earlier events elided)
[werewolf-actor-963896794b09] They look like team tokens.
[werewolf-actor-a1e6045d2a02] Okay.
[werewolf-actor-a9751be3b249] Oh that's probably what we play on the board.
[werewolf-actor-e8c809c922a6] Are these it?
[werewolf-actor-963896794b09] Those are score markers.
[werewolf-actor-e8c809c922a6] These?

CURRENT OBSERVATION:
These?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yeah.
```

### Example 29 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:54549f9caedf6a76`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (22 earlier events elided)
[werewolf-actor-a1e6045d2a02] Okay.
[werewolf-actor-a9751be3b249] Oh that's probably what we play on the board.
[werewolf-actor-e8c809c922a6] Are these it?
[werewolf-actor-963896794b09] Those are score markers.
[werewolf-actor-e8c809c922a6] These?
[werewolf-actor-963896794b09] Yeah.

CURRENT OBSERVATION:
Yeah.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Okay.
```

### Example 30 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:f797a9a8775f335e`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (23 earlier events elided)
[werewolf-actor-a9751be3b249] Oh that's probably what we play on the board.
[werewolf-actor-e8c809c922a6] Are these it?
[werewolf-actor-963896794b09] Those are score markers.
[werewolf-actor-e8c809c922a6] These?
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-e8c809c922a6] Okay.

CURRENT OBSERVATION:
Okay.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
So then the tiny things-
```

### Example 31 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:6ab3a38d97295689`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (24 earlier events elided)
[werewolf-actor-e8c809c922a6] Are these it?
[werewolf-actor-963896794b09] Those are score markers.
[werewolf-actor-e8c809c922a6] These?
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-e8c809c922a6] Okay.
[werewolf-actor-963896794b09] So then the tiny things-

CURRENT OBSERVATION:
So then the tiny things-

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
These tiny...
```

### Example 32 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:4c4a8ed73871266c`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (25 earlier events elided)
[werewolf-actor-963896794b09] Those are score markers.
[werewolf-actor-e8c809c922a6] These?
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-e8c809c922a6] Okay.
[werewolf-actor-963896794b09] So then the tiny things-
[werewolf-actor-e8c809c922a6] These tiny...

CURRENT OBSERVATION:
These tiny...

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Are the build markers. So everyone gets one of them.
```

### Example 33 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:45d410f156fa1e6f`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (26 earlier events elided)
[werewolf-actor-e8c809c922a6] These?
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-e8c809c922a6] Okay.
[werewolf-actor-963896794b09] So then the tiny things-
[werewolf-actor-e8c809c922a6] These tiny...
[werewolf-actor-963896794b09] Are the build markers. So everyone gets one of them.

CURRENT OBSERVATION:
Are the build markers. So everyone gets one of them.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Or is it these?
```

### Example 34 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:cd9e12a4f8c9e3f7`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (27 earlier events elided)
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-e8c809c922a6] Okay.
[werewolf-actor-963896794b09] So then the tiny things-
[werewolf-actor-e8c809c922a6] These tiny...
[werewolf-actor-963896794b09] Are the build markers. So everyone gets one of them.
[werewolf-actor-e8c809c922a6] Or is it these?

CURRENT OBSERVATION:
Or is it these?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Loot dice.
```

### Example 35 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:207d5d79e9c4ef90`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (28 earlier events elided)
[werewolf-actor-e8c809c922a6] Okay.
[werewolf-actor-963896794b09] So then the tiny things-
[werewolf-actor-e8c809c922a6] These tiny...
[werewolf-actor-963896794b09] Are the build markers. So everyone gets one of them.
[werewolf-actor-e8c809c922a6] Or is it these?
[werewolf-actor-a9751be3b249] Loot dice.

CURRENT OBSERVATION:
Loot dice.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Okay. This is the leader token.
```

### Example 36 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:47040c239a31e27f`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (29 earlier events elided)
[werewolf-actor-963896794b09] So then the tiny things-
[werewolf-actor-e8c809c922a6] These tiny...
[werewolf-actor-963896794b09] Are the build markers. So everyone gets one of them.
[werewolf-actor-e8c809c922a6] Or is it these?
[werewolf-actor-a9751be3b249] Loot dice.
[werewolf-actor-963896794b09] Okay. This is the leader token.

CURRENT OBSERVATION:
Okay. This is the leader token.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Hi sweetie.
```

### Example 37 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:4e5b1cb4328c1459`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (30 earlier events elided)
[werewolf-actor-e8c809c922a6] These tiny...
[werewolf-actor-963896794b09] Are the build markers. So everyone gets one of them.
[werewolf-actor-e8c809c922a6] Or is it these?
[werewolf-actor-a9751be3b249] Loot dice.
[werewolf-actor-963896794b09] Okay. This is the leader token.
[werewolf-actor-e8c809c922a6] Hi sweetie.

CURRENT OBSERVATION:
Hi sweetie.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
So I guess I get...
```

### Example 38 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:e419fdb0ad19b2b9`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (31 earlier events elided)
[werewolf-actor-963896794b09] Are the build markers. So everyone gets one of them.
[werewolf-actor-e8c809c922a6] Or is it these?
[werewolf-actor-a9751be3b249] Loot dice.
[werewolf-actor-963896794b09] Okay. This is the leader token.
[werewolf-actor-e8c809c922a6] Hi sweetie.
[werewolf-actor-963896794b09] So I guess I get...

CURRENT OBSERVATION:
So I guess I get...

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Approves? Maybe this is it? Oh, it's just like a way to hide the notions. And project a crew.
```

### Example 39 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:c54606197df546fb`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (32 earlier events elided)
[werewolf-actor-e8c809c922a6] Or is it these?
[werewolf-actor-a9751be3b249] Loot dice.
[werewolf-actor-963896794b09] Okay. This is the leader token.
[werewolf-actor-e8c809c922a6] Hi sweetie.
[werewolf-actor-963896794b09] So I guess I get...
[werewolf-actor-a9751be3b249] Approves? Maybe this is it? Oh, it's just like a way to hide the notions. And project a crew.

CURRENT OBSERVATION:
Approves? Maybe this is it? Oh, it's just like a way to hide the notions. And project a crew.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Oh I see. I see. Yeah. So we get those vote tokens.
```

### Example 40 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:d434db2206d5ec3a`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (33 earlier events elided)
[werewolf-actor-a9751be3b249] Loot dice.
[werewolf-actor-963896794b09] Okay. This is the leader token.
[werewolf-actor-e8c809c922a6] Hi sweetie.
[werewolf-actor-963896794b09] So I guess I get...
[werewolf-actor-a9751be3b249] Approves? Maybe this is it? Oh, it's just like a way to hide the notions. And project a crew.
[werewolf-actor-963896794b09] Oh I see. I see. Yeah. So we get those vote tokens.

CURRENT OBSERVATION:
Oh I see. I see. Yeah. So we get those vote tokens.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Do we need... how many of each?
```

### Example 41 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:d5f52ef8abee3275`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (34 earlier events elided)
[werewolf-actor-963896794b09] Okay. This is the leader token.
[werewolf-actor-e8c809c922a6] Hi sweetie.
[werewolf-actor-963896794b09] So I guess I get...
[werewolf-actor-a9751be3b249] Approves? Maybe this is it? Oh, it's just like a way to hide the notions. And project a crew.
[werewolf-actor-963896794b09] Oh I see. I see. Yeah. So we get those vote tokens.
[werewolf-actor-a9751be3b249] Do we need... how many of each?

CURRENT OBSERVATION:
Do we need... how many of each?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Five, I guess. Yeah.
```

### Example 42 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:25a3dfebfc2e523a`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a1e6045d2a02

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (35 earlier events elided)
[werewolf-actor-e8c809c922a6] Hi sweetie.
[werewolf-actor-963896794b09] So I guess I get...
[werewolf-actor-a9751be3b249] Approves? Maybe this is it? Oh, it's just like a way to hide the notions. And project a crew.
[werewolf-actor-963896794b09] Oh I see. I see. Yeah. So we get those vote tokens.
[werewolf-actor-a9751be3b249] Do we need... how many of each?
[werewolf-actor-e8c809c922a6] Five, I guess. Yeah.

CURRENT OBSERVATION:
Five, I guess. Yeah.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
For each quest?
```

### Example 43 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:5b60ba89703fe44c`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (36 earlier events elided)
[werewolf-actor-963896794b09] So I guess I get...
[werewolf-actor-a9751be3b249] Approves? Maybe this is it? Oh, it's just like a way to hide the notions. And project a crew.
[werewolf-actor-963896794b09] Oh I see. I see. Yeah. So we get those vote tokens.
[werewolf-actor-a9751be3b249] Do we need... how many of each?
[werewolf-actor-e8c809c922a6] Five, I guess. Yeah.
[werewolf-actor-a1e6045d2a02] For each quest?

CURRENT OBSERVATION:
For each quest?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Five of each.
```

### Example 44 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:a84017819482e0fd`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (37 earlier events elided)
[werewolf-actor-a9751be3b249] Approves? Maybe this is it? Oh, it's just like a way to hide the notions. And project a crew.
[werewolf-actor-963896794b09] Oh I see. I see. Yeah. So we get those vote tokens.
[werewolf-actor-a9751be3b249] Do we need... how many of each?
[werewolf-actor-e8c809c922a6] Five, I guess. Yeah.
[werewolf-actor-a1e6045d2a02] For each quest?
[werewolf-actor-a9751be3b249] Five of each.

CURRENT OBSERVATION:
Five of each.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Yeah.
```

### Example 45 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:adfbb97d597b279a`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (38 earlier events elided)
[werewolf-actor-963896794b09] Oh I see. I see. Yeah. So we get those vote tokens.
[werewolf-actor-a9751be3b249] Do we need... how many of each?
[werewolf-actor-e8c809c922a6] Five, I guess. Yeah.
[werewolf-actor-a1e6045d2a02] For each quest?
[werewolf-actor-a9751be3b249] Five of each.
[werewolf-actor-963896794b09] Yeah.

CURRENT OBSERVATION:
Yeah.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
I don't think we have that many tokens.
```

### Example 46 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:65a1b34558f725ac`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (39 earlier events elided)
[werewolf-actor-a9751be3b249] Do we need... how many of each?
[werewolf-actor-e8c809c922a6] Five, I guess. Yeah.
[werewolf-actor-a1e6045d2a02] For each quest?
[werewolf-actor-a9751be3b249] Five of each.
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-a9751be3b249] I don't think we have that many tokens.

CURRENT OBSERVATION:
I don't think we have that many tokens.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Everyone will have a crew 
```

### Example 47 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:bd845e559de14db6`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-e8c809c922a6

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (40 earlier events elided)
[werewolf-actor-e8c809c922a6] Five, I guess. Yeah.
[werewolf-actor-a1e6045d2a02] For each quest?
[werewolf-actor-a9751be3b249] Five of each.
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-a9751be3b249] I don't think we have that many tokens.
[werewolf-actor-963896794b09] Everyone will have a crew 

CURRENT OBSERVATION:
Everyone will have a crew 

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
Whoa. And then what are these?
```

### Example 48 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:c1476362ac010933`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (41 earlier events elided)
[werewolf-actor-a1e6045d2a02] For each quest?
[werewolf-actor-a9751be3b249] Five of each.
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-a9751be3b249] I don't think we have that many tokens.
[werewolf-actor-963896794b09] Everyone will have a crew 
[werewolf-actor-e8c809c922a6] Whoa. And then what are these?

CURRENT OBSERVATION:
Whoa. And then what are these?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
What are these neutral cards?
```

### Example 49 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:84c396bbcb6fd992`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-963896794b09

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (42 earlier events elided)
[werewolf-actor-a9751be3b249] Five of each.
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-a9751be3b249] I don't think we have that many tokens.
[werewolf-actor-963896794b09] Everyone will have a crew 
[werewolf-actor-e8c809c922a6] Whoa. And then what are these?
[werewolf-actor-a9751be3b249] What are these neutral cards?

CURRENT OBSERVATION:
What are these neutral cards?

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
These are the team tokens. So these are what I would use to assign.
```

### Example 50 — PREDICT_NEXT_MESSAGE — `werewolf:PREDICT_NEXT_MESSAGE:78bcb9939dd83578`
```
TASK: PREDICT_NEXT_MESSAGE

ACTOR:
role=player, id=werewolf-actor-a9751be3b249

PRIVATE STATE BEFORE:
(none recorded)

KNOWN HISTORY:
... (43 earlier events elided)
[werewolf-actor-963896794b09] Yeah.
[werewolf-actor-a9751be3b249] I don't think we have that many tokens.
[werewolf-actor-963896794b09] Everyone will have a crew 
[werewolf-actor-e8c809c922a6] Whoa. And then what are these?
[werewolf-actor-a9751be3b249] What are these neutral cards?
[werewolf-actor-963896794b09] These are the team tokens. So these are what I would use to assign.

CURRENT OBSERVATION:
These are the team tokens. So these are what I would use to assign.

AVAILABLE ACTIONS:
<UNKNOWN_ACTION_SPACE>

TARGET:

--- TARGET ---
We just need one of approve and one reject?
```

## 25 most-suspicious examples (warnings / possible leakage)

- `werewolf:PREDICT_FINAL_OUTCOME:52d5be41b0fec6db` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:d0129b738b744d5b` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:8a8bb3d6366aad4c` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:04fa312d341f1df7` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:c04868be946cc72c` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:8154f37a0c92f494` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:3e3f22a6ef9651e4` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:d1cb39049f174fdd` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:cc8e2140fbcc95a5` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:b2233acba80bfe92` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:b2305cfd46c4e3ce` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:44d259a8c726f6c4` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:3fac9078b2f0a554` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:9285cb1ec3c40751` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:f5f897f8a7eb13ac` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:acf1a955b8881117` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:b3bfa1756ba799cc` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:d7fa6b957eced9a3` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:68808a417d6cbaff` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:65319b30b05c6cef` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:617ab4b2a27087c0` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:20f2c72269c586d0` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:5257b3d7c3406df1` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:df989188152c40fd` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
- `werewolf:PREDICT_FINAL_OUTCOME:52b032749925ed4c` (PREDICT_FINAL_OUTCOME): warnings=['roles are revealed ground truth, present only in the target'] possible_leakage=False
