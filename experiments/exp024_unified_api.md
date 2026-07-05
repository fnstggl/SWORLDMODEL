# EXP-024 — The unified `simulate()` API: is its confidence trustworthy? (the launch-readiness test)

EXP-020–023 validated the *architecture* (map known + inferred variables, predict from them) across
response, persuasion, and population opinion. But "launch-ready for **any** prediction" needs one more
thing: a single entry point that, pointed at an arbitrary query, returns a number **you can trust or
knowingly distrust**. That is `swm.api.Simulator.simulate()`. This experiment tests whether its
confidence and abstention actually mean something on real data.

## What the API adds over the raw `VariableWorld`
One call — `simulate(entity, action, context, user_context=…, llm_inference=…)` → `Prediction` — that,
on top of the calibrated probability, carries:
- **regime routing** — every query is classified into the regime that carries its signal:
  `entity_state` (WHO dominates — GitHub/Enron reply), `inference_driven` (no entity state; LLM-inferred
  latent variables carry it — persuasion), `message_only` (only cheap heuristics), `cold_start` (only
  population priors). Routing doesn't change the readout; it sets honest confidence and abstention.
- **honest confidence** — a ceiling from the regime + a within-regime position from **prediction
  extremity** (how far the model moves from the base rate).
- **an OOD guard** — `fit()` records which regimes the training stream actually covered; a query whose
  regime the model barely saw is discounted and abstained.
- **a safety shrink** — when abstaining, the probability is pulled toward the calibrated base rate so an
  out-of-envelope query can never emit an overconfident 0.00/1.00.
- **a calibration badge** — `fit()` grades calibration on a held-out temporal tail; the grade rides on
  every `Prediction` (a prediction without a grade is not allowed out the door).

## Test (no-cheat, real CMV persuasion corpus; n_test = 360; base rate ≈ 0.66)

### 1. Does the confidence track accuracy? (selective prediction)
Keep only the most-confident fraction of test predictions and re-measure log loss:

| coverage kept | n | log loss |
|---|---|---|
| 100% | 360 | 0.5972 |
| 75% | 270 | 0.5776 |
| 50% | 180 | 0.5593 |
| 25% | 90 | **0.4893** |

**Monotone — log loss falls by 0.038 from full coverage to the top-50% most-confident.** The confidence
score is usable as a launch dial: set a threshold and you know your reliability above it.

### 2. Does the abstain flag separate reliable from unreliable?
| | n | log loss |
|---|---|---|
| kept (not abstained) | 254 | **0.5728** |
| abstained | 106 | 0.6556 |

**Abstained predictions are measurably worse (+0.083 log loss)** — the flag correctly identifies the
queries the model shouldn't be trusted on.

### 3. Does the OOD guard catch out-of-envelope queries?
A model fit on this inference regime, handed a fabricated **entity-state** query it never trained on,
returns `regime=message_only, confidence=0.09, abstain=True` and shrinks toward the base rate — instead
of a confident wrong answer.

### An honest negative that shaped the design
The **first** confidence signal we tried — the LLM's *self-reported* confidence — did **not** track
accuracy (selective log loss was non-monotone: keeping the "most confident" 50% was *worse* than keeping
all). We measured this, discarded it, and switched to **prediction extremity**, which is monotone. The
lesson: a model's self-reported confidence is not a reliability estimate; distance-from-base-rate is.

## Why this matters
- It converts the validated architecture into a **deployable interface** with a trustworthy uncertainty
  contract — the concrete meaning of "launch-ready for any prediction."
- The confidence is **empirically validated**, not asserted: it orders predictions by accuracy, its
  abstention flags the hard cases, and its OOD guard refuses queries outside what the model was trained
  on. The safety shrink even *improves* aggregate log loss (0.597 vs 0.608 raw) by damping overconfident
  hard cases.

## Honest limits
- Single dataset (CMV, inference regime); the entity-state and cold-start regimes are exercised by unit
  tests and EXP-014/016 but not re-benchmarked here.
- Confidence is validated as an **ordering** (selective prediction) and via the calibration badge (ECE),
  not yet as a per-prediction interval (conformal prediction is the next uncertainty step).
- A `Simulator` is trustworthy for the regimes its training stream covered; multi-regime deployment
  means either one model per domain (as validated) or a stream that spans regimes. The OOD guard makes
  the boundary explicit rather than silent.

## Reproduce
`python -m experiments.exp024_unified_api` (uses the committed CMV inference artifacts).
`python -m pytest tests/test_api_simulate.py` covers routing, confidence, abstention, and the OOD guard.
