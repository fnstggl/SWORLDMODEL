# EXP-060 — Level-1 individual simulator, validated on real persuasion data

**Level 1 of the three-level framework:** simulate ONE person as a little dynamical system — who they
are (stable variables) + how they are right now (mutable state) — and choose the best action on them.
This is the one level where "simulate the person" has no cheap substitute, and it's now built end-to-end
and earns its place on real data.

---

## The assembly

```
person → VariableMap (who they are) + mutable STATE (mood, busyness, load, reciprocity)
                          │
        message → ────────┤ response_fn(variables, state, message) → P(respond)
                          │      structured/grounded (validated here) | LLM-as-the-person (production)
                          ▼
   predict_response  ·  best_message (argmax do(x))  ·  simulate_thread (state carries forward)
```

Three new modules:

- **`swm/simulation/individual_agent.py`** — `IndividualAgent`: a person with `variables` (stable) and a
  mutable `state`. `apply(message, responded)` moves the state on first principles (a pushy/rude ask sours
  mood; a high-effort ask raises cognitive load; being contacted raises the sense of owing a reply;
  responding spends attention and discharges that debt). `relax()` decays the state back toward baseline
  as time passes. This is the part a static variable vector cannot do.
- **`swm/simulation/response_model.py`** — the response function. `StructuredResponseModel` fits a grounded
  logistic over four interpretable quantities: **receptivity** (the person's susceptibility), **quality**
  (the message's fit/force), their **interaction** (the crux — a strong argument moves an open mind far
  more than an entrenched one), and a grounded **state gate** (a worse transient state lowers the odds,
  zero at rest so it never disturbs the fit). `llm_response_fn` is the production backend — the LLM reasons
  *as the person*.
- **`swm/api/individual_simulate.py`** — the front door: `predict_response` ("how will A respond to this
  email?"), `best_message` ("what's the best email to send A for outcome X?" — the InterventionSelector's
  do(x), grounded in this specific person), `simulate_thread` (roll the person forward, state carrying
  over between contacts).

---

## Validation — ChangeMyView (real persuasion, 1,200 threads, temporal split)

Each row: an OP states a view, a challenger argues, the label is whether the OP was persuaded (awarded a
delta). Every row carries LLM-inferred **person** variables (openness, skepticism, entrenchment) and
**message** variables (addresses-crux, evidence, clarity, respectfulness, expertise). Fit on the past,
scored on the held-out future.

### A. Does modeling the PERSON beat modeling only the message?

| Arm | log-loss | brier | uplift@20 |
|---|---|---|---|
| base rate | 0.6440 | 0.2258 | 0.053 |
| message only (quality) | 0.6380 | 0.2231 | 0.094 |
| person only (receptivity) | 0.6309 | 0.2201 | 0.108 |
| **INDIVIDUAL (person × message)** | **0.6236** | **0.2169** | **0.108** |

**The person × message interaction earns its place: +0.0144 log-loss over message-only, +0.0204 over base
rate.** A message-quality score that ignores who is reading is strictly worse than one that reads the
argument *through* the person. (Consistent with the earlier EXP-021 finding that the LLM-inferred variable
map helped; here it's isolated to the interaction term.)

### B. Best message — the action layer, on a real natural experiment

For OPs who received **several** arguments with **mixed** outcomes, the person is fixed and only the
message varies — a natural do(x). Ranking each OP's arguments with the individual model and picking its
top one:

| | |
|---|---|
| OPs (held-out, mixed outcomes) | 23 |
| **model precision@1** | **0.739** |
| random-pick success rate | 0.518 |
| **lift** | **+0.221** |

**Picking the model's top-ranked argument persuades ~74% of the time vs ~52% for a random pick — a +22-point
causal lift.** This is `best_message` ("what's the best email to send this person") working on real data.

### C. The person as a dynamical system

Same closing ask, sent as a rapid follow-up after two different openers:

| Opener | P(respond to the identical closing ask) |
|---|---|
| pushy / rude | **0.572** |
| respectful / personalized | **0.643** |

The ask is byte-identical; only the state the opener left behind differs (mood, attention). A static
variable vector predicts the same number both times — this is why the person has to be a *dynamical
system*, not a fixed probability. The state gate is zero at the resting state (verified in tests), so this
effect is the *evolving state*, not a re-tuned baseline.

---

## Tests — `tests/test_individual_simulator.py` (9, all pass)

State initialization + dynamics (apply/relax move the state the right way), the person × message
interaction (the individual model discounts a strong message to a closed person; message-only can't), the
state gate being exactly zero at rest, `predict_response` / `best_message` / `simulate_thread`, and the
production `llm_response_fn` backend.

---

## Honest boundaries

- The **person × message coefficients are fit on real data**; the **state dynamics are grounded
  first-principles**, not fit, because CMV is one-shot (no within-thread state variation to identify them).
  They are validated as a *mechanism* (C) and would be calibrated on threaded reply data (a named data
  gap) — kept explicitly separate rather than hidden as a fitted knob.
- The gains are real but modest in absolute log-loss (persuasion is hard and noisy); the **best-message
  lift (+22 pts)** is the headline because choosing the right action is exactly the product.
- Production swaps the structured `response_fn` for `llm_response_fn` (the LLM reasons as the person)
  behind the identical interface — the validated harness calibrates the real thing.
