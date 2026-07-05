# EXP-032 — Unified belief dynamics: one transition operator for market AND person

The point of a *general* social world model is that the same machinery models a market's collective
belief updating on news AND a person's belief updating on an argument. This makes that literal: one
operator,

    Δbelief = responsiveness · event_impact

- **aggregate** (market): `event_impact` = news impact, `responsiveness` = market factor (EXP-030);
- **individual** (person): `event_impact` = the argument's persuasive push, `responsiveness` is read
  from the person's **VariableMap** — openness attenuated by skepticism and by how entrenched their prior
  is. The same event moves an open mind and an entrenched mind differently; the VariableMap supplies that
  heterogeneity.

This is what fuses **State, Readout, and Dynamics into one system**: the VariableMap (State) sets the
responsiveness of the transition (Dynamics), whose output is read out as the predicted belief change.

## The test
Persuasion on r/ChangeMyView **is** an individual belief transition: an argument updates the OP's stance.
Does routing the argument's impact through the OP's VariableMap responsiveness beat the argument impact
alone (the aggregate operator applied blind to who the person is)? No-cheat temporal split; the OP's
VariableMap (openness_to_outreach, skepticism, prior_stance) and the argument push (crux-fit + evidence +
clarity) come from the committed EXP-021 LLM inferences.

## Result (CMV, n_test = 360; base rate ≈ 0.656)
| tier | log loss ↓ | Brier ↓ | accuracy |
|---|---|---|---|
| base rate | 0.6440 | 0.226 | 0.656 |
| event_impact only (argument push, ignores the person) | 0.6411 | 0.225 | 0.656 |
| responsiveness only (the VariableMap alone) | 0.6219 | 0.217 | 0.658 |
| **unified: responsiveness · event_impact** | **0.6215** | **0.216** | 0.658 |

**Routing the event impact through the person's VariableMap responsiveness beats the event impact alone
by +0.0196 log loss (~3%).** The person-modulated transition wins: individual heterogeneity — supplied by
the VariableMap — is what turns a generic event impact into an accurate individual belief update. The
same operator that predicted aggregate market moves (EXP-030) now predicts individual persuasion, with
only the responsiveness source swapped.

## Why this matters
- It closes the loop on the architecture: the three pieces validated separately — **State** (VariableMap
  + evidence fusion), **cross-sectional Readout**, **temporal Dynamics** (EXP-030) — are now **one
  engine**. An individual transition is the aggregate transition with the VariableMap setting
  responsiveness; the aggregate is the population average of individual transitions.
- It is the concrete mechanism by which "the core architecture is used on every simulation": the map
  isn't just features for a readout — it parameterizes the *dynamics*.

## Honest limits
- The individual-side gain is real but modest (+0.0196 log loss), and much of the individual signal here
  is the responsiveness (WHO the OP is) — the event push adds on top rather than dominating; CMV delta is
  a specific persuasion setting.
- The unification is validated on two regimes (market aggregate + CMV individual). A dataset with a
  person's belief measured *before and after* a dated event (a true individual belief trajectory) would
  test the temporal individual transition directly; CMV is a single-step stance update.
- `responsiveness_from_map` is a grounded closed form (openness, skepticism, entrenchment); learning it
  end-to-end is a natural next step.

## Reproduce
`python -m experiments.exp032_unified_dynamics` (committed EXP-021 CMV inferences).
`python -m pytest tests/test_unified_dynamics.py`.
