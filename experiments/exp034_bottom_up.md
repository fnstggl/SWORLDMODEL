# EXP-034 — Bottom-up simulate-and-aggregate vs top-down: does simulating individuals win?

The core hypothesis behind a *general* social world model: to predict a GROUP's collective opinion, is it
better to **simulate each heterogeneous individual** (from their VariableMap / value profile) and
aggregate — the Park et al. generative-agents bet — or to model the aggregate directly as one number?
This tests it no-cheat on OpinionQA.

## Setup (no-cheat)
Split respondents train/test. For each demographic GROUP (by ideology, party, religion, age) and each
question, predict the group's answer distribution three ways and compare to the group's TRUE distribution
(its members' actual answers, never used to predict):
- **top_down_global** — the population marginal ("the average person")
- **top_down_group** — the group's demographic-cell marginal from train
- **bottom_up** — simulate each group member individually (value-similarity to OTHER people, EXP-028) and
  average their predicted distributions

Metric: mean total-variation distance to the true group distribution (lower is better), over 1,035
(group, question) pairs.

## Result — bottom-up wins, and wins more where it should
| method | mean TV to true group dist ↓ |
|---|---|
| top-down global marginal | 0.1460 |
| top-down group-cell marginal | 0.1439 |
| **bottom-up (simulate individuals, aggregate)** | **0.1335** |

On the **distinctive groups** (the 579 group-questions whose true opinion is farthest from the global
average — where *who is in the group* should matter most):

| method | mean TV ↓ |
|---|---|
| top-down global marginal | 0.2225 |
| **bottom-up** | **0.1960** |

**Simulating individuals and aggregating beats modeling the aggregate directly — by ~9% overall and ~12%
on distinctive groups.** The gain concentrates exactly where the hypothesis predicts: for groups that
differ from the population average, composing the group out of its (simulated) members captures what a
single aggregate number cannot. It also beats the group-cell marginal, so it is not just "condition on
the group" — the individual-level composition adds.

## Why this matters
- It is the empirical warrant for building the **full bottom-up loop**: represent a population as
  VariableMaps, simulate each under the query/event, aggregate to the collective outcome. This experiment
  validates the *aggregation* half on real opinion data; combined with EXP-030's *event transition*, the
  path to "apply an event to a simulated population and read off the collective belief shift" is clear.
- It reconciles the two halves of the model: the aggregate is the composition of individuals, not a
  separate object — so State (VariableMap) and the population forecast are one system.

## Honest limits
- The win is real and consistent but **modest** (~9%/~12%), not the order-of-magnitude the strongest
  form of the hypothesis imagined — because on OpinionQA the individual predictions are demographic-
  driven, so bottom-up ≈ a well-composed demographic reweighting. Richer per-person profiles (text/
  history, EXP-025) should widen the gap; that is the next test.
- Cross-sectional (one opinion snapshot), not yet coupled to the event transition over time.
- Groups here are single demographic cells; finer composition (intersectional cells, or the full
  respondent set reweighted to a target population) is the productionization.

## Reproduce
`python -m experiments.exp034_bottom_up` (uses the committed OpinionQA parsed cache).
