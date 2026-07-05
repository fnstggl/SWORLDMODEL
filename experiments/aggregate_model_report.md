# Aggregate model report (spec Phase 3)

**Question:** does an explicit **population state** that evolves — `PopulationState_t + Action →
Outcome + PopulationState_{t+1}` — predict aggregate/community response better than a static
content model, on real data, calibrated?

**Headline:** on **1,800 real HN stories (Feb–Apr 2026, time-ordered, no-cheat)** the aggregate
state-transition model beats a content-only model and the base rate, is **calibrated (grade A,
ECE 0.019)**, and its **free-running multi-step rollout does not catastrophically drift** at
horizons 1–4. The lift from state is **small but real** — exactly as EXP-005 found and honestly
consistent with it: at HN scale most signal is content/author-prior, and evolving state adds a real
but modest increment. Reproducible from `models/hn_aggregate.json` +
`experiments/results/exp010_aggregate_hn.json`.

## The state (what an aggregate query conditions on)
`swm/state/population.py::PopulationState` carries every field the spec asks for, each a posterior:
population base rate; **subgroup priors** (topic×format, topic, format, domain); **topic salience**
(fast EMA); **domain/source reputation** (slow EMA); **attention & competition**; recent-event
context; **incentives/stakes** (`swm/state/incentives.py` — stakes, controversy, novelty, reward
gradient, effort cost); optional **network/diffusion state** (`swm/state/graph.py` +
`swm/transition/diffusion.py`); and **nonstationarity/drift** indicators
(`swm/transition/nonstationarity.py::DriftTracker`).

`swm/transition/aggregate_transition.py::AggregateTransition` makes the state **actually enter the
prediction**: the calibrated head's feature vector includes the base-rate logit, the most-specific
subgroup rate, salience, reputation, competition, incentives, and the drift indicator — all read
from the *current* state. After each outcome, `transition()` updates all of them. (The content-only
ablation strips exactly the state features, so the comparison is clean.)

## Backtest (temporal split, target P(score ≥ 40), test = last 30% by time)
| model | log loss | ECE | uplift@20 |
|---|---|---|---|
| base rate (no model) | 0.3442 | — | — |
| content + time only | 0.3405 | 0.0406 | — |
| **+ evolving population state (FULL)** | **0.3362** | **0.0190** | 0.0352 |

Test base rate P(≥40) = 0.076. **State lowers held-out log loss (−0.0043 vs content-only, −0.0080
vs base) and more than halves ECE (0.041 → 0.019 → grade A).** The improvement is carried by the
stateful features (subgroup/reputation/salience), the same finding as EXP-005 — this is a state
model, not a feature model.

Honest magnitude: −1.3% relative log loss over content-only. Small. On HN the ceiling is low because
front-page ranking has a large irreducible luck component a title/state cannot reveal (the item-level
aleatoric floor discussed in EXP-003). The state machinery's value grows with (a) entity depth,
(b) multi-step questions, (c) domains where state genuinely evolves.

## Decision lift (the buyer's metric)
Ranking the held-out stories by the model's P(hit) and acting on the top-K:

| act on top | model hit-capture | random | oracle |
|---|---|---|---|
| 10% | 0.107 | 0.100 | 0.964 |
| **20%** | **0.268** | 0.200 | 1.000 |
| 30% | 0.375 | 0.300 | 1.000 |

**+5.4pp over random at top-20%** (capture 27% of winners in 20% of the effort). Modest — this is a
*pure statistical state model on a random sample of stories*, not the LLM reasoning about titles
(EXP-004's 50%-capture used Claude's language judgment on curated rounds). The two are complementary;
EXP-009 measures them head-to-head.

## Multi-step (the distinctive world-model claim, finally evaluated honestly)
`swm/simulation/rollout.py::calibration_by_horizon` runs the eval the prior repo admitted it never
did: for **held-out test-slice authors**, started from the **warm trained population**, it compares
- **teacher-forced** (state advanced by the ACTUAL outcomes — an upper bound), vs.
- **free-running** (state advanced by the model's OWN sampled outcomes — the real regime).

| horizon | teacher-forced log loss / ECE | free-running log loss / ECE |
|---|---|---|
| 1 | 0.122 / 0.115 | 0.122 / 0.115 |
| 2 | 0.117 / 0.110 | 0.108 / 0.103 |
| 3 | 0.275 / 0.028 | 0.268 / 0.023 |
| 4 | 0.119 / 0.112 | 0.113 / 0.107 |

**Free-running tracks teacher-forced closely at every horizon — it does not blow up.** That is the
key positive multi-step result: rolling the state forward on the model's own sampled outcomes does
not diverge from the ground-truth-fed rollout over 1–4 steps. Caveat (honest): the per-horizon
*absolute* numbers are dominated by the shifting realized hit-rate across the small per-author
cohorts (h3's spike is a cohort effect, not degradation), so this shows *no catastrophic drift*, not
a precise degradation curve — that still needs more per-author depth. The API therefore validates
`/v1/rollout` at horizon 1 (grade A) and labels deeper horizons honestly.

## Second domain
The **Manifold market** backtest (`experiments/market_harness.py`, market report) is the second
aggregate domain — market/event outcomes rather than community upvotes. It is where the model is
*supposed* to lose (a liquid market prices exactly the current information the model lacks), and it
does, honestly. Two aggregate domains, opposite regimes, both backtested.

## What is real vs. not
- **Real:** an evolving-population state model that beats content-only and the base rate on real HN,
  calibrated grade A, with the lift carried by stateful features; a persisted, reloadable fitted
  model wired into the API; a genuine free-running multi-step eval showing no short-horizon drift.
- **Not yet:** large lift (HN's ceiling is low); a precise multi-step degradation curve (needs depth);
  belief/stance-distribution transitions (proxied by salience/reputation, not measured directly);
  diffusion dynamics validated against real cascade sizes (built, not backtested — no networked-
  outcome wedge yet).
