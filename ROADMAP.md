# Roadmap — a general social world model simulator

**The goal.** Take an arbitrary natural-language question ("Where is LeBron more likely to win a title,
Miami or Cleveland, as of today?" / "Will the Fed cut in March?" / "Will this user churn if we email
them?"), automatically construct the belief state, map every variable acting on it, roll the simulation
forward under uncertainty, and return a **calibrated distribution over outcomes** — and the **best action**
to reach a desired outcome. This document maps everything needed to get there, grounded in what we have
built and, crucially, in what the no-cheat experiments have proven about what's hard.

## What the experiments already settled (these constrain the whole design)

1. **State + cross-sectional readout work** (EXP-014/016/021/023/028): mapping a person/population to a
   VariableMap and reading out a response/opinion beats strong baselines, no-cheat.
2. **The one-step event transition works** (EXP-030): an LLM-judged event impact predicts the *next*
   belief move; on genuine event steps it calls direction 85% right vs a martingale's 0%.
3. **Multi-step extrapolation of dynamics FAILS** (EXP-033): prediction-market belief is a near-martingale
   — *nothing beats "belief stays put"* over 1–10 days, and learned momentum/mean-reversion make it
   worse. **The one-step event edge is local; it does not survive rollout.** ⇒ Long-horizon forecasting
   is a **future-event-forecasting problem, not a better-dynamics problem.**
4. **Bottom-up aggregation beats top-down** (EXP-034): simulating individuals from VariableMaps and
   aggregating beats modeling the aggregate as one number (~9% overall, ~12% on distinctive groups). ⇒
   Build the population as a set of individuals, not a scalar.
5. **Grounded-variable simulation beats the crowd composite — and the frontier is estimation, not
   enumeration** (EXP-040): on a real social outcome (OpinionQA population shares), mapping each person's
   *real* variables and simulating their answer beats the aggregate composite (individual log-loss
   0.612 → 0.585; accuracy 0.656 → 0.690), and beats the top-down aggregate by **24%** on distinctive
   subgroups. **Crucially, "map more variables" only helps once you can *estimate* their joint effect** —
   naive Bayes over 11 correlated variables (party≈ideology) overfit and was *worse* than one variable
   (log-loss 0.92 vs 0.60) until regularized. ⇒ The differentiated capability (compute an outcome from its
   constituents, beating a single aggregate number) is **real**; the binding constraint is variable
   *estimation quality* (grounding, partial pooling, correlation-aware readouts), not variable count.

6. **The estimator, not the variable list, is the fix** (EXP-041): a correlation-aware, partially-pooled
   readout makes "add more variables" safe — where naive Bayes and un-pooled logistic *collapse* as
   variables grow (log-loss 0.61 → 0.74 / → 0.86), the pooled readout stays flat (0.61 → 0.606) and cuts
   data-poor-question error **29%** (0.85 → 0.60). Partial pooling toward the prior is the load-bearing
   piece. ⇒ `swm/variables/pooled_readout.py` is the deployable grounded-variable estimator.
7. **The forward operator works; its structure earns mechanism, not aggregate accuracy** (EXP-042): on
   real opinion-change events (CMV), coupling a grounded actor to the event-transition operator beats
   persistence (+0.025 log-loss) and one-sided baselines, and the **gating mechanism is verified** — the
   same argument predicts change ~38% more for responsive than entrenched people (slope 0.71 vs 0.51).
   But the multiplicative coupling doesn't separably beat an additive model at this scale/noise. ⇒ The
   operator is validated per-step; the interaction is real but needs grounded (low-noise) variables and
   more data to pay in aggregate.
8. **Grounding is necessary but not sufficient — content extraction is the frontier** (EXP-043): crude
   features from the *real* as-of news (volume, result-cue, polarity) also fail to beat the base rate
   (corr 0.047), even though the market extracts the same articles into a decisive signal (corr 0.84).
   The LLM gestalt (EXP-037) and shallow real-news reading fail for the same reason from opposite ends. ⇒
   The bottleneck for question-level forecasting is **reading real content well** (entity-linked,
   resolution-aware stance detection), not listing or nominally-grounding drivers.

9. **A grounded POPULATION is not a martingale — multi-step forward simulation beats persistence**
   (EXP-045, GSS 1972–2024, 72,707 respondents, 15 items, 406 rolling-origin forecasts): where a market
   belief cannot be rolled forward past persistence (item 3), a grounded population *can* — composing the
   opinion from demographic cells at the target year's composition beats persistence (MAE 0.0264 vs
   0.0288 overall) and **the edge grows with horizon** (4–7y: 0.0288 vs 0.0419, −31%), with change
   directional accuracy 0.593 vs 0.480. A population has predictable structure (evolving composition) a
   price does not. ⇒ The "simulate forward N steps" thesis is validated on real longitudinal data — for
   outcomes that are the aggregate of a modelable, evolving population, exactly where no market exists.
10. **Resolution-aware content extraction recovers more signal but not enough — the frontier is semantic**
    (EXP-044): entity-linked, resolution-aware lexical stance recovers 1.26× the raw outcome-correlation
    of crude features (0.112 vs 0.088) but only **13%** of what the market extracts (0.84), and doesn't
    beat the base rate in calibrated prediction. ⇒ Confirms item 8: the bottleneck is semantic stance
    detection against the specific resolution criterion (embeddings / LLM judge), not lexical features.

11. **Opinion change decomposes into composition (near-term) + period drift (far-term)** (EXP-046, GSS):
    coupling the compositional rollout (item 9) with a forecast of the composition-removed period residual
    helps *exactly at long horizons* — 4–7y MAE 0.0251 vs compositional-only 0.0288 (−13%) and persistence
    0.0419 (−40%) — but adds noise near-term, so compositional-only stays best overall. Responsiveness-
    gating the period shock (the EXP-042 operator at the aggregate) adds nothing. ⇒ A production population
    forecaster should weight the period term **by horizon** (off near-term, on far-term); the individual
    gating mechanism is real per-person but washes out in the aggregate share.

Design corollary: the simulator's power comes from (a) **who** is in the population (VariableMaps), (b)
**what events** will hit and their impact, and (c) honest **uncertainty** over both — *not* from
extrapolating a belief curve. And its *edge over the crowd* comes from **well-estimated grounded
variables** simulated bottom-up, most of all where the population is heterogeneous — not from a longer
variable list, and not from predicting a market's price (which is a near-martingale — item 3). The two
regimes are now cleanly separated by evidence: a **price is a martingale** (item 3, don't roll it
forward), a **grounded population is not** (item 9, roll it forward — the edge grows with horizon). The
concrete frontiers, in order: **estimation** (done — item 6), **grounded low-noise variables** (item 7),
**real-content extraction** (items 8/10, the current bottleneck — now known to require *semantic* stance,
not lexical), and **coupling population rollout to period/event shocks** (item 9's next step, joining
EXP-045's compositional dynamics with EXP-042's event operator).

## The honest north-star boundary (corrected)
An earlier framing over-claimed that you "cannot beat a liquid market's probability." That conflated two
different claims: you cannot predict a liquid market's price *path* (item 3, a near-martingale), but a
**structural bottom-up simulation is a different information source than the price** — it computes the
outcome from its constituent decisions, which the crowd's noisy aggregate only approximates. EXP-040 is
the first direct evidence that this simulation *beats* the aggregate composite on a real outcome. On the
most liquid mega-markets professional modelers already do this, so the price impounds it and the edge is
thin; but on the vast space of **off-market / niche / individual-scale** questions no one has run the
simulation, and grounded bottom-up simulation is both the only method available and the entire value
proposition. The project's job is that simulation — not a price-prediction heuristic.

## The target architecture (pipeline), with build status

```
 NL question
   │  [A] QUESTION INTAKE  →  proposition, resolution criterion, horizon, relevant entities/segments
   ▼
 [B] STATE CONSTRUCTION  →  current belief s_0 (retrieved market/poll/prior) + the population of
   │                        VariableMaps (individuals or demographic segments) acting on it
   ▼
 [C] VARIABLE MAPPING    →  for each actor, the known+inferred variables (VariableMap + EvidenceFusion)
   ▼
 [D] EVENT MODEL         →  the distribution of FUTURE events over the horizon + each event's impact
   ▼
 [E] ROLLOUT ENGINE      →  Monte-Carlo over event paths; per step apply the transition (per actor,
   │                        modulated by their VariableMap responsiveness), aggregate bottom-up
   ▼
 [F] OUTCOME DISTRIBUTION → calibrated P(outcome), widening bands, pivotal-branch decomposition
   ▼
 [G] DECISION / ACTION   →  for each candidate action, expected utility over F; argmax = best action
```

| stage | status | what exists / what's missing |
|---|---|---|
| A. Question intake | ❌ not built | need: LLM parses question → proposition + resolution + horizon + entities. The "front door." |
| B. State construction | ◑ partial | retrieval scaffolding exists (`swm/retrieval/`); VariableMap/EvidenceFusion build actor state. Missing: auto-retrieve the current belief (market/poll) for an arbitrary proposition, and instantiate the population. |
| C. Variable mapping | ✅ built | VariableMap + inference + EvidenceFusion + deep per-person inference (EXP-020/25/29). |
| D. Event model | ◑ variance/timing built; direction open | EXP-035: forecasts the *distribution* of belief moves (heteroskedastic variance) — a calibrated predictive distribution that beats persistence 24% / a constant band 14% on CRPS. Key finding: for an efficient series you forecast event *variance/timing*, not *direction* (direction is unforecastable — EXP-033). Open: an event *calendar* (known future dates) to place variance even better; directional forecasting only where the series is inefficient. |
| E. Rollout engine | ✅ distributional MC built | one-step transition (EXP-030), unified individual/aggregate form (EXP-032), Monte-Carlo distributional rollout with heteroskedastic variance (EXP-035), bottom-up aggregation (EXP-034). Open: pivotal-branch decomposition for multimodal futures; coupling the population rollout to the event rollout. |
| F. Outcome distribution | ◑ calibrated one-var; multivar open | EXP-035: CRPS-scored, horizon-calibrated interval bands (80% coverage) — a proper predictive distribution per horizon. Open: pivotal-branch conditionals; joint distributions across coupled questions. |
| G. Decision/action | ❌ not built | counterfactual scaffolding (`swm/simulation/counterfactuals.py`); need action → outcome-distribution → expected-utility argmax. |

## The hard core: future events and branching realities

EXP-033 is unambiguous: **you cannot forecast forward without forecasting the events.** This is the
branching-realities problem. Worked example:

> **"Will the Fed cut rates at the March 2026 meeting?"** Today: 55%. Between now and March, pivotal
> events each split the future: Jan jobs report (strong→↓ / weak→↑), Feb CPI (hot→↓ / cool→↑), Feb FOMC
> (hold→↓ / cut→↑). With *n* binary pivots there are 2ⁿ reality branches — combinatorial by months out.

You do **not** enumerate 2ⁿ, and you do **not** pick one branch. Three tractable architectures, in
increasing sophistication:

1. **Monte-Carlo trajectory sampling (the default).** Sample K trajectories. In each, step T+1 by day:
   draw whether a pivotal event fires and its outcome from the **event model (D)**, apply the transition
   (E), continue to the horizon. The K terminal beliefs form the forecast; `P(cut) = fraction of
   trajectories resolving "cut."` Scales **linearly in K**, not 2ⁿ; gives a calibrated distribution;
   uncertainty widens naturally. **This is how "vast simulation space" reduces to percentages.**
2. **Particle filter / beam over realities.** Keep M weighted "live realities" (belief-state,
   probability). At each pivot, branch each by the outcomes, reweight by outcome probability, then
   resample/prune back to M — concentrating compute on plausible worlds. Better when a few branches
   dominate.
3. **Moment propagation.** Propagate only mean+variance analytically (events as calibrated noise).
   Cheapest, but **loses multimodality** — so use it only when the future is unimodal.

**Multimodality is the real trap.** When a single pivot makes the future genuinely bimodal (e.g. "if the
Feb FOMC holds → 25%; if it cuts → 85%"), the *mean* (≈55%) is a lie no one should act on. The honest
output is the **pivotal-branch decomposition**: surface the branch explicitly — "conditional on the Feb
decision: 25% / 85%, and here's P(cut in Feb)" — rather than averaging over it. The rollout engine (E)
must detect high-variance pivots and report conditionals, not just a marginal.

**Decisions never collapse the distribution.** For "best action to reach a desired outcome," run the
ensemble **once per candidate action** (the action is an intervention in the rollout), get an outcome
distribution per action, and choose `argmax_a  E[utility | action a]` (or `argmax_a P(desired | a)`).
The branching is handled by the ensemble; the decision is an expected-utility argmax over percentages —
never a single predicted reality.

## Build order (dependency-ranked, each with a no-cheat test)

1. **Event model (D) — the unlock.** Two sub-parts: (i) *event forecasting* — over a horizon, the
   distribution of pivotal events and their timing (calendar events like FOMC/elections are known;
   surprises are a base-rate hazard model); (ii) *event→impact* — reuse EXP-030's LLM channel per
   sampled event. Test: multi-step rollout **with sampled future events** vs persistence on SWM-Bench
   futures — the real version of EXP-033. This is where long-horizon accuracy is won or lost.
2. **Rollout engine (E) — the Monte-Carlo loop** composing D + the unified transition (EXP-032) + bottom-
   up aggregation (EXP-034), with pivotal-branch decomposition. Test: horizon calibration + does it beat
   persistence *given* the event model.
3. **Horizon uncertainty (F).** Conformalize the terminal distribution per horizon; bands must hit their
   coverage at each h (EXP-033's band was over-wide — recalibrate). Test: coverage vs horizon.
4. **Question intake + state construction (A/B) — the front door.** LLM parses the question; auto-retrieve
   the current belief (Kalshi/Polymarket/poll/prior) and instantiate the population. Test: end-to-end on
   held-out resolved questions (incl. a real NBA-champion market, no-cheat as-of).
5. **Individual temporal transition.** Validate the person-level rollout directly — needs a dataset of
   individual belief *before/after dated events* (**ANES panel**, **USC Understanding America Study**;
   both registration-gated, not on HF). With it: learn `responsiveness_from_map` end-to-end (today it's a
   grounded closed form, EXP-032). Test: predict a held-out person's post-event belief.
6. **Decision/action layer (G).** Action as a rollout intervention; expected-utility argmax over the
   outcome distribution. Test: on data with observed interventions (A/B tests, outreach) — does the
   recommended action raise the desired-outcome rate?

## Honest feasibility

- **Reachable now** on public data: #1–#3 on SWM-Bench (weeks-ahead), #4's retrieval for market-backed
  questions. A short-horizon, no-cheat NBA-championship-*market* backtest is doable (pull the market +
  dated news); a trustworthy months-out "who wins 2026" forecast is **not** until #1 is built and shown
  to beat persistence with a real future-event model.
- **Data-blocked**: #5 (individual temporal — gated panels), long-horizon (months/years — SWM-Bench caps
  at ~16 days; need longer trajectories).
- **The efficiency ceiling is real — but direction is still forecastable.** Where the belief source is
  efficient (liquid markets) you cannot beat its *probability* on the point (EXP-033/035). BUT the
  *direction* implied by that probability is correct at the calibration rate — EXP-036: the lean predicts
  the move direction at 0.6–0.8 (0.85 on confident beliefs), far above chance; momentum is useless. So the
  honest product is calibrated probabilities + a directional call + pivotal conditionals. For questions
  with **no market**, the whole game is *inferring the lean (P(outcome)) from the drivers* — the VariableMap
  applied to the question; the direction then follows. That inference is the highest-leverage open build,
  and it is where a general SWM is most differentiated from just reading a market.

## One-line status

State ✅ · cross-sectional Readout ✅ · one-step event Dynamics ✅ · unified individual/aggregate form ✅ ·
bottom-up aggregation ✅ · **calibrated multi-step distributional rollout ✅ (EXP-035)** — **missing: the
question-intake front door (A/B), pivotal-branch decomposition, and the decision/action layer (G) that
turn these into "ask anything, simulate forward, choose the best action."** The forecasting core now
produces calibrated distributions over horizons; what remains is the front door (parse a question →
construct state) and the back door (outcome distribution → best action).
