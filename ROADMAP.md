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

Design corollary: the simulator's power comes from (a) **who** is in the population (VariableMaps), (b)
**what events** will hit and their impact, and (c) honest **uncertainty** over both — *not* from
extrapolating a belief curve.

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
| D. Event model | ❌ not built (the bottleneck) | EXP-033 proves this is THE gap. Need a model of *which events will occur* over the horizon and their impact distribution. Today we only score a *given* event (EXP-030). |
| E. Rollout engine | ◑ one-step + endogenous | one-step transition (EXP-030), unified individual/aggregate form (EXP-032), endogenous multi-step (EXP-033), bottom-up aggregation (EXP-034). Missing: the loop that samples event paths and composes them. |
| F. Outcome distribution | ◑ partial | Monte-Carlo band + conformal (one-step) exist. Missing: horizon calibration, pivotal-branch decomposition. |
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
- **The efficiency ceiling is real.** Where the belief source is efficient (liquid markets), even a
  perfect event model only helps *at* events; the honest product is calibrated probabilities + pivotal
  conditionals, not a crystal ball. Where the source is *inefficient* (slow social attitudes, niche
  questions), the model can add more — and that is where a general SWM is most differentiated.

## One-line status

State ✅ · cross-sectional Readout ✅ · one-step event Dynamics ✅ · unified individual/aggregate form ✅ ·
bottom-up aggregation ✅ — **missing: the future-event model and the Monte-Carlo rollout/decision loop
that turn these into "ask anything, simulate forward, choose the best action."** That is the build that
makes it a *general* simulator rather than a set of validated parts.
