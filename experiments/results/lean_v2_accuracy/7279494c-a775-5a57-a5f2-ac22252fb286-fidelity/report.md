# Lean V2 accuracy run — 7279494c-a775-5a57-a5f2-ac22252fb286-fidelity

**Question:** Will the Bank of Japan raise its short-term policy interest rate at the June 15–16, 2026 Monetary Policy Meeting?

**Status:** completed | **probability:** 0.0385 | **source:** deliberative_institution_vote | **grounding:** partially_grounded | **confidence:** low


## Causal world

- thesis: The BOJ Policy Board votes on a proposal to raise the uncollateralized overnight call rate from 0.75% to 1.0%. A simple majority of the 9 members is required. The outcome hinges on whether at least 5 members vote for the hike, given the 6-3 split at the prior meeting with three dissenting hawks. Mar
- actors: 5 | action templates: 1

## Shared world conditions

- **boj_gradual_hiking_cycle**: The Bank of Japan is in a gradual tightening cycle, having raised rates from negative territory to 0.75% since March 2024, with a cautious pace dependent on data and inflation outlook. — counted rate 0.9 (n=4)
- **dissenting_votes_on_hold**: At the April 2026 meeting, three board members dissented in favor of a hike to 1.0%, indicating internal pressure for tightening. — counted rate 0.75 (n=1)
- **market_expectations_elevated**: Market-implied probability of a June hike is around 63% as of mid-May 2026, with Reuters reporting the BOJ has signaled a possible hike. — counted rate 0.75 (n=1)

## Forecast decomposition

- prior_forecast (grounded prior): 0.9 (n=4, source=counted_outcome_reference_class)
- simulation_forecast (conditional on resolved mass): 0.0385 (resolved mass 1.0)
- simulation probability bounds (residual-widened): [0.0, 0.5853] (residual bound 0.58528)
- headline_forecast: 0.0385 via deliberative_institution_vote
- prior/simulation disagreement: 0.8615
  - no leakage-audited reliability combiner is fitted — prior and simulation are reported separately with the feasible combined range; no fixed blend is applied and Lean V2 must not become default on this basis

## Simulation completion audit

- resolved mass: 0.999999 of 0.999999 (target ≥0.8 met: True)
- unresolved by cause: {}
- unknown-state terminal mass: 0.0 (must be 0: True)
- missing-mechanism terminal mass: 0.0 (ok: True)
- readiness verdict: ready | round-trip ok: True

## Unresolved mass by cause


## Cost

- calls: 32 | wall: 220.37s | peak nodes: 3456
- deliberations: 8 | challenger: False
- limitation: forecast decomposition — grounded prior 0.9 (counted n=4); simulation-conditional 0.0385 (resolved mass 1.00); headline 0.0385 via combiner_unavailable_range_only. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: bounded omitted-state residual 0.585: the interval widens to [0.0, 0.5853] — private-state omissions are BOUNDS, never unknown-state worlds
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the mass-based recovery blend (resolved mass keeps its simulated conditional; unresolved mass takes the grounded prior); the two inputs are reported separately and never blended by a fixed rule
