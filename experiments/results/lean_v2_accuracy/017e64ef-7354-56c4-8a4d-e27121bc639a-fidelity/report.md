# Lean V2 accuracy run — 017e64ef-7354-56c4-8a4d-e27121bc639a-fidelity

**Question:** Will daily oil tanker transits through the Strait of Hormuz reach 50 or more on any day between April 29 and June 1, 2026?

**Status:** completed | **probability:** 0.125 | **source:** mass_weighted:completed_rollouts+grounded_prior | **grounding:** exploratory | **confidence:** very_low


## Causal world

- thesis: The Strait remains effectively closed due to Iranian mining and US blockade since late February 2026. Resumption to 50 tankers/day requires coordinated de-escalation: Iran clearing mines and lifting closure, US lifting blockade, and tanker operators resuming transit. Without such actions, traffic st
- actors: 4 | action templates: 3

## Shared world conditions

- **hormuz_closure_regime**: The Strait of Hormuz is effectively closed to commercial oil tanker traffic due to military conflict and blockade. — counted rate 0.8333 (n=2)
- **pre_conflict_traffic_baseline**: Pre-conflict daily oil tanker traffic through the strait was approximately 100 tankers per day (half of 125-140 total vessels). — counted rate 0.75 (n=1)

## Forecast decomposition

- prior_forecast (grounded prior): 0.5 (n=2, source=counted_outcome_reference_class)
- simulation_forecast (conditional on resolved mass): 0.125 (resolved mass 1.0)
- simulation probability bounds (residual-widened): [0.0, 1.0] (residual bound 0.3141)
- headline_forecast: 0.125 via mass_weighted:completed_rollouts+grounded_prior
- prior/simulation disagreement: 0.375
  - no leakage-audited reliability combiner is fitted — prior and simulation are reported separately with the feasible combined range; no fixed blend is applied and Lean V2 must not become default on this basis

## Simulation completion audit

- resolved mass: 0.999999 of 0.999999 (target ≥0.8 met: True)
- unresolved by cause: {}
- unknown-state terminal mass: 0.0 (must be 0: True)
- missing-mechanism terminal mass: 0.0 (ok: True)
- readiness verdict: ready | round-trip ok: True

## Unresolved mass by cause


## Cost

- calls: 29 | wall: 118.53s | peak nodes: 72
- deliberations: 6 | challenger: False
- limitation: forecast decomposition — grounded prior 0.5 (counted n=2); simulation-conditional 0.125 (resolved mass 1.00); headline 0.125 via combiner_unavailable_range_only. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: weight_sensitive: within the counted reference-class intervals P(YES) spans [0.0, 1.0]
- limitation: bounded omitted-state residual 0.314: the interval widens to [0.0, 1.0] — private-state omissions are BOUNDS, never unknown-state worlds
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the mass-based recovery blend (resolved mass keeps its simulated conditional; unresolved mass takes the grounded prior); the two inputs are reported separately and never blended by a fixed rule
