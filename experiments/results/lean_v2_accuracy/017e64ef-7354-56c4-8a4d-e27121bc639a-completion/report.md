# Lean V2 accuracy run — 017e64ef-7354-56c4-8a4d-e27121bc639a-completion

**Question:** Will daily oil tanker transits through the Strait of Hormuz reach 50 or more on any day between April 29 and June 1, 2026?

**Status:** partially_resolved | **probability:** 0.8333 | **source:** mass_weighted:partial_rollouts+grounded_prior | **grounding:** exploratory | **confidence:** low


## Causal world

- thesis: The Strait remains effectively closed due to Iranian mining and US blockade since late February 2026. Resumption of normal traffic requires either a diplomatic agreement to lift the blockade and clear mines, or a unilateral US decision to allow tankers through. As of late April, only ~7 ships transi
- actors: 4 | action templates: 3

## Shared world conditions

- **hormuz_closure_regime**: The Strait of Hormuz is effectively closed to commercial oil tanker traffic due to military conflict and blockade. — counted rate 0.8333 (n=2)
- **global_oil_market_disruption**: Global oil markets are under severe supply disruption due to the Hormuz closure, with EIA estimating reduced Middle East oil production. — counted rate 0.75 (n=1)

## Forecast decomposition

- prior_forecast (grounded prior): 0.5 (n=2, source=counted_outcome_reference_class)
- simulation_forecast (conditional on resolved mass): 1.0 (resolved mass 0.6667)
- simulation probability bounds (residual-widened): [0.6859, 1.0] (residual bound 0.3141)
- headline_forecast: 0.8333 via mass_weighted:partial_rollouts+grounded_prior
- prior/simulation disagreement: 0.5
  - no leakage-audited reliability combiner is fitted — prior and simulation are reported separately with the feasible combined range; no fixed blend is applied and Lean V2 must not become default on this basis

## Simulation completion audit

- resolved mass: 0.666668 of 1.000002 (target ≥0.8 met: False)
- unresolved by cause: {'unresolved_missing_mechanism': 0.333336}
- unknown-state terminal mass: 0.0 (must be 0: True)
- missing-mechanism terminal mass: 0.333336 (ok: True)
- readiness verdict: repairable | round-trip ok: True

## Unresolved mass by cause

- unresolved_missing_mechanism: 0.333336 — one repair attempt else under-modeled disclosure

## Cost

- calls: 18 | wall: 101.14s | peak nodes: 72
- deliberations: 3 | challenger: False
- limitation: forecast decomposition — grounded prior 0.5 (counted n=2); simulation-conditional 1.0 (resolved mass 0.67); headline 0.8333 via combiner_unavailable_range_only. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: bounded omitted-state residual 0.314: the interval widens to [0.6859, 1.0] — private-state omissions are BOUNDS, never unknown-state worlds
- limitation: unresolved [unresolved_missing_mechanism]: 0.333 — one repair attempt else under-modeled disclosure
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the mass-based recovery blend (resolved mass keeps its simulated conditional; unresolved mass takes the grounded prior); the two inputs are reported separately and never blended by a fixed rule
