# Lean V2 accuracy run — cfb43147-d9d2-5bd9-903f-f449e9a5aecf-fidelity

**Question:** Will the Banxico Governing Board's June 25, 2026, interest rate decision be unanimous (5-0 vote)?

**Status:** completed | **probability:** 0.4398 | **source:** deliberative_institution_vote | **grounding:** exploratory | **confidence:** low


## Causal world

- thesis: The May 7, 2026, 3-2 split revealed deep internal disagreement, with Heath and Espinosa preferring a hold. The board declared the easing cycle over, so the June meeting tests whether the majority can unify on a hold or if dissenters persist, potentially joined by others if growth concerns or inflati
- actors: 5 | action templates: 2

## Shared world conditions

- **easing_cycle_ended**: Banxico declared its two-year easing cycle over in May 2026, shifting to a data-dependent hold stance. — counted rate 0.75 (n=1)
- **internal_dissent_on_hold**: Two board members (Heath, Espinosa) dissented from the cut, preferring to hold, indicating a hawkish minority. — counted rate 0.75 (n=1)
- **economic_contraction_and_inflation_above_target**: Mexico's economy contracted and inflation forecast revised upward to 4.1% for Q2 2026, above 3% target. — counted rate 0.75 (n=1)

## Forecast decomposition

- prior_forecast (grounded prior): 0.8333 (n=2, source=counted_outcome_reference_class)
- simulation_forecast (conditional on resolved mass): 0.4398 (resolved mass 1.0)
- simulation probability bounds (residual-widened): [0.0, 0.4284] (residual bound 0.142625)
- headline_forecast: 0.4398 via deliberative_institution_vote
- prior/simulation disagreement: 0.3935
  - no leakage-audited reliability combiner is fitted — prior and simulation are reported separately with the feasible combined range; no fixed blend is applied and Lean V2 must not become default on this basis

## Simulation completion audit

- resolved mass: 1.0 of 1.0 (target ≥0.8 met: True)
- unresolved by cause: {}
- unknown-state terminal mass: 0.0 (must be 0: True)
- missing-mechanism terminal mass: 0.0 (ok: True)
- readiness verdict: ready | round-trip ok: True

## Unresolved mass by cause


## Cost

- calls: 37 | wall: 238.97s | peak nodes: 1944
- deliberations: 11 | challenger: False
- limitation: forecast decomposition — grounded prior 0.8333 (counted n=2); simulation-conditional 0.4398 (resolved mass 1.00); headline 0.4398 via combiner_unavailable_range_only. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: dependence_sensitive: the actor-state joint dependence is unidentified; the answer moves across (0.0833, 1.0) between the independent and comonotonic structures — no single correlation was assumed
- limitation: bounded omitted-state residual 0.143: the interval widens to [0.0, 0.4284] — private-state omissions are BOUNDS, never unknown-state worlds
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the mass-based recovery blend (resolved mass keeps its simulated conditional; unresolved mass takes the grounded prior); the two inputs are reported separately and never blended by a fixed rule
