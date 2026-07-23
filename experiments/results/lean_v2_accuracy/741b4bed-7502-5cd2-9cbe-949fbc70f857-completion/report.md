# Lean V2 accuracy run — 741b4bed-7502-5cd2-9cbe-949fbc70f857-completion

**Question:** Will Matthew Wale be elected as the next Prime Minister of Solomon Islands following the May 2026 no-confidence vote?

**Status:** completed | **probability:** 0.1302 | **source:** mass_weighted:completed_rollouts+grounded_prior | **grounding:** exploratory | **confidence:** low


## Causal world

- thesis: Matthew Wale, as opposition leader, seeks to convert his coalition's parliamentary majority into his own election as Prime Minister, but fluid coalition politics and rival candidates like John Agovaka may produce a compromise candidate, determining the outcome.
- actors: 6 | action templates: 1

## Shared world conditions

- **solomon_islands_coalition_instability**: Solomon Islands coalition politics are notoriously fluid; the leader of a no-confidence motion does not always become PM, as coalition negotiations can produce compromise candidates. — counted rate 0.8333 (n=2)
- **china_taiwan_diplomatic_rivalry**: Solomon Islands' shift to China in 2019 and subsequent tensions with Taiwan create external pressure on leadership choices, affecting coalition stability. — counted rate 0.75 (n=1)
- **economic_dependence_on_aid**: Solomon Islands relies heavily on foreign aid, particularly from China and Australia, which influences political alignments and coalition formation. — counted rate 0.75 (n=1)

## Forecast decomposition

- prior_forecast (grounded prior): 0.1667 (n=2, source=counted_outcome_reference_class)
- simulation_forecast (conditional on resolved mass): 0.1302 (resolved mass 1.0)
- simulation probability bounds (residual-widened): [0.0, 0.8607] (residual bound 0.737856)
- headline_forecast: 0.1302 via mass_weighted:completed_rollouts+grounded_prior
- prior/simulation disagreement: 0.0365
  - no leakage-audited reliability combiner is fitted — prior and simulation are reported separately with the feasible combined range; no fixed blend is applied and Lean V2 must not become default on this basis

## Simulation completion audit

- resolved mass: 0.999999 of 0.999999 (target ≥0.8 met: True)
- unresolved by cause: {}
- unknown-state terminal mass: 0.0 (must be 0: True)
- missing-mechanism terminal mass: 0.0 (ok: True)
- readiness verdict: ready | round-trip ok: True

## Unresolved mass by cause


## Cost

- calls: 30 | wall: 168.78s | peak nodes: 1458
- deliberations: 9 | challenger: False
- limitation: forecast decomposition — grounded prior 0.1667 (counted n=2); simulation-conditional 0.1302 (resolved mass 1.00); headline 0.1302 via combiner_unavailable_range_only. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: bounded omitted-state residual 0.738: the interval widens to [0.0, 0.8607] — private-state omissions are BOUNDS, never unknown-state worlds
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the mass-based recovery blend (resolved mass keeps its simulated conditional; unresolved mass takes the grounded prior); the two inputs are reported separately and never blended by a fixed rule
