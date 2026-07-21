# Lean V2 accuracy run — 017e64ef-7354-56c4-8a4d-e27121bc639a

**Question:** Will daily oil tanker transits through the Strait of Hormuz reach 50 or more on any day between April 29 and June 1, 2026?

**Status:** unresolved | **probability:** 0.5 | **source:** combined_calibrated | **grounding:** exploratory | **confidence:** very_low


## Causal world

- thesis: The Strait of Hormuz remains effectively closed due to Iranian mining and a US naval blockade, with only ~7 ships/day and no oil tankers as of April 27. The EIA assumes the conflict does not persist, but no diplomatic or military breakthrough is evident by April 30. Without a coordinated de-mining a
- actors: 4 | action templates: 3

## Shared world conditions

- **hormuz_closure_regime**: The Strait of Hormuz is effectively closed to commercial oil tanker traffic due to military conflict and blockade. — counted rate 0.8333 (n=2)
- **pre_conflict_traffic_baseline**: Pre-conflict daily oil tanker traffic through the Strait of Hormuz was approximately 125-140 ships per day, with oil tankers comprising a significant portion. — counted rate 0.75 (n=1)
- **us_blockade_policy**: The U.S. imposed a naval blockade of Iranian ports after the strait closure, preventing Iranian oil tankers from transiting. — counted rate 0.75 (n=1)

## Forecast decomposition

- grounded prior: 0.5 (n=2, source=counted_outcome_reference_class)
- simulation-conditional: None (resolved mass 0.0)
- combined: 0.5 via prior_only_simulation_unavailable
- prior/simulation disagreement: 0.0
  - simulation produced no resolved forecast; prior served, labeled

## Unresolved mass by cause

- unresolved_missing_mechanism: 0.999996 — one repair attempt else under-modeled disclosure

## Cost

- calls: 12 | wall: 76.66s | peak nodes: 54
- deliberations: 0 | challenger: False
- limitation: forecast decomposition — grounded prior 0.5 (counted n=2); simulation-conditional None (resolved mass 0.00); headline 0.5 via prior_only_simulation_unavailable. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: unresolved [unresolved_missing_mechanism]: 1.000 — one repair attempt else under-modeled disclosure
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the simulation-conditional forecast; the grounded prior is reported separately; this is a stated reason Lean V2 must not become the default yet
