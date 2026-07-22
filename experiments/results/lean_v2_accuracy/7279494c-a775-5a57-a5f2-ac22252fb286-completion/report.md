# Lean V2 accuracy run — 7279494c-a775-5a57-a5f2-ac22252fb286-completion

**Question:** Will the Bank of Japan raise its short-term policy interest rate at the June 15–16, 2026 Monetary Policy Meeting?

**Status:** partially_resolved | **probability:** 0.875 | **source:** mass_weighted:grounded_reference_prior+grounded_prior | **grounding:** exploratory | **confidence:** very_low


## Causal world

- thesis: The BOJ Policy Board votes on a proposal to raise the uncollateralized overnight call rate from 0.75% to 1.0%. A simple majority of the 9 members is required. The outcome hinges on whether the three April dissenters (Nakagawa, Takata, Tamura) are joined by at least two more members, or whether Gover
- actors: 5 | action templates: 1

## Shared world conditions

- **boj_gradual_hiking_cycle**: The Bank of Japan is in a gradual tightening cycle, raising rates in small increments after ending negative rates in March 2024. — counted rate 0.75 (n=1)
- **dissenting_pressure_for_hike**: Multiple board members have consistently dissented in favor of a rate hike, indicating internal pressure for tightening. — counted rate 0.75 (n=1)
- **market_expectations_elevated**: Market participants and economists widely expect a rate hike at the June meeting, with Polymarket probability at 63%. — counted rate 0.75 (n=1)

## Forecast decomposition

- prior_forecast (grounded prior): 0.875 (n=3, source=counted_outcome_reference_class)
- simulation_forecast (conditional on resolved mass): 0.0 (resolved mass 0.375)
- simulation probability bounds (residual-widened): [0.0, 0.6314] (residual bound 0.63136)
- headline_forecast: 0.875 via mass_weighted:grounded_reference_prior+grounded_prior
- prior/simulation disagreement: 0.875
  - no leakage-audited reliability combiner is fitted — prior and simulation are reported separately with the feasible combined range; no fixed blend is applied and Lean V2 must not become default on this basis

## Simulation completion audit

- resolved mass: 0.375 of 1.0 (target ≥0.8 met: False)
- unresolved by cause: {'unresolved_future_decision': 0.624902}
- unknown-state terminal mass: 0.0 (must be 0: True)
- missing-mechanism terminal mass: 0.0 (ok: True)
- readiness verdict: repairable | round-trip ok: True

## Unresolved mass by cause

- unresolved_future_decision: 0.624902 — advance simulated time to the deadline; the obligation reopens the decision

## Cost

- calls: 27 | wall: 120.98s | peak nodes: 1536
- deliberations: 6 | challenger: False
- limitation: forecast decomposition — grounded prior 0.875 (counted n=3); simulation-conditional 0.0 (resolved mass 0.37); headline 0.875 via combiner_unavailable_range_only. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: bounded omitted-state residual 0.631: the interval widens to [0.0, 0.6314] — private-state omissions are BOUNDS, never unknown-state worlds
- limitation: unresolved [unresolved_future_decision]: 0.625 — advance simulated time to the deadline; the obligation reopens the decision
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the mass-based recovery blend (resolved mass keeps its simulated conditional; unresolved mass takes the grounded prior); the two inputs are reported separately and never blended by a fixed rule
