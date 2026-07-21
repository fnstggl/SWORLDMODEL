# Lean V2 accuracy run — 7279494c-a775-5a57-a5f2-ac22252fb286

**Question:** Will the Bank of Japan raise its short-term policy interest rate at the June 15–16, 2026 Monetary Policy Meeting?

**Status:** unresolved | **probability:** 0.875 | **source:** combined_calibrated | **grounding:** exploratory | **confidence:** very_low


## Causal world

- thesis: The BOJ Policy Board votes on a proposal to raise the short-term policy rate from 0.75% to 1.0%. The outcome hinges on whether the three dissenting members from April (Nakagawa, Takata, Tamura) gain one more vote, or whether a majority coalesces to keep rates unchanged. Market expectations and econo
- actors: 5 | action templates: 1

## Shared world conditions

- **japan_economic_regime**: Japan is in a moderate inflation and growth regime, with the BOJ gradually normalizing policy after years of ultra-loose stance. — counted rate 0.875 (n=3)
- **consensus_pressure_for_hike**: Market and internal pressure for a June hike is high, with dissenting members advocating for 1.0%. — counted rate 0.75 (n=1)
- **leadership_stance**: Governor Ueda has signaled gradual normalization but has not committed to a June hike, emphasizing data dependence. — counted rate 0.75 (n=1)

## Forecast decomposition

- grounded prior: 0.875 (n=3, source=counted_outcome_reference_class)
- simulation-conditional: None (resolved mass 0.0)
- combined: 0.875 via prior_only_simulation_unavailable
- prior/simulation disagreement: 0.0
  - simulation produced no resolved forecast; prior served, labeled

## Unresolved mass by cause

- unresolved_unknown_state: 0.936 — explicit other_unknown_state mass; feasible-action bounds computed, never assigned prior/0.5
- unresolved_future_decision: 0.064002 — advance simulated time to the deadline; the obligation reopens the decision

## Cost

- calls: 16 | wall: 112.22s | peak nodes: 96
- deliberations: 1 | challenger: False
- limitation: forecast decomposition — grounded prior 0.875 (counted n=3); simulation-conditional None (resolved mass 0.00); headline 0.875 via prior_only_simulation_unavailable. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: unresolved [unresolved_unknown_state]: 0.936 — explicit other_unknown_state mass; feasible-action bounds computed, never assigned prior/0.5
- limitation: unresolved [unresolved_future_decision]: 0.064 — advance simulated time to the deadline; the obligation reopens the decision
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the simulation-conditional forecast; the grounded prior is reported separately; this is a stated reason Lean V2 must not become the default yet
