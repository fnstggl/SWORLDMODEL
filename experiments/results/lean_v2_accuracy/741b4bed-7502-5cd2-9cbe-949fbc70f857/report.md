# Lean V2 accuracy run — 741b4bed-7502-5cd2-9cbe-949fbc70f857

**Question:** Will Matthew Wale be elected as the next Prime Minister of Solomon Islands following the May 2026 no-confidence vote?

**Status:** partially_resolved | **probability:** 0.8333 | **source:** mass_weighted:grounded_reference_prior+grounded_prior | **grounding:** exploratory | **confidence:** very_low


## Causal world

- thesis: Matthew Wale, as opposition leader, seeks to convert the no-confidence victory into his own premiership, but coalition fluidity and rival candidate John Agovaka create uncertainty. The 26-member coalition must vote; Wale's success depends on securing a majority among them and potentially attracting 
- actors: 5 | action templates: 1

## Shared world conditions

- **solomon_islands_coalition_instability**: Solomon Islands coalition governments are highly unstable, with frequent no-confidence motions and party switching. — counted rate 0.5 (n=2)
- **china_influence_pressure**: China exerts significant economic and diplomatic influence on Solomon Islands politics, often backing incumbents or preferred candidates. — counted rate 0.75 (n=1)

## Forecast decomposition

- grounded prior: 0.8333 (n=2, source=counted_outcome_reference_class)
- simulation-conditional: 0.0 (resolved mass 0.0088)
- combined: None via combiner_unavailable_range_only
- prior/simulation disagreement: 0.8333
  - no leakage-audited reliability combiner is fitted — prior and simulation are reported separately with the feasible combined range; no fixed blend is applied and Lean V2 must not become default on this basis

## Unresolved mass by cause

- unresolved_unknown_state: 0.2775 — explicit other_unknown_state mass; feasible-action bounds computed, never assigned prior/0.5
- unresolved_future_decision: 0.71373 — advance simulated time to the deadline; the obligation reopens the decision

## Cost

- calls: 50 | wall: 214.68s | peak nodes: 324
- deliberations: 9 | challenger: True
- limitation: forecast decomposition — grounded prior 0.8333 (counted n=2); simulation-conditional 0.0 (resolved mass 0.01); headline 0.8333 via combiner_unavailable_range_only. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: unresolved [unresolved_unknown_state]: 0.278 — explicit other_unknown_state mass; feasible-action bounds computed, never assigned prior/0.5
- limitation: unresolved [unresolved_future_decision]: 0.714 — advance simulated time to the deadline; the obligation reopens the decision
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the simulation-conditional forecast; the grounded prior is reported separately; this is a stated reason Lean V2 must not become the default yet
