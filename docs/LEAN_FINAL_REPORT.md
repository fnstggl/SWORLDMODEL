# World Model V2 — Lean-Adaptive Final Report

> STATUS: lean arm (EXP-108) final; full-fidelity arm (EXP-107) cells marked **FF-PENDING**
> fill from `experiments/results/exp109_comparison.json` when the baseline completes.

Baseline: PR #127 merge commit `6e79fa345974031d5f2f1a17f3372cc28e76919e`.
Model/provider both arms: `deepseek-v4-flash`, temperature 0.2, max_tokens 3600, seed 0,
sealed-replay frozen-background bundle (`experiments/btf3_frozen_bundle.py`), full LLM actor
cognition, no numeric actors.

## Controlled gates (deterministic fixtures — the semantic-safety evidence)

* **Isolated caching parity** (caches on vs off, everything else identical): identical
  decisions, distributions, statuses, operator censuses and forecasts. Decision provider calls
  **70 → 4** on 35 particles containing exactly 4 genuinely distinct decision situations
  (101 cache hits). `experiments/results/exp109_controlled_cache_isolation.json`.
* **Concurrency identity**: lean sequential == lean `SWM_BRANCH_THREADS=4` on all semantics.
* Full-fidelity byte-untouchedness, research-first arming, replicate-index keying, no-empty-
  rollout: `tests/test_lean_integration.py`, `tests/test_lean_units.py` (46 tests).

## Live lean arm (EXP-108, five frozen BTF-3 questions) — FINAL

| Question | status | calls | unique ctx | ctx reuses | particles (exec/budget) | wall |
|---|---|---|---|---|---|---|
| BoJ June hike | under_modeled | 115 | 30 | 449 | 108/108 (full kept) | 20.3 m |
| visionOS 27 | under_modeled | 119 | 9 | 70 | 56/101 (early stop) | 17.3 m |
| Wale PM | unresolved | 418 | 53 | 331 | 48/111 (early stop) | 43.9 m |
| Hormuz transits | under_modeled | 170 | 19 | 335 | 111/111 (full kept) | 22.4 m |
| Banxico unanimity | unresolved | 230 | 40 | 303 | 48/148 (early stop) | 30.1 m |
| **Totals** | — | **1,052** | **151** | **1,488** | **371/579 (208 avoided)** | **134.0 m** |

Arm-level §23 manifest (`experiments/results/exp108_arm_lean_manifest.json`): consequence
compiles 95 with **620 reuses**; one-call successes 82; escalations 69 (67 blocked-on-
missing-fact, 2 provider — every one recorded and its staged result cached); invalidated
cache hits **0**; largest single decision context served **32 branches**; prompt chars sent
943,568 vs 3,774,272 full-re-render equivalent (**75% reduction**); provider prompt-cache
tokens 217,472; execution classifications: 1,528 human_discretion_required (the deterministic
layer never suppressed a genuine human choice in these worlds); challengers generated 5/5
questions (each critic found a reversal-capable alternative); structurally underidentified
4/5 (full-fidelity escalation offered on the result face); stability replicates ran 5/5.

Terminal statuses: all five are honest §NAP refusals (`under_modeled`/`unresolved`) under the
frozen-background protocol — the compiled worlds' required causal processes lacked validated
mechanism families and the runtime refuses to manufacture probability mass. Status-gated
scoring (one rule, both arms): refusals have no scoreable forecast.


## Forecast-availability contract (user directive, applied after the first runs)

Grounding quality and forecast availability are now SEPARATE (`forecast_recovery.py`): every
coherent binary question returns its best defensible probability with
`probability_source` / `grounding_grade` / `confidence` / `unresolved_mass` /
`probability_conditional_on_resolved` / `uncertainty_interval` / `weight_sensitive` as
separate fields; execution status describes the run and never erases the probability
(`has_forecast()` is status-independent — regression-pinned). No neutral 0.5 exists anywhere
(AST-pinned); with no defensible source the probability honestly stays None.

**EXP-110** recovered all five lean forecasts from the EXISTING checkpoints (no reruns, no new
calls, original as_of/evidence untouched — pure readout re-derivation from stored weighted
distributions + evidence-updated posterior means):

| Question | recovered p | outcome | Brier | side | source | grade |
|---|---|---|---|---|---|---|
| BoJ June hike | 0.565 | 1 | 0.190 | ✓ | completed_rollouts+evidence_prior | exploratory |
| visionOS 27 | 0.417 | 1 | 0.340 | ✗ | evidence_prior+partial_rollouts | exploratory |
| Wale PM | 0.435 | 1 | 0.319 | ✗ | evidence_conditioned_prior | exploratory |
| Hormuz transits | 0.885 | 0 | 0.782 | ✗ | evidence_prior+partial_rollouts | exploratory |
| Banxico unanimity | 0.769 | 1 | 0.053 | ✓ | evidence_conditioned_prior | exploratory |

Lean arm: Brier 0.337, 2/5 correct side — every row weight-sensitive and exploratory-grade
(the labels say exactly how weak these are; for context the pre-#127 EXP-104 full system
scored 0.393 / 1/5 on the same frozen set; FutureSearch SOTA 0.165). FF visionOS recovered
0.834 → Brier 0.028 (its arm completes separately). The five-question set remains an
architecture/performance diagnostic, not an accuracy claim.

## Full-fidelity baseline (EXP-107) — FF-PENDING

Per-question and total: prediction/status/Brier/side, calls, calls by stage, tokens,
provider-cache tokens, wall-clock, cost, structural models generated/simulated, particles,
censuses → fill from `exp109_comparison.json`.

## The 30 questions (§27) — fill FF-PENDING cells on baseline completion

1. **PR #127 full-fidelity five-question wall-clock**: FF-PENDING.
2. **Lean adaptive wall-clock**: 134.0 minutes summed; 43.9 minutes maximum single question
   (the five ran in parallel).
3. **Total LLM calls removed**: FF-PENDING (lean total is 1,052).
4. **Actor calls removed**: FF-PENDING (lean spent 82 one-call + 69 escalation-staged
   decision rounds over 1,528 decision invocations; 1,488 invocations were served by reuse).
5. **Unique actor decision contexts**: 151 across the arm (9–53 per question).
6. **Particles sharing each major actor context**: largest 32; four contexts ≥ 17.
7. **Unsafe merges**: none observed — invalidated cache hits 0; the isolated parity gate
   showed byte-identical semantics; every reuse carries a certificate
   (`explain_equivalence()`).
8. **Calls avoided by unchanged-decision reuse**: 0 — in these worlds every trigger carried
   new decision-relevant content, so the standing-decision layer never fired (its tests prove
   it fires on true duplicates).
9. **Calls avoided by duplicate-notification suppression**: 0 live (same reason); proven by
   deterministic tests.
10. **Compact prompting token reduction**: 75% of actor-prompt chars vs full re-render
    (943,568 sent vs 3,774,272 equivalent).
11. **Provider prompt caching achieved**: 217,472 cached prompt tokens (of 1,333,710 input).
12. **Consequence compilations reused**: 620 of 715 (87%).
13. **Structural models lean avoided**: FF-PENDING (lean generated 1 primary + 1 challenger
    per question = 10 simulated; FF default generates ≥4 candidates + pilots each).
14. **Particles avoided by progressive stopping**: 208 of 579 budgeted (36%).
15. **Questions requiring the full particle budget**: BoJ and Hormuz (stability conditions
    correctly refused early stop); visionOS/Wale/Banxico stopped early with records.
16. **Questions requiring a challenger**: all five (each reversal critic found a plausible
    reversal-capable alternative; each challenger compiled, deduped and simulated).
17. **Questions requiring repeated-run escalation**: all five ran the capped
    execution-replicate probe (underidentification is an escalation signal); results reported,
    never averaged.
18. **Genuinely load-bearing calls**: structural generation/critic/compile, world boundary,
    per-model conditioning, the 151 unique decision contexts, 95 consequence compiles, and
    per-model finalization — everything else was reuse.
19. **Did predictions materially change?** FF-PENDING; both arms are status-gated — lean
    produced five honest refusals.
20. **Why did each changed prediction change?** FF-PENDING (per-question §23 cause rows in
    exp109).
21. **Brier**: FF-PENDING (lean: no scoreable forecasts under §NAP honesty).
22. **Wrong side of 0.5**: FF-PENDING.
23. **Did ensure_outcome_pathway repair any lean plan?** No repairs were required at prepare
    time (`outcome_pathway.repaired` false across checkpoints); the invariant ran on every
    prepared model.
24. **Did any lean optimization attempt to remove the terminal pathway?** No — pathway
    validation runs inside `prepare_persistence_run` on every lean model; no
    empty rollouts occurred.
25. **Total token reduction**: FF-PENDING (lean: 1,333,710 in / 814,374 out).
26. **Total cost reduction**: FF-PENDING (recorded price assumptions in exp109).
27. **Total wall-clock reduction**: FF-PENDING.
28. **Largest safe improvement**: the decision-equivalence cache (1,488 reuses at zero
    invalidations, parity-gated), with consequence-response reuse (620) second and
    progressive particles (208) third.
29. **Should lean adaptive become the consumer default?** Decision recorded in
    `§ Default-switch` below — FF-PENDING for the accuracy leg.
30. **What stays exclusive to full-fidelity research mode?** The independent ≥4-candidate
    structural ensemble with per-model full budgets, multi-call staged cognition as the
    default, per-particle behavioral variance without context sharing, model-family pools,
    and mean-of-K stability studies.

## Default-switch (§25) — decision pending FF baseline

Static acceptance (`experiments/results/exp109_acceptance_static.json`): full_fidelity
available and default; PR-#127 protection tests green; lean gates green; no numeric-actor
paths in lean modules; controlled isolation parity recorded. Remaining legs: no catastrophic
forecast degradation vs FF; explainable prediction changes; material call/token/cost/time
reduction (lean side measured; FF side pending).
