# WMV2 Phase 6 — Final Report (brutally honest)

Phase 6 builds a **production, evidence-backed, executable mechanism registry** on top of the Phase-1
universal execution path. This report grades it without inflation. All numbers below are regenerated from
committed artifacts by `experiments/wmv2_phase6_report.py`; the registry is `swm/world_model_v2/registry/`.

## 1. Before / after (regenerated, not asserted)

| Quantity | Before (base `claude/world-model-v2`) | After Phase 6 |
|---|---|---|
| Named families | 47 | **61** |
| Software-implemented families (executable + tests) | ~47 | **57** |
| Empirically-validated families (passed held-out/PPC/transfer) | 3 | **5** |
| Production-eligible families | 2 | **3** |
| Parameter packs | 6 | **16** |
| — with local held-out validation | 3 | **4** |
| — with a transfer record (passed) | 2 | **3** |
| — published-estimate (verified) packs | 0 | **6** |
| Statuses | impl 43 / local 1 / prod 2 / quar 1 | impl 45 / **research_encoded 4** / local 2 / **domain_restricted 6** / prod 3 / quar 1 |
| Preserved negative results | (Hawkes, BehaviorBench PG, ...) | **7** (all prior + telco negative-transfer + StackExchange/CMV nulls) |

Selection quality (mechanism routing, NOT outcome accuracy) over a 12-scenario / 22-process stratified bank
(`experiments/results/wmv2_phase6_selection_eval.json`):

| Config | Tier 1–4 | Tier 6–7 | **Valid selection** (family actually answers the process) |
|---|---|---|---|
| names-only registry | 0% | 100% | 0% |
| **before** (one scenario winner reused for every process) | 100% | 0% | **22.7%** |
| **after** (Phase-6 per-process selection) | 86.4% | 13.6% | **100%** |

The headline: the pre-Phase-6 compiler stamped **one scenario winner onto every causal process** — high tier,
but only **22.7%** of those selections were for a family that actually answers the process (false coverage).
Phase-6 per-process selection makes **100%** of selections valid; the honest cost is that **13.6%** of
processes correctly fall to Tier 6–7 because no evidence-backed family answers them yet.

## 2. The 20 anti-scaffolding answers

1. **How many genuinely implemented families now exist?** 57 software-implemented (executable transition +
   tests). 61 named (4 are `research_encoded` structural records with no executable numeric transition yet).
2. **How many are empirically validated?** 5 (passed a real held-out / posterior-predictive / transfer check):
   `content_response_click`, `engagement_momentum_persistence`, `social_preference_population` (transfer),
   `attrition_dropout_hazard` (held-out), `exposure_response_hazard` (held-out).
3. **How many are production eligible?** 3: `content_response_click` (new), `engagement_momentum_persistence`,
   `social_preference_population`.
4. **How many genuine parameter packs exist?** 16 (4 with local held-out, 6 verified published estimates, 6
   with a preserved failed/null record). No duplicated packs.
5. **Which families are strongest?** `content_response_click` (randomized→causal, in-distribution held-out
   pairwise 0.738 + out-of-time transfer 0.719, population ablation, LLM-free); `exposure_response_hazard` &
   `engagement_momentum_persistence` (Higgs / OmniBehavior held-out).
6. **Which remain structural candidates only?** `weak_tie_transmission`, `network_targeting_seeding`,
   `altruistic_punishment`, `persuasion_minimal_effects` (`research_encoded`: verified research + formal
   model, no executable numeric transition this run) plus ~19 registry families the priority matrix marks
   `structural_candidate_only`.
7. **Which are quarantined?** `hawkes_self_excitation` (held-out count forecast FAILED vs Poisson on Higgs —
   preserved, never overwritten).
8. **Which were rejected?** None outright rejected; the honest NULLs (`response_occurrence_hazard`,
   `argument_persuasion_success`) are kept `implemented` with their failed held-out preserved — a null is not
   a rejection.
9. **Which prior negative results remain unresolved?** Hawkes (aggregate-count forecast); BehaviorBench
   public-goods misfit; StackExchange + CMV surface-feature nulls; telco cross-subpopulation negative
   transfer. All preserved in `wmv2_phase6_failures.json`.
10. **Does the compiler select by applicability, not name similarity?** Yes — `select_for_process` matches a
    required causal process to families that DECLARE they answer it (`answers_processes`), then scores
    applicability. Adversarial test: diffusion families score 0 for `offer_response` and are never selected.
11. **Do selected mechanisms instantiate scenario-specific parameter posteriors?** Yes — the pack's values
    (with source labels + uncertainty) are bound into a scenario instance (a `hazard_spec`) with a transport
    widening from `assess_transport`.
12. **Do they execute through WorldState and StateDelta?** Yes — `BehavioralMechanismOperator` and
    `FeatureHazardOperator` read typed state, sample, write a typed quantity, and emit a `StateDelta` (5
    traces executed end-to-end in `wmv2_phase6_forensic_traces.json`; pinned by `tests/test_wmv2_phase6.py`).
13. **Do they materially affect terminal outcomes?** Yes — terminal-sensitivity sweeps move the readout (e.g.
    social-pressure control 0.287 → neighbors 0.387; ultimatum accept 0.05→0.40 offer moves acceptance >0.5).
14. **Has reliance on Tier 6–7 fallback decreased?** For processes an evidence-backed family answers, yes
    (now Tier 2–4 with valid selection). Overall the honest figure is 86% Tier 1–4 / 14% Tier 6–7, with the
    critical fix being valid-selection 22.7%→100%.
15. **Are transport limits and uncertainty honest?** Yes — every published pack carries a broad transport
    prior (never the tiny within-study SE), an explicit `transport_note`, and forbidden interpretations;
    `assess_transport` can reject a decisive-axis mismatch.
16. **Is Phase 6 software implemented?** Yes.
17. **Does Phase 6 execute end to end?** Yes, for the executable families (behavioral + feature-hazard).
18. **Is Phase 6 empirically validated?** Partially — 5 families / 4 packs on real held-out data; the bulk of
    new coverage is verified-published (Tier-4 `domain_restricted`), which is NOT the same as locally
    validated and is labeled as such.
19. **Is the registry production eligible?** As infrastructure, yes. As coverage, 3 families are
    production-eligible; the rest are honestly graded below that.
20. **Why is this more than a paper citation / registry name / synthetic test?** Every counted family has an
    executable transition + tests; every validated pack has a real held-out artifact; every published pack
    has a core-verified primary source + broad transport uncertainty; failures are preserved; and the whole
    chain (select → instantiate → execute → StateDelta → terminal) runs and is pinned by tests + traces.

## 3. Honest family table (the strongest + the caveats)

| Family | Software | Executes | Real packs | Local validation | Transfer | Production | Main limitation |
|---|---|---|---|---|---|---|---|
| content_response_click | ✓ | ✓ | upworthy_archive_2021 | pairwise 0.738 (held-out) | 0.719 (time-fwd) | **✓** | Upworthy clickbait era; ~22% caching randomization issue (2024 correction) |
| engagement_momentum_persistence | ✓ | ✓ | omnibehavior_kuaishou | Brier −0.0065 held-out | person-disjoint | ✓ | short-video engagement only |
| social_preference_population | ✓ | ✓ | behaviorbench_moblab | loses to per-game hist in-dist | LOGO passes (PG fails) | ✓ | lab economic games; PG misfit preserved |
| attrition_dropout_hazard | ✓ | ✓ | telco_ibm_churn | Brier .141 vs .198 | **FAILS** (preserved) | — (blocked) | observational; negative cross-subpop transfer |
| exposure_response_hazard | ✓ | ✓ | higgs_2012_rumor (real θ) | held-out tied fitted logistic | no transfer record | — (blocked) | one cascade; no transfer test |
| bass_diffusion | ✓ | ✓ | sultan_farley_lehmann_1990 | published estimate only | — | — (domain_restricted) | wide prior (SD≈mean); durable goods |
| ultimatum_offer_response | ✓ | ✓ | oosterbeek_2004 | published + MobLab cross-check | — | — (domain_restricted) | lab one-shot; not repeated |
| trust_game_transfer | ✓ | ✓ | johnson_mislin_2011 | published estimate only | — | — (domain_restricted) | lab trust game ≠ real trust |
| social_pressure_turnout | ✓ | ✓ | ggl_2008_michigan | published estimate only | — | — (domain_restricted) | low-salience primary; ceiling on transport |
| matching_donation_response | ✓ | ✓ | karlan_list_2007 | published estimate only | — | — (domain_restricted) | prior donors; ratio-flat null |
| reputation_updating | ✓ | ✓ | ebay_resnick_2006 | published estimate only | — | — (domain_restricted) | eBay; gameable ratings |
| response_occurrence_hazard | ✓ | ✓ | stackexchange_answered | **NULL** (preserved) | — | — | surface features ≈ base rate |
| argument_persuasion_success | ✓ | ✓ | cmv_delta | **NULL** (preserved) | — | — | surface features ≈ base rate |
| hawkes_self_excitation | ✓ | ✓ | higgs_2012_stream | **FAILED** (preserved) | — | **quarantined** | underfits burst vs Poisson |
| weak_tie_transmission / network_targeting_seeding / altruistic_punishment / persuasion_minimal_effects | — | — | 0 (research record) | — | — | research_encoded | verified research; no numeric transition yet |

## 4. Exactly what remains (not relabeled as done)

- **Executable numeric transitions** for the 4 `research_encoded` families and ~19 `structural_candidate_only`
  matrix families (weak ties, targeting, punishment, coalition defection/Gamson, DeGroot vs Bayesian,
  position-bias propensity form, enforcement, queueing).
- **Local validation** of the 6 `domain_restricted` published packs against real local data (the blocker to
  promotion is stated per-family in `wmv2_phase6_validation_summary.json`): mobilization/turnout on a real
  GOTV dataset, Bass on real product-sales curves, trust/ultimatum against per-subject MobLab distributions.
- **Downstream dependency (documented, not faked):** the compiler binds packs to scenario instances using
  the evidence/posterior interfaces available through Phase 1–2; a full Phase-2/3 posterior service and a
  richer evidence-to-parameter estimator are out of scope and recorded in `WMV2_PHASE6_AUDIT_AND_ARCHITECTURE.md`.
- **Congress/VoteView refit** for coalition defection (builder present; raw uncommitted).

## 5. Verdict

- **Software implemented:** YES.
- **Executes end to end:** YES (behavioral + feature-hazard families; 5 forensic traces + tests).
- **Empirically validated:** PARTIAL — 5 families / 4 packs on real held-out; the +6 published packs are
  verified-but-transported, honestly graded `domain_restricted`, not locally validated.
- **Registry production eligible:** the infrastructure is; the *coverage* is 3 production families. This run
  materially increased validated + parameterized coverage across 9 categories and fixed the selection flaw,
  but the majority of the ~40-family / 100–200-pack target remains — **Phase 6 is substantially advanced,
  not complete.** Graded honestly: **incomplete on the numeric target, complete on architecture + a real
  first tranche of evidence-backed mechanisms.**

## 6. Reproducibility

```
PYTHONPATH=. python -m experiments.wmv2_phase6_fits            # real-data fits + held-out (telco/SE/CMV/Upworthy)
PYTHONPATH=. python -m experiments.wmv2_phase6_matrix          # priority matrix + study registry
PYTHONPATH=. python -m swm.world_model_v2.registry.build_registry  # build committed registry.json/packs.json
PYTHONPATH=. python -m experiments.wmv2_phase6_selection_eval  # before/after fallback-tier + validity
PYTHONPATH=. python -m experiments.wmv2_phase6_traces          # forensic traces (end-to-end executions)
PYTHONPATH=. python -m experiments.wmv2_phase6_report          # the four/six counts + failures index
PYTHONPATH=. python -m pytest tests/test_wmv2_phase6.py        # 15 tests (execution, selection, gates)
```

Artifact index: see `WMV2_PHASE6_AUDIT_AND_ARCHITECTURE.md` §Artifacts.
