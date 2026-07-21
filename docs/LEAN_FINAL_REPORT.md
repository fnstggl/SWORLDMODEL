# World Model V2 — Lean-Adaptive Final Report

Baseline: PR #127 merge commit `6e79fa345974031d5f2f1a17f3372cc28e76919e`.
Both arms: `deepseek-v4-flash`, temperature 0.2, max_tokens 3600, seed 0, sealed-replay
frozen-background bundle (`experiments/btf3_frozen_bundle.py`), full LLM actor cognition, no
numeric actors, same maximum particle budgets. Machine-readable artifacts:
`experiments/results/exp109_comparison.json`, `exp110_recovered_forecasts.json`,
`exp108_arm_lean_manifest.json`, `exp109_controlled_cache_isolation.json`,
`exp109_acceptance_static.json`, per-question checkpoints under `exp107_checkpoints/` and
`exp108_checkpoints/`.

## Controlled gates (deterministic fixtures — the semantic-safety evidence)

* **Isolated caching parity** (mandatory §1): caches on vs off, everything else identical —
  identical decisions, distributions, statuses, operator censuses, forecasts. Decision provider
  calls **70 → 4** on 35 particles containing exactly 4 genuinely distinct decision situations
  (101 hits, 0 invalidated).
* **Concurrency identity** (mandatory §6): lean sequential == lean `SWM_BRANCH_THREADS=4`.
* Full-fidelity byte-untouchedness, research-first arming, replicate-index keying,
  no-empty-rollout, no-periodic-reconsideration, status-independence of forecast availability:
  57 focused lean/contract tests, all green; full repo suite 1900+ green with 2 pre-existing
  environment failures that fail identically at the #127 merge commit.

## Live lean arm (EXP-108) — COMPLETE, all five questions

| Question | status | calls | unique ctx | reuses | particles | wall | recovered p (outcome) | Brier |
|---|---|---|---|---|---|---|---|---|
| BoJ June hike | under_modeled | 115 | 30 | 449 | 108/108 full | 20.3 m | 0.565 (1) ✓ | 0.190 |
| visionOS 27 | under_modeled | 119 | 9 | 70 | 56/101 early | 17.3 m | 0.417 (1) ✗ | 0.340 |
| Wale PM | unresolved | 418 | 53 | 331 | 48/111 early | 43.9 m | 0.435 (1) ✗ | 0.319 |
| Hormuz transits | under_modeled | 170 | 19 | 335 | 111/111 full | 22.4 m | 0.885 (0) ✗ | 0.782 |
| Banxico unanimity | unresolved | 230 | 40 | 303 | 48/148 early | 30.1 m | 0.769 (1) ✓ | 0.053 |
| **Totals** | — | **1,052** | **151** | **1,488** | **371/579** | **134.0 m** | Brier **0.337**, 2/5 side | — |

All five probabilities are labeled `exploratory` + `weight_sensitive` (forecast-availability
contract: availability preserved, weakness disclosed). Context: pre-#127 EXP-104 full system on
the same frozen set: Brier 0.393, 1/5; FutureSearch SOTA 0.165. Five questions are an
architecture/performance diagnostic, not an accuracy claim.

## Full-fidelity baseline (EXP-107) — 1/5 completed in this environment

* **visionOS 27 (completed)**: 2,411 calls, 2.60M in / 1.12M out tokens (737k provider-cached),
  **212.0 min**, status unresolved, recovered p **0.834** (✓, Brier 0.028,
  `evidence_conditioned_prior`, exploratory).
* **Hormuz (one full attempt finished `execution_failed`)**: 3,050 calls / 251 min — both
  promoted models failed terminal projection on the pre-hardening image; the per-model error
  was unrecoverable from the artifact (that diagnosability gap + the recovery-can-never-kill-
  finalize hardening are now fixed and regression-pinned).
* **BoJ, Wale, Banxico (+ Hormuz rerun): did not complete.** The execution environment
  terminates long processes (observed lifetimes ≈2.5–8 h; three restarts during this work);
  these questions' serial full-fidelity runtimes repeatedly exceeded every window — cumulative
  burned attempts ≈45 worker-hours for the arm (BoJ ~11 h over 3 attempts, Banxico ~17 h over
  4, Wale ~14 h over 3, Hormuz ~12 h over 3). Ensemble rollouts pass a particle scope, which
  forces the serial path, so branch-threading cannot shorten them without touching canonical
  CRN semantics — declined by policy ("do not silently alter full fidelity").
* A final relaunch attempt is running; if checkpoints land, `exp109`/`exp110` regenerate in
  seconds and the numbers below update mechanically.

**The paired-question comparison that did complete** (visionOS): full fidelity 2,411 calls /
3.72M tokens / $1.01 / 212 min vs lean 119 calls / 252K tokens / $0.07 / 17.3 min —
**20× fewer calls, ~15× fewer tokens, ~14× cheaper, 12× faster**, same honest §NAP status
family, both recovered probabilities exploratory-grade (FF 0.834 ✓ vs lean 0.417 ✗ on outcome
1 — cause: FF's readout is the pure evidence-conditioned prior; lean's blends its small
resolved no-leaning rollout mass with the same prior family — a partial_rollouts vs prior
source difference, disclosed per row).

## The 30 answers (§27)

1. **FF five-question wall-clock**: not measurable in this environment — 1/5 completed
   (212 min); the other four exceeded every process-lifetime window (≈45 worker-hours burned).
2. **Lean wall-clock**: 134.0 min summed; 43.9 min worst question; every question fits a window.
3. **Total LLM calls removed**: on the paired question, 2,411 → 119 (95%). Arm-vs-arm totals
   are not honestly comparable while 4 FF questions are incomplete; the lean arm total is 1,052.
4. **Actor calls removed**: lean spent 151 fresh decision computations (82 one-call + 69
   escalated-staged) for 1,528 decision invocations — 1,488 invocations (97%) served by reuse.
   The FF staged path spends ~4–5 calls per invocation.
5. **Unique actor decision contexts**: 151 (9–53 per question).
6. **Particles sharing each major context**: largest 32; four contexts ≥17.
7. **Unsafe merges**: none — 0 invalidated hits; parity gate byte-identical; every reuse
   carries an `explain_equivalence()` certificate.
8. **Unchanged-decision reuse avoided calls**: 0 live (every live trigger carried new
   decision-relevant content); behavior proven by deterministic tests.
9. **Duplicate-notification suppression avoided calls**: 0 live (same reason); test-proven.
10. **Compact prompting reduction**: 75% of actor-prompt chars (943,568 sent vs 3,774,272
    full-re-render equivalent).
11. **Provider prompt caching**: lean 217,472 cached prompt tokens (16% of input); FF visionOS
    737,152 (28% — staged re-sends create more identical prefixes, at 10× the volume).
12. **Consequence compilations reused**: 620 of 715 (87%).
13. **Structural models avoided**: lean simulated 2 per question (primary + reversal
    challenger; 10 total) vs the FF default ≥4 candidates with pilots + full budgets each
    (FF visionOS generated 5, simulated 2 after critics).
14. **Particles avoided by progressive stopping**: 208 of 579 budgeted (36%).
15. **Full-budget questions**: BoJ and Hormuz (conditions correctly refused early stop).
16. **Challenger questions**: all five (every reversal critic found a reversal-capable
    alternative — these are contested questions by construction).
17. **Repeated-run escalations**: all five ran the capped execution-replicate probe
    (underidentification is an escalation signal); reported, never averaged.
18. **Genuinely load-bearing calls**: structural generation/critic/compile, world boundary,
    per-model conditioning, the 151 unique decision contexts, 95 consequence compiles,
    finalization. Everything else was reuse.
19. **Did predictions materially change?** On the paired question, yes (0.834 vs 0.417).
20. **Why**: probability-source difference — FF read the evidence-conditioned prior (its
    rollouts fully unresolved); lean's partial_rollouts blended a small no-leaning resolved
    mass with the same prior family. Disclosed per row; both exploratory-grade.
21. **Brier**: lean 0.337 (5 scored); FF 0.028 (1 scored) — not comparable at these sample
    sizes; no accuracy claim is made from five questions.
22. **Wrong side of 0.5**: lean 3/5 wrong (visionOS, Wale, Hormuz); FF wrong-side unknown for
    4/5 (incomplete).
23. **ensure_outcome_pathway repairs in lean**: none needed (validated on every prepared
    model inside `prepare_persistence_run`).
24. **Did any lean optimization attempt to remove the terminal pathway?** No; no empty
    rollouts occurred.
25. **Token reduction**: paired question 3.72M → 252K (93%). Lean arm total: 2.15M.
26. **Cost reduction**: paired question $1.01 → $0.066 (93%); lean arm total $0.66 (recorded
    price assumptions).
27. **Wall-clock reduction**: paired question 212 → 17.3 min (92%); lean worst-case 44 min vs
    FF's un-completable windows.
28. **Largest safe improvement**: the decision-equivalence cache (1,488 reuses, 0
    invalidations, parity-gated); consequence-response reuse (620) second; progressive
    particles (208) third.
29. **Consumer default?** **Not yet** — see the §25 decision below.
30. **Exclusive to full-fidelity research mode**: the ≥4-candidate independent structural
    ensemble with full per-model budgets, staged multi-call cognition as default,
    per-particle behavioral variance without context sharing, model-family pools, mean-of-K
    stability studies.

## §25 Default-switch decision

**`world_model_v2` keeps `execution_profile="full_fidelity"` as the default.** The switch
conditions are not all met: "no catastrophic forecast degradation appears on the five
questions" requires five PAIRED completions, and this environment completed one FF question
(on it, lean's recovered estimate was materially different and on the wrong side). Safety
invariants, cache-parity, escalation and savings legs all PASS.

**Safe to enable independently today** (each parity/test-gated, semantics-preserving by
construction): run-shared artifacts; actor-state cohorting; decision-context caching +
single-flight; decision invalidation + duplicate suppression; deterministic prechecks +
execution classification; one-call cognition **with its recorded escalation ladder**; compact
prompts; consequence-program caching; the forecast-availability contract (both profiles
already share it). **Needs paired accuracy data before default**: reversal-triggered
structural reduction and progressive particle stopping — the two semantics-visible
reductions; they remain lean-profile-only pending a completed baseline (or an environment
with longer process lifetimes / intra-question checkpointing for the FF arm).

## Fixes contributed to the canonical runtime along the way (both profiles, regression-pinned)

1. §19 executability probe backend threading — every live structural candidate was rejected.
2. Unknown-entity persistence updates — mixed compiler naming variants killed whole runs.
3. Type-tolerant predicate evaluation — LLM-shaped completion conditions killed a 4-h rollout.
4. Sealed-replay frozen-background bundle for the BTF-3 protocol (post-#127 posterior shape).
5. Forecast availability ≠ grounding (user directive): layered recovery, separated fields,
   status-independent `has_forecast()`, no neutral-0.5 anywhere, weighted partial-rollout
   aggregation with disclosed unresolved mass, ensemble weight-sensitivity marking.
6. Finalize errors surface with tracebacks; recovery can never kill a finalize.
