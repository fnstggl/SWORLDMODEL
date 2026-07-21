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

## Full-fidelity baseline (EXP-107) — 3/5 completed in this environment

* **visionOS 27 (completed)**: 2,411 calls, 2.60M in / 1.12M out tokens (737k provider-cached),
  **212.0 min**, status unresolved, recovered p **0.834** (✓, Brier 0.028,
  `evidence_conditioned_prior`, exploratory).
* **Wale PM (completed)**: 5,897 calls, 6.18M in / 2.57M out tokens (2.15M provider-cached,
  35%), **692.4 min**, status under_modeled, native-contract p **0.158** (✗, Brier 0.708,
  `mixed:completed_rollouts+partial_rollouts`, partially_grounded, interval [0.00, 0.79],
  weight-sensitive).
* **BoJ June hike (completed)**: 2,687 calls, 3.31M in / 1.44M out tokens (1.28M
  provider-cached, 39%), **261.2 min**, status under_modeled, native-contract p **0.22** (✗,
  Brier 0.608, `mixed:grounded_reference_prior+partial_rollouts`, exploratory, interval
  [0.13, 0.37]).
* **Hormuz (one full attempt finished `execution_failed`)**: 3,050 calls / 251 min — both
  promoted models failed terminal projection on the pre-hardening image; the per-model error
  was unrecoverable from the artifact (that diagnosability gap + the recovery-can-never-kill-
  finalize hardening are now fixed and regression-pinned).
* **Banxico (+ Hormuz rerun): still running at report time.** The execution environment
  terminates long processes (observed lifetimes ≈2.5–8 h; three restarts during this work);
  these questions' serial full-fidelity runtimes repeatedly exceeded every window — cumulative
  burned attempts ≈45 worker-hours for the arm (BoJ ~11 h over 3 attempts before the 4.4 h
  fourth completed, Banxico ~17 h over 4, Wale ~14 h over 3 before the 11.5 h fourth
  completed, Hormuz ~12 h over 3). Ensemble rollouts pass a particle scope, which forces the
  serial path, so branch-threading cannot shorten them without touching canonical CRN
  semantics — declined by policy ("do not silently alter full fidelity").
* If the remaining checkpoints land, `exp109`/`exp110` regenerate in seconds and the numbers
  below update mechanically.

**The paired-question comparisons that did complete** (visionOS, Wale, BoJ; all outcomes 1):

* visionOS: full fidelity 2,411 calls / 3.72M tokens / $1.01 / 212 min vs lean 119 calls /
  252K tokens / $0.07 / 17.3 min — **20× fewer calls, ~15× fewer tokens, ~14× cheaper,
  12× faster**. FF 0.834 ✓ vs lean 0.417 ✗ — cause: FF's readout is the pure
  evidence-conditioned prior; lean's blends its small resolved no-leaning rollout mass with
  the same prior family.
* Wale: full fidelity 5,897 calls / 8.75M tokens / $2.27 / 692 min vs lean 418 calls /
  745K tokens / $0.22 / 43.9 min — **14× fewer calls, ~12× fewer tokens, ~10× cheaper,
  16× faster**. The accuracy sign reverses: FF 0.158 ✗ (Brier 0.708) vs lean 0.435 ✗
  (Brier 0.319) — FF's rollouts resolved substantial no-leaning mass and its readout followed
  them; lean's stayed unresolved and read the evidence-conditioned prior, which sat nearer
  the truth.
* BoJ: full fidelity 2,687 calls / 4.75M tokens / $1.21 / 261 min vs lean 115 calls /
  314K tokens / $0.10 / 20.3 min — **23× fewer calls, ~15× fewer tokens, ~12× cheaper,
  13× faster**. Again lean is closer AND on the correct side: FF 0.22 ✗ (Brier 0.608,
  `mixed:grounded_reference_prior+partial_rollouts`) vs lean 0.565 ✓ (Brier 0.190,
  `mixed:completed_rollouts+evidence_conditioned_prior`) — lean's rollouts resolved
  hike-leaning mass; FF's readout leaned on a no-leaning reference prior plus partial mass.

Across all three pairs the source of divergence is the same disclosed mechanism — which layer
of the forecast-recovery ladder each arm's resolved mass selects — not silent behavior drift;
each row carries `probability_source`, grade, and interval. Paired-3 Brier: FF 0.448 vs lean
0.283; correct side 1/3 each. One pair favors FF, two favor lean; three pairs still decide
nothing about accuracy at this sample size.

*(Scoring-integrity note: exp110 initially recomputed FF BoJ from artifacts as 1.0, which
would have flattered FF's Brier; the recomputation reads a strict SUBSET of the runtime's
recovery inputs. Fixed with native-precedence — a post-contract checkpoint's own recovery
fields are authoritative; artifact recomputation is attached as a diagnostic with any
discrepancy disclosed.)*

## The 30 answers (§27)

1. **FF five-question wall-clock**: not fully measurable in this environment — 3/5 completed
   (212.0 + 692.4 + 261.2 min = 19.4 h); the other two exceeded every process-lifetime window
   so far (≈45 worker-hours burned; relaunches still running).
2. **Lean wall-clock**: 134.0 min summed; 43.9 min worst question; every question fits a window.
3. **Total LLM calls removed**: paired questions — visionOS 2,411 → 119 (95%), Wale
   5,897 → 418 (93%), BoJ 2,687 → 115 (96%). Arm-vs-arm totals are not honestly comparable
   while 2 FF questions are incomplete; the lean arm total is 1,052 vs 10,995 for the three
   completed FF questions alone.
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
19. **Did predictions materially change?** On all three paired questions, yes — visionOS
    0.834 vs 0.417, Wale 0.158 vs 0.435, BoJ 0.22 vs 0.565 — with no consistent direction.
20. **Why**: the same disclosed mechanism every time — which forecast-recovery layer each
    arm's resolved mass selects. visionOS: FF read the evidence-conditioned prior (rollouts
    fully unresolved) while lean blended a small no-leaning resolved mass into it. Wale and
    BoJ: the inversion — FF's rollouts/reference prior leaned no while lean's resolved mass
    (BoJ) or prior readout (Wale) sat nearer the outcome. Disclosed per row via
    `probability_source`.
21. **Brier**: lean 0.337 (5 scored); FF 0.448 (3 scored; paired-3 lean is 0.283) — small
    samples, no accuracy claim is made from five questions; what the pairs do show is no
    evidence of systematic lean degradation.
22. **Wrong side of 0.5**: lean 3/5 wrong (visionOS, Wale, Hormuz); FF 2/3 wrong (Wale,
    BoJ); unknown for the 2 incomplete FF questions.
23. **ensure_outcome_pathway repairs in lean**: none needed (validated on every prepared
    model inside `prepare_persistence_run`).
24. **Did any lean optimization attempt to remove the terminal pathway?** No; no empty
    rollouts occurred.
25. **Token reduction**: paired question 3.72M → 252K (93%). Lean arm total: 2.15M.
26. **Cost reduction**: paired question $1.01 → $0.066 (93%); lean arm total $0.66 (recorded
    price assumptions).
27. **Wall-clock reduction**: paired questions 212 → 17.3 min (92%), 692 → 43.9 min (94%),
    261 → 20.3 min (92%); lean worst-case 44 min vs FF questions that repeatedly outlive the
    environment's process windows.
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
questions" requires five PAIRED completions, and this environment has completed three FF
questions. On those three the estimates differ materially with no consistent direction (FF
closer on visionOS; lean closer on Wale and BoJ — all via the same disclosed
probability-source mechanism; paired-3 Brier FF 0.448 vs lean 0.283). The paired evidence
shows no lean degradation so far but remains insufficient at 3/5. Safety invariants,
cache-parity, escalation and savings legs all PASS.

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
