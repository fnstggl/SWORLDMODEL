# World Model V2 — Integration Completion: Validation & Gate Scorecard

**Honest bottom line.** One execution-level break was found, fixed, and verified end-to-end
(Phase‑10 rule executability: **0 → 171** executable institution rules across the 46‑question
corpus, every institution-bearing plan now executable). The broader claim — that *every* causally
required phase activates (execute + StateDelta + matched causal effect) at near‑100% recall with
low false activation — is **NOT** met. Most activation gates are reported **FAILED** here with an
exact continuation path, per the mandate's rule that a gate must either pass or be honestly
reported as failed with a continuation path. No gate was lowered after seeing results; no
activation was manufactured by running irrelevant phases.

## Method

- **Corpus (Part B):** `experiments/integration_corpus.py` — 46 questions with **independent
  per-phase relevance labels** (`p4, p6, p7, p9pop, p9net, p10, p11`) + 10 irrelevant controls
  (`ctrl_*`). Labels are assigned from the question's structure, not from what the compiler emits.
- **Measurement (Parts L/M):** `experiments/integration_activation.py` compiles every question
  through the **real** compiler and scores, per phase, **emission** recall (emitted│required) and
  **false activation** (emitted│not-required) against the independent labels, plus the Phase‑10
  executable-rule count before/after normalization. Machine-readable:
  `experiments/results/integration/activation.json`.
- **Honest scope note (in the artifact itself):** this measures the **emission** stage +
  Phase‑10 executability. It does **not** measure full execute+StateDelta+matched-ablation, which
  the ≥95%/≥90% gates require. That is why those gates are marked FAILED, not PASSED.

## Measured results (n = 46)

| Phase | required | emission recall | false-activation | verdict |
|---|---|---|---|---|
| **p4** actor | 15 | **0.733** | 0.742 | fails recall **and** precision |
| **p6** mechanism | 3 | **0.000** | 0.093 | not emitted when required |
| **p7** nonlinear | 9 | **0.000** | 0.000 | operators registered, never named |
| **p9pop** populations | 12 | 1.000 | **1.000** | emitted everywhere → no discrimination |
| **p9net** networks | 9 | 1.000 | **0.946** | emitted almost everywhere → no discrimination |
| **p10** institutions | 13 | 1.000 | **1.000** | emitted everywhere; **but now executable** (fix below) |

**Phase‑10 executability fix (verified):** institution-bearing rows executable **before = 0 / 46**,
**after = 46 / 46**; total executable rules **0 → 171**. This is the one causal-execution-level
improvement this run lands and proves.

Two honest readings of the table:
1. **p6/p7** don't even *emit* when required (recall 0) — the compiler never names their operators.
2. **p9pop/p9net/p10** emit on nearly every question (false-activation ≈ 0.95–1.0) — high recall
   but no relevance discrimination, so they fail the *low-false-activation* half of the standard.
   High raw activation here is **not** success; it is exactly the "manufactured activation" failure
   mode the mandate warns against, and is reported as such rather than dressed up as a pass.

## Part P — 20 acceptance gates (pre-registered; not lowered after results)

| # | Gate | Status | Evidence / continuation |
|---|---|---|---|
| 1 | One canonical runtime, no new top-level pipeline | **PASS** | No new runtime; fix is in `integration_completion.py` wired into `unified_runtime.py`. |
| 2 | Part-A activation-chain audit exists, per phase | **PASS** | `activation_chain_audit.json`. |
| 3 | Independent relevance-labeled corpus ≥ real coverage | **PASS** | 46 Q + 10 controls, labels independent of emission. |
| 4 | No benchmark question IDs hardcoded in runtime | **PASS** | `test_no_benchmark_question_id_hardcoding`. |
| 5 | P10 rules become executable (root-cause fix) | **PASS** | 0 → 171 executable rules; diagnostics clear. |
| 6 | P10 completeness validator flags ornamental institutions | **PASS** | `completeness_diagnostics`; `test_completeness_flags_ornamental_institution`. |
| 7 | Normalization never drops a rule / idempotent | **PASS** | `test_normalization_never_drops_a_rule_and_is_idempotent`. |
| 8 | Normalization default-on in the runtime | **PASS** | wired after populations/networks; `test_unified_runtime_records_institution_normalization`. |
| 9 | Runtime fingerprint deterministic | **PASS** | `runtime_fingerprint.py`; `test_runtime_fingerprint_deterministic_and_corpus_status`. |
| 10 | Old Phase‑12 corpus demoted to diagnostic_only | **PASS** | `phase12_compatibility.json`; `corpus_status` demotes non-matching hashes. |
| 11 | Old Phase‑12 calibrator marked INCOMPATIBLE | **PASS** | `phase12_compatibility.json`. |
| 12 | Refit command requires the fingerprint | **PASS** | `phase12_refit.py` prints/requires the fingerprint. |
| 13 | P4 activation recall ≥ 0.95 on actor-required | **FAIL (0.733)** | Continuation: requirement-inference forces an actor operator + decision event on actor-required plans. |
| 14 | P4 false activation ≤ 0.10 on controls | **FAIL (0.742)** | Continuation: gate actor emission on a strategic-actor detector, not generic priors. |
| 15 | P6 recall ≥ 0.95 on mechanism-required | **FAIL (0.0)** | Continuation: map causal process → registry family + operator; add to `accepted_mechanisms`. |
| 16 | P7 recall ≥ 0.95 on nonlinear-required | **FAIL (0.0)** | Continuation: nonlinear-structure detector → nonlinear operator + matching scheduled event. |
| 17 | P9pop causal consumer moves terminal ≥ 0.90 | **FAIL (no consumer)** | Continuation: population-aggregation operator that affects the terminal; false-activation gate. |
| 18 | P9net multilayer + layer rewiring moves terminal ≥ 0.90 | **FAIL (no layers)** | Continuation: typed multilayer layers + layer-specific consumers. |
| 19 | P10 institution StateDelta ≥ 0.90 (execute, not just executable) | **FAIL (executable only)** | Continuation: `institution_action` operator + `institutional_action` event carrying an `InstitutionRuntime`. |
| 20 | P11 trigger recall ≥ 0.90 on injected shocks, migration integrity ≥ 0.98 | **FAIL (unvalidated)** | Continuation: adversarial shock/migration corpus verifying trigger→revision→migration→continuation. |

**12 / 20 PASS, 8 / 20 FAIL.** The 12 passes are the infrastructure + the one real
execution-level fix; the 8 fails are the per-phase execute+causal-effect gates, each with a
concrete continuation path (also enumerated in `activation_chain_audit.json`).

## Matched ablations (Part N) — status

Matched causal ablations (irrelevant controls + heterogeneity/layer/institution removal comparing
terminal shift) are **not run to conclusion** this run, because for p6/p7/p9 the causal *consumer*
that an ablation would toggle does not yet exist — ablating a component that has no terminal effect
would trivially show "no change" and prove nothing. The honest precondition for meaningful
ablations is landing the consumers (gates 15–19). Reported as a failed gate, not a passed one.

## Tests (Part Q)

`tests/test_wmv2_integration_completion.py` — 8 tests, all passing: rule normalization
(maps non-canonical kinds, never drops, idempotent, preserves executable), completeness flags,
`infer_required_phases`, fingerprint determinism + corpus status, no-question-ID-hardcoding, and
the runtime-records-normalization check.

## Constraints honored

Did **not**: merge the PR · modify agent-engine-on-main · begin Phase 13 · run/claim a definitive
historical accuracy benchmark · reuse the contaminated old corpus as product evidence · add
another top-level runtime or phase-specific production pipeline · lower any gate after seeing
results · claim predictive improvement · manufacture activation by executing irrelevant phases ·
add empty calls to light up manifests · hardcode benchmark questions.

## Continuation manifest (next run, in priority order)

1. **P10 execution** (nearest to done): add `institution_action` operator + `institutional_action`
   event carrying an `InstitutionRuntime`; target ≥0.90 institution StateDelta (gate 19).
2. **P9 consumers**: population-aggregation operator + typed multilayer network consumers so
   emission (already ~1.0 recall) gains a terminal effect and matched ablations become meaningful
   (gates 17–18); simultaneously add relevance gating to cut the ≈1.0 false-activation.
3. **P7/P6 emission**: nonlinear-structure + causal-process detectors that name the registered
   operators and emit matching events (gates 15–16).
4. **P4 precision+recall**: strategic-actor detector gating actor-operator emission (gates 13–14).
5. **P11 validation**: adversarial shock/migration corpus (gate 20).
6. **Then** regenerate the corpus through `simulate_world` under fingerprint `8cf389ba0ec96da8`
   and run `phase12_refit.py --regen` for product-eligible calibration.
