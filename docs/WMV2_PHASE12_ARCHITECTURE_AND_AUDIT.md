# WMv2 Phase 12 — Architecture and Audit

*Production calibration, support grading, uncertainty diagnostics and model criticism for World Model V2.
This document records the ruthless Part-A audit of the real system, the Phase-12 architecture built on top of
it, and the exact unresolved dependencies. Every empirical number lives in the machine-readable artifacts
under `experiments/results/phase12/` and in `WMV2_PHASE12_VALIDATION.md`; nothing here is claimed complete
that the artifacts do not support.*

## Part A — ruthless audit of the current system (verified in code, not from docs)

**Two entry points, and the shipped one is the weaker.**
- Public facade `swm/facade.py:75` `forecast(question, *, architecture, ...)`. For `architecture="world_model_v2"`
  it calls `swm/world_model_v2/pipeline.py:103` `simulate(...)`, which runs `compile_world → run_from_plan →
  result_from_run`. **This shipped path does NOT compute a numeric posterior.**
- The richer maximum-capacity path is `phase3_pipeline.py:55` `simulate_with_posterior(...)` (evidence →
  posterior hidden-state → structural hypotheses → rollout). It is invoked **only from experiments/tests**, not
  from the facade. So "max capacity" is a research entry, not the default product path. **This is recorded, not
  hidden.**

**Phase wiring (what is actually on the question→forecast path):**
- On the main compile/execute path (conditionally): Phase 4 actor policy (`transitions.py:510` →
  `phase4_execution`), the Phase 6/7 mechanism **registry** (`compiler.py:561-597`), and Phase 10
  `institutions_v2` (`compiler.py:599`).
- **NOT on the main path:** Phase 8 persistence (`phase8_pipeline.simulate_with_persistence` — separate entry),
  Phase 9 populations/networks (`phase9_pipeline.simulate_with_populations_networks` — separate entry), and the
  nonlinear package (`world_model_v2/nonlinear/` is CLI-only, no runtime importer).
- **Phase 11 (dynamic recompilation) is ABSENT from the base branch** (being developed in parallel). Only the
  Phase-2 `evidence_recompile.py` exists.

**Existing Phase-12 skeleton was mostly ornamental.** `calibration.py` already defined `PlattCalibrator`,
`IsotonicCalibrator`, `ConditionedCalibrator`, `grade_support`, `run_critic`, `build_result`, and
`decompose_uncertainty` — **but only `decompose_uncertainty` was imported by any pipeline**. `calibrated_probability`
was `None` by default; `grade_support`/`run_critic`/`build_result` were never called; the calibrators operated
on scalar `(p, y)` pairs, never fitted on real held-out full-system outputs. `support_grade` on the main path
was purely a function of mechanism-tier selection (`fallback.overall_support_grade`), not evidence quality,
calibration, or critic disagreement. `uncertainty.py` is actually a Phase-2B mechanism-compiler, unwired.

**Consequence for Phase 12:** the honest task is to (a) fit + select + validate calibration on **real**
max-capacity forecasts with data governance, (b) **wire** the calibrated result so it is no longer ornamental,
(c) build the missing governance / calibration-uncertainty / empirical support-grade validation / real
uncertainty decomposition / sensitivity / baseline-and-critic evaluation / monitoring, and (d) mark everything
**provisional** because the corpus is a pre-Phase-11, not-fully-integrated distribution.

## Real full-force call path (as it exists today)

```
question + as_of  →  compile_world (Phase-1/2, registry 6/7, actor-policy 4, institutions_v2 10 conditionally)
  →  gather_evidence (strict as-of)  →  tag_claims  →  infer_posterior (hidden-state + structural)  [Phase 3]
  →  materialize posterior onto plan  →  run_from_plan rollout  →  terminal WorldState distribution
  →  raw forecast (raw_probability)
  →  [Phase 12] load_phase12_bundle → calibrated_result:
        selected calibrator (identity) → calibrated_probability
        fitted support-grade model → support_grade + reasons
        uncertainty decomposition (posterior LTV + LOO-group)
        dominant sensitivity contributors
        critic (vs direct/ensemble) → disagreement flags (never overwrites)
        calibration provenance + effective_calibration_n
  →  user-facing SimulationResult
NOT on this path: Phase 8 persistence, Phase 9 populations/networks, nonlinear, Phase 11 recompilation.
```

## Architecture built in Phase 12

- **Data governance (Parts B/C/D)** — `experiments/phase12_corpus.py` pools **148 REAL max-capacity posterior
  forecasts** (93 phase3acc + 34 phase3b + 23 diagnostic) with per-row provenance + an **active-component
  manifest**, and assigns each **event family** to exactly one of {calibration 75, validation 32, test 41}
  with a seeded hash → immutable `split_manifest.json` + content hash. Fresh Phase-12 splits, distinct from the
  Phase-3 splits and from any Phase-15 benchmark. `maximum_capacity_available=False` (Phase 11 absent).
- **Calibrator registry + selection (Parts E/F/G/H)** — `calibration.py` gains `IdentityCalibrator`,
  `BetaCalibrator`, `CALIBRATOR_REGISTRY`, `select_calibrator` (promotes a non-identity calibrator ONLY if it
  beats identity on BOTH validation Brier and log-loss), conditioned/hierarchical partial pooling
  (`fit_conditioned`), and `bootstrap_calibration_uncertainty`.
- **Empirical support grading (Part I)** — `phase12_support.py` fits a transparent monotonic reliability model
  (expected squared error from pre-outcome features) on calibration+validation; frozen thresholds; grade never
  sees the outcome.
- **Uncertainty decomposition + sensitivity (Parts J/K)** — `phase12_uncertainty.py` recomputes the posterior
  and applies the law of total variance + leave-one-evidence-group-out; a synthetic-recovery check validates
  attribution.
- **Baselines + critic (Part L)** — `phase12_baselines.py` (grounded direct-LLM + ensemble) and `run_critic`
  (disagreement flags; cannot overwrite the simulation).
- **Serving (Part M)** — `phase12_serve.py` loads the frozen bundle and **populates** the calibrated result
  contract (fixing the ornamental finding); the raw number is never changed.
- **Monitoring (Part S)** — `compatible_with()` refuses a provisional pre-Phase-11 calibrator once Phase 11
  lands; `phase12_refit.py` is the resumable final-integration command.

## Data governance summary

fit on `calibration` only; select method on `validation` only; evaluate once on `test`. No test outcome enters
any fit or selection. Split unit = event family (temporal + family disjoint). Manifest hashed. Distinct from
the Phase-15 locked benchmark.

## Exact unresolved dependencies (before this is "complete")

1. **Phase 11 integration** — absent from base; final calibrators are PROVISIONAL. Rerun `phase12_refit.py`
   after Phase 11 lands and the corpus is regenerated from the post-Phase-11 path.
2. **Full-force single path** — the facade default is `pipeline.simulate` (no posterior); Phases 8/9 are
   separate pipelines; nonlinear is CLI-only. Unifying every causally-relevant subsystem into one
   question→forecast path is not done here.
3. **Scale** — 148 verified real forecasts across 9 domains, far below the 1,000 / 10-domain target. Resumable
   pipeline provided; the count is reported, not relabelled.
4. **Support-grade full monotonicity** — validated only at the extremes on 41 test rows (see validation doc).
