# p=None investigation — root cause, fix, and reproduction (Knesset)

**Malformed run under investigation** (`experiments/results/prop_check.json`):
`{qid: 0851f82c… (Knesset "Associations Bill" first-reading by 2026-07-01), p: null, status: under_modeled, calls: 189, secs: 1707}`.
A usable grounded ~5% outside-view prior existed (deadline-aware estimator: stage
`mere_proposal_or_speculation`, base_rate ≈ 0.05) — and the FutureSearch SOTA for this question is 3%,
actual outcome **NO**. So the ~5% the system discarded was an *excellent* forecast. The system returned
nothing.

The two problems were deliberately kept separate (they are not the same bug):
1. **Output failure** — a grounded prior existed but the final result was `None`.
2. **Runtime** — where did the 189 calls / 28 min go?

---

## 1. Output failure — exactly where the 5% disappeared

Traced statically through the canonical path
`unified_runtime.simulate_world → structural_runtime.simulate_structural_ensemble`.

The forensic questions, answered:

| question | answer |
|---|---|
| Was the prior passed into the simulation? | **Yes.** `_phase3_block` builds `build_outcome_rate_prior(plan)` and stashes it on the plan (`plan._outcome_prior_spec`) and on every per-model record (`rec["prior_spec"]`, structural_runtime.py:343). |
| Did §NAP's `unresolved_mass ≥ 0.999` branch reject it? | **No** — that branch (pipeline.py:125) produces `unresolved`; the run was `under_modeled`. |
| Did `harden_general_path` overwrite it? | **No.** |
| Did unresolved mass force `p=None`? | **Partly** — the ensemble headline is emptied (`headline = {}`) under material structural disagreement, and `raw_probability` is then `None`. But that is downstream of the real cause. |
| **Was there no defined fallback?** | **This is the root cause.** |
| Did the final output formatter discard the prior? | **Yes, effectively** — the ensemble aggregator *is* the final formatter, and it never consulted the prior. |

### Root cause (one sentence)
`simulate_structural_ensemble` assembles its final `SimulationResult` fresh (structural_runtime.py:899-940)
and returns it **without ever running the no-silent-None guard** — `unified_runtime._apply_result_guards`
runs only **per-model** (structural_runtime.py:526), *before* aggregation. So the aggregated result the
caller actually receives has no fallback-to-grounded-prior path at all.

Two things then combine:
- **Status ladder (structural_runtime.py:886-898):** the moment any promoted model carries an
  `under_subtype` (a missing boundary, an unmodeled external process, or a suppressed non-human mechanism),
  the status is set to `under_modeled`.
- **Headline (structural_runtime.py:567, 904-905):** under material structural disagreement the headline
  mixture is suppressed to `{}`, so `raw_probability` is `None`.

The grounded `prior_spec` — carried on every model and on the plan — is **never read** when building the
ensemble headline. A supported ~5% reference-class estimate was discarded and the run shipped a silent `None`.

This is bad architecture **for forecasting mode**: `under_modeled` is meant to *lower confidence*, not
*erase a supported probability*.

---

## 2. The fix — grounded outside-view survives an under-modeled rollout

`claude/wmv2-grounded-outside-view` (commit `ebc664f`).

- **`phase3_priors.py`** — centralize the grounded/generic boundary so it can never drift:
  `GROUNDED_SOURCE_CLASSES = {reference_class, llm_estimated_reference, recurrence}` and
  `is_grounded_prior(spec)`. A `generic_weakly_informative` prior (fixed qualitative-lean Beta, **no**
  reference class) is **not** grounded.
- **`structural_runtime.py`** — a **forecasting-mode no-silent-None guard at the ensemble level**
  (`_ensemble_grounded_forecast` + `_serve_grounded_outside_view`), run right before the result is
  returned. When the headline is `None` and the status is not a genuine engineering failure, it serves the
  **grounded** outside-view forecast (evidence-updated posterior mean if any model had one, else the
  grounded reference-class prior mean) as the headline, **keeps** the epistemically-weak status as a loud
  warning, and records `provenance.grounded_outside_view_fallback`. A bare `unresolved` is retagged
  `under_modeled` with a *named* gap (in forecasting mode "no model validated a complete mechanism" is
  under-modeling, not a refusal to forecast).

**Contract:** an epistemically-weak status LOWERS confidence — it must never ERASE a supported
probability. **§NAP is preserved:** a GENERIC (~0.5, no reference class) prior is still never manufactured
into a headline; only a grounded reference-class estimate survives.

Example intended output for Knesset: *"Forecast 5%, status under_modeled — the actor rollout could not
validate a complete causal mechanism, so the forecast falls back to the grounded outside-view estimate."*

### Test (commit `ebc664f`)
`tests/test_grounded_forecast_survives_under_modeled.py` pins both directions:
- forecasting mode cannot return `None` when a grounded forecast exists (`has_forecast()` True, weak
  status kept as a warning, `unresolved`→`under_modeled` with a named gap);
- a generic-only prior still refuses (§NAP).

244 pre-existing world-model tests still pass.

---

## 3. Reproduction exposed a SECOND failure mode — and the deeper fix

`python -m experiments.exp109_knesset_repro` — single run, full LLM actors, on the fixed runtime — did
**not** reproduce the `under_modeled`/189-call path. It hit a *different* failure:
`status=execution_failed`, **zero promoted models**, limitation *"no executable structural candidate
remained after generation, critics and bounded repair"*, **38 calls / 403 s**.

**The run is stochastic.** The original malformed run reached the rollout (models survived → under_modeled,
189 calls, 1707 s); this re-run collapsed at ensemble **compilation** before any model — or any prior —
existed. The ensemble-level guard correctly did *not* fire (a genuine engineering failure, and there was
no per-model `prior_spec` to serve).

But the grounded outside-view estimate does not depend on the ensemble at all. So a **top-level
forecasting-mode floor** (`unified_runtime._forecasting_mode_floor`, commit `17f5d07`) now computes the
grounded reference-class prior *independently* and serves it as an `under_modeled` forecast whenever the
structural engine returns no forecast — while keeping the original status/taxonomy named in provenance and
a loud limitation, so the engineering failure is surfaced, not hidden. §NAP preserved (grounded-only;
generic never manufactured; `clarification_required` never floored). Two composed guards now:

| failure mode | who serves the grounded prior |
|---|---|
| models executed but rollout under-modeled (headline None) | ensemble-level guard (`structural_runtime`) |
| ensemble produced NO forecast at all (execution_failed, 0 models) | top-level floor (`unified_runtime`) |

### Timing note (what the 38-call vs 189-call gap means)
The 38-call run died at compilation (generation + critics + bounded repair produced zero survivors); the
189-call run got *further* — surviving models each ran full actor rollouts + particles + boundary/outside-
world across the ensemble. So the 189 calls are the architecture's **normal ensemble cost when models
survive** (≈ compile + N models × per-model conditioning/actor rollout), **not an accidental loop**. The
definitive per-stage `calls_by_stage` breakdown comes from the frozen-5 full-actor pass (EXP-110), where
runs that complete/under-model expose the stage histogram.

- **Fix verification (Step 4): CONFIRMED.** The Knesset re-run now returns
  `p=0.14, status=under_modeled, has_forecast=True, raw_distribution={True:0.14, False:0.86}` — no longer a
  silent `None`. It again hit the compilation-collapse path (`execution_failed`, zero models, 43 calls /
  446 s), so the **top-level floor** fired: it built the grounded prior independently (reference class
  *"Israeli Knesset committee approvals of private member bills with short deadlines"*, stage
  `formally_initiated` → 0.14) and served it, with the failure NAMED
  (`grounded_outside_view_fallback.original_status=execution_failed`,
  `original_failure_taxonomy=invalid_execution_plan`) and a loud limitation. Against the held-out answer
  (outcome NO, SOTA 3 %), 0.14 is on the correct side. The prior varies run-to-run with the stage
  classification (0.05 `mere_proposal` ↔ 0.14 `formally_initiated`) — expected LLM-estimate variance.

  **Engineering signal:** Knesset hit compilation collapse on BOTH re-runs (38 then 43 calls). The ensemble
  compiler frequently rejects every candidate for this question (`no_executable_structural_candidate` /
  `invalid_execution_plan`). The floor guarantees a forecast regardless, but the compiler's fragility on
  short-deadline institutional questions is a genuine, separate robustness gap worth a dedicated fix.

## 4. Lean §8-9 deadline-prior forecaster (EXP-111, Step 6)

The lean path = the grounded deadline-aware prior mean as the forecast (no rollout, ~2 calls/q).

| set | Brier | AUC | acc@0.5 | const base-rate | FutureSearch SOTA |
|---|---|---|---|---|---|
| frozen 5 (base 0.8) | 0.297 | **0.875** | 0.60 | 0.16 | 0.164 |
| 25-set (base 0.4) | 0.258 | 0.603 | 0.64 | 0.24 | 0.176 |
| combined 30 | 0.265 | 0.625 | 0.63 | 0.249 | 0.174 |

The lever gives real **discrimination** (AUC 0.60–0.88) — a large repair over the older rich-numeric code
(AUC 0.413, *anti*-discriminative). But absolute Brier (~0.26) is still slightly worse than the constant
base-rate and well behind SOTA (0.174): the prior UNDER-predicts specific advanced YES cases (BoJ 0.22→YES,
Wale 0.14→YES). That residual is exactly what evidence + actor simulation must close — the outside view is
the floor, not the ceiling. Whether the rich full-actor pass beats this floor is the open question
(EXP-110), and the malformed run does not settle it.
