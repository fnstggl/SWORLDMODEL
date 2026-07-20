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

## 3. Reproduction + per-stage timing (EXP-109)  — LIVE RESULTS PENDING

`python -m experiments.exp109_knesset_repro` — single run, full LLM actors, on the fixed runtime.

- **Fix verification (Step 4):** _pending_ — expect `p ≈ 0.05`, `status = under_modeled`,
  `has_forecast = True`, `grounded_outside_view_fallback.used = True`.
- **Timing profile (Step 1):** _pending_ — `llm_calls_by_stage` / `_by_model`, total calls, wall time,
  and the timing wrapper's true backend-call count vs. the ledger total (the gap = actor-rollout cost).
  This determines whether the 189 calls are an accidental loop / repeated validation or the architecture's
  normal cost.
