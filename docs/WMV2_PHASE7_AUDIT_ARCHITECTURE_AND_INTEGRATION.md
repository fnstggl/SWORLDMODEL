# WMV2 Phase 7 — Audit, Architecture & Integration

Phase 7 adds a **production nonlinear- and context-dependent-mechanism subsystem** to World-Model v2. This
document is one of exactly four authoritative Phase-7 Markdown files; all detailed records live in the
machine-readable artifacts under `experiments/results/wmv2_phase7_*.json` and the committed sidecar registry
`swm/world_model_v2/registry/data/nonlinear_extensions.json`.

- **Audit** → `experiments/results/wmv2_phase7_audit.json`
- **Form registry** → `experiments/results/wmv2_phase7_form_registry.json`
- **Mechanism↔form compatibility** → `experiments/results/wmv2_phase7_mechanism_form_compat.json`
- **Context / history schemas** → `wmv2_phase7_context_schema.json`, `wmv2_phase7_history_schema.json`
- **Merge notes for Phase 9/10** → `experiments/results/wmv2_phase7_merge_integration.json`

---

## Part 0 — Ruthless nonlinearity audit (summary; full rows in the artifact)

All **63 Phase-6 families** were audited for whether their current form assumes linearity, which nonlinear
phenomena are *possible* for their causal process, and which are *already modeled*.

**What the registry already does well.** Nonlinearity is not new to the registry. `exposure_response_hazard`
(log-linear, degree-conditioned) is `locally_validated`; `complex_contagion_hazard` (Hill) is implemented;
`bass_diffusion`, `finite_population_saturation`, `quantal_response_choice`, `position_bias_propensity`,
`platform_examination` are genuinely nonlinear. The Hawkes point process is **quarantined** (a preserved
held-out failure) and must never silently return.

**The three genuinely under-served phenomena** (modeled-count near zero across all 63 families):

| Phenomenon | State before Phase 7 | Phase-7 action |
|---|---|---|
| **parametric interactions** | the *only* interaction term in any executable form was `k/deg` in the Higgs hazard | added evidence-selected interaction terms (with a dedicated interaction standardizer); tested on telco (`tenure×contract`), cmv, stackexchange |
| **fatigue / habituation** | absent — decay forms decay a *signal*, not the responder's *sensitivity* | added `fatigue`, `habituation`, `refractory` forms + a typed exposure-count history |
| **hysteresis / path dependence** | only `trust_update`'s gain/loss asymmetry (no bistable band) | added a true two-threshold `hysteresis` form (state-dependent) |

Disposition counts (from `wmv2_phase7_audit.json`): `test_nonlinear` 46, `extend` 5, `retain_linear` 6,
`retain` 1, `retain_linear_as_comparator` 1, `quarantine_preserved` 1 (Hawkes), `structural_candidate` 1,
`test_nonlinear_expect_null` 2 (StackExchange, CMV — the known Phase-6 nulls, used as adversarial checks).

The audit's guiding principle, enforced downstream: **the goal is not to make every mechanism nonlinear — it
is to detect where nonlinear structure is real, useful, stable, and causally relevant, and to keep the simpler
form everywhere else.**

---

## Architecture — before / after

### Before (Phase 6)
```
question → compile_world → WorldExecutionPlan → build_world → WorldState
  → registered TransitionOperators (mostly linear / logit-linear) → StateDelta → rollout → SimulationResult
```
The one nonlinear execution template is `FeatureHazardOperator` (hazard.py) — a fitted hazard carried on the
event payload as a `hazard_spec`. Diffusion nonlinear forms (`HillHazard`, `LogLinearHazard`) exist in
`registry/families/diffusion.py` but run in a *standalone* survival loop (Mode B), not the shared rollout.

### After (Phase 7) — additive package `swm/world_model_v2/nonlinear/`
```
compiler identifies causal process → Phase 6 selects family + pack
  → Phase 7 nonlinear.applicability decides nonlinear-vs-linear (keep_linear on weak support)
  → nonlinear.forms selects a structural form; nonlinear.fit's serialized params load
  → a scenario nonlinear_spec is bound to real state/history/context
  → WorldState materializes typed state; the triggering event carries the spec on its payload
  → nonlinear.operators (Mode-A TransitionOperators) read typed context (nonlinear.context) + typed
    history (nonlinear.history, leakage-guarded) + Phase-3 posterior (nonlinear.posterior, per-particle)
  → evaluate the fitted form → nonlinear.safety guards the value
  → emit an explicit StateDelta + schedule future events (retransmission / recurrence / next state-step)
  → later actor/state/terminal outcome changes; provenance + structural uncertainty preserved
```

The subsystem's modules (all pure-Python runtime; numerics offline-only):

| Module | Role | Spec parts |
|---|---|---|
| `forms.py` | typed structural-form registry, 31 evaluable forms + full metadata | 1 |
| `context.py` | typed context-conditioning schema + leakage audit | 2 |
| `history.py` | typed event-history/memory schema (strictly ≤ now) | 11 |
| `posterior.py` | **the missing** Phase-3 per-particle propagator, `E[f(X)] ≠ f(E[X])` | 12 |
| `pooling.py` | hierarchical partial pooling (empirical Bayes) | 3 |
| `structural_uncertainty.py` | evidence-weighted competing forms + disagreement | 9 |
| `applicability.py` | nonlinear applicability + transport/extrapolation gates | 14, 15 |
| `composition.py` | nonlinear composition + stability guards | 16 |
| `safety.py` | numerical safeguards + append-only failure records | 17, 26 |
| `operators.py` | execution plane: nonlinear TransitionOperators (StateDelta + future events) | 18 |
| `registry_ext.py` | additive integrity-hashed Phase-6 sidecar registry | 13 |
| `fit.py` / `compare.py` | **offline** fitting (numpy/scipy/sklearn optional) + model comparison | 6-8 |
| `audit.py`, `cli.py` | Part-0 audit + reproducible CLI | 0, 28 |

### Dependency separation (per the mid-run directive)
- **Runtime** (everything the rollout touches) is **pure-Python, dependency-free** — the core `swm/`
  invariant is preserved, the existing dependency-free test suite stays green, and the runtime evaluates only
  *serialized* fitted parameters (evaluating a fitted spline/Hill/GAM is trivial pure Python).
- **Offline fitting** (`fit.py` + the experiments) *may* use NumPy / SciPy / scikit-learn where they
  materially improve fit quality/stability (L-BFGS logistic, isotonic regression, Nelder–Mead NLS), with a
  **pure-Python fallback** so nothing is a hard dependency. Versions are captured in every fit's provenance
  (`software` field). Environment: Python 3.11, numpy 2.4.6, scipy 1.17.1, sklearn 1.9.0.

---

## Phase-3 integration (Part 12)

The Phase-3 audit confirmed **no generic per-particle `E[f(X)]` evaluator existed** in the codebase.
`nonlinear/posterior.py` adds it: `ParamPosterior` adapts every Phase-3 posterior representation
(`PosteriorResult.outcome_rate_particles`, a fitted grid, a `{mean,sd,lo,hi}` envelope, raw samples) behind a
uniform `.sample(n)`; `propagate(form, param_posteriors, inputs)` evaluates the form **once per particle** and
returns the posterior-correct `E[f(X)]`, its spread, quantiles, and the **measured Jensen gap** vs the naive
`f(E[X])`. `delta_method_gap` gives the cheap curvature estimate and flags when per-particle evaluation is
mandatory. `NonlinearMechanismOperator` uses this automatically whenever the spec carries `param_posteriors`,
and stamps `posterior_propagated` + `jensen_gap` onto the StateDelta's `uncertainty`.

## Phase-6 integration (Part 13)

Phase 7 does **not** modify `registry.json` / `packs.json` (Phase 9/10 are editing those in parallel).
Instead it writes an **integrity-hashed sidecar** `registry/data/nonlinear_extensions.json`. Each
`NonlinearExtension` references a Phase-6 `family_id` and adds `candidate_forms`, `selected_form`,
`form_posterior`, `context_conditioning`, `history_requirements`, `nonlinear_validation`,
`nonlinear_ablation`, `extrapolation_limits`, `status`, and `failures`. Promotion is gated
(`promotion_blockers`) exactly like Phase 6: a nonlinear extension cannot reach `locally_validated` without a
PASSED held-out record that names a beaten baseline. `verify-registry` recomputes the hash and checks every
extension joins to a real family (currently: 5 extensions, 0 dangling refs).

## Shared-world execution (Part 18)

Three registered operators (all via the public `register_operator` API — **no edit to `transitions.py`**):
`nonlinear_mechanism` (form → outcome, with context/history/posterior), `nonlinear_contagion` (diffusion:
exposure history → hazard → activation → retransmission future events), `nonlinear_state_step` (a scalar state
stepped through a growth form, scheduling the next step). All emit `StateDelta`s and follow-up events into the
existing rollout; the terminal answer is read from world state, never returned by a bypass.

---

## Compatibility & merge-conflict notes for Phase 9 and Phase 10 (Part 33)

**Phase 7 edits no shared core file.** Full detail in `experiments/results/wmv2_phase7_merge_integration.json`.
Integration is entirely through public seams:
- `transitions.register_operator` (registers the three operators at import, like `hazard.py`);
- `state.register_entity_extension` (registers `p7_history` + `p7_mechanism_fields` typed fields);
- `events.register_event_type` (registers `nonlinear_transition` / `contagion_exposure` / `state_step`);
- `event.payload` (the `nonlinear_spec` / `contagion_spec` / `step_spec` ride the payload, like `hazard_spec`).

**Phase 9 (population/network) and Phase 10 (institutions) are consumed, not rebuilt.** Phase 7 reads
population/network/institution context only through typed `ContextVariable`s with `source ∈
{population, network, institution}` and a `state_path`; when 9/10 land, those paths bind to their accessors,
and until then the variables fall back to typed defaults. No substitute population/network/institution system
is built. **Recommended merge order:** 9 → 10 → 7 (7 only adds consumers), though 7-first is also safe because
its hooks default gracefully. **No manual conflict resolution is expected** — if 9/10 rebuild `registry.json`,
the sidecar is a separate file and is unaffected (re-run `python -m experiments.wmv2_phase7_build_registry` to
refresh join-key health).

## Migrations

One additive committed data file: `swm/world_model_v2/registry/data/nonlinear_extensions.json` (new; not a
migration of an existing schema). No existing artifact is rewritten. The Hawkes quarantine and all Phase-6
failure records are untouched and re-verified by `tests/test_wmv2_phase7_adversarial.py`.

## Artifact index

| Artifact | Contents |
|---|---|
| `wmv2_phase7_audit.json` | Part-0 audit, all 63 families |
| `wmv2_phase7_form_registry.json` | the 31 structural forms + metadata |
| `wmv2_phase7_mechanism_form_compat.json` | family → causally-meaningful forms |
| `wmv2_phase7_context_schema.json` / `_history_schema.json` | typed context + history schemas |
| `wmv2_phase7_validation.json` | component fits, identical-split comparison, calibration |
| `wmv2_phase7_ablations.json` | ablation ladder |
| `wmv2_phase7_historical_backtests.json` | end-to-end WorldState backtests (3 categories) |
| `wmv2_phase7_counterfactuals.json` | sensitivity sweeps (invariants) |
| `wmv2_phase7_forensic_traces.json` | full question→StateDelta traces |
| `wmv2_phase7_failures.json` | append-only failures (Hawkes preserved + Phase-7 nulls) |
| `wmv2_phase7_scenario_instances.json` | bound nonlinear_spec exemplars |
| `wmv2_phase7_merge_integration.json` | Phase 9/10 merge notes |
| `registry/data/nonlinear_extensions.json` | the committed sidecar registry |
