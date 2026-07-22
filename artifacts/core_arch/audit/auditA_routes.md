# Audit A — Default route walkthroughs (World Model V2, commit 5e7646c)

## (a) A forecast question, default path (no special env vars)

Caller: `swm.facade.forecast(q, architecture="world_model_v2", llm=..., as_of=..., horizon=...)`
(facade.py:75). The facade only validates the architecture name and stamps the RunRecord; the whole
simulation is `unified_runtime.simulate_world` (facade.py:89-94).

1. **Mode dispatch** — `simulate_world` (unified_runtime.py:88) reads
   `execution_policy["structural_mode"]`, default **"ensemble"** (:101). Everything below is
   `structural_runtime.simulate_structural_ensemble` (:55). `single_structural_model` is an explicit
   ablation only.
2. **Ledger + knobs** — a `CallLedger` and content-addressed cache store are created
   (structural_runtime.py:67-73); `compute_budget="maximum_capacity"` only raises the generation ceiling.
3. **Individual-reaction check** — if `user_context["individual"]` is a dict and the question is a
   personal reaction (`actor_selection.is_individual_reaction_question`), the run short-circuits into
   `_route_individual_reaction_ensemble` (:757): recon + critics generate causal *frames*; each surviving
   frame runs `individual_reaction.simulate_individual_reaction` (individual_reaction.py:140 —
   send → channel delivery → attention (timezone/sleep/checking habits) → qualitative decision) with the
   FULL n_hypotheses×samples budget; result is the equal-weight mixture over frames (:862).
4. **Stage A recon** — `ensemble_compiler.reconnoiter_structures` (ensemble_compiler.py:114): ≥3,
   normally 4, independent blind LLM calls, one per perspective (constants structural_contracts.py:63-66).
   `llm=None` → loud `execution_failed / unavailable_service` — there is no deterministic fallback model.
5. **Shared evidence** — gated on `as_of` being non-empty: `union_evidence_requirements` (:206) →
   `evidence_orchestrator.gather_evidence` (evidence_orchestrator.py:95) ONCE (Google News RSS +
   Wikipedia; ≤16 LLM claim-extraction calls). Replay drivers instead inject `prebuilt_bundle`
   (structural_runtime.py:107). Evidence failure never blocks the forecast.
6. **Critics + Stage B** — omission critic (:342), contrast critic (:418), adaptive expansion
   (:441/:468), per-candidate critics (:370); then `compile_candidates` (:616) calls the canonical
   `compiler.compile_world` (compiler.py:337) once per candidate with that candidate's
   `_structural_directive`, verifies executability by actually building the world
   (build_world + check_readout_binding + operators_from_plan, :692-706), and allows one bounded repair
   recompile. Conservative dedup (:547) + survivorship certificate (:710).
7. **Per-model conditioning** (`_condition_and_pilot_model`, structural_runtime.py:254) — for each
   surviving model’s own plan: evidence recompile (`_apply_evidence_to_plan`), its OWN Phase-3 posterior
   (`_phase3_block` → `plan.posterior_rate_particles`), then `_condition_plan` (unified_runtime.py:282):
   Phase-9 populations/networks, Phase-10 rule-kind normalization (integration_completion.py:35),
   the fidelity block (resolution criterion, actor decomposition, scheduled facts, mode graph for
   when-questions, evidence-grounded actor intentions, actor capacity), activation synthesis, the
   **temporal compiler** (temporal_compiler.py:241 — 2 generation + 2 critic LLM calls, cached per
   (plan, structural_model_id)), **event-time conversion** (event_time.py: when→:1087, categorical→:1087
   with options, binary→:1321 — removes `resolve_outcome` resolvers, installs a posterior-calibrated
   continuous first-passage "resolution" process, and swaps `plan.outcome_contract` for an
   `EventTimeContract`; when/categorical raise n_particles to ≥200 at :1254), and Phase-11
   recompilation (max 2 recompiles).
8. **Pilot through the ONE funnel** — `phase8_pipeline.prepare_persistence_run` (phase8_pipeline.py:129):
   `materialize.build_world` (:42) → `_bind_scenario_schema` (:441 — because `SWM_CONSEQUENCES` defaults
   to `generated_actor_mediated_world`, the scenario semantic schema + causal-boundary mechanism model
   are compiled and bound; failure stamps `structurally_under_modeled`) → `check_readout_binding` →
   `operators_from_plan` (:176 — plan operators + the actor-policy runtime for
   `SWM_ACTOR_POLICY=hybrid_relevant_actor_policy` (default) + the generated-world control plane:
   delivery, attention, semantic-event routing, observation delivery, actor invocation, scenario plan
   steps, MechanismRuntimeOperator, ScheduledAttemptOperator; materialize.py:229-277) →
   `WorldModelV2Run`. Budget: `plan.compute_plan["n_particles"]` (12..80 standard, ≥200 event-time).
   Pilot = `max(8, ceil(0.20·n_full))` particles, rolled as an index-keyed prefix slice:
   `run_persistence_slice` → `run_particle_range` (rollout.py:54) → per branch
   `temporal_runtime.run_branch_temporal` (temporal_runtime.py:892): pop earliest event batch, advance
   continuous processes over the exact interval, microstep same-time causality, run every applicable
   operator (each emits StateDeltas), reproject hazards, until quiescence/horizon; hitting
   `safety_max_events=2000` or the actor-LLM budget (SWM_ACTOR_LLM_BUDGET=240) records a
   **temporal truncation**, never fake quiescence.
9. **Promotion + full budgets** — `_promote_models` (:321): promote unless behaviorally
   indistinguishable (spread ≤0.02) from a structurally similar, better-supported model with a
   non-noisy pilot. `_extend_to_full` (:390) continues the SAME prepared run to ≥ its full budget
   (pilot reused as prefix; +25% extension if the models materially disagree). Budget invariant
   enforced at :222-229 (never divided across models).
10. **Per-model finalize** — `_finalize_model` (:416) → `finalize_persistence_run`
    (phase8_pipeline.py:226): `run.project(branches)` → `OutcomeContract.project` (contracts.py:65) or
    `EventTimeContract.project` (event_time.py:233 — answer = F(deadline) / first-passage read from
    trajectories), attach actor-decision distributions + consequence report, `pipeline.result_from_run`
    (pipeline.py:55 — statuses, temporally_truncated surfacing, temporal-model block, provenance),
    degradation surfacing, ONE checkpoint commit (first promoted model only), operator-phase manifest +
    phase supervision (`phase_supervision.assess/finalize`).
11. **Aggregation** — `_assemble_ensemble_result` (:443): equal-weight mixture (labeled UNCALIBRATED),
    robust range, uncertainty decomposition (between/within model), sensitivity classification
    (`classify_forecast_sensitivity`, structural_contracts.py:313 — stable <0.05 spread, mild <0.15,
    material ≥0.15; `ensemble_execution_incomplete` if any model died or a single survivor lacks a
    convergence certificate), reversal conditions, structural VOI, human summary. `SimulationResult`
    (:559) carries `structural_ensemble` + a live `_ensemble_handle` for Phase 13.

## (b) A recommend_action question, default path

Caller runs (a) first, then:

```python
res = simulate_world(question, ...)                      # SimulationResult with _ensemble_handle
r = phase13.api.recommend_action(problem, res, llm=...)  # world_context = the ensemble
```

1. `recommend_action` (phase13/api.py:90) → `extract_ensemble_models` (ensemble.py:33) resolves
   `res._ensemble_handle` into `{model_id: plan}`. A bare `WorldExecutionPlan` raises
   `SingleModelContextError` unless `allow_single_structural_model=True` (api.py:29-44).
2. `recommend_action_across_models` (ensemble.py:69) deep-copies the problem and re-enters
   `recommend_action` per model with `allow_single_structural_model=True`.
3. Inside each model, `mode="auto"` + `is_generated_context(plan)` is True (the plan carries its bound
   scenario schema), so the run routes to the **scenario-generated action layer**:
   `scenario_actions.api.discover_best_action` (scenario_actions/api.py:290) — goal-backward discovery,
   generated action language, kernel-compiled `ConcreteAction`s, matched simulation via
   `MatchedEvaluator` (counterfactual.py:69 — same build_world/_bind_scenario_schema/
   _inject_posterior_rate/operators_from_plan as forecasts, hypothesis-stratified matched particles,
   CRN per particle), diagnosis-driven revision. The fixed-v1 catalog is reachable only for
   non-generated (controlled) contexts or explicit `mode="legacy_fixed_v1"`.
   (The non-generated single-model pipeline in api.py:131-226 — affordances → feasibility →
   abstention gates → `select_and_run` under `SearchBudget.tiered(budget)` (standard=64 arms) →
   `evaluate_bundle` → Pareto → VOI → ranking — serves controlled benchmarks.)
4. Cross-model synthesis (`_synthesize`, ensemble.py:167): winner per model, minimax regret across
   models over shared actions, worst-model downside, labeled equal-mixture ranking. Decision form:
   ALL models agree → `recommendation_kind="action"`; disagree with discriminating observations →
   `"gather_information"` + `structurally_sensitive_recommendation` abstention; otherwise `"pareto"`
   with the minimax-regret robust set. `recommendation_stability`
   (structurally_stable / mildly / materially, ensemble.py:335) lands in
   `provenance.structural_ensemble`; per-model DecisionResults are preserved verbatim.
5. Every result stamps `causal_claim="simulated_mechanism_counterfactual"`, decision-action
   causal-boundary reports, and `human_approval_required` semantics (api.py docstring): nothing here
   executes an action in the world.

## Key default facts (for the refactor)

- Runtime identity: `unified_runtime` = orchestrator + shared per-plan phase helpers;
  `structural_runtime` = default driver; `phase8_pipeline` prepare/slice/finalize = the ONE terminal
  funnel; `temporal_runtime.run_branch_temporal` = the ONLY branch event loop; `generated_world` +
  `causal_boundary` operators = the default consequence plane (env-selected, default ON);
  `temporal_compiler` = per-model LLM temporal model (cache keyed by structural_model_id).
- Statuses: 5 simulation statuses (incl. `temporally_truncated`), 10 failure taxonomy codes,
  5 structural-sensitivity classes (incl. `ensemble_execution_incomplete`,
  `structurally_underidentified`), `structurally_under_modeled` consequence flag,
  phase13 recommendation kinds (5) + abstention codes. `run_classification` as a name does NOT exist.
- LLM default: DeepSeek `deepseek-v4-flash` via `swm.api.deepseek_backend.default_chat_fn`
  (DEEPSEEK_API_KEY, else HF Qwen router, else None → V2 ensemble fails loudly). Budgets: actor
  cognition 240 calls/run (SWM_ACTOR_LLM_BUDGET) enforced in QualitativeDecisionEngine + pre-gated by
  GeneratedActorInvocationOperator; 40 invocations/actor/branch; evidence ≤16 claim docs;
  CallLedger meters everything into `structural_ensemble.cost_manifest` (no ensemble-level hard cap).
- Calibration: Phase-12 serving is NOT wired in; both runtimes stamp
  `calibration_compatibility=INCOMPATIBLE`; `calibrated_probability` is None by default.
- Posterior injection asymmetry: `_inject_posterior_rate` runs on `run_from_plan` and Phase-13 paths,
  NOT in `prepare_persistence_run`; on the default path the posterior reaches execution through the
  event-time calibration payloads instead (event_time.py:1424 → :520/:709).
