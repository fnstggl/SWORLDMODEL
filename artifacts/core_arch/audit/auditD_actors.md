# Audit D — Qualitative-actor decision path & integration surfaces (branch claude/worldmodel-v2-core-arch-73s7eq, 2026-07-19)

## 1. The actor decision path today

The production decision mechanism is the **persistent qualitative LLM actor** (`swm/world_model_v2/qualitative_actor.py`). Default mode is `hybrid_relevant_actor_policy` (materialize.py:285-298); `materialize.operators_from_plan` (201-215) binds `ProductionActorPolicyOperator` with a `QualitativeActorPolicyRuntime` built by `build_qualitative_runtime` (qualitative_actor.py:1404). Three seams invoke it, all funneling into **`QualitativeActorPolicyRuntime.decide` (qualitative_actor.py:902-997)**:

1. **Fixed-v1 events** — `ProductionActorPolicyOperator.run` (phase4_execution.py:637-690) on `decision_opportunity` / `actor_reaction` events.
2. **Generated world (production default)** — `GeneratedActorInvocationOperator.run` (generated_world.py:1196-1308) on `ctrl_invoke_actor` events, which arrive only through the temporal chain: `emit_semantic_event` (verified observability only) → `route_semantic_event` (per-recipient channel delivery timing, 803-870) → `GeneratedObservationDeliveryOperator` (availability, NOT exposure; 1047-1116) → `schedule_attention`/`collect_attention_bundle` (temporal_runtime.py:331-389; exposure into the InformationLedger happens at notice time) → `GeneratedAttentionOperator` (one noticed bundle → one DecisionTrigger → one invocation, 1119-1193).
3. **Personal-reaction route** — `simulate_individual_reaction` (individual_reaction.py:231-233) after the stimulus travels delivery→attention.

Inside `decide`: multi-particle bridge stays numeric (909-913); tier routing (`_routes_qualitative` 889-899, `RelevantActorSelector` actor_selection.py:59-185); then **view build (933) → action space + menu (935-936) → persistent-state load/init (937-943) → `engine.decide(view, state, situation, menu)` (946)**. Failure ladder: parse (`parse_qualitative_decision` 553-598, truncation salvage 535-550) → retries (1) → fallback model families once each (752-772) → numeric fallback marked `excluded_from_qualitative_aggregation` (947-961). Budget: `SWM_ACTOR_LLM_BUDGET` default 240 across the whole run; exhaustion at operator level = recorded temporal truncation, never an invented decision (phase4_execution.py:663-682; generated_world.py:1232-1242). One obstacle-revision round on perceived infeasibility (965-976).

### Context assembly is FIXED slicing
`build_prompt` (775-806): observations = `reversed(view.observed_events)[:10]`, each `content[:220]`; constraints = rules[:8] + authority + binding commitments + resource names; history = last 6 action names; situation[:400]; menu[:14]. `_render_state` (808-847) caps every state section (beliefs[:8], memories[-8:] of 12 stored, etc.). Upstream, `ActorViewBuilder.build` (phase4_policy.py:385-505) puts **ALL** ledger-visible items into `observed_events` (no cap, no salience use — Exposure.salience decays but is never consulted). In generated mode the situation string is pre-truncated even earlier (bundle 8×220 chars / `exact_content[:400]`, generated_world.py:1258-1268). There is no attention-within-bundle, no memory retrieval (beyond fixed tail slices), no separate interpretation stage, no action search — the single decide call does everything.

### State, particles, output
Private state = `QualitativeActorState` under `entity.latent_state["qualitative_actor_state"]` per **branch world** (117-205); K=3 hypotheses generated once per (actor, time, structural_frame) and assigned branch_index mod K (297-443); state evolves via `_post_execute` (1114-1157) on the action's own StateDelta. No cross-branch decision caching; `ScopedActorCache` shares byte-identical calls only across structural models at the same particle index (CRN). Output = one TypedAction (menu/modified/novel-compiled, 1003-1045 + NovelActionCompiler 602-689), degenerate posterior {selected:1.0} with rich `provenance.qualitative` (1047-1097); distributions arise later in `aggregate_actor_decisions` (1327-1400: cluster-2.0 → raw frequency → external calibration or `unvalidated`). Cross-session: qualitative state is NOT persisted; phase8 persists typed variables + episodic memory exposure only (phase8_pipeline.py:129-287, 356-378).

Adjacent but separate: `llm_actor.py` (persona blend, arm B, numeric minting) and `actor_cognition.py` (Phase 2C numeric interpretation + fitted policy + attention latents — the only existing "attention/interpretation" code, deliberately outside the qualitative mind). `event_time.py` contains NO working-memory/attention content (only hazard "memoryless continuation").

## 2. Insertion points for the staged bounded-cognition pipeline

- **Primary: `QualitativeActorPolicyRuntime.decide`, qualitative_actor.py:933-946** — between view/menu/state assembly and `engine.decide`. All three seams pass through here with the branch world and persistent state in hand. Stage observation→attention→WM update→memory retrieval→interpretation here; limited action search filters `actions`/`menu` before the choice call.
- **`QualitativeDecisionEngine.build_prompt`:775-806** — replace the fixed slicing with staged outputs (deterministic per particle: prompt bytes are the CRN cache key and prompt_hash provenance).
- **`QualitativeDecisionEngine.decide`:743-773** — the terminal "one choice" call; extra stages must share its budget/abstain ladder.
- **`QualitativeActorState` (+`_post_execute`:1114-1157)** — working-memory home on the branch world (non-`phase4_policy_` latent key; gated on `config.persistent`).
- **`GeneratedActorInvocationOperator.run`, generated_world.py:1258-1280** — pass the structured bundle through `decision{}` instead of a pre-truncated string.
- **`collect_attention_bundle`, temporal_runtime.py:354-389** — the existing observation→attention stage (currently takes ALL available items); capacity bounds extend here without re-deciding noticing.

## 3. Integration surfaces that must be preserved

- **Structural ensemble** (structural_contracts/ensemble_compiler/structural_runtime): 3-4→8(12) independent models, per-model plan/posterior/particle isolation (integrity-enforced), pilot=max(8, 20%N) prefix, full ≥N budget per model, equal-weight labeled mixture + qualitative support classes only, sensitivity thresholds 0.05/0.15/regret 0.15, disagreement/decomposition/reversal/VOI blocks.
- **Temporal**: event-driven only — DecisionTrigger required on every decision; per-actor channel-check cycles (anchor phase + sampled gap × latent factor), urgency interrupts, waking windows; delivery≠attention≠decision with delay quantiles in TemporalRunStats; deferrals compile to calendar/conditional triggers, never retry timers; particle-root RNG for CRN matched arms; safety budgets → recorded truncation.
- **Phase 13**: `human_approval_required=True` on every DecisionProblem (contracts.py:139); recommendation_kind action|policy|pareto|abstain|gather_information; ensemble-default evaluation (SingleModelContextError guard); scenario-generated action search with typed gates, blind diagnosis-driven revision, matched CRN evaluation.
- **PR#115 messaging**: general planner decides target/goal/timing; `make_message_realizer` (message_bridge.py:18-41) hands ONLY the wording to `ReplyFirstPlanner` (truth/language/outcome judges); realized `exact_content` rides the candidate step into kernel ops and reaches the recipient's own qualitative actor through delivery→attention→invocation. Today the recipient's prompt truncates it to 400/220 chars.
- **Personal-reaction**: `_route_individual_reaction_ensemble` (structural_runtime.py:757-928) runs several causal frames, each `simulate_individual_reaction` with `structural_frame` conditioning hypotheses; outcomes responded / read_but_deferred / unread_by_horizon (unread = never-invoked, explicit mass).
- **Persistence**: phase8 prepare/slice/finalize funnel; index-keyed particle slices (pilot prefix reuse); single checkpoint commit; leakage-safe replay; episodic memory → entity.memory → view.

## 4. Top risks (detail in auditD_actors.json "risks")
1. Prompt-byte determinism per particle (CRN cache + prompt_hash + replay).
2. Message truncation vs. PR#115 realized text (WM must carry the full message).
3. Budget multiplication + truncation semantics for mid-pipeline failures; revision re-entrancy.
4. No-numeric-cognition invariant for interpretation/attention outputs.
5. Branch isolation + stateless-arm gating of working memory; provenance/aggregation contract (`decision_source`, `hypothesis_id`, `state_hash`, `act_or_wait`, `revisit`) and deferral compilation must survive unchanged.
