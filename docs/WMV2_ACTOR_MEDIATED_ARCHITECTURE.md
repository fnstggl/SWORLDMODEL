# Universal Actor-Mediated Causal Execution — architecture map, implementation, and forensic map

Status: LIVING DOCUMENT for the actor-mediated-execution phase. Section A is the
pre-implementation audit map (frozen as written before code changed). Section B is the
implemented architecture. Section C is the post-implementation forensic map. Section D is the
honest verdict.

## A. Pre-implementation architecture map (audit of `claude/world-model-v2` @ 48cd060)

### A.1 The production execution chain (verified by direct read)

```
simulate_world (unified_runtime.py:51)
→ compile_world (compiler.py) → ONE WorldExecutionPlan
→ gather_evidence → recompile_with_evidence → attach_evidence_observations
→ infer_posterior (phase3) → plan.posterior_rate_particles / structural_posterior
→ fidelity_expand + scheduled facts + canonical modes + ground_actor_intentions
  + declare_actor_capacity + declare_pathway_processes
→ activation_synthesis (relevance-gated completion) → event_time conversion
→ phase11 recompilation
→ _project_terminal → phase8_pipeline.run_with_persistence
   → materialize.build_world → operators_from_plan
     (binds QualitativeActorPolicyRuntime when an LLM backend exists —
      hybrid_relevant_actor_policy default, SWM_ACTOR_POLICY override)
   → WorldModelV2Run.run: InitialStateModel.sample_particles(n) → per-particle
     queue_builder_from_plan → RolloutEngine.run_branch (the ONE canonical queue;
     follow_up_events on StateDelta are validated + queued — the A4 endogenous chain)
   → contract.project(branches) → result_from_run → SimulationResult
```

Best-action: phase13 (api.py) generates the action space, clones matched joint particles
(counterfactual.py + crn.py), applies the intervention event, and rolls the SAME operator
set per arm; ranking/abstention from matched terminal outcomes.

### A.2 What already exists and is preserved

* Persistent qualitative actor state per branch (`qualitative_actor.QualitativeActorState`,
  stored under `latent_state[qualitative_actor_state]`, branch-isolated by deep copy).
* Per-branch LLM choice (`QualitativeActorPolicyRuntime.decide` — one action per particle;
  degenerate observed-choice posterior; NO probability minting).
* Counted action distributions (`aggregate_actor_decisions` — cluster → raw frequency →
  external calibration or `unvalidated`).
* Typed actions + separate perceived/actual feasibility (`phase4_policy.FeasibilityEngine`),
  attempted-but-blocked as a real outcome (`phase4_execution.execute`).
* Actor-local views (`ActorViewBuilder` — fail-closed; PRIVATE_ACTOR_FIELDS and
  SIMULATOR_ONLY_FIELDS excluded; information via `InformationLedger.visible_to`).
* Relevant-actor tiering (`actor_selection.RelevantActorSelector` — causal, question-specific,
  with dynamic event-time promotion).
* Event-driven real-time rollout with hazards, background dynamics, stance dynamics
  (`world_dynamics.StanceReviewOperator`), persistence semantics, sampled coupling constants.
* Novel-action compilation with ontology anchoring or explicit `novel_action_unmodeled`
  (`NovelActionCompiler`).

### A.3 Scalar actor-mediated bypasses found (each with location)

1. **Direct recipient belief writes** — `phase4_execution._apply_immediate_consequences`
   (`kind == "belief_delta"`): an action writes `target.beliefs[key] ± delta` directly. The
   recipient never decides anything.
2. **Ontology-level pathway coefficients as recipient reactions** —
   `phase4_policy.ACTION_PATHWAY_EFFECTS` + `phase4_execution._apply_pathway_effects`: every
   executed action immediately moves `pathway_progress:*` by `sampled_coupling × effect`,
   including actions whose causal substance is another actor's decision (persuade, coordinate,
   support/oppose by non-holders, messaging reveals). `accept → cooperative_agreement +1.0`
   stands in for the counterpart's response.
3. **Fixed reaction menus, target-only scheduling** — `phase4_execution._follow_up_events`
   `reaction_scheduling`: exactly ONE `actor_reaction` event for the explicitly named target,
   candidate set defaulting to `["acknowledge", "ignore"]`. No frontier discovery, no
   audience, no institutional participants, no propagation beyond one hop.
4. **Implicit actor visibility** — event visibility is a filtering boundary
   (`ActorViewBuilder._event_visible`, `InformationLedger.visible_to`) rather than a delivery
   system: nothing converts an executed action into per-actor observations with channel,
   perceived source, credibility, or distortion; most executed actions never enter any other
   actor's information set at all.
5. **Independent actor hypotheses without joint-world conditioning** —
   `QualitativeParticleHypothesizer` builds ONE hypothesis set per actor (keyed only by
   actor+time) and assigns `branch_index mod K` per actor independently. Two actors' hidden
   realities in one particle are uncorrelated by construction; nothing prevents jointly
   incoherent combinations.
6. **Numeric reaction priors as world truth** — `SubjectiveConsequenceModel.predict` reaction
   priors {observe/respond/ignore} feed the numeric utility; acceptable as actor-subjective
   anticipation but also the only "reaction model" for non-routed actors.
7. **Polarity aggregation** — `phase_consumers.ActorActionPolarityOperator` writes
   `actor_action_share` from lexical polarity of chosen actions (population-style consumer;
   acceptable as aggregate but must be stamped).
8. **Pooled multi-particle decisions** — `decide()` with >1 world routes to the numeric
   multi-particle bridge (`multi_particle_bridge_is_numeric`), recorded; per-branch qualitative
   decisions only when the rollout passes exactly one world (which the production rollout does).
9. **Institutional decisions from posterior rates, not executed votes** —
   `phase_consumers.CollectiveThresholdDecisionOperator` draws member yes-propensity from
   posterior rate particles rather than counting executed member actions when members are
   representable actors.
10. **Ornamental information ledger on the action path** — executed actions write
    `current_action`/`past_actions` but publish nothing into `world.information`, so
    ActorViews of other actors don't see them except through `observed_events` passed ad hoc.

### A.4 Constraint inventory (what must NOT change)

* The canonical queue (`events.EventQueue`) + A4 follow-up validation stays the only event path.
* `StateDelta` remains the only mutation record; operators registered via `register_operator`.
* No LLM probability minting; terminal probabilities only from `contract.project(branches)`.
* Phase 13 counterfactuals must keep matched particles + per-(particle, stream) seeds.
* No new parallel simulator; no bypass of `simulate_world` / `run_with_persistence`.

## B. Implemented architecture (this phase)

New modules (all wired into the production funnel, none demo-only):

* `semantic_events.py` — versioned `SemanticEvent` (`semantic.event.v1`) preserving exact
  content, semantic commitments, channel, intended audience, observability, provenance,
  credibility and institutional context, parent events, and the branch's world-hypothesis id.
  `compile_semantic_events(action, world, decision)` turns ANY executed TypedAction (menu,
  known-ontology, or novel) into one or more semantic events — multi-target private outreach
  compiles into one private communication event per target with no ontology coefficient.
* `observation_delivery.py` — `ObservationRouter`: for each semantic event and world particle
  decides who CAN receive, who DOES receive, when, in what representation (original / summary /
  relayed, with distortion metadata), through which channel, with what perceived source and
  credibility, honoring institutional information boundaries; appends actor-local observations
  to the branch's `InformationLedger` (publish + expose) with provenance back to the original
  event. The router owns reach; the LLM actor never decides whether information reached it.
* `causal_frontier.py` — `CausalFrontierDiscovery`: event-specific recipient discovery
  (direct targets → intended audience → actual recipients → institutional decision/veto
  holders → relevant network neighbors → threshold-relevant members → dynamic promotion via
  `RelevantActorSelector.promote_if_consequential`), tier assignment, per-event budgets, and
  approximation stamps for every Tier-3 substitution.
* `actor_propagation.py` — `SemanticPropagationEngine` + `actor_reconsideration` event type:
  executed action → semantic events → routed observations → frontier → reconsideration events
  queued through `StateDelta.follow_up_events` (the canonical A4 path) → each consequential
  recipient's OWN qualitative decision per particle → typed execution → new semantic events →
  recursion, terminating on quiescence (semantic dedup), per-branch cascade budgets, depth
  caps, or the LLM-call budget — all stamped in the event-cascade manifest.
* `joint_world.py` — `JointWorldHypothesis` (`joint.world.v1`) + `JointWorldHypothesizer`:
  world-level coherent hidden realities generated FIRST (evidence-cited, assumptions labeled,
  adverse/private-collapse regimes required where consistent), assigned per particle with
  ancestry + weights; actor private states are then generated CONDITIONAL on the branch's
  world hypothesis (replacing independent per-actor mod-K sampling). Incoherent
  world/actor combinations are rejected or marked.
* `semantic_clustering.py` — versioned mapping v2 (`cluster-2.0`): exact typed match →
  canonical target normalization → ontology-equivalent synonym map → (optional) LLM-assisted
  equivalence constrained to the candidate ontology with structured justification and refusal
  to merge → strategy-class match → novel → unresolved. Every mapping recorded and
  replayable; locked human-graded fixture + metrics (exact/semantic accuracy, false-merge,
  false-split, unresolved).
* `run_classification.py` — every run self-classifies as `full_numeric_forecast` /
  `rank_only` / `scenario_distribution` / `structurally_underidentified` /
  `execution_failed`, and the product response carries the epistemic contract (what ran, what
  degraded, what was approximated).

Demotions in existing code:

* `phase4_execution._apply_immediate_consequences`: recipient `belief_delta` writes are
  DEMOTED — routed through the recipient's own reconsideration when the recipient is a
  representable consequential actor; the direct write survives only as a stamped fallback for
  non-representable targets.
* `phase4_execution._apply_pathway_effects`: every ontology effect is classified
  (`ACTION_EFFECT_CLASS`) as structural / actor_mediated / population / residual.
  Actor-mediated effects no longer write pathway progress when consequential recipients are
  representable and propagation is active — the process moves when the recipients' own
  executed reactions move it. Population effects write with an aggregate stamp. Structural
  effects (the actor's own vote, launch, resource spend, own-side process step) stay
  deterministic.
* `phase4_execution._follow_up_events` `reaction_scheduling`: replaced by frontier-based
  reconsideration scheduling (the old single-target path survives only when propagation is
  explicitly disabled, stamped).

## C. Post-implementation forensic map

### C.1 Where every former bypass now routes (file:behavior)

| Former bypass (§A.3) | Now |
|---|---|
| direct recipient `belief_delta` | demoted in `phase4_execution._apply_immediate_consequences`: skipped for representable person/institution targets with propagation live, recorded in `demoted_scalar_writes`; direct write survives only propagation-off, stamped `belief_delta_scalar_fallback_propagation_disabled` |
| pathway coefficients as reactions | classified per (family, action) in `phase4_policy.ACTION_EFFECT_CLASS`; actor-mediated effects skip the write (stamped `pathway_effect_demoted_actor_mediated`), population effects stamped `population_aggregate_pathway_write`, structural effects unchanged |
| single-target fixed reaction menu | `_follow_up_events` legacy branch gated on `propagation_enabled() == False` and marked `legacy_reaction_scheduling`; live path = frontier discovery + broad affordances + free actor choice |
| filtering-only visibility | `ObservationRouter` performs actual delivery into the branch ledger (channel, credibility, representation, distortion, timing, boundaries) |
| per-actor independent hypotheses | `JointWorldHypothesizer` world-first generation; `QualitativeParticleHypothesizer` conditions per-actor sets on the branch hypothesis; hypothesis ids are `<world>/<actor>` scoped |
| silent LLM-runtime construction fallback | `materialize._actor_policy_runtime` records `actor_runtime_fallback` on plan provenance → surfaces in the epistemic contract |
| institutional votes from posterior rates | `InstitutionalVoteOperator` coerces EXECUTED ontology actions to votes via lexical polarity (support→yes, defect→no); `CollectiveThresholdDecisionOperator` remains for population-scale bodies (stamped aggregate) |
| pooled multi-particle decisions | unchanged and still recorded (`multi_particle_bridge_is_numeric`); the production rollout passes exactly one branch world per decision |

### C.2 Deterministic architecture evidence (all offline, scripted mock actors)

* `tests/test_actor_mediated_architecture.py` — 15 tests, all passing: public-statement
  cascade with an executed-action vote; private-concession boundary + conditional relay
  (C reacts only when B actually relays, and to B's account); joint-world coherence +
  incoherence rejection; particle-isolated private state; novel two-target coordination with
  zero coefficients; event-time tier promotion; unauthorized-action structural block with no
  downstream semantic success; scalar-bypass regression; mutual-acknowledgement quiescence;
  Phase-13 matched-arm parity (exogenous traces identical, reactions diverge); private-state /
  simulator-truth prompt leakage; summary-vs-original representations; terminal-probability
  write refusal; same-seed replay.
* `tests/test_semantic_clustering.py` — 13 tests; locked fixture (44 cases, SHA-256-pinned):
  deterministic v2 exact_accuracy 0.386→semantic_accuracy 0.977, false-merge 0.0,
  false-split 0.033, unresolved 0.045.
* Three-arm benchmark (`experiments/actor_mediated_three_arm.py`, 12 particles, scripted
  actors, identical inputs/seeds): scalar arm — 0 reconsiderations, members never decide
  (the legacy target-only narrowness made visible), vote fails for lack of executed votes;
  one-hop — 36 reconsiderations, depth 1, all-support (nobody observes later defections),
  P(bloc holds)=1.0; recursive — 276 reconsiderations, depth 3, observed defections cascade,
  P(bloc holds)=0.33. Same operators, same particles, same seeds — the propagation regime is
  causally load-bearing. This is an ARCHITECTURE result, not an accuracy claim.

(Sections C.3 — live-LLM forensic traces — and D — honest verdict — are appended after the
live runs complete.)

### C.2b Live-LLM three-arm run (DeepSeek backend, 6 particles, same seeds)

`experiments/results/actor_mediated/three_arm_report_real.json`:

| arm | LLM calls | wall | reconsiderations | max depth | member decisions |
|---|---|---|---|---|---|
| scalar | 6 | 51s | 0 | 0 | none — members never decide (legacy target-only path) |
| one_hop | 24 | 191s | 18 | 1 | one response each (support/acknowledge/coordinate mix) |
| recursive | 138 | 1175s | 132 | 4 | multi-actor interaction webs (reply/coordinate/support toward several counterparts), terminated by depth+event budgets |

All actor distributions labeled `unvalidated` (no fitted calibrator — correct). Cost scales
with cascade regime and is hard-bounded by the declared budgets. Architectural evidence only —
the synthetic scenario has no real-world resolution and supports no accuracy claim.
