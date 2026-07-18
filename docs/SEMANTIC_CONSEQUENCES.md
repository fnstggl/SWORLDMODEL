# Semantic World Consequences — audit, design, and status

**The defect this phase removes.** A qualitative actor now chooses realistic actions, but the
executor compressed every action into `nearest ontology label × ACTION_PATHWAY_EFFECTS ×
sampled pathway_step(~0.04)` on abstract `pathway_progress:*` bars. The production causal
audit *measured* the damage: decisions shifted trajectory distributions decisively (max CDF
gap 0.50) while the binary answer stayed frozen in both arms — the world's response to
actions, not the actors, was the binding constraint. This phase makes **structured semantic
world transitions** the default consequence path: actions create and modify typed objects,
facts, relations, communications with real content, commitments, institutional procedures,
and staged processes; other actors and mechanisms produce downstream effects; numbers change
only where the world is genuinely numerical; the answer is read from the evolved typed world.

## 1. Audit of the pre-phase consequence path (the ten questions)

1. **What changes when a `TypedAction` executes** (`phase4_execution.execute`):
   `entity.current_action`, `entity.past_actions` (append), `entity.resources[k] −= declared
   cost`, `entity.commitments` (append declared), numeric `quantities[name] += delta` and
   bounded `target.beliefs[key] += clamp(delta)` from `possible_consequences`, and
   `pathway_progress:*` / `mode_progress:*` via the scalar coupling; plus queued follow-ups.
2. **Direct structured consequences:** history append, declared resource costs, declared
   commitment records. That is the complete structured set.
3. **Arbitrary numerical deltas:** `possible_consequences` of kind `quantity_delta` (any
   named quantity) and `belief_delta` (±0.25 clamp) — compiler-proposed scalars with no
   mechanism behind them.
4. **Generic pathway increments:** every ontology action with an `ACTION_PATHWAY_EFFECTS`
   entry writes `sampled pathway_step (~0.02–0.08) × effect × capacity × principal-share`
   onto `pathway_progress:*` / `mode_progress:*` (clamped 0.05–0.95). This was the PRIMARY
   causal channel from human action to answer.
5. **Follow-up events with real consumers:** `decision_opportunity` and `actor_reaction`
   (ProductionActorPolicyOperator), `stance_review` / `persistence_check` (world_dynamics),
   `hazard_round` (event-time consumers), `collective_vote` (transitions vote operator;
   phase8 transitions), `background_tick`.
6. **Effectively inert emissions:** `message_delivered` (no operator consumed it; content
   field usually empty), `institution_submission` (emitted, but the vote operator listens for
   `collective_vote`), `delayed_action_effect` (queued; no general operator turns its prose
   payload into world change).
7. **Novel-action reduction:** `NovelActionCompiler` token-matched the phrasing to the
   nearest ontology anchor and inherited its scalar effects; no anchor ⇒ `record_action`
   only, flagged `novel_action_unmodeled` — the action happened in history and did nothing.
8. **Semantic detail lost before execution:** the decision's exact content/terms, target
   framing, timing, secrecy split (public/private components), linked actions, and the
   intended effect all survived only as provenance strings — none shaped the world.
9. **Terminal dependence on pathway progress:** hazard rounds consume `pathway_progress:*`
   as rate inputs; binary/event-time readouts threshold those bars (the audit's readout was
   `cooperative_agreement ≥ 0.5`); so the answer was a function of the scalar bars.
10. **Retained architecture (unchanged by this phase):** ActorView boundaries; persistent
    qualitative actors and branch-specific decisions; `TypedAction`; perceived/actual
    feasibility; institutional rule validation; the event queue; `StateDelta` provenance;
    branch-local worlds; matched counterfactuals; outcome contracts + terminal readout;
    mechanism registry; external calibration; fitted numeric mechanisms where the process is
    numerical.

## 2. The new architecture (module `semantic_consequences.py`)

```
qualitative decision (exact content, target, timing, observability, intent)
→ SemanticConsequenceCompiler        LLM proposes a decomposition — UNTRUSTED
→ validated CausalActionProgram      closed primitive registry; schema/authority/referential/
                                     resource/temporal checks; numeric-minting REJECTED
→ direct typed world changes         objects, facts, relations, artifacts, communications,
                                     commitments, processes, submissions — all StateDelta'd
→ follow-ups with real content       delivered messages open the RECIPIENT's decision;
                                     submissions enter REAL procedures; processes stage
→ derived summaries (optional)       pathway bars become read-only projections of typed state
→ readout from the evolved world     object-predicate contracts; first passage on predicates
```

- **`WorldObject`** — typed, provenance-bearing registry on `WorldState.objects`: closed
  `OBJECT_TYPES` (organization, product, service, feature, brand_identity, campaign,
  public_statement, private_communication, proposal, offer, contract, agreement, policy,
  regulation, legal_filing, project, operational_initiative, team, role, asset,
  market_offering, event_record, submission, obligation_record, process). Every object:
  stable id, type, attributes, status, created/updated_at, creator, source action,
  visibility, valid time, provenance.
- **Typed processes** — objects of type `process` with per-`process_type` stage machines
  (`product_launch`, `negotiation`, `acquisition`, `institutional_procedure`,
  `regulatory_review`, `adoption`, `generic`); stages advance only along declared
  transitions, with history. Progress = completed stages and satisfied prerequisites, never
  a universal scalar.
- **Primitives** — a closed registry (`create/update/terminate_world_object`,
  `set/remove_object_fact`, `create/update/remove_relation`, `publish_artifact`,
  `deliver_information`, `create_commitment`, `create_obligation`,
  `allocate/consume/transfer_resource` (conservation-checked), `start_process`,
  `advance_process_stage`, `complete_process`, `fail_process`, `submit_to_institution`,
  `open_actor_decision`, `open_population_response`, `schedule_event`,
  `record_observation`). Each has a strict schema, deterministic executor, and validators;
  none may set terminal probabilities, forecast distributions, pathway bars, utilities, or
  belief scalars — such ops are **quarantined**, loudly.
- **Communications carry real content**: `deliver_information` creates a
  `private_communication`/`public_statement` object AND an `InformationItem` with the exact
  message, schedules `message_delivered`; the new `CommunicationDeliveryOperator` exposes
  the item to recipients at delivery time and opens their decision with the message as the
  situation — the recipient's next `ActorView` contains the actual text, and their reaction
  comes from THEIR policy, never from the sender's expectation.
- **Institutions**: `submit_to_institution` creates a `submission` object, starts an
  `institutional_procedure` process, validates eligibility/deadline rules, schedules the
  procedure's real events (`collective_vote` — the type the vote operator actually
  consumes) and opens member decisions where decision rights exist; the vote writes a typed
  institutional outcome onto the submission.
- **Compiler paths**: (a) LLM decomposition for qualitative decisions — proposal-only,
  validated op-by-op; (b) deterministic ontology→primitive programs for numeric/Tier-3
  actions and as the loud fallback (`semantic_consequence_unmodeled` marks what could not be
  modeled; partial programs report exactly what executed).
- **Derived pathway summaries**: `derive_pathway_summaries(world)` recomputes
  `pathway_progress:*` as projections of typed state (open channels, live proposals,
  acceptance/signature status, launch/procedure stages) so existing hazard/readout consumers
  keep functioning — facts → summary → readout, never action → bar → answer.
- **Modes** (`SWM_CONSEQUENCES`): `semantic_world_consequences` (default),
  `legacy_scalar_pathway_consequences` (benchmark-only), `dual_run_consequence_audit`
  (semantic applied; what legacy would have written recorded, unapplied). Every run result
  carries a `consequence_report` (requested/actual mode, actions compiled, ops applied by
  primitive, events scheduled, processes started, objects created, submissions, decisions
  opened, unsupported semantics, fallbacks + reasons, legacy scalar writes). Failures are
  loud; silent scalar fallback is structurally impossible (the scalar writer asserts the
  mode).

## 3. Honest status (claims ladder)

- **Implemented & mechanically verified**: everything in §2, wired default-on through
  `operators_from_plan` (both funnels), the qualitative runtime, individual reactions, and
  matched counterfactuals; full test suite green including the new invariants (no scalar
  writes in default mode; exact-message delivery round trip; sender expectation ≠ recipient
  reaction; institutional submissions enter real procedures and votes write typed outcomes;
  novel actions compile or are loudly unmodeled; no primitive can touch terminal state).
- **Concrete world state changed**: demonstrated end-to-end (launch, negotiation,
  institutional decision, individual communication demos) and the matched-mode evaluation
  runs the causal-audit settlement scenario (typed readout: signed agreement or negotiation
  reaching provisional acceptance) under legacy scalar / semantic×{numeric, stateless,
  persistent} on identical worlds and seeds. Results:
  `experiments/results/SEMANTIC_CONSEQUENCES_REPORT.md`.
- **Not yet demonstrated**: predictive improvement. Consequence-accuracy scoring against
  frozen historical intermediate facts is measured only at pilot scale in this phase's
  evaluation artifact; fitted adoption/market mechanisms and large-scale trajectory
  validation remain open, and are the declared next measurement work.
