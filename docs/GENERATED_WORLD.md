# Generated actor-mediated world — the production consequence architecture

**The principle.** Hardcode reality integrity, not social causality. The engine's fixed
machinery covers time ordering, identity, provenance, branch isolation, visibility and
information access, physical/legal feasibility, authority, resource conservation, queue
execution, recursion budgets, schema validation, institutional arithmetic, aggregation, and
calibration. It contains **no fixed semantic model** of messages, proposals, launches,
negotiations, approvals, adoptions, or reactions. Each scenario **generates** its own
semantic types, facts, relations, events, processes, institutional procedures, causal
mechanisms, and outcome predicates; when a world change affects a consequential human, the
engine delivers the actor-specific observation and **invokes that actor's persistent LLM** —
the actor decides whether anything should be done, and what, including nothing, or an action
no menu anticipated. Probabilities come afterward, by counting independently realized
trajectories.

**The causal truth boundary (default-on).** Every consequence belongs to one causal layer:
**A** actor-controlled direct effect (the actor's own records, artifacts, recorded decision,
initiated attempt, committed own resources); **B** mechanism-mediated effect (anything
needing a channel, platform, technical system, institution, administrative/legal procedure,
market, or physical process — delivery, publication, intake, registration, settlement);
**C** actor-mediated effect (another actor perceiving, interpreting, deciding, acting);
**D** the terminal readout. The direct-effect compiler may produce **only Layer A plus
explicit invocations of Layer-B mechanisms** — never C or D. The premise handed to it is
*"the actor selected an intended action and is attempting the steps under their control"*,
never *"the actor successfully performed the action"*. The full chain:

```
actor intent
→ actor-controlled attempt            (Layer A: exact content, own records, attempt events)
→ scenario-specific mechanism         (Layer B: generated per scenario, generic runtime)
→ actual world transition             (mechanism success/failure/unresolved — never assumed)
→ actual observation                  (verified recipients only; intended ≠ actual)
→ affected actor reaction             (Layer C: that actor's own simulation)
→ downstream world evolution          (recursion through the same boundary)
```

Intended visibility is **not** actual observability; an intended target is **not** an actual
recipient; an attempted action is **not** a completed action; scheduling is **not**
occurrence; submission is **not** acceptance; transmission is **not** delivery; one
signature is **not** a bilateral agreement; a request is **not** the other actor's act.
Unresolved mechanisms remain explicitly unresolved — success is never assumed.

## Two planes

**Control plane** (`generated_world.py`, invisible machinery): three queue task types —
`ctrl_semantic_event` (route observations + discover the causal frontier),
`ctrl_deliver_observation` (apply information-access rules to ONE actor's local state),
`ctrl_invoke_actor` (the actor's perceived world materially changed: rebuild their
`ActorView`, invoke their persistent qualitative LLM) — plus per-actor invocation budgets,
(actor, event) dedup, cascade-depth caps, and deterministic institutional arithmetic
(`run_institutional_aggregation` counts actual member decision records; it never chooses a
vote). Control tasks never appear in the world's semantic history.

**World semantic plane**: `WorldState.semantic_log` (scenario-typed `SemanticWorldEvent`s —
what actually happened, with the exact content) and `WorldState.objects` as **records** whose
types come from the branch's `ScenarioSemanticModel`. The world's meaning lives entirely in
scenario-generated definitions.

## The pieces

- **`scenario_schema.ScenarioSemanticModel`** — per-question generated semantics: entity/
  fact/relation/event types, process definitions, institutional definitions (holders +
  decision record type + executable aggregation arithmetic; evidence or `assumed`), physical
  constraints (executable or `unresolved`), conserved resources, information rules, actor
  roles (affordance **examples**, never required menus), frozen outcome predicates,
  unresolved mechanisms, evidence basis, assumptions, provenance. Compiled by an LLM
  (`SchemaCompiler`) as an **untrusted proposal**: deterministic validation (stable ids, no
  numeric-minting fields, no action→human-reaction coefficients, executable arithmetic,
  predicates over declared records, no outcome smuggled into initial records), mechanical
  honesty-preserving auto-repairs (label unexecutable constraints unresolved, declare
  missing predicate record types, mark unevidenced institutions assumed — all stamped),
  one LLM repair round, an adversarial critic pass, then **frozen**. Runtime extension is
  versioned, ancestry-preserving, additive-only, and **branch-local** (the schema lives on
  the branch's world; `clone()` isolates it).
- **Scenario mechanisms** (`ScenarioSemanticModel.mechanism_definitions` +
  `causal_boundary.py`) — each scenario generates its OWN Layer-B mechanisms (channels,
  platforms, institutions, administrative/physical processes): triggers (attempt event
  types), accepted inputs, controlling system, authority, preconditions, a real state
  machine (initial/intermediate/success/failure/unresolved), executable transition rules
  over branch state, typed outputs, record updates, observation rules, timing, evidence/
  assumptions, uncertainty source, and a semantically neutral `executor_binding`
  (institutional arithmetic, conserved-resource settlement, transport, scheduling). There is
  **no global catalog** of email/launch/application/meeting/payment mechanisms. Generation
  uses ACTUAL LLM calls: one proposal, one **independent causal-boundary critic** ("could
  the actor do every step under their control and the effect still fail?"), one bounded
  repair — traced and content-addressed cached. One generic `MechanismRuntimeOperator`
  executes every definition: declared rules first, neutral bindings second, single declared
  paths third, LLM adjudication of ONE concrete next transition per branch fourth (never a
  probability, labeled `model_based_unvalidated`), and honest `unresolved` last.
- **The kernel** (semantically empty storage + integrity): `declare_schema_definition`,
  `create_or_update_record`, `remove_record`, `create_or_remove_relation`,
  `emit_semantic_event`, `schedule_semantic_event`, `transfer_conserved_quantity`,
  `invoke_scenario_mechanism`. It validates instances against the **branch schema** — there
  is no `create_product` or `on_message_delivered` anywhere. The boundary is kernel-enforced
  on the direct-action plane: no writing externally-controlled or another party's records,
  no non-unilateral relations, no emitting mechanism-output event types, no scheduling
  future successes (a scheduled attempt re-enters the boundary at fire time), no direct
  transfers past a declared settlement mechanism, no outcome-satisfying writes
  (terminal smuggling; institution decision holders recording their OWN decision are the
  scenario-declared exception), no runtime causal-semantics extension without the boundary
  critic. Untrusted-op failures quarantine loudly; ops that try to write another human's
  mind/choice are rejected and counted.
- **`causal_boundary.CausalActionCompiler` + `DirectnessValidator`** — the actor's exact
  chosen action (their words, target, timing, secrecy) → a typed `DirectActionProgram`:
  actor-controlled operations, attempt events, mechanism invocations, deferred actor
  dependencies, unresolved claims, rejected claims, completion conditions, compiler + critic
  provenance. Deterministic directness tests (ownership, external-acceptance, terminal
  smuggling) convert or reject failed claims; an LLM directness critic challenges the rest
  (unilateral-control, social-agency, temporal, observability, institutional, physical) —
  failed claims become mechanism invocations, deferred actor decisions, unresolved, or
  rejected — never silently retained. Op budget is complexity-aware; overflow marks the
  program partially modeled, never silently truncated. Total fallback preserves the exact
  action as the schema-scoped `unmodeled_actor_action` scaffolding attempt, counted.
- **Observation routing** — deterministic, and only for VERIFIED observability: a
  mechanism's successful output carries `actual_recipients` (and `availability="public"`
  only after an actual publication/availability mechanism succeeded); the router delivers
  those and only those, with per-recipient channel, delay, and representation. An
  unprocessed attempt routes to nobody. Delivery updates the recipient's information ledger
  with the exact (or rule-degraded) content and schedules THEIR reconsideration.
- **Causal frontier** — per-event discovery of actors who may now matter, run ONLY on
  verified-observable events (actual recipients, institutional decision holders touched by
  the matter, network neighbors on actually-published events, optional LLM extension) with
  deterministic validation, dedup, and budgets. Nobody is invoked by an intended target
  list.
- **Action history as attempts** — `past_actions` records
  `action_attempt_initiated → mechanism_pending → mechanism_succeeded/failed/unresolved →
  action_partially_completed/action_completed` (plus `execution_incomplete`, `blocked`),
  with intended action, attempted action, actor-controlled effects, mechanisms invoked,
  mechanism results, unresolved steps, completion conditions, failure reason, provenance.
  An action is `action_completed` only when its scenario-specific completion conditions
  actually hold in the world.
- **Actor invocation** — rebuilds the view, presents the new observation as the situation,
  passes schema affordances only as examples, and lets the persistent qualitative actor
  decide: wait (first-class, counted), a listed affordance, or any novel feasible action.
  Feasibility/authority validation is unchanged (`execute()`); consequences compile through
  the generated compiler; new events recurse through the same canonical queue.
- **Modes** — `generated_actor_mediated_world` (default),
  `fixed_semantic_consequence_policy_v1` (the previous fixed-catalog baseline, explicitly
  requested ONLY; `semantic_world_consequences` is its alias),
  `legacy_scalar_pathway_consequences` (benchmark-only), `dual_run_consequence_audit`.
  **Generated mode never degrades to fixed-v1** — silent or stamped. A world with no
  scenario schema (or no mechanism model) is `structurally_under_modeled` /
  `execution_incomplete`: the exact attempt is preserved, only deterministically provable
  actor-controlled effects apply, unresolved mechanisms are marked, the support grade is
  capped, and NO old consequence path serves in its place. Both scalar writers raise under
  every non-legacy mode.
- **Reporting** — every result carries the full contract: requested/actual mode, schema id +
  version, types generated, events emitted, schema extensions, observations delivered,
  actors reconsidered/invoked/declined, actions executed, cascade depth, plus the
  causal-boundary counters: `action_attempts`, `actor_controlled_effects`,
  `mechanisms_invoked` / `mechanism_successes` / `mechanism_failures` /
  `mechanism_unresolved`, `intended_deliveries` vs `actual_deliveries`,
  `intended_publications` vs `actual_publications`, `directness_claims_rejected`,
  `deliveries_unresolved_no_mechanism`, `scheduled_attempts(_fired)`,
  `structurally_under_modeled`, and per-action `causal_action_reports` (selected/attempted
  action, exact content, proposed vs rejected direct effects, mechanism instances,
  deferred actor dependencies, unresolved claims, completion conditions + status, LLM
  calls, compiler/critic provenance). Pure-run invariants:
  `human_reactions_written_directly == 0`, `external_successes_written_directly == 0`,
  `fixed_ontology_uses == 0`, `numeric_fallbacks == 0`.

## Final audit (the required fifteen questions, answered honestly)

1. **Which fixed ontology lists remain?** `semantic_consequences.py` keeps `OBJECT_TYPES`,
   `PROCESS_STAGES`, `PRIMITIVES`, and the fixed action ontology
   (`phase4_policy.ACTION_ONTOLOGY`/`ACTION_PATHWAY_EFFECTS`) — all baseline-only. The
   kernel has `KERNEL_OPS` (storage mechanics) and three `ctrl_*` task types (scheduler
   instructions), which are integrity machinery, not scenario semantics. The event-type
   registry still exists for control/mechanistic events; scenario events bypass it by
   design (they ride the `ctrl_semantic_event` envelope).
2. **Are any load-bearing in production mode?** No. Tests 1–3/28–29 prove records/events are
   accepted solely by the branch schema, and the fixed catalog/scalar writers cannot run in
   generated mode. A schema-less world is `execution_incomplete` / structurally
   under-modeled — the attempt is preserved and **no fixed-v1 consequence is served**
   (`tests/test_causal_boundary.py::test_11`).
3. **Can a novel scenario create a new type without code changes?** Yes — demoD's supplier-
   qualification and demoC's reaction types exist nowhere in the repository; test 5 proves
   it offline.
4. **Novel semantic event without global registration?** Yes (test 6 asserts the type is
   absent from the global registry while the event flows).
5. **Does receiving an event invoke the actor rather than a reaction handler?** Yes:
   mechanism success → verified observability → delivery → `ctrl_invoke_actor` → the
   actor's own policy. The fixed `message_delivered → acknowledge/ignore` path is
   baseline-only; in generated mode the legacy mechanism emissions are disabled (schema or
   not) so a reaction has exactly one route — and nobody is invoked before an ACTUAL
   observation (`test_causal_boundary.py::test_2/test_14`).
6. **Can an actor choose an action outside every supplied menu?** Yes (test 13; the
   qualitative schema always invited novel actions; affordances are examples).
7. **Deliberate inaction?** Yes — `wait` is first-class and counted
   (`actors_declined_to_act`).
8. **Does any code directly modify another consequential actor's belief/support/compliance/
   choice?** No: the kernel exposes no mind-write operation, attempts are quarantined and
   counted, and the legacy `belief_delta` writer raises outside the legacy mode.
9. **Are internal scheduler tasks separated from world events?** Yes — `ctrl_*` tasks never
   enter `semantic_log` (tests 9–10).
10. **Are institutional outcomes aggregated from actual actor decisions?** Yes —
    `run_institutional_aggregation` counts member decision records; an empty tally decides
    nothing (tests 24–25).
11. **Can the schema evolve during rollout?** Yes — `declare_schema_definition`: versioned,
    additive-only, ancestry-preserving, branch-isolated (tests 7–8).
12. **Does the public default path use the generated architecture?** Yes —
    `resolve_consequence_mode()` defaults to it across `run_from_plan`, `simulate_world`,
    phase-8 persistence (which now binds the scenario schema + mechanisms too), Phase-13
    matched evaluation, and individual reactions. The schema-less case is
    `execution_incomplete`, never a fixed-v1 swap
    (`experiments/causal_boundary_smoke.py` proves the default route end to end).
13. **Do Phase-13 rollouts use it?** Yes — matched counterfactuals clone worlds carrying the
    schema and run the same operators/engine with paired seeds (test 30–31).
14. **Can fixed-v1 still run as a baseline?** Yes — explicit mode, full artifact set kept
    (test 34).
15. **What remains mechanically verified vs unvalidated?** Mechanically verified: everything
    above (1336 tests, incl. the 34 invariants). Demonstrated end-to-end with a real
    backend: the five demos + the matched evaluation
    (`experiments/results/GENERATED_WORLD_REPORT.md`). **Not demonstrated**: predictive
    improvement over the baselines — consequence/trajectory accuracy against frozen
    historical intermediate facts remains the declared next measurement; population
    mechanisms remain opened-not-resolved without fitted models; non-human mechanisms
    without validated implementations stay labeled `unresolved` rather than simulated.

**Definitive acceptance statement.** The engine contains fixed machinery only for integrity,
scheduling, access, feasibility, and aggregation. Each scenario generates its own semantic
types, facts, events, processes, and outcome predicates. A world change that affects a
consequential human delivers an actor-specific observation and invokes that actor's
persistent LLM; the actor decides whether and how to respond; its action creates new
scenario-defined semantic changes that may affect other actors recursively. No predefined
social reaction handler, fixed action menu, hardcoded social coefficient, or global scenario
ontology substitutes for those decisions.
