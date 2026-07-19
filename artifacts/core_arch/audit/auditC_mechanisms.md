# Audit C — Mechanism/Transition Architecture & Kernel Inventory

Repo `/home/user/SWORLDMODEL`, branch `claude/worldmodel-v2-core-arch-73s7eq`, 2026-07-19.
Companion data: `auditC_mechanisms.json` (same directory).

## 1. Prose map of the V2 mechanism plane

Everything in V2 executes on one shared substrate: `WorldState` (`state.py` — entities with
provenance-stamped `StateField`s, populations, one typed `RelationGraph`, `RuleSystem`
institutions, typed `Quantity` registry, information ledger, `objects` (typed world facts),
branch-local `scenario_schema`, `mechanism_instances`) + `SimulationClock{now, as_of}`
(monotonic, real calendar) + an `EventQueue` (`events.py`) that interleaves scheduled events,
`StochasticHazard` samples and endogenous follow-ups, popped in same-timestamp batches that the
temporal runtime layers into causal microsteps. Every change is a `StateDelta` (`transitions.py`)
— `{at, event_type, operator, changes[{path,before,after}], reason_codes, uncertainty,
evidence_deps, follow_up_events}` — produced by a `TransitionOperator`
(`applicable → propose → validate → apply`), with institutional `validate_action` run before any
apply. This contract is uniformly honored: I found no operator that mutates state outside a delta.

There are, however, **five parallel "mechanism registries"**, three status vocabularies and two
operator binding modes:

1. **Lean registry** (`mechanisms.py`): 11 `MechanismEntry`s the compiler may instantiate.
   `calibration_status ∈ {calibrated, prior, uncalibrated, experimental}` (unenforced free
   string). Two entries (`poll_error_aggregation`, `whipcount_binomial`) are deliberately dead —
   experimental, empty operator, compiler rejects them loudly (pinned in
   `tests/test_wmv2_tier_a_fixes.py`). `poisson_arrival` is the one ported v1 kernel
   (`RareEventArrivalOperator`).
2. **Operator registry** (`transitions.py` `_OPERATORS`): 32 live registrations across 13
   modules (foundational ops; `production_actor_policy`; `scheduled_fact`; world-dynamics
   `stance_review`/`persistence_check`; event-time `first_passage`/`absorption_monitor`/
   `hazard_round`; six `phase_consumers`; three `nonlinear`; `feature_hazard`;
   `behavioral_mechanism`; `institution_action`; phase-8 `persistence_update`/
   `memory_consolidation`; `communication_delivery`; `evidence_observation`;
   `generic_outcome_prior`). **All 32 carry `validated=True`, none `experimental`** — the flag
   means "contract-tested", not "empirically calibrated"; the operator-level experimental gate is
   dead in practice. `requires`/`modifies` are advisory strings.
3. **Phase-6 heavyweight registry** (`registry/`): 63 committed `MechanismRecord`s (45
   implemented, 3 production_eligible, 2 locally_validated, 7 domain_restricted published packs,
   5 research_encoded, 1 quarantined — `hawkes_self_excitation`), 17 `ParameterPack`s, a 9-status
   lifecycle, an 8-value `PARAMETER_SOURCES` vocabulary, citations-with-limits, preserved failed
   validations, applicability/transport rules. `store.py` mirrors production records into the
   lean registry. **This registry already has ~90% of the target MechanismSpec fields.**
4. **Scenario-generated mechanisms** (`scenario_schema.mechanism_definitions` + the
   `causal_boundary` runtime): per-question LLM-compiled Layer-B mechanisms (channels, platforms,
   institutions, physical/administrative processes) with real state machines,
   `triggering_event_types`, `possible_output_event_types`, `executor_binding ∈
   MECHANISM_EXECUTOR_BINDINGS` (generic_state_machine, institutional_aggregation,
   conserved_resource_settlement, information_transport, event_scheduling, population_response).
   Executed by ONE generic `MechanismRuntimeOperator` (declared rules → neutral bindings → single
   path → LLM adjudication labeled `model_based_unvalidated` → honest unresolved), with escrow/
   settle/refund conservation. **The only place event I/O and per-mechanism state machines are
   declared** — but per-branch and untrusted, never global.
5. **Sidecars**: nonlinear `forms` registry (evaluable shapes with `param_schema`, monotonicity,
   extrapolation metadata, maturity), `registry_ext`, and the semantic-feature registry
   (evidence-gated, 10 statuses, interpretation channel quarantined on measured harm).

Binding mode two: `materialize.py` directly instantiates ~8 operators (mechanism runtime,
scheduled attempts, generated-world semantic/observation/actor/attention/plan operators) outside
the string registry; `phase9_temporal`'s 8 typed edge transitions emit their own `Phase9Delta`.

Temporal/exogenous machinery is strong and consistent: `temporal_hazards.CumulativeHazardState`
(particle-rooted Exp(1) thresholds, preserved Λ across rate changes, generation-guarded
crossings, mass-conservation identity), `family_hazards` (fitted pack PRESENT on disk),
`scheduled_facts` (deterministic dated facts), `world_dynamics` (stance dynamics, persistence
checks, 10 documented coupling priors sampled per branch — `coupling_pack.json` NOT yet fitted),
`phase8_events.EventLog` (append-only, observed-time filtered, tamper-evident) and
`evidence_temporal.TemporalVerifier` as the external adapter.

## 2. Validated legacy kernels (outside `world_model_v2`)

22 kernels inventoried (full records in the JSON). Highlights:

- **Port (10)**: `sim_aggregation` (poll-error measurement — fills the dead
  `poll_error_aggregation` entry), `sim_whipcount` (institution — fills `whipcount_binomial`),
  `sim_contest` (Elo, numerical), `future_events.py` FutureEvent calendar + SurpriseHazard
  (+ `simulation/event_model.py` calibrated jump variance, `directional_event_model` conditional
  impacts) as the **exogenous OutsideWorldProcess residual**, `belief_dynamics.BeliefTransition`
  and EXP-035 heteroskedastic drift/variance (fitted market/belief-series numerics),
  `mean_field.MeanFieldRollout` + `population_simulator` turnout coupling (the only validated
  coupled population dynamics; phase9 has composition but no within-horizon coupling),
  `bayes_logistic` (weight posteriors for packs), `elasticity_fit` (graded pack producer),
  `demographic_values` (population trait priors), `conformal`/`calibration_grade` +
  `nonstationarity` drift (measurement/validation tooling), `direction_model` (lean prior).
- **Reject (9)**: unfitted `transition/diffusion.py` IC/LT/Hawkes (superseded by Higgs-fitted
  Phase-6/7 families; Hawkes quarantined on evidence), `engine/diffusion.py` LLM cascade
  (RC4-quarantined probability minting), `response_model`/`individual_transition` stack (already
  re-fitted natively in `reference/enron.py`), `aggregate_transition`/`aggregate_world`
  (the quarantined arbitrary-variable logistic path), HN engine (platform packs cover it),
  `agent_society` LLM personas, `world/substrate.py`, `sim_escalation`/`sim_persistence`
  (subsumed by mode-graph pathways + first passage), stubs (`graph/diffusion`,
  `inference/filter`, `transition/mechanistic` are `IMPLEMENTED=False`).

## 3. Shortest path to a typed MechanismSpec (no operator breakage)

1. New `mechanism_spec.py`: `MechanismSpec` dataclass with the 19 target fields + the 14-kind
   enum. Touch nothing existing.
2. **Join, don't migrate**: `from_registries(mech_id)` merges lean entry + operator metadata
   (`requires→read_set`, `modifies→write_set`, `temporal_scale`, `invariants→validation_rules`)
   + event-type `reads`/`deltas` + Phase-6 record/packs (`parameter_schema`, `parameter_sources`,
   `known_limits`, citations) where present.
3. Add optional `event_inputs=()` to `register_operator` and backfill the 32 registrations from
   their `applicable()` etypes (static table; zero behavior change). `event_outputs` from
   declared follow-up tables.
4. Kind mapping tables for `ontology_type`, Phase-6 `ONTOLOGY_TYPES`, and scenario
   `executor_binding` (aggregation→institution, settlement→resource, transport→network,
   scheduling→queue, population_response→population, state_machine→operational). Actor-cognition
   entries (decision/belief/relationship) stay Phase-4 policy, outside MechanismSpec.
5. Adopt Phase-6 vocabularies as the single source (`PARAMETER_SOURCES`, 9-status lifecycle with
   a lean-4→lifecycle mapping); extend `store.py`'s mirror to emit specs.
6. Units from the quantity types a mechanism writes (pre-register with units instead of at
   apply time); `conservation_rules` mirrored from `resource_definitions.conserved` + escrow
   semantics.
7. Optional flag-gated enforcement: diff `StateDelta.changes` paths against `write_set`, log
   violations — advisory strings become checked contracts incrementally.
8. `spec_from_mechanism_definition(md)` adapter for scenario-generated mechanisms
   (version = schema version + content hash; `calibration_status=experimental` unless
   `evidence_basis`).

## 4. Three biggest contract gaps

1. **Fragmented identity & status**: five registries, three incompatible status vocabularies, no
   `version` on the lean entries the compiler consumes, and the enum is bypassable
   (`scheduled_facts` injects an inline mech dict with status `"deterministic"` that never passes
   `register_mechanism`).
2. **Declared I/O is advisory and scattered**: read/write sets live in three places (operator
   `requires/modifies`, event-type `reads/deltas`, `Event.read_set/write_set`), never validated
   against actual `StateDelta.changes`; `event_inputs` are hardcoded in `applicable()`; units
   exist only on quantity types (often registered ad-hoc at apply time) and conservation only as
   code in two operators — none of units/conservation/event-I/O is declarable per mechanism.
3. **`validated=True` is overloaded**: every registered operator claims it, meaning only
   "contract-tested"; empirical calibration evidence (packs, held-out validations, transport
   limits) lives solely in the Phase-6 registry and is not joined to the operator that executes —
   so a spec consumer today cannot distinguish a Higgs-fitted contagion from a broad-prior
   network_diffusion by looking at the execution registry.
