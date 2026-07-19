"""Strict typed MechanismSpec — §22. ONE contract JOINED over the five parallel registries.

The mechanism audit (auditC) found five mechanism registries with three incompatible status
vocabularies, advisory-only I/O declarations, and empirical evidence (Phase-6 packs, held-out
validations, transport limits) never joined to the operator that executes — so a spec consumer
could not distinguish a Higgs-fitted contagion from a broad-prior network_diffusion by looking
at the execution layer. This module is the audit's recommended shortest path: a strict typed
``MechanismSpec`` built by JOIN, not migration. Nothing existing is touched — the lean
``mechanisms.MechanismEntry`` vocabulary, the ``transitions._OPERATORS`` execution registry, the
event-type registry and the Phase-6 heavyweight registry all keep their exact behavior; this
module reads all of them and emits one spec per mechanism with:

  * a single extensible ``mechanism_kind`` vocabulary (NOT a claim of universal coverage — new
    kinds register through ``register_mechanism_kind``, never by stringly-typed drift);
  * a single unified ``calibration_status`` vocabulary with explicit mapping tables from each of
    the three legacy vocabularies (transitions' overloaded ``validated`` flag, the lean 4-value
    enum plus its observed ``deterministic`` bypass, the Phase-6 9-status lifecycle) — so
    "contract-tested" can never again masquerade as "empirically calibrated";
  * declared ``read_set``/``write_set``/``event_inputs``/``event_outputs`` gathered from the
    three scattered declaration sites plus a static backfill table for the event inputs that
    today live only inside each operator's ``applicable()`` code;
  * ``declared_write_violations`` — the pure checking function that turns today's advisory
    write-set strings into a checkable contract (enforcement wiring lives elsewhere, behind a
    flag; this function never mutates anything).

§25 external simulators: ``ExternalSimulatorAdapter`` is the ONLY door for outside numerical
simulators. Adapters are code-reviewed classes with declared inputs/outputs/units/time
semantics/seed behavior/version/data cutoff — NEVER arbitrary generated code: the registry
refuses callables whose ``__module__`` starts with ``'<'`` (dynamically generated), and refuses
adapters missing ``version`` or ``data_cutoff`` (an unversioned simulator with an unknown
training cutoff is a leakage and reproducibility hole, not a mechanism).
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------- kind vocabulary (§22)
#: The mechanism type system. EXTENSIBLE (register_mechanism_kind) — this tuple is the curated
#: core, not a claim that fifteen kinds cover the world. `qualitative_actor` marks actor-
#: cognition mechanisms (decision/belief/memory policies) so the spec layer can see them without
#: pretending they are numerical world mechanisms; Phase-4 policy machinery remains their owner.
MECHANISM_KINDS = ("qualitative_actor", "institution", "population", "network", "market",
                   "algorithm", "queue", "resource", "operational", "physical",
                   "legal_procedural", "numerical", "exogenous", "measurement",
                   "external_simulator_adapter")

_EXTRA_KINDS: set = set()


def register_mechanism_kind(kind: str, *, rationale: str) -> str:
    """The extension door for the kind vocabulary. A new kind must arrive with a rationale —
    silent vocabulary drift is exactly the fragmentation this module exists to end."""
    if not kind or not kind.replace("_", "").isalnum():
        raise ValueError(f"bad mechanism kind {kind!r}")
    if not rationale.strip():
        raise ValueError("a new mechanism kind requires a rationale (provenance)")
    _EXTRA_KINDS.add(kind)
    return kind


def known_kinds() -> tuple:
    return MECHANISM_KINDS + tuple(sorted(_EXTRA_KINDS))


# ------------------------------------------------------------- unified calibration vocabulary
#: ONE status vocabulary for "how much should you trust this mechanism's parameters", replacing
#: the three parallel ones the audit found. Ordering is informational, not a lattice claim:
#:   fitted_validated      fitted on data AND passed a real held-out/posterior-predictive check
#:   domain_validated      validated/parameterized but valid ONLY in declared domains
#:   transfer_validated    retained value under >=1 meaningful transfer
#:   documented_prior      parameters are documented/published priors (labeled, not fitted)
#:   grounded_scenario     parameters supplied grounded from THIS scenario's data at run time
#:   experimental_visible  experimental/uncalibrated — runnable only where loudly opted in
#:   unresolved            quarantined/rejected/unknown — never silently trusted
CALIBRATION_STATUSES = ("fitted_validated", "domain_validated", "transfer_validated",
                        "documented_prior", "grounded_scenario", "experimental_visible",
                        "unresolved")

#: legacy vocabulary 1 — mechanisms.MechanismEntry.calibration_status (4 declared values plus
#: the `deterministic` bypass scheduled_facts injects without register_mechanism; the audit
#: pinned that the enum is unenforced, so the map covers the observed off-enum value too).
LEAN_STATUS_MAP = {"calibrated": "fitted_validated",
                   "prior": "documented_prior",
                   "uncalibrated": "experimental_visible",
                   "experimental": "experimental_visible",
                   "deterministic": "grounded_scenario"}

#: legacy vocabulary 2 — the Phase-6 registry's 9-status lifecycle (registry/record.STATUSES).
PHASE6_STATUS_MAP = {"proposed": "experimental_visible",
                     "research_encoded": "documented_prior",
                     "implemented": "experimental_visible",
                     "locally_validated": "fitted_validated",
                     "transfer_validated": "transfer_validated",
                     "production_eligible": "fitted_validated",
                     "domain_restricted": "domain_validated",
                     "quarantined": "unresolved",
                     "rejected": "unresolved"}

#: legacy vocabulary 3 — transitions.register_operator's validated/experimental booleans. The
#: audit's finding: ALL 32 registrations carry validated=True, meaning only "contract-tested".
#: The unified map makes that overload explicit: contract-tested maps to `documented_prior` (a
#: conservative floor — the parameters are at best documented at registration), never to
#: `fitted_validated`; genuinely fitted mechanisms earn their status from the Phase-6 join.
TRANSITIONS_FLAG_MAP = {"validated": "documented_prior",
                        "experimental": "experimental_visible",
                        "unvalidated": "experimental_visible"}

_VOCABULARIES = {"lean": LEAN_STATUS_MAP, "phase6": PHASE6_STATUS_MAP,
                 "transitions": TRANSITIONS_FLAG_MAP}


def unify_calibration_status(status: str, *, vocabulary: str) -> str:
    """Map one legacy status into the unified vocabulary. Unknown statuses map to `unresolved`
    — an unrecognized trust claim is never rounded UP."""
    table = _VOCABULARIES.get(vocabulary)
    if table is None:
        raise KeyError(f"unknown legacy vocabulary {vocabulary!r} (known: {sorted(_VOCABULARIES)})")
    return table.get(status, "unresolved")


def status_from_flags(*, validated: bool, experimental: bool) -> str:
    """The transitions-registry booleans as a unified status (the WEAKEST signal — used only
    when neither a lean entry nor a Phase-6 record covers the mechanism)."""
    if experimental:
        return TRANSITIONS_FLAG_MAP["experimental"]
    return TRANSITIONS_FLAG_MAP["validated" if validated else "unvalidated"]


# ------------------------------------------------------------------- kind mapping tables (§22)
#: lean mechanisms.MechanismEntry.ontology_type -> mechanism_kind
ONTOLOGY_KIND_MAP = {"decision": "qualitative_actor", "belief": "qualitative_actor",
                     "relationship": "network", "resource": "resource", "diffusion": "network",
                     "institutional": "institution", "numerical": "numerical",
                     "exogenous": "exogenous", "measurement": "measurement"}

#: Phase-6 registry/record.ONTOLOGY_TYPES -> mechanism_kind
PHASE6_ONTOLOGY_KIND_MAP = {"observation": "measurement", "attention": "qualitative_actor",
                            "memory": "qualitative_actor", "interpretation": "qualitative_actor",
                            "decision": "qualitative_actor", "learning": "qualitative_actor",
                            "belief": "qualitative_actor", "relationship": "network",
                            "norm": "institution", "bargaining": "qualitative_actor",
                            "coalition": "network", "participation": "population",
                            "diffusion": "network", "influence": "network", "network": "network",
                            "platform": "algorithm", "resource": "resource",
                            "institutional": "institution", "measurement": "measurement",
                            "exogenous": "exogenous"}

#: scenario_schema.MECHANISM_EXECUTOR_BINDINGS -> mechanism_kind (for per-scenario mechanisms)
EXECUTOR_BINDING_KINDS = {"generic_state_machine": "operational",
                          "institutional_aggregation": "institution",
                          "conserved_resource_settlement": "resource",
                          "information_transport": "network",
                          "event_scheduling": "queue",
                          "population_response": "population"}

#: curated kind per registered operator (the audit's per-operator analysis; ports append theirs)
OPERATOR_KINDS = {
    "agent_decision": "qualitative_actor", "fitted_decision": "qualitative_actor",
    "production_actor_policy": "qualitative_actor", "belief_update": "qualitative_actor",
    "relationship_update": "network", "resource_update": "resource",
    "institutional_vote": "institution", "background_dynamics": "exogenous",
    "poisson_arrival": "exogenous", "scheduled_fact": "exogenous",
    "stance_review": "qualitative_actor", "persistence_check": "operational",
    "generic_outcome_prior": "numerical", "institutional_decision": "institution",
    "population_aggregation": "population", "aggregate_outcome_mechanism": "numerical",
    "structural_process_prior": "numerical", "actor_action_aggregation": "institution",
    "network_diffusion": "network", "communication_delivery": "network",
    "absorption_monitor": "measurement", "hazard_round": "exogenous",
    "first_passage": "numerical", "nonlinear_state_step": "numerical",
    "nonlinear_mechanism": "numerical", "nonlinear_contagion": "network",
    "feature_hazard": "numerical", "behavioral_mechanism": "population",
    "institution_action": "institution", "persistence_update": "qualitative_actor",
    "memory_consolidation": "qualitative_actor", "evidence_observation": "measurement",
}

# --------------------------------------------------- static event-I/O backfill (audit step 3)
#: event_inputs per operator, backfilled from each operator's applicable() analysis (the audit
#: found these hardcoded in code with no declaration anywhere). Notation: an event-type name;
#: "payload:<key>" for operators gated on a payload key rather than an etype; "*" for operators
#: applicable on any event (pure state monitors). ZERO behavior change — pure metadata.
OPERATOR_EVENT_INPUTS = {
    "agent_decision": ("decision_opportunity",),
    "fitted_decision": ("decision_opportunity",),
    "production_actor_policy": ("decision_opportunity", "actor_reaction"),
    "belief_update": ("exposure",),
    "relationship_update": ("relationship_effect", "payload:relationship_shift"),
    "resource_update": ("payload:resource_delta",),
    "institutional_vote": ("collective_vote",),
    "background_dynamics": ("background_tick",),
    "poisson_arrival": ("external_shock",),
    "scheduled_fact": ("scheduled_fact",),
    "stance_review": ("stance_relevant_change", "stance_review"),
    "persistence_check": ("persistence_check",),
    "generic_outcome_prior": ("resolve_outcome",),
    "institutional_decision": ("institutional_decision",),
    "population_aggregation": ("population_aggregation",),
    "aggregate_outcome_mechanism": ("aggregate_outcome_resolution",),
    "structural_process_prior": ("structural_process_prior",),
    "actor_action_aggregation": ("actor_action_aggregation",),
    "network_diffusion": ("network_diffusion",),
    "communication_delivery": ("message_delivered",),
    "absorption_monitor": ("*",),
    "hazard_round": ("hazard_round",),
    "first_passage": ("first_passage",),
    "nonlinear_state_step": ("state_step",),
    "nonlinear_mechanism": ("nonlinear_transition",),
    "nonlinear_contagion": ("contagion_exposure",),
    "feature_hazard": ("outcome_hazard",),
    "behavioral_mechanism": ("behavioral_mechanism", "resolve_outcome"),
    "institution_action": ("institutional_action",),
    "persistence_update": ("actor_action", "policy_feedback", "actor_reaction",
                           "external_shock", "public_outcome", "collective_vote"),
    "memory_consolidation": ("background_tick",),
    "evidence_observation": ("observe_evidence",),
}

#: event_outputs per operator — the follow-up event types each operator is known to emit
#: (from the declared follow-up tables in code; most operators emit none).
OPERATOR_EVENT_OUTPUTS = {
    "production_actor_policy": ("actor_action", "message_delivered", "institution_submission",
                                "delayed_action_effect"),
    "first_passage": ("persistence_check",),
    "hazard_round": ("persistence_check",),
    "persistence_check": ("first_passage",),
    "institution_action": ("institutional_action",),
    "nonlinear_mechanism": ("nonlinear_transition",),
    "nonlinear_state_step": ("state_step",),
    "communication_delivery": ("ctrl_attention",),
}


# ---------------------------------------------------------------------- the spec itself (§22)
@dataclass
class MechanismSpec:
    """The strict typed contract for ONE mechanism: identity, kind, causal role, declared state
    I/O, declared event I/O, temporal behavior, parameter schema + provenance vocabulary, units,
    conservation and validation rules, unified calibration status, domains, known limits, and
    the executing operator. Built by ``build_spec_index`` as a JOIN over the existing
    registries; hand-authored only for ported kernels (kernel_ports)."""
    mechanism_id: str
    version: str = "0.0.0-unversioned"          # audit gap: lean entries carry no version
    mechanism_kind: str = "numerical"           # known_kinds()
    causal_role: str = ""
    required_state: tuple = ()
    read_set: tuple = ()                        # declared state paths read (join of 3 sites)
    write_set: tuple = ()                       # declared state paths written (checkable)
    event_inputs: tuple = ()                    # event types that trigger the operator
    event_outputs: tuple = ()                   # follow-up event types the operator may emit
    temporal_behavior: dict = field(default_factory=dict)   # {"scale": ..., "semantics": ...}
    parameter_schema: tuple = ()                # ({name, description, lo, hi, source}, ...)
    parameter_sources: tuple = ()               # Phase-6 PARAMETER_SOURCES values (or labeled
    #                                             free text when no controlled record exists)
    units: dict = field(default_factory=dict)   # {written quantity/path: units}
    conservation_rules: tuple = ()              # declared invariant conservation statements
    validation_rules: tuple = ()                # executable/declared invariants (operator
    #                                             registration `invariants` + port rules)
    calibration_status: str = "unresolved"      # CALIBRATION_STATUSES (unified vocabulary)
    domains: tuple = ("*",)
    known_limits: tuple = ()                    # transport limits, failure modes, citations'
    #                                             limits — what this spec does NOT license
    operator: str = ""                          # transitions-registry operator name (string)

    def __post_init__(self):
        if self.mechanism_kind not in known_kinds():
            raise ValueError(f"{self.mechanism_id}: unknown mechanism_kind "
                             f"{self.mechanism_kind!r} (register_mechanism_kind first — the "
                             f"vocabulary is extensible, never implicit)")
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise ValueError(f"{self.mechanism_id}: calibration_status "
                             f"{self.calibration_status!r} is not in the unified vocabulary "
                             f"{CALIBRATION_STATUSES} — map legacy statuses through "
                             f"unify_calibration_status")

    def as_dict(self) -> dict:
        return asdict(self)


#: hand-authored specs for ported kernels — richest source, overlaid last in the join
_PORTED_SPECS: dict = {}


def register_ported_spec(spec: MechanismSpec) -> str:
    _PORTED_SPECS[spec.mechanism_id] = spec
    return spec.mechanism_id


# ---------------------------------------------------------------------------- the JOIN (§22)
#: the operator-registering modules (audit's live inventory). Imported best-effort so the index
#: reflects the full execution registry; a module that fails to import is recorded, not fatal.
_OPERATOR_MODULES = (
    "swm.world_model_v2.transitions",           # foundational 8 (+ phase4 self-registration)
    "swm.world_model_v2.scheduled_facts",
    "swm.world_model_v2.world_dynamics",
    "swm.world_model_v2.event_time",
    "swm.world_model_v2.phase_consumers",
    "swm.world_model_v2.fallback",
    "swm.world_model_v2.nonlinear.operators",
    "swm.world_model_v2.registry.families.hazard",
    "swm.world_model_v2.registry.families.behavioral",
    "swm.world_model_v2.institutions_v2.operators",
    "swm.world_model_v2.phase8_transitions",
    "swm.world_model_v2.semantic_consequences",
    "swm.world_model_v2.evidence_materialize",
    "swm.world_model_v2.kernel_ports",          # §23/§24 ported legacy kernels
)


def _import_operator_modules() -> list:
    failures = []
    for mod in _OPERATOR_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 — an optional module must not kill the index
            failures.append({"module": mod, "error": f"{type(e).__name__}: {e}"[:160]})
    return failures


def _load_phase6_records() -> dict:
    """Read the Phase-6 registry RAW (registry.json + packs.json) — deliberately NOT through
    ``RegistryStore.load()``, whose lean-mirror side effect mutates the compiler vocabulary.
    Building a spec index must never change what the compiler may instantiate."""
    try:
        from swm.world_model_v2.registry.store import PACKS_FILE, REGISTRY_FILE, _read_checked
        if not REGISTRY_FILE.exists():
            return {}
        records = _read_checked(REGISTRY_FILE)
        packs = _read_checked(PACKS_FILE) if PACKS_FILE.exists() else {}
        for fid, rec in records.items():
            rec["packs"] = packs.get(fid, [])
        return records
    except Exception:  # noqa: BLE001 — a corrupted registry file degrades the join, loudly
        return {}


def _operator_code_key(obj) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _phase6_join_fields(rec: dict) -> dict:
    """The evidence fields one Phase-6 record contributes to a spec."""
    packs = rec.get("packs") or []
    sources = sorted({str(v.get("source", "unsupported"))
                      for p in packs for v in (p.get("values") or {}).values()})
    limits = ([str(m) for m in rec.get("known_failure_modes") or []] +
              [c.get("limits", "") for c in rec.get("citations") or [] if c.get("limits")] +
              [str(x) for x in (rec.get("applicability") or {}).get("exclusion_conditions") or []])
    schema = tuple({"name": p.get("name", ""), "description": p.get("description", ""),
                    "lo": p.get("lo"), "hi": p.get("hi"),
                    "source": p.get("default_source", "unsupported")}
                   for p in rec.get("parameters") or [])
    return {"version": str(rec.get("version") or "0.0.0-unversioned"),
            "parameter_schema": schema,
            "parameter_sources": tuple(sources),
            "known_limits": tuple(x for x in limits if x),
            "domains": tuple((rec.get("applicability") or {}).get("domains") or ("*",)),
            "calibration_status": unify_calibration_status(str(rec.get("status", "")),
                                                           vocabulary="phase6"),
            "phase6_status": str(rec.get("status", "")),
            "phase6_ontology": str(rec.get("ontology_type", ""))}


def build_spec_index(*, import_modules: bool = True) -> dict:
    """dict[mechanism_id, MechanismSpec]: introspect ``transitions._OPERATORS`` (every
    registered operator gets a spec), join the lean ``mechanisms`` registry (mech_id, causal
    role, required_state, calibration status, domains), the event-type registry (reads/deltas
    for the declared event inputs), the static event-I/O backfill tables, and the Phase-6
    registry's empirical evidence (matched by family_id OR by the operator's code reference) —
    so fitted and broad-prior mechanisms are DISTINGUISHABLE at the execution layer. Hand-
    authored ported specs overlay last. Pure read: no registry is mutated."""
    if import_modules:
        _import_operator_modules()
    from swm.world_model_v2 import mechanisms as lean
    from swm.world_model_v2 import transitions
    from swm.world_model_v2.events import _EVENT_TYPES

    phase6 = _load_phase6_records()
    code_key_to_family = {}
    for fid, rec in phase6.items():
        if rec.get("code_ref"):
            code_key_to_family[str(rec["code_ref"])] = fid

    index: dict = {}

    def _event_io(op_name: str) -> tuple:
        inputs = tuple(OPERATOR_EVENT_INPUTS.get(op_name, ()))
        outputs = tuple(OPERATOR_EVENT_OUTPUTS.get(op_name, ()))
        extra_reads, extra_writes = [], []
        for et in inputs:
            meta = _EVENT_TYPES.get(et)
            if meta:
                extra_reads.extend(meta.get("reads") or ())
                extra_writes.extend(meta.get("deltas") or ())
        return inputs, outputs, tuple(extra_reads), tuple(extra_writes)

    def _dedupe(*seqs) -> tuple:
        out, seen = [], set()
        for seq in seqs:
            for x in seq or ():
                if x not in seen:
                    seen.add(x)
                    out.append(x)
        return tuple(out)

    def _build(mech_id: str, op_name: str, entry) -> MechanismSpec:
        meta = transitions._OPERATORS.get(op_name, {})
        inputs, outputs, ev_reads, ev_writes = _event_io(op_name)
        kind = OPERATOR_KINDS.get(op_name)
        if kind is None and entry is not None:
            kind = ONTOLOGY_KIND_MAP.get(entry.ontology_type)
        joined = {}
        fid = mech_id if mech_id in phase6 else code_key_to_family.get(
            _operator_code_key(meta["operator"]) if meta else "", "")
        if fid and fid in phase6:
            joined = _phase6_join_fields(phase6[fid])
            if kind is None:
                kind = PHASE6_ONTOLOGY_KIND_MAP.get(joined.get("phase6_ontology", ""))
        # calibration precedence: Phase-6 EVIDENCE statuses > lean declared status > Phase-6
        # implementation-only statuses > transitions flags. `proposed`/`implemented` say nothing
        # about parameters (code + tests only), so they never override a lean declaration.
        if joined.get("phase6_status") in ("research_encoded", "locally_validated",
                                           "transfer_validated", "production_eligible",
                                           "domain_restricted", "quarantined", "rejected"):
            cal = joined["calibration_status"]
        elif entry is not None:
            cal = unify_calibration_status(entry.calibration_status, vocabulary="lean")
        elif joined.get("calibration_status"):
            cal = joined["calibration_status"]
        elif meta:
            cal = status_from_flags(validated=bool(meta.get("validated")),
                                    experimental=bool(meta.get("experimental")))
        else:
            cal = "unresolved"
        p_sources = joined.get("parameter_sources") or ()
        if not p_sources:
            free = (entry.parameter_source if entry is not None else "") or \
                (meta.get("parameter_source", "") if meta else "")
            p_sources = (f"free_text:{free}",) if free else ()
        return MechanismSpec(
            mechanism_id=mech_id,
            version=joined.get("version", "0.0.0-unversioned"),
            mechanism_kind=kind or "numerical",
            causal_role=(entry.causal_role if entry is not None else ""),
            required_state=tuple(entry.required_state) if entry is not None else (),
            read_set=_dedupe(entry.required_state if entry is not None else (),
                             meta.get("requires", ()) if meta else (), ev_reads),
            write_set=_dedupe(meta.get("modifies", ()) if meta else (), ev_writes),
            event_inputs=inputs, event_outputs=outputs,
            temporal_behavior={"scale": (meta.get("temporal_scale", "") if meta else "") or
                               (entry.temporal_scale if entry is not None else "")},
            parameter_schema=joined.get("parameter_schema", ()),
            parameter_sources=p_sources,
            conservation_rules=(),
            validation_rules=tuple(meta.get("invariants", ())) if meta else (),
            calibration_status=cal,
            domains=_dedupe(entry.domains if entry is not None else ("*",),
                            joined.get("domains", ())),
            known_limits=joined.get("known_limits", ()),
            operator=op_name)

    claimed_ops = set()
    for mech_id, entry in lean.known_mechanisms().items():
        op_name = entry.operator or ""
        if op_name:
            claimed_ops.add(op_name)
        index[mech_id] = _build(mech_id, op_name, entry)
    for op_name in transitions._OPERATORS:
        if op_name in claimed_ops:
            continue
        row = index.get(op_name)
        if row is not None and row.operator == op_name:
            continue
        # name collision with an operator-less lean mechanism id: the OPERATOR still needs its
        # own spec (import-order independent — suites importing extra registering modules must
        # not leave any registered operator uncovered)
        key = op_name if row is None else f"op:{op_name}"
        index[key] = _build(op_name, op_name, None)
    for mech_id, spec in _PORTED_SPECS.items():
        index[mech_id] = spec                       # hand-authored: richest, overlays the join
    return index


# ------------------------------------------------------- write-set checking (audit step 7)
_CONTAINER_PREFIXES = ("quantities", "objects", "edge(", "network", "institutions",
                       "information", "populations", "event_queue")


def _pattern_matches(pattern: str, path: str) -> bool:
    """Path-prefix match between one declared write pattern and one StateDelta change path.
    Handles the repo's real declaration dialects: trailing `*` wildcards; `entity.<field>` /
    `entities` (delta paths spell the concrete entity id, so the first segment is a wildcard);
    `network.edges` (delta paths spell `edge(src,rel,dst).dim`); bare container names."""
    pattern = pattern.strip()
    if not pattern:
        return False
    if pattern.endswith("*"):
        pattern = pattern[:-1].rstrip(".")
    if pattern in ("entity", "entities") or pattern.startswith("entity."):
        tail = pattern[len("entity."):] if pattern.startswith("entity.") else ""
        if path.startswith(_CONTAINER_PREFIXES) or "." not in path:
            return False
        return path.split(".", 1)[1].startswith(tail)
    if pattern in ("network.edges", "network"):
        return path.startswith("edge(") or path.startswith("network")
    if pattern in ("quantities", "objects", "institutions", "information", "populations"):
        return path.startswith(pattern)
    return path.startswith(pattern)


def declared_write_violations(spec: MechanismSpec, state_delta) -> list:
    """PURE checking function (enforcement wiring lives elsewhere, behind a flag): compare a
    ``StateDelta``'s actual change paths against the spec's declared ``write_set`` patterns.
    Returns one violation record per out-of-set change. An EMPTY write_set means the mechanism
    declared nothing — advisory legacy mode, no violations (the gap is visible in the spec
    itself, not punished retroactively). Never mutates the delta or the spec."""
    if not spec.write_set:
        return []
    violations = []
    for ch in getattr(state_delta, "changes", None) or []:
        path = str(ch.get("path", "")) if isinstance(ch, dict) else str(ch)
        if not any(_pattern_matches(p, path) for p in spec.write_set):
            violations.append({"mechanism_id": spec.mechanism_id, "operator": spec.operator,
                               "path": path, "declared_write_set": list(spec.write_set)})
    return violations


# ------------------------------------------------------------- external simulators (§25)
def is_dynamically_generated(fn) -> bool:
    """True when a callable (or its class) was minted at run time — ``__module__`` starting
    with ``'<'`` (e.g. ``<string>``, ``<generated>``) or absent. Such callables are refused:
    adapters are code-reviewed classes living in real modules, never generated code."""
    if fn is None:
        return False
    target = fn if isinstance(fn, type) or callable(fn) else type(fn)
    mod = getattr(target, "__module__", None)
    if not isinstance(mod, str) or not mod:
        return True
    if mod.startswith("<"):
        return True
    cls_mod = getattr(type(fn), "__module__", "") if not isinstance(fn, type) else mod
    return isinstance(cls_mod, str) and cls_mod.startswith("<")


@dataclass
class ExternalSimulatorAdapter:
    """§25 — the ONLY contract for plugging an outside numerical simulator into the world.
    Everything is declared up front: what it accepts, what it emits (schema + units), how its
    clock relates to the simulation clock, how seeding behaves, its version and data cutoff
    (leakage boundary), unified calibration status, failure behavior, declared state I/O and
    whether it is safe under counterfactual branching (shared-randomness matched worlds).
    ``simulate`` is a code-reviewed callable from a real module — ``validate_adapter`` refuses
    dynamically generated code and adapters missing version/data_cutoff."""
    adapter_id: str
    version: str = ""                            # REQUIRED — refused when empty
    data_cutoff: str = ""                        # REQUIRED — training/knowledge cutoff (leakage)
    accepted_inputs: tuple = ()                  # typed input names the adapter consumes
    output_schema: dict = field(default_factory=dict)   # {output name: type/shape description}
    units: dict = field(default_factory=dict)    # {output name: units}
    time_semantics: str = ""                     # how adapter time maps onto SimulationClock
    deterministic_seed_behavior: str = ""        # what a fixed seed guarantees
    calibration_status: str = "experimental_visible"    # CALIBRATION_STATUSES
    failure_behavior: str = "reject"             # reject | degrade_with_reason — never silent
    read_set: tuple = ()
    write_set: tuple = ()
    counterfactual_safe: bool = False            # safe under matched-branch cloning?
    simulate: object = None                      # callable(inputs: dict, *, seed) -> dict

    def as_dict(self) -> dict:
        d = asdict(self)
        d["simulate"] = _operator_code_key(self.simulate) if callable(self.simulate) else ""
        return d


_ADAPTERS: dict = {}


def validate_adapter(adapter: ExternalSimulatorAdapter) -> list:
    """Reasons this adapter is refused (empty list = admissible). Enforced, not aspirational:
    no version, no data cutoff, an off-vocabulary status, or dynamically generated code each
    refuse registration — an external simulator with unknown provenance is not a mechanism."""
    reasons = []
    if not adapter.adapter_id.strip():
        reasons.append("missing adapter_id")
    if not adapter.version.strip():
        reasons.append("missing version — an unversioned external simulator is unreproducible")
    if not adapter.data_cutoff.strip():
        reasons.append("missing data_cutoff — an unknown training cutoff is a leakage hole")
    if adapter.calibration_status not in CALIBRATION_STATUSES:
        reasons.append(f"calibration_status {adapter.calibration_status!r} not in the unified "
                       f"vocabulary")
    if adapter.failure_behavior not in ("reject", "degrade_with_reason"):
        reasons.append(f"failure_behavior {adapter.failure_behavior!r} must be 'reject' or "
                       f"'degrade_with_reason' — silent failure is not an option")
    if adapter.simulate is not None:
        if not callable(adapter.simulate):
            reasons.append("simulate is not callable")
        elif is_dynamically_generated(adapter.simulate):
            reasons.append("simulate is dynamically generated code (__module__ starts with "
                           "'<') — adapters are code-reviewed classes, never generated code")
    return reasons


def register_external_adapter(adapter: ExternalSimulatorAdapter) -> str:
    reasons = validate_adapter(adapter)
    if reasons:
        raise ValueError(f"adapter {adapter.adapter_id!r} refused: " + "; ".join(reasons))
    _ADAPTERS[adapter.adapter_id] = adapter
    return adapter.adapter_id


def get_external_adapter(adapter_id: str) -> ExternalSimulatorAdapter:
    a = _ADAPTERS.get(adapter_id)
    if a is None:
        raise KeyError(f"unknown external adapter {adapter_id!r} (known: {sorted(_ADAPTERS)})")
    return a


def known_external_adapters() -> dict:
    return dict(_ADAPTERS)
