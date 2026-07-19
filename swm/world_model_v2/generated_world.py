"""Generated actor-mediated world — the production consequence architecture.

Two planes, explicitly separated:

CONTROL PLANE (fixed, invisible machinery — never the meaning of what happened): three
registered queue task types — ``ctrl_semantic_event`` (route a world event: observation
delivery + causal-frontier discovery), ``ctrl_deliver_observation`` (apply information-access
rules and update ONE actor's local information state), ``ctrl_invoke_actor`` (an actor's
perceived world materially changed: rebuild their view and invoke their persistent LLM) —
plus budgets, dedup, provenance, and deterministic institutional arithmetic.

WORLD SEMANTIC PLANE (generated per scenario): records and ``SemanticWorldEvent``s whose
types come from the branch's ``ScenarioSemanticModel``. There is NO global catalog of object
types, event names, process stages, or reaction menus here — the kernel below is semantically
empty storage + integrity, and every meaning is scenario-generated.

The social causal mechanism is the actor loop, not a handler: a semantic event reaches an
actor → their local information changes → the control plane schedules reconsideration → THEIR
persistent LLM interprets it and decides whether anything should be done and what — including
waiting, or an action no menu anticipated. No code path writes another consequential human's
belief, support, compliance, or choice; violations are quarantined and counted
(``human_reactions_written_directly`` must be 0 in a pure run).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field

from swm.world_model_v2.events import Event, event_type_registered, register_event_type
from swm.world_model_v2.information import InformationItem
from swm.world_model_v2.scenario_schema import (
    UNMODELED_EVENT_TYPE, ScenarioSemanticModel, evaluate_predicate, extend_schema,
)
from swm.world_model_v2.semantic_consequences import WorldObject, _FORBIDDEN_KEYS
from swm.world_model_v2.state import F
from swm.world_model_v2.transitions import StateDelta, ValidationResult

#: control-plane task types — scheduler instructions, never world semantics
CONTROL_EVENT_TYPES = ("ctrl_semantic_event", "ctrl_deliver_observation", "ctrl_invoke_actor",
                       "ctrl_mechanism_step", "ctrl_scheduled_attempt", "ctrl_attention")
for _et in CONTROL_EVENT_TYPES:
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", validated=True,
                            parameter_source="generated world control plane")

#: kernel operation names — storage/integrity mechanics only, semantically empty.
#: `invoke_scenario_mechanism` is the ONLY door from a direct action to any external effect
#: (delivery, publication, intake, settlement, physical completion) — the causal boundary.
KERNEL_OPS = ("declare_schema_definition", "create_or_update_record", "remove_record",
              "create_or_remove_relation", "emit_semantic_event", "schedule_semantic_event",
              "transfer_conserved_quantity", "invoke_scenario_mechanism")

#: SAFETY budgets — service protection, NOT models of reality (§12/§26). The real stopping
#: conditions are causal quiescence / horizon / actor-chosen inaction; these ceilings sit far
#: above natural cascade sizes, and REACHING one marks the branch temporally_truncated with
#: the pending actors/events recorded — never a silent omission or a fake quiescence.
DEFAULT_BUDGETS = {"max_invocations_per_actor": 40, "max_semantic_events": 1000,
                   "max_cascade_depth": 64}


def _hash(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:14]


def _sanitize(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw)).strip("_")[:80]


def generated_report() -> dict:
    """The §15 result contract — every generated-mode run carries these counters, plus the
    causal-boundary counters (attempts vs deliveries, mechanism outcomes, rejected directness
    claims). Pure-run invariants: human_reactions_written_directly == 0,
    external_successes_written_directly == 0, fixed_ontology_uses == 0, numeric_fallbacks == 0."""
    from swm.world_model_v2.causal_boundary import causal_boundary_report_fields
    return {"scenario_schema_id": "", "scenario_schema_version": "",
            "scenario_types_generated": 0, "scenario_events_emitted": 0,
            "schema_extensions": 0, "observations_delivered": 0, "actors_reconsidered": 0,
            "actors_invoked": 0, "actor_actions_executed": 0, "actors_declined_to_act": 0,
            "recursive_cascade_depth": 0, "human_reactions_written_directly": 0,
            "fixed_ontology_uses": 0, "unsupported_semantics": 0, "mechanistic_fallbacks": 0,
            "numeric_fallbacks": 0, "fallback_reasons": [],
            "attention_events": 0, "observation_bundles_delivered": 0,
            "temporal_truncations": [],
            **causal_boundary_report_fields()}


# ---------------------------------------------------------------- world-plane event envelope
@dataclass
class SemanticWorldEvent:
    """What actually happened, in the scenario's OWN terms. `semantic_type_id` references the
    branch schema — never a global registry.

    CAUSAL TRUTH BOUNDARY: `direct_targets` and `intended_visibility` are the actor's INTENT;
    `actual_recipients`/`availability` are set ONLY when `observability_verified` is True —
    i.e. when a scenario mechanism actually delivered/published the content (causal_layer
    `mechanism_output`) or the schema declares the act itself directly perceivable
    (`unmediated`). An unverified actor-controlled attempt is observable to nobody but its
    source."""

    event_id: str = ""
    semantic_type_id: str = ""
    schema_id: str = ""
    schema_version: str = ""
    source_actor_id: str = ""
    direct_targets: list = field(default_factory=list)          # INTENDED targets
    exact_content: str = ""
    structured_fields: dict = field(default_factory=dict)
    occurred_at: float = 0.0
    intended_visibility: str = "participants"      # public | participants | private (INTENT)
    actual_observability: dict = field(default_factory=dict)   # recipient -> representation
    causal_layer: str = "actor_controlled"         # actor_controlled | mechanism_output
    observability_verified: bool = False           # True only via mechanism or unmediated act
    actual_recipients: list = field(default_factory=list)      # verified recipients ONLY
    availability: str = ""                         # "public" ONLY after a publication/
    #                                                availability mechanism succeeded
    representation: str = "complete"
    mechanism_id: str = ""                         # producing mechanism, when mechanism_output
    linked_action_id: str = ""
    parent_event_ids: list = field(default_factory=list)
    branch_id: str = ""
    cascade_depth: int = 0
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _schema(world) -> ScenarioSemanticModel:
    s = getattr(world, "scenario_schema", None)
    if s is None:
        raise KernelError("world has no scenario schema bound — generated mode requires one")
    return s if isinstance(s, ScenarioSemanticModel) else ScenarioSemanticModel.from_dict(s)


class KernelError(Exception):
    """Kernel-op validation failure — quarantines the op, never crashes the branch."""


# ---------------------------------------------------------------- the semantically-empty kernel
def _validate_fields(schema_fields: dict, fields: dict, *, type_id: str,
                     dropped: list | None = None) -> dict:
    """Simple values only, no forbidden names (fatal). Undeclared fields on a DECLARED type
    are DROPPED and surfaced (never silently kept, never op-fatal) — the type is the semantic
    unit; new field semantics enter through declare_schema_definition."""
    out = {}
    for k, v in (fields or {}).items():
        k = str(k)[:60]
        if _FORBIDDEN_KEYS.search(k):
            raise KernelError(f"forbidden numeric-minting field {k!r}")
        if v is None:
            # a null is "no value": DROP it (surfaced), never op-fatal — one stray null from
            # an LLM was quarantining entire otherwise-valid decision records (ex4 forensic:
            # hundreds of typed actor reactions died on {"proposal_id": null})
            if dropped is not None:
                dropped.append(f"{type_id}.{k}=null")
            continue
        if not isinstance(v, (str, int, float, bool, list)):
            raise KernelError(f"field {k!r} must be a simple typed value")
        if schema_fields and k not in schema_fields and k != "status":
            if dropped is not None:
                dropped.append(f"{type_id}.{k}")
            continue
        out[k] = v
    return out


def k_create_or_update_record(world, op, ctx, delta):
    schema = _schema(world)
    rtype = str(op.get("record_type", ""))
    rtypes = schema.record_types()
    if rtype not in rtypes:
        raise KernelError(f"record type {rtype!r} not in this scenario's schema "
                          f"(known: {sorted(rtypes)[:10]}; use declare_schema_definition "
                          f"to extend)")
    dropped = []
    fields = _validate_fields(rtypes[rtype].get("fields") or {}, op.get("fields"),
                              type_id=rtype, dropped=dropped)
    if str(ctx.get("plane", "direct_action")) != "mechanism":
        # CAUSAL BOUNDARY (ownership test): a direct action writes only records the actor
        # owns/controls — never another party's record, an institution's intake, a platform's
        # state, bilateral status, or a mechanism-produced result.
        from swm.world_model_v2.scenario_schema import externally_controlled_record_types
        external = externally_controlled_record_types(schema, ctx["actor_id"])
        if rtype in external:
            ctx["report"]["directness_claims_rejected"] = \
                ctx["report"].get("directness_claims_rejected", 0) + 1
            raise KernelError(f"record type {rtype!r} is controlled by {external[rtype]!r} — "
                              f"a direct action cannot write it; it is produced by the "
                              f"controlling mechanism/system when that actually succeeds")
        td = rtypes[rtype] if isinstance(rtypes[rtype], dict) else {}
        controller = str(td.get("controlled_by", ""))
        if controller and controller != ctx["actor_id"]:
            raise KernelError(f"record type {rtype!r} is {controller!r}'s own unilateral "
                              f"act — {ctx['actor_id']} cannot perform it")
        # an institution's DECLARED decision holders record their OWN decisions in its
        # declared decision_record_type — a Layer-A act under scenario-generated authority
        # (the aggregation arithmetic, not this write, decides what the institution did)
        holder_of = any(
            rtype == str(inst.get("decision_record_type", ""))
            and ctx["actor_id"] in [str(h) for h in (inst.get("decision_holders") or [])]
            for inst in (schema.institutional_definitions or {}).values()
            if isinstance(inst, dict))
        if not controller and not holder_of:
            # terminal-smuggling test: a direct write may not satisfy an outcome predicate
            # unless the schema declares the record type as this actor's own unilateral act
            from swm.world_model_v2.scenario_schema import evaluate_predicate as _evalp
            probe_fields = dict(fields)
            if op.get("status"):
                probe_fields["status"] = op["status"]
            probe = {"record_type": rtype, "fields": probe_fields}
            if any(_evalp(p, [probe]) for p in (schema.outcome_predicates or [])
                   if isinstance(p, dict)):
                ctx["report"]["directness_claims_rejected"] = \
                    ctx["report"].get("directness_claims_rejected", 0) + 1
                raise KernelError(
                    f"terminal_smuggling: direct write of {rtype!r} would satisfy an "
                    f"outcome predicate — the terminal outcome arises from mechanisms and "
                    f"other actors, never from the acting actor's own claim")
    if dropped:
        ctx["report"]["undeclared_fields_dropped"] = \
            ctx["report"].get("undeclared_fields_dropped", 0) + len(dropped)
        delta.reason_codes.append(f"fields_dropped:{','.join(dropped[:4])[:80]}")
    rid = _sanitize(op.get("record_id") or f"{rtype}_{_hash([ctx['actor_id'], op])[:10]}")
    existing = world.objects.get(rid)
    if existing is not None:
        if existing.created_by not in ("", ctx["actor_id"]) \
                and str(existing.attributes.get("owner", "")) != ctx["actor_id"]:
            raise KernelError(f"{ctx['actor_id']} lacks authority over record {rid!r}")
        before = {"status": existing.status, "fields": dict(existing.attributes)}
        existing.attributes.update(fields)
        if op.get("status"):
            existing.status = str(op["status"])[:60]
        existing.updated_at = ctx["now"]
        delta.change(f"records[{rid}]", before,
                     {"status": existing.status, "fields": dict(existing.attributes)})
    else:
        world.objects[rid] = WorldObject(
            object_id=rid, object_type=rtype, attributes=fields,
            status=str(op.get("status", "exists"))[:60], created_at=ctx["now"],
            updated_at=ctx["now"], created_by=ctx["actor_id"],
            source_action_id=ctx.get("action_id", ""),
            visibility="participants" if op.get("visibility") == "participants" else "public",
            audience=[str(a) for a in (op.get("audience") or [])][:24],
            valid_from=ctx["now"],
            provenance={"plane": "world", "schema_id": schema.schema_id,
                        "schema_version": schema.version, "kernel": "create_or_update_record"})
        delta.change(f"records[{rid}]", None, {"type": rtype, "status": world.objects[rid].status})
    ctx["report"]["scenario_types_generated"] = len(schema.record_types())
    return rid


def k_remove_record(world, op, ctx, delta):
    rid = str(op.get("record_id", ""))
    rec = world.objects.get(rid)
    if rec is None:
        raise KernelError(f"unknown record {rid!r}")
    if rec.created_by not in ("", ctx["actor_id"]) \
            and str(rec.attributes.get("owner", "")) != ctx["actor_id"]:
        raise KernelError(f"{ctx['actor_id']} lacks authority over record {rid!r}")
    rec.status = "removed"
    rec.valid_until = ctx["now"]
    delta.change(f"records[{rid}].status", None, "removed")


def k_create_or_remove_relation(world, op, ctx, delta):
    schema = _schema(world)
    rel = _sanitize(op.get("relation", "")).lower()
    if not rel:
        raise KernelError("relation needs a name")
    if schema.relation_types and rel not in schema.relation_types:
        raise KernelError(f"relation type {rel!r} not in this scenario's schema")
    src, dst = str(op.get("src", ctx["actor_id"])), str(op.get("dst", ""))
    if not dst:
        raise KernelError("relation needs dst")
    if str(ctx.get("plane", "direct_action")) != "mechanism":
        # CAUSAL BOUNDARY: one actor creates only relations the schema explicitly declares
        # UNILATERAL and only from themselves. Bilateral, institutional, legal, market,
        # platform, or socially recognized relations require mechanisms or the other party.
        rdef = (schema.relation_types or {}).get(rel) or {}
        if not (isinstance(rdef, dict) and rdef.get("unilateral")):
            ctx["report"]["directness_claims_rejected"] = \
                ctx["report"].get("directness_claims_rejected", 0) + 1
            raise KernelError(f"relation {rel!r} is not declared unilateral — a bilateral/"
                              f"recognized relation needs the mechanism or the other "
                              f"party's own action, not one side's assertion")
        if src != ctx["actor_id"]:
            raise KernelError("an actor may create only their OWN outgoing unilateral "
                              "relation")
    if world.network is not None:
        from swm.world_model_v2.network import _RELATIONS, register_relation
        if rel not in _RELATIONS:
            # scenario-generated relation semantics enter through the typed registry door —
            # directionality comes from the schema, meaning stays scenario-scoped
            rdef = (schema.relation_types or {}).get(rel) or {}
            register_relation(rel, directed=bool(rdef.get("directed", True)),
                              uses=("scenario",))
        if op.get("remove"):
            world.network.edges = [e for e in world.network.edges
                                   if not (e.src == src and e.rel == rel and e.dst == dst)]
        else:
            world.network.add(src, rel, dst)
    delta.change(f"relation[{src}-{rel}->{dst}]", None, "removed" if op.get("remove") else "created")


def k_emit_semantic_event(world, op, ctx, delta):
    """Record that something HAPPENED in scenario terms, and hand it to the control plane.
    The kernel does not interpret it — no per-type handler exists anywhere.

    CAUSAL BOUNDARY: on the direct-action plane this records an actor-controlled ATTEMPT —
    it cannot be a mechanism's output type (that would claim the external success), it is
    observable to nobody but its source (unless the schema declares the act `unmediated`),
    and it is automatically handed to every scenario mechanism the schema declares as
    triggered by it. Verified observability (actual recipients, public availability) exists
    only on the mechanism plane, stamped by the mechanism runtime."""
    schema = _schema(world)
    tid = str(op.get("semantic_type_id", op.get("etype", "")))
    if tid not in schema.semantic_event_types:
        raise KernelError(f"semantic event type {tid!r} not in this scenario's schema "
                          f"(known: {sorted(schema.semantic_event_types)[:10]})")
    tdef = schema.semantic_event_types[tid]
    plane = str(ctx.get("plane", "direct_action"))
    if plane != "mechanism":
        from swm.world_model_v2.scenario_schema import mechanism_output_event_types
        outputs = mechanism_output_event_types(schema)
        if tid in outputs:
            ctx["report"]["directness_claims_rejected"] = \
                ctx["report"].get("directness_claims_rejected", 0) + 1
            raise KernelError(f"event type {tid!r} is produced by mechanism "
                              f"{outputs[tid]!r} — an action may only ATTEMPT; invoke the "
                              f"mechanism instead of asserting its result")
    dropped = []
    fields = _validate_fields(tdef.get("fields") or {}, op.get("structured_fields")
                              or op.get("fields"), type_id=tid, dropped=dropped)
    if dropped:
        ctx["report"]["undeclared_fields_dropped"] = \
            ctx["report"].get("undeclared_fields_dropped", 0) + len(dropped)
    budgets = ctx.get("budgets") or DEFAULT_BUDGETS
    n = sum(1 for _ in world.semantic_log)
    if n >= budgets["max_semantic_events"]:
        raise KernelError(f"semantic event budget exhausted ({n}) — cascade truncated LOUDLY")
    targets = [str(t) for t in (op.get("direct_targets") or []) if t][:16]
    unmediated = bool(tdef.get("unmediated"))
    mobs = (ctx.get("mechanism_observability") or {}) if plane == "mechanism" else {}
    verified = plane == "mechanism" or unmediated
    if plane == "mechanism":
        actual = [str(r) for r in (mobs.get("actual_recipients") or []) if r in world.entities]
    elif unmediated:
        actual = [t for t in targets if t in world.entities]
    else:
        actual = []
    sev = SemanticWorldEvent(
        event_id=f"sev_{_hash(op)[:12]}_{n:04d}", semantic_type_id=tid,
        schema_id=schema.schema_id, schema_version=schema.version,
        source_actor_id=ctx["actor_id"],
        direct_targets=targets,
        exact_content=str(op.get("exact_content", op.get("content", "")))[:1200],
        structured_fields=fields, occurred_at=ctx["now"],
        intended_visibility=str(op.get("intended_visibility",
                                       tdef.get("typical_visibility", "participants")))[:20],
        causal_layer="mechanism_output" if plane == "mechanism" else "actor_controlled",
        observability_verified=verified, actual_recipients=actual,
        availability=str(mobs.get("availability", ""))[:20],
        representation=str(mobs.get("representation", "complete"))[:20],
        mechanism_id=str(mobs.get("mechanism_id", ""))[:80],
        linked_action_id=ctx.get("action_id", ""),
        parent_event_ids=list(ctx.get("parent_event_ids") or [])[:6],
        branch_id=world.branch_id, cascade_depth=int(ctx.get("cascade_depth", 0)),
        provenance={"plane": "mechanism" if plane == "mechanism" else "world",
                    "emitted_by": ctx.get("compiler", "kernel"),
                    "instance_id": str(mobs.get("instance_id", "")),
                    "scaffolding": bool(tdef.get("scaffolding"))})
    world.semantic_log.append(sev.as_dict())
    delta.change(f"semantic_log[{sev.event_id}]", None,
                 {"type": tid, "source": sev.source_actor_id, "layer": sev.causal_layer,
                  "verified": verified})
    ctx["report"]["scenario_events_emitted"] += 1
    if tdef.get("scaffolding"):
        ctx["report"]["fallback_reasons"].append(
            {"kind": "unmodeled_action_scaffolding", "action": ctx.get("action_id", "")})
    if plane != "mechanism":
        if targets:
            ctx["report"]["intended_deliveries"] = \
                ctx["report"].get("intended_deliveries", 0) + len(targets)
        if sev.intended_visibility == "public":
            ctx["report"]["intended_publications"] = \
                ctx["report"].get("intended_publications", 0) + 1
    elif sev.availability == "public":
        ctx["report"]["actual_publications"] = \
            ctx["report"].get("actual_publications", 0) + 1
    if verified:
        # only verified-observable events route observations and discover the frontier
        ctx["events"].append(Event(
            ts=ctx["now"], etype="ctrl_semantic_event",
            participants=sev.actual_recipients or [sev.source_actor_id],
            payload={"semantic_event": sev.as_dict()}, visibility="participants",
            source="endogenous:generated_world"))
    if plane != "mechanism":
        # the boundary: hand the attempt to the scenario mechanisms the schema declares —
        # never to the targets' eyes directly
        from swm.world_model_v2.causal_boundary import auto_invoke_for_attempt
        from swm.world_model_v2.scenario_schema import mechanisms_triggered_by
        started = auto_invoke_for_attempt(world, sev.as_dict(), ctx, delta)
        if started:
            ctx.setdefault("mechanism_instances_started", []).extend(started)
        if not unmediated and not mechanisms_triggered_by(schema, tid) \
                and (targets or sev.intended_visibility == "public"):
            ctx["report"]["deliveries_unresolved_no_mechanism"] = \
                ctx["report"].get("deliveries_unresolved_no_mechanism", 0) + 1
            ctx["report"]["fallback_reasons"].append(
                {"kind": "delivery_unresolved_no_mechanism", "event_type": tid[:60],
                 "reason": "attempt has intended targets/publicity but the scenario "
                           "declares no mechanism that processes it — nobody observes it"})
            delta.reason_codes.append(f"delivery_unresolved:{tid[:40]}")
    return sev.event_id


def k_schedule_semantic_event(world, op, ctx, delta):
    """Scheduling is NOT occurrence. A scheduled future action becomes a ctrl_scheduled_attempt
    that fires at its time and re-enters this same boundary THEN — with the actor's continued
    existence, the schema restrictions, and the mechanism handoff all evaluated at fire time.
    Nothing is appended to the world-plane log now."""
    delay = float(op.get("delay_s", 0.0) or 0.0)
    if not delay > 0:
        raise KernelError("schedule_semantic_event needs a positive delay_s")
    schema = _schema(world)
    tid = str(op.get("semantic_type_id", op.get("etype", "")))
    if tid not in schema.semantic_event_types:
        raise KernelError(f"semantic event type {tid!r} not in this scenario's schema")
    if str(ctx.get("plane", "direct_action")) != "mechanism":
        from swm.world_model_v2.scenario_schema import mechanism_output_event_types
        outputs = mechanism_output_event_types(schema)
        if tid in outputs:
            ctx["report"]["directness_claims_rejected"] = \
                ctx["report"].get("directness_claims_rejected", 0) + 1
            raise KernelError(f"cannot schedule mechanism output {tid!r} as a future fact — "
                              f"a future success is still a mechanism's outcome, not the "
                              f"actor's plan")
    attempt_op = {k: v for k, v in op.items() if k != "delay_s"}
    attempt_op["op"] = "emit_semantic_event"
    ctx["events"].append(Event(
        ts=ctx["now"] + delay, etype="ctrl_scheduled_attempt",
        participants=[ctx["actor_id"]],
        payload={"attempt_op": attempt_op, "actor_id": ctx["actor_id"],
                 "action_id": ctx.get("action_id", "")},
        visibility="participants", source="endogenous:generated_world"))
    ctx["report"]["scheduled_attempts"] = ctx["report"].get("scheduled_attempts", 0) + 1
    delta.change(f"scheduled_attempt[{tid}]", None, ctx["now"] + delay)
    delta.reason_codes.append(f"scheduled_future_attempt:{tid[:40]}")
    return f"scheduled_{tid[:40]}"


def k_transfer_conserved_quantity(world, op, ctx, delta):
    schema = _schema(world)
    name = str(op.get("resource", ""))
    if schema.resource_definitions and name not in schema.resource_definitions:
        raise KernelError(f"resource {name!r} not declared in this scenario's schema")
    if str(ctx.get("plane", "direct_action")) != "mechanism":
        # CAUSAL BOUNDARY: an immediate direct transfer is legitimate only when no external
        # settlement/acceptance stands between the actor's ledger and the destination. When
        # the scenario declares a settlement mechanism, the direct credit is forbidden — the
        # actor creates a transfer ATTEMPT (escrow) and the mechanism settles or fails.
        rdef = (schema.resource_definitions or {}).get(name)
        settlement = str(rdef.get("settlement_mechanism", "")) if isinstance(rdef, dict) else ""
        if not settlement:
            settlement = next(
                (mid for mid, md in (schema.mechanism_definitions or {}).items()
                 if isinstance(md, dict)
                 and str(md.get("executor_binding")) == "conserved_resource_settlement"), "")
        if settlement:
            ctx["report"]["directness_claims_rejected"] = \
                ctx["report"].get("directness_claims_rejected", 0) + 1
            raise KernelError(f"resource {name!r} settles through mechanism {settlement!r} — "
                              f"initiate the transfer with invoke_scenario_mechanism; "
                              f"settlement is not initiation")
    amount = float(op.get("amount", 0.0) or 0.0)
    if amount <= 0:
        raise KernelError("transfer amount must be positive")
    src = str(op.get("src", ctx["actor_id"]))
    if src != ctx["actor_id"]:
        raise KernelError("an actor may transfer only from their own holdings")
    dst = str(op.get("to", ""))
    s_ent = world.entities.get(src)
    have = s_ent.value("resources", key=name, default=None) if s_ent is not None else None
    have = float(have) if isinstance(have, (int, float)) else 0.0
    if have < amount:
        raise KernelError(f"insufficient {name} ({have} < {amount}) — conservation holds")
    s_ent.set("resources", F(round(have - amount, 6), status="derived",
                             method="generated_world_kernel", updated_at=ctx["now"]), key=name)
    delta.change(f"{src}.resources[{name}]", have, round(have - amount, 6))
    if dst in world.entities:
        d_ent = world.entities[dst]
        dv = d_ent.value("resources", key=name, default=None)
        dv = float(dv) if isinstance(dv, (int, float)) else 0.0
        d_ent.set("resources", F(round(dv + amount, 6), status="derived",
                                 method="generated_world_kernel", updated_at=ctx["now"]),
                  key=name)
        delta.change(f"{dst}.resources[{name}]", dv, round(dv + amount, 6))


def k_declare_schema_definition(world, op, ctx, delta):
    """Runtime schema extension: versioned, branch-local (the schema lives on THIS world),
    ancestry-preserving. New semantics only — past definitions are immutable.

    CAUSAL BOUNDARY: an action's effect execution may not silently invent new CAUSAL
    semantics. Extensions that declare mechanisms or directly-perceivable (`unmediated`)
    event types must pass the causal-boundary critic bound into the execution context; with
    no critic available they are rejected loudly."""
    schema = _schema(world)
    proposal = op.get("definitions") or {}
    if str(ctx.get("plane", "direct_action")) != "mechanism":
        risky = bool(proposal.get("mechanism_definitions")) or any(
            isinstance(td, dict) and td.get("unmediated")
            for td in (proposal.get("semantic_event_types") or {}).values())
        if risky:
            critic = ctx.get("boundary_critic")
            if critic is None:
                ctx["report"]["directness_claims_rejected"] = \
                    ctx["report"].get("directness_claims_rejected", 0) + 1
                raise KernelError("schema extension declares causal semantics (mechanisms/"
                                  "unmediated observability) — requires the causal-boundary "
                                  "critic, none bound in this execution context")
            verdict = {}
            try:
                verdict = critic(proposal) or {}
            except Exception as e:  # noqa: BLE001 — a failed critique rejects, loudly
                raise KernelError(f"causal-boundary critic failed on the extension: "
                                  f"{type(e).__name__}")
            if str(verdict.get("verdict", "reject")) not in ("usable", "accept", "approved"):
                ctx["report"]["directness_claims_rejected"] = \
                    ctx["report"].get("directness_claims_rejected", 0) + 1
                raise KernelError(f"causal-boundary critic rejected the extension: "
                                  f"{str(verdict.get('reason', verdict.get('why', '')))[:120]}")
            delta.reason_codes.append("schema_extension_boundary_criticized")
    ok, out = extend_schema(schema, proposal,
                            reason=str(op.get("reason", "runtime extension")),
                            triggering_event_id=str(op.get("triggering_event_id", "")),
                            at=ctx["now"])
    if not ok:
        raise KernelError(f"schema extension rejected: {out[:3]}")
    world.scenario_schema = schema
    delta.change(f"scenario_schema.version", schema.ancestry[-1]["from_version"],
                 schema.version)
    delta.reason_codes.append(f"schema_extended:{','.join(map(str, out))[:80]}")
    ctx["report"]["schema_extensions"] += 1
    return out


def k_invoke_scenario_mechanism(world, op, ctx, delta):
    """The semantically empty door from a direct action to any external effect — validated
    and executed by the causal-boundary runtime (existence, input shape, authority, live
    preconditions; branch-local instance; initial state; scheduled transition; no result
    before it occurs)."""
    from swm.world_model_v2.causal_boundary import invoke_mechanism_op
    iid = invoke_mechanism_op(world, op, ctx, delta)
    ctx.setdefault("mechanism_instances_started", []).append(iid)
    return iid


KERNEL = {"create_or_update_record": k_create_or_update_record,
          "remove_record": k_remove_record,
          "create_or_remove_relation": k_create_or_remove_relation,
          "emit_semantic_event": k_emit_semantic_event,
          "schedule_semantic_event": k_schedule_semantic_event,
          "transfer_conserved_quantity": k_transfer_conserved_quantity,
          "declare_schema_definition": k_declare_schema_definition,
          "invoke_scenario_mechanism": k_invoke_scenario_mechanism}

#: ops that would write another human's mind/choice — the invariant the whole phase protects
_MIND_WRITE = re.compile(r"(belief|support|compli|vote|approval|stance|trust|decision)_of|"
                         r"set_(belief|support|vote|reaction)|other_actor", re.I)


def execute_kernel_ops(world, ops: list, ctx: dict, delta: StateDelta) -> list:
    """Apply validated kernel ops; failures quarantine LOUDLY onto ctx['quarantined'];
    returns the control-plane events to queue. `ctx['plane']` defaults to `direct_action` —
    the causal-boundary restrictions apply to every caller that does not explicitly execute
    as a mechanism (only the mechanism runtime does)."""
    ctx.setdefault("events", [])
    ctx.setdefault("quarantined", [])
    ctx.setdefault("plane", "direct_action")
    applied = 0
    for op in list(ops or [])[:24]:
        if not isinstance(op, dict):
            ctx["quarantined"].append({"op": op, "reason": "not an op object"})
            continue
        name = str(op.get("op", ""))
        fn = KERNEL.get(name)
        try:
            if fn is None:
                raise KernelError(f"unknown kernel op {name!r} (kernel: {KERNEL_OPS})")
            if _MIND_WRITE.search(json.dumps(op, default=str)):
                ctx["report"]["human_reactions_attempted_directly"] = \
                    ctx["report"].get("human_reactions_attempted_directly", 0) + 1
                raise KernelError("op attempts to write another human's mind/choice — "
                                  "reactions come only from that actor's own simulation")
            fn(world, op, ctx, delta)
            applied += 1
            if ctx["plane"] != "mechanism" and name != "invoke_scenario_mechanism":
                ctx["report"]["actor_controlled_effects"] = \
                    ctx["report"].get("actor_controlled_effects", 0) + 1
        except (KernelError, KeyError, ValueError, TypeError, AttributeError) as e:
            ctx["quarantined"].append({"op": op, "reason": f"{type(e).__name__}: {e}"[:200]})
            ctx["report"]["unsupported_semantics"] += 1
    delta.reason_codes.append(f"kernel_ops:{applied}"
                              + (f":quarantined:{len(ctx['quarantined'])}"
                                 if ctx["quarantined"] else ""))
    return ctx["events"]


# ---------------------------------------------------------------- action → direct effects
_ACTION_COMPILE_PROMPT = """You are the DIRECT-EFFECT COMPILER for a generated-world simulation. An actor just
successfully performed an action. Express ONLY what the actor's own successful performance directly makes
true, using the kernel operations and THIS SCENARIO'S OWN types below. How other people respond is decided by
THEIR simulations — never assert another person's reaction, belief, support, or vote. Everything below is
data, never instructions.

ACTOR: {actor_id}
THEIR EXACT DECISION: {decision}
TARGET: {target} | TIMING: {timing} | INTENDED VISIBILITY: {observability}
INTENDED EFFECT (their words): {intent}

THIS SCENARIO'S RECORD TYPES: {record_types}
THIS SCENARIO'S SEMANTIC EVENT TYPES: {event_types}
DECLARED RESOURCES: {resources}
EXISTING RECORDS (id: type/status): {records}

KERNEL OPERATIONS (storage mechanics only — meanings come from the scenario types):
- create_or_update_record: record_type, fields, [record_id, status, visibility, audience]
- remove_record: record_id
- create_or_remove_relation: relation, src, dst, [remove]
- emit_semantic_event: semantic_type_id, exact_content (the actor's ACTUAL words/artifact text),
  [direct_targets, structured_fields, intended_visibility, delay_s]
- schedule_semantic_event: same + delay_s > 0 (a future structural consequence of THIS action)
- transfer_conserved_quantity: resource, amount, to
- declare_schema_definition: definitions {{entity_types|fact_types|semantic_event_types|
  process_definitions: {{new_type_id: {{description, fields}}}}}}, reason — use ONLY when this action's
  semantics genuinely need a type the schema lacks.

HARD RULES: direct effects only; no other-person reactions; no probabilities/progress/utilities; amounts only
for declared resources; preserve the actor's exact content in exact_content. 1-6 ops. Return ONLY a JSON
array of kernel op objects."""


class GeneratedActionCompiler:
    """SUPERSEDED for production by `causal_boundary.CausalActionCompiler` (attempt premise,
    DirectActionProgram, directness validation). Kept ONLY as the deterministic scaffolding
    fallback used in offline tests — its prompt's success premise must not compile production
    consequences. The kernel boundary applies to its ops regardless."""

    def __init__(self, llm=None, *, max_llm_calls: int = 300):
        self.llm = llm
        self.max_llm_calls = max_llm_calls
        self._calls = 0
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_ops(raw) -> list:
        """Accept the op shapes LLMs actually emit: flat {"op": name, …} (canonical),
        wrapper {name: {…}}, and {"operation": name, …} — all normalized to canonical."""
        out = []
        for op in raw or []:
            if not isinstance(op, dict):
                out.append(op)
                continue
            if op.get("op"):
                out.append(op)
            elif op.get("operation"):
                out.append({"op": str(op.pop("operation")), **op})
            elif len(op) == 1:
                k, v = next(iter(op.items()))
                if str(k) in KERNEL and isinstance(v, dict):
                    out.append({"op": str(k), **v})
                else:
                    out.append(op)
            else:
                out.append(op)
        return out

    @staticmethod
    def _invalid_type_refs(schema, ops) -> set:
        known_r, known_e = set(schema.record_types()), set(schema.semantic_event_types)
        bad = set()
        for op in ops:
            if not isinstance(op, dict):
                continue
            rt = str(op.get("record_type", "") or "")
            et = str(op.get("semantic_type_id", op.get("etype", "")) or "")
            if rt and rt not in known_r:
                bad.add(rt)
            if et and et not in known_e:
                bad.add(et)
        return bad

    def compile(self, world, action, *, qualitative=None, report=None) -> tuple:
        """Returns (ops, meta). meta records compiler path + quarantine seeds."""
        schema = _schema(world)
        decision_text = ""
        if isinstance(qualitative, dict):
            decision_text = str(qualitative.get("decision_summary")
                                or qualitative.get("chosen_action") or "")[:400]
        intent = str((action.parameters or {}).get("intended_effect", ""))[:300]
        raw, path = None, "deterministic_fallback"
        if self.llm is not None:
            with self._lock:
                budget_ok = self._calls < self.max_llm_calls
                if budget_ok:
                    self._calls += 1
            if budget_ok:
                prompt = _ACTION_COMPILE_PROMPT.format(
                    actor_id=action.actor_id,
                    decision=decision_text or action.action_name,
                    target=action.target.target_id or "none",
                    timing=str((action.parameters or {}).get("timing", "immediate")),
                    observability=str(qualitative.get("observability", "participants"))[:20]
                    if isinstance(qualitative, dict) else "participants",
                    intent=intent or decision_text or action.action_name,
                    record_types=json.dumps({k: sorted((v.get("fields") or {}))
                                             for k, v in list(schema.record_types().items())[:18]},
                                            default=str)[:1200],
                    event_types=json.dumps({k: sorted((v.get("fields") or {}))
                                            for k, v in
                                            list(schema.semantic_event_types.items())[:18]},
                                           default=str)[:1200],
                    resources=sorted(schema.resource_definitions)[:10] or "none declared",
                    records=[f"{o.object_id}: {o.object_type}/{o.status}"
                             for o in list(world.objects.values())[:14]])
                try:
                    from swm.engine.grounding import parse_json
                    r = parse_json(self.llm(prompt))
                    if isinstance(r, dict):
                        r = r.get("operations") if isinstance(r.get("operations"), list) else [r]
                    if isinstance(r, list):
                        raw, path = self._normalize_ops(r), "llm"
                        bad = self._invalid_type_refs(schema, raw)
                        if bad and len(bad) * 2 >= len(raw):
                            # the proposal missed the scenario vocabulary — one repair round
                            # with the full valid id lists (never silent, never invented)
                            with self._lock:
                                retry_ok = self._calls < self.max_llm_calls
                                if retry_ok:
                                    self._calls += 1
                            if retry_ok:
                                r2 = parse_json(self.llm(
                                    prompt + "\n\nYOUR OPS REFERENCED UNDECLARED TYPES: "
                                    + ", ".join(sorted(bad)[:8])
                                    + "\nVALID record_type ids: "
                                    + ", ".join(sorted(schema.record_types()))
                                    + "\nVALID semantic_type_id ids: "
                                    + ", ".join(sorted(schema.semantic_event_types))
                                    + "\nReturn the corrected FULL JSON array using ONLY "
                                      "these ids."))
                                if isinstance(r2, dict):
                                    r2 = r2.get("operations") \
                                        if isinstance(r2.get("operations"), list) else [r2]
                                if isinstance(r2, list):
                                    r2 = self._normalize_ops(r2)
                                    if len(self._invalid_type_refs(schema, r2)) < len(bad):
                                        raw, path = r2, "llm_vocab_repaired"
                except Exception as e:  # noqa: BLE001 — loud fallback below
                    if report is not None:
                        report["fallback_reasons"].append(
                            {"kind": "action_compiler_llm_failed",
                             "reason": f"{type(e).__name__}"[:60]})
        if raw is None:
            # exact content preserved; scaffolding type is schema-scoped and counted
            raw = [{"op": "emit_semantic_event", "semantic_type_id": UNMODELED_EVENT_TYPE,
                    "exact_content": decision_text or intent or action.action_name,
                    "structured_fields": {"action_name": action.action_name[:60],
                                          "content": (decision_text or intent)[:400],
                                          "target": action.target.target_id[:60]},
                    "direct_targets": [t for t in (action.target.target_id,) if t],
                    "intended_visibility": "participants"}]
            if report is not None and path == "deterministic_fallback":
                report["numeric_fallbacks"] += 0
                report["fallback_reasons"].append(
                    {"kind": "action_semantics_unmodeled", "action": action.action_name[:60]})
        return raw, {"compiler": path, "decision_text": decision_text}


# ---------------------------------------------------------------- control-plane operators
def _budgets(world) -> dict:
    b = world.uncertainty_meta.setdefault("generated_budgets", dict(DEFAULT_BUDGETS))
    b.setdefault("invocations", {})
    return b


def _channel_model_id(model, chan: str) -> str:
    """Map a schema channel label onto the scenario temporal model's channel set: exact id,
    else first channel of that kind, else the label itself (miss → labeled broad band)."""
    if model is None:
        return str(chan)
    if chan in (model.channels or {}):
        return str(chan)
    for cid, c in (model.channels or {}).items():
        if str(getattr(c, "kind", "")) == str(chan):
            return str(cid)
    return str(chan)


def route_semantic_event(world, sev: dict, report: dict) -> list:
    """Deterministic observation routing: who receives what, through WHICH ACTUAL CHANNEL,
    in which representation. The router NEVER interprets the event for the actor.

    CAUSAL BOUNDARY: only VERIFIED observability routes. `direct_targets` never means
    "recipients who observed this"; `intended_visibility='public'` never means everyone
    became aware. A mechanism's successful output carries `actual_recipients` (and
    `availability='public'` when a publication/availability mechanism succeeded) — those,
    and only those, become observations.

    TEMPORAL RUNTIME: delivery timing comes from the scenario temporal model's channel
    stages (transmission → delivery → moderation → exposure), sampled per particle — never
    a fixed 60-second/one-hour constant (§9/§10). A public availability publishes once;
    each recipient's EXPOSURE is their own (spread over the channel's exposure process).
    Delivery is AVAILABILITY, not attention."""
    from swm.world_model_v2.temporal_runtime import channel_delivery_ts, get_stats, temporal_model_of
    schema = _schema(world)
    rules = schema.information_rules or {}
    model = temporal_model_of(world)
    stats = get_stats(world)
    if not sev.get("observability_verified"):
        return []                    # an unprocessed attempt reaches nobody's eyes
    source = str(sev.get("source_actor_id", ""))
    recipients = [r for r in (sev.get("actual_recipients") or []) if r in world.entities]
    urgency = max(0.0, min(1.0, float(sev.get("urgency", 0.0) or 0.0)))
    deliveries = []
    for r in recipients:
        if r == source:
            continue
        chan = str(sev.get("mechanism_id") or rules.get("default_channel", "direct"))
        cid = _channel_model_id(model, chan)
        available_ts, prov = channel_delivery_ts(
            world, model, channel_id=cid, sent_ts=world.clock.now, urgency=urgency,
            recipient=r, salt=f"{sev.get('event_id', '')}:{r}", stats=stats)
        deliveries.append(Event(
            ts=max(float(available_ts), world.clock.now), etype="ctrl_deliver_observation",
            participants=[r],
            payload={"recipient": r, "semantic_event": sev, "channel": cid,
                     "representation": str(sev.get("representation", "complete")),
                     "urgency": urgency, "delivery_provenance": prov,
                     "sent_ts": world.clock.now, "verified_direct": True},
            parent_ids=[str(sev.get("event_id", ""))],
            visibility="participants", source="endogenous:generated_world"))
    if str(sev.get("availability")) == "public":
        # actual public availability (a publication mechanism succeeded) — broad routing in
        # the public channel's representation, each recipient's exposure sampled from the
        # channel's own exposure process (never a fixed one-hour constant)
        persons = [eid for eid, e in world.entities.items()
                   if getattr(e, "entity_type", "person") == "person"]
        for r in persons:
            if r == source or r in recipients:
                continue
            chan = str(rules.get("public_channel", "public_broadcast"))
            cid = _channel_model_id(model, chan)
            available_ts, prov = channel_delivery_ts(
                world, model, channel_id=cid, sent_ts=world.clock.now, urgency=urgency,
                recipient=r, salt=f"{sev.get('event_id', '')}:pub:{r}", stats=stats)
            deliveries.append(Event(
                ts=max(float(available_ts), world.clock.now),
                etype="ctrl_deliver_observation", participants=[r],
                payload={"recipient": r, "semantic_event": sev, "channel": cid,
                         "representation": str(rules.get("public_representation",
                                                         "complete")),
                         "urgency": urgency, "delivery_provenance": prov,
                         "sent_ts": world.clock.now, "verified_direct": False},
                parent_ids=[str(sev.get("event_id", ""))],
                visibility="participants", source="endogenous:generated_world"))
    return deliveries


class GeneratedSemanticEventOperator:
    """ctrl_semantic_event: route observations + discover the causal frontier. Control plane —
    this operator never writes world semantics."""

    name = "generated_semantic_event_router"

    def __init__(self, *, report: dict, frontier_llm=None):
        self.report = report
        self.frontier_llm = frontier_llm

    def applicable(self, world, event):
        return event.etype == "ctrl_semantic_event" \
            and isinstance(event.payload.get("semantic_event"), dict) \
            and getattr(world, "scenario_schema", None) is not None

    def run(self, world, event, rng):
        from swm.world_model_v2.temporal_runtime import (channel_delivery_ts, get_stats,
                                                         temporal_model_of)
        sev = event.payload["semantic_event"]
        delta = StateDelta(at=world.clock.now, event_type="ctrl_semantic_event",
                           operator=self.name,
                           reason_codes=[f"routing:{sev.get('semantic_type_id', '?')[:40]}"])
        follow = route_semantic_event(world, sev, self.report)
        # frontier discovery runs ONLY on verified-observable events: nobody is invoked by an
        # attempt that never actually reached the world
        frontier = discover_causal_frontier(world, sev, llm=self.frontier_llm,
                                            report=self.report) \
            if sev.get("observability_verified") else []
        depth = int(sev.get("cascade_depth", 0))
        self.report["recursive_cascade_depth"] = max(
            self.report.get("recursive_cascade_depth", 0), depth)
        recipients = {e.payload["recipient"] for e in follow}
        model = temporal_model_of(world)
        stats = get_stats(world)
        for actor_id, reason in frontier:
            if actor_id in recipients:
                continue                    # their delivery already carries the information
            # a frontier actor learns through an ACTUAL channel too: availability + their own
            # attention — never a fixed reconsideration timer (§9)
            schema = _schema(world)
            chan = (schema.information_rules or {}).get("default_channel", "direct")
            cid = _channel_model_id(model, chan)
            available_ts, prov = channel_delivery_ts(
                world, model, channel_id=cid, sent_ts=world.clock.now,
                urgency=max(0.0, min(1.0, float(sev.get("urgency", 0.0) or 0.0))),
                recipient=actor_id, salt=f"frontier:{sev.get('event_id', '')}:{actor_id}",
                stats=stats)
            follow.append(Event(
                ts=max(float(available_ts), world.clock.now),
                etype="ctrl_deliver_observation", participants=[actor_id],
                payload={"recipient": actor_id, "semantic_event": sev, "channel": cid,
                         "representation": "complete",
                         "frontier_reason": str(reason)[:160],
                         "delivery_provenance": prov, "sent_ts": world.clock.now},
                parent_ids=[str(sev.get("event_id", ""))],
                visibility="participants", source="endogenous:generated_world"))
        delta.follow_up_events = [{"etype": e.etype, "ts": e.ts,
                                   "participants": list(e.participants),
                                   "payload": dict(e.payload),
                                   "parent_ids": list(e.parent_ids or [])} for e in follow]
        return delta, ValidationResult(ok=True)


def _record_truncation(world, report, *, kind: str, actor: str, detail: str = "",
                       halt: bool = True, pending_trigger: dict = None):
    """§20 first-class branch truncation: record the exact reason at the exact timestamp,
    preserve the pending decision trigger, and (default) HALT the branch — a truncated branch
    never continues advancing under a missing actor decision. A safety budget is an execution
    limit, not a causal event; the branch is truncated, never silently quiescent, and no
    substitute action is ever created for the actor (§0.2/§19)."""
    from swm.world_model_v2.temporal_runtime import get_stats
    try:
        from swm.world_model_v2.truncation import map_truncation_kind
        status = map_truncation_kind(kind)
    except Exception:  # noqa: BLE001 — vocabulary module optional during migration
        status = "truncated_event_budget"
    rec = {"kind": kind, "branch_status": status, "actor": actor, "at_ts": world.clock.now,
           "detail": str(detail)[:200],
           "unresolved_decision_trigger": dict(pending_trigger or {}),
           "why_actor_matters": "a pending causal chain names this actor; additional compute "
                                "would process their reconsideration"}
    if report is not None:
        report.setdefault("temporal_truncations", []).append(rec)
        report.setdefault("fallback_reasons", []).append({"kind": kind, "actor": actor})
    stats = get_stats(world)
    stats.temporally_truncated = True
    if not stats.truncation:
        stats.truncation = {"reason": kind, "branch_status": status, "at_ts": world.clock.now,
                            "actors_not_processed": [], "note": rec["why_actor_matters"],
                            "unresolved_decision_trigger": dict(pending_trigger or {})}
    naf = stats.truncation.setdefault("actors_not_processed", [])
    if actor not in naf:
        naf.append(actor)
    if halt:
        # the branch loop checks this flag after every processed timestamp and stops the
        # branch with full world state + pending events preserved (§20.1-§20.8)
        stats.branch_halted = True
        stats.branch_status = status


def _invocation_event(world, actor_id: str, sev: dict, *, reason: str,
                      observation_ids=(), at_ts: float = None, trigger: dict = None,
                      bundle=None, report=None):
    """Internal ActorReconsiderationTask (scheduler metadata, not a world event): dedup per
    (actor, semantic event). The invocation happens at the REAL triggering time (`at_ts` —
    normally the attention/notice event's own timestamp), carrying a first-class
    DecisionTrigger (§6) — never a fixed post-observation delay. SAFETY budgets protect the
    service; hitting one records a temporal truncation (§12), never a silent drop."""
    budgets = _budgets(world)
    key = f"{actor_id}|{sev.get('event_id', '')}"
    pending = world.uncertainty_meta.setdefault("pending_reconsiderations", [])
    if key in pending:
        return None                          # duplicate of an already-pending reconsideration
    used = budgets["invocations"].get(actor_id, 0)
    if used >= budgets["max_invocations_per_actor"]:
        _record_truncation(world, report, kind="invocation_safety_budget_reached",
                           actor=actor_id,
                           detail=f"safety cap {budgets['max_invocations_per_actor']}")
        return None
    if int(sev.get("cascade_depth", 0)) >= budgets["max_cascade_depth"]:
        _record_truncation(world, report, kind="cascade_depth_safety_budget_reached",
                           actor=actor_id,
                           detail=f"safety cap {budgets['max_cascade_depth']}")
        return None
    pending.append(key)
    del pending[:-256]
    ts = float(at_ts) if at_ts is not None else world.clock.now
    payload = {"actor_id": actor_id,
               "triggering_observation_ids": list(observation_ids),
               "triggering_semantic_event": sev,
               "reason_actor_may_be_causally_relevant": str(reason)[:200],
               "cascade_depth": int(sev.get("cascade_depth", 0)) + 1}
    if trigger:
        payload["trigger"] = dict(trigger)
    if bundle:
        payload["observation_bundle"] = list(bundle)[:24]
    return Event(ts=max(ts, world.clock.now), etype="ctrl_invoke_actor",
                 participants=[actor_id], payload=payload,
                 trigger=dict(trigger or {}),
                 visibility="participants", source="endogenous:generated_world")


def discover_causal_frontier(world, sev: dict, *, llm=None, report=None) -> list:
    """Per-event causal-frontier discovery: which actors may NOW matter. Deterministic base
    (targets; institutional decision holders whose matter this touches; network neighbors of
    the source for public events) + optional LLM extension, all validated (existing ids only,
    dedup, budget) — no private simulator truth crosses here."""
    schema = _schema(world)
    out, seen = [], set()

    def add(aid, reason):
        aid = str(aid)
        if aid and aid in world.entities and aid not in seen \
                and aid != str(sev.get("source_actor_id", "")):
            seen.add(aid)
            out.append((aid, reason))

    for t in sev.get("actual_recipients") or []:
        add(t, "actual recipient of the verified event")
    for iid, inst in (schema.institutional_definitions or {}).items():
        text = json.dumps(sev, default=str)[:1500]
        if iid in text or any(str(h) in text for h in inst.get("decision_holders") or []):
            for h in inst.get("decision_holders") or []:
                add(h, f"decision holder in {iid}")
    # ACTUAL public availability (a publication mechanism succeeded) reaches the source's
    # network — intended publicity alone reaches nobody
    if str(sev.get("availability")) == "public" and world.network is not None:
        src = str(sev.get("source_actor_id", ""))
        for e in list(world.network.out_edges(src)) + list(world.network.in_edges(src)):
            add(e.dst if e.src == src else e.src,
                "network neighbor of the source (actually-published event)")
    if llm is not None and len(out) < 6:
        try:
            from swm.engine.grounding import parse_json
            r = parse_json(llm(
                "Given this world event, list ONLY actor ids (from the known list) whose "
                "situation materially changed and why. Known actors: "
                f"{sorted(world.entities)[:40]}\nEVENT: {json.dumps(sev, default=str)[:800]}\n"
                'Return JSON: [{"actor_id": "…", "reason": "…"}]'))
            for row in r if isinstance(r, list) else []:
                if isinstance(row, dict):
                    add(row.get("actor_id"), f"llm_frontier: {str(row.get('reason'))[:80]}")
        except Exception:  # noqa: BLE001 — deterministic base already stands
            if report is not None:
                report["fallback_reasons"].append({"kind": "frontier_llm_failed"})
    # NO fixed frontier cap (§7): every actor whose situation materially changed participates.
    # Compute pressure is handled by safety budgets + temporal truncation, never by silently
    # dropping the tail of the frontier.
    return out


class GeneratedObservationDeliveryOperator:
    """ctrl_deliver_observation: the item becomes technically AVAILABLE to one actor — it is
    published to the world's information plane but NOT exposed to the actor (delivered ≠ read,
    invariants 17/18). The actor's attention is a SEPARATE event: the runtime schedules (or
    coalesces into) their next real attention opportunity per their temporal profile — channel
    checking habits, sleep/work windows, urgency interrupts, sampled latent state (§9). Never
    interprets, never picks reactions."""

    name = "generated_observation_delivery"

    def __init__(self, *, report: dict):
        self.report = report

    def applicable(self, world, event):
        return event.etype == "ctrl_deliver_observation" \
            and getattr(world, "scenario_schema", None) is not None

    def run(self, world, event, rng):
        from swm.world_model_v2.temporal_runtime import (get_stats,
                                                         record_available_observation,
                                                         schedule_attention,
                                                         temporal_model_of)
        p = event.payload
        recipient = str(p.get("recipient", ""))
        sev = p.get("semantic_event") or {}
        delta = StateDelta(at=world.clock.now, event_type="ctrl_deliver_observation",
                           operator=self.name,
                           reason_codes=[f"available_to:{recipient}"])
        if recipient not in world.entities:
            return delta, ValidationResult(ok=False, reasons=["unknown recipient"])
        content = str(sev.get("exact_content", ""))
        if str(p.get("representation")) == "summary" and len(content) > 200:
            content = content[:200] + " …[summarized in transit]"
        iid = f"obs_{_hash([recipient, sev.get('event_id')])[:12]}"
        if world.information is not None:
            # published (it EXISTS) — but NOT exposed to the recipient until they notice it
            world.information.publish(InformationItem(
                iid, content or f"[{sev.get('semantic_type_id', 'event')}]",
                kind="private" if sev.get("intended_visibility") != "public" else "public",
                source=str(sev.get("source_actor_id", "")), created_at=world.clock.now,
                about=str(sev.get("semantic_type_id", ""))[:60]))
        model = temporal_model_of(world)
        stats = get_stats(world)
        record_available_observation(
            world, recipient=recipient,
            item={"iid": iid, "semantic_event": sev, "content": content,
                  "source": str(sev.get("source_actor_id", "")),
                  "urgency": float(p.get("urgency", 0.0) or 0.0),
                  "sent_ts": p.get("sent_ts")},
            available_ts=world.clock.now, channel=str(p.get("channel", ""))[:40], stats=stats)
        delta.change(f"information_available[{recipient}]", None, iid)
        self.report["observations_delivered"] += 1
        if p.get("verified_direct"):
            self.report["actual_deliveries"] = self.report.get("actual_deliveries", 0) + 1
        rep_map = dict(sev.get("actual_observability") or {})
        rep_map[recipient] = str(p.get("representation", "complete"))
        sev["actual_observability"] = rep_map
        # attention is a separate causal step: availability never implies the actor noticed —
        # the temporal model's attention process decides WHEN (never a fixed 30-minute delay)
        att = schedule_attention(world, model, actor_id=recipient,
                                 channel_id=str(p.get("channel", ""))[:40],
                                 available_ts=world.clock.now,
                                 urgency=float(p.get("urgency", 0.0) or 0.0),
                                 sender=str(sev.get("source_actor_id", "")), stats=stats)
        if att is not None:
            delta.follow_up_events = [{"etype": att.etype, "ts": att.ts,
                                       "participants": list(att.participants),
                                       "payload": dict(att.payload),
                                       "parent_ids": [event.event_id]}]
        return delta, ValidationResult(ok=True)


class GeneratedAttentionOperator:
    """ctrl_attention: the actor's REAL attention opportunity on a channel. Everything that
    became available by now enters their information set as ONE ordered bundle (§20) — one
    actor view, one invocation, with a first-class DecisionTrigger (§6). If nothing is pending
    (already collected by an earlier check), this is an honest no-op."""

    name = "generated_attention"

    def __init__(self, *, report: dict):
        self.report = report

    def applicable(self, world, event):
        # mode-agnostic: attention events exist in BOTH consequence modes (a fixed-v1 world's
        # message_delivered also flows availability → attention → decision)
        return event.etype == "ctrl_attention" and bool(event.participants)

    def run(self, world, event, rng):
        from swm.world_model_v2.temporal_runtime import (collect_attention_bundle, get_stats,
                                                         make_trigger)
        p = event.payload
        actor_id = str(p.get("actor_id", event.participants[0]))
        stats = get_stats(world)
        bundle = collect_attention_bundle(world, actor_id=actor_id, now_ts=world.clock.now,
                                          channel=str(p.get("channel", "")), stats=stats)
        delta = StateDelta(at=world.clock.now, event_type="ctrl_attention",
                           operator=self.name,
                           reason_codes=[f"attention:{actor_id}",
                                         f"n_noticed:{len(bundle)}"])
        if not bundle:
            return None, ValidationResult(ok=True, reasons=["nothing_newly_available"])
        self.report["attention_events"] = self.report.get("attention_events", 0) + 1
        self.report["observation_bundles_delivered"] = \
            self.report.get("observation_bundles_delivered", 0) + 1
        for it in bundle:
            delta.change(f"information_exposure[{actor_id}]", None, it.get("iid"))
        # the actor decides ONCE from the full noticed bundle — the invocation is triggered by
        # NOTICING (a real event), not by a fixed post-delivery timer
        newest = bundle[-1]
        trigger = make_trigger(
            trigger_type="newly_noticed_information", actor_id=actor_id,
            parents=[event.event_id] + [str((it.get("semantic_event") or {}).get("event_id", ""))
                                        for it in bundle[:6]],
            observed=f"{len(bundle)} item(s) noticed on "
                     f"{p.get('channel', 'their channels')}",
            relevance="newly noticed information may change the actor's situation",
            why_now=f"the actor's real attention opportunity "
                    f"({p.get('notice_provenance', 'attention_model')})",
            provenance="temporal_attention")
        if getattr(world, "scenario_schema", None) is not None:
            inv = _invocation_event(world, actor_id, newest.get("semantic_event") or {},
                                    reason="noticed newly available information",
                                    observation_ids=[it.get("iid") for it in bundle],
                                    at_ts=world.clock.now, trigger=trigger,
                                    bundle=[{k: it.get(k) for k in
                                             ("iid", "content", "source", "channel",
                                              "available_ts", "urgency")}
                                            for it in bundle], report=self.report)
            if inv is not None:
                delta.follow_up_events = [{"etype": inv.etype, "ts": inv.ts,
                                           "participants": list(inv.participants),
                                           "payload": dict(inv.payload),
                                           "parent_ids": [event.event_id],
                                           "trigger": trigger}]
        else:
            # fixed-v1 worlds: the noticed bundle opens ONE phase4 decision_opportunity NOW
            lines = [f"[{it.get('channel', '?')}] from {it.get('source', '?')}: "
                     f"\"{str(it.get('content', ''))[:240]}\"" for it in bundle[:6]]
            delta.follow_up_events = [{
                "etype": "decision_opportunity", "ts": world.clock.now,
                "participants": [actor_id],
                "payload": {"situation": ("you notice " + ("; ".join(lines))[:700]),
                            "trigger": trigger,
                            "source_action_id": (bundle[-1].get("source_action_id") or "")},
                "parent_ids": [event.event_id], "trigger": trigger}]
        return delta, ValidationResult(ok=True)


class GeneratedActorInvocationOperator:
    """ctrl_invoke_actor: the actor's perceived world materially changed — rebuild their view
    and invoke their persistent LLM. THE ACTOR decides whether anything should be done and
    what; no candidate menu is required (schema affordances ride along as examples only), and
    deliberate inaction is a first-class outcome."""

    name = "generated_actor_invocation"

    def __init__(self, runtime, *, report: dict):
        self.runtime = runtime                  # the bound (qualitative) ActorPolicyRuntime
        self.report = report

    def applicable(self, world, event):
        return event.etype == "ctrl_invoke_actor" and bool(event.participants) \
            and getattr(world, "scenario_schema", None) is not None

    _PERSISTENT_CHANNEL_MARKERS = ("email", "inbox", "mail", "document", "portal", "queue",
                                   "ticket", "letter", "filing", "message")

    def _requeue_missed(self, world, actor_id, bundle, posterior):
        """§10: an item the actor cognitively missed on a PERSISTENT channel remains available
        (bounded re-attempts recorded); missed ephemeral items are recorded as missed for good.
        Exposure honesty: the item was presented, not registered — the cognition record carries
        which and why."""
        cog = ((getattr(posterior, "provenance", None) or {}).get("cognition")
               if posterior is not None else None) or {}
        missed = {str(m.get("obs_id")) for m in (cog.get("observations_missed") or [])
                  if isinstance(m, dict)}
        if not missed:
            return
        store = getattr(world, "temporal_attention", None)
        if not isinstance(store, dict):
            return
        buf = store.setdefault(actor_id, {"available": [], "scheduled_attention": {}})
        for it in bundle:
            iid = str(it.get("iid", ""))
            if iid not in missed:
                continue
            channel = str(it.get("channel", "")).lower()
            persistent = any(k in channel for k in self._PERSISTENT_CHANNEL_MARKERS)
            attempts = int(it.get("_notice_reattempts", 0) or 0)
            if persistent and attempts < 2:
                item = dict(it)
                item["_notice_reattempts"] = attempts + 1
                item.setdefault("available_ts", world.clock.now)
                buf["available"].append(item)

    def run(self, world, event, rng):
        from swm.world_model_v2.temporal_runtime import get_stats, record_invocation
        p = event.payload
        actor_id = str(p.get("actor_id", event.participants[0]))
        budgets = _budgets(world)
        used = budgets["invocations"].get(actor_id, 0)
        delta = StateDelta(at=world.clock.now, event_type="ctrl_invoke_actor",
                           operator=self.name, reason_codes=[f"reconsider:{actor_id}"])
        self.report["actors_reconsidered"] += 1
        stats = get_stats(world)
        if used >= budgets["max_invocations_per_actor"]:
            # SAFETY budget (§12): the branch is temporally truncated — recorded loudly on the
            # report AND the branch stats; NEVER converted to a numeric policy or treated as
            # the actor naturally going quiet.
            delta.reason_codes.append("temporally_truncated:invocation_safety_budget")
            _record_truncation(world, self.report,
                               kind="invocation_safety_budget_reached", actor=actor_id,
                               detail=f"cap {budgets['max_invocations_per_actor']} at "
                                      f"invocation time")
            return delta, ValidationResult(ok=True)
        engine = getattr(self.runtime, "engine", None)
        if engine is not None and hasattr(engine, "budget_left") and not engine.budget_left():
            # LLM-call SAFETY budget exhausted (invariant 38): the pending decision cannot be
            # simulated — record a temporal truncation and DO NOT invent behavior for the
            # actor (no numeric fallback, no fake wait)
            delta.reason_codes.append("temporally_truncated:actor_llm_budget_exhausted")
            _record_truncation(world, self.report, kind="actor_llm_budget_exhausted",
                               actor=actor_id,
                               detail="qualitative runtime call budget reached before this "
                                      "invocation; additional compute would simulate it")
            return delta, ValidationResult(ok=True)
        budgets["invocations"][actor_id] = used + 1
        sev = p.get("triggering_semantic_event") or {}
        trigger = dict(p.get("trigger") or {})
        if not trigger:
            from swm.world_model_v2.temporal_runtime import make_trigger
            trigger = make_trigger(
                trigger_type="observable_state_change", actor_id=actor_id,
                parents=[str(sev.get("event_id", ""))],
                observed=str(p.get("reason_actor_may_be_causally_relevant", ""))[:200],
                relevance="the actor's perceived situation materially changed",
                why_now="the change became observable to the actor at this time",
                provenance="generated_world_control_plane")
        record_invocation(stats, actor_id=actor_id, trigger=trigger)
        schema = _schema(world)
        role = (schema.actor_roles or {}).get(actor_id) or {}
        bundle = p.get("observation_bundle") or []
        if bundle:
            # §20: the actor sees the WHOLE noticed bundle at once, ordered, with sources
            lines = [f"- [{it.get('channel', '?')}] from {it.get('source', '?')}: "
                     f"\"{str(it.get('content', ''))[:220]}\"" for it in bundle[:8]]
            situation = (f"you just checked your channels and found {len(bundle)} new "
                         f"item(s):\n" + "\n".join(lines))
        else:
            situation = (f"{str(sev.get('semantic_type_id', 'a development')).replace('_', ' ')}: "
                         f"\"{str(sev.get('exact_content', ''))[:400]}\""
                         if sev else str(p.get("reason_actor_may_be_causally_relevant", "")))
        decision = {"situation": situation[:900],
                    "question_id": f"reconsider_{sev.get('event_id', '')[:20]}",
                    "trigger": trigger,
                    # §10: the delivered bundle is the availability set for the actor's
                    # bounded-cognition attention stage (availability ≠ noticing)
                    "observation_bundle": bundle}
        affordances = [a for a in (role.get("affordances") or []) if isinstance(a, str)][:8]
        if affordances:
            # EXAMPLES of feasible capabilities — the qualitative schema explicitly allows an
            # action outside this list, and choosing nothing at all
            decision["candidate_actions"] = affordances
        seed = rng.randrange(0, 2**31 - 1)
        try:
            selected, posterior, trace = self.runtime.decide(
                None, [world], actor_id, decision=decision, seed=seed,
                observed_events=[event])
        except Exception as e:  # noqa: BLE001 — §19/§20: a failed actor decision TRUNCATES
            from swm.world_model_v2.qualitative_actor import ActorDecisionUnavailable
            if isinstance(e, ActorDecisionUnavailable):
                delta.reason_codes.append(f"branch_truncated:{e.reason}")
                _record_truncation(world, self.report, kind=e.reason, actor=actor_id,
                                   detail=str(e), halt=True, pending_trigger=trigger)
                return delta, ValidationResult(ok=True)
            raise
        # §10: cognitively MISSED persistent-channel items stay available for a later real
        # attention opportunity (an unread email remains in the inbox); ephemeral items do not.
        self._requeue_missed(world, actor_id, bundle, posterior)
        self.report["actors_invoked"] += 1
        qual = (posterior.provenance or {}).get("qualitative") or {}
        # delivery→attention→decision accounting (§27)
        for it in bundle[:8]:
            if isinstance(it.get("available_ts"), (int, float)):
                stats.attention_to_decision_s.append(
                    max(0.0, world.clock.now - float(it["available_ts"])))
        if qual.get("act_or_wait") == "wait" or selected.action_name in ("wait", "abstain"):
            self.report["actors_declined_to_act"] += 1
            delta.reason_codes.append("actor_considered_no_action_warranted")
            delta.uncertainty["decision_summary"] = str(qual.get("decision_summary", ""))[:200]
            # §11: a DEFERRAL with stated timing/condition compiles to a real trigger — never
            # an automatic retry timer
            fu = compile_actor_deferral(world, actor_id, qual, sev, trigger)
            if fu:
                delta.follow_up_events = [fu]
                delta.reason_codes.append("deferral_compiled_to_real_trigger")
            return delta, ValidationResult(ok=True)
        exec_delta, _events = self.runtime.execute(world, selected, posterior, trace,
                                                   seed=seed)
        self.report["actor_actions_executed"] += 1
        # ride the execution's world changes + follow-ups on THIS control step's delta
        delta.changes.extend(exec_delta.changes)
        delta.reason_codes.extend(exec_delta.reason_codes[:6])
        delta.uncertainty["executed_action"] = selected.action_name[:80]
        delta.uncertainty["trigger_type"] = trigger.get("trigger_type")
        delta.follow_up_events = list(exec_delta.follow_up_events)
        return delta, ValidationResult(ok=True)


def compile_actor_deferral(world, actor_id: str, qual: dict, sev: dict, trigger: dict):
    """§11: an actor who chose to wait may have stated WHEN or ON WHAT CONDITION they will
    revisit. Compile that intent into a real event:
      * calendar expression ("tomorrow_morning", "end_of_day", "next_business_day") → exact tz-
        aware timestamp in the ACTOR's calendar;
      * an explicit revisit time → exact timestamp;
      * a condition ("when they reply", "after the vote") → a registered conditional trigger
        that fires when the condition's event occurs;
      * no stated intent, or unresolvable intent → NOTHING is scheduled (deliberate inaction is
        a real outcome; the runtime keeps unresolved intent as an unresolved timing mechanism).
    NEVER an automatic fixed-delay retry."""
    from swm.world_model_v2.temporal_runtime import (get_stats, make_trigger,
                                                     register_conditional,
                                                     resolve_timing_spec, temporal_model_of)
    model = temporal_model_of(world)
    stats = get_stats(world)
    intent = qual.get("revisit") if isinstance(qual.get("revisit"), dict) else {}
    timing_label = str(intent.get("when", "") or qual.get("timing", "")
                       or intent.get("timing", "")).strip().lower()
    cond = intent.get("condition") or qual.get("timing_condition")
    if isinstance(cond, str) and cond.strip():
        # a natural-language condition without a mappable event type stays UNRESOLVED (§11) —
        # a structured condition {etype, participant} registers a real watcher
        stats.unresolved_timing.append({"kind": "deferral_condition_text",
                                        "actor": actor_id, "condition": cond[:120]})
        return None
    if isinstance(cond, dict) and cond.get("etype"):
        register_conditional(world, condition={"etype": str(cond["etype"]),
                                               "participant": str(cond.get("participant", ""))},
                             event_spec={"etype": "ctrl_invoke_actor",
                                         "participants": [actor_id],
                                         "payload": {"actor_id": actor_id,
                                                     "triggering_semantic_event": sev,
                                                     "reason_actor_may_be_causally_relevant":
                                                         "the condition the actor was waiting "
                                                         "for occurred",
                                                     "cascade_depth":
                                                         int(sev.get("cascade_depth", 0)) + 1}},
                             actor_id=actor_id, provenance="actor_deferral_condition")
        return None                                            # fires via the conditional watcher
    revisit_spec = intent.get("at") if isinstance(intent.get("at"), dict) else None
    if revisit_spec is None and timing_label in ("tomorrow_morning", "end_of_day",
                                                 "this_evening", "next_business_day"):
        revisit_spec = {"kind": "calendar", "calendar_expr": timing_label,
                        "calendar_of": actor_id}
    if revisit_spec is None:
        return None                                            # plain inaction — nothing scheduled
    ts = resolve_timing_spec(revisit_spec, world=world, model=model, ref_ts=world.clock.now,
                             calendar_of=actor_id, salt=f"revisit:{actor_id}", stats=stats)
    if ts is None or ts <= world.clock.now:
        return None                                            # unresolved stays unresolved
    return {"etype": "ctrl_invoke_actor", "ts": float(ts), "participants": [actor_id],
            "payload": {"actor_id": actor_id, "triggering_semantic_event": sev,
                        "reason_actor_may_be_causally_relevant":
                            "the actor deferred and chose this revisit time",
                        "cascade_depth": int(sev.get("cascade_depth", 0)) + 1,
                        "trigger": make_trigger(
                            trigger_type="self_scheduled_revisit", actor_id=actor_id,
                            parents=[str(sev.get("event_id", ""))],
                            observed="own earlier deferral",
                            why_now="the actor's own chosen revisit time arrived",
                            provenance="actor_deferral")},
            "trigger": {"trigger_type": "self_scheduled_revisit"}}


# ---------------------------------------------------------------- institutions (arithmetic
# only — the engine counts actual actor choices, it never chooses them)
def run_institutional_aggregation(world, inst_id: str, *, matter_record_id: str = "",
                                  delta: StateDelta = None, report: dict = None) -> dict:
    schema = _schema(world)
    inst = (schema.institutional_definitions or {}).get(inst_id)
    if inst is None:
        return {"ok": False, "reason": f"institution {inst_id!r} not in schema"}
    holders = [str(h) for h in inst.get("decision_holders") or []]
    drt = str(inst.get("decision_record_type", ""))
    decisions = {}
    for rec in world.objects.values():
        if rec.object_type == drt and rec.created_by in holders \
                and (not matter_record_id
                     or str(rec.attributes.get("matter", "")) == matter_record_id):
            decisions[rec.created_by] = str(rec.attributes.get("position",
                                                               rec.status)).lower()
    agg = inst.get("aggregation") or {}
    kind = str(agg.get("kind", "majority"))
    yes = sum(1 for v in decisions.values() if v in ("yes", "approve", "for", "support",
                                                     "aye", "in_favor"))
    no = sum(1 for v in decisions.values() if v in ("no", "reject", "against", "oppose",
                                                    "nay"))
    n, total = len(decisions), len(holders)
    if kind == "single_authority":
        passed = yes >= 1
    elif kind == "unanimous":
        passed = n == total and yes == total
    elif kind == "threshold":
        passed = yes >= int(agg.get("threshold", max(1, total // 2 + 1)))
    elif kind == "quorum_majority":
        quorum = int(agg.get("threshold", max(1, total // 2 + 1)))
        passed = n >= quorum and yes > no
    else:                                   # majority of votes actually cast
        passed = n > 0 and yes > no
    result = {"ok": True, "institution": inst_id, "kind": kind, "passed": bool(passed),
              "yes": yes, "no": no, "cast": n, "holders": total,
              "decisions": decisions, "note": "engine counted actual actor decision records; "
                                              "it chose none of them"}
    if delta is not None:
        delta.change(f"institutional_result[{inst_id}]", None,
                     {"passed": result["passed"], "yes": yes, "no": no, "cast": n})
    return result


# ---------------------------------------------------------------- outcome predicates
def make_generated_predicate_readout(schema: ScenarioSemanticModel):
    """Contract readout over the branch's generated records: the FROZEN predicates resolve
    the question from the evolved world."""
    preds = list(schema.outcome_predicates or [])

    def readout(world):
        for p in preds:
            if evaluate_predicate(p, list(getattr(world, "objects", {}).values())):
                return str(p.get("option_true", "True"))
        return str(preds[0].get("option_false", "False")) if preds else "False"

    return readout
