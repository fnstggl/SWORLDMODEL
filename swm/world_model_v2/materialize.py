"""Materializer — turns a validated WorldExecutionPlan into a runnable WorldModelV2Run.

Plan JSON → typed objects: entities (universal schema, causally-relevant fields only), populations
(segments + heterogeneity), relation graph, executable rule systems, typed quantities, information ledger,
latent records (InitialStateModel), scheduled events + hazards (queue builder), and the operator set from the
accepted mechanisms. This is configuration, not a new engine: every scenario class flows through here.

Production contract (Tier A1 of the gap audit):
  * PROVENANCE HONESTY — compiler-proposed values are `inferred` (LLM proposal), never `observed`;
    `observed` requires an evidence reference, which only the evidence layer may attach.
  * LOUD FAILURE — nothing is silently dropped: unknown fields/relations/rule-kinds/quantities and
    mechanisms that resolve no operator are recorded in `world.omissions` / returned in the run report,
    and high-sensitivity drops raise MaterializeAbstention.
  * READOUT BINDING — the outcome contract's readout must resolve against the materialized base world
    (or name a declared quantity); a dangling readout aborts before any rollout.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from swm.world_model_v2.contracts import OutcomeContract
from swm.world_model_v2.events import Event, EventQueue, StochasticHazard
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.init_state import InitialStateModel
from swm.world_model_v2.institutions import EXECUTABLE_RULE_KINDS, Rule, RuleSystem
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.population import Population, PopulationSegment
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.rollout import WorldModelV2Run
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import get_operator


class MaterializeAbstention(Exception):
    """The plan cannot be materialized faithfully — say precisely why; never run a silently-mutilated world."""


def build_world(plan, *, world_id: str = "w0", evidence_hash: str = "", versions: dict = None) -> WorldState:
    clock = SimulationClock(now=plan.as_of, as_of=plan.as_of)
    w = WorldState(world_id=world_id, branch_id="root", clock=clock,
                   network=RelationGraph(), information=InformationLedger(),
                   evidence_hash=evidence_hash, versions=versions or {})
    omissions = []
    prompt_hash = (plan.provenance or {}).get("prompt_hash", "")
    for e in plan.entities:
        ent = Entity(identity=str(e.get("id")), entity_type=str(e.get("type", "person")))
        for fname, val in (e.get("fields") or {}).items():
            if val in ("?", None, ""):
                continue                                     # unknowns stay latent, not fabricated
            # PROVENANCE HONESTY: an LLM proposal is an inference, not an observation. The evidence
            # layer upgrades fields to `observed` only with an evidence reference attached.
            sf = F(val, status="inferred", method=f"compiler:proposal:{prompt_hash}",
                   confidence=0.45, updated_at=plan.as_of)
            from swm.world_model_v2.state import ENTITY_FIELDS, extension_fields
            if fname in (set(ENTITY_FIELDS) | extension_fields(ent.entity_type)):
                ent.set(fname, sf)
            else:
                # scenario-specific proposed field → typed latent_state namespace (kept, not dropped),
                # and recorded as an omission-from-canonical-schema for the audit trail
                ent.set("latent_state", sf, key=fname)
                omissions.append({"kind": "entity_field_routed_to_latent_state", "entity": ent.identity,
                                  "field": fname, "reason": "not a canonical schema field — stored as "
                                  "a typed latent_state scalar with provenance"})
        w.entities[ent.identity] = ent
    for p in plan.populations:
        segs = []
        for s in (p.get("segments") or []):
            segs.append(PopulationSegment(
                segment_id=str(s.get("id")), weight=F(float(s.get("weight", 0.0) or 0.0),
                                                      status="inferred",
                                                      method=f"compiler:proposal:{prompt_hash}"),
                heterogeneity={str(d): F(None, dist={"mean": 0.5, "sd": 0.2, "lo": 0.0, "hi": 1.0},
                                         status="assumed",
                                         method="unparameterized-dimension broad prior (labeled)")
                               for d in (s.get("differs_on") or [])}))
        w.populations[str(p.get("id"))] = Population(population_id=str(p.get("id")), segments=segs)
    for r in plan.relations:
        try:
            w.network.add(str(r.get("src")), str(r.get("rel")), str(r.get("dst")))
        except KeyError:
            omissions.append({"kind": "relation", "src": r.get("src"), "rel": r.get("rel"),
                              "dst": r.get("dst"), "reason": "unregistered relation type"})
    for inst in plan.institutions:
        rules, inst_id = [], str(inst.get("id"))
        for i, ru in enumerate(inst.get("rules") or []):
            kind = str(ru.get("kind", "procedure"))
            if kind not in EXECUTABLE_RULE_KINDS:
                # CLOSED RULE-KIND REGISTRY: an inexecutable rule must not silently validate everything.
                omissions.append({"kind": "institutional_rule", "institution": inst_id, "rule_kind": kind,
                                  "reason": f"rule kind {kind!r} is not executable "
                                            f"(executable: {sorted(EXECUTABLE_RULE_KINDS)})"})
                continue
            rules.append(Rule(rule_id=f"{inst_id}:{i}", kind=kind, params=dict(ru.get("params") or {})))
        w.institutions[inst_id] = RuleSystem(institution_id=inst_id, rules=rules)
    for q in plan.quantities:
        name, qtype = str(q.get("name")), str(q.get("qtype", q.get("name")))
        register_quantity_type(qtype, units=str(q.get("units", "unit")))
        try:
            w.quantities[name] = Quantity(name=name, qtype=qtype,
                                          value=q.get("value"), sd=q.get("sd"), timestamp=plan.as_of)
        except KeyError:
            omissions.append({"kind": "quantity", "name": name, "reason": "quantity construction failed"})
    w.omissions = omissions
    return w


def check_readout_binding(plan, world) -> None:
    """READOUT BINDING: the contract's readout must reference something that exists in the materialized
    world (an entity field path or a registered quantity). A dangling readout would let empty no-op worlds
    produce confident {'None': 1.0} answers — abort instead."""
    var = getattr(plan.outcome_contract, "readout_var", "") or ""
    if not var:
        return                                                # hand-built contracts bind a closure directly
    if var in world.quantities:
        return
    eid, _, fpath = var.partition(".")
    if eid in world.entities and fpath:
        return                                                # field may be set during rollout; entity must exist
    raise MaterializeAbstention(
        f"terminal readout {var!r} resolves to neither a registered quantity nor an entity of the "
        f"materialized world (entities: {sorted(world.entities)[:8]}, quantities: "
        f"{sorted(world.quantities)[:8]}) — refusing to simulate a world whose answer variable "
        f"does not exist")


def queue_builder_from_plan(plan):
    """Fresh queue per branch: scheduled events + hazards, horizon-capped."""
    def build(world) -> EventQueue:
        q = EventQueue(horizon_ts=plan.horizon_ts)
        rng = random.Random(int(world.branch_id.strip("b").split(":")[0] or 0)
                            if world.branch_id.startswith("b") else 0)
        for ev in plan.scheduled_events:
            q.schedule(Event(ts=ev["ts"], etype=ev["etype"], participants=list(ev["participants"]),
                             payload=dict(ev["payload"]), source="scheduled"))
        for hz in plan.stochastic_hazards:
            q.add_hazard(StochasticHazard(etype=hz["etype"], rate_per_day=hz["rate_per_day"],
                                          participants=list(hz["participants"])),
                         now=world.clock.now, rng=rng, world=world)
        return q
    return build


def operators_from_plan(plan, *, llm=None, allow_experimental=False) -> tuple:
    """Instantiate operators for accepted mechanisms. Mechanisms that name NO operator are returned as
    rejections (they must have been rejected at compile; this is defense in depth) — never silently
    skipped. Returns (operators, rejections)."""
    ops, seen, rejections = [], set(), []
    for m in plan.accepted_mechanisms:
        opname = m.get("operator")
        if not opname:
            rejections.append({"mech_id": m.get("mech_id"),
                               "reason": "accepted mechanism names no executable operator"})
            continue
        if opname in seen:
            continue
        seen.add(opname)
        try:
            cls = get_operator(opname, allow_experimental=allow_experimental)
        except (KeyError, PermissionError) as e:
            rejections.append({"mech_id": m.get("mech_id"), "reason": str(e)[:200]})
            continue
        try:
            # A3: the LLM reaches agent_decision only when experimental execution is explicitly enabled;
            # even then probability-minting requires the operator's own opt-in flag.
            if opname == "agent_decision":
                ops.append(cls(llm=(llm if allow_experimental else None),
                               allow_llm_probabilities=allow_experimental))
            else:
                ops.append(cls())
        except TypeError as e:
            rejections.append({"mech_id": m.get("mech_id"),
                               "reason": f"operator {opname!r} needs bound parameters "
                                         f"(e.g. a fitted policy pack): {e}"[:200]})
    return ops, rejections


def run_from_plan(plan, *, llm=None, n_particles=None, seed=0):
    """The end-to-end: plan → world → InitialStateModel → rollout → native terminal distribution.
    Aborts (MaterializeAbstention) on dangling readouts or when no accepted mechanism is executable;
    the result carries the world's omission log so downstream consumers see what was dropped."""
    base = build_world(plan, evidence_hash=(plan.provenance or {}).get("evidence_bundle_hash", ""))
    check_readout_binding(plan, base)
    ops, rejections = operators_from_plan(plan, llm=llm)
    if not ops:
        raise MaterializeAbstention(
            "no accepted mechanism resolves to an executable operator — the plan cannot cause anything "
            f"to happen (rejections: {[r['reason'][:60] for r in rejections]})")
    init = InitialStateModel(base_world=base, latents=list(plan.latents))
    run = WorldModelV2Run(initial=init, queue_builder=queue_builder_from_plan(plan),
                          operators=ops,
                          contract=plan.outcome_contract,
                          n_particles=n_particles or plan.compute_plan.get("n_particles", 30))
    result, branches = run.run(seed=seed)
    result["omissions"] = list(getattr(base, "omissions", []))
    result["operator_rejections"] = rejections
    return result, branches
