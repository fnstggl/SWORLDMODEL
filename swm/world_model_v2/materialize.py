"""Materializer — turns a validated WorldExecutionPlan into a runnable WorldModelV2Run.

Plan JSON → typed objects: entities (universal schema, causally-relevant fields only), populations
(segments + heterogeneity), relation graph, executable rule systems, typed quantities, information ledger,
latent records (InitialStateModel), scheduled events + hazards (queue builder), and the operator set from the
accepted mechanisms. This is configuration, not a new engine: every scenario class flows through here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from swm.world_model_v2.contracts import OutcomeContract
from swm.world_model_v2.events import Event, EventQueue, StochasticHazard
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.init_state import InitialStateModel
from swm.world_model_v2.institutions import Rule, RuleSystem
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.population import Population, PopulationSegment
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.rollout import WorldModelV2Run
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import get_operator


def build_world(plan, *, world_id: str = "w0", evidence_hash: str = "", versions: dict = None) -> WorldState:
    clock = SimulationClock(now=plan.as_of, as_of=plan.as_of)
    w = WorldState(world_id=world_id, branch_id="root", clock=clock,
                   network=RelationGraph(), information=InformationLedger(),
                   evidence_hash=evidence_hash, versions=versions or {})
    for e in plan.entities:
        ent = Entity(identity=str(e.get("id")), entity_type=str(e.get("type", "person")))
        for fname, val in (e.get("fields") or {}).items():
            if val in ("?", None, ""):
                continue                                     # unknowns stay latent, not fabricated
            try:
                ent.set(fname, F(val, status="observed", method="compiler:evidence",
                                 updated_at=plan.as_of))
            except KeyError:
                continue                                     # non-schema field proposed → dropped, not smuggled
        w.entities[ent.identity] = ent
    for p in plan.populations:
        segs = []
        for s in (p.get("segments") or []):
            segs.append(PopulationSegment(
                segment_id=str(s.get("id")), weight=F(float(s.get("weight", 0.0) or 0.0)),
                heterogeneity={str(d): F(None, dist={"mean": 0.5, "sd": 0.2, "lo": 0.0, "hi": 1.0},
                                         status="assumed") for d in (s.get("differs_on") or [])}))
        w.populations[str(p.get("id"))] = Population(population_id=str(p.get("id")), segments=segs)
    for r in plan.relations:
        try:
            w.network.add(str(r.get("src")), str(r.get("rel")), str(r.get("dst")))
        except KeyError:
            continue                                         # unregistered relation → dropped loudly at compile
    for inst in plan.institutions:
        rules = [Rule(rule_id=f"{inst.get('id')}:{i}", kind=str(ru.get("kind", "procedure")),
                      params=dict(ru.get("params") or {}))
                 for i, ru in enumerate(inst.get("rules") or [])]
        w.institutions[str(inst.get("id"))] = RuleSystem(institution_id=str(inst.get("id")), rules=rules)
    for q in plan.quantities:
        name, qtype = str(q.get("name")), str(q.get("qtype", q.get("name")))
        register_quantity_type(qtype, units=str(q.get("units", "unit")))
        try:
            w.quantities[name] = Quantity(name=name, qtype=qtype,
                                          value=q.get("value"), sd=q.get("sd"), timestamp=plan.as_of)
        except KeyError:
            continue
    return w


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


def operators_from_plan(plan, *, llm=None, allow_experimental=False) -> list:
    ops, seen = [], set()
    for m in plan.accepted_mechanisms:
        opname = m.get("operator")
        if not opname or opname in seen:
            continue
        seen.add(opname)
        cls = get_operator(opname, allow_experimental=allow_experimental)
        ops.append(cls(llm=llm) if opname == "agent_decision" else cls())
    return ops


def run_from_plan(plan, *, llm=None, n_particles=None, seed=0):
    """The end-to-end: plan → world → InitialStateModel → rollout → native terminal distribution."""
    base = build_world(plan)
    init = InitialStateModel(base_world=base, latents=list(plan.latents))
    run = WorldModelV2Run(initial=init, queue_builder=queue_builder_from_plan(plan),
                          operators=operators_from_plan(plan, llm=llm),
                          contract=plan.outcome_contract,
                          n_particles=n_particles or plan.compute_plan.get("n_particles", 30))
    return run.run(seed=seed)
