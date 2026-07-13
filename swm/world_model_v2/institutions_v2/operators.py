"""Phase 10 — shared-world institutional execution (Part 18).

InstitutionRuntime binds a real TEMPLATE (evidence-backed, as-of-versioned) + the executable engines
(authority, stage, decision, queue) + a scenario INSTANCE. InstitutionOperator runs it inside the Phase-1
WorldState: an incoming institutional_action event is AUTHORIZED or BLOCKED, the stage/decision state
changes, an explicit StateDelta is emitted, future procedural events are scheduled, and terminal outcomes
change. An unauthorized action never mutates state (Part 5) — it is blocked, recorded, and explained.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.institutions_v2.authority import AuthorityGraph, InformationBoundary
from swm.world_model_v2.institutions_v2.decisions import ThresholdSpec, evaluate_decision
from swm.world_model_v2.institutions_v2.procedure import StageEngine
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            register_operator)


@dataclass
class InstitutionRuntime:
    """A ready-to-execute institution: template + engines + scenario instance. Constructed by the compiler
    (compile.py) or a testbed; carried on the event payload so the operator can run it."""
    template: object
    instance: object
    as_of: str = ""
    authority: AuthorityGraph = None
    stages: StageEngine = None
    info: InformationBoundary = None
    thresholds: dict = field(default_factory=dict)      # {decision: ThresholdSpec}
    blocked_actions: list = field(default_factory=list)

    def __post_init__(self):
        if self.authority is None:
            self.authority = AuthorityGraph(edges=list(self.template.authority))
        if self.stages is None:
            self.stages = StageEngine.from_stages(self.template.stages)
        if self.info is None:
            self.info = InformationBoundary(rights=self._info_rights())

    def _info_rights(self):
        rights = {}
        fam_rights = getattr(self.template, "information_rights", None)
        return fam_rights or rights


class InstitutionOperator(TransitionOperator):
    """Executes one institutional action against the runtime. Payload:
      {institution: InstitutionRuntime, action: {actor, type, matter?, subject?}, decision?: {...}}
    Authorization is checked FIRST; a blocked action produces a StateDelta that records the block and
    mutates nothing. An authorized stage/decision action advances the matter, writes the outcome quantity,
    and schedules the next procedural event."""
    name = "institution_action"

    def applicable(self, world, event):
        return event.etype == "institutional_action" and isinstance(
            event.payload.get("institution"), InstitutionRuntime)

    def propose(self, world, event, rng):
        rt = event.payload["institution"]
        action = dict(event.payload.get("action") or {})
        stage_permits = rt.stages.permitted_actions(rt.instance.current_stage)
        ok, reason = rt.authority.authorize(rt.instance, action, stage_permits=stage_permits)
        return TransitionProposal(operator=self.name, action={
            "authorized": ok, "reason": reason, "action": action,
            "decision": event.payload.get("decision"), "outcome_var": event.payload.get("outcome_var")},
            reason_codes=[f"authorized={ok}", action.get("type", "")])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        rt = self._rt
        a = proposal.action
        action = a["action"]
        d = StateDelta(at=world.clock.now, event_type="institutional_action", operator=self.name,
                       reason_codes=proposal.reason_codes)
        if not a["authorized"]:
            # BLOCK: record, mutate nothing (Part 5 — invalid action cannot execute)
            rt.blocked_actions.append({"action": action, "reason": a["reason"], "at": world.clock.now})
            d.uncertainty = {"blocked": True, "reason": a["reason"]}
            d.reason_codes = d.reason_codes + ["blocked_invalid_action"]
            return d

        matter = rt.instance.matter
        stage_before = rt.instance.current_stage
        outcome = action.get("type", "")

        # a decision action runs the threshold engine on real votes
        if a.get("decision"):
            dec = a["decision"]
            spec = rt.thresholds.get(dec.get("decision_id")) or ThresholdSpec(**dec.get("spec", {"kind": "simple_majority"}))
            res = evaluate_decision(spec, dec.get("votes", {}), eligible=dec.get("eligible", []),
                                    weights=dec.get("weights"), recused=set(dec.get("recused", [])),
                                    tie_break=dec.get("tie_break"))
            outcome = "passed" if res.passed else "failed"
            d.uncertainty = {"decision": res.as_dict()}
            matter["last_decision"] = res.as_dict()

        # advance the stage graph
        nxt = rt.stages.next_stage(stage_before, outcome)
        rt.instance.current_stage = nxt if nxt else stage_before
        if isinstance(matter, dict):
            matter["stage"] = rt.instance.current_stage
        d.change(f"institution[{rt.template.template_id}].stage", stage_before, rt.instance.current_stage)

        # terminal outcome projection
        var = a.get("outcome_var")
        if var and (rt.stages.is_terminal(rt.instance.current_stage) or nxt is None):
            register_quantity_type(var, units="institutional_outcome")
            before = world.quantities[var].value if var in world.quantities else None
            val = outcome
            world.quantities[var] = Quantity(name=var, qtype=var, value=val, timestamp=world.clock.now)
            d.change(f"quantities[{var}]", before, val)

        # schedule the next procedural event (future-event generation)
        if nxt and not rt.stages.is_terminal(nxt):
            permits = rt.stages.permitted_actions(nxt)
            if permits:
                d.follow_up_events = [{"etype": "institutional_action", "ts": world.clock.now + 86400.0,
                                       "participants": [], "payload": {"stage": nxt}}]
        return d

    def run(self, world, event, rng):
        self._rt = event.payload["institution"]
        return super().run(world, event, rng)


register_operator("institution_action", InstitutionOperator(), requires=("entities",),
                  modifies=("quantities", "institutions"), temporal_scale="event",
                  parameter_source="evidence-backed institutional template (authority/stage/threshold rules)",
                  validated=True)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
if not event_type_registered("institutional_action"):
    register_event_type("institutional_action", scheduling="scheduled", reads=("entities", "institutions"),
                        deltas=("quantities", "institutions"),
                        parameter_source="institutional template rules", validated=True)
