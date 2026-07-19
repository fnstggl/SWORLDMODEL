"""Shared offline fixtures for the scenario-generated action layer.

Builds complete generated-world contexts (schema + entities + ledger + network + control-
plane operators + scripted actor runtime + queue builder) with PARAMETRIZABLE names, so
cross-domain and randomized tests can generate scenario vocabularies at test time that
cannot have been memorized from source literals. No LLM anywhere; actor reactions are
scripted through the SAME kernel ops production reactions use.
"""
from __future__ import annotations

import copy

from swm.world_model_v2.events import EventQueue
from swm.world_model_v2.generated_world import (GeneratedObservationDeliveryOperator,
                                                GeneratedSemanticEventOperator,
                                                execute_kernel_ops, generated_report)
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase13.scenario_actions.execution import ScenarioPlanOperator
from swm.world_model_v2.scenario_schema import ScenarioSemanticModel
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta, ValidationResult

T0 = 1_700_000_000.0
DAY = 86400.0


def council_schema(*, maker="rivera", officer="chen", panel="zoning_panel",
                   record="variance_petition", decision_record="panel_member_decision",
                   event="petition_filed_notice", outcome_record="variance_grant",
                   horizon_days=45):
    """A municipal zoning-variance scenario, fully parameterizable so randomized tests can
    rename every type at test time. Carries its OWN causal mechanism (the clerk's posting
    board) — the CAUSAL TRUTH BOUNDARY means the filing notice reaches nobody's eyes until
    that mechanism actually posts it, and the grant record is scenario-declared as the
    officer's own act (`controlled_by`), never the petitioner's claim."""
    return ScenarioSemanticModel(
        question=f"Will {maker} obtain the variance?",
        prediction_timestamp=T0, horizon=T0 + horizon_days * DAY,
        entity_types={"resident": {"description": "person", "fields": {"name": "str"}}},
        fact_types={
            record: {"description": "the petition", "fields":
                     {"parcel": "str", "request": "str", "status": "str", "matter": "str"}},
            decision_record: {"description": "one panel member's decision",
                              "fields": {"position": "str", "matter": "str"}},
            outcome_record: {"description": "the granted variance — the officer's own "
                                            "issued instrument",
                             "controlled_by": officer,
                             "fields": {"parcel": "str", "status": "str"}}},
        semantic_event_types={
            event: {"description": "ATTEMPT: notice handed to the clerk",
                    "fields": {"parcel": "str"}, "typical_visibility": "public"},
            f"{event}_posted": {"description": "the clerk's board made the notice available",
                                "fields": {"parcel": "str"},
                                "typical_visibility": "public"},
            "neighbor_objection_sent": {"description": "an objection",
                                        "fields": {"reason": "str"},
                                        "typical_visibility": "participants"}},
        institutional_definitions={
            panel: {"procedure": "votes on petitions", "decision_holders": [officer],
                    "decision_record_type": decision_record,
                    "aggregation": {"kind": "single_authority"}, "assumed": True}},
        resource_definitions={"filing_credits": {"unit": "credits", "conserved": True}},
        actor_roles={maker: {"role": "petitioner", "why_consequential": "decides",
                             "affordances": ["file the petition", "withdraw"]},
                     officer: {"role": "panel officer", "why_consequential": "votes",
                               "affordances": ["approve", "reject"]}},
        mechanism_definitions={
            "clerk_posting_board": {
                "description": "the municipal clerk posting filed notices for public view",
                "triggering_event_types": [event],
                "accepted_inputs": {"parcel": "str"},
                "controlling_actor_or_system": "municipal_clerk_office",
                "state_machine": {"at_clerk_window": ["posted"]},
                "initial_state": "at_clerk_window",
                "success_states": ["posted"], "failure_states": ["misfiled"],
                "unresolved_states": [],
                "possible_output_event_types": {"on_success": [f"{event}_posted"],
                                                "on_failure": []},
                "observation_rules": {"recipients": "direct_targets",
                                      "availability": "public",
                                      "representation": "complete"},
                "timing_rules": {"delay_s": 60.0},
                "assumptions": ["the clerk posts same-day"],
                "uncertainty_source": "clerk workload"}},
        outcome_predicates=[{"predicate_id": "granted", "record_type": outcome_record,
                             "op": "exists", "option_true": "granted",
                             "option_false": "not_granted"}],
        information_rules={"default_channel": "clerk_window", "default_delay_s": 60.0},
        provenance={"compiler": "test"}).freeze()


def build_world(schema, actors, *, maker_resources=None, branch="b0"):
    w = WorldState("gen", branch, SimulationClock(T0, T0),
                   network=RelationGraph(), information=InformationLedger())
    w.horizon = float(schema.horizon or T0 + 30 * DAY)
    for name in actors:
        e = Entity(name)
        e.set("roles", F(["person"], status="observed"))
        for rname, amount in (maker_resources or {}).items():
            e.set("resources", F(float(amount), status="observed"), key=rname)
        e.set("past_actions", F([], status="observed"))
        w.entities[name] = e
    for a in actors[1:]:
        w.network.add(actors[0], "communicates_with", a)
    w.scenario_schema = copy.deepcopy(schema)
    return w


class ScriptedActorRuntime:
    """Offline stand-in for the qualitative actor runtime: `script` maps actor_id ->
    callable(world, situation) -> list of kernel ops (their REACTION, through the same
    kernel production uses) or None for deliberate inaction."""

    def __init__(self, script=None):
        self.script = dict(script or {})
        self.invocations = []

    def decide(self, _store, worlds, actor_id, *, decision=None, seed=0, observed_events=None):
        self.invocations.append((actor_id, (decision or {}).get("situation", "")))
        fn = self.script.get(actor_id)
        situation = (decision or {}).get("situation", "")
        ops = fn(worlds[0], situation) if fn is not None else None

        class _Sel:
            action_name = "react" if ops else "wait"

        class _Post:
            provenance = {"qualitative": {"act_or_wait": "act" if ops else "wait",
                                          "decision_summary": situation[:80]}}

        return _Sel(), _Post(), {"ops": ops}

    def execute(self, world, selected, posterior, trace, *, seed=0):
        d = StateDelta(at=world.clock.now, event_type="actor_reaction",
                       operator="scripted_actor")
        ctx = {"actor_id": "scripted", "action_id": "reaction", "now": world.clock.now,
               "report": generated_report(), "events": [], "quarantined": [],
               "compiler": "scripted"}
        ops = (trace or {}).get("ops") or []
        for op in ops:
            ctx["actor_id"] = op.pop("_actor", ctx["actor_id"])
        events = execute_kernel_ops(world, ops, ctx, d)
        d.follow_up_events = [{"etype": e.etype, "ts": e.ts,
                               "participants": list(e.participants),
                               "payload": dict(e.payload)} for e in events]
        return d, events


class ScriptedInvocationOperator:
    """ctrl_invoke_actor for scripted worlds: rebuilds nothing, invokes the scripted actor,
    rides its kernel-ops reaction on this delta — same shape as production."""

    name = "scripted_actor_invocation"

    def __init__(self, runtime: ScriptedActorRuntime, report=None):
        self.runtime = runtime
        self.report = report if report is not None else generated_report()

    def applicable(self, world, event):
        return event.etype == "ctrl_invoke_actor" and bool(event.participants)

    def run(self, world, event, rng):
        actor_id = str(event.payload.get("actor_id", event.participants[0]))
        sev = event.payload.get("triggering_semantic_event") or {}
        situation = f"{sev.get('semantic_type_id', '')}: {sev.get('exact_content', '')[:200]}"
        delta = StateDelta(at=world.clock.now, event_type="ctrl_invoke_actor",
                           operator=self.name, reason_codes=[f"reconsider:{actor_id}"])
        selected, posterior, trace = self.runtime.decide(
            None, [world], actor_id, decision={"situation": situation}, seed=0)
        self.report["actors_invoked"] += 1
        if selected.action_name == "wait":
            self.report["actors_declined_to_act"] += 1
            delta.reason_codes.append("actor_considered_no_action_warranted")
            return delta, ValidationResult(ok=True)
        ops = (trace or {}).get("ops") or []
        ctx = {"actor_id": actor_id, "action_id": f"react_{actor_id}",
               "now": world.clock.now, "report": self.report, "events": [],
               "quarantined": [], "compiler": "scripted"}
        events = execute_kernel_ops(world, [dict(o) for o in ops], ctx, delta)
        self.report["actor_actions_executed"] += 1
        delta.follow_up_events = [{"etype": e.etype, "ts": e.ts,
                                   "participants": list(e.participants),
                                   "payload": dict(e.payload)} for e in events]
        return delta, ValidationResult(ok=True)


class FixtureInitial:
    """InitialStateModel-compatible: n independent particle worlds (optionally varied by a
    per-particle hook, e.g. to realize different structural hypotheses)."""

    def __init__(self, schema, actors, *, maker_resources=None, vary=None):
        self.schema = schema
        self.actors = list(actors)
        self.maker_resources = dict(maker_resources or {})
        self.vary = vary

    def sample_particles(self, n, seed=0):
        worlds = []
        for i in range(n):
            w = build_world(self.schema, self.actors, maker_resources=self.maker_resources,
                            branch=f"p{i}")
            if self.vary is not None:
                self.vary(w, i)
            worlds.append(w)
        return worlds


def build_context(schema, actors, *, script=None, maker_resources=None, n_particles=6,
                  hypotheses=None, vary=None, report=None):
    """A complete generated-mode world_context dict for the Phase 13 evaluator — including
    the causal-boundary mechanism runtime, exactly like production `operators_from_plan`."""
    from swm.world_model_v2.causal_boundary import (MechanismRuntimeOperator,
                                                    ScheduledAttemptOperator)
    rep = report if report is not None else generated_report()
    runtime = ScriptedActorRuntime(script)
    operators = [ScenarioPlanOperator(report=rep),
                 GeneratedSemanticEventOperator(report=rep),
                 GeneratedObservationDeliveryOperator(report=rep),
                 ScriptedInvocationOperator(runtime, report=rep),
                 MechanismRuntimeOperator(report=rep),
                 ScheduledAttemptOperator(report=rep)]
    return {"initial": FixtureInitial(schema, actors, maker_resources=maker_resources,
                                      vary=vary),
            "queue_builder": lambda w: EventQueue(horizon_ts=float(
                getattr(w, "horizon", w.clock.now + 30 * DAY))),
            "operators": operators, "contract": None, "n_particles": n_particles,
            "hypotheses": hypotheses or [], "max_events": 200}, rep, runtime
