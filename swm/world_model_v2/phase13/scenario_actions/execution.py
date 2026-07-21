"""Plan execution through the canonical runtime — one operator, generated kernel semantics.

A compiled candidate enters a matched arm as an Intervention that only SCHEDULES
`scenario_plan_step` events (queue surgery, never state surgery). When a step event fires,
`ScenarioPlanOperator`:

  1. re-checks stop conditions and the step's observation-predicates against the
     decision-maker's OBSERVABLE projection (visible records + delivered information —
     structurally never hidden state);
  2. re-validates state-dependent feasibility live (resources, authority — the kernel's own
     checks run again on the branch world);
  3. applies the step's PRE-COMPILED kernel ops via `execute_kernel_ops` — the exact program
     fixed at compile time, identical across matched worlds;
  4. queues the resulting control-plane events, so observation routing, causal-frontier
     discovery, and affected actors' own reconsiderations happen through the SAME generated
     control plane as every other event in the world;
  5. schedules dependent steps as follow-ups, records step failures/lapses loudly, and never
     converts an infeasible step into a silent no-op.

There is no second executor: the decision-maker's steps and other actors' reactions both
resolve through the generated-world kernel.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from swm.world_model_v2.contracts import Intervention
from swm.world_model_v2.events import Event, event_type_registered, register_event_type
from swm.world_model_v2.generated_world import execute_kernel_ops
from swm.world_model_v2.phase13.scenario_actions.candidates import (ConcreteAction,
                                                                    ConditionSpec, PlanStep)
from swm.world_model_v2.transitions import StateDelta, ValidationResult

PLAN_STEP_ETYPE = "scenario_plan_step"
if not event_type_registered(PLAN_STEP_ETYPE):
    register_event_type(PLAN_STEP_ETYPE, scheduling="scheduled", validated=True,
                        parameter_source="phase13 scenario action layer")

# Conditional steps are CHANGE-TRIGGERED, not polled (§14): an unmet guard registers a state
# watch and the step re-checks when the observable state actually changes; dependent steps
# fire the INSTANT their parents complete (same timestamp, next causal microstep) — no
# synthetic re-check interval, no synthetic inter-step gap.


# ---------------------------------------------------------------- observable projection
def observable_projection(world, actor_id: str) -> dict:
    """Everything the decision-maker can legitimately condition on in THIS branch: their
    visible records, information delivered to them, their own resources, and the clock.
    Contingent conditions evaluate against THIS dict only."""
    records = []
    for o in (getattr(world, "objects", {}) or {}).values():
        if o.visible_to(actor_id):
            records.append({"record_id": o.object_id, "record_type": o.object_type,
                            "status": o.status, "fields": dict(o.attributes),
                            "updated_at": o.updated_at})
    observations = []
    info = getattr(world, "information", None)
    if info is not None:
        try:
            for item, exp in info.visible_to(actor_id, at=world.clock.now):
                observations.append({"item_id": getattr(item, "item_id", ""),
                                     "content": str(getattr(item, "content", "")),
                                     "source": getattr(item, "source", ""),
                                     "observed_at": getattr(exp, "at", None)})
        except Exception:  # noqa: BLE001 — no ledger => no observations, never omniscience
            pass
    # an actor knows what they THEMSELVES did: their own emitted semantic events are
    # observations (delivery routes to recipients only, so without this a step conditioned
    # on the maker's own prior announcement could never fire)
    for sev in (getattr(world, "semantic_log", []) or []):
        if str(sev.get("source_actor_id", "")) == actor_id:
            observations.append({"item_id": str(sev.get("event_id", "")),
                                 "content": str(sev.get("exact_content", "")),
                                 "source": actor_id,
                                 "observed_at": sev.get("occurred_at")})
    resources = {}
    ent = (getattr(world, "entities", {}) or {}).get(actor_id)
    if ent is not None:
        res = ent.get("resources")
        if isinstance(res, dict):
            for k, sf in res.items():
                if isinstance(getattr(sf, "value", None), (int, float)):
                    resources[str(k)] = float(sf.value)
    return {"actor": actor_id, "t": float(world.clock.now), "records": records,
            "observations": observations, "resources": resources}


def condition_holds(cond, projection: dict) -> bool:
    """Evaluate ONE ConditionSpec against the observable projection. Unknown kinds fail
    closed (a condition that cannot be evaluated never fires its step)."""
    c = cond.as_dict() if isinstance(cond, ConditionSpec) else dict(cond or {})
    kind = str(c.get("kind", "record"))
    op = str(c.get("op", "exists"))
    value = c.get("value")

    def cmp(got):
        if op == "exists":
            return got is not None
        if op == "eq":
            return got == value
        if op == "ne":
            return got != value
        if op == "in":
            return got in (value or [])
        if op == "contains":
            return isinstance(got, (str, list)) and value is not None and value in got
        if op in ("gte", "lte") and isinstance(got, (int, float)):
            try:
                bound = float(value or 0)
            except (TypeError, ValueError):
                return False                     # non-numeric bound: fail closed, never crash
            return got >= bound if op == "gte" else got <= bound
        return False

    if kind == "record":
        rt, fieldname = str(c.get("record_type", "")), str(c.get("field", "") or "status")
        for r in projection["records"]:
            if r["record_type"] != rt:
                continue
            got = r["status"] if fieldname == "status" else r["fields"].get(fieldname)
            if op == "exists" or cmp(got):
                return True
        return False
    if kind == "information":
        needle = str(value or c.get("field", ""))
        return any(needle.lower() in str(o["content"]).lower()
                   for o in projection["observations"])
    if kind == "time":
        return cmp(projection["t"])
    if kind == "resource":
        return cmp(projection["resources"].get(str(c.get("field", ""))))
    return False


# ---------------------------------------------------------------- per-branch plan state
def _plan_state(world, candidate_id: str) -> dict:
    root = world.uncertainty_meta.setdefault("scenario_plan_exec", {})
    return root.setdefault(candidate_id, {"completed": [], "failed": [], "lapsed": [],
                                          "halted": False, "checks": {}})


# ---------------------------------------------------------------- the intervention
def plan_intervention(candidate: ConcreteAction, *, problem=None) -> Intervention:
    """Queue surgery only: schedule the candidate's ROOT steps (no unmet dependencies).
    Dependent steps ride follow-ups from completed parents; nothing mutates state here."""
    steps = list(candidate.steps)

    def apply(world, queue):
        if not steps:                                   # do_nothing / defer: change NOTHING
            return
        now = float(world.clock.now)
        for step in steps:
            if step.after_steps:
                continue
            ts = float(step.timing_ts) if step.timing_ts and step.timing_ts >= now else now
            queue.schedule(Event(
                ts=ts, etype=PLAN_STEP_ETYPE,
                participants=[candidate.actor_id] + [str(t) for t in step.target_ids][:8],
                payload={"candidate_id": candidate.candidate_id, "step_id": step.step_id,
                         "plan": candidate, "source_decision": True},
                source="phase13:scenario_plan"))

    return Intervention(intervention_id=candidate.candidate_id,
                        description=candidate.title[:120], apply=apply,
                        kind="policy" if candidate.is_contingent() else "discrete")


# ---------------------------------------------------------------- the operator
@dataclass
class StepOutcome:
    fired: bool
    reason: str = ""


class ScenarioPlanOperator:
    """Executes scenario_plan_step events for compiled ConcreteActions. One instance serves
    every particle of every arm — ALL execution state is branch-local (on the world)."""

    name = "scenario_plan_step"

    def __init__(self, *, report: dict = None):
        self.report = report if report is not None else {}
        self.report.setdefault("steps_fired", 0)
        self.report.setdefault("steps_failed_at_execution", 0)
        self.report.setdefault("steps_lapsed", 0)
        self.report.setdefault("plans_halted", 0)
        self.report.setdefault("fallback_reasons", [])

    def applicable(self, world, event) -> bool:
        return event.etype == PLAN_STEP_ETYPE and \
            isinstance(event.payload.get("plan"), ConcreteAction) and \
            getattr(world, "scenario_schema", None) is not None

    def run(self, world, event, rng):
        plan: ConcreteAction = event.payload["plan"]
        step_id = str(event.payload.get("step_id", ""))
        step = next((s for s in plan.steps if s.step_id == step_id), None)
        state = _plan_state(world, plan.candidate_id)
        delta = StateDelta(at=world.clock.now, event_type=PLAN_STEP_ETYPE, operator=self.name,
                           reason_codes=[f"plan:{plan.candidate_id}", f"step:{step_id}"])
        if step is None:
            return delta, ValidationResult(ok=False, reasons=[f"unknown step {step_id!r}"])
        if state["halted"] or step_id in state["completed"] or step_id in state["failed"]:
            delta.reason_codes.append("step_skipped:plan_halted_or_done")
            return delta, ValidationResult(ok=True)

        projection = observable_projection(world, plan.actor_id)
        # stop conditions: any holding => the plan halts, loudly. A stop rule stops IN-FLIGHT
        # activity: it is not evaluated before the first step has fired (an LLM-authored stop
        # condition that is true at t0 is an inversion, and aborting an unstarted plan on it
        # made whole candidates silently degenerate — ex2 forensic). The skip is recorded.
        started = bool(state["completed"] or state["failed"])
        for sc in plan.stop_conditions:
            if not started:
                if condition_holds(sc, projection):
                    delta.reason_codes.append("stop_condition_true_at_start_ignored")
                break
            if condition_holds(sc, projection):
                state["halted"] = True
                self.report["plans_halted"] += 1
                delta.reason_codes.append("plan_stop_condition_met")
                delta.uncertainty["stop_condition"] = json.dumps(
                    sc.as_dict() if isinstance(sc, ConditionSpec) else sc, default=str)[:200]
                return delta, ValidationResult(ok=True)
        # step guards: unmet => CHANGE-TRIGGERED re-check (a state watch fires the step when
        # the observable world actually changes), bounded, then a RECORDED lapse (never silent,
        # never a fixed polling interval)
        unmet = [c for c in step.conditions if not condition_holds(c, projection)]
        if unmet:
            checks = state["checks"].get(step_id, 0) + 1
            state["checks"][step_id] = checks
            if checks <= int(step.max_condition_checks):
                from swm.world_model_v2.temporal_runtime import (get_stats,
                                                                 register_state_watch)
                delta.reason_codes.append(
                    f"step_condition_unmet:watching_state_change_{checks}")
                register_state_watch(
                    world,
                    match_substrings=("objects[", ".resources[", "information",
                                      "quantities["),
                    event_spec={"etype": PLAN_STEP_ETYPE,
                                "participants": [plan.actor_id],
                                "payload": {"candidate_id": plan.candidate_id,
                                            "step_id": step_id, "plan": plan,
                                            "source_decision": True}},
                    max_fires=1, provenance="plan_step_condition_watch",
                    stats=get_stats(world))
            else:
                state["lapsed"].append(step_id)
                self.report["steps_lapsed"] += 1
                delta.reason_codes.append("step_lapsed:conditions_never_held")
            return delta, ValidationResult(ok=True)

        # live feasibility re-check: resource commitments against THIS branch's holdings
        for rname, amt in (step.resource_commitments or {}).items():
            have = projection["resources"].get(str(rname), 0.0)
            if have < float(amt):
                state["failed"].append(step_id)
                self.report["steps_failed_at_execution"] += 1
                delta.reason_codes.append(
                    f"step_infeasible_at_execution:insufficient_{rname}")
                self.report["fallback_reasons"].append(
                    {"kind": "step_infeasible_at_execution", "candidate": plan.candidate_id,
                     "step": step_id, "reason": f"insufficient {rname} ({have} < {amt})"})
                if plan.fallback == "halt_plan":
                    state["halted"] = True
                    self.report["plans_halted"] += 1
                return delta, ValidationResult(ok=True)

        # apply the PRE-COMPILED kernel ops — the same program on every matched world
        ctx = {"actor_id": plan.actor_id, "action_id": f"{plan.candidate_id}:{step_id}",
               "now": world.clock.now, "report": _kernel_report_view(self.report),
               "compiler": step.compile_meta.get("compiler", "precompiled"),
               "events": [], "quarantined": [], "cascade_depth": 0,
               "parent_event_ids": [str(event.payload.get("parent_event_id", ""))]}
        follow = execute_kernel_ops(world, list(step.compiled_ops), ctx, delta)
        if ctx["quarantined"] and len(ctx["quarantined"]) >= len(step.compiled_ops):
            state["failed"].append(step_id)
            self.report["steps_failed_at_execution"] += 1
            delta.reason_codes.append("step_failed_at_execution:all_ops_quarantined")
            self.report["fallback_reasons"].append(
                {"kind": "step_ops_quarantined", "candidate": plan.candidate_id,
                 "step": step_id,
                 "reasons": [q.get("reason", "")[:80] for q in ctx["quarantined"][:3]]})
            if plan.fallback == "halt_plan":
                state["halted"] = True
                self.report["plans_halted"] += 1
            return delta, ValidationResult(ok=True)

        state["completed"].append(step_id)
        self.report["steps_fired"] += 1
        delta.uncertainty["step_intent"] = step.intent[:120]
        fu = [{"etype": e.etype, "ts": e.ts, "participants": list(e.participants),
               "payload": dict(e.payload)} for e in follow]
        # dependent steps whose parents are now ALL complete
        done = set(state["completed"])
        for nxt in plan.steps:
            if nxt.step_id in done or nxt.step_id in state["failed"]:
                continue
            if nxt.after_steps and set(nxt.after_steps) <= done:
                # the dependent step's earliest start is its own declared timing, else the
                # INSTANT the dependency completes — same timestamp, next causal microstep
                # (the parent link orders it), no synthetic gap
                ts = float(nxt.timing_ts) if nxt.timing_ts and \
                    nxt.timing_ts > world.clock.now else world.clock.now
                fu.append({"etype": PLAN_STEP_ETYPE, "ts": ts,
                           "participants": [plan.actor_id],
                           "payload": {"candidate_id": plan.candidate_id,
                                       "step_id": nxt.step_id, "plan": plan,
                                       "source_decision": True},
                           "parent_ids": [event.event_id]})
        delta.follow_up_events = fu
        return delta, ValidationResult(ok=True)


class _KernelReportView(dict):
    """Adapter: execute_kernel_ops increments generated-world counter names; keep them on the
    shared plan report without requiring the full generated_report() shape."""


def _kernel_report_view(report: dict) -> dict:
    for k in ("scenario_events_emitted", "scenario_types_generated", "schema_extensions",
              "unsupported_semantics", "fallback_reasons"):
        report.setdefault(k, [] if k == "fallback_reasons" else 0)
    return report


def plan_execution_trace(world, candidate_id: str) -> dict:
    """The branch-local record of what the plan actually did here — read by trajectory
    diagnosis, never by the policy itself."""
    state = (world.uncertainty_meta.get("scenario_plan_exec") or {}).get(candidate_id) or {}
    return {"completed": list(state.get("completed", [])),
            "failed": list(state.get("failed", [])),
            "lapsed": list(state.get("lapsed", [])),
            "halted": bool(state.get("halted", False)),
            "condition_checks": dict(state.get("checks", {}))}
