"""Phase 13 intervention semantics (Part 7) — action → intervention → Event → StateDelta, canonically.

The four-way distinction is enforced structurally:
  ACTION        — an `ontology.ActionSchema` the decision-maker chose;
  INTERVENTION  — its causal representation: a canonical `contracts.Intervention` whose apply() only
                  SCHEDULES a registered `decision_action` Event into the branch's queue (time-family
                  operations may also re-time existing queue events — still queue-level, never state);
  EVENT         — the queued `decision_action` Event the shared rollout pops like any other;
  STATE DELTA   — what the registered `DecisionActionOperator` emits when the event fires, after
                  re-checking state-dependent preconditions and institutional rules AT FIRE TIME.

Hard rules: no direct terminal-probability mutation, no Phase-13-only hidden state, implementation
failure is a first-class stochastic outcome drawn from the action's OWN CRN stream, and every effect on
other actors travels through follow-up events the plan's own operators react to (that is what makes
other-actor response real rather than a static delta).
"""
from __future__ import annotations

from swm.world_model_v2.contracts import Intervention
from swm.world_model_v2.events import Event, event_type_registered, register_event_type
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult)

# the ONE event type Phase-13 actions enter the world through (registered like any other)
register_event_type("decision_action", scheduling="scheduled", validated=True,
                    participants="acting actor + recipients",
                    deltas=("resources", "quantities", "information", "network", "queue"))

#: operation -> follow-up event type OTHER operators already react to (the actor-response bridge)
_FOLLOW_UP_ETYPE = {
    "communicate": "message_delivered", "contact": "message_delivered", "propose": "message_delivered",
    "counteroffer": "message_delivered", "request": "message_delivered", "query": "message_delivered",
    "signal": "information_published", "disclose": "information_published",
    "publish": "information_published", "commit_disclosure": "information_published",
    "endorse": "information_published",
    "submit": "decision_opportunity", "call_vote": "collective_vote", "nominate": "decision_opportunity",
    "run_experiment": "measurement", "observe": "measurement", "investigate": "measurement",
    "gather_information": "measurement",
}


def to_intervention(action, problem=None) -> Intervention:
    """Wrap an ActionSchema as a canonical Intervention. apply(world, queue) NEVER mutates entity/
    quantity state directly — it schedules the decision_action Event (and, for time-family operations,
    re-times matching queued events, which is queue surgery, not state surgery)."""

    def apply(world, queue):
        now = float(world.clock.now)
        ts = float(action.timing_ts) if action.timing_ts and action.timing_ts >= now else now
        if action.operation in ("delay", "accelerate", "cancel", "pause"):
            _retime_queue(queue, action, now)
        if action.operation in ("do_nothing",):
            return                                        # the reference arm changes NOTHING
        queue.schedule(Event(
            ts=ts, etype="decision_action",
            participants=[action.actor] + list(action.recipients or []),
            payload={"action": action, "problem": problem, "source_decision": True},
            source="phase13:intervention"))

    return Intervention(intervention_id=action.action_id,
                        description=f"{action.operation} {action.object}".strip(),
                        apply=apply,
                        kind=("policy" if action.operation.startswith("choose_") else "discrete"))


def _retime_queue(queue, action, now: float):
    """Time-family semantics: shift/cancel already-scheduled events matching the target etype."""
    import heapq
    target = str(action.params.get("etype", action.object))
    shift = float(action.params.get("shift_s", 0.0))
    kept = []
    for ev in queue.events:
        if ev.etype == target and ev.ts >= now:
            if action.operation == "cancel":
                continue
            ev.ts = max(now, ev.ts + (shift if action.operation != "accelerate" else -abs(shift)))
        kept.append(ev)
    heapq.heapify(kept)
    queue.events = kept


class DecisionActionOperator(TransitionOperator):
    """The registered operator that executes decision_action events through the CANONICAL
    propose → validate → apply funnel. Institutional rules re-check at fire time (base validate),
    state-dependent preconditions re-check here (Part 6 state-dependent feasibility), implementation
    failure draws from the action's own CRN stream (Part 8 partitioning)."""
    name = "decision_action"

    def applicable(self, world, event) -> bool:
        return event.etype == "decision_action" and bool(event.payload.get("action"))

    def propose(self, world, event, rng) -> TransitionProposal:
        a = event.payload["action"]
        # state-dependent feasibility AT FIRE TIME — an action that became infeasible mid-policy
        # fails loudly with a typed reason, it does not silently execute
        for i, pre in enumerate(getattr(a, "preconditions", []) or []):
            try:
                ok = bool(pre(world))
            except Exception as e:  # noqa: BLE001 — failing closed
                ok = False
            if not ok:
                return TransitionProposal(operator=self.name,
                                          action={"actor": a.actor, "type": a.operation,
                                                  "target": a.object, "_infeasible": f"pre[{i}]"},
                                          reason_codes=["precondition_failed_at_execution"])
        # implementation failure: the action's OWN stream (never desynchronizes unrelated shocks)
        if getattr(a, "failure_prob", 0.0) > 0.0:
            u = rng.use(f"impl|{a.action_id}").random() if hasattr(rng, "use") else rng.random()
            if u < float(a.failure_prob):
                return TransitionProposal(operator=self.name,
                                          action={"actor": a.actor, "type": a.operation,
                                                  "target": a.object, "_failed": True},
                                          reason_codes=["implementation_failed"])
        amount = float(a.params.get("amount",
                                    next(iter((a.required_resources or {}).values()), 0.0) or 0.0))
        return TransitionProposal(operator=self.name,
                                  action={"actor": a.actor, "type": a.operation, "target": a.object,
                                          "amount": amount, "_schema": a},
                                  reason_codes=[f"decision:{a.action_id}"])

    def validate(self, world, proposal) -> ValidationResult:
        act = dict(proposal.action or {})
        if act.get("_infeasible"):
            return ValidationResult(ok=False,
                                    reasons=[f"precondition failed at execution: {act['_infeasible']}"])
        act.pop("_schema", None)
        act.pop("_failed", None)
        return super().validate(world, _P(act))

    def apply(self, world, proposal) -> StateDelta:
        a = proposal.action.get("_schema")
        d = StateDelta(at=world.clock.now, event_type="decision_action", operator=self.name,
                       reason_codes=list(proposal.reason_codes))
        if proposal.action.get("_failed"):
            d.change("implementation", "attempted", "failed")
            d.uncertainty["failure_prob"] = getattr(a, "failure_prob", None) if a else None
            return d
        if a is None:
            return d
        fam = a.spec()["family"]
        actor_ent = (world.entities or {}).get(a.actor)

        # ---- resource semantics: consume required resources; transfer credits the recipient --------
        for rname, amt in (a.required_resources or {}).items():
            if actor_ent is not None:
                rf = actor_ent.get("resources", key=rname)
                if rf is not None and getattr(rf, "value", None) is not None:
                    before = float(rf.value)
                    rf.value = before - float(amt)
                    d.change(f"{a.actor}.resources[{rname}]", before, rf.value)
        if a.operation in ("transfer", "allocate", "invest") and a.recipients:
            rname = str(a.params.get("resource", ""))
            amt = float(a.params.get("amount", 0.0))
            for r in a.recipients:
                rec = (world.entities or {}).get(r)
                if rec is not None and rname:
                    rf = rec.get("resources", key=rname)
                    before = float(rf.value) if rf is not None and rf.value is not None else 0.0
                    from swm.world_model_v2.state import F
                    rec.set("resources", F(before + amt, status="derived",
                                           method="phase13:transfer"), key=rname)
                    d.change(f"{r}.resources[{rname}]", before, before + amt)

        # ---- parameter semantics: set a declared quantity the actor controls ----------------------
        if a.operation in ("set_parameter", "set_threshold", "price") and a.object:
            q = (world.quantities or {}).get(a.object)
            if q is not None:
                before = q.value
                q.value = float(a.params.get("value", before if before is not None else 0.0))
                q.timestamp = world.clock.now
                d.change(f"quantities.{a.object}", before, q.value)

        # ---- information semantics: publish/disclose/communicate enter the LEDGER -----------------
        if fam == "information" or a.operation in ("contact", "propose", "counteroffer", "endorse"):
            info = getattr(world, "information", None)
            if info is not None:
                from swm.world_model_v2.information import InformationItem
                iid = f"p13:{a.action_id}:{int(world.clock.now)}"
                content = str(a.content.get("text", a.params.get("content",
                                                                 f"{a.operation} {a.object}")))[:300]
                item = InformationItem(item_id=iid, content=content,
                                       kind=("public" if a.observability == "public" else "private"),
                                       source=a.actor, created_at=world.clock.now)
                info.publish(item)
                d.change(f"information.items[{iid}]", None, content[:80])
                for r in (a.recipients or []):
                    info.expose(r, iid, world.clock.now, channel=str(a.params.get("channel", "")))
                    d.change(f"information.exposures[{r}]", None, iid)

        # ---- relationship semantics: connect/ally/block edit the typed network --------------------
        if a.operation in ("connect", "ally", "coordinate") and a.recipients and \
                getattr(world, "network", None) is not None:
            for r in a.recipients:
                try:
                    world.network.add(a.actor, "communicates_with", r)
                    d.change(f"network.{a.actor}<->{r}", None, "communicates_with")
                except KeyError as e:
                    d.reason_codes.append(f"edge_rejected:{e}")

        # ---- meta: gather_information marks the observation intent (VOI consumes it) --------------
        if a.operation in ("gather_information", "observe", "investigate", "choose_observation"):
            d.change("meta.information_request", None, str(a.params.get("about", a.object))[:80])

        # ---- the actor-response bridge: emit the registered follow-up event the plan's own
        #      operators (Phase 4 policy, diffusion, institutions, populations) react to ------------
        fu_type = _FOLLOW_UP_ETYPE.get(a.operation)
        if fu_type and event_type_registered(fu_type):
            d.follow_up_events.append({
                "etype": fu_type, "ts": world.clock.now + 1.0,
                "participants": [a.actor] + list(a.recipients or []),
                "payload": {"from_decision_action": a.action_id, "operation": a.operation,
                            "content": {k: v for k, v in (a.content or {}).items()
                                        if isinstance(v, (int, float, str, bool))}}})
        # record the actor's own committed action on their entity (past_actions is canonical schema)
        if actor_ent is not None:
            from swm.world_model_v2.state import F
            actor_ent.set("past_actions", F(a.operation, status="derived",
                                            method=f"phase13:{a.action_id}"), key=a.action_id)
            d.change(f"{a.actor}.past_actions[{a.action_id}]", None, a.operation)
        return d


class _P:
    """Tiny adapter so the base-class institutional validate() sees `.action` (it reads proposal.action)."""
    def __init__(self, action):
        self.action = action
