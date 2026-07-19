"""Phase 13 sequential & contingent policies (Parts 12–13) — belief-state in, action out, no omniscience.

A `Policy` maps the decision-maker's OBSERVABLE belief state to an action (or None = wait). The rollout
executes it at `decision_opportunity` events through `PolicyExecutionOperator`; the chosen action enters
the world as a canonical `decision_action` follow-up event (validated + queued by the engine), so a
policy's actions flow through exactly the same intervention semantics as one-step actions.

The belief state is built from `transitions.observable_view` (the canonical actor-visibility boundary:
own fields, exposed information, public state — never other actors' private fields, latent ground truth,
or simulator metadata), plus received observations, remaining resources, time, and the policy's own
prior actions. Conditioning on hidden state is structurally impossible: the policy never receives the
WorldState, only this view.

Replanning: the policy is re-invoked at every decision point with the UPDATED belief state; a
`ContingentPlan` encodes explicit observation-triggered branches and stop rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.events import Event, register_event_type
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            observable_view)

register_event_type("decision_opportunity", scheduling="scheduled", validated=True)


def belief_state(world, actor_id: str, *, prior_actions: list = None) -> dict:
    """The ONLY input a policy sees. Built from the canonical observable-view boundary."""
    view = {}
    try:
        view = observable_view(world, actor_id)
    except Exception:  # noqa: BLE001 — an actor missing from the world sees nothing, not everything
        view = {}
    obs = []
    info = getattr(world, "information", None)
    if info is not None:
        try:
            # visible_to returns (InformationItem, Exposure) pairs — the canonical actor-visibility API
            for item, exp in info.visible_to(actor_id, at=world.clock.now):
                obs.append({"item_id": getattr(item, "item_id", ""),
                            "content": str(getattr(item, "content", ""))[:120],
                            "source": getattr(item, "source", ""),
                            "observed_at": getattr(exp, "at", None)})
        except Exception:  # noqa: BLE001
            pass
    resources = {}
    ent = (world.entities or {}).get(actor_id)
    if ent is not None:
        res = ent.get("resources")
        if isinstance(res, dict):
            for k, sf in res.items():
                if getattr(sf, "value", None) is not None:
                    resources[k] = sf.value
    return {"actor": actor_id, "t": float(world.clock.now), "view": view, "observations": obs,
            "resources": resources, "prior_actions": list(prior_actions or [])}


@dataclass
class Policy:
    """decide(belief) -> ActionSchema | None. `policy_id` names it in results."""
    policy_id: str
    decide: object
    description: str = ""


@dataclass
class ContingentPlan:
    """Explicit observation-triggered branching + stop rule: [(trigger(belief)->bool, action_factory
    (belief)->ActionSchema)] evaluated in order; `stop(belief)->bool` ends the policy's activity."""
    policy_id: str
    branches: list = field(default_factory=list)
    stop: object = None
    default: object = None                       # action_factory(belief) when no trigger fires

    def as_policy(self) -> Policy:
        def decide(belief):
            if self.stop is not None and self.stop(belief):
                return None
            for trigger, factory in self.branches:
                try:
                    if trigger(belief):
                        return factory(belief)
                except Exception:  # noqa: BLE001 — a broken trigger never crashes the world
                    continue
            return self.default(belief) if self.default is not None else None
        return Policy(policy_id=self.policy_id, decide=decide,
                      description=f"contingent plan ({len(self.branches)} branches)")


class PolicyExecutionOperator(TransitionOperator):
    """Executes a Policy at decision_opportunity events for the decision-maker. The chosen action is
    emitted as a decision_action FOLLOW-UP event — same funnel, same validation, same CRN streams."""
    name = "phase13_policy"

    def __init__(self, policy: Policy, actor_id: str, problem=None):
        self.policy = policy
        self.actor_id = actor_id
        self.problem = problem
        # per-BRANCH action history: one operator instance serves every particle in the arm, so a
        # shared list would leak "already acted" state across particles (belief pollution — particle
        # 1's policy would see particle 0's actions). Keyed by branch_id.
        self.taken: dict = {}

    def applicable(self, world, event) -> bool:
        return event.etype == "decision_opportunity" and \
            self.actor_id in (event.participants or [self.actor_id])

    def propose(self, world, event, rng) -> TransitionProposal:
        branch_taken = self.taken.setdefault(world.branch_id, [])
        belief = belief_state(world, self.actor_id, prior_actions=branch_taken)
        try:
            action = self.policy.decide(belief)
        except Exception as e:  # noqa: BLE001 — a crashing policy is a recorded no-op, not a crash
            return TransitionProposal(operator=self.name,
                                      reason_codes=[f"policy_error:{type(e).__name__}"])
        if action is None:
            return TransitionProposal(operator=self.name, reason_codes=["policy_wait"])
        branch_taken.append(action.operation)
        return TransitionProposal(
            operator=self.name, action={"actor": self.actor_id, "type": "choose_policy_step"},
            reason_codes=[f"policy:{self.policy.policy_id}"],
            follow_up_events=[{"etype": "decision_action", "ts": world.clock.now,
                               "participants": [self.actor_id] + list(action.recipients or []),
                               "payload": {"action": action, "problem": self.problem,
                                           "source_decision": True}}])

    def apply(self, world, proposal) -> StateDelta:
        d = StateDelta(at=world.clock.now, event_type="decision_opportunity", operator=self.name,
                       reason_codes=list(proposal.reason_codes))
        d.change(f"{self.actor_id}.policy_step", None,
                 proposal.reason_codes[0] if proposal.reason_codes else "step")
        return d


def schedule_decision_points(queue_builder, decision_points: list, actor_id: str):
    """Wrap a queue builder so every branch's queue carries the policy's decision points."""
    def build(world):
        q = queue_builder(world)
        for ts in decision_points:
            q.schedule(Event(ts=float(ts), etype="decision_opportunity",
                             participants=[actor_id], payload={}, source="phase13:policy"))
        return q
    return build


def one_step_policy(action, decision_ts: float) -> Policy:
    """Adapter: a one-step action as a degenerate policy (fires once at its decision point).

    Fired-state is PER BRANCH (keyed on the belief's prior_actions, which
    PolicyExecutionOperator already tracks per branch_id) — a shared closure flag would let
    the action fire in only the FIRST particle that reached a decision point and silently
    no-op in every other matched world (audit finding: cross-particle state leak)."""
    def decide(belief):
        already = any(str(a) == str(action.operation)
                      for a in (belief.get("prior_actions") or []))
        return None if already else action
    return Policy(policy_id=f"onestep:{action.action_id}", decide=decide)
