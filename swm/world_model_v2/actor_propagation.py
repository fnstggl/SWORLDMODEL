"""Recursive actor-mediated propagation — the default consequence path for social actions.

The behavioral primitive this engine realizes, per world particle:

    actor-visible world + persistent private actor state + newly observed semantic event
    + feasible action affordances → ONE chosen action

An executed action compiles into semantic events; the observation router delivers
actor-specific observations; frontier discovery finds the causally engaged actors; each Tier-1/2
recipient gets an `actor_reconsideration` event on the CANONICAL branch queue (scheduled through
`StateDelta.follow_up_events` — the same A4 path every endogenous chain uses, in both the base
`RolloutEngine` and Phase 13's `MatchedRolloutEngine`). When that event pops, the production
actor-policy operator runs the recipient's OWN decision in that particle; their executed action
emits new semantic events, recursing.

Termination is structural, never hopeful:
  * semantic dedup — a (source, act, target, gist)-duplicate event schedules no new frontier;
  * per-branch cascade budget (`max_cascade_events`) and depth cap (`max_depth`);
  * one reconsideration per (actor, source-event);
  * the shared LLM-call budget (numeric fallback thereafter, loudly marked);
  * queue horizon and rollout max_events remain the outer bounds.
Every stop reason is recorded in the branch's event-cascade manifest
(`world.uncertainty_meta["event_cascade"]`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from swm.world_model_v2.causal_frontier import CausalFrontierDiscovery
from swm.world_model_v2.events import register_event_type
from swm.world_model_v2.observation_delivery import ObservationRouter
from swm.world_model_v2.semantic_events import compile_semantic_events

PROPAGATION_VERSION = "actor-propagation-1.0"

register_event_type("actor_reconsideration", scheduling="endogenous", visibility="participants",
                    parameter_source="actor-mediated semantic event propagation", validated=True)

#: broad response affordances offered at reconsideration — a MENU, never a restriction: the
#: qualitative prompt says "you are NOT restricted to this list" and novel actions compile.
RECONSIDERATION_AFFORDANCES = (
    "reply_now", "acknowledge", "clarify", "ignore",
    "support", "oppose", "coordinate", "defect", "persuade",
    "counteroffer", "concede", "hold_position", "escalate", "reveal", "conceal",
    "wait",
)


def propagation_enabled(world=None) -> bool:
    """Actor-mediated propagation is DEFAULT-ON. `SWM_ACTOR_PROPAGATION=off` disables it
    globally (the scalar-coupling legacy path then serves, stamped); a world may carry an
    explicit ablation flag (three-arm benchmark) in `uncertainty_meta`."""
    if os.environ.get("SWM_ACTOR_PROPAGATION", "").strip().lower() == "off":
        return False
    if world is not None:
        flag = (world.uncertainty_meta or {}).get("actor_propagation")
        if flag is not None:
            return bool(flag)
    return True


@dataclass
class PropagationBudget:
    max_cascade_events: int = 24            # reconsiderations scheduled per branch
    max_depth: int = 4                      # recursion depth (0 = the initiating action)
    max_frontier_per_event: int = 5
    reaction_latency_s: float = 1800.0      # a recipient reconsiders ~half an hour later

    @classmethod
    def one_hop(cls) -> "PropagationBudget":
        return cls(max_depth=1)

    @classmethod
    def from_env(cls) -> "PropagationBudget":
        """Declared budgets, env-overridable (`SWM_PROPAGATION_DEPTH`,
        `SWM_PROPAGATION_EVENTS`) — how the three-arm benchmark pins the one-hop arm
        without a parallel code path."""
        b = cls()
        try:
            if os.environ.get("SWM_PROPAGATION_DEPTH", ""):
                b.max_depth = max(0, int(os.environ["SWM_PROPAGATION_DEPTH"]))
            if os.environ.get("SWM_PROPAGATION_EVENTS", ""):
                b.max_cascade_events = max(0, int(os.environ["SWM_PROPAGATION_EVENTS"]))
        except ValueError:
            pass
        return b


def _cascade(world) -> dict:
    return world.uncertainty_meta.setdefault("event_cascade", {
        "version": PROPAGATION_VERSION, "scheduled": 0, "max_depth_reached": 0,
        "suppressed_duplicate": 0, "suppressed_budget": 0, "suppressed_depth": 0,
        "reconsidered": [], "seen_signatures": [], "quiescence": ""})


class SemanticPropagationEngine:
    """Bound into `ActorPolicyRuntime`; called by `execute()` after typed consequences."""

    version = PROPAGATION_VERSION

    def __init__(self, *, router: ObservationRouter | None = None,
                 frontier: CausalFrontierDiscovery | None = None,
                 budget: PropagationBudget | None = None,
                 tiers: dict | None = None, selector=None):
        self.budget = budget or PropagationBudget.from_env()
        self.router = router or ObservationRouter()
        self.frontier = frontier or CausalFrontierDiscovery(
            selector=selector, max_actors=self.budget.max_frontier_per_event)
        self.tiers = dict(tiers or {})

    def propagate(self, world, action, *, decision=None, trace=None, delta=None,
                  depth: int = 0) -> list:
        """executed action → semantic events → deliveries → frontier → reconsideration events.
        Returns follow-up event RECORDS (dicts) for `StateDelta.follow_up_events` — the engine
        never touches the queue directly; the rollout engine validates and schedules them."""
        if not propagation_enabled(world):
            return []
        cascade = _cascade(world)
        parent_ids = [str((trace or {}).trace_id)] if hasattr(trace, "trace_id") else []
        sevs = compile_semantic_events(action, world, decision=decision,
                                       parent_event_ids=parent_ids, depth=depth)
        follow_ups = []
        for sev in sevs:
            deliveries = self.router.deliver(world, sev, delta=delta)
            world.uncertainty_meta.setdefault("semantic_events", []).append(sev.as_dict())
            sig = sev.semantic_signature()
            if sig in cascade["seen_signatures"]:
                cascade["suppressed_duplicate"] += 1
                cascade["quiescence"] = "duplicate_semantic_event"
                continue                                   # quiescence: nothing new happened
            cascade["seen_signatures"] = (cascade["seen_signatures"] + [sig])[-200:]
            if depth + 1 > self.budget.max_depth:
                cascade["suppressed_depth"] += 1
                cascade["quiescence"] = "depth_budget"
                continue
            frontier = self.frontier.discover(world, sev, deliveries, tiers=self.tiers)
            for assign in frontier:
                if assign.tier > 2:
                    continue                               # Tier-3: stamped by discovery, no event
                key = (assign.actor_id, sig)
                if key in {tuple(k) for k in cascade["reconsidered"]}:
                    cascade["suppressed_duplicate"] += 1
                    continue
                if cascade["scheduled"] >= self.budget.max_cascade_events:
                    cascade["suppressed_budget"] += 1
                    cascade["quiescence"] = "event_budget"
                    from swm.world_model_v2.causal_frontier import stamp_approximation
                    stamp_approximation(world, actor_id=assign.actor_id,
                                        why="cascade event budget exhausted",
                                        approximation_type="no_reconsideration_scheduled",
                                        sensitivity="depends_on_remaining_cascade")
                    continue
                cascade["scheduled"] += 1
                cascade["reconsidered"] = (cascade["reconsidered"] + [list(key)])[-200:]
                cascade["max_depth_reached"] = max(cascade["max_depth_reached"], depth + 1)
                delivery = next((d for d in deliveries if d.recipient_id == assign.actor_id), None)
                follow_ups.append({
                    "etype": "actor_reconsideration",
                    "ts": (delivery.delivered_at if delivery is not None else world.clock.now)
                          + self.budget.reaction_latency_s,
                    "participants": [assign.actor_id, sev.actor_id],
                    "payload": {
                        "semantic_event": sev.as_dict(),
                        "situation": self._situation(sev, delivery),
                        "candidate_actions": self._affordances(world, assign.actor_id, sev),
                        "trigger_action_id": action.action_id,
                        "depth": depth + 1,
                        "tier_assignment": assign.as_dict(),
                        "reaction_to": sev.actor_id,
                    }})
        if not follow_ups and not cascade["quiescence"]:
            cascade["quiescence"] = "frontier_empty"
        return follow_ups

    @staticmethod
    def _situation(sev, delivery) -> str:
        """The situation text a recipient's decision prompt receives — the delivered
        representation (what THEY received), never the simulator's omniscient record."""
        received = delivery.received_content if delivery is not None else \
            (sev.exact_content or sev.semantic_content.get("intended_effect", ""))
        source = delivery.perceived_source if delivery is not None else sev.actor_id
        via = delivery.delivery_path if delivery is not None else "direct"
        kind = sev.event_type.replace("_", " ")
        return (f"You have just observed a {kind} from {source} (via {via}): "
                f"\"{str(received)[:300]}\" — decide how you respond, if at all.")[:400]

    @staticmethod
    def _affordances(world, actor_id: str, sev) -> list:
        """Reconsideration candidates: broad response ontology aimed at the event's source,
        plus any institutional actions this actor holds decision rights for. Feasibility and
        the actor's own (possibly novel) choice do the rest — no fixed two-item menu."""
        out = [{"name": n, "target": sev.actor_id} if n not in ("wait", "ignore")
               else {"name": n} for n in RECONSIDERATION_AFFORDANCES]
        for inst_id, inst in (world.institutions or {}).items():
            for rule in getattr(inst, "rules", []) or []:
                params = getattr(rule, "params", {}) or {}
                if getattr(rule, "kind", "") == "decision_right" and \
                        actor_id in [str(h) for h in (params.get("holders") or [])]:
                    for name in params.get("actions") or []:
                        out.append({"name": str(name), "family": "institutional",
                                    "target": inst_id,
                                    "institutional_permissions": [str(name)],
                                    "mechanisms_triggered": ["institution_processing"]})
        return out[:18]


def cascade_manifest(worlds) -> dict:
    """Aggregate the per-branch cascade manifests for result provenance."""
    rows = []
    for w in worlds:
        c = (getattr(w, "uncertainty_meta", None) or {}).get("event_cascade")
        if c:
            rows.append({"branch": w.branch_id, "scheduled": c["scheduled"],
                         "max_depth_reached": c["max_depth_reached"],
                         "suppressed_duplicate": c["suppressed_duplicate"],
                         "suppressed_budget": c["suppressed_budget"],
                         "suppressed_depth": c["suppressed_depth"],
                         "quiescence": c["quiescence"]})
    return {"version": PROPAGATION_VERSION, "branches": rows,
            "total_reconsiderations": sum(r["scheduled"] for r in rows)}
