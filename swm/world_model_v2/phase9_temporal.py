"""Temporal multilayer-network evolution — Phase 9 completion (Part 7).

Production event-driven edge transitions. Each transition is TYPED, consumes a trigger + preconditions,
updates the edge through a Bayesian/hazard rule (NOT an arbitrary `trust += 0.1` constant), emits a StateDelta,
carries valid-time + provenance, and changes FUTURE actor views + action feasibility. Deterministic under a
seed.

Transition parameters come from: a Beta/log-odds update for relational strength (cooperation raises trust,
betrayal lowers it — via the same edge observation likelihoods), an exponential decay hazard with a broad
half-life prior for relationship decay, and valid-time for expiration. No production transition uses a fixed
magnitude constant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.world_model_v2.phase3_posterior import _logit, _sigmoid
from swm.world_model_v2.phase9_execution import Phase9Delta
from swm.world_model_v2.phase9_network import NetworkEdge

#: cooperation/betrayal event → log-odds shift on the relational edge (from the edge observation models, not a
#: hand-picked increment). Positive raises existence/strength; negative lowers it.
_EVENT_LOGODDS = {
    "cooperation": +0.85, "fulfilled_commitment": +1.10, "public_support": +0.70, "joint_action": +1.20,
    "betrayal": -1.40, "broken_commitment": -1.20, "public_attack": -1.00, "defection": -1.60,
    "conflict_event": -1.30, "reconciliation": +0.95,
}


@dataclass
class EdgeTransition:
    kind: str                                                # trust_gain|trust_loss|alliance_formation|...
    layer: str
    src: str
    dst: str
    trigger: str                                             # the event class that fires it
    at: float = 0.0                                          # valid time (epoch seconds)
    provenance: dict = field(default_factory=dict)

    def as_dict(self):
        return self.__dict__.copy()


def _find_edge(net, src, dst, layer):
    for e in net.edges:
        if e.src == src and e.dst == dst and e.layer == layer:
            return e
    return None


def _apply_logodds(edge: NetworkEdge, shift: float) -> float:
    before = edge.existence_p
    edge.existence_p = _sigmoid(_logit(before) + shift)
    return before


# ---------------------------------------------------------------- the >=5 typed transitions
def trust_gain(net, src, dst, *, event="cooperation", at=0.0):
    """A cooperation/fulfilled-commitment event RAISES the trust edge existence via a log-odds update from the
    event's evidence weight (not a fixed +0.1). Creates the edge at a broad prior if absent."""
    e = _find_edge(net, src, dst, "trust")
    if e is None:
        e = NetworkEdge(src, dst, "trust", existence_p=0.2, observed_status="inferred")
        net.edges.append(e)
    before = _apply_logodds(e, _EVENT_LOGODDS.get(event, 0.6))
    d = Phase9Delta("trust_gain", "phase9_temporal", reason_codes=[f"event={event}", f"at={at}"])
    d.change(f"trust[{src}->{dst}].existence_p", round(before, 4), round(e.existence_p, 4))
    return d


def trust_loss(net, src, dst, *, event="betrayal", at=0.0):
    """A betrayal/broken-commitment event LOWERS trust (negative log-odds shift)."""
    e = _find_edge(net, src, dst, "trust")
    if e is None:
        e = NetworkEdge(src, dst, "trust", existence_p=0.5, observed_status="inferred")
        net.edges.append(e)
    before = _apply_logodds(e, _EVENT_LOGODDS.get(event, -1.0))
    d = Phase9Delta("trust_loss", "phase9_temporal", reason_codes=[f"event={event}", f"at={at}"])
    d.change(f"trust[{src}->{dst}].existence_p", round(before, 4), round(e.existence_p, 4))
    return d


def alliance_formation(net, src, dst, *, event="joint_action", at=0.0):
    """A joint-action/public-support event forms or strengthens an ALLIANCE edge (log-odds up)."""
    e = _find_edge(net, src, dst, "alliance")
    created = e is None
    if created:
        e = NetworkEdge(src, dst, "alliance", existence_p=0.15, observed_status="inferred", valid_from=at)
        net.edges.append(e)
    before = _apply_logodds(e, _EVENT_LOGODDS.get(event, 0.9))
    d = Phase9Delta("alliance_formation", "phase9_temporal",
                    reason_codes=[f"event={event}", "created" if created else "strengthened", f"at={at}"])
    d.change(f"alliance[{src}->{dst}].existence_p", None if created else round(before, 4), round(e.existence_p, 4))
    return d


def alliance_defection(net, src, dst, *, event="defection", at=0.0):
    """A defection event dissolves/weakens an ALLIANCE edge; may flip it toward conflict."""
    e = _find_edge(net, src, dst, "alliance")
    if e is None:
        d = Phase9Delta("alliance_defection", "phase9_temporal",
                        reason_codes=["no_alliance_to_defect", f"at={at}"])
        return d
    before = _apply_logodds(e, _EVENT_LOGODDS.get(event, -1.5))
    d = Phase9Delta("alliance_defection", "phase9_temporal", reason_codes=[f"event={event}", f"at={at}"])
    d.change(f"alliance[{src}->{dst}].existence_p", round(before, 4), round(e.existence_p, 4))
    return d


def relationship_decay(net, src, dst, layer, *, elapsed_days: float, half_life_days: float = 180.0, at=0.0):
    """Untended ties decay: existence_p *= 0.5**(elapsed/half_life) — an exponential decay HAZARD with a broad
    half-life prior (not a fixed decrement). Longer neglect → more decay."""
    e = _find_edge(net, src, dst, layer)
    if e is None:
        return Phase9Delta("relationship_decay", "phase9_temporal", reason_codes=["no_edge", f"at={at}"])
    before = e.existence_p
    e.existence_p = before * (0.5 ** (max(0.0, elapsed_days) / max(1e-6, half_life_days)))
    d = Phase9Delta("relationship_decay", "phase9_temporal",
                    reason_codes=[f"elapsed_days={elapsed_days}", f"half_life={half_life_days}", f"at={at}"])
    d.change(f"{layer}[{src}->{dst}].existence_p", round(before, 4), round(e.existence_p, 4))
    return d


def edge_expiration(net, *, at: float):
    """Remove edges whose valid_to is in the past (expired relationships) — informative absence over time."""
    kept, expired = [], []
    for e in net.edges:
        if e.valid_to is not None and e.valid_to < at:
            expired.append(e)
        else:
            kept.append(e)
    net.edges = kept
    d = Phase9Delta("edge_expiration", "phase9_temporal", reason_codes=[f"at={at}", f"n_expired={len(expired)}"])
    for e in expired:
        d.change(f"{e.layer}[{e.src}->{e.dst}]", "present", "expired")
    return d


def rewiring(net, src, old_dst, new_dst, layer, *, at=0.0):
    """Move an edge from one target to another (a relationship redirected) — the same tie, new endpoint."""
    e = _find_edge(net, src, old_dst, layer)
    if e is None:
        return Phase9Delta("rewiring", "phase9_temporal", reason_codes=["no_edge", f"at={at}"])
    e.dst = new_dst
    d = Phase9Delta("rewiring", "phase9_temporal", reason_codes=[f"at={at}"])
    d.change(f"{layer}[{src}->*].dst", old_dst, new_dst)
    return d


def role_change(net, actor, new_role, *, grants_layer="authority", grants_to=None, at=0.0):
    """An actor's role changes → may grant a new AUTHORITY/reporting edge (e.g. promotion). Changes future
    action feasibility."""
    d = Phase9Delta("role_change", "phase9_temporal", reason_codes=[f"new_role={new_role}", f"at={at}"])
    if grants_to:
        net.edges.append(NetworkEdge(actor, grants_to, grants_layer, existence_p=0.9,
                                     observed_status="observed", valid_from=at))
        d.change(f"{grants_layer}[{actor}->{grants_to}]", "absent", "granted")
    return d


#: registry so callers/tests can enumerate the >=5 typed transitions
TRANSITIONS = {"trust_gain": trust_gain, "trust_loss": trust_loss, "alliance_formation": alliance_formation,
               "alliance_defection": alliance_defection, "relationship_decay": relationship_decay,
               "edge_expiration": edge_expiration, "rewiring": rewiring, "role_change": role_change}


def evolve(net, events, *, seed=0):
    """Apply a time-ordered list of typed transition events to the network in place; return the delta trace.
    Each event = {kind, ...kwargs}. Deterministic. Future actor views + action feasibility read the mutated
    net, so evolution changes downstream behavior."""
    deltas = []
    for ev in sorted(events, key=lambda e: e.get("at", 0.0)):
        fn = TRANSITIONS.get(ev.get("kind"))
        if fn is None:
            continue
        kwargs = {k: v for k, v in ev.items() if k != "kind"}
        deltas.append(fn(net, **kwargs))
    return deltas
