"""Universal observation delivery — the subsystem that OWNS information reach.

For each semantic event and world particle the router decides which actors CAN receive it,
which actually DO, when, in what representation (original / summary / relayed account), through
which channel, with what perceived source and credibility — honoring institutional information
boundaries and network structure. The LLM actor never decides whether information magically
reached itself; it only interprets what the router delivered.

Every delivery is appended to the branch's actor-local information state (the
`InformationLedger`: publish + expose) with provenance back to the original semantic event, so
`ActorViewBuilder` — the one projection actors see the world through — picks deliveries up with
zero new leakage surface. A recipient whose delivery timestamp lies in the future simply does
not see the item until the clock passes it (the ledger's `visible_to` is time-gated).

Deterministic by default: reach follows structure (observability class, network edges,
institutional boundaries), not coin flips, so architecture tests replay exactly. Attention-based
unread-delivery uses the entity's own attention state, recorded, never a hidden die roll.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from swm.world_model_v2.information import InformationItem

OBSERVATION_ROUTER_VERSION = "obsrouter-1.0"

#: default source credibility by channel when no trust edge exists (documented priors, visible)
_CHANNEL_CREDIBILITY = {"direct_private": 0.9, "institutional_channel": 0.8,
                        "public_broadcast": 0.7, "relayed": 0.5}
#: content longer than this reaches NON-TARGET public observers as a summary (distortion recorded)
_SUMMARY_THRESHOLD = 400
_SUMMARY_LEN = 280


@dataclass
class DeliveredObservation:
    """One actor's receipt of one semantic event — the ledger-backed delivery record."""
    event_id: str
    recipient_id: str
    delivered_at: float
    received_content: str
    perceived_source: str
    delivery_path: str                      # direct | broadcast | institutional | network:<edge>
    visibility_class: str
    credibility: float
    representation: str = "original"        # original | summary | relayed
    distortion: dict = field(default_factory=dict)
    observed: bool = True                   # False = delivered but not yet seen
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class ObservationRouter:
    """semantic event + world particle → actor-specific deliveries, written into the ledger."""

    version = OBSERVATION_ROUTER_VERSION

    def deliver(self, world, sev, *, delta=None) -> list:
        if world.information is None:
            return []
        recipients = self._recipients(world, sev)
        deliveries = []
        published: dict = {}
        for rid, path in recipients:
            if rid == sev.actor_id:
                continue                                   # the source already knows its own act
            rep, content, distortion = self._representation(sev, rid, path)
            credibility = self._credibility(world, sev, rid, path)
            at = self._delivery_time(world, sev, rid, path)
            observed = self._attended(world, rid)
            item_key = (rep, content)
            iid = published.get(item_key)
            if iid is None:
                iid = sev.event_id if rep == "original" else f"{sev.event_id}:{rep}"
                if iid not in world.information.items:
                    world.information.publish(InformationItem(
                        item_id=iid, content=content,
                        kind=("public" if sev.observability == "public" else "private"),
                        source=self._perceived_source(sev, path),
                        credibility=credibility, created_at=sev.timestamp,
                        about=",".join(sev.targets)[:80]))
                published[item_key] = iid
            world.information.expose(rid, iid, at, channel=path,
                                     salience=0.7 if rep == "original" else 0.5,
                                     observed=observed)
            deliveries.append(DeliveredObservation(
                event_id=sev.event_id, recipient_id=rid, delivered_at=at,
                received_content=content, perceived_source=self._perceived_source(sev, path),
                delivery_path=path, visibility_class=sev.observability,
                credibility=credibility, representation=rep, distortion=distortion,
                observed=observed,
                provenance={"router": self.version, "source_event": sev.event_id,
                            "ledger_item": iid, "world_hypothesis": sev.world_hypothesis_id}))
        sev.actual_recipients = sorted({d.recipient_id for d in deliveries})
        if delta is not None and deliveries:
            delta.change(f"information.deliveries[{sev.event_id}]", None,
                         [(d.recipient_id, d.representation) for d in deliveries])
            delta.reason_codes.append(f"observation_delivery:{len(deliveries)}")
        return deliveries

    # ---- reach -----------------------------------------------------------------------
    def _recipients(self, world, sev) -> list:
        """(recipient_id, delivery_path) pairs. Reach is structural: observability class first,
        then institutional boundaries, then network edges for second-hand public reach."""
        entities = world.entities or {}
        out: list = []
        if sev.observability == "public":
            direct = set(sev.targets)
            for eid in entities:
                out.append((eid, "direct" if eid in direct else "broadcast"))
        elif sev.observability == "institutional":
            inst_id = sev.institutional_context.get("institution_id", "")
            members = self._institution_members(world, inst_id)
            boundary = (world.uncertainty_meta.get("institution_boundaries") or {}).get(inst_id)
            for eid in members | set(sev.targets):
                if eid not in entities:
                    continue
                if boundary is not None:
                    role = self._role(entities.get(eid))
                    try:
                        if not boundary.visible_to(role, "internal"):
                            continue
                    except Exception:  # noqa: BLE001 — a malformed boundary must fail closed
                        continue
                out.append((eid, "institutional"))
        else:                                              # participants / private
            for t in sev.targets:
                if t in entities:
                    out.append((t, "direct"))
        return out

    @staticmethod
    def _institution_members(world, inst_id: str) -> set:
        members: set = set()
        inst = (world.institutions or {}).get(inst_id)
        for rule in getattr(inst, "rules", []) or []:
            params = getattr(rule, "params", {}) or {}
            for key in ("holders", "members"):
                for m in params.get(key) or []:
                    members.add(str(m))
        return members

    @staticmethod
    def _role(entity) -> str:
        if entity is None:
            return "unknown"
        roles = entity.value("roles", default=None)
        if isinstance(roles, list) and roles:
            return str(roles[0])
        return str(roles or entity.entity_type)

    # ---- representation / credibility / timing ---------------------------------------
    @staticmethod
    def _representation(sev, rid: str, path: str):
        content = sev.exact_content or sev.semantic_content.get("intended_effect", "") or \
            f"{sev.actor_id} {sev.semantic_content.get('action_name', 'acted')}"
        if path == "broadcast" and len(content) > _SUMMARY_THRESHOLD:
            summary = content[:_SUMMARY_LEN] + "…"
            return "summary", summary, {"summarized": True, "original_len": len(content),
                                        "summary_len": len(summary)}
        return "original", str(content)[:800], {}

    def _credibility(self, world, sev, rid: str, path: str) -> float:
        # a trust edge from recipient toward the source overrides the channel default
        net = getattr(world, "network", None)
        if net is not None:
            for edge in (net.out_edges(rid) if hasattr(net, "out_edges") else []):
                if edge.dst == sev.actor_id and getattr(edge, "rel", "") == "trusts":
                    strength = getattr(edge, "strength", None)
                    v = getattr(strength, "value", None)
                    if isinstance(v, (int, float)):
                        return max(0.05, min(0.99, float(v)))
        base = _CHANNEL_CREDIBILITY.get(
            "relayed" if path == "broadcast" and sev.channel != "public_broadcast" else
            sev.channel if path != "broadcast" else "public_broadcast", 0.6)
        return base

    @staticmethod
    def _delivery_time(world, sev, rid: str, path: str) -> float:
        delay = 0.0
        net = getattr(world, "network", None)
        if net is not None and path == "broadcast":
            delay = 3600.0                                  # second-hand public reach: ~an hour
            for edge in list(getattr(net, "edges", []) or []):
                if {edge.src, edge.dst} == {sev.actor_id, rid} and getattr(edge, "delay_hours", 0):
                    delay = float(edge.delay_hours) * 3600.0
                    break
        return float(sev.timestamp) + delay

    @staticmethod
    def _perceived_source(sev, path: str) -> str:
        if path == "broadcast" and sev.channel != "public_broadcast":
            return f"reported:{sev.actor_id}"               # second-hand account of a private act
        return sev.actor_id

    @staticmethod
    def _attended(world, rid: str) -> bool:
        ent = (world.entities or {}).get(rid)
        if ent is None:
            return True
        att = ent.value("attention", default=None)
        if isinstance(att, (int, float)) and float(att) < 0.2:
            return False                                    # delivered but unread — inbox != read
        return True
