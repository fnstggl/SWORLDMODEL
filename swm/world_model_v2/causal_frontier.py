"""Event-specific causal-frontier discovery — WHO must actually decide after a semantic event.

Extends plan-time tiering (`actor_selection.RelevantActorSelector`) into per-event discovery:
for each semantic event, find the actors whose information, incentives, obligations, authority,
relationships, or decision rights the event genuinely engages — direct targets, intended and
actual recipients, institutional decision/veto holders, relevant network neighbors,
threshold-relevant members — and promote newly consequential actors at event time.

Not everyone reacts: the production hierarchy is
    Tier 1  persistent full LLM actor (individual choice could change the answer)
    Tier 2  persistent lower-cost LLM actor (potentially consequential secondary)
    Tier 3  aggregate / rule-based / no reconsideration (routine)
and EVERY Tier-3 substitution is stamped into the world's approximation manifest with the
reason, the affected actor, the approximation type, and expected sensitivity — visible, never
silent. Budgets cap the per-event frontier; actors dropped by budget are stamped the same way.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

FRONTIER_VERSION = "frontier-1.0"


@dataclass
class FrontierAssignment:
    actor_id: str
    tier: int
    reasons: list = field(default_factory=list)
    discovered_via: str = ""                # direct_target|recipient|institutional|network|promotion
    approximation: dict | None = None       # set for Tier-3 substitutions (stamped)

    def as_dict(self) -> dict:
        return asdict(self)


def stamp_approximation(world, *, actor_id: str, why: str, approximation_type: str,
                        support: str = "structural_rule", sensitivity: str = "unknown"):
    """Record one Tier-3/aggregate substitution in the world's approximation manifest."""
    world.uncertainty_meta.setdefault("approximation_manifest", []).append({
        "at": world.clock.now, "affected": actor_id, "why_not_explicit_actor": why,
        "approximation_type": approximation_type, "support": support,
        "expected_sensitivity": sensitivity, "promoted_later": False})


class CausalFrontierDiscovery:
    """semantic event + deliveries + world → ranked, tiered, budgeted reconsideration frontier."""

    version = FRONTIER_VERSION

    def __init__(self, *, selector=None, max_actors: int = 5):
        self.selector = selector
        self.max_actors = max(1, int(max_actors))

    def discover(self, world, sev, delivered, *, tiers: dict | None = None) -> list:
        tiers = tiers or {}
        entities = world.entities or {}
        reasons: dict = {}

        def note(aid, why, via):
            if aid and aid in entities and aid != sev.actor_id:
                row = reasons.setdefault(aid, {"reasons": [], "via": via})
                row["reasons"].append(why)

        for t in sev.targets:
            note(t, "direct_target_of_event", "direct_target")
        for t in sev.intended_audience:
            if t != "*":
                note(t, "intended_recipient", "direct_target")
        received = {d.recipient_id for d in delivered if getattr(d, "observed", True)}
        for r in received:
            note(r, "actually_received_observation", "recipient")

        # institutional participants: decision-right / veto holders of any institution the
        # event submits to or whose members the event's actor/targets belong to
        inst_ids = set()
        if sev.institutional_context.get("institution_id"):
            inst_ids.add(str(sev.institutional_context["institution_id"]))
        for inst_id, inst in (world.institutions or {}).items():
            for rule in getattr(inst, "rules", []) or []:
                params = getattr(rule, "params", {}) or {}
                holders = [str(h) for h in (params.get("holders") or params.get("members") or [])]
                engaged = (inst_id in inst_ids or sev.actor_id in holders
                           or any(t in holders for t in sev.targets))
                kind = getattr(rule, "kind", "")
                for h in holders:
                    if engaged and kind == "decision_right":
                        note(h, f"decision_right:{inst_id}", "institutional")
                    elif engaged and kind in ("quorum", "procedure", "capacity"):
                        note(h, f"threshold_relevant:{inst_id}:{kind}", "institutional")
                    elif engaged:
                        note(h, f"institutional_participant:{inst_id}:{kind}", "institutional")
                    elif h in received and kind in ("decision_right", "quorum"):
                        # a THRESHOLD VOTER who actually observed the event: their formal
                        # decision right makes the new information decision-relevant even when
                        # the event itself is not institutional (coalition members whose vote
                        # can alter a threshold)
                        note(h, f"threshold_relevant:{inst_id}:{kind}_holder_informed",
                             "institutional")

        # network neighbors of the SOURCE with consequential edge semantics: allies,
        # principals, and those the source reports to care about what the source just did —
        # PUBLIC events only: a neighbor cannot be engaged by a private act it never observed
        net = getattr(world, "network", None)
        if net is not None and sev.observability == "public":
            for edge in list(getattr(net, "edges", []) or []):
                rel = str(getattr(edge, "rel", "")).lower()
                if not any(k in rel for k in ("alli", "coalition", "report", "command",
                                              "influen", "trust", "partner", "member")):
                    continue
                if edge.src == sev.actor_id:
                    note(edge.dst, f"network_neighbor:{rel}", "network")
                elif edge.dst == sev.actor_id:
                    note(edge.src, f"network_neighbor:{rel}", "network")

        # THE INFORMATION GATE: reconsideration requires that the actor's information actually
        # changed — they received the observation, are a direct target, or hold institutional
        # standing on an institutional event. Nobody reacts to an event they never observed.
        out = []
        for aid, row in reasons.items():
            informed = (aid in received or aid in sev.targets
                        or (sev.observability in ("public", "institutional")
                            and any(w.startswith(("decision_right", "threshold_relevant",
                                                  "institutional_participant"))
                                    for w in row["reasons"]))
                        or (sev.observability == "public" and row["via"] == "network"))
            if not informed:
                stamp_approximation(world, actor_id=aid,
                                    why="causally adjacent but received no observation of the "
                                        "event (information gate)",
                                    approximation_type="no_reconsideration_unobserved",
                                    sensitivity="low")
                continue
            tier, extra = self._tier(world, aid, row, tiers, received)
            out.append(FrontierAssignment(actor_id=aid, tier=tier,
                                          reasons=row["reasons"] + extra,
                                          discovered_via=row["via"]))
        # deterministic ranking: strongest causal engagement first
        out.sort(key=lambda a: (a.tier, -len(a.reasons), a.actor_id))

        kept, dropped = out[:self.max_actors], out[self.max_actors:]
        for a in dropped:
            a.tier = 3
            a.approximation = {"type": "frontier_budget_drop",
                               "why": f"per-event frontier capped at {self.max_actors}"}
            stamp_approximation(world, actor_id=a.actor_id,
                               why=f"per-event frontier budget ({self.max_actors}) reached",
                               approximation_type="no_reconsideration_scheduled",
                               sensitivity="low_by_ranking")
        for a in kept:
            if a.tier >= 3:
                a.approximation = {"type": "tier3_routine",
                                   "why": "no live causal signal strong enough for cognition"}
                stamp_approximation(world, actor_id=a.actor_id,
                                    why="; ".join(a.reasons[:3]) or "routine observer",
                                    approximation_type="aggregate_or_no_reaction",
                                    sensitivity="low")
        return kept + dropped

    def _tier(self, world, aid: str, row: dict, tiers: dict, received: set):
        """Tier for one candidate. Plan-time tier map first; then event-time causal signals;
        then dynamic promotion via the selector; else Tier 3."""
        pre = tiers.get(aid)
        if isinstance(pre, dict) and int(pre.get("tier", 3)) <= 2:
            return int(pre["tier"]), [f"plan_tier:{pre['tier']}"]
        why = row["reasons"]
        strong = [w for w in why if w.startswith(("direct_target_of_event", "decision_right"))]
        medium = [w for w in why if w.startswith(("threshold_relevant", "intended_recipient",
                                                  "institutional_participant"))]
        # an informed recipient with a consequential network relation to the source (ally,
        # principal, influence) is a secondary-consequential actor, not a routine bystander
        if aid in received and any(w.startswith("network_neighbor") for w in why):
            medium.append("informed_network_neighbor")
        if strong:
            return 1, ["event_tier:direct_or_authority"]
        if self.selector is not None:
            promo = self.selector.promote_if_consequential(world, aid, {"situation": "event"})
            if promo is not None and int(promo.get("tier", 3)) <= 2:
                for m in world.uncertainty_meta.get("actor_tier_promotions", []):
                    if m.get("actor") == aid:
                        m["promoted_by"] = "causal_frontier"
                return int(promo["tier"]), list(promo.get("reasons", []))[:4]
        if medium:
            return 2, ["event_tier:secondary_consequential"]
        return 3, []
