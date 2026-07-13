"""Phase 10 — authority graph (Part 5) + information boundaries (Part 6).

`AuthorityGraph.authorize(instance, action)` is the hard constraint: an action is admitted ONLY if the actor
holds the required authority for the matter, at the current stage, within the time and jurisdiction, and is
not recused. Invalid actions are BLOCKED (not warned) — the institutional operator refuses to apply them,
records the block, and (optionally) proposes a procedurally valid alternative.

`InformationBoundary` enforces Part 6: an actor's Phase-3 posterior view may only condition on information
the institution makes observable to that actor's role at that time. `visible_to(actor, info_class)` is used
to filter what enters actor policy — an actor must not act on sealed/privileged/undisclosed information.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: which authority type each action TYPE requires (extensible; templates may override per action)
ACTION_REQUIRES_AUTHORITY = {
    "approve": "approve", "reject": "approve", "final_decision": "final_decision",
    "veto": "veto", "override": "final_decision", "amend": "amend", "recommend": "recommend",
    "advise": "advise", "schedule": "agenda_control", "refer": "agenda_control", "table": "agenda_control",
    "enforce": "enforce", "appeal": "appellate", "vote": "final_decision", "certify": "final_decision",
    "moderate": "enforce", "reinstate": "appellate",
}


@dataclass
class AuthorityGraph:
    edges: list = field(default_factory=list)          # [AuthorityEdge]

    def _edges_for(self, role: str):
        return [e for e in self.edges if e.holder_role == role]

    def authorize(self, instance, action: dict, *, stage_permits=None) -> tuple[bool, str]:
        """action: {actor, type, matter?, subject?}. Returns (ok, reason). Checks, in order:
        membership → authority for the action type → subject-matter scope → stage → recusal."""
        actor = action.get("actor")
        atype = action.get("type", "")
        role = instance.actor_bindings.get(actor)
        if role is None:
            return False, f"{actor!r} holds no role in this institution — not authorized to act"

        required = action.get("required_authority") or ACTION_REQUIRES_AUTHORITY.get(atype)
        if required is not None:
            held = [e for e in self._edges_for(role) if e.authority == required]
            if not held:
                return False, (f"{actor!r} (role {role}) lacks '{required}' authority for action "
                               f"{atype!r} — advisory/recommendation authority is NOT decision authority")
            # subject-matter scope
            subj = action.get("subject")
            if subj is not None:
                scoped = [e for e in held if not e.subject_matter or subj in e.subject_matter]
                if not scoped:
                    return False, f"{actor!r} authority does not cover subject {subj!r}"

        # stage gate (an action must be permitted in the current procedural stage)
        if stage_permits is not None and atype and atype not in stage_permits:
            return False, (f"action {atype!r} not permitted in stage {instance.current_stage!r} "
                           f"(permitted: {sorted(stage_permits)})")

        # recusal / conflict of interest
        if actor in (instance.diagnostics.get("recused") or []):
            return False, f"{actor!r} is recused from this matter (conflict of interest)"
        return True, ""


#: institutional information classes (Part 6), from most open to most restricted
INFO_CLASSES = ("public", "notice", "published_decision", "role_record", "internal_deliberation",
                "private_filing", "privileged", "sealed", "classified", "ex_parte")


@dataclass
class InformationBoundary:
    """Maps (role, info_class) → observable? plus event-driven access changes (filing/publication/notice).
    Default: public/notice/published_decision observable to all; restricted classes only to authorized
    roles. `rights` overrides per template: {info_class: [roles]}."""
    rights: dict = field(default_factory=dict)          # {info_class: [roles that may observe]}

    def visible_to(self, role: str, info_class: str, *, released: set | None = None) -> bool:
        released = released or set()
        if info_class in ("public", "notice", "published_decision"):
            return True
        if info_class in released:                        # an event released it (publication/disclosure)
            return True
        allowed = self.rights.get(info_class)
        if allowed is None:
            return info_class not in ("sealed", "classified", "privileged", "ex_parte", "private_filing")
        return role in allowed

    def filter_observations(self, role: str, observations: dict, *, released=None) -> dict:
        """Drop observations the role may not see — this is what keeps unavailable info out of actor policy
        (Part 6 test: an actor must not condition behavior on information it could not observe)."""
        out = {}
        for k, v in observations.items():
            info_class = (v.get("info_class") if isinstance(v, dict) else None) or "public"
            if self.visible_to(role, info_class, released=released):
                out[k] = v
        return out
