"""Mandatory institutional participation — a deadline forces one procedurally-allowed terminal
choice; waiting during deliberation is fine, a missing vote at the deadline is not.

Generic across central-bank/legislative/board/judicial votes, approvals, elections,
contractual and scheduled decisions. An actor MAY wait before the deadline. When the deadline
arrives the actor's decision is REOPENED with the final feasible action set restricted to the
procedurally-permitted terminal actions (a vote option, or abstain/recuse/absent/delegate ONLY
where the institution actually permits them). The substantive choice is never forced — but the
actor cannot simply be missing because they earlier chose to wait."""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm, parse_day


@dataclass
class ParticipationObligation:
    institution_id: str
    deadline_day: str
    required_participants: list = field(default_factory=list)
    vote_options: list = field(default_factory=list)
    abstention_allowed: bool = False
    recusal_allowed: bool = False
    absence_allowed: bool = False
    delegation_allowed: bool = False
    waiting_allowed_before_deadline: bool = True
    quorum: str = ""
    consequence_of_nonparticipation: str = ""

    def terminal_action_set(self) -> list:
        """The procedurally-allowed terminal actions at the deadline (menu lines the actor
        must choose one of)."""
        acts = [f"vote:{o}" for o in self.vote_options]
        if self.abstention_allowed:
            acts.append("abstain")
        if self.recusal_allowed:
            acts.append("recuse")
        if self.absence_allowed:
            acts.append("be_absent")
        if self.delegation_allowed:
            acts.append("delegate")
        return acts

    def as_dict(self) -> dict:
        return {"institution_id": self.institution_id, "deadline_day": self.deadline_day,
                "required_participants": list(self.required_participants),
                "vote_options": list(self.vote_options),
                "abstention_allowed": self.abstention_allowed,
                "recusal_allowed": self.recusal_allowed,
                "absence_allowed": self.absence_allowed,
                "delegation_allowed": self.delegation_allowed,
                "waiting_allowed_before_deadline": self.waiting_allowed_before_deadline,
                "quorum": self.quorum,
                "consequence_of_nonparticipation": self.consequence_of_nonparticipation,
                "terminal_action_set": self.terminal_action_set()}


def build_obligations(bp, grounding: dict) -> dict:
    """One `ParticipationObligation` per institution, merging blueprint structure with the
    grounding call's institutional-obligation facts. institution_id -> obligation."""
    g_obl = (grounding or {}).get("institutional_obligations") or {}
    out = {}
    for inst in bp.institutions:
        iid = inst.get("id")
        g = g_obl.get(iid) or {}
        # vote options come from the mechanical action templates (authoritative), not the LLM
        options = []
        for t in bp.action_templates:
            for e in t.get("effects") or []:
                if e.get("kind") == "record_vote":
                    p = e.get("params") or {}
                    if p.get("institution_id") in (iid, "", None):
                        options += [str(o) for o in (p.get("options") or [])]
        options = sorted(set(options))
        deadline = (g.get("deadline_day")
                    or str(bp.terminal.get("evaluation_day") or "")[:10]
                    or next((str(s.get("day"))[:10] for s in (inst.get("procedure") or [])
                             if s.get("day")), ""))
        out[iid] = ParticipationObligation(
            institution_id=iid, deadline_day=deadline,
            required_participants=list(g.get("required_participants")
                                       or inst.get("members") or []),
            vote_options=options,
            abstention_allowed=bool(g.get("abstention_allowed")),
            recusal_allowed=bool(g.get("recusal_allowed")),
            absence_allowed=bool(g.get("absence_allowed")),
            delegation_allowed=bool(g.get("delegation_allowed")),
            waiting_allowed_before_deadline=bool(g.get("waiting_allowed_before_deadline", True)),
            quorum=norm(g.get("quorum"), 60),
            consequence_of_nonparticipation=norm(g.get("consequence_of_nonparticipation"), 200))
    return out


def is_deadline(obligation: ParticipationObligation, day: str) -> bool:
    d, dd = parse_day(day), parse_day(obligation.deadline_day)
    return d is not None and dd is not None and d >= dd
