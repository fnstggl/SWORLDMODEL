"""Unresolved mass, separated BY CAUSE — each cause has its own treatment, none renormalized.

The generic "unresolved" bucket is replaced by ten explicit causes. Each carries its mass and
a treatment; the combiner and the recovery see the causes, never a single lump. Nothing here
drops mass, assigns the prior automatically, or invents 0.5."""
from __future__ import annotations

from dataclasses import dataclass, field

UNRESOLVED_CAUSES = (
    "unresolved_future_decision",        # deadline not yet reached in sim time
    "unresolved_private_information",    # inaccessible private info → stays actor-state uncertainty
    "unresolved_unknown_state",          # mass on the explicit other_unknown_state
    "unresolved_missing_mechanism",      # no modeled mechanism resolves it → repair/under-model
    "unresolved_truncation",             # branch mass cut by a budget/node cap
    "unresolved_provider_failure",       # a stage call failed after its one retry
    "unresolved_valid_abstention",       # an EXECUTED institutional abstention (not a failure)
    "unresolved_valid_absence",          # an EXECUTED permitted absence
    "unresolved_structural_disagreement",  # primary/challenger disagree beyond evidence
    "unresolved_weight_uncertainty",     # weights unidentified → sensitivity range
)

#: causes that are actually RESOLVED institutional outcomes, not failures — an abstention or a
#: permitted absence is an executed action with a defined effect on the vote, not missing mass
EXECUTED_INSTITUTIONAL = ("unresolved_valid_abstention", "unresolved_valid_absence")


@dataclass
class UnresolvedLedger:
    by_cause: dict = field(default_factory=dict)     # cause -> mass
    detail: dict = field(default_factory=dict)       # cause -> [notes]

    def add(self, cause: str, mass: float, note: str = ""):
        if cause not in UNRESOLVED_CAUSES:
            cause = "unresolved_missing_mechanism"
        self.by_cause[cause] = round(self.by_cause.get(cause, 0.0) + float(mass), 6)
        if note:
            self.detail.setdefault(cause, []).append(note[:200])

    def total(self) -> float:
        return round(sum(self.by_cause.values()), 6)

    def genuinely_unresolved(self) -> float:
        """Mass that truly lacks a resolved outcome — abstentions/absences are executed
        institutional actions and are NOT counted here (they resolve the vote)."""
        return round(sum(m for c, m in self.by_cause.items()
                         if c not in EXECUTED_INSTITUTIONAL), 6)

    def treatment(self, cause: str) -> str:
        return {
            "unresolved_future_decision": "advance simulated time to the deadline; the "
                                          "obligation reopens the decision",
            "unresolved_private_information": "retained as actor-state uncertainty; bounded, "
                                              "never point-estimated",
            "unresolved_unknown_state": "explicit other_unknown_state mass; feasible-action "
                                        "bounds computed, never assigned prior/0.5",
            "unresolved_missing_mechanism": "one repair attempt else under-modeled disclosure",
            "unresolved_truncation": "preserved as unfinished branch mass; disclosed",
            "unresolved_provider_failure": "failed stage retried once; else disclosed",
            "unresolved_valid_abstention": "EXECUTED institutional action — resolves the vote "
                                           "per the rule, not a failure",
            "unresolved_valid_absence": "EXECUTED permitted absence — resolves per quorum rule",
            "unresolved_structural_disagreement": "kept separated by model; reported",
            "unresolved_weight_uncertainty": "sensitivity range across plausible weights",
        }.get(cause, "disclosed")

    def as_dict(self) -> dict:
        return {"by_cause": dict(self.by_cause), "total": self.total(),
                "genuinely_unresolved": self.genuinely_unresolved(),
                "detail": {k: v[:6] for k, v in self.detail.items()},
                "treatments": {c: self.treatment(c) for c in self.by_cause}}


def classify_unresolved_reason(reason: str) -> str:
    """Map an engine node's unresolved_reason string to a cause."""
    r = str(reason or "").lower()
    if "votes_missing" in r or "vote" in r and "missing" in r:
        return "unresolved_future_decision"
    if "abstain" in r:
        return "unresolved_valid_abstention"
    if "absent" in r or "absence" in r:
        return "unresolved_valid_absence"
    if "unknown_state" in r or "unknown state" in r:
        return "unresolved_unknown_state"
    if "state_predicate_not_mechanically_bound" in r or "missing" in r and "mechanism" in r:
        return "unresolved_missing_mechanism"
    if "truncat" in r:
        return "unresolved_truncation"
    if "provider" in r or "decision_unavailable" in r:
        return "unresolved_provider_failure"
    if "horizon_reached" in r:
        return "unresolved_future_decision"
    if "unknown_rule" in r:
        return "unresolved_missing_mechanism"
    return "unresolved_missing_mechanism"
