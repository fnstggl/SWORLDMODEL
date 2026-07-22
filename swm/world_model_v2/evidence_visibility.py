"""Actor-specific visibility — Phase 2.

The omniscient WorldState may hold evidence individual actors cannot access. For every claim we model WHICH
actors could plausibly know it and WHEN. Actor views filter by identity, visibility state, and earliest
observation time, so a manager cannot act on a private finance discussion they were not part of, a
negotiation party cannot see the other side's reservation value, and public news is available only after its
verified publication time. Simulator-level evidence never leaks into actor cognition unless the visibility
model grants it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

VISIBILITY_STATES = ("public", "private_actor", "private_group", "institutional", "confidential",
                     "leaked", "inferred", "unavailable", "uncertain")
#: source types whose claims are public by default, observable from their publication time
_PUBLIC_SOURCE_TYPES = ("news", "wire", "archive_snapshot", "wikipedia_revision", "official_filing",
                        "social", "dataset", "market")


@dataclass
class ClaimVisibility:
    claim_id: str
    visibility: str                                    # VISIBILITY_STATES
    actors: list = field(default_factory=list)         # actor/group ids who can observe ("*" = everyone)
    earliest_observation_time: float | None = None     # unix; None = unknown
    method: str = "source_type_default"
    uncertainty: float = 0.2
    communication_path: str = ""
    evidence: str = ""

    def observable_by(self, actor_id: str, at_time: float | None) -> bool:
        """True iff `actor_id` can observe this claim by `at_time`. Unknown visibility is NOT public by
        default — it stays restricted (fail-safe)."""
        if self.visibility in ("unavailable", "confidential"):
            return False
        if self.earliest_observation_time is not None and at_time is not None \
                and at_time < self.earliest_observation_time:
            return False                                # not yet knowable at this time
        if self.visibility == "public":
            return True
        if self.visibility in ("private_actor", "private_group", "institutional", "leaked", "inferred"):
            return actor_id in self.actors or "*" in self.actors
        # uncertain visibility → restricted unless the actor is explicitly listed (fail-safe)
        return actor_id in self.actors

    def as_dict(self):
        return asdict(self)


def assign_visibility(*, claim_id: str, source_type: str, publication_time: float | None,
                      claim_class: str = "", hint: dict | None = None) -> ClaimVisibility:
    """Assign a visibility record to a claim. Explicit `hint` (from the evidence requirement or source
    metadata: {visibility, actors, communication_path}) wins; otherwise public sources → public-from-
    publication, and anything else defaults to `uncertain` (restricted), never silently public."""
    hint = hint or {}
    if hint.get("visibility") in VISIBILITY_STATES:
        return ClaimVisibility(
            claim_id=claim_id, visibility=hint["visibility"], actors=list(hint.get("actors") or []),
            earliest_observation_time=hint.get("earliest_observation_time", publication_time),
            method="explicit_hint", uncertainty=float(hint.get("uncertainty", 0.15)),
            communication_path=str(hint.get("communication_path", "")),
            evidence=str(hint.get("evidence", "")))
    if source_type in _PUBLIC_SOURCE_TYPES:
        return ClaimVisibility(
            claim_id=claim_id, visibility="public", actors=["*"],
            earliest_observation_time=publication_time, method="public_source_default",
            uncertainty=0.1, evidence=f"public source type={source_type}")
    # user_provided / prior_world_state / unknown with no hint → restricted, uncertain
    return ClaimVisibility(
        claim_id=claim_id, visibility="uncertain", actors=[], earliest_observation_time=publication_time,
        method="no_public_basis_default_restricted", uncertainty=0.4,
        evidence=f"source type={source_type} with no visibility hint — restricted (fail-safe)")


def actor_view(visibilities: list, *, actor_id: str, at_time: float | None) -> list:
    """The subset of claim_ids an actor can observe by `at_time`. Enforced everywhere an actor policy reads
    evidence — the omniscient store is never handed to an actor directly."""
    return [v.claim_id for v in visibilities if v.observable_by(actor_id, at_time)]
