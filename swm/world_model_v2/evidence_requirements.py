"""Typed evidence requirements — Phase 2 (planning).

The universal compiler emits typed requirements describing exactly what evidence is causally needed, why, and
under what temporal/entity/visibility constraints. Requirements are prioritized by expected value of
information (sensitivity × how much a fact could move the outcome), not by ease of retrieval. The evidence
system is subordinate to the compiler on WHAT to answer, but may report a requirement unmet, contradictory,
ambiguous, or newly-discovered — new requirements route back through iterative compilation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict

SOURCE_TYPES = ("news", "wire", "official_filing", "regulatory", "court", "legislative", "election",
                "academic", "social", "archive_snapshot", "wikipedia_revision", "dataset", "user_provided",
                "prior_world_state")


@dataclass
class EvidenceRequirement:
    requirement_id: str
    claim_or_quantity: str                             # exactly what is needed
    why_relevant: str                                  # causal justification
    affected_component: str                            # actor/population/institution/mechanism/terminal id
    expected_sensitivity: float = 0.5                  # how much this could move the outcome
    expected_voi: float = 0.5                          # value of information (sensitivity × resolvability gap)
    preferred_source_types: list = field(default_factory=lambda: ["news", "wire"])
    fallback_source_types: list = field(default_factory=lambda: ["wikipedia_revision"])
    disallowed_source_types: list = field(default_factory=list)
    as_of_constraint: str = ""                         # ISO; retrieval must not use content after this
    event_time_scope: str = ""
    publication_time_scope: str = ""                   # e.g. "after:2023-08-01 before:2023-09-30"
    geographic_scope: str = ""
    jurisdiction: str = ""
    entity_scope: list = field(default_factory=list)
    structured_fields: list = field(default_factory=list)
    actor_visibility_assumption: str = "public"
    absence_informative: bool = False                  # does absence of evidence carry signal?
    retrieval_cost_estimate: float = 1.0
    stopping_criteria: str = "coverage>=1 claim or 2 queries"
    missing_consequence: str = "widen prior; lower support grade"
    uncertainty_if_unmet: str = "broad"
    status: str = "open"                               # open/fulfilled/partial/contradictory/ambiguous/unmet

    def as_dict(self):
        return asdict(self)


def _rid(*parts) -> str:
    return "req_" + hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def requirements_from_plan(plan, *, as_of_iso: str, question: str, max_reqs: int = 14) -> list:
    """Deterministically derive typed evidence requirements from a compiled WorldExecutionPlan. One
    requirement for the terminal outcome, plus: the highest-sensitivity latents; key institutions/rules;
    each structural hypothesis's discriminating fact; RECENT STATEMENTS by each principal actor (the
    stance-grounding substrate); MEASURED VALUES for the plan's declared quantities (order-of-battle-
    class facts); and the SCHEDULED CALENDAR (aid packages, votes, deadlines, meetings) inside the
    window. Prioritized by expected VoI (sensitivity-weighted). Evidence was measured THIN relative to
    the questions (~13 items for a world war): breadth scales with the plan's own declared structure —
    never with scenario keywords."""
    reqs = []
    oc = plan.outcome_contract
    # 1) the terminal outcome itself — what is the current standing / base rate as of the question date
    reqs.append(EvidenceRequirement(
        requirement_id=_rid("outcome", question), claim_or_quantity=oc.resolution_rule or question,
        why_relevant="the terminal outcome standing as of the question date sets the base rate",
        affected_component="terminal_outcome", expected_sensitivity=1.0, expected_voi=1.0,
        as_of_constraint=as_of_iso, publication_time_scope="paired_after_before",
        absence_informative=True, entity_scope=[str(e.get("id")) for e in plan.entities[:4]
                                                if isinstance(e, dict)]))
    # 2) per-ACTOR recent statements — the substrate the stance grounding classifies; without them
    #    intentions fall back to model knowledge (weaker reliability, lower fidelity)
    actors = sorted([e for e in (plan.entities or []) if isinstance(e, dict) and e.get("id")],
                    key=lambda e: -float(e.get("sensitivity", 0.5) or 0.5))
    for e in actors[:4]:
        aid = str(e.get("id"))
        reqs.append(EvidenceRequirement(
            requirement_id=_rid("actor_statements", aid),
            claim_or_quantity=f"recent public statements positions of {aid.replace('_', ' ')}",
            why_relevant="a principal's dated public statements ground their stances (commitment "
                         "level, pathway, target mode) — behavior conditioning needs them",
            affected_component=aid, expected_sensitivity=0.85, expected_voi=0.85,
            as_of_constraint=as_of_iso, publication_time_scope="paired_after_before",
            entity_scope=[aid]))
    # 3) high-sensitivity latents → the hidden variable each needs a contemporaneous fact for
    for l in sorted(plan.latents, key=lambda x: -getattr(x, "sensitivity", 0.5))[:3]:
        s = getattr(l, "sensitivity", 0.5)
        reqs.append(EvidenceRequirement(
            requirement_id=_rid("latent", l.path), claim_or_quantity=f"value/context of {l.path}",
            why_relevant=f"high-sensitivity latent ({s:.2f}) driving the outcome distribution",
            affected_component=l.path, expected_sensitivity=float(s), expected_voi=round(float(s), 3),
            as_of_constraint=as_of_iso, publication_time_scope="paired_after_before"))
    # 4) institutions / rules — the decision procedure that governs the outcome
    for inst in (plan.institutions or [])[:2]:
        iid = inst.get("id") if isinstance(inst, dict) else str(inst)
        reqs.append(EvidenceRequirement(
            requirement_id=_rid("institution", iid), claim_or_quantity=f"decision rules / process of {iid}",
            why_relevant="institutional rules determine which actions are feasible and when the outcome resolves",
            affected_component=str(iid), expected_sensitivity=0.7, expected_voi=0.7,
            preferred_source_types=["official_filing", "regulatory", "news"],
            as_of_constraint=as_of_iso, absence_informative=True))
    # 5) structural hypotheses — the discriminating fact between competing world structures
    for h in (plan.structural_hypotheses or [])[:2]:
        hid = h.get("id") if isinstance(h, dict) else str(h)
        reqs.append(EvidenceRequirement(
            requirement_id=_rid("hypothesis", hid),
            claim_or_quantity=f"evidence discriminating structural hypothesis {hid}",
            why_relevant="a fact that reweights competing structural hypotheses materially changes the outcome",
            affected_component=f"structural_hypothesis:{hid}", expected_sensitivity=0.8, expected_voi=0.8,
            as_of_constraint=as_of_iso, publication_time_scope="paired_after_before"))
    # 6) declared QUANTITIES — measured values for the state variables the mechanisms consume
    #    (force levels, counts, rates — the order-of-battle class, derived from the plan itself)
    for q in [x for x in (plan.quantities or []) if isinstance(x, dict) and x.get("name")][:2]:
        qn = str(q["name"])
        if qn.startswith(("pathway_progress:", "absorbed_", "absorbing_", "actor_intentions")):
            continue                                       # internal machinery, not world measurements
        reqs.append(EvidenceRequirement(
            requirement_id=_rid("quantity", qn),
            claim_or_quantity=f"current measured value of {qn.replace('_', ' ')}",
            why_relevant="mechanisms consume this declared quantity; an as-of measurement replaces "
                         "a broad prior", affected_component=qn,
            expected_sensitivity=0.6, expected_voi=0.6, as_of_constraint=as_of_iso,
            preferred_source_types=["news", "dataset", "official_filing"],
            publication_time_scope="paired_after_before"))
    # 7) the SCHEDULED CALENDAR — dated commitments inside the window (votes, packages, summits,
    #    deadlines) become scheduled-reality events; absence is itself informative
    reqs.append(EvidenceRequirement(
        requirement_id=_rid("calendar", question),
        claim_or_quantity="scheduled dated events, votes, deadlines, packages, meetings relevant "
                          "to the question window",
        why_relevant="dated public commitments become deterministic scheduled events in the "
                     "trajectory (the scheduled-reality layer)",
        affected_component="scheduled_reality", expected_sensitivity=0.65, expected_voi=0.65,
        as_of_constraint=as_of_iso, publication_time_scope="paired_after_before",
        absence_informative=True,
        entity_scope=[str(e.get("id")) for e in plan.entities[:3] if isinstance(e, dict)]))
    reqs.sort(key=lambda r: -r.expected_voi)
    return reqs[:max_reqs]
