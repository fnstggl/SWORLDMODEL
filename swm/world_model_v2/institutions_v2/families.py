"""Phase 10 — reusable institutional FAMILY execution (Part 19).

Each family is a reusable STRUCTURAL pattern; a template supplies the real roles/authority/stages/thresholds.
The characteristic PROCEDURE of each family is a genuinely distinct executable transition here (bicameral
concurrence + veto/override; appellate routing; approval chains; queue service timing). These are the
`code_ref` targets that make each family executable; the shared InstitutionOperator drives them in the world.
All numeric thresholds come from the template's verified rules — never invented here.
"""
from __future__ import annotations

from swm.world_model_v2.institutions_v2.decisions import (ThresholdSpec, apply_veto_and_override,
                                                          evaluate_decision)


# ---------------- A. hierarchical approval chain ----------------
def hierarchical_approval(chain: list, approvals: dict) -> dict:
    """A multi-level approval chain: EVERY required level must approve, in order, for the matter to pass.
    chain: [role_id in required order]; approvals: {role_id: 'approve'|'reject'|'pending'}. A rejection at
    any level stops the chain (subsequent levels never act). Returns the outcome + the level reached."""
    for i, level in enumerate(chain):
        decision = approvals.get(level, "pending")
        if decision == "reject":
            return {"outcome": "rejected", "stopped_at": level, "level_index": i, "approved_levels": chain[:i]}
        if decision != "approve":
            return {"outcome": "pending", "waiting_on": level, "level_index": i, "approved_levels": chain[:i]}
    return {"outcome": "approved", "approved_levels": list(chain), "level_index": len(chain)}


# ---------------- B. collective vote body ----------------
def collective_vote_body(spec: ThresholdSpec, votes: dict, *, eligible: list, **kw) -> dict:
    """A single collective decision under a threshold/quorum rule."""
    return evaluate_decision(spec, votes, eligible=eligible, **kw).as_dict()


# ---------------- C. legislative process (bicameral + veto/override) ----------------
def legislative_process(*, chamber_specs: dict, chamber_votes: dict, chamber_eligible: dict,
                        vetoed: bool = False, override_spec: ThresholdSpec | None = None,
                        override_votes: dict | None = None) -> dict:
    """Bicameral passage + presentment + veto/override (US Art I §7, evidence-backed thresholds).
    Each chamber must pass under its own threshold (majority of a quorum); then presentment; a veto is only
    overcome by a successful override in BOTH chambers (2/3 of a quorum). chamber_* keyed by chamber id."""
    results, all_passed = {}, True
    for ch, spec in chamber_specs.items():
        res = evaluate_decision(spec, chamber_votes.get(ch, {}), eligible=chamber_eligible.get(ch, []))
        results[ch] = res.as_dict()
        all_passed = all_passed and res.passed
    if not all_passed:
        return {"outcome": "failed_passage", "chambers": results, "vetoed": False}
    if not vetoed:
        return {"outcome": "enacted", "chambers": results, "vetoed": False}
    # veto → override required in every chamber
    ov = {}
    overridden_all = True
    for ch, ospec in (override_spec or {}).items():
        res = evaluate_decision(ospec, (override_votes or {}).get(ch, {}), eligible=chamber_eligible.get(ch, []))
        ov[ch] = res.as_dict()
        overridden_all = overridden_all and res.passed
    return {"outcome": "enacted_over_veto" if overridden_all else "vetoed_sustained",
            "chambers": results, "vetoed": True, "override": ov, "overridden": overridden_all}


# ---------------- D. adjudicative court + appeal ----------------
def adjudicative_court(*, decision: str, appealed: bool, appellate_decision: str | None = None) -> dict:
    """Trial decision → optional appeal → appellate outcome (affirm/reverse/remand). The appellate court's
    decision governs; a remand sends the matter back."""
    if not appealed:
        return {"outcome": decision, "final": True, "path": ["trial"]}
    ad = appellate_decision or "affirm"
    final_outcome = {"affirm": decision, "reverse": _reverse(decision), "remand": "remanded"}.get(ad, decision)
    return {"outcome": final_outcome, "final": ad != "remand", "path": ["trial", "appeal"],
            "appellate": ad}


def _reverse(d):
    return {"granted": "denied", "denied": "granted", "guilty": "not_guilty",
            "not_guilty": "guilty"}.get(d, f"reversed_{d}")


# ---------------- E. administrative agency (application → review → decision → appeal) ----------------
def administrative_agency(*, complete: bool, staff_recommendation: str, decision_authority_decision: str,
                          appealed: bool = False, appellate_decision: str | None = None) -> dict:
    """Completeness gate → staff recommendation → decision-maker's formal decision → optional admin appeal.
    An incomplete application cannot be decided (blocked upstream)."""
    if not complete:
        return {"outcome": "returned_incomplete", "final": False}
    formal = decision_authority_decision
    if appealed and appellate_decision:
        formal = appellate_decision
    return {"outcome": formal, "final": True, "staff_recommendation": staff_recommendation,
            "appealed": appealed}


# ---------------- F. queue / capacity-constrained service ----------------
def queue_capacity_service(queue, *, periods: int, target_matter: str) -> dict:
    """Serve a queue for N periods; report when the target matter completes (real timing effect). A backlog
    that exceeds capacity delays completion — the queue is a state constraint, not a label."""
    wait = queue.wait_periods(target_matter)
    completed_at = None
    for p in range(periods):
        served = queue.service_period(p)
        if target_matter in served:
            completed_at = p
            break
    return {"outcome": "served" if completed_at is not None else "still_waiting",
            "completed_at_period": completed_at, "estimated_wait_periods": wait,
            "backlog_remaining": queue.backlog()}


# ---------------- G. corporate board (delegated management + reserved board matters) ----------------
def corporate_board(*, matter_is_reserved: bool, management_decision: str, board_spec: ThresholdSpec | None,
                    board_votes: dict | None, board_eligible: list | None, recused: set | None = None) -> dict:
    """Management decides delegated matters; RESERVED matters require a board vote (with recusal for
    conflicts). A conflicted director's recusal changes the eligible base."""
    if not matter_is_reserved:
        return {"outcome": management_decision, "decided_by": "management", "final": True}
    res = evaluate_decision(board_spec, board_votes or {}, eligible=board_eligible or [],
                            recused=recused or set())
    return {"outcome": "approved" if res.passed else "rejected", "decided_by": "board",
            "decision": res.as_dict(), "final": True}


# ---------------- H. platform moderation + appeal ----------------
def moderation_appeals(*, violates_policy: bool, penalty: str, appealed: bool = False,
                       appeal_upheld: bool | None = None) -> dict:
    """Report → policy check → penalty → optional appeal → reinstatement if the appeal is upheld."""
    if not violates_policy:
        return {"outcome": "no_action", "final": True}
    if appealed and appeal_upheld:
        return {"outcome": "reinstated", "final": True, "original_penalty": penalty}
    return {"outcome": penalty, "final": True, "appealed": appealed}
