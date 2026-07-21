"""Phase 10 — compiler integration (Part 17): select institutions BY CAUSAL NEED, not keyword routing.

The compiler requests an institutional causal process (determine_authorized_decision_maker,
evaluate_quorum_and_threshold, process_required_approval_chain, issue_formal_decision, process_appeal, …).
`select_institution` matches the process to families that DECLARE they answer it, filters candidate templates
by jurisdiction + as-of validity, prefers evidence-backed executable templates, and returns the selection
with a fallback tier + missing-evidence record. No `if domain == legislature`, no hardcoded titles/thresholds.

Fallback tiers (institutional):
  1 evidence-backed executable template, verified rules, as-of valid, jurisdiction match
  2 evidence-backed template, transported jurisdiction (widened)
  3 structural family only (no verified template) — competing rule hypotheses, broadened uncertainty
  4 generic collective-decision fallback (no institution answers the process)
"""
from __future__ import annotations

from dataclasses import dataclass, field

INSTITUTIONAL_PROCESSES = (
    "determine_authorized_decision_maker", "evaluate_matter_eligibility", "place_matter_on_agenda",
    "process_required_approval_chain", "evaluate_quorum_and_threshold", "enforce_deadline",
    "allocate_review_capacity", "issue_formal_decision", "enforce_decision", "process_appeal",
    "escalate_to_next_authority", "resolve_competing_jurisdiction")


@dataclass
class InstitutionSelection:
    process: str
    family_id: str | None
    template_id: str | None
    jurisdiction: str
    as_of: str
    tier: int
    support_grade: str
    transported: bool = False
    competing_templates: list = field(default_factory=list)
    missing_evidence: list = field(default_factory=list)
    reason: str = ""

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


TIER_GRADE = {1: "evidence_backed", 2: "transfer_supported", 3: "structural", 4: "generic_fallback"}


def _family_answers(fam, process: str) -> bool:
    procs = [p.lower() for p in (fam.answers_processes or [])]
    p = process.lower()
    if p in procs:
        return True
    ptoks = set(p.split("_")) - {"the", "a", "of", "to", "for"}
    return any(ptoks & set(d.split("_")) and len(ptoks & set(d.split("_"))) >= 2 for d in procs)


def select_institution(store, process: str, scenario: dict, *, as_of: str = "",
                       jurisdiction: str = "") -> InstitutionSelection:
    """Select the institution for ONE institutional causal process. scenario may carry
    {jurisdiction, organization, family_hint}. Returns the selection + fallback tier."""
    jur = jurisdiction or scenario.get("jurisdiction", "")
    # families that ANSWER this process
    fams = [f for f in store.families.values() if _family_answers(f, process)]
    if not fams:
        return InstitutionSelection(process, None, None, jur, as_of, 4, TIER_GRADE[4],
                                    reason=f"no institutional family answers process {process!r}")
    fam_ids = {f.family_id for f in fams}

    # candidate templates for those families, valid as-of, matching jurisdiction where given
    cands = [t for t in store.templates.values()
             if t.family_id in fam_ids and (not as_of or t.active_at(as_of))]
    in_jur = [t for t in cands if not jur or jur.lower() in (t.jurisdiction or "").lower()
              or (t.jurisdiction or "").lower() in jur.lower()]
    evidence_backed = [t for t in (in_jur or cands)
                       if t.status in ("evidence_encoded", "executable", "locally_reconstructed",
                                       "historically_replayed", "cross_institution_tested",
                                       "production_eligible") and t.has_official_evidence()]

    if evidence_backed and in_jur:
        tpl = _best(evidence_backed)
        return InstitutionSelection(process, tpl.family_id, tpl.template_id, jur, as_of, 1, TIER_GRADE[1],
                                    competing_templates=[t.template_id for t in evidence_backed if t is not tpl],
                                    reason="evidence-backed executable template, as-of valid, jurisdiction match")
    if evidence_backed:                                   # transported jurisdiction (widen)
        tpl = _best(evidence_backed)
        return InstitutionSelection(process, tpl.family_id, tpl.template_id, jur, as_of, 2, TIER_GRADE[2],
                                    transported=True,
                                    competing_templates=[t.template_id for t in evidence_backed if t is not tpl],
                                    missing_evidence=[f"no verified template for jurisdiction {jur!r}"],
                                    reason="evidence-backed template transported to a different jurisdiction")
    # structural family only (no verified template) — competing rule hypotheses
    fam = sorted(fams, key=lambda f: 0 if f.executable() else 1)[0]
    return InstitutionSelection(process, fam.family_id, None, jur, as_of, 3, TIER_GRADE[3],
                                missing_evidence=[f"no verified template for {jur!r}/{process!r}"],
                                reason="structural institutional family only — rule uncertainty, "
                                       "competing models, broadened outcome")


def _best(templates):
    order = {"production_eligible": 0, "cross_institution_tested": 1, "historically_replayed": 2,
             "locally_reconstructed": 3, "executable": 4, "evidence_encoded": 5}
    return sorted(templates, key=lambda t: order.get(t.status, 9))[0]
