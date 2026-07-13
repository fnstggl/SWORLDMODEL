"""Phase 8 completion — executable family-support enforcement + default selection + fallback tiers.

The distinction the completion pass enforces is NOT "validated → usable / experimental → disabled". It is
"validated → usable with stronger support / experimental → usable with broader uncertainty, explicit
assumptions, a lower support grade, and a sensitivity contribution". Only `quarantined` or `incompatible`
families are blocked from automatic production execution.

This module makes that policy EXECUTABLE (not documentation):
  * ``select_families`` — given the causally-relevant families, returns which execute by default (blocks only
    quarantined/incompatible) with a per-family reason;
  * ``family_runtime_manifest`` — the full runtime provenance a forecast must expose per used family;
  * ``support_grade_effect`` — how a load-bearing experimental family lowers the overall support grade and
    widens uncertainty;
  * ``resolve_transition_tier`` — the 7-tier fallback (scenario-fitted → … → competing hypotheses); an
    uncertain transition is NEVER removed, it drops to the strongest available tier.

No coefficients are minted here — this is policy over the specs in ``phase8_persistence``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.phase8_persistence import (BLOCKED_STATUSES, RUNTIME_STATUSES, get_persistent_variable,
                                                   registered_variables)

# support grade ranking (strongest→weakest) — mirrors result.SUPPORT_GRADES
_GRADE_RANK = {"empirically_supported": 0, "transfer_supported": 1, "exploratory": 2, "highly_speculative": 3}
_STATUS_TO_GRADE = {"empirically_supported": "empirically_supported",
                    "transfer_supported": "transfer_supported",
                    "exploratory": "exploratory", "highly_speculative": "highly_speculative"}

# the 7-tier persistence-transition fallback (Part 4). An uncertain transition drops to the strongest
# AVAILABLE tier — it is never silently removed.
TRANSITION_TIERS = {
    1: "scenario-fitted and held-out validated transition",
    2: "domain-validated transition pack",
    3: "cross-domain / cross-population transfer-validated pack",
    4: "published empirical transition with widened transport uncertainty",
    5: "reference-class estimated transition",
    6: "generic structural transition with broad parameter priors",
    7: "multiple competing persistence hypotheses (materialized + disagreement propagated)",
}
#: which fallback tier each runtime status maps to when no stronger pack is bound
_STATUS_DEFAULT_TIER = {"empirically_supported": 1, "transfer_supported": 3, "exploratory": 6,
                        "highly_speculative": 7, "quarantined": 7, "incompatible": 7}


@dataclass
class FamilySelection:
    variable_id: str
    runtime_status: str
    selected: bool
    reason: str
    transition_tier: int
    support_grade: str
    transport_risk: str
    uncertainty_widening: float


def select_families(candidate_variable_ids, *, allow_blocked=False) -> list:
    """Given the families the applicability system judged causally relevant, decide which execute by DEFAULT.
    Blocks ONLY `quarantined`/`incompatible` (unless `allow_blocked`, used by ablations). Every other family
    — including exploratory/highly_speculative — is selected, carrying its transition tier, support-grade
    contribution, transport risk, and an uncertainty-widening factor. Returns [FamilySelection]."""
    out = []
    for vid in candidate_variable_ids:
        try:
            spec = get_persistent_variable(vid)
        except KeyError:
            out.append(FamilySelection(vid, "incompatible", False,
                                       "unknown family — not registered", 7, "highly_speculative", "high", 2.0))
            continue
        status = spec.runtime_status
        blocked = status in BLOCKED_STATUSES and not allow_blocked
        tier = _STATUS_DEFAULT_TIER.get(status, 6)
        widen = {"empirically_supported": 1.0, "transfer_supported": 1.2, "exploratory": 1.6,
                 "highly_speculative": 2.0, "quarantined": 2.5, "incompatible": 2.5}.get(status, 1.6)
        if blocked:
            reason = (f"BLOCKED: status {status!r} — quarantined/incompatible families are the only ones "
                      "prevented from automatic execution")
        else:
            reason = (f"selected by default (status {status!r}); "
                      + ("strong empirical support" if status == "empirically_supported"
                         else "usable with broader uncertainty + lower support grade + sensitivity contribution"))
        out.append(FamilySelection(
            variable_id=vid, runtime_status=status, selected=not blocked, reason=reason,
            transition_tier=tier, support_grade=_STATUS_TO_GRADE.get(status, "highly_speculative"),
            transport_risk=spec.transport_risk, uncertainty_widening=widen))
    return out


def family_runtime_manifest(variable_id: str, *, posterior=None, applicability=None,
                            sensitivity_contribution=None) -> dict:
    """The full runtime provenance a forecast must expose for a used family (Part 3). Combines the static
    spec with the live posterior/applicability/sensitivity when supplied. Unsupported precision is refused
    at the source (posteriors carry sd); this manifest surfaces it so a reviewer sees the uncertainty."""
    spec = get_persistent_variable(variable_id)
    m = {"family_id": spec.variable_id, "implementation_version": spec.schema_version,
         "runtime_status": spec.runtime_status, "transition_family": spec.transition_family,
         "parameter_source": spec.transition_param_source, "supporting_evidence": spec.supporting_evidence,
         "transport_risk": spec.transport_risk, "state_variables_affected": [spec.materializes_into],
         "downstream_consumers": list(spec.consumed_by),
         "identifiability": spec.identifiability, "terminal_sensitivity": spec.terminal_sensitivity,
         "limitations": _family_limitations(spec)}
    if posterior is not None:
        m["posterior_uncertainty"] = {"mean": round(getattr(posterior, "mean", 0.0), 4),
                                      "sd": round(getattr(posterior, "sd", 0.0), 4),
                                      "n_events": getattr(posterior, "n_events_assimilated", 0),
                                      "ess": round(getattr(posterior, "ess", 0.0), 3)}
    if applicability is not None:
        m["applicability_result"] = applicability
    if sensitivity_contribution is not None:
        m["sensitivity_contribution"] = sensitivity_contribution
    return m


def _family_limitations(spec) -> list:
    lims = []
    if spec.runtime_status in ("exploratory", "highly_speculative"):
        lims.append(f"{spec.runtime_status}: transition parameters are {spec.transition_param_source} "
                    "(labeled, not held-out validated); coefficients are NOT precise point estimates")
    if spec.transport_risk == "high":
        lims.append("high transport risk — out-of-domain behavior widens terminal uncertainty")
    if spec.identifiability != "identified":
        lims.append(f"{spec.identifiability} — the latent may be only weakly recoverable from history")
    return lims


def support_grade_effect(base_grade: str, selections, *, load_bearing_ids=None) -> dict:
    """Compute the overall support-grade effect when experimental families are materially load-bearing
    (Part 3). A load-bearing exploratory/highly_speculative family lowers the overall grade to (at most) its
    own status and records a widening factor + a limitation. Returns {support_grade, uncertainty_widening,
    limitations, downgraded_by}."""
    load_bearing_ids = set(load_bearing_ids or [s.variable_id for s in selections if s.selected])
    grade = base_grade if base_grade in _GRADE_RANK else "exploratory"
    widen, lims, downgraded_by = 1.0, [], []
    for s in selections:
        if not s.selected or s.variable_id not in load_bearing_ids:
            continue
        if _GRADE_RANK.get(s.support_grade, 3) > _GRADE_RANK.get(grade, 2):
            grade = s.support_grade
            downgraded_by.append(s.variable_id)
        widen = max(widen, s.uncertainty_widening)
        if s.runtime_status in ("exploratory", "highly_speculative"):
            lims.append(f"load-bearing {s.runtime_status} family {s.variable_id!r}: broadened parameter "
                        f"uncertainty (×{s.uncertainty_widening}); include in sensitivity analysis")
    return {"support_grade": grade, "uncertainty_widening": round(widen, 3),
            "limitations": lims, "downgraded_by": downgraded_by}


def resolve_transition_tier(variable_id: str, *, fitted_pack=False, domain_pack=False,
                            transfer_pack=False, published=False) -> dict:
    """Resolve the strongest available fallback tier for a family's transition (Part 4). An uncertain
    transition is NEVER removed — it drops to the strongest tier available, and Tier 6/7 materialize competing
    parameter/structural hypotheses through the shared world with disagreement propagated. Returns the tier +
    whether competing hypotheses should be simulated."""
    spec = get_persistent_variable(variable_id)
    if fitted_pack:
        tier = 1
    elif domain_pack:
        tier = 2
    elif transfer_pack:
        tier = 3
    elif published:
        tier = 4
    else:
        tier = _STATUS_DEFAULT_TIER.get(spec.runtime_status, 6)
    return {"variable_id": variable_id, "tier": tier, "description": TRANSITION_TIERS[tier],
            "simulate_competing_hypotheses": tier >= 6,
            "lower_support_grade": tier >= 5, "preserve_path_dependence": True,
            "note": ("uncertain transition kept and dropped to the strongest available tier — never removed"
                     if tier >= 5 else "validated/transfer transition available")}


def runtime_status_table() -> list:
    """The family-by-family runtime status table (the deliverable's final table source)."""
    rows = []
    for vid, spec in registered_variables().items():
        sel = select_families([vid])[0]
        rows.append({"family": vid, "runtime_status": spec.runtime_status,
                     "default_selectable": sel.selected, "transition_family": spec.transition_family,
                     "parameter_source": spec.transition_param_source, "transport_risk": spec.transport_risk,
                     "materializes_into": spec.materializes_into,
                     "production_usable": sel.selected,   # usable iff not blocked (validation ≠ usability)
                     "supporting_evidence": spec.supporting_evidence[:120]})
    return rows
