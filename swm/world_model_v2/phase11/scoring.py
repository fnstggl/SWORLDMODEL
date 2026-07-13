"""Phase 11 — reproducible plan scoring (spec §13).

The LLM does NOT choose the winner. Each candidate gets a vector of transparent COMPONENT scores (stored
separately, never collapsed into one unexplained number); the total is a fixed weighted sum and the plan
posterior is a softmax (Bayesian-model-weight style) over totals. The CURRENT (unchanged) plan is always
scored, so when there is no real structural evidence it wins on parsimony + continuity and the system decides
NOT to recompile. When the top candidates are within a margin, several are RETAINED as a normalized mixture
and the rollout continues across all of them (structural uncertainty preserved, not collapsed to top-1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# fixed, documented weights (reproducible; not tuned on test). Predictive fit dominates; parsimony + continuity
# protect against needless/destructive recompiles.
WEIGHTS = {
    "residual_reduction": 2.0,        # expected drop in predictive surprise from adopting this structure
    "evidence_fit": 1.2,              # does the candidate structurally address the evidenced change?
    "structural_plausibility": 0.8,   # reference-class prior over the proposed structure
    "mechanism_applicability": 0.6,
    "institutional_consistency": 0.5,
    "network_consistency": 0.4,
    "continuity": 1.0,                # fraction of prior state that survives migration (anti-destruction)
    "complexity_penalty": -0.6,       # parsimony
    "compute_cost": -0.3,
    "transport_risk": -0.5,
    "calibration": 0.4,
}
TEMPERATURE = 0.7
MIXTURE_MARGIN = 0.15                  # totals within this of the max are retained in the mixture


@dataclass
class CandidateScore:
    candidate_id: str = ""
    is_current_plan: bool = False
    components: dict = field(default_factory=dict)
    total: float = 0.0
    uncertainty: float = 0.0
    weight: float = 0.0               # posterior model weight (filled by normalize)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ScoreResult:
    scores: list = field(default_factory=list)         # [CandidateScore]
    mixture: list = field(default_factory=list)        # [{candidate_id, weight}] retained plans
    top_candidate_id: str = ""
    recompile_warranted: bool = False                  # False iff the CURRENT plan is (in) the mixture alone
    multimodal: bool = False
    notes: list = field(default_factory=list)

    def as_dict(self):
        return {"scores": [s.as_dict() for s in self.scores], "mixture": self.mixture,
                "top_candidate_id": self.top_candidate_id, "recompile_warranted": self.recompile_warranted,
                "multimodal": self.multimodal, "notes": self.notes}


def _components(cand, ops, plan, observation, fused, *, migration_report=None):
    """Transparent component scores in [~0,1] (penalties negative). Deterministic in the inputs."""
    is_current = cand.is_current_plan
    fam_impact = float(getattr(fused, "fused_probability", 0.0))
    exp_impact = 0.0
    for e in []:
        pass
    # expected terminal impact of the evidenced change (from the observation)
    exp_impact = float((getattr(observation, "uncertainty", {}) or {}).get("terminal_sensitivity", 0.5))

    if is_current:
        # the current plan explains NONE of the new structural evidence, but is maximally parsimonious/continuous
        residual_reduction = 0.0
        evidence_fit = 1.0 - fam_impact                # if strong structural evidence, current fits poorly
        complexity = 0.0
        continuity = 1.0
        compute = 0.0
        transport = 0.0
        struct_plaus = 0.6
        mech_appl = 1.0
        inst_cons = 1.0
        net_cons = 1.0
        calib = 0.7
    else:
        full = any(t.op == "full_recompile" for t in ops)
        addresses = 1.0 if fused.scope_candidates else 0.6
        # a candidate that adds the evidenced structure is credited with removing the surprise it explains
        residual_reduction = round(fam_impact * exp_impact * (1.0 if not full else 0.9), 4)
        evidence_fit = round(0.5 + 0.5 * addresses, 4)
        complexity = float(getattr(cand, "complexity", len(cand.changed_components)))
        # migration completeness → continuity; full recompile discards most continuity
        if migration_report is not None:
            continuity = float(migration_report.get("object_retention_rate", 0.7))
        else:
            continuity = 0.35 if full else 0.85
        compute = 3.0 if full else 1.0
        transport = 0.6 if full else 0.2
        struct_plaus = 0.5 if full else 0.65
        mech_appl = float((cand.mechanism_applicability or {}).get("score", 0.8))
        inst_cons = 1.0
        net_cons = 1.0
        calib = 0.6

    comp = {
        "residual_reduction": residual_reduction,
        "evidence_fit": round(evidence_fit, 4),
        "structural_plausibility": struct_plaus,
        "mechanism_applicability": mech_appl,
        "institutional_consistency": inst_cons,
        "network_consistency": net_cons,
        "continuity": round(continuity, 4),
        "complexity_penalty": round(complexity, 4),
        "compute_cost": round(compute, 4),
        "transport_risk": round(transport, 4),
        "calibration": calib,
    }
    return comp


def score_candidates(candidates_with_ops, plan, observation, fused, *, migration_reports=None) -> ScoreResult:
    """Score every candidate (including the current plan); return the normalized plan-posterior mixture."""
    migration_reports = migration_reports or {}
    scored = []
    for cand, ops in candidates_with_ops:
        comp = _components(cand, ops, plan, observation, fused,
                           migration_report=migration_reports.get(cand.candidate_id))
        total = sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)
        # uncertainty ≈ spread of weighted contributions (how score-determining any one component is)
        contribs = [abs(WEIGHTS[k] * comp[k]) for k in WEIGHTS]
        m = sum(contribs) / len(contribs)
        unc = round(math.sqrt(sum((c - m) ** 2 for c in contribs) / len(contribs)), 4)
        scored.append(CandidateScore(candidate_id=cand.candidate_id, is_current_plan=cand.is_current_plan,
                                     components=comp, total=round(total, 4), uncertainty=unc))

    # softmax over totals → posterior model weights (Bayesian-model-weight style, reproducible)
    mx = max(s.total for s in scored)
    for s in scored:
        s.weight = math.exp((s.total - mx) / TEMPERATURE)
    z = sum(s.weight for s in scored) or 1.0
    for s in scored:
        s.weight = round(s.weight / z, 4)

    scored.sort(key=lambda s: s.weight, reverse=True)
    top = scored[0]
    mixture = [{"candidate_id": s.candidate_id, "weight": s.weight}
               for s in scored if (top.total - s.total) <= MIXTURE_MARGIN * max(1.0, abs(top.total))]
    zt = sum(m["weight"] for m in mixture) or 1.0
    for m in mixture:
        m["weight"] = round(m["weight"] / zt, 4)

    current_ids = {s.candidate_id for s in scored if s.is_current_plan}
    mixture_ids = {m["candidate_id"] for m in mixture}
    # recompilation is warranted iff a NON-current plan is in (or dominates) the retained mixture
    recompile = bool(mixture_ids - current_ids)
    multimodal = len(mixture) > 1
    res = ScoreResult(scores=scored, mixture=mixture, top_candidate_id=top.candidate_id,
                      recompile_warranted=recompile, multimodal=multimodal)
    if not recompile:
        res.notes.append("current plan retained — evidence does not warrant recompilation")
    elif multimodal:
        res.notes.append(f"structural uncertainty retained across {len(mixture)} plans (mixture, not top-1)")
    return res
