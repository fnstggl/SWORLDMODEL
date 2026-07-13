"""Phase 10 (continuation) — competing rule-model EXECUTION + live Phase-3 multi-particle institutions.

When the institution's rules/membership/authority/interpretation are uncertain, we do NOT collapse to one
mean institution. Instead:
  1. `institutional_hypothesis_posterior` draws REAL posterior weights over discrete institutional hypotheses
     from the merged Phase-3 engine (`infer_compositional_posterior` — a Dirichlet over a hypothesis simplex).
     No Phase-10-local posterior is invented.
  2. `execute_competing_models` executes each structurally-distinct hypothesis SEPARATELY (its own thresholds
     / quorum / membership) and aggregates the terminal outcomes as a WEIGHTED distribution over particles —
     incompatible rule models are never averaged into a single rule.
  3. `divergence` reports whether the hypotheses actually disagree on the terminal outcome (so the forensic
     trace shows the structural uncertainty MATTERS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.institutions_v2.decisions import ThresholdSpec, evaluate_decision


@dataclass
class RuleHypothesis:
    """One competing institutional interpretation (Part 13). `overrides` may set threshold/quorum/eligible."""
    model_id: str
    weight: float = 0.0
    threshold: ThresholdSpec | None = None
    eligible: list = field(default_factory=list)
    recused: set = field(default_factory=set)
    supporting_evidence: list = field(default_factory=list)
    contradicting_evidence: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)


def institutional_hypothesis_posterior(hypothesis_ids, prior_alpha, evidence_counts, *, seed=0):
    """Real Phase-3 posterior weights over a set of competing institutional hypotheses. `evidence_counts` is
    a list of dicts {counts: {hyp_id: n}, reliability, source, dependence_group} — e.g. how many sources
    support each interpretation. Returns {hyp_id: posterior_weight} + the CompositionalPosteriorResult."""
    from swm.world_model_v2.phase3_posterior import infer_compositional_posterior
    obs = [{"counts": {hid: ec.get("counts", {}).get(hid, 0) for hid in hypothesis_ids},
            "reliability": ec.get("reliability", 1.0), "source": ec.get("source", ""),
            "dependence_group": ec.get("dependence_group", "")} for ec in evidence_counts]
    res = infer_compositional_posterior(list(hypothesis_ids), list(prior_alpha), obs, seed=seed)
    weights = {hid: res.posterior_mean[i] for i, hid in enumerate(hypothesis_ids)}
    return weights, res


@dataclass
class ParticleOutcome:
    model_id: str
    weight: float
    passed: bool
    decision: dict
    assumptions: list = field(default_factory=list)


def execute_competing_models(hypotheses: list, votes: dict, *, base_eligible: list) -> dict:
    """Execute EACH hypothesis's decision separately on the same votes, then aggregate as a weighted
    distribution over outcomes. Returns the per-particle outcomes, the weighted P(pass), and a divergence
    report. Incompatible rules are kept as separate particles — never averaged into one threshold."""
    parts = []
    for h in hypotheses:
        spec = h.threshold or ThresholdSpec("simple_majority", 0.5, base="present")
        elig = h.eligible or base_eligible
        res = evaluate_decision(spec, votes, eligible=elig, recused=set(h.recused))
        parts.append(ParticleOutcome(h.model_id, h.weight, res.passed, res.as_dict(), h.assumptions))
    wsum = sum(p.weight for p in parts) or 1.0
    p_pass = sum(p.weight for p in parts if p.passed) / wsum
    outcomes = {}
    for p in parts:
        k = "pass" if p.passed else "fail"
        outcomes[k] = outcomes.get(k, 0.0) + p.weight / wsum
    return {"particles": [p.__dict__ for p in parts], "p_pass_weighted": round(p_pass, 4),
            "terminal_distribution": {k: round(v, 4) for k, v in outcomes.items()},
            "divergence": divergence(parts),
            "note": "weighted over competing rule models; incompatible rules kept as separate particles "
                    "(not averaged) — structural uncertainty propagates to the terminal distribution"}


def divergence(parts: list) -> dict:
    """Do the competing hypotheses disagree on the terminal outcome? If they all agree, the structural
    uncertainty is immaterial for this scenario; if they split, it MATTERS (report the split)."""
    passed = {p.model_id for p in parts if p.passed}
    failed = {p.model_id for p in parts if not p.passed}
    disagree = bool(passed and failed)
    return {"disagree": disagree, "models_pass": sorted(passed), "models_fail": sorted(failed),
            "interpretation_matters": disagree,
            "reason": ("competing rule models produce DIFFERENT terminal outcomes on these votes — the "
                       "institutional interpretation is outcome-determining" if disagree else
                       "all competing models agree on the outcome here — interpretation is immaterial")}
