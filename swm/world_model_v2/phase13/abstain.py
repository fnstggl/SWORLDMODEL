"""Phase 13 abstention & safety gates (Part 23) — refuse with reasons + what is needed, never fabricate.

Simulation, recommendation, approval and execution are separate concerns; this layer only ever
recommends, and every result carries `human_approval_required`. The gates below run BEFORE simulation;
post-simulation exclusions (rights/floors/chance constraints) live in robust.rank. Prohibited-harm
classes are hard-coded conservative: an action whose operation or params match coercive/deceptive
manipulation of protected or vulnerable targets is rejected at feasibility and re-checked here."""
from __future__ import annotations

from swm.world_model_v2.phase13.contracts import Abstention

_PROHIBITED_MARKERS = ("coerce", "threaten_unlawful", "deceive", "blackmail", "doxx",
                       "target_protected_class", "exploit_vulnerable")


def abstention_check(problem, *, defects: list, missing: list, feasible: list):
    """Returns None (proceed), or an Abstention. An underspecified-utility abstention still lets the
    pipeline continue to a Pareto frontier (partial['continue_for_pareto'])."""
    reasons, needed = [], []

    if defects:
        reasons.append({"code": "invalid_contract", "detail": "; ".join(defects)[:300]})
        needed.append("fix the decision contract defects")
    if not problem.decision_maker:
        reasons.append({"code": "authority_unclear", "detail": "no decision-maker declared"})
        needed.append("declare the decision-maker and their authority")

    for a in feasible:
        text = f"{a.operation} {a.object} {sorted(map(str, a.params.values()))}".lower()
        if any(m in text for m in _PROHIBITED_MARKERS):
            reasons.append({"code": "prohibited_harm",
                            "detail": f"action {a.action_id} matches a prohibited-harm marker"})
            needed.append(f"remove action {a.action_id} from the candidate set")

    only_baselines = all(a.provenance == "baseline" for a in feasible)
    if only_baselines and not problem.candidate_actions:
        reasons.append({"code": "no_substantive_actions",
                        "detail": "feasibility rejected every non-baseline action"})
        needed.append("grant authority/resources or supply candidate actions")

    if missing:
        ab = Abstention(reasons=[{"code": "utility_underspecified", "detail": m} for m in missing]
                        + reasons,
                        needed=needed + ["supply stakeholder utilities / an aggregation rule"],
                        partial={"continue_for_pareto": not reasons})
        return ab
    if reasons:
        return Abstention(reasons=reasons, needed=needed)
    return None
