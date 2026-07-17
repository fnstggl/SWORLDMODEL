"""Phase 13 robust decision evaluation (Parts 15–16) — never rank on the mean alone.

Builds the full per-action report from a MatchedBundle + UtilitySpec: expected utility, distribution
quantiles, CVaR, P(improvement)/P(material improvement)/P(harm), constraint-violation probabilities,
expected + minimax regret, per-structural-hypothesis values with a FRAGILITY flag (an action that wins
under only one supported hypothesis is reported as such, Part 16), implementation cost, reversibility.
The ranking objective is selected by the contract's RiskSpec — expected | cvar | lower_confidence |
minimax_regret | worst_hypothesis — and the chosen objective is recorded on the result.
"""
from __future__ import annotations

from swm.world_model_v2.phase13.counterfactual import paired_report, variance_reduction
from swm.world_model_v2.phase13.utility import cvar as _cvar, evaluate_utility


def evaluate_bundle(bundle, actions, problem) -> dict:
    """Returns {action_id: evaluation dict}, plus '_ranking' and '_regret' blocks."""
    spec = problem.utility
    by_id = {a.action_id: a for a in actions}
    breakdowns, agg = {}, {}
    for aid, arm in bundle.arms.items():
        bd = evaluate_utility(aid, arm.outcomes, spec)
        breakdowns[aid] = bd
        # implementation costs subtract in utility units (documented, not hidden)
        a = by_id.get(aid)
        cost = (a.direct_cost + a.indirect_cost) if a is not None else 0.0
        cost += float(problem.implementation_costs.get(aid, 0.0))
        agg[aid] = [u - cost for u in bd.aggregate]

    ref = bundle.reference
    n = bundle.n_particles
    material = _material_threshold(agg)
    evals = {}
    for aid, us in agg.items():
        a = by_id.get(aid)
        arm = bundle.arms[aid]
        diffs = [x - y for x, y in zip(us, agg[ref])] if ref in agg else []
        ev = {
            "action_id": aid,
            "operation": a.operation if a is not None else "",
            "expected_utility": _m(us), "median_utility": _q(us, 0.5),
            "q10": _q(us, 0.10), "q90": _q(us, 0.90),
            "cvar": round(_cvar(us, problem.utility.cvar_alpha), 6),
            "utility_distribution": _hist(us),
            "p_improvement": round(sum(1 for d in diffs if d > 0) / n, 4) if diffs else None,
            "p_material_improvement": (round(sum(1 for d in diffs if d > material) / n, 4)
                                       if diffs else None),
            "p_harm": round(sum(1 for d in diffs if d < 0) / n, 4) if diffs else None,
            "paired_vs_reference": paired_report(diffs) if diffs else {},
            "variance_reduction": variance_reduction(us, agg[ref]) if ref in agg else {},
            "stakeholder_breakdown": breakdowns[aid].summary(),
            "constraint_violations": _constraints(arm.outcomes, problem),
            "rights_violations": breakdowns[aid].rights_violations,
            "floor_violations": breakdowns[aid].floor_violations,
            "implementation_cost": round((a.direct_cost + a.indirect_cost) if a else 0.0, 6),
            "reversible": a.is_reversible() if a is not None else True,
            "by_hypothesis": _by_hypothesis(us, bundle.hypothesis_assignment),
            "n_state_deltas": arm.n_deltas,
        }
        evals[aid] = ev

    # regret: per-particle best across arms
    regret = {aid: 0.0 for aid in agg}
    worst_regret = {aid: 0.0 for aid in agg}
    for i in range(n):
        row = {aid: agg[aid][i] for aid in agg}
        best = max(row.values())
        for aid, v in row.items():
            regret[aid] += (best - v) / n
            worst_regret[aid] = max(worst_regret[aid], best - v)
    for aid in evals:
        evals[aid]["expected_regret"] = round(regret[aid], 6)
        evals[aid]["max_regret"] = round(worst_regret[aid], 6)
        evals[aid]["fragile"] = _fragile(evals[aid]["by_hypothesis"], evals, aid)

    evals["_ranking"] = rank(evals, problem)
    evals["_regret"] = {"minimax_regret_action": min((aid for aid in agg),
                                                     key=lambda a: worst_regret[a])}
    return evals


def rank(evals: dict, problem) -> dict:
    """Rank feasible arms by the RiskSpec's robustness objective; report the objective used."""
    mode = problem.risk.robustness
    rows = []
    for aid, ev in evals.items():
        if aid.startswith("_"):
            continue
        if ev["rights_violations"] > 0 or ev["floor_violations"]:
            rows.append((aid, float("-inf"), "excluded: rights/floor violation"))
            continue
        chance_bad = any(c["kind"] == "chance" and c["violated"]
                         for c in ev["constraint_violations"])
        if chance_bad:
            rows.append((aid, float("-inf"), "excluded: chance constraint violated"))
            continue
        if mode == "cvar":
            score = ev["cvar"]
        elif mode == "lower_confidence":
            score = ev["q10"] if problem.risk.lower_confidence <= 0.15 else ev["q10"]
        elif mode == "minimax_regret":
            score = -ev["max_regret"]
        elif mode == "worst_hypothesis":
            byh = ev["by_hypothesis"]
            score = min((h["mean"] for h in byh.values()), default=ev["expected_utility"])
        else:
            score = ev["expected_utility"]
        if problem.risk.ambiguity_aversion > 0:
            byh = ev["by_hypothesis"]
            worst = min((h["mean"] for h in byh.values()), default=score)
            lam = min(1.0, max(0.0, problem.risk.ambiguity_aversion))
            score = (1 - lam) * score + lam * worst
        rows.append((aid, score, mode))
    rows.sort(key=lambda r: -r[1])
    return {"objective": mode, "order": [{"action_id": aid, "score": (round(s, 6)
                                          if s != float("-inf") else "excluded"), "note": note}
                                         for aid, s, note in rows]}


def _constraints(outcomes, problem) -> list:
    out = []
    n = max(1, len(outcomes))
    for c in problem.constraints or []:
        if c.outcome_pred is None:
            continue
        bad = 0
        for o in outcomes:
            try:
                if not c.outcome_pred(o):
                    bad += 1
            except Exception:  # noqa: BLE001 — failing closed
                bad += 1
        p_viol = bad / n
        out.append({"constraint_id": c.constraint_id, "kind": c.kind,
                    "p_violation": round(p_viol, 4),
                    "violated": (p_viol > (c.max_prob if c.kind == "chance" else 0.0))})
    return out


def _by_hypothesis(us, assignment) -> dict:
    groups = {}
    for u, h in zip(us, assignment):
        groups.setdefault(h, []).append(u)
    return {h: {"mean": _m(g), "n": len(g)} for h, g in groups.items()}


def _fragile(byh: dict, evals: dict, aid: str) -> bool:
    """Fragile = this action's win is carried by ONE hypothesis: it leads under some hypothesis but is
    below the field's median under the others (only meaningful with >=2 hypotheses)."""
    if len(byh) < 2:
        return False
    leads, trails = 0, 0
    for h in byh:
        field_means = [e["by_hypothesis"].get(h, {}).get("mean") for k, e in evals.items()
                       if not k.startswith("_") and e["by_hypothesis"].get(h)]
        mine = byh[h]["mean"]
        if not field_means:
            continue
        if mine >= max(field_means) - 1e-12:
            leads += 1
        elif mine < sorted(field_means)[len(field_means) // 2]:
            trails += 1
    return leads >= 1 and trails >= 1


def _m(xs):
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def _q(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 6)


def _hist(xs, bins: int = 8) -> dict:
    if not xs:
        return {}
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return {str(round(lo, 4)): len(xs)}
    w = (hi - lo) / bins
    out = {}
    for x in xs:
        b = min(bins - 1, int((x - lo) / w))
        key = f"[{round(lo + b * w, 4)},{round(lo + (b + 1) * w, 4)})"
        out[key] = out.get(key, 0) + 1
    return out


def _material_threshold(agg: dict) -> float:
    """Material improvement = 10% of the cross-arm utility spread (documented heuristic)."""
    all_us = [u for us in agg.values() for u in us]
    if not all_us:
        return 0.0
    return 0.1 * (max(all_us) - min(all_us))
