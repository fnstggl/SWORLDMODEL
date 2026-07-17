"""Phase 13 multi-stakeholder utility evaluation (Part 3) — decomposed, never one hidden scalar.

Input: per-particle OUTCOME dicts (the terminal readout of a matched rollout). Each stakeholder maps an
outcome to a utility; aggregations combine stakeholders into the ranking scalar. Rights and floors are
lexicographic: a violation cannot be bought back by aggregate gain — the action is marked violating and
excluded from recommendation (reported, not silently dropped).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class UtilityBreakdown:
    """Per-action utility evaluation across matched particles, fully decomposed."""
    action_id: str
    per_stakeholder: dict = field(default_factory=dict)   # sid -> [u per particle]
    aggregate: list = field(default_factory=list)         # [aggregate u per particle]
    rights_violations: int = 0
    floor_violations: dict = field(default_factory=dict)  # sid -> count of particles below floor
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        agg = self.aggregate
        return {"action_id": self.action_id,
                "per_stakeholder": {sid: {"mean": _mean(us), "q10": _q(us, 0.10), "q90": _q(us, 0.90)}
                                    for sid, us in self.per_stakeholder.items()},
                "aggregate_mean": _mean(agg), "aggregate_median": _q(agg, 0.5),
                "rights_violations": self.rights_violations,
                "floor_violations": dict(self.floor_violations), "notes": self.notes[:6]}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _q(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def cvar(xs, alpha=0.2):
    """Mean of the worst alpha fraction — the downside tail, not the average."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(1, int(alpha * len(s)))
    return sum(s[:k]) / k


def evaluate_utility(action_id: str, outcomes: list, spec) -> UtilityBreakdown:
    """outcomes: [outcome dict per matched particle]; spec: contracts.UtilitySpec."""
    bd = UtilityBreakdown(action_id=action_id)
    for s in spec.stakeholders:
        us = []
        below = 0
        for o in outcomes:
            u = s.utility(o)
            us.append(u)
            if s.floor is not None and u < s.floor:
                below += 1
            for right in s.rights:
                try:
                    if not right(o):
                        bd.rights_violations += 1
                except Exception:  # noqa: BLE001 — a broken rights predicate must not pass silently
                    bd.rights_violations += 1
                    bd.notes.append(f"rights predicate error for {s.stakeholder_id}")
        if below:
            bd.floor_violations[s.stakeholder_id] = below
        bd.per_stakeholder[s.stakeholder_id] = us
    bd.aggregate = _aggregate(bd.per_stakeholder, spec)
    return bd


def _aggregate(per_stakeholder: dict, spec) -> list:
    sids = list(per_stakeholder)
    if not sids:
        return []
    n = len(per_stakeholder[sids[0]])
    ws = {s.stakeholder_id: float(s.weight) for s in spec.stakeholders}
    z = sum(abs(w) for w in ws.values()) or 1.0
    agg = []
    for i in range(n):
        row = {sid: per_stakeholder[sid][i] for sid in sids}
        if spec.aggregation in ("weighted_sum", "cvar", "chance_constrained", "minimax_regret",
                                "pareto_only", "lexicographic"):
            # cvar/chance/regret/pareto act at the DISTRIBUTION level (robust.py); the per-particle
            # aggregate is still the weighted sum. Lexicographic ordering acts via floors/rights.
            agg.append(sum(ws[sid] * row[sid] for sid in sids) / z)
        elif spec.aggregation == "maximin":
            agg.append(min(row.values()))
        elif spec.aggregation == "nash_social_welfare":
            # Nash SW needs nonnegative utilities; shift is recorded as a note upstream if used.
            vals = [max(1e-9, row[sid]) for sid in sids]
            agg.append(math.exp(sum(math.log(v) for v in vals) / len(vals)))
        else:
            agg.append(sum(ws[sid] * row[sid] for sid in sids) / z)
    return agg


def pareto_frontier(breakdowns: list) -> list:
    """Non-dominated actions on stakeholder MEAN utilities. Returns [{action_id, means, dominated_by}]
    for every action, frontier first — callers get the frontier AND why the rest fell off it."""
    means = {b.action_id: {sid: _mean(us) for sid, us in b.per_stakeholder.items()}
             for b in breakdowns}
    rows = []
    for a, ma in means.items():
        dominated_by = None
        for b, mb in means.items():
            if a == b:
                continue
            keys = set(ma) | set(mb)
            if all(mb.get(k, 0.0) >= ma.get(k, 0.0) + 1e-12 or
                   abs(mb.get(k, 0.0) - ma.get(k, 0.0)) <= 1e-12 for k in keys) and \
               any(mb.get(k, 0.0) > ma.get(k, 0.0) + 1e-12 for k in keys):
                dominated_by = b
                break
        rows.append({"action_id": a, "stakeholder_means": {k: round(v, 6) for k, v in ma.items()},
                     "on_frontier": dominated_by is None, "dominated_by": dominated_by})
    rows.sort(key=lambda r: (not r["on_frontier"], -sum(r["stakeholder_means"].values())))
    return rows
