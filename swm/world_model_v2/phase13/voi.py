"""Phase 13 value of information (Part 14) — computed from the matched utility matrix, not asserted.

Given the paired per-particle utility matrix U[action][particle] that the matched evaluator already
produced, EVPI is exact over the represented uncertainty:  E[max_a U] − max_a E[U]  — the value of
knowing the particle (the hidden state) before committing. EVSI for a candidate observation partitions
the SAME particles by the observation's simulated answer and re-optimizes within each cell:
E_signal[ max_a E[U | signal] ] − max_a E[U]. Because both use the matched matrix, information value is
never confounded with world luck. The recommendation to gather information triggers when net EVSI
(minus observation cost and delay risk) exceeds the best immediate commitment's margin.
"""
from __future__ import annotations


def evpi(agg_utils: dict) -> dict:
    """agg_utils: {action_id: [utility per particle]} (implementation costs already applied)."""
    ids = [a for a in agg_utils if not a.startswith("_")]
    if not ids:
        return {"evpi": 0.0}
    n = len(agg_utils[ids[0]])
    best_ex_ante = max(sum(agg_utils[a]) / n for a in ids)
    perfect = sum(max(agg_utils[a][i] for a in ids) for i in range(n)) / n
    return {"evpi": round(perfect - best_ex_ante, 6),
            "best_ex_ante": round(best_ex_ante, 6), "e_max_posterior": round(perfect, 6)}


def evsi(agg_utils: dict, signal_of_particle: list, *, cost: float = 0.0,
         signal_name: str = "observation") -> dict:
    """signal_of_particle[i] = the observation's simulated answer under particle i (any hashable).
    Partition particles by signal, re-optimize per cell, weight by cell mass."""
    ids = [a for a in agg_utils if not a.startswith("_")]
    n = len(signal_of_particle)
    if not ids or n == 0:
        return {"evsi": 0.0, "signal": signal_name}
    cells = {}
    for i, s in enumerate(signal_of_particle):
        cells.setdefault(s, []).append(i)
    best_ex_ante = max(sum(agg_utils[a]) / n for a in ids)
    post = 0.0
    per_cell = {}
    for s, idx in cells.items():
        w = len(idx) / n
        cell_best_action = max(ids, key=lambda a: sum(agg_utils[a][i] for i in idx) / len(idx))
        cell_val = sum(agg_utils[cell_best_action][i] for i in idx) / len(idx)
        post += w * cell_val
        per_cell[str(s)] = {"mass": round(w, 4), "best_action": cell_best_action,
                            "value": round(cell_val, 6)}
    gross = post - best_ex_ante
    return {"signal": signal_name, "evsi_gross": round(gross, 6), "cost": cost,
            "evsi_net": round(gross - cost, 6), "cells": per_cell,
            "would_change_decision": len({c["best_action"] for c in per_cell.values()}) > 1}


def information_report(agg_utils: dict, candidate_observations: list, *, delay_risk: float = 0.0) -> dict:
    """candidate_observations: [{"name":..., "signal_of_particle":[...], "cost": float}].
    Reports EVPI, per-candidate EVSI, and whether gathering information DOMINATES committing now."""
    rep = {"evpi": evpi(agg_utils), "candidates": [], "delay_risk": delay_risk}
    best = None
    for c in candidate_observations or []:
        r = evsi(agg_utils, c.get("signal_of_particle") or [], cost=float(c.get("cost", 0.0)),
                 signal_name=str(c.get("name", "obs")))
        r["evsi_net_after_delay"] = round(r["evsi_net"] - delay_risk, 6)
        rep["candidates"].append(r)
        if best is None or r["evsi_net_after_delay"] > best["evsi_net_after_delay"]:
            best = r
    rep["recommend_gathering"] = bool(best and best["evsi_net_after_delay"] > 0 and
                                      best.get("would_change_decision"))
    rep["best_observation"] = best["signal"] if best else None
    rep["reduces_uncertainty_about"] = ("hidden-state particles that flip the best action"
                                        if rep["recommend_gathering"] else None)
    return rep
