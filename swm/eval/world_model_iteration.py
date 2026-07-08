"""Reusable iteration helpers (Phase 2/9): error clustering + failure classification + factor
proposal, used by the simulation iteration loop and the world-model reports.

Kept small and dependency-free. These turn a benchmark's per-item errors into a ranked list of
where the model loses and a machine-suggested next fix, so the iteration loop's "identify biggest
loss source -> propose targeted fix" step is not hand-waved.
"""
from __future__ import annotations

from swm.eval.metrics import log_loss


def cluster_errors(rows: list[dict], y: list[int], model_p: list[float], ref_p: list[float],
                   slicers: dict) -> list[dict]:
    """For each named slice, the model's log loss vs a reference (e.g. raw LLM + context). Returns
    slices ranked by how much the model TRAILS the reference (biggest failure first)."""
    out = []
    for name, fn in slicers.items():
        idx = [i for i, r in enumerate(rows) if fn(r)]
        if len(idx) < 12 or sum(y[i] for i in idx) < 3:
            continue
        ys = [y[i] for i in idx]
        mp = [min(1 - 1e-6, max(1e-6, model_p[i])) for i in idx]
        rp = [min(1 - 1e-6, max(1e-6, ref_p[i])) for i in idx]
        gap = log_loss(ys, mp) - log_loss(ys, rp)
        out.append({"slice": name, "n": len(idx), "model_ll": round(log_loss(ys, mp), 4),
                    "ref_ll": round(log_loss(ys, rp), 4), "gap": round(gap, 4)})
    out.sort(key=lambda d: d["gap"], reverse=True)
    return out


# heuristic map from failing slice -> a failure class and a candidate fix
_FAILURE_MAP = {
    "cold": ("weak actor state / no history", "raise the LLM gate weight on cold entities"),
    "low_context": ("no useful retrieval", "defer to calibrated LLM prior (gate ~0)"),
    "semantics": ("weak action representation", "add richer LLM-extracted text features"),
    "domain": ("bad source-reputation transition", "couple domain reputation into exposure/front-page"),
    "strong_domain": ("segment priors / small-n noise", "regularize; more trajectories; shrink to prior"),
    "ai": ("segment affinity mis-set", "refit AI/ML segment affinity + weight"),
    "security": ("segment affinity mis-set", "refit security segment affinity + weight"),
}


def classify_and_propose(worst_slice: str) -> dict:
    for key, (cls, fix) in _FAILURE_MAP.items():
        if key in worst_slice.lower():
            return {"failure_class": cls, "proposed_fix": fix}
    return {"failure_class": "calibration / over-simulation noise",
            "proposed_fix": "recalibrate readout; reduce trajectory variance"}
