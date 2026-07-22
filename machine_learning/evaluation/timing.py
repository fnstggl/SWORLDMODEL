"""Timing family metrics (with censoring awareness).

Metrics: timing MAE on uncensored + censored fraction.
"""
from __future__ import annotations

from . import metrics as M

def evaluate(pairs, censored=None):
    """pairs = [(pred_seconds, true_seconds)] for UNCENSORED; censored = list[bool]."""
    res = {"n": len(pairs), "timing_mae": round(M.mae(pairs), 3), "rmse": round(M.rmse(pairs), 3)}
    if censored is not None:
        res["censored_fraction"] = round(sum(1 for c in censored if c)/max(len(censored),1), 4)
    return res
