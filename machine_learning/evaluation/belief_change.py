"""Belief-change family metrics.

Metrics: belief MAE + direction accuracy.
"""
from __future__ import annotations

from . import metrics as M

def evaluate(pairs):
    """pairs = [(pred_delta, true_delta)] numeric belief shifts."""
    dir_pairs = [((p > 0) - (p < 0), (t > 0) - (t < 0)) for p, t in pairs
                 if isinstance(p,(int,float)) and isinstance(t,(int,float))]
    return {"n": len(pairs), "belief_change_mae": round(M.mae(pairs), 4),
            "direction_accuracy": round(M.accuracy(dir_pairs), 4) if dir_pairs else None}
