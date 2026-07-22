"""Trajectory-continuation family metrics.

Metrics: set/prefix similarity.
"""
from __future__ import annotations

from . import metrics as M

def evaluate(pairs):
    """pairs = [(pred_events, true_events)] lists; scored by first-step accuracy + Jaccard of action types."""
    first = [(p[0] if p else None, t[0] if t else None) for p, t in pairs]
    jacc = [M.jaccard_set(set(map(str,p)), set(map(str,t))) for p, t in pairs]
    return {"n": len(pairs), "first_step_accuracy": round(M.accuracy(first), 4),
            "set_jaccard": round(sum(jacc)/len(jacc), 4) if jacc else 0.0}
