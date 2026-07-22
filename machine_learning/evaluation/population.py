"""Population-response + population-time-series family metrics.

Metrics: rate MAE + distribution distance.
"""
from __future__ import annotations

from . import metrics as M

def evaluate(rate_pairs=None, dist_pairs=None):
    """rate_pairs = [(pred_rate, true_rate)]; dist_pairs = [(pred_dist, true_dist)] dicts."""
    res = {}
    if rate_pairs:
        res["rate_mae"] = round(M.mae(rate_pairs), 5)
    if dist_pairs:
        tvs = [M.distribution_distance(p, t)["total_variation"] for p, t in dist_pairs]
        res["mean_total_variation"] = round(sum(tvs)/len(tvs), 4) if tvs else 0.0
    res["n"] = len(rate_pairs or dist_pairs or [])
    return res
