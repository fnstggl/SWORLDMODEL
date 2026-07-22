"""Causal intervention / policy-value / ranking family metrics.

Metrics: effect MAE, IPS value, nDCG.
"""
from __future__ import annotations

from . import metrics as M

def effect_error(pairs):
    """pairs = [(pred_effect, true_effect)]."""
    return {"effect_mae": round(M.mae(pairs), 5), "n": len(pairs)}

def ips_value(logged):
    """logged = [(reward, propensity, pi_action_prob)] -> inverse-propensity-scored value."""
    vals = [r * (pi/max(ps,1e-6)) for r, ps, pi in logged if ps]
    return {"ips_value": round(sum(vals)/len(vals), 5) if vals else 0.0, "n": len(logged)}

def ranking(ranking_list, relevance, k=10):
    return {"ndcg@%d"%k: round(M.ndcg_at_k(ranking_list, relevance, k), 4)}
