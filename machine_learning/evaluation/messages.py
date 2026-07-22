"""Message-generation family metrics (semantic similarity + dialogue-act).

Metrics: token-F1 (semantic proxy), dialogue-act accuracy.
"""
from __future__ import annotations

from . import metrics as M

def evaluate(pairs, act_pairs=None):
    """pairs = [(pred_text, true_text)]; act_pairs = [(pred_act, true_act)] optional."""
    f1s = [M.token_f1(p, t) for p, t in pairs]
    res = {"n": len(pairs), "token_f1": round(sum(f1s)/len(f1s), 4) if f1s else 0.0}
    if act_pairs:
        res["dialogue_act_accuracy"] = M.accuracy(act_pairs)
    return res
