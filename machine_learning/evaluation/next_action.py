"""Next-action / next-choice / next-speaker family metrics.

Metrics: accuracy, macro-F1, no-action accuracy.
"""
from __future__ import annotations

from . import metrics as M

def evaluate(pairs, no_action_token="<NO_ACTION>"):
    """pairs = [(pred_label, true_label)]. Returns accuracy + macro-F1 + no-action accuracy."""
    res = {"n": len(pairs), "accuracy": M.accuracy(pairs), "macro_f1": M.macro_f1(pairs)}
    na = [(p, t) for p, t in pairs if t == no_action_token]
    if na:
        res["no_action_accuracy"] = M.accuracy(na)
    return res
