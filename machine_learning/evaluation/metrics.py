"""Metric primitives. Each task family uses the subset that fits it — there is NO single
universal metric. Pure-python (no sklearn) so evaluation runs anywhere.
"""
from __future__ import annotations

import math
from collections import Counter


def accuracy(pairs: list[tuple]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for p, t in pairs if p == t) / len(pairs)


def macro_f1(pairs: list[tuple]) -> float:
    labels = set(t for _, t in pairs) | set(p for p, _ in pairs)
    f1s = []
    for lab in labels:
        tp = sum(1 for p, t in pairs if p == lab and t == lab)
        fp = sum(1 for p, t in pairs if p == lab and t != lab)
        fn = sum(1 for p, t in pairs if p != lab and t == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def brier_binary(pairs: list[tuple]) -> float:
    """pairs = [(prob_of_1, label_0_or_1)]."""
    if not pairs:
        return 0.0
    return sum((p - t) ** 2 for p, t in pairs) / len(pairs)


def log_loss_binary(pairs: list[tuple], eps: float = 1e-9) -> float:
    if not pairs:
        return 0.0
    s = 0.0
    for p, t in pairs:
        p = min(max(p, eps), 1 - eps)
        s += -(t * math.log(p) + (1 - t) * math.log(1 - p))
    return s / len(pairs)


def mae(pairs: list[tuple]) -> float:
    vals = [(p, t) for p, t in pairs if _num(p) is not None and _num(t) is not None]
    if not vals:
        return float("nan")
    return sum(abs(_num(p) - _num(t)) for p, t in vals) / len(vals)


def rmse(pairs: list[tuple]) -> float:
    vals = [(p, t) for p, t in pairs if _num(p) is not None and _num(t) is not None]
    if not vals:
        return float("nan")
    return math.sqrt(sum((_num(p) - _num(t)) ** 2 for p, t in vals) / len(vals))


def expected_calibration_error(pairs: list[tuple], n_bins: int = 10) -> float:
    """pairs = [(prob, label)]. Standard ECE."""
    if not pairs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for p, t in pairs:
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append((p, t))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(t for _, t in b) / len(b)
        ece += (len(b) / len(pairs)) * abs(conf - acc)
    return ece


def distribution_distance(pred: dict, true: dict) -> dict:
    """Total-variation + L1 between two categorical distributions."""
    keys = set(pred) | set(true)
    tv = 0.5 * sum(abs(pred.get(k, 0.0) - true.get(k, 0.0)) for k in keys)
    l1 = sum(abs(pred.get(k, 0.0) - true.get(k, 0.0)) for k in keys)
    return {"total_variation": tv, "l1": l1}


def token_f1(pred: str, true: str) -> float:
    pt, tt = Counter(pred.lower().split()), Counter(true.lower().split())
    overlap = sum((pt & tt).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(sum(pt.values()), 1)
    rec = overlap / max(sum(tt.values()), 1)
    return 2 * prec * rec / (prec + rec)


def jaccard_set(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def ndcg_at_k(ranking: list, relevance: dict, k: int = 10) -> float:
    def dcg(items):
        return sum((relevance.get(it, 0.0)) / math.log2(i + 2) for i, it in enumerate(items[:k]))
    ideal = sorted(relevance, key=lambda x: relevance[x], reverse=True)
    idcg = dcg(ideal)
    return dcg(ranking) / idcg if idcg > 0 else 0.0


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
