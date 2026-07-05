"""Decision lift — the buyer's metric, as a reusable library function (audit E.2, EXP-004).

Calibration is not what anyone pays for. The operator acts on the top-K% the model ranks highest
and asks: did I capture more winners / more outcome than my current method? This module computes
that, for any set of predictions vs. baselines, with a bootstrap CI on the lift over the *strong*
baseline (the honest bar — beating random is easy).

- `hit_capture(y, score, k)`: fraction of all positives captured in the top-K by `score`.
- `decision_lift(y, model_score, baseline_score, ...)`: hit-capture curves for model / baseline /
  random / oracle across K, plus a paired bootstrap on (model - baseline) capture at a target K.
"""
from __future__ import annotations

import random


def hit_capture(y: list[int], score: list[float], k: float) -> float:
    n = len(y)
    total_pos = sum(y)
    if total_pos == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: score[i], reverse=True)
    top = order[:max(1, int(round(k * n)))]
    return sum(y[i] for i in top) / total_pos


def decision_lift(y: list[int], model_score: list[float], baseline_score: list[float], *,
                  ks: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3), target_k: float = 0.2,
                  n_boot: int = 2000, seed: int = 0) -> dict:
    n = len(y)
    rng = random.Random(seed)
    random_capture = {k: round(k, 4) for k in ks}   # expected capture of random ~ k
    curve = []
    for k in ks:
        curve.append({
            "k": k,
            "model": round(hit_capture(y, model_score, k), 4),
            "baseline": round(hit_capture(y, baseline_score, k), 4),
            "random": random_capture[k],
            "oracle": round(hit_capture(y, [float(v) for v in y], k), 4),
        })
    # paired bootstrap on capture difference at target_k
    def cap(idx, sc):
        yy = [y[i] for i in idx]
        tp = sum(yy)
        if tp == 0:
            return 0.0
        order = sorted(range(len(idx)), key=lambda j: sc[idx[j]], reverse=True)
        top = order[:max(1, int(round(target_k * len(idx))))]
        return sum(yy[j] for j in top) / tp
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(cap(idx, model_score) - cap(idx, baseline_score))
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    p_no_lift = sum(1 for d in diffs if d <= 0) / n_boot
    return {
        "curve": curve, "target_k": target_k,
        "lift_over_baseline_at_target_k": round(
            hit_capture(y, model_score, target_k) - hit_capture(y, baseline_score, target_k), 4),
        "lift_ci95": [round(lo, 4), round(hi, 4)], "p_no_lift_over_baseline": round(p_no_lift, 4),
        "n": n, "n_positives": sum(y),
    }
