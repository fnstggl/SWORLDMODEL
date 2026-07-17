"""Pure scoring metrics (no outcome-store import — unit-testable anywhere)."""
from __future__ import annotations

import math
import random


def _ll(p, y):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _auroc(pairs):
    pos = sorted(p for p, y in pairs if y == 1)
    neg = sorted(p for p, y in pairs if y == 0)
    if not pos or not neg:
        return None
    wins = sum((sum(1 for n in neg if p > n) + 0.5 * sum(1 for n in neg if p == n))
               for p in pos)
    return round(wins / (len(pos) * len(neg)), 4)


def _ece(pairs, bins=10):
    if not pairs:
        return None
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if sel:
            conf = sum(p for p, _ in sel) / len(sel)
            acc = sum(y for _, y in sel) / len(sel)
            tot += len(sel) / len(pairs) * abs(conf - acc)
    return round(tot, 4)


def _cal_line(pairs):
    """Logistic recalibration line (slope, intercept) via 40-step Newton on logit(p)."""
    import statistics
    if len(pairs) < 8:
        return None, None
    xs = [math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6)))
          for p, _ in pairs]
    ys = [y for _, y in pairs]
    a, b = 0.0, 1.0
    for _ in range(40):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for x, y in zip(xs, ys):
            z = a + b * x
            mu = 1 / (1 + math.exp(-max(-30, min(30, z))))
            w = mu * (1 - mu)
            g_a += mu - y
            g_b += (mu - y) * x
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-9:
            break
        a -= (g_a * h_bb - g_b * h_ab) / det
        b -= (g_b * h_aa - g_a * h_ab) / det
    return round(b, 3), round(a, 3)


def crps_event_time(evt: dict, *, outcome: int, resolution_ts: float, deadline_ts: float):
    """Censoring-aware discrete CRPS over the forecast CDF grid: truth CDF is a step at the
    realized event time when the event occurred (YES under occurrence polarity), else 0 through
    the horizon (censored)."""
    grid, cdf = evt.get("cdf_grid_ts") or [], evt.get("cdf") or []
    if not grid or not cdf or len(grid) != len(cdf):
        return None
    occurred = outcome == 1 if str(evt.get("occurrence_resolves", "yes")).endswith("yes") \
        else outcome == 0
    tot = 0.0
    for g, f in zip(grid, cdf):
        truth = 1.0 if (occurred and resolution_ts <= float(g)) else 0.0
        tot += (float(f) - truth) ** 2
    return round(tot / len(grid), 4)


def interval_cover(evt: dict, *, outcome: int, resolution_ts: float, q_lo: str, q_hi: str):
    qs = evt.get("first_passage_quantiles_ts") or {}
    lo, hi = qs.get(q_lo), qs.get(q_hi)
    occurred = outcome == 1 if str(evt.get("occurrence_resolves", "yes")).endswith("yes") \
        else outcome == 0
    if not occurred:
        return None                                          # censored rows excluded from coverage
    if not isinstance(lo, (int, float)):
        return 0 if occurred else None
    hi_v = hi if isinstance(hi, (int, float)) else float("inf")
    return 1 if float(lo) <= resolution_ts <= hi_v else 0


