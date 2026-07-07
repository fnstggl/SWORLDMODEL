"""Population-outcome metrics — the RIGHT scoreboard for large-scale demographic prediction.

Why not just log-loss (the honest KPI critique):
  Log-loss is a proper score for a probabilistic BINARY label. But a large-scale demographic outcome is a
  CONTINUOUS population share (support %, adoption %, vote share), sometimes a distribution or a winner —
  not a coin flip. Three problems with leaning on log-loss here:
    1. it scores a probability against a 0/1 label, but our target is a proportion in [0,1] — the natural
       error is on the SHARE, not a classification loss;
    2. it conflates CALIBRATION and SHARPNESS into one number, so you can't see which is failing;
    3. it does not isolate the one thing this whole level rests on — does the COUPLING beat the
       marginal-average composite? "otherwise it's a fancy poll average."

So the scoreboard here leads with three things log-loss can't give:
  - SHARE ERROR (rmse/mae on the predicted proportion) — the accuracy that matters for a share;
  - COUPLING SKILL — the decisive number: how much the coupled simulation reduces error RELATIVE to the
    marginal-average baseline, across a benchmark. > 0 means interaction earned its place;
  - INTERVAL COVERAGE — a real calibration check for a continuous outcome (do the model's X% predictive
    intervals actually contain the truth X% of the time?), which log-loss cannot express.
Binary reductions (brier / a winner-accuracy) are kept as SECONDARY, for the "who wins" framing.
"""
from __future__ import annotations

import math
from typing import Sequence


def share_rmse(truth: Sequence[float], pred: Sequence[float]) -> float:
    n = len(truth)
    return math.sqrt(sum((t - p) ** 2 for t, p in zip(truth, pred)) / n) if n else float("nan")


def share_mae(truth: Sequence[float], pred: Sequence[float]) -> float:
    n = len(truth)
    return sum(abs(t - p) for t, p in zip(truth, pred)) / n if n else float("nan")


def coupling_skill(truth: Sequence[float], marginal: Sequence[float], coupled: Sequence[float]) -> dict:
    """The decisive metric. Skill = 1 - RMSE(coupled)/RMSE(marginal): the fractional error reduction the
    coupling buys over the marginal-average composite. > 0 => interaction helps; ~0 => a fancy poll
    average; < 0 => coupling hurts. Also returns both RMSEs and the win rate (per-item)."""
    rm = share_rmse(truth, marginal)
    rc = share_rmse(truth, coupled)
    wins = sum(1 for t, m, c in zip(truth, marginal, coupled) if abs(c - t) < abs(m - t))
    n = len(truth)
    return {"marginal_rmse": round(rm, 4), "coupled_rmse": round(rc, 4),
            "skill": round(1 - rc / rm, 4) if rm > 1e-9 else 0.0,
            "coupled_wins_frac": round(wins / n, 4) if n else float("nan"), "n": n}


def interval_coverage(truth: Sequence[float], lo: Sequence[float], hi: Sequence[float],
                      nominal: float = 0.8) -> dict:
    """Calibration for a continuous outcome: fraction of truths inside [lo,hi]. Well-calibrated =>
    empirical coverage ~ nominal. Also the mean interval width (sharpness) — tight AND covered is the goal."""
    n = len(truth)
    inside = sum(1 for t, a, b in zip(truth, lo, hi) if a <= t <= b)
    width = sum(b - a for a, b in zip(lo, hi)) / n if n else float("nan")
    cov = inside / n if n else float("nan")
    return {"nominal": nominal, "empirical_coverage": round(cov, 4),
            "coverage_gap": round(cov - nominal, 4), "mean_width": round(width, 4), "n": n}


def winner_accuracy(truth: Sequence[float], pred: Sequence[float], threshold: float = 0.5) -> float:
    """Secondary 'who wins' framing: did we get the majority side right (share vs threshold)?"""
    n = len(truth)
    return sum(1 for t, p in zip(truth, pred) if (t > threshold) == (p > threshold)) / n if n else float("nan")


def brier_share(truth: Sequence[float], pred: Sequence[float]) -> float:
    """Brier on the share treated as a probability (secondary; = MSE on the proportion)."""
    n = len(truth)
    return sum((t - p) ** 2 for t, p in zip(truth, pred)) / n if n else float("nan")


def population_scorecard(truth, marginal, coupled, *, lo=None, hi=None, nominal=0.8, threshold=0.5) -> dict:
    """The full scoreboard for a benchmark of population outcomes. `coupled` is the model under test;
    `marginal` is the no-interaction composite it must beat. Intervals (lo/hi) are the COUPLED model's."""
    card = {
        "share_rmse": {"marginal": round(share_rmse(truth, marginal), 4),
                       "coupled": round(share_rmse(truth, coupled), 4)},
        "share_mae": {"marginal": round(share_mae(truth, marginal), 4),
                      "coupled": round(share_mae(truth, coupled), 4)},
        "coupling_skill": coupling_skill(truth, marginal, coupled),
        "winner_accuracy": {"marginal": round(winner_accuracy(truth, marginal, threshold), 4),
                            "coupled": round(winner_accuracy(truth, coupled, threshold), 4)},
    }
    if lo is not None and hi is not None:
        card["interval_coverage"] = interval_coverage(truth, lo, hi, nominal)
    card["headline"] = ("coupling helps" if card["coupling_skill"]["skill"] > 0.02
                        else "marginal-dominated (coupling ~ poll average)"
                        if card["coupling_skill"]["skill"] > -0.02 else "coupling hurts")
    return card
