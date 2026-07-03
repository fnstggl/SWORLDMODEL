"""Proper scoring rules + calibration + decision metrics (audit E.2, C.10).

This module is implemented first, on purpose. The whole thesis is that the evaluator must be
able to embarrass the model. These are the functions that do it. No sklearn dependency — the
math is short and worth reading.

All functions take plain lists/tuples of floats. Probabilities are for a binary outcome unless
noted. Everything here is deterministic and unit-testable.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

_EPS = 1e-12


def _clip(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, p))


def log_loss(y_true: Sequence[int], p_pred: Sequence[float]) -> float:
    """Mean negative log-likelihood (a.k.a. cross-entropy). Lower is better.
    Punishes confident wrong predictions hard — the point of a proper scoring rule."""
    n = len(y_true)
    if n == 0 or n != len(p_pred):
        raise ValueError("y_true and p_pred must be same non-zero length")
    total = 0.0
    for y, p in zip(y_true, p_pred):
        p = _clip(p)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / n


def brier_score(y_true: Sequence[int], p_pred: Sequence[float]) -> float:
    """Mean squared error of the probability. Lower is better. Proper scoring rule."""
    n = len(y_true)
    if n == 0:
        raise ValueError("empty input")
    return sum((p - y) ** 2 for y, p in zip(y_true, p_pred)) / n


def expected_calibration_error(
    y_true: Sequence[int], p_pred: Sequence[float], n_bins: int = 10
) -> float:
    """ECE: average gap between predicted confidence and observed frequency, across bins.
    A model can have good log-loss and still be miscalibrated; ECE catches that.
    0.0 = perfectly calibrated. Report the reliability diagram alongside this in practice."""
    if not y_true:
        raise ValueError("empty input")
    bins: list[list[tuple[int, float]]] = [[] for _ in range(n_bins)]
    for y, p in zip(y_true, p_pred):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((y, p))
    n = len(y_true)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(y for y, _ in b) / len(b)
        conf = sum(p for _, p in b) / len(b)
        ece += (len(b) / n) * abs(acc - conf)
    return ece


def base_rate(y_true: Sequence[int]) -> float:
    """The baseline every model must beat: predict the mean outcome for everyone."""
    return sum(y_true) / len(y_true) if y_true else 0.0


def uplift_at_k(
    y_true: Sequence[int], score: Sequence[float], k: float = 0.1
) -> float:
    """Decision metric the buyer actually cares about (audit E.2).

    If you act on the top-k fraction the model ranks highest, how much better is the outcome
    rate there than the overall base rate? Returns (rate_in_top_k - base_rate). Positive means
    the model's ranking concentrates outcomes — i.e. acting on it beats acting at random.
    """
    n = len(y_true)
    if n == 0 or not 0 < k <= 1:
        raise ValueError("need non-empty input and 0 < k <= 1")
    order = sorted(range(n), key=lambda i: score[i], reverse=True)
    top = order[: max(1, int(round(k * n)))]
    top_rate = sum(y_true[i] for i in top) / len(top)
    return top_rate - base_rate(y_true)


def crps_ensemble(y_obs: float, samples: Sequence[float]) -> float:
    """Continuous Ranked Probability Score for a distribution given as samples (audit E.2).
    Generalizes MAE to probabilistic forecasts; lower is better. Uses the
    CRPS = E|X - y| - 0.5 E|X - X'| estimator over the empirical sample distribution.
    Use this when the outcome is a rate/count/continuous quantity, not a single binary."""
    m = len(samples)
    if m == 0:
        raise ValueError("need at least one sample")
    term1 = sum(abs(s - y_obs) for s in samples) / m
    term2 = sum(abs(a - b) for a in samples for b in samples) / (m * m)
    return term1 - 0.5 * term2
