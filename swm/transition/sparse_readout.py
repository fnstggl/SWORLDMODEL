"""Sparse logistic readout — L1/elastic-net feature discipline learned end-to-end.

EXP-025 found that dumping all 23 deep-persona traits into a dense logistic OVERFITS, and only a
hand-picked subset (intellectual_humility, …) won. Hand-picking doesn't scale and risks snooping. This
readout removes the hand: it fits an elastic-net (L1 + L2) logistic by proximal gradient — the L1 term
drives irrelevant coefficients to exactly zero, so the model performs its OWN feature selection on the
training data only. Given all 23 traits, it should keep the few that carry signal and discard the rest,
recovering the win without a human choosing the features.

Proximal gradient (ISTA): each epoch takes a gradient step on the smooth part (mean logistic loss +
L2/2·‖w‖²) then applies the soft-threshold prox of the L1 term, which is what yields exact zeros.
Features are standardized internally so the single L1 strength is comparable across coefficients.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.transition.readout import LogisticReadout


def _sigmoid(z: float) -> float:
    if z < -35:
        return 1e-15
    if z > 35:
        return 1.0 - 1e-15
    return 1.0 / (1.0 + math.exp(-z))


def _soft_threshold(x: float, t: float) -> float:
    """Prox of t·|x|: shrink toward 0 by t, clamp at 0 — the operator that produces exact sparsity."""
    if x > t:
        return x - t
    if x < -t:
        return x + t
    return 0.0


@dataclass
class SparseLogisticReadout:
    """Elastic-net (L1 + L2) logistic regression via proximal gradient. Standardizes features."""
    l1: float = 0.02          # sparsity strength (drives coefficients to exactly zero)
    l2: float = 0.01          # ridge shrinkage (stabilizes correlated features)
    lr: float = 0.1
    epochs: int = 400
    seed: int = 0
    w: list[float] = field(default_factory=list)
    b: float = 0.0
    _mu: list[float] = field(default_factory=list)
    _sd: list[float] = field(default_factory=list)

    def fit(self, X: list[list[float]], y: list[int]) -> "SparseLogisticReadout":
        n, d = len(X), len(X[0])
        self._mu = [sum(row[j] for row in X) / n for j in range(d)]
        self._sd = [max(1e-9, math.sqrt(sum((row[j] - self._mu[j]) ** 2 for row in X) / n)) for j in range(d)]
        Xs = [self._standardize(row) for row in X]
        self.w = [0.0] * d
        p0 = min(1 - 1e-9, max(1e-9, sum(y) / n))
        self.b = math.log(p0 / (1 - p0))
        idx = list(range(n))
        rng = random.Random(self.seed)
        for _ in range(self.epochs):
            rng.shuffle(idx)
            gw = [0.0] * d
            gb = 0.0
            for i in idx:
                p = _sigmoid(self.b + sum(wj * xj for wj, xj in zip(self.w, Xs[i])))
                err = p - y[i]
                for j in range(d):
                    gw[j] += err * Xs[i][j]
                gb += err
            # gradient step on smooth part (loss + L2), then L1 prox (soft-threshold) -> exact zeros
            for j in range(d):
                wj = self.w[j] - self.lr * (gw[j] / n + self.l2 * self.w[j])
                self.w[j] = _soft_threshold(wj, self.lr * self.l1)
            self.b -= self.lr * gb / n
        return self

    def predict_proba(self, x: list[float]) -> float:
        xs = self._standardize(x)
        return _sigmoid(self.b + sum(wj * xj for wj, xj in zip(self.w, xs)))

    def _standardize(self, row: list[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(row, self._mu, self._sd)]

    def nonzero(self, names: list[str]) -> list[tuple[str, float]]:
        """The features the model KEPT (non-zero standardized weight), by magnitude — its learned choice."""
        return sorted([(names[j], round(self.w[j], 4)) for j in range(len(self.w)) if abs(self.w[j]) > 1e-8],
                      key=lambda t: -abs(t[1]))

    def sparsity(self) -> float:
        """Fraction of coefficients driven to exactly zero."""
        return sum(1 for wj in self.w if abs(wj) <= 1e-8) / max(1, len(self.w))


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy * n)


@dataclass
class ScreenedLogisticReadout:
    """Filter-based feature discipline: keep the top-k features by |correlation with y| on the TRAINING
    data, then fit a dense L2 logistic on just those. k is chosen by an inner train/val split (no test
    leakage). At small sample sizes this is far more STABLE than L1 (which scatters across correlated
    features), and it recovers the signal without a human hand-picking which features to use.

    Selection uses only training labels, so it is leakage-free as long as `fit` sees only training rows.
    """
    k: int | None = None                     # if None, tuned over k_grid by inner CV
    k_grid: tuple = (1, 2, 3, 5, 8)
    l2: float = 1.0
    epochs: int = 400
    seed: int = 0
    keep_: list[int] = field(default_factory=list)
    _model: LogisticReadout = None           # type: ignore

    def _rank(self, X, y):
        d = len(X[0])
        return sorted(range(d), key=lambda j: -abs(_corr([row[j] for row in X], [float(v) for v in y])))

    def fit(self, X: list[list[float]], y: list[int]) -> "ScreenedLogisticReadout":
        k = self.k if self.k is not None else self._tune_k(X, y)
        self.keep_ = self._rank(X, y)[:k]
        Xs = [[row[j] for j in self.keep_] for row in X]
        self._model = LogisticReadout(l2=self.l2, epochs=self.epochs, seed=self.seed).fit(Xs, y)
        return self

    def _tune_k(self, X, y) -> int:
        n = len(X)
        rng = random.Random(self.seed + 7)
        idx = list(range(n)); rng.shuffle(idx)
        c = max(1, int(0.7 * n)); tr, va = idx[:c], idx[c:]
        if len(va) < 5 or len(set(y[i] for i in tr)) < 2:
            return min(self.k_grid[0], len(X[0]))
        rank = self._rank([X[i] for i in tr], [y[i] for i in tr])
        best, best_ll = self.k_grid[0], 1e9
        for k in self.k_grid:
            keep = rank[:k]
            m = LogisticReadout(l2=self.l2, epochs=self.epochs, seed=self.seed).fit(
                [[X[i][j] for j in keep] for i in tr], [y[i] for i in tr])
            preds = [min(1 - 1e-9, max(1e-9, m.predict_proba([X[i][j] for j in keep]))) for i in va]
            ll = -sum(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p) for i, p in zip(va, preds)) / len(va)
            if ll < best_ll:
                best_ll, best = ll, k
        return best

    def predict_proba(self, x: list[float]) -> float:
        return self._model.predict_proba([x[j] for j in self.keep_])

    def selected(self, names: list[str]) -> list[str]:
        return [names[j] for j in self.keep_]
