"""Discriminative readout p(reply | persona, message) — THE WORKHORSE (audit C.8).

v1 is a pure-python L2 logistic regression trained by gradient descent, plus a bagged ensemble
for epistemic intervals. Deliberately boring: at v1 data sizes (10^2–10^4 sends, 14 features)
this is at the frontier of what's honest, it has no dependencies, and it is the baseline that
anything fancier (GBM, LLM features) must beat on the harness before earning its place.

The persona enters two ways:
- as a feature: the pooled-responsiveness logit (the hierarchical person effect), and
- inside the encoder's interaction features (style match).
Ablation rungs in exp001 toggle these.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def _sigmoid(z: float) -> float:
    if z < -35:
        return 1e-15
    if z > 35:
        return 1.0 - 1e-15
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class LogisticReadout:
    """L2-regularized logistic regression. fit() standardizes features internally."""
    l2: float = 0.1
    lr: float = 0.1
    epochs: int = 300
    seed: int = 0
    w: list[float] = field(default_factory=list)
    b: float = 0.0
    _mu: list[float] = field(default_factory=list)
    _sd: list[float] = field(default_factory=list)

    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticReadout":
        n, d = len(X), len(X[0])
        self._mu = [sum(row[j] for row in X) / n for j in range(d)]
        self._sd = [
            max(1e-9, math.sqrt(sum((row[j] - self._mu[j]) ** 2 for row in X) / n))
            for j in range(d)
        ]
        Xs = [self._standardize(row) for row in X]
        rng = random.Random(self.seed)
        self.w = [0.0] * d
        self.b = math.log(max(1e-9, sum(y) / n) / max(1e-9, 1 - sum(y) / n))
        idx = list(range(n))
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
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j] / n)
            self.b -= self.lr * gb / n
        return self

    def predict_proba(self, x: list[float]) -> float:
        xs = self._standardize(x)
        return _sigmoid(self.b + sum(wj * xj for wj, xj in zip(self.w, xs)))

    def _standardize(self, row: list[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(row, self._mu, self._sd)]

    def drivers(self, x: list[float], names: list[str], top: int = 5) -> list[tuple[str, float]]:
        """Per-feature contribution to the logit (standardized value x weight). Signed."""
        xs = self._standardize(x)
        contrib = [(names[j], self.w[j] * xs[j]) for j in range(len(names))]
        return sorted(contrib, key=lambda t: abs(t[1]), reverse=True)[:top]


@dataclass
class EnsembleReadout:
    """Bagged logistic ensemble: mean = prediction, spread = epistemic interval (audit C.9)."""
    n_members: int = 20
    l2: float = 0.1
    seed: int = 0
    members: list[LogisticReadout] = field(default_factory=list)

    def fit(self, X: list[list[float]], y: list[int]) -> "EnsembleReadout":
        rng = random.Random(self.seed)
        n = len(X)
        self.members = []
        for m in range(self.n_members):
            idx = [rng.randrange(n) for _ in range(n)]  # bootstrap resample
            Xb, yb = [X[i] for i in idx], [y[i] for i in idx]
            if len(set(yb)) < 2:  # degenerate resample; fall back to full data
                Xb, yb = X, y
            self.members.append(LogisticReadout(l2=self.l2, seed=self.seed + m).fit(Xb, yb))
        return self

    def predict(self, x: list[float]) -> tuple[float, tuple[float, float]]:
        """(mean probability, 80% ensemble interval). Interval reflects EPISTEMIC spread only;
        the aleatoric floor of a single binary outcome is communicated by the probability itself."""
        ps = sorted(m.predict_proba(x) for m in self.members)
        mean = sum(ps) / len(ps)
        lo = ps[max(0, int(0.10 * len(ps)) - 0)]
        hi = ps[min(len(ps) - 1, int(0.90 * len(ps)))]
        return mean, (lo, hi)

    def drivers(self, x: list[float], names: list[str], top: int = 5) -> list[tuple[str, float]]:
        agg: dict[str, float] = {}
        for m in self.members:
            for name, c in m.drivers(x, names, top=len(names)):
                agg[name] = agg.get(name, 0.0) + c / len(self.members)
        return sorted(agg.items(), key=lambda t: abs(t[1]), reverse=True)[:top]


IMPLEMENTED = True
