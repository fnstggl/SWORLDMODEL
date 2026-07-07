"""Bayesian logistic with a Laplace posterior over the WEIGHTS — "how sure are we of each weight?".

This is the piece the weighting problem needs and the point estimators (PooledLogisticReadout,
LLMPriorReadout) were missing: every variable's weight comes with a POSTERIOR UNCERTAINTY, and the forecast
INTEGRATES OVER it. The honest answer to "we won't know how much mood affects the reaction" is not to guess
a number — it is to carry a distribution over the mood-weight and integrate it out, so an uncertain weight
WIDENS the prediction (correctly) instead of biasing it, and the data + prior decide the weight.

Mechanism (first-principles, cheap for small feature dims):
  - MAP fit of an L2-regularized logistic whose coefficients are shrunk toward a PRIOR MEAN `w0` (the LLM /
    literature elasticity prior), not toward zero — so a data-poor weight stays near its world-knowledge
    prior and a data-rich one moves to fit (this is the LLMPriorReadout idea, made Bayesian).
  - Laplace posterior: covariance ≈ H⁻¹ at the MAP, H = Σ pᵢ(1−pᵢ)xᵢxᵢᵀ + Λ (Λ from the L2/prior precision).
    `weight_sd[j] = sqrt(H⁻¹[j,j])` IS the honest "how sure are we of variable j's weight".
  - `predict_dist` samples weights from that posterior and returns the predictive mean AND spread — the
    reducible (epistemic) uncertainty that shrinks as the weight is pinned down by data.

`variance_contribution` implements the triage principle: a variable's share of outcome variance ≈
weight² × Var(feature) across the population, so you calibrate precisely only the few high-leverage weights.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def _sigmoid(z):
    if z < -35:
        return 1e-15
    if z > 35:
        return 1 - 1e-15
    return 1.0 / (1.0 + math.exp(-z))


def _matinv(A):
    """Inverse of a small symmetric positive-definite matrix via Gauss-Jordan (d small: features + intercept)."""
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            M[col][col] += 1e-9                       # nudge a near-singular pivot (ridge already added)
            piv = col
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [x / d for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [row[n:] for row in M]


@dataclass
class BayesianLogistic:
    """L2-to-prior logistic with a Laplace posterior over the coefficients. `w0` is the prior mean per
    feature (the elasticity prior; 0 => shrink toward zero, the ordinary ridge). `l2` is the prior PRECISION:
    a scalar (same for all weights) OR a per-weight list — a weight with a TIGHT prior CI (high precision)
    shrinks hard toward its prior mean; a weight with a wide/absent prior (low precision) is free to fit.
    Feature vectors are dense lists; the intercept is unpenalized."""
    l2: float = 1.0
    epochs: int = 300
    lr: float = 0.3
    w0: list = None                      # prior mean per feature (None => zeros)
    w: list = field(default=None)        # MAP weights (posterior mean)
    b: float = 0.0
    cov: list = field(default=None)      # posterior covariance (Laplace); diagonal => weight variances

    def _prec(self, d):
        return list(self.l2) if isinstance(self.l2, (list, tuple)) else [self.l2] * d

    def fit(self, X, y):
        n = len(X)
        d = len(X[0]) if n else 0
        w0 = list(self.w0) if self.w0 is not None else [0.0] * d
        prec = self._prec(d)
        w = list(w0)
        b = math.log((sum(y) + 1) / (n - sum(y) + 1)) if n else 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for xi, yi in zip(X, y):
                p = _sigmoid(b + sum(w[j] * xi[j] for j in range(d)))
                e = p - yi
                for j in range(d):
                    gw[j] += e * xi[j]
                gb += e
            for j in range(d):
                w[j] -= self.lr * (gw[j] / n + prec[j] * (w[j] - w0[j]) / n)   # shrink toward the prior mean
            b -= self.lr * gb / n
        self.w, self.b = w, b
        # Laplace posterior covariance at the MAP: (XᵀSX + diag(prec))⁻¹  (S = diag p(1-p))
        H = [[0.0] * d for _ in range(d)]
        for xi in X:
            p = _sigmoid(b + sum(w[j] * xi[j] for j in range(d)))
            s = p * (1 - p)
            for a in range(d):
                if xi[a] == 0.0:
                    continue
                for c in range(d):
                    H[a][c] += s * xi[a] * xi[c]
        for j in range(d):
            H[j][j] += prec[j]                          # per-weight prior precision (1/prior_var)
        self.cov = _matinv(H) if d else []
        return self

    def predict_proba(self, x) -> float:
        return _sigmoid(self.b + sum(self.w[j] * x[j] for j in range(len(x))))

    def weight_sd(self):
        """Posterior SD of each weight — the honest 'how sure are we of this variable's weight'."""
        return [math.sqrt(max(0.0, self.cov[j][j])) for j in range(len(self.w))] if self.cov else \
            [float("inf")] * len(self.w)

    def predict_dist(self, x, *, n_samples=200, seed=0) -> dict:
        """Integrate over WEIGHT uncertainty: sample coefficients from the Laplace posterior (diagonal), push
        each through the logistic, and return the predictive mean + spread. An uncertain weight -> a wider
        prediction (reducible/epistemic uncertainty). Reduces to the point prediction as the posterior tightens."""
        rng = random.Random(seed)
        sd = self.weight_sd()
        ps = []
        for _ in range(n_samples):
            wj = [self.w[j] + rng.gauss(0, sd[j]) for j in range(len(self.w))]
            ps.append(_sigmoid(self.b + sum(wj[j] * x[j] for j in range(len(x)))))
        m = sum(ps) / len(ps)
        var = sum((p - m) ** 2 for p in ps) / len(ps)
        return {"p": m, "sd": var ** 0.5, "map": self.predict_proba(x)}

    def weight_report(self, names=None):
        """Per-variable (weight, posterior_sd, |weight|/sd = signal-to-noise) — which weights are pinned down
        vs still-unknown. A weight with |w|/sd < ~1 is not distinguishable from its prior given the data."""
        sd = self.weight_sd()
        out = []
        for j in range(len(self.w)):
            nm = names[j] if names and j < len(names) else f"x{j}"
            snr = abs(self.w[j]) / sd[j] if sd[j] > 0 else float("inf")
            out.append({"name": nm, "weight": round(self.w[j], 4), "sd": round(sd[j], 4), "snr": round(snr, 2)})
        return out


def variance_contribution(X, weights):
    """Triage: a variable's approximate share of outcome-logit variance = weight² · Var(feature) over the
    population. Ranks which weights actually matter (high elasticity × high variance) — the ones worth
    calibrating precisely; the rest can stay at a rough prior. Returns [(index, share)] sorted desc."""
    n = len(X)
    d = len(X[0]) if n else 0
    contribs = []
    for j in range(d):
        col = [row[j] for row in X]
        mu = sum(col) / n
        var = sum((v - mu) ** 2 for v in col) / n
        contribs.append(weights[j] ** 2 * var)
    tot = sum(contribs) or 1.0
    return sorted([(j, c / tot) for j, c in enumerate(contribs)], key=lambda t: -t[1])
