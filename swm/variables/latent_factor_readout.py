"""Latent-factor readout — kill double-counting by modelling the correlation structure (estimation frontier).

EXP-041's pooled logistic *shrinks* correlated variables toward zero — it damps double-counting but can't
distinguish "party and ideology are the same latent axis, count it once" from "both are independently
weak." The first-principles fix is to model the correlation structure explicitly: decompose the correlated
one-hot variables into a few ORTHOGONAL latent factors (the value axes the whole thesis is about —
left/right, traditional/progressive, religiosity), and estimate the outcome's dependence on the FACTORS.

Two correlated variables that measure one latent axis then contribute ONE effect (the factor's), not two
— double-counting is impossible by construction, and the estimator spends its few degrees of freedom on
the real axes instead of ~40 collinear dummies (data-efficient, the win on thin questions).

Pure Python: PCA via power iteration + deflation on the centered one-hot covariance (global, fit once);
then a small logistic per question on the K factor scores. This is the grounded "map the person's latent
value variables, then simulate" — the factors ARE the latent value profile.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.transition.readout import LogisticReadout
from swm.variables.pooled_readout import encode, onehot_vocab


def _cov(rows_x, mean):
    d = len(mean)
    cov = [[0.0] * d for _ in range(d)]
    for x in rows_x:
        c = [x[j] - mean[j] for j in range(d)]
        for i in range(d):
            ci = c[i]
            if ci:
                row = cov[i]
                for j in range(d):
                    row[j] += ci * c[j]
    n = max(1, len(rows_x))
    for i in range(d):
        for j in range(d):
            cov[i][j] /= n
    return cov


def _matvec(m, v):
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def _top_factors(cov, k, iters=60, seed=0):
    """Top-k eigenvectors of the covariance by power iteration + deflation (orthogonal latent factors)."""
    d = len(cov)
    rng = random.Random(seed)
    M = [row[:] for row in cov]
    factors = []
    for _ in range(min(k, d)):
        v = [rng.gauss(0, 1) for _ in range(d)]
        nrm = math.sqrt(sum(x * x for x in v)) or 1.0
        v = [x / nrm for x in v]
        lam = 0.0
        for _ in range(iters):
            w = _matvec(M, v)
            nrm = math.sqrt(sum(x * x for x in w))
            if nrm < 1e-12:
                break
            v = [x / nrm for x in w]
            lam = nrm
        factors.append(v)
        for i in range(d):                          # deflate: M -= lam * v v^T
            for j in range(d):
                M[i][j] -= lam * v[i] * v[j]
    return factors


@dataclass
class LatentFactorReadout:
    attrs: list = field(default_factory=list)
    k: int = 6                                        # number of latent value factors
    tau: float = 20.0                                 # partial pooling toward the marginal (as EXP-041)
    l2: float = 1.0
    epochs: int = 200
    max_cov_rows: int = 8000
    vocab: dict = field(default_factory=dict)
    mean: list = field(default_factory=list)
    factors: list = field(default_factory=list)
    _models: dict = field(default_factory=dict)       # qid -> (LogisticReadout|None, marginal, n)

    def _score(self, demo):
        x = encode(demo, self.attrs, self.vocab)
        c = [x[j] - self.mean[j] for j in range(len(x))]
        return [sum(c[j] * f[j] for j in range(len(f))) for f in self.factors]

    def fit(self, rows, min_q=12):
        self.vocab = onehot_vocab(rows, self.attrs)
        X = [encode(r["demo"], self.attrs, self.vocab) for r in rows]
        self.mean = [sum(row[j] for row in X) / len(X) for j in range(len(self.vocab))]
        cov_rows = X if len(X) <= self.max_cov_rows else X[:: max(1, len(X) // self.max_cov_rows)][:self.max_cov_rows]
        self.factors = _top_factors(_cov(cov_rows, self.mean), self.k)
        by_q = {}
        for r in rows:
            by_q.setdefault(r["qid"], []).append(r)
        self._models = {}
        for q, rs in by_q.items():
            y = [int(r["answer_idx"]) for r in rs]
            marg = (sum(y) + 1.0) / (len(y) + 2.0)
            model = None
            if len(rs) >= min_q and len(set(y)) == 2:
                Z = [self._score(r["demo"]) for r in rs]
                model = LogisticReadout(l2=self.l2, epochs=self.epochs).fit(Z, y)
            self._models[q] = (model, marg, len(rs))
        return self

    def predict(self, qid, demo):
        entry = self._models.get(qid)
        if entry is None:
            return 0.5
        model, marg, n = entry
        if model is None:
            return marg
        p = model.predict_proba(self._score(demo))
        w = n / (n + self.tau)
        return w * p + (1 - w) * marg
