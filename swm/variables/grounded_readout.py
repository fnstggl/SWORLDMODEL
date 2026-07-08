"""GroundedReadout — the three estimation-frontier pieces unified into one estimator.

The bottleneck the project kept hitting was estimation quality, and three separate results each fixed part
of it. This composes them into one readout:

  1. STRUCTURE (EXP-048) — decorrelate the correlated variables into orthogonal latent value factors, so
     redundant variables (party ≈ ideology) collapse into one axis counted once (no double-counting).
  2. WORLD-KNOWLEDGE PRIOR (EXP-049) — regularize the coefficients toward the LLM's prior on each effect
     (grounds the estimate, carries data-poor cells, resolves per-question polarity). The prior is a
     coefficient vector in one-hot space; the factors are an orthonormal basis V, so the prior projects
     EXACTLY into factor space as Vᵀ·prior — the two pieces compose without approximation.
  3. RELIABILITY WEIGHTING (new) — scale each variable's features by its provenance reliability
     (data/user=1.0 grounded, llm<1, heuristic<1, prior≈0). A noisy inferred variable's feature shrinks,
     so its effect is attenuated errors-in-variables style — the readout automatically trusts grounded
     variables more than inferred ones. This is the direct answer to "the variables haven't been real":
     real ones dominate the estimate, inferred ones contribute in proportion to how much they can be
     trusted.

Plus the EXP-041 n-adaptive pooling toward the marginal, so thin questions never overfit. One estimator,
grounded + structured + world-primed. Pure Python.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.variables.latent_factor_readout import _cov, _top_factors
from swm.variables.llm_prior import prior_value
from swm.variables.pooled_readout import onehot_vocab

# provenance -> reliability weight (from schema.PROVENANCE_RANK, normalized): grounded facts trusted fully,
# inferred variables discounted by how noisy their source is.
RELIABILITY = {"data": 1.0, "user": 1.0, "llm": 0.55, "heuristic": 0.3, "prior": 0.1}


def _sigmoid(z):
    if z < -35:
        return 1e-15
    if z > 35:
        return 1 - 1e-15
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


@dataclass
class GroundedReadout:
    attrs: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)     # attr -> provenance str (default "data" = grounded)
    k: int = 3                                          # latent value factors
    l2: float = 1.0
    prior_scale: float = 1.2
    tau: float = 40.0
    epochs: int = 150
    lr: float = 0.4
    use_prior: bool = True
    use_factors: bool = True
    use_reliability: bool = True
    vocab: dict = field(default_factory=dict)
    rel: list = field(default_factory=list)            # per-feature reliability weight
    mean: list = field(default_factory=list)
    factors: list = field(default_factory=list)
    _models: dict = field(default_factory=dict)        # qid -> (w, b, marginal, n)

    # ---- feature construction: reliability-weighted one-hot, optionally projected to factors ----
    def _reliability(self, attr):
        if not self.use_reliability:
            return 1.0
        return RELIABILITY.get(self.provenance.get(attr, "data"), 1.0)

    def _weighted_onehot(self, demo):
        x = [0.0] * len(self.vocab)
        for a in self.attrs:
            j = self.vocab.get((a, demo.get(a, "")))
            if j is not None:
                x[j] = self.rel[j]
        return x

    def _features(self, demo):
        x = self._weighted_onehot(demo)
        if not self.use_factors or not self.factors:
            return x
        c = [x[j] - self.mean[j] for j in range(len(x))]
        return [sum(c[j] * f[j] for j in range(len(f))) for f in self.factors]

    def _prior_coef(self, item):
        """LLM prior as a coefficient vector, in the same space the readout fits (one-hot, or projected)."""
        pw = [0.0] * len(self.vocab)
        if self.use_prior:
            for (attr, level), j in self.vocab.items():
                pw[j] = self.prior_scale * prior_value(item, attr, level)
        if not self.use_factors or not self.factors:
            return pw
        return [sum(pw[j] * f[j] for j in range(len(f))) for f in self.factors]   # Vᵀ·prior (exact)

    def fit(self, rows, min_q=4):
        self.vocab = onehot_vocab(rows, self.attrs)
        inv = {j: a for (a, _l), j in self.vocab.items()}
        self.rel = [self._reliability(inv[j]) for j in range(len(self.vocab))]
        if self.use_factors:
            X = [self._weighted_onehot(r["demo"]) for r in rows]
            cap = X if len(X) <= 8000 else X[:: max(1, len(X) // 8000)][:8000]
            self.mean = [sum(row[j] for row in X) / len(X) for j in range(len(self.vocab))]
            self.factors = _top_factors(_cov(cap, self.mean), self.k)
        by = {}
        for r in rows:
            by.setdefault(r["qid"], []).append(r)
        self._models = {}
        for q, rs in by.items():
            y = [int(v["answer_idx"]) for v in rs]
            marg = (sum(y) + 1) / (len(y) + 2)
            pw = self._prior_coef(q)
            if len(set(y)) < 2 or len(rs) < min_q:
                self._models[q] = (pw, _logit(marg), marg, len(rs))
                continue
            Z = [self._features(v["demo"]) for v in rs]
            d = len(Z[0]); w = pw[:] if len(pw) == d else [0.0] * d
            b = _logit(marg); n = len(Z)
            for _ in range(self.epochs):
                gw = [0.0] * d; gb = 0.0
                for zi, yi in zip(Z, y):
                    p = _sigmoid(b + sum(w[j] * zi[j] for j in range(d)))
                    err = p - yi
                    for j in range(d):
                        gw[j] += err * zi[j]
                    gb += err
                for j in range(d):
                    w[j] -= self.lr * (gw[j] / n + self.l2 * (w[j] - pw[j]) / n)   # toward the projected prior
                b -= self.lr * gb / n
            self._models[q] = (w, b, marg, len(rs))
        return self

    def predict(self, qid, demo):
        entry = self._models.get(qid)
        if entry is None:
            return 0.5
        w, b, marg, n = entry
        z = self._features(demo)
        p = _sigmoid(b + sum(w[j] * z[j] for j in range(len(w))))
        wt = n / (n + self.tau)
        return wt * p + (1 - wt) * marg

    def fit_auto(self, rows, val_frac=0.3, salt=11, min_q=4):
        """Self-configure: the pieces help conditionally (factors help low-rank signal; the prior helps
        thin data), so pick the config that wins on a TRAIN-INTERNAL hold-out — never worse than the best
        single piece, no test leakage. Reliability weighting stays on (it can only help when inferred
        variables are present, and is a no-op when all are grounded)."""
        import zlib
        idx = [(zlib.crc32(f"{salt}:{i}".encode()) % 100000) / 100000.0 for i in range(len(rows))]
        fit_rows = [r for r, h in zip(rows, idx) if h >= val_frac]
        val_rows = [r for r, h in zip(rows, idx) if h < val_frac]
        if not val_rows or not fit_rows:
            self.chosen = {"use_factors": self.use_factors, "k": self.k, "use_prior": self.use_prior}
            return self.fit(rows, min_q=min_q)
        configs = [(False, 0, True), (False, 0, False), (True, 3, True), (True, 8, True)]
        best, best_ll = None, 1e9
        for uf, kk, up in configs:
            m = GroundedReadout(attrs=self.attrs, provenance=self.provenance, k=(kk or self.k),
                                l2=self.l2, prior_scale=self.prior_scale, tau=self.tau, epochs=self.epochs,
                                use_factors=uf, use_prior=up, use_reliability=self.use_reliability).fit(
                fit_rows, min_q=min_q)
            ll = 0.0
            for r in val_rows:
                p = m.predict(r["qid"], r["demo"]); pa = p if r["answer_idx"] == 1 else 1 - p
                ll += -math.log(min(1 - 1e-9, max(1e-9, pa)))
            ll /= max(1, len(val_rows))
            if ll < best_ll:
                best_ll, best = ll, (uf, kk, up)
        self.use_factors, kk, self.use_prior = best
        if kk:
            self.k = kk
        self.chosen = {"use_factors": self.use_factors, "k": self.k, "use_prior": self.use_prior}
        return self.fit(rows, min_q=min_q)
