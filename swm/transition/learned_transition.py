"""Learned transition: p(outcome | state, action, context) fit from data, not hand-coded.

The EXP-009 failure audit's central charge is that the "world model" was a *linear* logistic over
shallow factors — it could not represent interactions (deep-author × hot-topic, credible-source ×
technical-depth) that a raw LLM captures implicitly. This module answers it with a real learned,
NON-LINEAR outcome model, dependency-free (no numpy/sklearn in this environment):

- `DecisionTreeRegressor` — CART regression tree (squared-error splits).
- `GradientBoostedClassifier` — logistic-loss gradient boosting over those trees. This is the
  learned `p(outcome | features)`; it captures feature interactions a logistic cannot.
- `LearnedTransition` — a drop-in outcome head (same `.fit(X, scores)` / `.predict(x)->band dict`
  contract as `OutcomeHead`) so it slots into `AggregateWorld`/benchmarks and can be A/B'd against
  the hand-coded logistic head.

On `p(next_state | state, action, outcome)`: the world model's next-state update is the *conjugate /
EMA sufficient-statistic update* in `PopulationState`/`EntityHistory` — that is the principled
Bayesian transition, not a hand-waved rule, and it is what should evolve state. We do NOT fit a GBDT
to hallucinate next-state; we learn the hard part (the outcome distribution) and update state by the
correct estimator. `compare_learned_vs_handcoded` quantifies whether the learned outcome model beats
the logistic one on held-out data.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.transition.transition_head import BAND_EDGES

_SIG = lambda z: 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))  # noqa: E731


# --------------------------------------------------------------------------- CART regression tree
class _Node:
    __slots__ = ("feat", "thr", "left", "right", "value")

    def __init__(self):
        self.feat = -1
        self.thr = 0.0
        self.left = None
        self.right = None
        self.value = 0.0


@dataclass
class DecisionTreeRegressor:
    """Squared-error CART. Fits residuals in gradient boosting; small depths, robust at n~10^3."""
    max_depth: int = 3
    min_leaf: int = 10
    max_features: int | None = None       # feature subsampling per split (adds RF-style variance red.)
    seed: int = 0
    _root: _Node = None

    def fit(self, X, g, feat_idx=None):
        self._rng = random.Random(self.seed)
        n_feat = len(X[0])
        self._all_feats = list(range(n_feat))
        self._root = self._build(X, g, list(range(len(X))), 0)
        return self

    def _build(self, X, g, idx, depth):
        node = _Node()
        node.value = sum(g[i] for i in idx) / len(idx)
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
            return node
        feats = self._all_feats
        if self.max_features and self.max_features < len(feats):
            feats = self._rng.sample(feats, self.max_features)
        best = self._best_split(X, g, idx, feats)
        if best is None:
            return node
        f, thr, left_idx, right_idx = best
        node.feat, node.thr = f, thr
        node.left = self._build(X, g, left_idx, depth + 1)
        node.right = self._build(X, g, right_idx, depth + 1)
        node.value = 0.0
        return node

    def _best_split(self, X, g, idx, feats):
        best_gain, best = 0.0, None
        parent_sum = sum(g[i] for i in idx)
        parent_sse = parent_sum * parent_sum / len(idx)
        for f in feats:
            order = sorted(idx, key=lambda i: X[i][f])
            vals = [X[i][f] for i in order]
            gs = [g[i] for i in order]
            n = len(order)
            left_sum = 0.0
            for k in range(1, n):
                left_sum += gs[k - 1]
                if k < self.min_leaf or n - k < self.min_leaf:
                    continue
                if vals[k] == vals[k - 1]:
                    continue
                right_sum = parent_sum - left_sum
                gain = (left_sum * left_sum / k + right_sum * right_sum / (n - k)) - parent_sse
                if gain > best_gain:
                    thr = 0.5 * (vals[k] + vals[k - 1])
                    best_gain = gain
                    best = (f, thr, order[:k], order[k:])
        return best

    def predict_one(self, x):
        node = self._root
        while node.left is not None:
            node = node.left if x[node.feat] <= node.thr else node.right
        return node.value


@dataclass
class GradientBoostedClassifier:
    """Logistic-loss gradient boosting over shallow CART trees. Captures feature interactions."""
    n_estimators: int = 120
    learning_rate: float = 0.1
    max_depth: int = 3
    min_leaf: int = 12
    subsample: float = 0.8
    max_features: int | None = None
    l2_leaf: float = 1.0
    seed: int = 0
    trees: list = field(default_factory=list)
    base: float = 0.0

    def fit(self, X, y):
        n = len(X)
        p0 = max(1e-6, min(1 - 1e-6, sum(y) / n))
        self.base = math.log(p0 / (1 - p0))
        F = [self.base] * n
        rng = random.Random(self.seed)
        self.trees = []
        for m in range(self.n_estimators):
            g = [y[i] - _SIG(F[i]) for i in range(n)]          # negative gradient of logloss
            if self.subsample < 1.0:
                sample = [i for i in range(n) if rng.random() < self.subsample]
                if len(sample) < 2 * self.min_leaf:
                    sample = list(range(n))
            else:
                sample = list(range(n))
            Xs = [X[i] for i in sample]
            gs = [g[i] for i in sample]
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_leaf=self.min_leaf,
                                         max_features=self.max_features, seed=self.seed + m).fit(Xs, gs)
            self.trees.append(tree)
            for i in range(n):
                F[i] += self.learning_rate * tree.predict_one(X[i])
        return self

    def decision(self, x):
        return self.base + self.learning_rate * sum(t.predict_one(x) for t in self.trees)

    def predict_proba(self, x):
        return _SIG(self.decision(x))


@dataclass
class LearnedTransition:
    """Learned outcome head with the OutcomeHead contract: one GBDT per score-band threshold,
    monotone-corrected, returning a band distribution. Drop-in for the hand-coded logistic head."""
    thresholds: tuple[int, ...] = tuple(BAND_EDGES)
    n_estimators: int = 120
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 0.8
    max_features: int | None = None
    seed: int = 0
    models: dict = field(default_factory=dict)

    def fit(self, X, scores):
        for thr in self.thresholds:
            y = [1 if s >= thr else 0 for s in scores]
            if len(set(y)) == 2:
                self.models[thr] = GradientBoostedClassifier(
                    n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                    max_depth=self.max_depth, subsample=self.subsample,
                    max_features=self.max_features, seed=self.seed + thr).fit(X, y)
            else:
                self.models[thr] = None
        return self

    def predict(self, x):
        t = {thr: (m.predict_proba(x) if m else 0.0) for thr, m in
             ((thr, self.models.get(thr)) for thr in self.thresholds)}
        vals = [t[thr] for thr in self.thresholds]
        for i in range(1, len(vals)):
            vals[i] = min(vals[i], vals[i - 1])
        bands = [1 - vals[0]]
        for i in range(len(vals) - 1):
            bands.append(max(1e-6, vals[i] - vals[i + 1]))
        bands.append(vals[-1])
        s = sum(bands)
        return {"thresholds": {thr: t[thr] for thr in self.thresholds},
                "band_probs": [b / s for b in bands]}


def compare_learned_vs_handcoded(X_tr, s_tr, X_te, s_te, *, thr: int = 40) -> dict:
    """Fit the learned (GBDT) and hand-coded (logistic) heads on the SAME features; return held-out
    log loss / Brier / ECE for both at one threshold, so the learned transition earns (or doesn't)
    its place."""
    from swm.eval.metrics import brier_score, expected_calibration_error, log_loss
    from swm.transition.readout import LogisticReadout
    y_tr = [1 if s >= thr else 0 for s in s_tr]
    y_te = [1 if s >= thr else 0 for s in s_te]
    out = {}
    if len(set(y_tr)) == 2:
        lin = LogisticReadout(epochs=300).fit(X_tr, y_tr)
        gb = GradientBoostedClassifier(seed=0).fit(X_tr, y_tr)
        for name, m in (("handcoded_logistic", lin), ("learned_gbdt", gb)):
            p = [min(1 - 1e-6, max(1e-6, m.predict_proba(x))) for x in X_te]
            out[name] = {"log_loss": round(log_loss(y_te, p), 4),
                         "brier": round(brier_score(y_te, p), 4),
                         "ece": round(expected_calibration_error(y_te, p), 4)}
    return out
