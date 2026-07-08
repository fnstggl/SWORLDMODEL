"""Correlation-aware, partially-pooled readout — the estimator EXP-040 showed the north star needs.

EXP-040's lesson: mapping more real variables only helps if you can ESTIMATE their joint effect. Naive
Bayes assumes the variables are independent, so it DOUBLE-COUNTS correlated ones (party ≈ ideology) and
overfits thin per-question data — it got *worse* as variables were added until hand-tuned. Two fixes,
both here:

1. CORRELATION-AWARE: a logistic readout over the full one-hot feature vector shares credit among
   correlated features instead of multiplying them (the NB failure). This is the workhorse `LogisticReadout`
   with L2 — regularization distributes weight across collinear dummies rather than letting each fire.

2. PARTIAL POOLING: each question's per-person model is blended with that question's marginal by an
   n-adaptive weight  w_q = n_q / (n_q + tau)  — a data-rich question trusts its fitted model; a
   data-poor one shrinks toward its base rate (borrowing strength from the population prior instead of
   overfitting 12 respondents). `tau` is one global hyperparameter, tuned by empirical Bayes on a
   train-internal hold-out — so the estimator learns how much to pool without hand-tuning per question.

Together they turn "map more variables" from a regression into a monotone gain (EXP-041), and this is the
reusable readout the grounded forward-simulation (EXP-042) builds its per-person responsiveness on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.transition.readout import LogisticReadout


def onehot_vocab(rows, attrs, feat_key="demo"):
    """Stable index for every (attr, level) seen in `rows` — the one-hot feature space."""
    vocab = {}
    for r in rows:
        d = r[feat_key]
        for a in attrs:
            key = (a, d.get(a, "unknown"))
            if key not in vocab:
                vocab[key] = len(vocab)
    return vocab


def encode(demo, attrs, vocab):
    x = [0.0] * len(vocab)
    for a in attrs:
        j = vocab.get((a, demo.get(a, "unknown")))
        if j is not None:
            x[j] = 1.0
    return x


@dataclass
class PooledLogisticReadout:
    """Per-question binary logistic (correlation-aware) + n-adaptive partial pooling toward the marginal."""
    attrs: list = field(default_factory=list)
    tau: float = 20.0                      # pooling strength: w_q = n_q/(n_q+tau); tuned by empirical Bayes
    l2: float = 1.0
    epochs: int = 200
    _models: dict = field(default_factory=dict)     # qid -> (LogisticReadout|None, vocab, marginal, n)

    def fit(self, rows, min_q=12):
        by_q = {}
        for r in rows:
            by_q.setdefault(r["qid"], []).append(r)
        self._models = {}
        for q, rs in by_q.items():
            y = [int(r["answer_idx"]) for r in rs]
            marg = (sum(y) + 1.0) / (len(y) + 2.0)
            model = None
            if len(rs) >= min_q and self.attrs and len(set(y)) == 2:
                vocab = onehot_vocab(rs, self.attrs)
                X = [encode(r["demo"], self.attrs, vocab) for r in rs]
                model = LogisticReadout(l2=self.l2, epochs=self.epochs).fit(X, y)
            else:
                vocab = {}
            self._models[q] = (model, vocab, marg, len(rs))
        return self

    def predict(self, qid, demo):
        """P(answer_idx == 1). Blends the per-question logistic with the marginal by the pooling weight."""
        entry = self._models.get(qid)
        if entry is None:
            return 0.5
        model, vocab, marg, n = entry
        if model is None:
            return marg
        p = model.predict_proba(encode(demo, self.attrs, vocab))
        w = n / (n + self.tau)                          # data-rich -> trust the model; data-poor -> marginal
        return w * p + (1 - w) * marg

    def proba(self, qid, demo):
        """Two-class distribution [P(0), P(1)] for parity with the NB readout."""
        p1 = self.predict(qid, demo)
        return [1 - p1, p1]
