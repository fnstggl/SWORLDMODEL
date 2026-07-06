"""LLM-informed prior estimator — world knowledge as the prior, data as the update (estimation frontier).

The synthesis of the two threads. EXP-047 showed the LLM can READ the real world (news → stance); EXP-048
showed structured estimation beats shrinkage. This uses the LLM's world knowledge as a *prior on the
effects themselves*: it knows, without any dataset, that conservatives favor the death penalty and the
secular favor marijuana legalization. That prior

  - GROUNDS the estimate in real relationships (fixes "the variables aren't real"),
  - CARRIES data-poor cells (the prior predicts before the first datapoint — fixes data-poverty),
  - RESOLVES per-question polarity (the LLM knows which way each attribute pushes THIS outcome — the exact
    thing that blocked cross-question pooling in EXP-041),

and the DATA updates it: a logistic whose coefficients are regularized toward the LLM prior (β→1 on the
prior-signed features) rather than toward zero. Data-poor → stay near the world-knowledge prior; data-rich
→ move to fit. Bayesian in spirit: informative prior + likelihood.

Production path mirrors `semantic_stance`: `prior_axis_from_llm(item_desc, judge_fn)` would ask an LLM for
each attribute's signed push once and cache it; here the axis map is specified from world knowledge (the
cached equivalent) so the experiment is reproducible and the priors are coarse/general — NOT fitted to any
dataset (the data, held out, is what calibrates magnitude).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Each attribute level's position on a secular-liberal(−) ↔ religious-conservative(+) axis — general
# world knowledge, not tuned to any dataset. Keys match the GSS demo schema (datasets_gss).
AXIS = {
    "party": {"democrat": -0.7, "independent": 0.0, "republican": 0.7},
    "ideology": {"liberal": -0.9, "moderate": 0.0, "conservative": 0.9},
    "relig": {"none": -0.6, "jewish": -0.4, "catholic": 0.1, "protestant": 0.35, "other": 0.0},
    "attendance": {"low": -0.4, "medium": 0.0, "high": 0.5},
    "age": {"18-29": -0.3, "30-49": -0.1, "50-64": 0.1, "65+": 0.3},
    "degree": {"graduate": -0.3, "bachelor": -0.2, "junior_college": 0.0, "highschool": 0.1,
               "lt_highschool": 0.2},
    "sex": {"female": -0.05, "male": 0.05},
    "race": {"black": -0.4, "other": -0.1, "white": 0.2},
    "marital": {"married": 0.15, "never_married": -0.15, "divorced": -0.05, "separated": -0.05,
                "widowed": 0.1},
    "region": {},          # weak/uninformative on the axis without more detail -> neutral
}

# Whether a GSS item's binary "1" pole is the conservative/traditional side (+1) or the liberal side (−1).
ITEM_POLE = {
    "cappun": +1, "natcrime": +1, "fepol": +1, "fefam": +1, "homosex": +1, "premarsx": +1,
    "gunlaw": -1, "grass": -1, "abany": -1, "letdie1": -1, "natheal": -1, "natenvir": -1,
    "natfare": -1, "nateduc": -1, "natrace": -1,
}
# Item-specific amplification where an attribute matters more than the pure lib-con axis (world knowledge).
ITEM_ATTR_SCALE = {
    ("natrace", "race"): 2.5, ("abany", "relig"): 1.6, ("abany", "attendance"): 1.6,
    ("homosex", "relig"): 1.6, ("homosex", "attendance"): 1.5, ("premarsx", "relig"): 1.6,
    ("premarsx", "attendance"): 1.6, ("fefam", "sex"): 2.0, ("fepol", "sex"): 2.0,
    ("natfare", "race"): 1.5, ("grass", "age"): 1.5,
}


def prior_value(item: str, attr: str, level: str) -> float:
    """The LLM prior: signed push of (attr=level) toward this item's YES(=1) pole, in ~[-1,1]."""
    pole = ITEM_POLE.get(item, 0)
    base = AXIS.get(attr, {}).get(level, 0.0)
    return pole * base * ITEM_ATTR_SCALE.get((item, attr), 1.0)


def prior_features(item: str, demo: dict, attrs: list) -> list:
    return [prior_value(item, a, demo.get(a, "")) for a in attrs]


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
class LLMPriorReadout:
    """Full one-hot logistic whose per-level coefficients are regularized TOWARD the LLM prior (not toward
    zero). This keeps one-hot expressiveness (no straitjacket at high data) AND grounds thin cells: with
    little data every level's coefficient stays near the world-knowledge prior for that (item, attr,
    level); with much data it fits freely. `prior_scale` maps the prior's [-1,1] push to a target logit
    coefficient; `l2` sets how hard the data must work to move a coefficient off the prior.

    prior_only=True fixes the coefficients at the prior (zero-shot world knowledge, calibrated only by the
    per-item marginal bias) — the N=0 anchor.
    """
    attrs: list = field(default_factory=list)
    l2: float = 1.0
    prior_scale: float = 1.2
    tau: float = 40.0                    # n-adaptive pooling toward the item marginal (anti-overconfidence)
    epochs: int = 150
    lr: float = 0.4
    prior_only: bool = False
    vocab: dict = field(default_factory=dict)
    _models: dict = field(default_factory=dict)     # item -> (w, b, marginal, n)

    def _prior_w(self, item):
        w = [0.0] * len(self.vocab)
        for (attr, level), j in self.vocab.items():
            w[j] = self.prior_scale * prior_value(item, attr, level)
        return w

    def fit(self, rows):
        from swm.variables.pooled_readout import onehot_vocab
        self.vocab = onehot_vocab(rows, self.attrs)
        by = {}
        for r in rows:
            by.setdefault(r["item"], []).append(r)
        self._models = {}
        for item, rs in by.items():
            y = [int(v["answer"]) for v in rs]
            marg = (sum(y) + 1) / (len(y) + 2)
            b0 = _logit(marg)
            pw = self._prior_w(item)
            if self.prior_only or len(set(y)) < 2 or len(rs) < 4:
                self._models[item] = (pw, b0, marg, len(rs))
                continue
            X = [self._encode(v["demo"]) for v in rs]
            w = pw[:]; b = b0; n = len(X); d = len(self.vocab)
            for _ in range(self.epochs):
                gw = [0.0] * d; gb = 0.0
                for xi, yi in zip(X, y):
                    p = _sigmoid(b + sum(w[j] for j in xi))       # xi = active feature indices
                    err = p - yi
                    for j in xi:
                        gw[j] += err
                    gb += err
                for j in range(d):
                    w[j] -= self.lr * (gw[j] / n + self.l2 * (w[j] - pw[j]) / n)   # shrink toward the prior
                b -= self.lr * gb / n
            self._models[item] = (w, b, marg, len(rs))
        return self

    def _encode(self, demo):
        idx = []
        for a in self.attrs:
            j = self.vocab.get((a, demo.get(a, "")))
            if j is not None:
                idx.append(j)
        return idx

    def predict(self, item: str, demo: dict) -> float:
        entry = self._models.get(item)
        if entry is None:                                # unseen item: zero-shot prior, neutral bias
            z = self.prior_scale * sum(prior_features(item, demo, self.attrs))
            return _sigmoid(z)
        w, b, marg, n = entry
        p = _sigmoid(b + sum(w[j] for j in self._encode(demo)))
        wt = n / (n + self.tau)                          # thin item -> shrink toward its marginal
        return wt * p + (1 - wt) * marg
