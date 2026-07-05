"""Configurable individual-response world model — the object we IMPROVE and ablate (Part C).

The individual regime models `this entity + this action + context -> P(response)` as a response
function over the entity's latent state. This module makes every realism upgrade a toggle, so each
can be added and measured on held-out no-cheat data until diminishing returns:

- pooled rate            : hierarchical partial pooling of the entity's response rate (baseline).
- multilevel pooling     : entity <- segment <- global (helps cold/mid evidence).
- recency (EWMA)         : recent behavior weighed more (nonstationarity).
- entity-state features  : depth, sufficiency, recency-rate, trend fed to the readout.
- interactions           : entity_rate x message features (the design note's core claim).
- learned readout        : GBDT over the full vector (captures interactions the logistic can't).
- calibration            : Platt layer on a validation tail.

All state is built ONLINE, as-of (observe an outcome only after predicting it), so there is no
leakage. The readout reads state features; it does not replace the state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.state.latent import BetaHierarchical
from swm.transition.learned_transition import GradientBoostedClassifier
from swm.transition.readout import LogisticReadout

_LOGIT = lambda p: math.log(max(1e-6, p) / max(1e-6, 1 - p))  # noqa: E731
_SIG = lambda z: 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))  # noqa: E731


@dataclass
class ResponseConfig:
    use_pooled_rate: bool = True
    use_multilevel: bool = False
    use_recency: bool = False
    use_state_features: bool = False
    use_interactions: bool = False
    use_message: bool = True
    readout: str = "logistic"          # "pooled" | "logistic" | "gbdt"
    calibrate: bool = False
    prior_strength: float = 4.0
    segment_prior_strength: float = 6.0
    recency_halflife: float = 15.0


@dataclass
class _EntityAcc:
    beta: BetaHierarchical
    ewma: float | None = None
    n: int = 0
    _sum: float = 0.0

    def observe(self, y: float, alpha: float) -> None:
        self.beta.observe(y)
        self.ewma = y if self.ewma is None else alpha * y + (1 - alpha) * self.ewma
        self.n += 1
        self._sum += y


@dataclass
class ResponseModel:
    message_feature_names: list[str]
    config: ResponseConfig = field(default_factory=ResponseConfig)
    global_rate: float = 0.3
    _seg: dict = field(default_factory=dict)          # segment_id -> [sum, n]
    _ent: dict = field(default_factory=dict)          # entity_id -> _EntityAcc
    readout: object = None
    calibrator: tuple | None = None

    # ---------------- state accessors (as-of) ----------------
    def _segment_rate(self, seg_id) -> float:
        if seg_id is None:
            return self.global_rate
        s = self._seg.get(seg_id)
        if not s:
            return self.global_rate
        # pool segment toward global
        return (s[0] + self.global_rate * self.config.segment_prior_strength) / (
            s[1] + self.config.segment_prior_strength)

    def _entity(self, eid, seg_id) -> _EntityAcc:
        e = self._ent.get(eid)
        if e is None:
            prior = self._segment_rate(seg_id) if self.config.use_multilevel else self.global_rate
            e = _EntityAcc(beta=BetaHierarchical(segment_rate=prior,
                                                 prior_strength=self.config.prior_strength))
            self._ent[eid] = e
        return e

    def _alpha(self) -> float:
        return 1.0 - math.exp(-math.log(2) / self.config.recency_halflife)

    def _state_features(self, eid, seg_id) -> dict:
        e = self._ent.get(eid)
        seg = self._segment_rate(seg_id)
        if e is None:
            return {"ent_rate": self.global_rate, "ent_recency": self.global_rate, "ent_depth": 0.0,
                    "ent_logdepth": 0.0, "ent_suff": 0.0, "seg_rate": seg}
        rate = e.beta.mean
        return {"ent_rate": rate,
                "ent_recency": e.ewma if e.ewma is not None else rate,
                "ent_depth": float(e.n), "ent_logdepth": math.log1p(e.n),
                "ent_suff": e.beta.n_obs / (e.beta.n_obs + 4.0),
                "seg_rate": seg}

    def _vector(self, eid, seg_id, mf) -> list[float]:
        st = self._state_features(eid, seg_id)
        row = []
        if self.config.use_message:
            row += [float(mf.get(n, 0.0)) for n in self.message_feature_names]
        if self.config.use_pooled_rate:
            row.append(_LOGIT(st["ent_rate"]))
            row.append(_LOGIT(st["seg_rate"]) if self.config.use_multilevel else _LOGIT(self.global_rate))
        if self.config.use_recency:
            row.append(_LOGIT(st["ent_recency"]))
        if self.config.use_state_features:
            row += [st["ent_logdepth"], st["ent_suff"]]
        if self.config.use_interactions and self.config.use_message:
            r = st["ent_rate"]
            row += [r * float(mf.get(n, 0.0)) for n in self.message_feature_names]
        return row

    # ---------------- fit ----------------
    def fit_stream(self, samples, *, global_rate=None, val_frac=0.15):
        if global_rate is None:
            ys = [o for _, _, _, o in samples]
            global_rate = (sum(ys) + 1) / (len(ys) + 2)
        self.global_rate = global_rate
        self._seg.clear(); self._ent.clear()
        alpha = self._alpha()
        X, y = [], []
        for eid, seg_id, mf, o in samples:
            X.append(self._vector(eid, seg_id, mf)); y.append(int(o))
            # transition (as-of: after recording features)
            self._entity(eid, seg_id).observe(o, alpha)
            if seg_id is not None:
                s = self._seg.setdefault(seg_id, [0.0, 0.0]); s[0] += o; s[1] += 1
        if self.config.readout == "pooled" or len(set(y)) < 2:
            self.readout = None
        elif self.config.readout == "gbdt":
            self.readout = GradientBoostedClassifier(n_estimators=80, max_depth=3, seed=0).fit(X, y)
        else:
            self.readout = LogisticReadout(epochs=250).fit(X, y)
        if self.config.calibrate and self.readout is not None and val_frac > 0:
            # refit a Platt layer on the last val_frac of the (already state-built) stream
            cut = int((1 - val_frac) * len(X))
            raw = [self._raw(X[i]) for i in range(cut, len(X))]
            yy = y[cut:]
            if len(set(yy)) == 2:
                m = LogisticReadout(epochs=200).fit([[_LOGIT(min(1 - 1e-6, max(1e-6, r)))] for r in raw], yy)
                lo, hi = min(raw), max(raw)
                if hi - lo > 1e-6:
                    a = (_LOGIT(min(1-1e-6,max(1e-6,m.predict_proba([_LOGIT(hi)])))) -
                         _LOGIT(min(1-1e-6,max(1e-6,m.predict_proba([_LOGIT(lo)]))))) / (_LOGIT(hi)-_LOGIT(lo))
                    b = _LOGIT(min(1-1e-6,max(1e-6,m.predict_proba([_LOGIT(lo)])))) - a*_LOGIT(lo)
                    self.calibrator = (a, b)
        return self

    def _raw(self, vec) -> float:
        if self.readout is None:
            return None
        return self.readout.predict_proba(vec)

    # ---------------- predict ----------------
    def predict(self, eid, seg_id, mf) -> float:
        if self.readout is None:                      # pooled: the state estimate itself
            st = self._state_features(eid, seg_id)
            return st["ent_recency"] if self.config.use_recency else st["ent_rate"]
        p = self.readout.predict_proba(self._vector(eid, seg_id, mf))
        if self.calibrator is not None:
            a, b = self.calibrator
            p = _SIG(a * _LOGIT(min(1 - 1e-6, max(1e-6, p))) + b)
        return p

    def observe(self, eid, seg_id, o):
        self._entity(eid, seg_id).observe(o, self._alpha())
        if seg_id is not None:
            s = self._seg.setdefault(seg_id, [0.0, 0.0]); s[0] += o; s[1] += 1
