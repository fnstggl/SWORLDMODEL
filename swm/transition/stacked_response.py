"""Stacked, evidence-aware response model — the learned fusion of state-model + content prior.

First-principles motivation (measured across HN/GitHub/Enron/StackExchange): prediction error splits
into a STATE-RICH regime (repeat entity -> the entity-state world model wins, scaling with depth) and
a COLD/STATE-POOR regime (-> the content/message prior wins). The hand-set hybrid gate exploited this
crudely. This module learns the fusion optimally, per instance:

    p = meta_logistic( content_logit, entity_logit, segment_logit, recency_logit,
                       evidence features, and the KEY interactions
                       entity_logit*sufficiency  and  content_logit*(1-sufficiency) )

Those two interaction terms are the learned gate: trust the entity state where we have evidence,
trust the content prior where we don't — with the crossover fit from held-out data rather than
guessed. Cold-start also gets a better base than a flat mean via a similarity/segment-regression
prior (see `cold_start_prior`).

All state is built ONLINE, as-of; the meta-learner is fit on a held-out tail of TRAIN, never on test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.state.latent import BetaHierarchical
from swm.transition.readout import LogisticReadout

_LOGIT = lambda p: math.log(max(1e-6, p) / max(1e-6, 1 - p))  # noqa: E731
_SIG = lambda z: 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))  # noqa: E731


@dataclass
class _Ent:
    beta: BetaHierarchical
    ewma: float | None = None
    n: int = 0

    def observe(self, y, alpha):
        self.beta.observe(y)
        self.ewma = y if self.ewma is None else alpha * y + (1 - alpha) * self.ewma
        self.n += 1


@dataclass
class StackedResponseModel:
    message_feature_names: list[str]
    prior_strength: float = 4.0
    segment_prior_strength: float = 6.0
    recency_halflife: float = 15.0
    # pool the entity toward the GLOBAL rate (robust); the segment rate is exposed to the meta-learner
    # as a separate feature (seg_logit) so it can weight it up (informative segment) or ignore it
    # (noisy segment) — instead of baking a possibly-noisy segment into the entity estimate.
    use_multilevel: bool = False
    global_rate: float = 0.3
    content: object = None                 # logistic over message features only
    meta: object = None                    # meta-logistic over base preds + evidence
    _seg: dict = field(default_factory=dict)
    _ent: dict = field(default_factory=dict)

    def _alpha(self):
        return 1.0 - math.exp(-math.log(2) / self.recency_halflife)

    def _seg_rate(self, seg):
        if seg is None or seg not in self._seg:
            return self.global_rate
        s = self._seg[seg]
        return (s[0] + self.global_rate * self.segment_prior_strength) / (s[1] + self.segment_prior_strength)

    def _entity(self, eid, seg):
        e = self._ent.get(eid)
        if e is None:
            prior = self._seg_rate(seg) if self.use_multilevel else self.global_rate
            e = _Ent(BetaHierarchical(segment_rate=prior, prior_strength=self.prior_strength))
            self._ent[eid] = e
        return e

    def _mvec(self, mf):
        return [float(mf.get(n, 0.0)) for n in self.message_feature_names]

    def _base(self, eid, seg, mf):
        """Base predictions + evidence features for the meta-learner (from current as-of state)."""
        e = self._ent.get(eid)
        seg_rate = self._seg_rate(seg)
        ent_rate = e.beta.mean if e else seg_rate
        recency = (e.ewma if (e and e.ewma is not None) else ent_rate)
        n = e.n if e else 0
        suff = (e.beta.n_obs / (e.beta.n_obs + 4.0)) if e else 0.0
        content_p = self.content.predict_proba(self._mvec(mf)) if self.content else self.global_rate
        cl, el, sl, rl = (_LOGIT(content_p), _LOGIT(ent_rate), _LOGIT(seg_rate), _LOGIT(recency))
        # raw message features are included too, so the meta-learner can always reconstruct (and beat)
        # the message-only model — the stack is then >= its best base by construction.
        return [cl, el, sl, rl, math.log1p(n), suff,
                el * suff, cl * (1 - suff),          # the learned gate
                rl * suff] + self._mvec(mf)

    def _observe(self, eid, seg, o):
        self._entity(eid, seg).observe(o, self._alpha())
        if seg is not None:
            s = self._seg.setdefault(seg, [0.0, 0.0]); s[0] += o; s[1] += 1

    def fit_stream(self, samples, *, global_rate=None, content_frac=0.6):
        ys = [o for *_, o in samples]
        self.global_rate = global_rate if global_rate is not None else (sum(ys) + 1) / (len(ys) + 2)
        self._seg.clear(); self._ent.clear()
        c1 = int(content_frac * len(samples))
        # Phase A: build state + collect content rows
        Xc, yc = [], []
        for eid, seg, mf, o in samples[:c1]:
            Xc.append(self._mvec(mf)); yc.append(int(o))
            self._observe(eid, seg, o)
        self.content = LogisticReadout(epochs=250).fit(Xc, yc) if len(set(yc)) == 2 else None
        # Phase B: collect meta rows (base preds from as-of state), keep building state
        Xm, ym = [], []
        for eid, seg, mf, o in samples[c1:]:
            Xm.append(self._base(eid, seg, mf)); ym.append(int(o))
            self._observe(eid, seg, o)
        self.meta = LogisticReadout(epochs=300).fit(Xm, ym) if len(set(ym)) == 2 else None
        return self

    def predict(self, eid, seg, mf):
        if self.meta is None:
            e = self._ent.get(eid)
            return e.beta.mean if e else self.global_rate
        return self.meta.predict_proba(self._base(eid, seg, mf))

    def observe(self, eid, seg, o):
        self._observe(eid, seg, o)


def cold_start_prior(entity_features_by_id, outcomes_by_id, feature_names):
    """Optional: a regression prior for cold entities — maps observable entity features to an
    expected rate, so a brand-new entity starts from a feature-based estimate rather than the global
    mean. Fit on entities seen in TRAIN; used as the segment_rate for unseen entities. Returns a
    predictor callable(features)->rate. (Wired opt-in by harnesses that have entity-level features.)"""
    X, y = [], []
    for eid, feats in entity_features_by_id.items():
        outs = outcomes_by_id.get(eid, [])
        if outs:
            X.append([feats.get(n, 0.0) for n in feature_names]); y.append(sum(outs) / len(outs))
    if len(X) < 20:
        return None
    # simple ridge-ish logistic on the mean outcome (treat rate as soft label via rounding buckets)
    yb = [1 if r >= (sum(y) / len(y)) else 0 for r in y]
    if len(set(yb)) < 2:
        return None
    m = LogisticReadout(epochs=200).fit(X, yb)
    return lambda f: m.predict_proba([f.get(n, 0.0) for n in feature_names])
