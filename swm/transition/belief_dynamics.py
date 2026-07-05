"""Event-conditioned belief-transition operator — the temporal DYNAMICS half of a world model.

Everything else in this repo is cross-sectional: predict an outcome from a state, now. This models how a
belief STATE EVOLVES over time when an EVENT hits: an estimate of the next belief from (a) the belief
trajectory (level, momentum, volatility) and (b) features of the candidate events. It is deliberately
general — the state is just a scalar belief trajectory, so the same operator applies to a population's
market belief or (in principle) a person's belief on a proposition; events are opaque text.

The null/persistence branch (Δ=0, the efficient-market martingale) is the baseline every event effect is
measured against — following classical event studies and the SWM paper (Yu et al. 2026). The learned
branch predicts a belief change Δ from pre-shift features only (no leakage). An optional
`event_impact_fn` injects an LLM-inferred directional impact per transition (our "infer variables, not
outcomes" thesis applied to events) — the operator works with or without it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_STOP = set("the a an of to in on for and or is will be by at as with from this that not it its "
            "what when who how does do 2024 2025 2026".split())


def _tokens(s: str):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP and len(w) > 2}


def state_features(history: list) -> dict:
    """Belief-trajectory features from the look-back window (all strictly before the target)."""
    ps = [float(h["p"]) for h in history] if history else [0.5]
    p = ps[-1]
    k = min(len(ps) - 1, 5)
    mom = p - ps[-1 - k] if k > 0 else 0.0
    diffs = [ps[i] - ps[i - 1] for i in range(1, len(ps))]
    vol = (sum(d * d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
    last = diffs[-1] if diffs else 0.0
    return {"level": p, "dist_half": p - 0.5, "momentum": mom, "last_change": last,
            "volatility": vol, "n_hist": min(1.0, len(ps) / 16.0),
            "room_up": 1.0 - p, "room_down": p}


def event_features(rec, impact: float | None = None) -> dict:
    """Cheap, pre-shift event features: how many candidate events, and their salience to the question.
    `impact` (optional) is an injected LLM-inferred signed directional impact in [-1,1]."""
    news = rec.get("news", []) or []
    qtok = _tokens(rec.get("question", "") + " " + rec.get("description", ""))
    sal = 0.0
    for n in news:
        nt = _tokens(n.get("title", "") + " " + n.get("description", ""))
        if qtok:
            sal = max(sal, len(qtok & nt) / len(qtok))
    return {"n_news": min(1.0, len(news) / 20.0), "salience": sal,
            "z_score": max(-3.0, min(3.0, float(rec.get("z_score", 0.0)))),
            "impact": 0.0 if impact is None else max(-1.0, min(1.0, impact))}


FEATURES = (list(state_features([]).keys()) + list(event_features({}).keys()))


def featurize(rec, impact=None):
    sf = state_features(rec.get("history", []))
    ef = event_features(rec, impact)
    return [sf[k] for k in state_features([])] + [ef[k] for k in event_features({})]


@dataclass
class _Ridge:
    """Tiny standardized ridge regression by gradient descent (predicts belief change Δ)."""
    l2: float = 1.0
    lr: float = 0.05
    epochs: int = 500
    w: list = field(default_factory=list)
    b: float = 0.0
    _mu: list = field(default_factory=list)
    _sd: list = field(default_factory=list)

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self._mu = [sum(r[j] for r in X) / n for j in range(d)]
        self._sd = [max(1e-9, (sum((r[j] - self._mu[j]) ** 2 for r in X) / n) ** 0.5) for j in range(d)]
        Xs = [self._std(r) for r in X]
        self.w = [0.0] * d
        self.b = sum(y) / n
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                pred = self.b + sum(wj * xj for wj, xj in zip(self.w, Xs[i]))
                e = pred - y[i]
                for j in range(d):
                    gw[j] += e * Xs[i][j]
                gb += e
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j] / n)
            self.b -= self.lr * gb / n
        return self

    def _std(self, r):
        return [(v - m) / s for v, m, s in zip(r, self._mu, self._sd)]

    def predict(self, x):
        return self.b + sum(wj * xj for wj, xj in zip(self.w, self._std(x)))


@dataclass
class BeliefTransition:
    """Learned event-conditioned transition: Δ̂ = f(state, event); next belief = clamp(p_t + Δ̂).

    With `gate_by_impact`, the predicted change is scaled by the event-impact magnitude so that a
    transition with no relevant event collapses to the persistence/null branch (Δ≈0) — the efficient-
    market martingale the paper falls back to when no event is attributed. This prevents the operator
    from injecting spurious movement on quiet periods (which would regress MAE), while still moving
    decisively when the LLM judges an event to be strong.
    """
    event_impact_fn: object = None      # optional callable(rec) -> signed impact in [-1,1]
    gate_by_impact: bool = False        # scale Δ by |impact| so no-event transitions -> persistence
    gate_scale: float = 0.5             # |impact| at which the gate saturates to 1
    model: _Ridge = None                # type: ignore

    def _impact(self, rec):
        if self.event_impact_fn is None:
            return None
        try:
            return self.event_impact_fn(rec)
        except Exception:
            return None

    def _gate(self, rec):
        if not self.gate_by_impact:
            return 1.0
        imp = self._impact(rec) or 0.0
        return min(1.0, abs(imp) / max(1e-9, self.gate_scale))

    def fit(self, records, l2=1.0):
        X, y = [], []
        for r in records:
            if not r.get("history") or not r.get("target"):
                continue
            X.append(featurize(r, self._impact(r)))
            y.append(float(r["target"]["p"]) - float(r["history"][-1]["p"]))
        self.model = _Ridge(l2=l2).fit(X, y)
        return self

    def predict_change(self, rec) -> float:
        return self._gate(rec) * self.model.predict(featurize(rec, self._impact(rec)))

    def predict_belief(self, rec) -> float:
        p = float(rec["history"][-1]["p"]) if rec.get("history") else 0.5
        return min(1.0, max(0.0, p + self.predict_change(rec)))
