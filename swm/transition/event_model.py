"""Future-event model — forecast the DISTRIBUTION of belief moves over a horizon.

EXP-033's verdict: you cannot beat the martingale on the POINT forecast, because the *direction* of
future surprises is unforecastable in an efficient belief series. So the event model's job is not a
better point — it is a **calibrated predictive distribution**: forecast *when* volatility will hit
(heteroskedastic per-step variance) and carry any *known* directional signal (the current event's impact,
decaying), then Monte-Carlo the belief forward into a distribution over outcomes. This is what turns
"predict the next nudge" into "simulate forward with honest uncertainty."

Two learned pieces, both no-cheat (fit on train trajectories):
  - drift(state)      : E[Δ | state] — the expected one-step change (near-zero for efficient series);
  - log_var(state)    : the per-step innovation variance as a function of state (recent volatility,
                        distance to the 0/1 boundary, level) — so the band is wide before turbulent
                        periods and tight in calm ones (heteroskedastic), instead of one constant width.
A known current event contributes a decaying directional drift (its effect persists then fades / resolves).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def _feats(prices: list) -> list:
    p = prices[-1]
    k = min(len(prices) - 1, 5)
    mom = (p - prices[-1 - k]) / k if k > 0 else 0.0
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    vol = (sum(d * d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.02
    return [p, abs(p - 0.5), vol, p * (1 - p), abs(mom)]


class _Reg:
    """Tiny standardized ridge (shared by drift and log-variance heads)."""
    def __init__(self, l2=1.0, lr=0.05, epochs=250):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w, self.b, self._mu, self._sd = [], 0.0, [], []

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self._mu = [sum(r[j] for r in X) / n for j in range(d)]
        self._sd = [max(1e-9, (sum((r[j] - self._mu[j]) ** 2 for r in X) / n) ** 0.5) for j in range(d)]
        Xs = [[(r[j] - self._mu[j]) / self._sd[j] for j in range(d)] for r in X]
        self.w = [0.0] * d; self.b = sum(y) / n
        for _ in range(self.epochs):
            gw = [0.0] * d; gb = 0.0
            for i in range(n):
                e = (self.b + sum(wj * xj for wj, xj in zip(self.w, Xs[i]))) - y[i]
                for j in range(d):
                    gw[j] += e * Xs[i][j]
                gb += e
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j] / n)
            self.b -= self.lr * gb / n
        return self

    def predict(self, x):
        xs = [(v - m) / s for v, m, s in zip(x, self._mu, self._sd)]
        return self.b + sum(wj * xj for wj, xj in zip(self.w, xs))


@dataclass
class EventModel:
    drift: _Reg = None                    # type: ignore
    var: _Reg = None                      # type: ignore   predicts E[Δ² | state] directly (not log)
    impact_scale: float = 0.1             # step-1 strength of a known event
    impact_decay: float = 0.6             # how the known event's directional effect fades per step
    drift_shrink: float = 0.0             # 0 = no endogenous drift (EXP-033: drift hurts); tune upward only if it helps
    sigma_mult: float = 1.0               # global variance calibration (tuned to hit interval coverage)
    sigma_floor: float = 0.005

    def fit(self, sequences):
        X, dy, vy = [], [], []
        for seq in sequences:
            for i in range(2, len(seq)):
                f = _feats(seq[:i]); delta = seq[i] - seq[i - 1]
                X.append(f); dy.append(delta); vy.append(delta * delta)     # E[Δ²|state], not log (no Jensen bias)
        self.drift = _Reg().fit(X, dy)
        self.var = _Reg().fit(X, vy)
        return self

    def step_sigma(self, prices) -> float:
        return self.sigma_mult * max(self.sigma_floor,
                                     math.sqrt(max(self.sigma_floor ** 2, self.var.predict(_feats(prices)))))

    def step_drift(self, prices) -> float:
        return self.drift_shrink * self.drift.predict(_feats(prices))

    def rollout_samples(self, history, horizon, *, impact=0.0, n_samples=400, seed=0):
        """Monte-Carlo forward -> array (n_samples x horizon) of sampled beliefs. Uses a seeded RNG."""
        rng = random.Random(seed)
        paths = [list(history) for _ in range(n_samples)]
        term = [[0.0] * horizon for _ in range(n_samples)]
        for s in range(n_samples):
            path = paths[s]
            for h in range(horizon):
                d = self.step_drift(path)
                d += self.impact_scale * impact * (self.impact_decay ** h)     # decaying known-event drift
                d += self.step_sigma(path) * rng.gauss(0, 1)                    # heteroskedastic innovation
                nxt = min(1.0, max(0.0, path[-1] + d))
                path.append(nxt); term[s][h] = nxt
        return term

    def forecast(self, history, horizon, *, impact=0.0, n_samples=400, seed=0, lo=0.1, hi=0.9):
        term = self.rollout_samples(history, horizon, impact=impact, n_samples=n_samples, seed=seed)
        out = []
        for h in range(horizon):
            col = sorted(term[s][h] for s in range(n_samples))
            out.append({"mean": sum(col) / n_samples, "lo": col[int(lo * n_samples)],
                        "hi": col[min(n_samples - 1, int(hi * n_samples))], "samples": col})
        return out
