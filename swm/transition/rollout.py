"""Multi-step belief rollout — forecasting a belief trajectory forward over a horizon, with uncertainty.

The one-step operator (EXP-030) predicts s_{t+1} from s_t + a known event. Real forecasting needs to roll
forward H steps to a horizon — days, weeks — where the hard truth is that FUTURE EVENTS ARE UNKNOWN. So
a rollout is fundamentally two regimes:

  - the step(s) where we have an event -> the event-conditioned transition (EXP-030's LLM impact);
  - all later steps with no known event -> ENDOGENOUS dynamics only: the belief's own momentum and its
    mean-reversion / drift toward resolution, learned from data. This is the honest ceiling — without
    information about future events you cannot anticipate the moves they cause.

Uncertainty is propagated by Monte-Carlo: each step adds a learned innovation (a random daily change
sampled from the residual distribution), so the band WIDENS with horizon. The forecast at horizon h is a
DISTRIBUTION over belief, summarized as (mean, lo, hi) — the honest way to handle branching futures: we
don't pick one branch, we report the percentages.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _traj_features(prices: list) -> list:
    """Endogenous state: level, recent momentum, volatility, distance to the 0/1 boundaries."""
    p = prices[-1]
    k = min(len(prices) - 1, 5)
    mom = (p - prices[-1 - k]) / k if k > 0 else 0.0
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    vol = (sum(d * d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
    return [p, mom, vol, p * (1 - p)]        # p*(1-p): variance/room to move, shrinks near 0/1


@dataclass
class MultiStepRollout:
    """Learns a per-step endogenous change model E[Δ|state] + an innovation scale, then rolls forward."""
    w: list = field(default_factory=list)
    b: float = 0.0
    sigma: float = 0.03                       # per-step innovation std (residual), learned
    _mu: list = field(default_factory=list)
    _sd: list = field(default_factory=list)

    def fit(self, sequences, l2=1.0, lr=0.05, epochs=300):
        """sequences: list of price lists (>=2 points). Learns Δ_next ~ f(traj features) + residual std."""
        X, y = [], []
        for seq in sequences:
            for i in range(2, len(seq)):
                X.append(_traj_features(seq[:i])); y.append(seq[i] - seq[i - 1])
        if not X:
            return self
        d = len(X[0]); n = len(X)
        self._mu = [sum(r[j] for r in X) / n for j in range(d)]
        self._sd = [max(1e-9, (sum((r[j] - self._mu[j]) ** 2 for r in X) / n) ** 0.5) for j in range(d)]
        Xs = [[(r[j] - self._mu[j]) / self._sd[j] for j in range(d)] for r in X]
        self.w = [0.0] * d; self.b = sum(y) / n
        for _ in range(epochs):
            gw = [0.0] * d; gb = 0.0
            for i in range(n):
                pred = self.b + sum(wj * xj for wj, xj in zip(self.w, Xs[i]))
                e = pred - y[i]
                for j in range(d):
                    gw[j] += e * Xs[i][j]
                gb += e
            for j in range(d):
                self.w[j] -= lr * (gw[j] / n + l2 * self.w[j] / n)
            self.b -= lr * gb / n
        resid = []
        for xs, yi in zip(Xs, y):
            resid.append(yi - (self.b + sum(wj * xj for wj, xj in zip(self.w, xs))))
        self.sigma = max(1e-4, (sum(e * e for e in resid) / len(resid)) ** 0.5)
        return self

    def _endo_step(self, prices) -> float:
        if not self.w:
            return 0.0
        xs = [(v - m) / s for v, m, s in zip(_traj_features(prices), self._mu, self._sd)]
        return self.b + sum(wj * xj for wj, xj in zip(self.w, xs))

    def rollout(self, history: list, horizon: int, *, first_step_impact: float = 0.0,
                impact_scale: float = 0.1, n_samples: int = 200, seed_prices=None):
        """Forecast belief at each of `horizon` future steps. `first_step_impact` (signed) applies the
        event-conditioned move at step 1 only; later steps are endogenous. Returns list of
        {mean, lo, hi} per step. Deterministic MC (no RNG dependency): innovations on a fixed grid."""
        base = list(history)
        # deterministic symmetric innovation grid (avoids Math.random; stable across runs)
        grid = [(-1.5 + 3.0 * k / (n_samples - 1)) for k in range(n_samples)] if n_samples > 1 else [0.0]
        paths = [list(base) for _ in range(n_samples)]
        out = []
        for h in range(horizon):
            vals = []
            for s, path in enumerate(paths):
                delta = self._endo_step(path)
                if h == 0:
                    delta += impact_scale * first_step_impact
                delta += self.sigma * grid[s]           # innovation -> widening band
                nxt = min(1.0, max(0.0, path[-1] + delta))
                path.append(nxt); vals.append(nxt)
            vals.sort()
            lo = vals[max(0, int(0.10 * n_samples))]
            hi = vals[min(n_samples - 1, int(0.90 * n_samples))]
            out.append({"mean": sum(vals) / len(vals), "lo": lo, "hi": hi})
        return out
