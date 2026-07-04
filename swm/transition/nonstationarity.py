"""Nonstationarity / drift tracking (spec: "nonstationarity/drift indicators").

Social base rates drift: HN's front-page bar moves, a segment's reply rate decays as a list ages,
a topic saturates. A model fit on old data grows stale silently. This module makes drift a
first-class, measurable state variable and gives the transition two levers:

- a DRIFT INDICATOR: recent-window mean vs. older-window mean, normalized — a signed, bounded
  number that says "the world is moving and in which direction".
- an UNCERTAINTY WIDENING factor: under detected drift, widen predictive intervals (lower the
  effective evidence weight) so a drifting model reports honest uncertainty instead of confident
  staleness.

Both are cheap, online, and leakage-free (they only look backward). The transition modules read
`DriftTracker.indicator()` into `PopulationState.drift` and may multiply predictive uncertainty by
`inflation()`.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class DriftTracker:
    """Two EMAs at different timescales; their gap is the drift signal.

    fast tracks the recent regime, slow the background. `indicator` in ~[-1,1] is the normalized
    gap. A CUSUM-style running sum flags a sustained shift (change-point) rather than noise.
    """
    fast_halflife: float = 30.0     # in observations
    slow_halflife: float = 300.0
    fast: float | None = None
    slow: float | None = None
    _var: float = 0.0               # running variance of observations (for normalization)
    _mean: float = 0.0
    _n: int = 0
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=50))

    def _alpha(self, halflife: float) -> float:
        return 1.0 - math.exp(-math.log(2) / halflife)

    def observe(self, value: float) -> None:
        self.recent.append(value)
        # Welford-ish running moments
        self._n += 1
        d = value - self._mean
        self._mean += d / self._n
        self._var += d * (value - self._mean)
        if self.fast is None:
            self.fast = self.slow = value
        else:
            self.fast += self._alpha(self.fast_halflife) * (value - self.fast)
            self.slow += self._alpha(self.slow_halflife) * (value - self.slow)
        # CUSUM on standardized deviation from the slow mean
        sd = self.std or 1.0
        z = (value - (self.slow or value)) / sd
        k = 0.5  # slack
        self.cusum_pos = max(0.0, self.cusum_pos + z - k)
        self.cusum_neg = max(0.0, self.cusum_neg - z - k)

    @property
    def std(self) -> float:
        if self._n < 2:
            return 0.0
        return math.sqrt(max(0.0, self._var / (self._n - 1)))

    def indicator(self) -> float:
        """Signed normalized drift in ~[-1,1]: (fast-slow)/scale, squashed."""
        if self.fast is None or self.slow is None:
            return 0.0
        scale = (self.std or abs(self.slow) or 1.0)
        return math.tanh((self.fast - self.slow) / (scale + 1e-9))

    def changepoint(self, threshold: float = 5.0) -> bool:
        """True if the CUSUM statistic indicates a sustained regime shift."""
        return max(self.cusum_pos, self.cusum_neg) > threshold

    def inflation(self, cap: float = 2.5) -> float:
        """Uncertainty-widening multiplier >= 1. Larger drift / a flagged change-point widens
        predictive uncertainty (equivalently, discounts effective evidence)."""
        base = 1.0 + 1.5 * abs(self.indicator())
        if self.changepoint():
            base *= 1.5
        return min(cap, base)

    def summary(self) -> dict[str, float]:
        return {
            "drift_indicator": round(self.indicator(), 4),
            "changepoint": 1.0 if self.changepoint() else 0.0,
            "uncertainty_inflation": round(self.inflation(), 4),
            "fast_mean": round(self.fast, 4) if self.fast is not None else 0.0,
            "slow_mean": round(self.slow, 4) if self.slow is not None else 0.0,
        }
