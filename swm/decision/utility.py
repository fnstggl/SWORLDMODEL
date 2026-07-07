"""Utilities and risk objectives for the action layer.

Two distinct things, deliberately separated:

  - a UTILITY maps a single outcome to the scalar you want to maximize (`P(reply)`, `profit`, ...). It is
    applied per Monte-Carlo sample.
  - an OBJECTIVE reduces the *sample* of utility values to the number arms are compared by, WITH a confidence
    interval. `Mean` is risk-neutral (maximize E[U]); `Quantile`/`CVaR` are risk-averse (maximize a lower
    quantile / the downside tail mean) — because a decision-maker often wants "the highest expected profit
    that doesn't blow up conversion", not the raw mean. Do not hardcode maximize-the-mean.

The objective owns its CI so best-arm racing can say "A beats B" or honestly "tie within noise": `Mean` uses
a normal standard-error interval; the risk objectives use a percentile bootstrap (valid for any statistic).
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist


@dataclass
class Utility:
    """Per-outcome scalar to maximize. `fn(outcome) -> float`."""
    fn: object
    desc: str = "utility"

    def __call__(self, outcome):
        return float(self.fn(outcome))


def prob_target(predicate, desc="P(target)") -> Utility:
    """Maximize the probability the outcome satisfies `predicate` (E[indicator] = P)."""
    return Utility(lambda o: 1.0 if predicate(o) else 0.0, desc)


def label_is(label, desc=None) -> Utility:
    """For a categorical outcome: maximize P(outcome == label)."""
    return Utility(lambda o: 1.0 if o == label else 0.0, desc or f"P({label})")


def value(fn, desc="E[value]") -> Utility:
    """Maximize an arbitrary numeric value read from the outcome (e.g. profit = price·P(buy))."""
    return Utility(fn, desc)


def identity(desc="outcome") -> Utility:
    return Utility(lambda o: float(o), desc)


_ND = NormalDist()


def _z(conf: float) -> float:
    return _ND.inv_cdf(1 - (1 - conf) / 2)


class Objective:
    """Reduce a sample of utility values to a comparison scalar, with a confidence interval."""
    name = "objective"

    def value(self, samples):
        raise NotImplementedError

    def ci(self, samples, conf, rng):
        raise NotImplementedError


class Mean(Objective):
    name = "mean"

    def value(self, samples):
        return sum(samples) / len(samples) if samples else 0.0

    def ci(self, samples, conf, rng):
        n = len(samples)
        if n < 2:
            return (float("-inf"), float("inf"))
        m = self.value(samples)
        var = sum((x - m) ** 2 for x in samples) / (n - 1)
        se = (var / n) ** 0.5
        h = _z(conf) * se
        return (m - h, m + h)


def _bootstrap_ci(samples, value_fn, conf, rng, resamples=200):
    n = len(samples)
    if n < 2:
        return (float("-inf"), float("inf"))
    boots = []
    for _ in range(resamples):
        rs = [samples[rng.randrange(n)] for _ in range(n)]
        boots.append(value_fn(rs))
    boots.sort()
    a = (1 - conf) / 2
    lo = boots[min(resamples - 1, int(a * resamples))]
    hi = boots[min(resamples - 1, int((1 - a) * resamples))]
    return (lo, hi)


class Quantile(Objective):
    """Maximize a lower quantile of utility (risk-averse: raise the floor). q=0.25 => the 25th percentile."""
    def __init__(self, q=0.25):
        self.q = q
        self.name = f"q{int(q * 100)}"

    def value(self, samples):
        if not samples:
            return 0.0
        s = sorted(samples)
        return s[min(len(s) - 1, int(self.q * len(s)))]

    def ci(self, samples, conf, rng):
        return _bootstrap_ci(samples, self.value, conf, rng)


class CVaR(Objective):
    """Maximize the mean of the WORST `alpha` fraction of outcomes (downside-averse). alpha=0.2 => average of
    the bottom 20% — you optimize the bad tail, not the average."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.name = f"cvar{int(alpha * 100)}"

    def value(self, samples):
        if not samples:
            return 0.0
        s = sorted(samples)
        k = max(1, int(self.alpha * len(s)))
        return sum(s[:k]) / k

    def ci(self, samples, conf, rng):
        return _bootstrap_ci(samples, self.value, conf, rng)
