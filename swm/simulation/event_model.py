"""Event model — forecast the DISTRIBUTION of pivotal future events over a horizon, and their impact.

The honest long-horizon gap: outcomes often move in discrete JUMPS at pivotal events (a rate decision, an
election, an earnings call, a scandal), not smooth diffusion. A persistence/diffusion forecaster misplaces
the variance — too narrow between events, wrong at the jumps. This model places it correctly:

  - a CALENDAR of known scheduled events (their times), each ACTIVE with some probability and carrying an
    IMPACT distribution (mean + sd of the jump);
  - a background HAZARD of surprise events (a Poisson rate over the horizon), same impact family;
  - `sample(horizon)` draws the event stream; `rollout` sums impacts onto a starting level to get the
    terminal DISTRIBUTION and horizon intervals.

`calibrate` fits the model from a history of per-step moves (how often a pivotal move happens and how big),
so the event variance is data-grounded — and `interval_coverage` checks it is CALIBRATED over the horizon
(the event model's core job: nominal-X% intervals contain X% of realized futures), where persistence is not.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class EventModel:
    calendar: list = field(default_factory=list)   # times of known scheduled events over the horizon
    p_active: float = 1.0                           # P(a scheduled event actually moves the outcome)
    impact_mean: float = 0.0                        # mean signed jump of an active event
    impact_sd: float = 0.0                          # spread of the jump
    hazard: float = 0.0                             # surprise-event rate per unit time (Poisson)

    def sample(self, horizon: float, rng) -> list:
        """Draw the event stream over [0, horizon]: scheduled events that fire + Poisson surprises, each with
        a sampled impact. Returns [(time, impact)]."""
        events = []
        for t in self.calendar:
            if t <= horizon and rng.random() < self.p_active:
                events.append((t, rng.gauss(self.impact_mean, self.impact_sd)))
        lam = self.hazard * horizon
        k = 0
        if lam > 0:                                 # Poisson(lam) via Knuth
            L, p = math.exp(-lam), 1.0
            while p > L:
                k += 1
                p *= rng.random()
            k -= 1
        for _ in range(k):
            events.append((rng.uniform(0, horizon), rng.gauss(self.impact_mean, self.impact_sd)))
        return sorted(events)

    def rollout(self, start: float, horizon: float, *, n: int = 4000, seed: int = 0,
                lo: float = None, hi: float = None) -> dict:
        """Monte-Carlo the terminal level = start + Σ event impacts. Returns mean + horizon interval."""
        rng = random.Random(seed)
        outs = []
        for _ in range(n):
            v = start + sum(imp for _, imp in self.sample(horizon, rng))
            if lo is not None or hi is not None:
                v = min(hi if hi is not None else v, max(lo if lo is not None else v, v))
            outs.append(v)
        outs.sort()
        m = sum(outs) / n
        return {"mean": m, "sd": (sum((o - m) ** 2 for o in outs) / n) ** 0.5,
                "p05": outs[int(.05 * n)], "p50": outs[int(.5 * n)], "p95": outs[int(.95 * n)], "n": n}

    @classmethod
    def calibrate(cls, moves, *, threshold=0.05, calendar=None, hazard=0.0):
        """Fit from a history of per-step moves: `p_active` = fraction that are pivotal (|move|>threshold),
        and the impact mean/sd from those pivotal moves. The event variance is then data-grounded."""
        pivotal = [m for m in moves if abs(m) > threshold]
        p_active = len(pivotal) / len(moves) if moves else 0.0
        mean = sum(pivotal) / len(pivotal) if pivotal else 0.0
        sd = (sum((m - mean) ** 2 for m in pivotal) / len(pivotal)) ** 0.5 if pivotal else 0.0
        return cls(calendar=calendar or [], p_active=p_active, impact_mean=mean, impact_sd=sd, hazard=hazard)


def interval_coverage(truths, los, his, nominal=0.8):
    """Fraction of realized futures inside the predicted interval — the event model's calibration check."""
    n = len(truths)
    inside = sum(1 for t, a, b in zip(truths, los, his) if a <= t <= b)
    return {"nominal": nominal, "empirical_coverage": round(inside / n, 4) if n else float("nan"),
            "n": n, "calibrated": abs(inside / n - nominal) < 0.1 if n else False}
