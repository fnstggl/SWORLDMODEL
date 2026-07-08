"""Calibrated outcome heads over score-bands — the shared readout for every transition model.

Factored out of swm/state/transition.py so aggregate and individual transitions share one honest
head instead of duplicating it. The head maps a feature vector to a distribution over score bands
via one monotone logistic per threshold. It is deliberately boring and dependency-free; it is the
baseline any fancier model must beat on the harness, and it is NEVER an LLM.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.transition.readout import LogisticReadout

BAND_EDGES = [10, 40, 100, 300]                 # bands: <10, 10-39, 40-99, 100-299, 300+
BAND_REPR = [3.0, 20.0, 65.0, 180.0, 500.0]     # representative magnitude per band (for state update)


def band_of(score: float, edges: list[int] | tuple[int, ...] = tuple(BAND_EDGES)) -> int:
    return sum(1 for e in edges if score >= e)


@dataclass
class OutcomeHead:
    """P(score >= thr) per threshold (monotone), from which a band distribution is derived."""
    thresholds: tuple[int, ...] = tuple(BAND_EDGES)
    models: dict[int, LogisticReadout] = field(default_factory=dict)

    def fit(self, X: list[list[float]], scores: list[float]) -> "OutcomeHead":
        for thr in self.thresholds:
            y = [1 if s >= thr else 0 for s in scores]
            self.models[thr] = (LogisticReadout(seed=thr).fit(X, y)
                                if len(set(y)) == 2 else None)
        return self

    def predict(self, x: list[float]) -> dict:
        t = {}
        for thr in self.thresholds:
            m = self.models.get(thr)
            t[thr] = m.predict_proba(x) if m else 0.0
        vals = [t[thr] for thr in self.thresholds]
        for i in range(1, len(vals)):
            vals[i] = min(vals[i], vals[i - 1])          # monotone
        bands = [1 - vals[0]]
        for i in range(len(vals) - 1):
            bands.append(max(1e-6, vals[i] - vals[i + 1]))
        bands.append(vals[-1])
        s = sum(bands)
        return {"thresholds": {thr: t[thr] for thr in self.thresholds},
                "band_probs": [b / s for b in bands]}

    def to_dict(self) -> dict:
        return {"thresholds": list(self.thresholds),
                "models": {str(thr): (m.to_dict() if m else None)
                           for thr, m in self.models.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "OutcomeHead":
        h = cls(thresholds=tuple(d["thresholds"]))
        h.models = {int(thr): (LogisticReadout.from_dict(md) if md else None)
                    for thr, md in d["models"].items()}
        return h


@dataclass
class PriorHead:
    """Uncalibrated prior band distribution for domains/horizons with NO backtest. Sampling works
    (so a rollout returns real trajectories) but the honesty gate labels the result 'unvalidated'.
    Never used where a fitted, backtested head exists. IGNORES the state vector by construction —
    do not use it to claim state-sensitivity."""
    band_probs: tuple[float, ...] = (0.80, 0.13, 0.045, 0.02, 0.005)
    thresholds: tuple[int, ...] = tuple(BAND_EDGES)

    def predict(self, x: list[float]) -> dict:
        bp = list(self.band_probs)
        thr = {e: sum(bp[i + 1:]) for i, e in enumerate(BAND_EDGES)}  # P(>=edge)
        return {"thresholds": thr, "band_probs": bp}


def rand_band(band_probs: list[float], rng: random.Random) -> int:
    u, c = rng.random(), 0.0
    for i, p in enumerate(band_probs):
        c += p
        if u <= c:
            return i
    return len(band_probs) - 1


def sample_in_band(band: int, rng: random.Random,
                   edges: list[int] | tuple[int, ...] = tuple(BAND_EDGES)) -> float:
    lo = 0 if band == 0 else edges[band - 1]
    hi = edges[band] if band < len(edges) else edges[-1] * 4
    return math.exp(rng.uniform(math.log(max(1, lo) + 1), math.log(hi + 1))) - 1
