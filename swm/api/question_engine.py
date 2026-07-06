"""Question engine — the front door: an arbitrary question -> mapped drivers -> inferred P(outcome).

This applies the VariableMap architecture to a QUESTION instead of a person: the "variables acting on"
the question's *resolution* are its DRIVERS (for "Will the Fed cut in March?" — the inflation trend, the
labor market, the Fed's stated stance, recent votes; for "Will LeBron win a title?" — roster, health,
standings, schedule). We infer those drivers, aggregate them into a calibrated P(outcome), and read off
the direction (EXP-036: direction follows the lean). For a question with no liquid market, this inferred
lean IS the forecast.

The aggregation deliberately implements Tetlock's *Superforecasting* tenets, not a black box:
  - FERMI-IZE: decompose the question into drivers (sub-questions) — the driver list.
  - BASE-RATE FIRST (outside view): start from the reference-class prior, then adjust — `base_rate`.
  - BAYESIAN LOG-ODDS UPDATE: each driver shifts the log-odds by (direction·strength·confidence); we
    accumulate in logit space (a log-linear opinion pool), the natural home of incremental updating.
  - AVOID OVER/UNDER-REACTION: a global shrink on the total evidence keeps updates calibrated, not
    hyperbolic — Tetlock's "don't overreact to the latest headline."
  - DRAGONFLY EYE: aggregate several independent driver-inference passes (median in logit space) to wash
    out any single view's bias.
  - BALANCED EVIDENCE: the driver spec carries signed direction, so YES-pushing and NO-pushing drivers
    both enter — actively open-minded, not one-sided.
Every forecast returns its driver breakdown, so it is auditable (which drivers moved it, and how).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(z):
    if z < -35:
        return 1e-15
    if z > 35:
        return 1 - 1e-15
    return 1 / (1 + math.exp(-z))


@dataclass
class Driver:
    name: str
    direction: float          # signed pull toward YES in [-1, 1] (+ = raises P(outcome))
    strength: float           # how much this driver matters, [0, 1]
    confidence: float         # how sure we are of this driver's state, [0, 1]
    evidence: str = ""

    def logit_shift(self, kappa: float) -> float:
        return kappa * self.direction * self.strength * self.confidence


@dataclass
class QuestionForecast:
    p_outcome: float
    direction: int            # +1 YES-leaning, -1 NO-leaning, 0 toss-up
    confidence: float         # |p - 0.5| * 2
    base_rate: float
    drivers: list             # the driver breakdown (auditable)
    n_views: int = 1          # dragonfly: how many independent inference passes were aggregated

    def as_dict(self):
        return {"p_outcome": round(self.p_outcome, 4), "direction": self.direction,
                "confidence": round(self.confidence, 4), "base_rate": round(self.base_rate, 4),
                "n_views": self.n_views,
                "drivers": [{"name": d.name, "direction": d.direction, "strength": d.strength,
                             "confidence": d.confidence, "evidence": d.evidence} for d in self.drivers]}


@dataclass
class QuestionEngine:
    kappa: float = 1.2            # per-driver evidence scale (log-odds per unit direction·strength·conf)
    evidence_shrink: float = 0.7  # global shrink on total driver evidence (anti-overreaction, Tetlock)

    def aggregate(self, base_rate: float, drivers: list) -> float:
        """Base-rate log-odds + shrunk sum of driver shifts -> P(outcome). The Bayesian core."""
        z = _logit(base_rate)
        z += self.evidence_shrink * sum(d.logit_shift(self.kappa) for d in drivers)
        return _sigmoid(z)

    def forecast_from_views(self, views: list) -> QuestionForecast:
        """views: list of (base_rate, [Driver,...]) from independent passes. Dragonfly-aggregate in logit
        space (median), keep the union of drivers for the breakdown."""
        if not views:
            return QuestionForecast(0.5, 0, 0.0, 0.5, [])
        ps = sorted(self.aggregate(b, ds) for b, ds in views)
        p = ps[len(ps) // 2]                                   # median of the views
        base = sum(b for b, _ in views) / len(views)
        drivers = max(views, key=lambda v: len(v[1]))[1]        # richest driver set for the audit trail
        direction = 1 if p > 0.55 else (-1 if p < 0.45 else 0)
        return QuestionForecast(p, direction, abs(p - 0.5) * 2, base, drivers, n_views=len(views))

    def forecast(self, question: str, driver_infer_fn, *, n_views: int = 1, context=None) -> QuestionForecast:
        """driver_infer_fn(question, context) -> (base_rate, [Driver,...]). Runs n_views passes (dragonfly)."""
        views = []
        for _ in range(max(1, n_views)):
            try:
                base, drivers = driver_infer_fn(question, context)
            except Exception:
                continue
            if drivers is not None:
                views.append((base if base is not None else 0.5,
                              [d if isinstance(d, Driver) else Driver(**d) for d in drivers]))
        return self.forecast_from_views(views)


def drivers_from_payload(payload: list) -> list:
    """Build Driver objects from a raw agent payload of dicts (tolerant of missing fields)."""
    out = []
    for d in payload or []:
        try:
            out.append(Driver(name=str(d.get("name", "?")), direction=float(d.get("direction", 0.0)),
                              strength=float(d.get("strength", 0.5)), confidence=float(d.get("confidence", 0.5)),
                              evidence=str(d.get("evidence", ""))[:200]))
        except Exception:
            continue
    return out
