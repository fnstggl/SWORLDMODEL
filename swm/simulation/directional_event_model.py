"""Directional event model — forecast the DIRECTION of pivotal future events, not just their variance.

The event model v1 (`event_model.py`) places calibrated VARIANCE over the horizon but its impacts are
symmetric — it says "a big move is likely" without saying which way. The frontier is DIRECTION: for each
scheduled event on a real calendar, forecast the sign + magnitude of its impact CONDITIONAL on the state at
that event, using the corpus-calibrated elasticities (the harvest). Then roll the calendar forward, evolving
the state between events, so the terminal forecast has a direction driven by the fundamentals — the thing a
persistence/momentum baseline or a symmetric-variance model cannot do, and the thing long-horizon
interventional questions ("what happens by date T if X") actually need.

  - `calendar` — the known future event times (a real event calendar).
  - `impact_fn(state, rng) -> signed impact` — the conditional (directional) impact model. For FOMC this is
    P(hike|macro) from the corpus-calibrated `rate_hike` weights × a calibrated magnitude; pluggable per kind.
  - `evolve_fn(state, dt, rng, last_impact) -> state'` — how the drivers move between events (e.g. momentum:
    recent_move carries the last impact forward), so event k conditions on the state events 1..k-1 produced.

`rollout` returns the terminal distribution AND `p_up` / `expected_move` — a directional, calibrated forecast.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class DirectionalEventModel:
    calendar: list                    # event times over the horizon
    impact_fn: object                 # (state: dict, rng) -> signed impact
    evolve_fn: object = None          # (state, dt, rng, last_impact) -> state'  (drivers between events)

    def _once(self, start, state0, horizon, rng):
        state, level, last, prev_t = dict(state0), start, 0.0, 0.0
        for t in sorted(self.calendar):
            if t > horizon:
                break
            if self.evolve_fn is not None:
                state = self.evolve_fn(state, t - prev_t, rng, last)
            imp = self.impact_fn(state, rng)
            level += imp
            last, prev_t = imp, t
        return level

    def rollout(self, start, state0, horizon, *, n=4000, seed=0, lo=None, hi=None) -> dict:
        rng = random.Random(seed)
        outs = []
        for _ in range(n):
            v = self._once(start, state0, horizon, rng)
            if lo is not None or hi is not None:
                v = min(hi if hi is not None else v, max(lo if lo is not None else v, v))
            outs.append(v)
        outs.sort()
        m = sum(outs) / n
        return {"mean": m, "sd": (sum((o - m) ** 2 for o in outs) / n) ** 0.5,
                "p_up": sum(1 for o in outs if o > start) / n, "expected_move": m - start,
                "p05": outs[int(.05 * n)], "p50": outs[int(.5 * n)], "p95": outs[int(.95 * n)], "n": n}

    def direction(self, start, state0, horizon, *, n=4000, seed=0) -> dict:
        """The directional call: expected move + P(up) + a confidence (distance of p_up from 0.5)."""
        r = self.rollout(start, state0, horizon, n=n, seed=seed)
        return {"expected_move": round(r["expected_move"], 4), "p_up": round(r["p_up"], 4),
                "call": "up" if r["p_up"] > 0.5 else "down", "confidence": round(abs(r["p_up"] - 0.5) * 2, 4)}


def calibrated_impact_fn(prob_up_given_move_fn, magnitude_mean, magnitude_sd, p_move=1.0):
    """Build a directional `impact_fn` with the correct decomposition P(move) × P(up | move): most events HOLD
    (no impact); when an event fires, its DIRECTION comes from the state-conditioned P(up | a move happens) and
    its SIZE from |N(mean, sd)|. Separating the move-FREQUENCY from the move-DIRECTION is essential — forcing a
    move every event compounds the base-rate asymmetry into a spurious drift over the horizon."""
    def fn(state, rng):
        if rng.random() > p_move:
            return 0.0                                    # HOLD — the common case
        p = prob_up_given_move_fn(state)
        direction = 1.0 if rng.random() < p else -1.0
        return direction * abs(rng.gauss(magnitude_mean, magnitude_sd))
    return fn


def momentum_evolve(feature="recent_move", scale=1.0):
    """An `evolve_fn` that carries the last impact into a momentum feature — so an event's direction feeds the
    next event's conditioning (policy inertia, opinion cascades, winning streaks)."""
    def fn(state, dt, rng, last_impact):
        s = dict(state)
        s[feature] = max(-1.0, min(1.0, scale * last_impact)) if last_impact else s.get(feature, 0.0)
        return s
    return fn
