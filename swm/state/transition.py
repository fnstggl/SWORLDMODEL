"""The transition model: p(next_state, outcome | current_state, action) — audit C.8, spec section 2.

This is the difference between a predictor and a world model. A predictor gives p(outcome | features).
A transition model ALSO evolves the state: after the action's outcome, the entity's latent traits,
the domain's reputation, and the topic's salience all update — so the NEXT prediction is conditioned
on a changed world. That recurrence is what makes multi-step rollouts meaningful.

Composition (audit): a statistical OUTCOME HEAD (calibrated) predicts the outcome distribution over
score-bands; deterministic FACTOR UPDATE RULES evolve the state; the LLM is used only upstream for
feature extraction / qualitative priors, never as the probability source.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from swm.state.factors import FactorRegistry
from swm.state.state import Action, OutcomeEvent, WorldState
from swm.transition.readout import LogisticReadout

BAND_EDGES = [10, 40, 100, 300]                 # bands: <10, 10-39, 40-99, 100-299, 300+
BAND_REPR = [3.0, 20.0, 65.0, 180.0, 500.0]     # representative magnitude per band (for state update)


def _band(score: float) -> int:
    return sum(1 for e in BAND_EDGES if score >= e)


@dataclass
class OutcomeHead:
    """Calibrated statistical outcome model over the factor vector: P(score>=thr) per threshold,
    from which a band distribution is derived. One logistic per threshold (small, robust)."""
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


@dataclass
class PriorHead:
    """Uncalibrated prior outcome distribution, used for rollouts on domains/horizons with NO
    backtest. Sampling still works (so /rollout returns real trajectories) but the honesty gate
    labels the result 'unvalidated'. Never used where a fitted, backtested head exists."""
    band_probs: tuple[float, ...] = (0.80, 0.13, 0.045, 0.02, 0.005)

    def predict(self, x: list[float]) -> dict:
        bp = list(self.band_probs)
        thr = {e: sum(bp[i + 1:]) for i, e in enumerate(BAND_EDGES)}  # P(>=edge)
        return {"thresholds": thr, "band_probs": bp}


@dataclass
class TransitionModel:
    registry: FactorRegistry
    head: OutcomeHead
    exclude: tuple[str, ...] = ()

    def predict_outcome(self, state: WorldState, action: Action) -> dict:
        e = state.entity(action.actor_id)
        x = self.registry.vector(e, action, state.context_state, self.exclude)
        return self.head.predict(x)

    def step(self, state: WorldState, action: Action, *, observed: float | None = None,
             rng: random.Random | None = None) -> tuple[WorldState, OutcomeEvent]:
        """One transition. Returns (next_state, outcome_event). If `observed` is given it is used
        (teacher-forced); else a magnitude is SAMPLED from the predicted band distribution."""
        pred = self.predict_outcome(state, action)
        if observed is not None:
            magnitude = observed
        else:
            rng = rng or random.Random(0)
            band = _rand_band(pred["band_probs"], rng)
            magnitude = _sample_in_band(band, rng)
        nxt = copy.deepcopy(state)
        e = nxt.entity(action.actor_id)
        self.registry.apply_update(e, nxt.context_state, action, magnitude)
        nxt.timestamp = action.timing.get("ts", state.timestamp)
        ev = OutcomeEvent(timestamp=nxt.timestamp, entity_id=action.actor_id,
                          action_id=action.action_id, observed=float(_band(magnitude)),
                          magnitude=magnitude, raw={"band_probs": pred["band_probs"]})
        return nxt, ev


def _rand_band(band_probs: list[float], rng: random.Random) -> int:
    u, c = rng.random(), 0.0
    for i, p in enumerate(band_probs):
        c += p
        if u <= c:
            return i
    return len(band_probs) - 1


def _sample_in_band(band: int, rng: random.Random) -> float:
    lo = 0 if band == 0 else BAND_EDGES[band - 1]
    hi = BAND_EDGES[band] if band < len(BAND_EDGES) else BAND_EDGES[-1] * 4
    # log-uniform within the band (heavy-tailed)
    return math.exp(rng.uniform(math.log(max(1, lo) + 1), math.log(hi + 1))) - 1
