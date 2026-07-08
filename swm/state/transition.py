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
import random
from dataclasses import dataclass

from swm.state.factors import FactorRegistry
from swm.state.state import Action, OutcomeEvent, WorldState
# canonical home of the head is swm/transition/transition_head.py; re-exported here for the
# existing imports (`from swm.state.transition import OutcomeHead, TransitionModel, _band`).
from swm.transition.transition_head import (BAND_EDGES, BAND_REPR, OutcomeHead, PriorHead,
                                            band_of as _band, rand_band as _rand_band,
                                            sample_in_band as _sample_in_band)

__all__ = ["BAND_EDGES", "BAND_REPR", "OutcomeHead", "PriorHead", "TransitionModel", "_band"]


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
