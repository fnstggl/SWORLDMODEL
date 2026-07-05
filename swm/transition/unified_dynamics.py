"""Unified belief dynamics — ONE event-conditioned transition operator for every scale.

The whole point of a *general* social world model is that the same machinery models a market's collective
belief updating on news AND a person's belief updating on an argument. This operator unifies them:

    Δbelief = responsiveness · event_impact

- AGGREGATE (population / market): `event_impact` is the news impact (EXP-030's LLM channel) and
  `responsiveness` is a market-efficiency factor — how much collective belief moves per unit of event.
- INDIVIDUAL (person): `event_impact` is the message/argument's persuasive force, and `responsiveness`
  is read from the person's `VariableMap` — their openness attenuated by skepticism and by the strength
  of their prior stance (entrenchment). The SAME event moves an open person and an entrenched person
  differently; that heterogeneity is exactly what the VariableMap supplies.

So the aggregate transition is the population average of individual transitions, and an individual
transition is the aggregate form with the person's VariableMap setting the responsiveness. State,
Readout, and Dynamics become one system: the VariableMap (State) feeds the responsiveness of the
transition (Dynamics), whose output is read out as the predicted belief change.
"""
from __future__ import annotations

from dataclasses import dataclass


def responsiveness_from_map(vm) -> float:
    """Person's belief-update responsiveness from their VariableMap: open minds move, skeptical and
    entrenched minds resist. Returns a factor in [0,1]."""
    openness = vm.get("openness_to_outreach", 0.5)
    skepticism = vm.get("skepticism", 0.5)
    entrenchment = abs(vm.get("prior_stance", 0.0))          # a strong prior (either sign) resists change
    return max(0.0, min(1.0, openness * (1.0 - 0.5 * skepticism) * (1.0 - 0.5 * entrenchment)))


@dataclass
class UnifiedBeliefDynamics:
    """Δbelief = responsiveness · event_impact — one operator, aggregate or individual."""
    scale: float = 1.0                       # global magnitude (fit per domain)

    def predict_update(self, event_impact: float, responsiveness: float) -> float:
        """Signed belief change. `event_impact` is the (signed) push; `responsiveness` gates it."""
        return self.scale * responsiveness * event_impact

    # ---- aggregate: responsiveness is a market factor ----
    def update_market(self, event_impact: float, market_responsiveness: float = 1.0) -> float:
        return self.predict_update(event_impact, market_responsiveness)

    # ---- individual: responsiveness comes from the person's VariableMap ----
    def update_person(self, event_impact: float, variable_map) -> float:
        return self.predict_update(event_impact, responsiveness_from_map(variable_map))
