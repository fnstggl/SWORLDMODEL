"""Scenario tree — branch over candidate actions to pick the decision with the best expected value.

A prediction becomes a decision when you compare *actions*. Given a state and a set of candidate
actions (message variants, post timings, framings), expand each into its outcome distribution and
score it by an operator-supplied value function (e.g. expected reach, P(reply), expected revenue).
Returns the ranked actions and the recommended one — the product surface of the world model.

Depth-1 by default (one decision now). `expand_plan` supports multi-step plans by rolling the state
forward greedily; deeper search is a simple recursion on top of this and is left for when a wedge
needs it (the audit's "don't build ahead of a real decision" rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.state.population import PopulationState
from swm.state.state import Action
from swm.transition.aggregate_transition import AggregateTransition
from swm.transition.transition_head import BAND_REPR


def expected_magnitude(band_probs: list[float]) -> float:
    """Expected outcome magnitude under the predicted band distribution."""
    return sum(p * r for p, r in zip(band_probs, BAND_REPR))


@dataclass
class Branch:
    action: Action
    pred: dict
    value: float


@dataclass
class ScenarioTree:
    transition: AggregateTransition
    pop: PopulationState
    value_fn: object = None      # callable(pred) -> float; default = P(score>=40)

    def _value(self, pred: dict) -> float:
        if self.value_fn is not None:
            return self.value_fn(pred)
        return pred["thresholds"].get(40, 0.0)   # default decision value: P(hit)

    def evaluate(self, candidates: list[Action]) -> dict:
        """Score each candidate action from the CURRENT state; return ranked branches + best."""
        branches = []
        for a in candidates:
            pred = self.transition.predict(self.pop, a)
            branches.append(Branch(a, pred, self._value(pred)))
        branches.sort(key=lambda b: b.value, reverse=True)
        best = branches[0]
        spread = branches[0].value - branches[-1].value if len(branches) > 1 else 0.0
        return {
            "report_type": "prediction",
            "ranked": [{"action_id": b.action.action_id, "value": round(b.value, 4),
                        "p_hit": round(b.pred["thresholds"].get(40, 0.0), 4),
                        "expected_magnitude": round(expected_magnitude(b.pred["band_probs"]), 2),
                        "title": b.action.meta.get("title", "")}
                       for b in branches],
            "recommended_action_id": best.action.action_id,
            "decision_spread": round(spread, 4),
        }

    def expand_plan(self, plan: list[Action]) -> list[dict]:
        """Roll a fixed plan forward (teacher-free): predict each step from the evolving state.
        Uses the expected magnitude to advance state deterministically (a mean-field rollout)."""
        import copy
        pop = copy.deepcopy(self.pop)
        tr = copy.deepcopy(self.transition)
        out = []
        for a in plan:
            pred = tr.predict(pop, a)
            em = expected_magnitude(pred["band_probs"])
            out.append({"action_id": a.action_id, "p_hit": round(pred["thresholds"].get(40, 0.0), 4),
                        "expected_magnitude": round(em, 2)})
            tr.transition(pop, a, em)
        return out
