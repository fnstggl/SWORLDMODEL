"""Counterfactuals — do(action) comparisons and instance-level decision lift.

The decision question is counterfactual: "if I sent B instead of A, to this entity, in this state,
what changes?" Because the world model conditions the outcome distribution on an explicit state +
action, swapping the action while holding the state fixed gives a clean do(action) contrast — no
new data required. This is the engine under compare-actions and under the decision-lift metric.

Honest scope: these are model-based counterfactuals under the fitted transition, not identified
causal effects. They are only as good as the head's calibration on held-out data (which the
backtest grades). We label them PREDICTION and carry the grade; we never claim a causal guarantee.
"""
from __future__ import annotations

from swm.simulation.scenario_tree import expected_magnitude
from swm.state.population import PopulationState
from swm.state.state import Action
from swm.transition.aggregate_transition import AggregateTransition


def contrast(transition: AggregateTransition, pop: PopulationState, action_a: Action,
             action_b: Action, *, threshold: int = 40) -> dict:
    """do(A) vs do(B) from the same state. Returns the outcome deltas."""
    pa = transition.predict(pop, action_a)
    pb = transition.predict(pop, action_b)
    return {
        "report_type": "prediction",
        "a": {"action_id": action_a.action_id, "p_hit": round(pa["thresholds"].get(threshold, 0), 4),
              "expected_magnitude": round(expected_magnitude(pa["band_probs"]), 2)},
        "b": {"action_id": action_b.action_id, "p_hit": round(pb["thresholds"].get(threshold, 0), 4),
              "expected_magnitude": round(expected_magnitude(pb["band_probs"]), 2)},
        "delta_p_hit": round(pa["thresholds"].get(threshold, 0) - pb["thresholds"].get(threshold, 0), 4),
        "delta_expected_magnitude": round(
            expected_magnitude(pa["band_probs"]) - expected_magnitude(pb["band_probs"]), 2),
        "prefer": action_a.action_id if pa["thresholds"].get(threshold, 0)
        >= pb["thresholds"].get(threshold, 0) else action_b.action_id,
    }


def best_of(transition: AggregateTransition, pop: PopulationState, candidates: list[Action], *,
            threshold: int = 40) -> dict:
    """Pick the action maximizing P(hit) from the current state (the compare-actions decision)."""
    scored = [(a, transition.predict(pop, a)["thresholds"].get(threshold, 0.0)) for a in candidates]
    scored.sort(key=lambda t: t[1], reverse=True)
    return {"recommended_action_id": scored[0][0].action_id,
            "p_hit": round(scored[0][1], 4),
            "lift_over_worst": round(scored[0][1] - scored[-1][1], 4) if len(scored) > 1 else 0.0}
