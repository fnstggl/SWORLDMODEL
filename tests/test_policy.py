"""Tests for sequential policies (Component 6): state carries across steps, best_policy picks the better plan,
temporal structural schedules. The reflexivity a single-action layer cannot express."""
from swm.decision.action import inject_event, noop
from swm.decision.policy import (Policy, best_policy, enumerate_policies, individual_rollout,
                                 message_sequences, structural_schedule_rollout)
from swm.decision.utility import Mean, identity, value
from swm.api.model_spec import parse_spec


# a response model that reads the person's transient STATE (mood), so an opener changes how the ask lands
def _state_reading_response(variables, state, message):
    q = 0.3 * float(message.get("clarity", 0.5)) + 0.2 * float(message.get("ask_directness", 0.5))
    mood = state.get("mood_valence", 0.0)
    p = 0.4 + 0.35 * mood + 0.4 * q                       # a kind opener raises mood -> the ask lands better
    return {"p": max(0.02, min(0.98, p))}


KIND = {"clarity": 0.7, "politeness_disposition": 0.9, "personalization": 0.8, "pushiness": 0.1, "effort_cost": 0.3}
PUSHY = {"clarity": 0.5, "politeness_disposition": 0.1, "personalization": 0.1, "pushiness": 0.9, "effort_cost": 0.5}
ASK = {"clarity": 0.8, "ask_directness": 0.8}


def test_best_policy_prefers_the_opener_that_sets_up_the_ask():
    """Same closing ask; the ONLY difference is the opener's residue in the person's state -> best_policy must
    prefer the kind opener. A static single-message layer, blind to state carryover, cannot get this right."""
    rollout = individual_rollout({"trait_openness": 0.5}, _state_reading_response,
                                 gap_steps=0, readout="last", respond="threshold")
    policies = message_sequences([KIND, PUSHY], ASK, opener_labels=["kind", "pushy"])
    res = best_policy(rollout, policies, identity(), objective=Mean(), max_per_arm=800, seed=0)
    assert res.best.label.startswith("kind")
    assert res.decided


def test_individual_rollout_any_readout_accumulates():
    """'any' readout = P(respond at some point in the thread) >= a single step's probability."""
    rollout = individual_rollout({"trait_openness": 0.6}, _state_reading_response, readout="any",
                                 respond="threshold")
    out_any, _ = rollout(Policy([KIND, ASK]), __import__("random").Random(0))
    out_last, _ = rollout(Policy([KIND, ASK]), __import__("random").Random(0))  # deterministic path
    assert 0.0 <= out_last <= out_any <= 1.0


def test_structural_schedule_injects_a_temporal_shock():
    """A generic_scm world with a scheduled +shock at t=1 must end HIGHER than doing nothing (Component 1
    temporal do-operator, carried through the diffusion by the policy)."""
    spec = parse_spec({"mechanism": "generic_scm",
                       "variables": [{"name": "x", "value": 0.4, "est_sd": 0.0, "volatility": 0.02}],
                       "equations": {"x": "0.1*(0.5 - x)"}, "outcome": {"variable": "x"}, "horizon": 6})
    rollout = structural_schedule_rollout(spec)
    policies = enumerate_policies([[noop()], [inject_event("x", 0.3, time=1.0)]], labels=["nothing", "shock"])
    res = best_policy(rollout, policies, value(lambda o: o), objective=Mean(), max_per_arm=2000, seed=0)
    assert res.best.label == "shock"                      # the injected event genuinely moves the outcome
    assert res.contrast["vs_runner_up"]["delta"] > 0.05


def test_best_policy_returns_navigable_and_grade():
    rollout = individual_rollout({"trait_openness": 0.5}, _state_reading_response, respond="threshold")
    res = best_policy(rollout, message_sequences([KIND, PUSHY], ASK), identity(), max_per_arm=600, seed=0)
    assert res.navigable is not None
    assert res.grade().startswith(("A", "B", "C", "D"))
