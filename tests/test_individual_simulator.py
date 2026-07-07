"""Tests for the Level-1 individual simulator (EXP-060): agent state, response model, front door."""
from swm.api.individual_simulate import IndividualSimulator
from swm.simulation.individual_agent import STATE_BASELINE, IndividualAgent
from swm.simulation.response_model import (StructuredResponseModel, llm_response_fn, quantities)
from swm.variables.variable_map import VariableMap


def _fit_model(**kw):
    # tiny synthetic set where open people + high-quality messages persuade (person x message interaction)
    rows = []
    for op in (0.1, 0.9):
        for q in (0.1, 0.9):
            y = 1 if (op > 0.5 and q > 0.5) else 0
            for _ in range(20):
                rows.append(({"trait_openness": op}, {}, {"clarity": q, "ask_directness": q}, y))
    return StructuredResponseModel(**kw).fit(rows)


def test_agent_state_initializes_to_baseline():
    a = IndividualAgent("p", VariableMap(entity_id="p"))
    for k, base in STATE_BASELINE.items():
        assert abs(a.state[k] - base) < 1e-9


def test_apply_dynamics_move_state():
    a = IndividualAgent("p", VariableMap(entity_id="p"))
    a.apply({"pushiness": 0.9, "politeness_disposition": 0.1, "effort_cost": 0.8}, responded=True)
    assert a.state["mood_valence"] < 0.0                 # a pushy, rude ask sours mood
    assert a.state["attention_availability"] < 0.6       # responding + a high-effort ask spends attention
    assert a.state["recency_of_contact"] == 1.0          # just contacted
    assert a.state["reciprocity_debt"] < 0.25            # responding discharged the debt


def test_relax_returns_toward_baseline():
    a = IndividualAgent("p", VariableMap(entity_id="p"))
    a.state["mood_valence"] = -0.8
    a.relax(steps=3)
    assert -0.8 < a.state["mood_valence"] <= 0.0         # decays back toward neutral


def test_interaction_beats_message_only_on_synthetic():
    full = _fit_model(features=("receptivity", "quality", "interaction"))
    msg = _fit_model(features=("quality",))
    # an open person + strong message should score high on the interaction model, and the two models
    # should DISAGREE for a closed person + strong message (msg-only can't see the person)
    open_strong = full({"trait_openness": 0.9}, {}, {"clarity": 0.9, "ask_directness": 0.9})["p"]
    closed_strong_full = full({"trait_openness": 0.1}, {}, {"clarity": 0.9, "ask_directness": 0.9})["p"]
    closed_strong_msg = msg({"trait_openness": 0.1}, {}, {"clarity": 0.9, "ask_directness": 0.9})["p"]
    assert open_strong > closed_strong_full              # the person matters
    assert closed_strong_full < closed_strong_msg        # interaction model discounts the closed person


def test_state_gate_is_zero_at_resting_state():
    gated = _fit_model(features=("receptivity", "quality", "interaction"), state_gate=True, gate_strength=3.0)
    plain = StructuredResponseModel(features=("receptivity", "quality", "interaction"))
    plain.readout, plain.base_rate = gated.readout, gated.base_rate
    a = IndividualAgent("p", VariableMap(entity_id="p"))
    msg = {"clarity": 0.7, "ask_directness": 0.6}
    # at the baseline state, friction == baseline, so the gate contributes nothing
    assert abs(gated(a.variables, dict(a.state), msg)["p"] - plain(a.variables, dict(a.state), msg)["p"]) < 1e-6


def test_predict_response_and_best_message():
    model = _fit_model(features=("receptivity", "quality", "interaction"))
    sim = IndividualSimulator(response_fn=model)
    person = {"trait_openness": 0.9}
    out = sim.predict_response(person, {"clarity": 0.9, "ask_directness": 0.9})
    assert 0.0 <= out["p_respond"] <= 1.0 and "drivers" in out
    pick = sim.best_message(person, [{"clarity": 0.1, "ask_directness": 0.1},
                                     {"clarity": 0.9, "ask_directness": 0.9}])
    assert pick["best"]["index"] == 1 and pick["lift_over_mean"] >= 0   # the stronger message wins


def test_simulate_thread_carries_state():
    model = _fit_model(features=("receptivity", "quality", "interaction"), state_gate=True, gate_strength=2.5)
    sim = IndividualSimulator(response_fn=model)
    person = {"trait_openness": 0.5}
    ask = {"clarity": 0.7, "ask_directness": 0.7, "effort_cost": 0.4}
    pushy = sim.simulate_thread(person, [{"pushiness": 0.9, "politeness_disposition": 0.1}, ask], gap_steps=0)
    kind = sim.simulate_thread(person, [{"pushiness": 0.1, "politeness_disposition": 0.9,
                                         "personalization": 0.8}, ask], gap_steps=0)
    # the identical closing ask lands better after a kind opener than a pushy one (state carried over)
    assert kind["turns"][1]["p_respond"] > pushy["turns"][1]["p_respond"]
    assert pushy["turns"][1]["state_before"]["mood_valence"] < kind["turns"][1]["state_before"]["mood_valence"]


def test_llm_response_fn_backend():
    # a stub LLM judge that reasons "as the person" and returns JSON
    def judge(prompt):
        return {"p": 0.42, "reason": "busy but relevant"}
    fn = llm_response_fn(judge)
    out = fn({"trait_openness": 0.6}, {"attention_availability": 0.3}, {"clarity": 0.8})
    assert out["p"] == 0.42 and "quantities" in out and "reason" in out["drivers"]


def test_quantities_are_general_over_missing_vars():
    # missing variables default to neutral, so the model is general across question types
    q = quantities({"trait_openness": 0.8}, {}, {"clarity": 0.9})
    assert 0.0 <= q["receptivity"] <= 1.0 and 0.0 <= q["quality"] <= 1.0 and 0.0 <= q["friction"] <= 1.0
