"""Tests for the assembled generative loop (EXP-057)."""
from swm.api.generative_simulator import (AgentSpec, GenerativeSimulator, build_identify_prompt,
                                          build_position_prompt, parse_agents)
from swm.simulation.agent_society import AgentSociety


def test_parse_agents_from_payload():
    specs = parse_agents({"agents": [{"id": "j1", "variables": {"ideo": 0.8}, "influence": 2.0},
                                     {"id": "j2", "variables": {"ideo": 0.2}}]})
    assert len(specs) == 2 and specs[0].agent_id == "j1"
    assert specs[0].variables["ideo"] == 0.8 and specs[0].influence == 2.0


def test_identify_and_position_prompts():
    p = build_identify_prompt("Will the committee approve?", context="9 members")
    assert "AGENTS" in p and "JSON" in p and "9 members" in p
    pp = build_position_prompt("Will it pass?", AgentSpec("a", {"ideo": 0.7}))
    assert "position" in pp and "0.7" in pp and "hindsight" in pp


def test_simulate_end_to_end_with_structured_position():
    specs = [AgentSpec(f"a{i}", {"ideo": 0.2 + 0.1 * i}, influence=1.0) for i in range(5)]
    sim = GenerativeSimulator(society=AgentSociety(rounds=4),
                              position_fn=lambda q, s, ctx: s.variables["ideo"])
    fc = sim.simulate("Q?", agents=specs)
    assert fc.n_agents == 5
    assert 0.0 <= fc.p_outcome <= 1.0
    assert len(fc.agents) == 5 and "initial" in fc.agents[0] and "final" in fc.agents[0]
    assert fc.independent_p is not None                  # the composite baseline is reported for contrast


def test_influential_bloc_flips_outcome_via_deliberation():
    # naive count: 6 weak-NO (0.42) beat 3 strong-YES; deliberation: influential YES bloc flips it
    specs = [AgentSpec(f"w{i}", {"v": 0.5}, influence=1.0, openness=0.7, conviction=0.1) for i in range(6)] + \
            [AgentSpec(f"L{i}", {"v": 0.5}, influence=4.0, openness=0.1, conviction=0.9) for i in range(3)]
    pos = {**{f"w{i}": 0.42 for i in range(6)}, **{f"L{i}": 0.95 for i in range(3)}}
    sim = GenerativeSimulator(society=AgentSociety(rounds=8),
                              position_fn=lambda q, s, ctx: pos[s.agent_id])
    fc = sim.simulate("pass?", agents=specs)
    assert fc.independent_passes is False                # composite vote count says NO (only 3 of 9)
    assert fc.passes is True                             # the emergent, deliberated outcome says YES


def test_identify_fn_is_called_when_no_agents_given():
    called = {}
    def identify(q, ctx):
        called["yes"] = True
        return [AgentSpec("a", {"v": 0.6})]
    sim = GenerativeSimulator(identify_fn=identify, position_fn=lambda q, s, ctx: s.variables["v"])
    sim.simulate("Q?")
    assert called.get("yes")
