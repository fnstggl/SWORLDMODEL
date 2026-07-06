"""Tests for the agent-based society simulation (EXP-055) — emergent, non-separable outcomes."""
from swm.simulation.agent_society import AgentSociety, PersonaAgent, independent_outcome


def _pf(a, p):
    return a.position


def test_influential_minority_can_flip_the_vote():
    agents = ([PersonaAgent(f"w{i}", {"v": 0.5}, position=0.45, influence=1.0, openness=0.6, conviction=0.1)
               for i in range(6)]
              + [PersonaAgent(f"L{i}", {"v": 0.5}, position=0.97, influence=5.0, openness=0.1, conviction=0.9)
                 for i in range(3)])
    indep = independent_outcome(agents, _pf, None)
    soc = AgentSociety(rounds=8).simulate(None, agents, _pf)
    assert indep["passes"] is False                      # independent count: the 6 win
    assert soc["passes"] is True                         # deliberation: the influential 3 pull it over


def test_consensus_pull_converges_a_split_body():
    split = [PersonaAgent(f"lo{i}", {"v": 0.1}, position=0.2, openness=0.5) for i in range(5)] + \
            [PersonaAgent(f"hi{i}", {"v": 0.9}, position=0.8, openness=0.5) for i in range(5)]
    out = AgentSociety(consensus_pull=0.7, rounds=10).simulate(None, split, _pf)
    spread = max(out["final_positions"]) - min(out["final_positions"])
    assert spread < 0.2                                  # deliberation drives consensus


def test_bounded_confidence_preserves_polarization():
    blocs = [PersonaAgent(f"L{i}", {"v": 0.1}, position=0.35, openness=0.6) for i in range(5)] + \
            [PersonaAgent(f"R{i}", {"v": 0.9}, position=0.65, openness=0.6) for i in range(5)]
    out = AgentSociety(homophily=1.0, confidence_bound=0.5, rounds=10).simulate(None, blocs, _pf)
    fp = out["final_positions"]
    gap = sum(fp[5:]) / 5 - sum(fp[:5]) / 5
    assert gap > 0.2                                     # the blocs stay apart (echo chambers)


def test_non_separability():
    # an agent's outcome depends on another agent's state -> not a mean of independents
    solo = [PersonaAgent("a", {"v": 0.5}, position=0.3, openness=0.5)]
    AgentSociety(consensus_pull=0.5, rounds=5).simulate(None, solo, _pf)
    p_solo = solo[0].position
    pair = [PersonaAgent("a", {"v": 0.5}, position=0.3, openness=0.5),
            PersonaAgent("b", {"v": 0.5}, position=0.95, openness=0.5)]
    AgentSociety(consensus_pull=0.5, rounds=5).simulate(None, pair, _pf)
    assert abs(pair[0].position - p_solo) > 1e-3


def test_independent_baseline_is_a_mean():
    agents = [PersonaAgent(f"a{i}", {"v": 0.5}, position=0.2 * i) for i in range(5)]
    out = independent_outcome(agents, _pf, None)
    assert abs(out["p_outcome"] - sum(0.2 * i for i in range(5)) / 5) < 1e-9
