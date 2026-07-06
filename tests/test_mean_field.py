"""Tests for the mean-field coupling loop (EXP-053) — non-separable aggregation."""
from swm.simulation.mean_field import Agent, MeanFieldRollout, agents_from_cells


def test_aggregate_is_influence_weighted():
    agents = [Agent(belief=0.0, influence=1.0), Agent(belief=1.0, influence=3.0)]
    mf = MeanFieldRollout()
    assert abs(mf.aggregate(agents) - 0.75) < 1e-9        # (0*1 + 1*3)/4


def test_conformity_pulls_toward_aggregate():
    agents = [Agent(belief=0.1, responsiveness=0.5), Agent(belief=0.9, responsiveness=0.5)]
    mf = MeanFieldRollout(k_social=0.4, k_proof=0.0)
    b0 = [a.belief for a in agents]
    mf.step(agents)
    # both move toward the aggregate (0.5): the low one up, the high one down
    assert agents[0].belief > b0[0] and agents[1].belief < b0[1]


def test_coupling_is_non_separable():
    # agent i's trajectory depends on agent j's belief (via the aggregate) — the whole point
    a_alone = [Agent(belief=0.2, responsiveness=0.5)]
    MeanFieldRollout(k_social=0.4).roll(a_alone, 3)
    solo = a_alone[0].belief
    a_with = [Agent(belief=0.2, responsiveness=0.5), Agent(belief=0.95, responsiveness=0.5)]
    MeanFieldRollout(k_social=0.4).roll(a_with, 3)
    assert abs(a_with[0].belief - solo) > 1e-3            # the second agent changed the first's outcome


def test_social_proof_amplifies_majority():
    # with a majority already holding the view, social proof pushes the aggregate further up
    agents = [Agent(belief=0.7, responsiveness=0.5) for _ in range(10)]
    _, final = MeanFieldRollout(k_social=0.0, k_proof=0.6).roll(agents, 5)
    assert final > 0.7                                    # prevalence begets adoption (nonlinear)


def test_roll_returns_trajectory():
    agents = [Agent(belief=0.3, responsiveness=0.4) for _ in range(5)]
    traj, final = MeanFieldRollout(k_proof=0.5).roll(agents, 4)
    assert len(traj) == 5 and traj[-1] == final


def test_agents_from_cells():
    agents = agents_from_cells([(0.3, 0.5, 2.0), (0.8, 0.2, 1.0)])
    assert len(agents) == 2 and agents[0].influence == 2.0
