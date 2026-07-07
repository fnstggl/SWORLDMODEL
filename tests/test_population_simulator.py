"""Tests for Level-3 population simulator + KPI + the Level-2 demographic backdrop (EXP-061/062)."""
from swm.eval.population_metrics import (coupling_skill, interval_coverage, population_scorecard,
                                         share_rmse, winner_accuracy)
from swm.simulation.agent_society import AgentSociety, PersonaAgent
from swm.simulation.mean_field import MeanFieldRollout
from swm.simulation.population_simulator import (DemographicCell, PopulationSimulator, cells_from_rows,
                                                 marginal_share, share_aggregator,
                                                 winner_take_all_aggregator)


# ---- KPI ----
def test_coupling_skill_sign():
    truth = [0.6, 0.4, 0.7]
    marginal = [0.5, 0.5, 0.5]
    good = [0.58, 0.42, 0.68]                     # closer to truth -> positive skill
    bad = [0.5, 0.5, 0.5]                          # same as marginal -> ~0 skill
    assert coupling_skill(truth, marginal, good)["skill"] > 0.3
    assert abs(coupling_skill(truth, marginal, bad)["skill"]) < 1e-9


def test_interval_coverage_and_scorecard():
    truth = [0.5, 0.6, 0.4, 0.7]
    lo = [0.4, 0.5, 0.3, 0.6]
    hi = [0.6, 0.7, 0.5, 0.8]
    cov = interval_coverage(truth, lo, hi, nominal=0.8)
    assert cov["empirical_coverage"] == 1.0 and cov["mean_width"] > 0
    card = population_scorecard(truth, [0.5] * 4, [0.52, 0.58, 0.42, 0.68], lo=lo, hi=hi)
    assert "coupling_skill" in card and "interval_coverage" in card and "headline" in card


def test_winner_accuracy():
    assert winner_accuracy([0.6, 0.4], [0.55, 0.45]) == 1.0
    assert winner_accuracy([0.6, 0.4], [0.45, 0.55]) == 0.0


# ---- aggregators ----
def test_share_and_marginal_aggregators():
    cells = [DemographicCell("a", weight=2, stance=1.0, turnout=1.0),
             DemographicCell("b", weight=2, stance=0.0, turnout=0.5)]
    assert abs(marginal_share(cells) - 0.5) < 1e-9            # size-weighted, ignores turnout
    # participation-weighted tilts toward the higher-turnout cell (the 1.0-stance one)
    assert share_aggregator(cells) > 0.5


def test_winner_take_all_rolls_up_regions():
    cells = [DemographicCell("a", 1, 0.9, region="north"), DemographicCell("b", 1, 0.8, region="north"),
             DemographicCell("c", 1, 0.2, region="south")]
    out = winner_take_all_aggregator(cells)
    assert out["by_region"]["north"] > 0.5 and out["by_region"]["south"] < 0.5
    assert 0.0 <= out["region_share_won"] <= 1.0


# ---- simulator: marginal vs coupled, and participation coupling ----
def test_simulate_returns_marginal_and_coupled():
    cells = [DemographicCell(f"c{i}", weight=1, stance=s) for i, s in enumerate([0.3, 0.5, 0.7, 0.9])]
    sim = PopulationSimulator(rollout=MeanFieldRollout(k_social=0.15, k_proof=0.0), aggregator=marginal_share)
    out = sim.simulate(cells, steps=5)
    assert "marginal" in out and "coupled" in out and len(out["trajectory"]) == 5
    # pure conformity is (approximately) mean-preserving on a size-weighted aggregate
    assert abs(out["coupled"] - out["marginal"]) < 0.05


def test_turnout_coupling_moves_the_outcome():
    # a majority-favoring bloc mobilizes under participation coupling -> outcome diverges from marginal
    cells = [DemographicCell("maj", weight=7, stance=0.8, turnout=0.5),
             DemographicCell("min", weight=3, stance=0.3, turnout=0.5)]
    static = PopulationSimulator(rollout=MeanFieldRollout(k_social=0.1), aggregator=share_aggregator,
                                 turnout_coupling=0.0).simulate(cells, steps=8)
    mobilized = PopulationSimulator(rollout=MeanFieldRollout(k_social=0.1), aggregator=share_aggregator,
                                    turnout_coupling=0.8).simulate(cells, steps=8)
    assert mobilized["coupled"] != static["coupled"]          # mobilization changed who showed up


def test_cells_from_rows():
    rows = [{"g": "x", "y": 1}, {"g": "x", "y": 0}, {"g": "y", "y": 1}, {"g": "y", "y": 1}]
    cells = {c.cell_id: c for c in cells_from_rows(rows, lambda r: r["g"], lambda r: r["y"])}
    assert abs(cells["x"].stance - 0.5) < 1e-9 and abs(cells["y"].stance - 1.0) < 1e-9


# ---- Level-2 demographic backdrop ----
def test_public_backdrop_pulls_stakeholders_and_is_backward_compatible():
    def pf(a, p):
        return a.variables["ideo"]
    agents_no = [PersonaAgent(f"j{i}", {"ideo": 0.8}, position=0.8, openness=0.4, conviction=0.3)
                 for i in range(5)]
    # no backdrop -> conservative body stays high
    soc0 = AgentSociety(rounds=6)
    out0 = soc0.simulate(None, agents_no, pf)
    # a liberal public backdrop with real sensitivity pulls the body down
    agents_bd = [PersonaAgent(f"j{i}", {"ideo": 0.8}, position=0.8, openness=0.4, conviction=0.3,
                              public_sensitivity=0.5) for i in range(5)]
    soc1 = AgentSociety(rounds=6, public_field=0.2)
    out1 = soc1.simulate(None, agents_bd, pf)
    assert out1["p_outcome"] < out0["p_outcome"]              # the backdrop moved the stakeholders
    # backward compat: public_sensitivity default 0 => backdrop has no effect even if field is set
    agents_ins = [PersonaAgent(f"j{i}", {"ideo": 0.8}, position=0.8, openness=0.4, conviction=0.3)
                  for i in range(5)]
    out2 = AgentSociety(rounds=6, public_field=0.2).simulate(None, agents_ins, pf)
    assert abs(out2["p_outcome"] - out0["p_outcome"]) < 1e-9