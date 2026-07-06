"""Tests for the assembled end-to-end GroundedSimulator (EXP-050)."""
from swm.api.grounded_simulate import GroundedForecast, GroundedSimulator


def _rows(n=120):
    out = []
    for i in range(n):
        rep = i % 2 == 0
        out.append({"qid": "cappun", "answer_idx": 1 if rep else 0,
                    "demo": {"party": "republican" if rep else "democrat",
                             "ideology": "conservative" if rep else "liberal"}})
    return out


def _sim():
    return GroundedSimulator(attrs=["party", "ideology"]).fit(_rows(), k=2)


def test_simulate_person_returns_prob_and_profile():
    out = _sim().simulate_person("cappun", {"party": "republican", "ideology": "conservative"})
    assert 0.0 <= out["p_answer"] <= 1.0
    assert out["value_profile"]                                   # non-empty value-factor profile


def test_simulate_population_aggregates():
    pop = [{"party": "republican", "ideology": "conservative"}] * 30 + \
          [{"party": "democrat", "ideology": "liberal"}] * 30
    fc = _sim().simulate_population("cappun", pop)
    assert isinstance(fc, GroundedForecast)
    assert 0.0 <= fc.p_outcome <= 1.0
    assert fc.n == 60
    assert 0.0 <= fc.confidence <= 1.0
    assert len(fc.value_drivers) >= 1                            # reports which value axes drove it


def test_population_share_tracks_composition():
    sim = _sim()
    mostly_rep = [{"party": "republican", "ideology": "conservative"}] * 50 + \
                 [{"party": "democrat", "ideology": "liberal"}] * 10
    mostly_dem = [{"party": "republican", "ideology": "conservative"}] * 10 + \
                 [{"party": "democrat", "ideology": "liberal"}] * 50
    # cappun "1"=favor death penalty (conservative); a more-conservative population -> higher share
    assert sim.simulate_population("cappun", mostly_rep).p_outcome > \
           sim.simulate_population("cappun", mostly_dem).p_outcome


def test_forecast_as_dict():
    fc = _sim().simulate_population("cappun", [{"party": "democrat", "ideology": "liberal"}] * 25)
    d = fc.as_dict()
    assert set(d) >= {"p_outcome", "n", "confidence", "value_drivers"}
