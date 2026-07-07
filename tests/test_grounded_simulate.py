"""Tests for the IndependentPopulationReadout (EXP-050) — the DEMOTED non-interacting mechanism leaf.
Not a simulation; a calibrated bottom-up compositor. Also guards the deprecated GroundedSimulator aliases."""
from swm.api.grounded_simulate import (GroundedForecast, GroundedSimulator, IndependentPopulationReadout)


def _rows(n=120):
    out = []
    for i in range(n):
        rep = i % 2 == 0
        out.append({"qid": "cappun", "answer_idx": 1 if rep else 0,
                    "demo": {"party": "republican" if rep else "democrat",
                             "ideology": "conservative" if rep else "liberal"}})
    return out


def _readout():
    return IndependentPopulationReadout(attrs=["party", "ideology"]).fit(_rows(), k=2)


def test_predict_person_returns_prob_and_profile():
    out = _readout().predict_person("cappun", {"party": "republican", "ideology": "conservative"})
    assert 0.0 <= out["p_answer"] <= 1.0
    assert out["value_profile"]                                   # non-empty value-factor profile


def test_predict_share_aggregates():
    pop = [{"party": "republican", "ideology": "conservative"}] * 30 + \
          [{"party": "democrat", "ideology": "liberal"}] * 30
    fc = _readout().predict_share("cappun", pop)
    assert isinstance(fc, GroundedForecast)
    assert 0.0 <= fc.p_outcome <= 1.0
    assert fc.n == 60
    assert 0.0 <= fc.confidence <= 1.0
    assert len(fc.value_drivers) >= 1                            # reports which value axes drove it


def test_share_tracks_composition():
    r = _readout()
    mostly_rep = [{"party": "republican", "ideology": "conservative"}] * 50 + \
                 [{"party": "democrat", "ideology": "liberal"}] * 10
    mostly_dem = [{"party": "republican", "ideology": "conservative"}] * 10 + \
                 [{"party": "democrat", "ideology": "liberal"}] * 50
    assert r.predict_share("cappun", mostly_rep).p_outcome > r.predict_share("cappun", mostly_dem).p_outcome


def test_forecast_as_dict():
    fc = _readout().predict_share("cappun", [{"party": "democrat", "ideology": "liberal"}] * 25)
    d = fc.as_dict()
    assert set(d) >= {"p_outcome", "n", "confidence", "value_drivers"}


def test_deprecated_aliases_still_work():
    """The old GroundedSimulator / simulate_population / simulate_person names remain as back-compat aliases."""
    sim = GroundedSimulator(attrs=["party", "ideology"]).fit(_rows(), k=2)   # alias for IndependentPopulationReadout
    assert isinstance(sim, IndependentPopulationReadout)
    assert 0.0 <= sim.simulate_person("cappun", {"party": "republican", "ideology": "conservative"})["p_answer"] <= 1.0
    fc = sim.simulate_population("cappun", [{"party": "republican", "ideology": "conservative"}] * 10)
    assert isinstance(fc, GroundedForecast) and fc.n == 10
