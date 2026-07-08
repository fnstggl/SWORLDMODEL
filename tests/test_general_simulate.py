"""Tests for the GeneralSimulator front door (routes + fuses any question -> outcome)."""
from swm.api.general_simulate import GeneralForecast, GeneralSimulator
from swm.api.grounded_simulate import GroundedSimulator


def _grounded():
    rows = []
    for i in range(120):
        rep = i % 2 == 0
        rows.append({"qid": "cappun", "answer_idx": 1 if rep else 0,
                     "demo": {"party": "republican" if rep else "democrat",
                              "ideology": "conservative" if rep else "liberal"}})
    return GroundedSimulator(attrs=["party", "ideology"]).fit(rows, k=2)


class _Judge:                                            # a stub SemanticStanceJudge
    def __init__(self, stance, conf=0.7):
        self.s = {"stance": stance, "confidence": conf, "relevant": 3}

    def stance(self, q, news, hint=""):
        return self.s


def _driver_fn(q, ctx):                                  # a stub driver-inference fn
    from swm.api.question_engine import Driver
    return 0.5, [Driver("d", direction=1.0, strength=0.8, confidence=0.8)]


def test_prior_only_when_nothing_fires():
    fc = GeneralSimulator().answer("Will X?", base_rate=0.4)
    assert fc.method == "prior_only" and fc.p_outcome == 0.4


def test_routes_to_population_readout():
    sim = GeneralSimulator(grounded=_grounded())
    pop = [{"party": "republican", "ideology": "conservative"}] * 40
    fc = sim.answer("death penalty?", known_item="cappun", population=pop)
    assert "population_readout" in fc.sources               # the demoted non-interacting leaf (not "simulation")
    assert fc.p_outcome > 0.5                            # conservative population favors it


def test_routes_to_news_stance():
    sim = GeneralSimulator(stance_judge=_Judge(stance=0.8))
    fc = sim.answer("Will the team win?", news=[{"title": "team wins"}], base_rate=0.5)
    assert "news_stance" in fc.sources and fc.p_outcome > 0.5


def test_routes_to_driver_engine():
    fc = GeneralSimulator().answer("Will it pass?", driver_infer_fn=_driver_fn)
    assert "driver_engine" in fc.sources and fc.p_outcome > 0.5


def test_fuses_multiple_sources():
    sim = GeneralSimulator(grounded=_grounded(), stance_judge=_Judge(stance=0.6))
    pop = [{"party": "republican", "ideology": "conservative"}] * 30
    fc = sim.answer("q", known_item="cappun", population=pop, news=[{"title": "x"}],
                    driver_infer_fn=_driver_fn)
    assert len(fc.sources) == 3                          # all three fired and fused
    assert "+" in fc.method
    assert isinstance(fc.as_dict(), dict)
