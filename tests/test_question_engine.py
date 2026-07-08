"""Tests for the question->driver->inferred-lean engine (Tetlock-structured aggregation)."""
from swm.api.question_engine import Driver, QuestionEngine, drivers_from_payload


def test_base_rate_only_returns_base_rate():
    eng = QuestionEngine()
    assert abs(eng.aggregate(0.3, []) - 0.3) < 1e-6      # no drivers -> the prior is unchanged


def test_yes_drivers_raise_and_no_drivers_lower():
    eng = QuestionEngine()
    up = eng.aggregate(0.5, [Driver("d", direction=1.0, strength=0.8, confidence=0.9)])
    down = eng.aggregate(0.5, [Driver("d", direction=-1.0, strength=0.8, confidence=0.9)])
    assert up > 0.5 > down
    assert abs((up - 0.5) - (0.5 - down)) < 1e-6         # symmetric


def test_strength_and_confidence_scale_the_shift():
    eng = QuestionEngine()
    weak = eng.aggregate(0.5, [Driver("d", 1.0, 0.2, 0.3)])
    strong = eng.aggregate(0.5, [Driver("d", 1.0, 0.9, 0.9)])
    assert strong > weak > 0.5


def test_evidence_shrink_dampens_overreaction():
    d = [Driver("a", 1.0, 1.0, 1.0), Driver("b", 1.0, 1.0, 1.0)]
    hot = QuestionEngine(evidence_shrink=1.0).aggregate(0.5, d)
    cool = QuestionEngine(evidence_shrink=0.4).aggregate(0.5, d)
    assert hot > cool > 0.5                              # less shrink -> more extreme


def test_dragonfly_median_and_forecast_object():
    eng = QuestionEngine()
    views = [(0.5, [Driver("a", 1.0, 0.8, 0.9)]),        # leans YES
             (0.5, [Driver("b", -1.0, 0.8, 0.9)]),       # leans NO
             (0.5, [Driver("c", 1.0, 0.5, 0.9)])]        # leans YES mildly
    f = eng.forecast_from_views(views)
    assert f.n_views == 3
    assert f.direction in (-1, 0, 1)
    assert 0.0 <= f.p_outcome <= 1.0
    assert f.drivers                                     # keeps a driver breakdown for the audit trail


def test_drivers_from_payload_tolerant():
    d = drivers_from_payload([{"name": "x", "direction": 0.5, "strength": 0.6, "confidence": 0.7},
                              {"bad": "row"}])            # missing fields -> defaults, no crash
    assert len(d) == 2 and d[0].name == "x"
