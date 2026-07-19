"""Pins the grounded, continuous, recurrence-aware outcome-rate prior (replaces the coarse 5-value lean as
the DEFAULT when a base rate can be grounded; lean kept only as weak fallback)."""
from types import SimpleNamespace

from swm.world_model_v2.phase3_priors import (build_outcome_rate_prior, estimate_reference_base_rate,
                                              grounded_estimate_prior)


def _plan(question="Will X happen?", lean="weak_no", as_of="2026-05-07"):
    return SimpleNamespace(question=question, provenance={"outcome_lean": lean, "as_of": as_of})


def test_grounded_estimate_prior_is_continuous_and_bounded():
    # a recurrence with base rate 0.95 grounds the prior mean HIGH (not the 0.70 lean ceiling)...
    spec = grounded_estimate_prior("annual OS at conference", 0.95, transport_risk="low",
                                   n_examples=8, is_recurrence=True)
    assert 0.80 < spec.mean < 0.98 and spec.source_class == "recurrence"
    # ...but stays weakly-informative: effective sample size capped low (never a data-backed certainty)
    assert spec.retained_effective_n <= 10.0
    # continuous: a 0.62 estimate lands off the coarse {.30,.41,.50,.59,.70} lean grid
    assert abs(grounded_estimate_prior("c", 0.62, n_examples=6).mean - 0.62) < 0.12


def test_estimate_parses_and_bounds(monkeypatch):
    def fake(_p):
        return ('{"reference_class":"annual visionOS at WWDC","is_recurrence":true,"base_rate":0.95,'
                '"n_examples":5,"transport_risk":"low","why":"ships every June"}')
    est = estimate_reference_base_rate("Will Apple announce visionOS 27 at WWDC 2026?", llm=fake)
    assert est["is_recurrence"] is True and est["base_rate"] == 0.95 and est["transport_risk"] == "low"


def test_visionos_prior_grounds_high_not_weak_no():
    # the exact failure: old path -> weak_no -> 0.41. Grounded path -> ~0.9. Uses a stub outside-view LLM.
    def fake(_p):
        return ('{"reference_class":"annual visionOS release at WWDC","is_recurrence":true,'
                '"base_rate":0.95,"n_examples":5,"transport_risk":"low","why":"annual cycle"}')
    spec = build_outcome_rate_prior(_plan("Will Apple announce visionOS 27 at WWDC 2026?", lean="weak_no"),
                                    llm=fake)
    assert spec.mean > 0.75 and spec.source_class == "recurrence"


def test_precedence_real_data_beats_estimate():
    # held-out data (successes/total) must win over an LLM estimate
    def fake(_p):
        return '{"reference_class":"x","is_recurrence":false,"base_rate":0.9,"n_examples":5,"transport_risk":"low"}'
    spec = build_outcome_rate_prior(_plan(), llm=fake,
                                    reference_data={"reference_class": "d", "successes": 2, "total": 40,
                                                    "transport_risk": "low"})
    assert spec.source_class == "reference_class" and spec.mean < 0.3


def test_falls_back_to_lean_when_no_grounding():
    spec = build_outcome_rate_prior(_plan(lean="weak_no"), llm=lambda p: "{}")   # estimator returns {}
    assert spec.source_class == "generic_weakly_informative"
    assert abs(spec.mean - 0.41) < 0.02                       # weak_no lean, the weak fallback
