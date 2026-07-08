"""Tests for the grounding COVERAGE layer — structured sources, the router (state + rate), and the calibrated
retrieval grounder (offline, mock backends)."""
from swm.api.grounding_sources import (GroundingRouter, StructuredSource, default_sources, macro_source)
from swm.api.retrieval_grounding import (CalibratedExtractor, build_retrieval_grounder, calibrate_extractor)


def test_structured_source_matches_content_not_generic_words():
    src = macro_source(fetch=lambda k, a: {"inflation": (3.2, 0.1), "unemployment": (4.1, 0.1)}.get(k))
    assert src.match("inflation rate")[0] == "inflation"          # distinctive token -> match
    assert src.match("unemployment rate")[0] == "unemployment"
    assert src.match("crime rate") is None                        # shares only the generic word "rate" -> no match
    assert src.match("current case rate") is None                 # not a macro series -> no spurious match
    gv = src.ground("inflation rate")
    assert gv is not None and gv.value == 3.2 and gv.source == "fred:inflation"


def test_structured_source_grounds_series_for_rate_layer():
    src = StructuredSource("product", "product", {"adoption": ["adoption", "market penetration"]},
                           fetch_series=lambda k, a, w: [0.1, 0.2, 0.3][-w:])
    key, seq = src.ground_series("market penetration", window=2)
    assert key == "adoption" and seq == [0.2, 0.3]
    assert src.ground_series("bitcoin price") is None             # no match -> no series


def test_router_prefers_structured_then_falls_back_to_retrieval():
    struct = macro_source(fetch=lambda k, a: (3.2, 0.1) if k == "inflation" else None)
    retr = build_retrieval_grounder(lambda q, a: ["passage"],
                                    lambda p: {"value": 42.0, "ci95": 3.0, "confidence": 0.6})
    router = GroundingRouter(sources=[struct], retrieval=retr)
    inf = router.ground("inflation rate")
    assert inf.source == "fred:inflation" and inf.value == 3.2    # structured wins
    tail = router.ground("candidate favorability")               # no structured match -> retrieval
    assert tail is not None and tail.source == "retrieval" and tail.value == 42.0


def test_router_falls_back_when_structured_matches_but_fetch_empty():
    struct = macro_source(fetch=lambda k, a: None)               # matches "inflation" but has no value
    retr = build_retrieval_grounder(lambda q, a: ["p"], lambda p: {"value": 3.0, "ci95": 0.5, "confidence": 0.7})
    router = GroundingRouter(sources=[struct], retrieval=retr)
    gv = router.ground("inflation rate")
    assert gv is not None and gv.source == "retrieval"           # structured empty -> retrieval fallback


def test_router_coverage_report_splits_structured_and_retrieval():
    retr = build_retrieval_grounder(lambda q, a: ["p"],
                                    lambda p: {"value": 1.0, "ci95": 0.2, "confidence": 0.6}
                                    if "favorability" in p else {"value": None})
    router = GroundingRouter(sources=default_sources(fetch=lambda k, a: {"inflation": (3.2, 0.1)}.get(k)),
                             retrieval=retr)
    cov = router.coverage(["inflation rate", "candidate favorability", "quantum flux capacitance"])
    assert cov["grounded"] == 2 and cov["via_structured"] == 1 and cov["via_retrieval"] == 1
    assert cov["coverage"] == round(2 / 3, 3)                     # the third is honestly uncovered


def test_retrieval_ci_calibration_widens_overconfident_intervals():
    # an overconfident extractor: tiny ci95, truth often outside -> calibration must widen it toward nominal
    kb = {"a": (10.0, 12.0), "b": (5.0, 8.0), "c": (20.0, 17.0), "d": (3.0, 5.0), "e": (7.0, 11.0)}  # (value, truth)
    llm = lambda p: next(({"value": v, "ci95": 0.2, "confidence": 0.9} for k, (v, t) in kb.items() if k in p),
                         {"value": None})
    ext = CalibratedExtractor(llm)
    labeled = [{"variable": k, "question": None, "evidence": ["e"], "truth": t} for k, (v, t) in kb.items()]
    cal = calibrate_extractor(ext, labeled, nominal=0.9)
    assert cal["coverage_before"] < 0.5                           # started overconfident
    assert cal["coverage_after"] >= cal["coverage_before"]        # widened
    assert cal["ci_multiplier"] > 1.0 and ext.ci_multiplier == cal["ci_multiplier"]


def test_calibrated_extractor_returns_none_when_not_derivable():
    ext = CalibratedExtractor(lambda p: {"value": None})
    assert ext.extract("anything", None, ["evidence"]) is None