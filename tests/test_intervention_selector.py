"""Tests for the semantic intervention selector (EXP-056)."""
from swm.api.intervention_selector import InterventionSelector, build_selection_prompt, cached_selector


def test_prompt_lists_options_and_asks_for_json():
    p = build_selection_prompt("maximize clicks", ["Headline A", "Headline B"])
    assert "Headline A" in p and "Headline B" in p and "best" in p and "JSON" in p


def test_select_parses_dict():
    s = InterventionSelector(judge_fn=lambda prompt: {"best": 1, "scores": [0.2, 0.9]})
    out = s.select("goal", ["a", "b"])
    assert out["best"] == 1 and out["scores"] == [0.2, 0.9]


def test_select_parses_raw_json_and_clamps_index():
    s = InterventionSelector(judge_fn=lambda prompt: 'noise {"best": 9} tail')
    out = s.select("goal", ["a", "b"])
    assert out["best"] == 1                              # clamped to valid range


def test_missing_scores_synthesized_from_best():
    s = InterventionSelector(judge_fn=lambda prompt: {"best": 0})
    out = s.select("goal", ["a", "b", "c"])
    assert out["scores"][0] == 1.0 and out["scores"][1] == 0.0


def test_empty_options():
    s = InterventionSelector(judge_fn=lambda p: {"best": 0})
    assert s.select("goal", [])["best"] is None


def test_cached_selector_replays_and_errors_on_miss():
    fn = cached_selector({"k": {"best": 2}})
    assert fn("k")["best"] == 2
    try:
        fn("nope")
        assert False
    except KeyError:
        pass
