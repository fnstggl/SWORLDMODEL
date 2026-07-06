"""Tests for the semantic stance judge (EXP-047) — the production-wired LLM content extractor."""
from swm.variables.semantic_stance import (SemanticStanceJudge, build_stance_prompt, cached_judge_fn)


_NEWS = [{"title": "Lakers surge to a decisive win, clinch playoff berth", "published_at": "2025-01-01T00:00:00Z"}]


def test_prompt_includes_question_news_and_json_instruction():
    p = build_stance_prompt("Will the Lakers make the playoffs?", _NEWS, resolution_hint="Playoffs by April")
    assert "Lakers" in p and "playoffs" in p.lower()
    assert "Playoffs by April" in p
    assert "JSON" in p and "stance" in p
    assert "how this actually resolved" in p          # the anti-contamination instruction is present


def test_judge_parses_dict_response():
    j = SemanticStanceJudge(judge_fn=lambda prompt: {"stance": 0.7, "confidence": 0.8, "relevant": 2})
    s = j.stance("Q?", _NEWS)
    assert s["stance"] == 0.7 and s["confidence"] == 0.8 and s["relevant"] == 2


def test_judge_parses_raw_json_string_and_clamps():
    j = SemanticStanceJudge(judge_fn=lambda prompt: 'noise {"stance": 1.9, "confidence": 0.5} trailing')
    s = j.stance("Q?", _NEWS)
    assert s["stance"] == 1.0                          # clamped to [-1,1]


def test_no_news_is_neutral():
    j = SemanticStanceJudge(judge_fn=lambda prompt: {"stance": 0.9})
    s = j.stance("Q?", [])
    assert s["stance"] == 0.0 and s["confidence"] == 0.0   # never calls the judge with no evidence


def test_feature_vector_shape():
    j = SemanticStanceJudge(judge_fn=lambda p: {"stance": -0.4, "confidence": 0.5, "relevant": 3})
    v = j.feature_vector("Q?", _NEWS)
    assert len(v) == 4
    assert v[0] == -0.4 and abs(v[1] - (-0.2)) < 1e-9     # confident_stance = stance*conf


def test_cached_judge_fn_replays_and_errors_on_miss():
    fn = cached_judge_fn({"q1": {"stance": 0.3, "confidence": 0.6, "relevant": 1}})
    assert fn("q1")["stance"] == 0.3
    try:
        fn("missing")
        assert False, "should raise on missing key"
    except KeyError:
        pass
