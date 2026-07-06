"""Tests for the resolution-aware content extractor (EXP-044)."""
from swm.variables.content_extractor import extract, feature_vector, parse_question


def test_parse_extracts_subject_and_threshold():
    f = parse_question("Will the unemployment rate be above 4.2% in November?")
    assert "unemployment" in f["subject"]
    assert f["threshold"] == 4.2
    assert f["direction"] == 1                       # "above" -> positive comparison


def test_parse_direction_below():
    assert parse_question("Will inflation be below 3% next year?")["direction"] == -1


def test_entity_linking_ignores_unrelated_news():
    news = [{"title": "Lakers beat the Celtics tonight", "published_at": "2025-01-01T00:00:00Z"},
            {"title": "Weather forecast for the weekend", "published_at": "2025-01-01T00:00:00Z"}]
    f = extract("Will the Lakers win the championship?", news, t_target=1735800000.0)
    assert f["n_linked"] == 1                         # only the Lakers item links to the subject


def test_positive_stance_toward_subject():
    news = [{"title": "Lakers surge ahead, clinch a decisive win", "published_at": "2025-01-01T00:00:00Z"}]
    f = extract("Will the Lakers win the championship?", news, t_target=1735900000.0)
    assert f["subject_stance"] > 0                    # positive outcome terms near the subject


def test_negative_stance_toward_subject():
    news = [{"title": "Lakers lose again, eliminated from contention", "published_at": "2025-01-01T00:00:00Z"}]
    f = extract("Will the Lakers win the championship?", news, t_target=1735900000.0)
    assert f["subject_stance"] < 0


def test_feature_vector_shape_and_no_link():
    from swm.variables.content_extractor import FEATURE_NAMES
    v = feature_vector("Will X happen?", [], t_target=0.0)
    assert len(v) == len(FEATURE_NAMES)
    assert all(isinstance(x, float) for x in v)
