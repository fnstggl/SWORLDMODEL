"""Tests for the posterior-guided forward event attributor."""
from swm.transition.attribution import ForwardAttributor, news_features, _tok


def test_news_features_salience_and_result_cue():
    qtok = _tok("Will Keir Starmer win the UK general election?")
    causal = news_features({"title": "Exit poll says Labour's Keir Starmer wins majority",
                            "description": ""}, qtok)
    irrelevant = news_features({"title": "A style guide to spring fashion", "description": ""}, qtok)
    assert causal[0] > irrelevant[0]          # salience: overlaps starmer/win/election
    assert causal[1] > 0                       # result cue: "wins", "exit poll"


def test_attributor_fits_and_scores_in_range():
    # synthetic transitions: news with result words + question overlap are labeled causal
    recs = []
    for _ in range(30):
        recs.append({"question": "Will team A win the final?",
                     "news": [{"title": "Team A wins the final in a rout", "description": ""},
                              {"title": "Weather forecast for the weekend", "description": ""}],
                     "attributions": [{"news_idx": 0, "score": 0.95}, {"news_idx": 1, "score": 0.0}]})
    att = ForwardAttributor().fit(recs)
    scores = att.score_news(recs[0])
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1]               # learns the causal news scores higher
    assert 0.0 <= att.event_strength(recs[0]) <= 1.0


def test_attributor_handles_empty_gracefully():
    att = ForwardAttributor()                  # unfitted
    assert att.score_news({"news": []}) == []
    assert att.event_strength({"news": []}) == 0.0
