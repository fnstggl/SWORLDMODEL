"""Tests for the event-conditioned belief-transition operator."""
from swm.transition.belief_dynamics import BeliefTransition, event_features, featurize, state_features


def test_state_features_capture_momentum_and_volatility():
    rising = state_features([{"p": 0.2}, {"p": 0.3}, {"p": 0.4}, {"p": 0.5}])
    assert rising["level"] == 0.5
    assert rising["momentum"] > 0                      # upward trajectory
    flat = state_features([{"p": 0.5}] * 5)
    assert flat["momentum"] == 0.0 and flat["volatility"] == 0.0
    choppy = state_features([{"p": 0.5}, {"p": 0.7}, {"p": 0.4}, {"p": 0.6}])
    assert choppy["volatility"] > flat["volatility"]


def test_event_features_salience_and_impact():
    rec = {"question": "Will the Federal Reserve cut interest rates in December?",
           "news": [{"title": "Fed signals interest rate cut likely in December", "description": ""}]}
    ef = event_features(rec)
    assert ef["salience"] > 0                           # overlapping tokens (fed, interest, rate, december)
    assert event_features({"news": []}, impact=0.7)["impact"] == 0.7


def test_transition_predicts_valid_belief_and_uses_impact():
    # synthetic: belief change follows the injected impact
    train = []
    for i in range(120):
        imp = (-1.0 if i % 2 else 1.0)
        p = 0.5
        train.append({"history": [{"p": p}], "target": {"p": min(1, max(0, p + 0.2 * imp))},
                      "question": "q", "news": [], "_impact": imp})
    bt = BeliefTransition(event_impact_fn=lambda r: r.get("_impact", 0.0)).fit(train)
    up = bt.predict_change({"history": [{"p": 0.5}], "question": "q", "news": [], "_impact": 1.0})
    down = bt.predict_change({"history": [{"p": 0.5}], "question": "q", "news": [], "_impact": -1.0})
    assert up > down                                    # positive impact -> larger predicted rise
    b = bt.predict_belief({"history": [{"p": 0.5}], "question": "q", "news": [], "_impact": 1.0})
    assert 0.0 <= b <= 1.0


def test_featurize_length_matches_feature_names():
    from swm.transition.belief_dynamics import FEATURES
    rec = {"history": [{"p": 0.4}, {"p": 0.5}], "question": "q", "news": []}
    assert len(featurize(rec)) == len(FEATURES)
