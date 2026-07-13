"""Phase 3 accuracy — unit tests for fitted observation models + causal-latent inference."""
from __future__ import annotations

from swm.world_model_v2 import phase3_fitted_obs as fo
from swm.world_model_v2 import phase3_causal_latents as cl


# ---------------- fitted observation models ----------------
def _q(outcome, tags):
    return {"outcome": outcome, "tags": tags}


def _tag(direction, cls="forecast", strength="strong", rel=0.9):
    return {"outcome_direction": direction, "claim_class": cls, "strength": strength, "reliability": rel}


def test_fit_learns_positive_global_weight():
    # supports_yes claims coincide with YES outcomes, supports_no with NO -> positive discrimination
    train = [_q(1, [_tag("supports_yes")]) for _ in range(6)] + [_q(0, [_tag("supports_no")]) for _ in range(6)]
    params = fo.fit(train)
    assert params["w_global"] > 0
    # predicting a strong supports_yes question should exceed 0.5
    assert fo.predict_rate(_q(1, [_tag("supports_yes")]), params) > 0.5
    assert fo.predict_rate(_q(0, [_tag("supports_no")]), params) < 0.5


def test_fitted_lr_direction():
    train = [_q(1, [_tag("supports_yes")]) for _ in range(6)] + [_q(0, [_tag("supports_no")]) for _ in range(6)]
    params = fo.fit(train)
    assert fo.fitted_lr(_tag("supports_yes"), params) > 1.0
    assert fo.fitted_lr(_tag("supports_no"), params) < 1.0


def test_predict_rate_bounded():
    train = [_q(1, [_tag("supports_yes")] * 5)]
    params = fo.fit(train)
    p = fo.predict_rate(_q(1, [_tag("supports_yes")] * 20), params)
    assert 0.0 < p < 1.0


# ---------------- causal latents ----------------
def test_necessary_conjunction_lowers_with_more_latents():
    lat = [{"id": f"l{i}", "type": "feasibility", "favorable_supports_yes": True} for i in range(3)]
    post = {f"l{i}": {"mean": 0.6, "favorable_supports_yes": True} for i in range(3)}
    r1 = cl.combine_to_rate(lat[:1], {"l0": post["l0"]}, "necessary_conjunction")
    r3 = cl.combine_to_rate(lat, post, "necessary_conjunction")
    assert r3 < r1                                            # more necessary conditions => lower joint prob


def test_sufficient_disjunction_raises():
    lat = [{"id": f"l{i}", "type": "hazard", "favorable_supports_yes": True} for i in range(3)]
    post = {f"l{i}": {"mean": 0.4, "favorable_supports_yes": True} for i in range(3)}
    r = cl.combine_to_rate(lat, post, "sufficient_disjunction")
    assert r > 0.4                                            # any-of raises above a single latent


def test_favors_raises_latent_against_lowers():
    lat = [{"id": "intent_a", "type": "intent", "favorable_supports_yes": True}]
    tags = {"c1": {"strength": "strong", "reliability": 0.9}}
    up = cl.infer_latent_posteriors(lat, {"c1": [{"latent_id": "intent_a", "direction": "favors"}]}, tags)
    dn = cl.infer_latent_posteriors(lat, {"c1": [{"latent_id": "intent_a", "direction": "against"}]}, tags)
    assert up["intent_a"]["mean"] > 0.5 > dn["intent_a"]["mean"]


def test_polarity_flip_when_favorable_supports_no():
    """A latent whose favorable state pushes toward NO must invert its contribution to P(yes)."""
    lat_yes = [{"id": "x", "type": "regime", "favorable_supports_yes": True}]
    lat_no = [{"id": "x", "type": "regime", "favorable_supports_yes": False}]
    post = {"x": {"mean": 0.8, "favorable_supports_yes": True}}
    r_yes = cl.combine_to_rate(lat_yes, post, "single_driver")
    post_no = {"x": {"mean": 0.8, "favorable_supports_yes": False}}
    r_no = cl.combine_to_rate(lat_no, post_no, "single_driver")
    assert abs(r_yes - 0.8) < 1e-6 and abs(r_no - 0.2) < 1e-6


def test_causal_rate_none_without_latents():
    rate, post = cl.causal_latent_rate({"latents": [], "combination": "weighted_mean"}, {}, {})
    assert rate is None
