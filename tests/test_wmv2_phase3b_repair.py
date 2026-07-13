"""Phase 3B repair — unit tests for the calibrated posterior, convex blend, gate, and reference priors."""
from __future__ import annotations
import math

from swm.world_model_v2.phase3b_repair import (calibrated_rate_posterior, combine, logit, sigmoid,
                                               reference_prior_ab, load_params)


def _tags(n_yes=0, n_no=0, strength="strong"):
    rows = []
    for i in range(n_yes):
        rows.append({"claim_id": f"y{i}", "outcome_direction": "supports_yes", "strength": strength,
                     "reliability": 0.9, "is_strategic": False, "dependence_group": f"gy{i}"})
    for i in range(n_no):
        rows.append({"claim_id": f"n{i}", "outcome_direction": "supports_no", "strength": strength,
                     "reliability": 0.9, "is_strategic": False, "dependence_group": f"gn{i}"})
    return rows


def test_shrinkage_pulls_toward_prior():
    """gamma<1 must move the posterior LESS far from the prior than gamma=1 (over-responsiveness fix)."""
    tags = _tags(n_yes=5)
    m_full, _, n = calibrated_rate_posterior(tags, 1.0, 1.0, gamma=1.0)
    m_shrunk, _, _ = calibrated_rate_posterior(tags, 1.0, 1.0, gamma=0.5)
    assert n == 5
    assert m_full > 0.5 and m_shrunk > 0.5              # yes-evidence pushes up
    assert abs(m_shrunk - 0.5) < abs(m_full - 0.5)      # shrinkage stays closer to the 0.5 prior


def test_no_info_mix_flattens():
    tags = _tags(n_yes=5)
    m0, _, _ = calibrated_rate_posterior(tags, 1.0, 1.0, gamma=1.0, no_info_mix=0.0)
    m1, _, _ = calibrated_rate_posterior(tags, 1.0, 1.0, gamma=1.0, no_info_mix=0.5)
    assert abs(m1 - 0.5) < abs(m0 - 0.5)


def test_neutral_tags_do_not_move_posterior():
    tags = [{"claim_id": "a", "outcome_direction": "neutral", "strength": "strong", "reliability": 0.9,
             "is_strategic": False, "dependence_group": "g"}]
    m, _, n = calibrated_rate_posterior(tags, 2.0, 2.0)
    assert n == 0
    assert abs(m - 0.5) < 1e-6                           # Beta(2,2) mean = 0.5, unchanged


def test_gate_falls_back_to_phase2_below_threshold():
    params = {"blend": {"w_phase2": 0.5}, "gate": {"min_effective_obs": 4}}
    p, mode = combine(0.7, 0.2, n_effective=2, params=params)
    assert mode == "gate_phase2_fallback"
    assert abs(p - 0.7) < 1e-9                           # returns Phase-2 exactly


def test_convex_blend_is_between_and_never_inverts():
    params = {"blend": {"w_phase2": 0.5}, "gate": {"min_effective_obs": 0}}
    p2, p3 = 0.8, 0.2
    p, mode = combine(p2, p3, n_effective=5, params=params)
    assert mode == "blended"
    # convex in logit space => between the two endpoints, and on the same side as the weighted mean
    lo, hi = sorted([p2, p3])
    assert lo < p < hi
    # w=0.5 => logit(p) is the average of the two logits
    assert abs(logit(p) - 0.5 * (logit(p2) + logit(p3))) < 1e-6


def test_w1_is_pure_phase2():
    params = {"blend": {"w_phase2": 1.0}, "gate": {"min_effective_obs": 0}}
    p, mode = combine(0.73, 0.11, n_effective=5, params=params)
    assert mode == "phase2_only"
    assert abs(p - 0.73) < 1e-9


def test_reference_prior_shutdown_is_low_and_widened():
    a, b, rd = reference_prior_ab("shutdown_x", "Will there be a US federal government shutdown on date?",
                                  "2024-01-01", "politics")
    assert a is not None and rd is not None
    mean = a / (a + b)
    assert mean < 0.35                                   # shutdowns are rare
    assert (a + b) < 30                                  # transport-widened (not a sharp prior)


def test_load_params_fallback_is_safe_phase2():
    """With no params file at a bogus path, the fallback must be pure Phase-2 (never touches Phase-3)."""
    p = load_params(path="/nonexistent/repair_params.json")
    assert p["blend"]["w_phase2"] == 1.0
    assert p["gate"]["min_effective_obs"] >= 999
