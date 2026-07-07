"""Tests for the uniform scored-validation harness (EXP-065)."""
from swm.eval.world_model_bench import score_binary, score_share


def test_score_binary_per_mechanism_and_skill():
    recs = [{"mechanism": "committee", "p": 0.9, "y": 1, "base": 0.5},
            {"mechanism": "committee", "p": 0.8, "y": 1, "base": 0.5},
            {"mechanism": "committee", "p": 0.2, "y": 0, "base": 0.5},
            {"mechanism": "single_agent", "p": 0.6, "y": 1, "base": 0.5}]
    out = score_binary(recs)
    assert out["committee"]["n"] == 3 and out["committee"]["accuracy"] == 1.0
    assert out["committee"]["brier_skill_vs_base"] > 0          # confident-correct beats the 0.5 baseline
    assert "single_agent" in out


def test_score_share_rmse_coverage_and_skill():
    recs = [{"mechanism": "electorate", "pred": 0.58, "truth": 0.6, "marginal": 0.5, "lo": 0.5, "hi": 0.66},
            {"mechanism": "electorate", "pred": 0.41, "truth": 0.4, "marginal": 0.5, "lo": 0.33, "hi": 0.49}]
    out = score_share(recs)["electorate"]
    assert out["n"] == 2 and out["rmse"] < out["rmse_marginal"]  # closer than the marginal
    assert out["coupling_skill"] > 0 and out["interval_coverage"]["empirical_coverage"] == 1.0
