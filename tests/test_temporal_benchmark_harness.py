"""The §31 harness is real infrastructure: metrics compute correctly on a synthetic resolved
case, the as-of guard refuses leaked cases, and the empty corpus reports validation INCOMPLETE
(never a calibration claim)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "temporal"))

import harness as H  # noqa: E402


def test_metrics_compute_on_a_synthetic_resolved_case():
    as_of, hz = 1_772_000_000.0, 1_772_000_000.0 + 30 * 86400.0
    actual_first = as_of + 9 * 86400.0
    case = {"case_id": "syn1", "as_of": as_of, "horizon": hz,
            "resolved": {"first_event_ts": actual_first,
                         "event_sequence": [{"label": "a", "ts": as_of + 2 * 86400},
                                            {"label": "b", "ts": as_of + 9 * 86400}],
                         "decision_events": [{"actor": "x", "ts": as_of + 9 * 86400}]}}
    arm = {"arm": "event_driven", "case_id": "syn1",
           "cdf_grid_ts": [as_of + k / 10 * (hz - as_of) for k in range(1, 11)],
           "cdf": [0.05, 0.15, 0.4, 0.55, 0.65, 0.7, 0.74, 0.76, 0.78, 0.8],
           "median_first_passage_ts": as_of + 11 * 86400.0,
           "quantiles": {"0.1": as_of + 3 * 86400.0, "0.5": as_of + 11 * 86400.0,
                         "0.9": as_of + 26 * 86400.0},
           "p_censored": 0.2,
           "predicted_triggers": [{"actor": "x", "ts": as_of + 8.5 * 86400,
                                   "trigger_type": "a"},
                                  {"actor": "y", "ts": as_of + 20 * 86400,
                                   "trigger_type": "b"}],
           "runtime_s": 12.0}
    s = H.score_case(case, arm)
    assert 0.0 <= s["crps_first_passage"] <= 1.0
    assert s["interval_coverage_10_90"] is True
    assert s["first_event_error_s"] == pytest.approx(2 * 86400.0)
    assert s["censoring_brier"] == pytest.approx(0.04)
    assert s["trigger_pr"]["recall"] == 1.0 and s["trigger_pr"]["precision"] == 0.5
    assert s["runtime_s_per_simulated_day"] == pytest.approx(0.4)
    # order accuracy: triggers a (8.5d) before b (20d) vs actual a (2d) before b (9d) — concordant
    assert s["order_accuracy"] == 1.0


def test_as_of_guard_refuses_leaked_cases():
    with pytest.raises(ValueError):
        H._check_as_of({"case_id": "leak", "as_of": "2026-03-02",
                        "resolved": {"event_sequence": [{"label": "pre", "ts": 100.0}]}})


def test_empty_corpus_reports_validation_incomplete():
    cases = H.load_cases()
    if cases:
        pytest.skip("a real corpus exists — incompleteness note no longer applies")
    report = {"validation_status": ("INCOMPLETE — no resolved-case corpus present; temporal "
                                    "calibration is NOT claimed" if not cases else "scored")}
    assert "NOT claimed" in report["validation_status"]
