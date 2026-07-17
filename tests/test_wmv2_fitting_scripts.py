"""Pure measurement/scoring functions of the fitting + frozen-vault scripts (offline; the network
loops are exercised only in open environments)."""
import math

import pytest

from experiments.replay_v3.fit_intention_hr import implied_hazard, statement_hazard_ratio
from experiments.replay_v3.fit_survival_pack import effective_resolution_fraction
from experiments.replay_v3.build_event_time_vault import canonical_bytes
from experiments.replay_v3.score_event_time_vault import market_baseline_cdf

T0 = 1_700_000_000.0
DAY = 86400.0


def _hist(points):
    return [{"t": T0 + d * DAY, "p": p} for d, p in points]


# ---------------------------------------------------------------- resolution-time proxy (v2)
def test_sticky_crossing_ignores_transient_spikes():
    # spikes to 0.92 on day 10 but collapses to 0.2 — the naive proxy called this "resolved at 10%"
    hist = _hist([(0, 0.3), (5, 0.5), (10, 0.92), (15, 0.2), (30, 0.25), (50, 0.3), (70, 0.35),
                  (90, 0.95), (95, 0.97), (100, 0.99)])
    frac, ok, proxy = effective_resolution_fraction(hist, end_ts=T0 + 100 * DAY)
    assert ok and proxy == "sticky_price_crossing"
    assert frac == pytest.approx(0.9)                          # the day-90 crossing, not day-10


def test_never_crossing_is_censored_not_unusable():
    hist = _hist([(d, 0.3 + 0.001 * d) for d in range(0, 100, 10)])
    frac, ok, proxy = effective_resolution_fraction(hist, end_ts=T0 + 100 * DAY)
    assert ok and frac is None and proxy == "censored"


def test_early_close_resolution_time_beats_crossing_and_fixes_denominator():
    # market scheduled for 100d resolves decisively and CLOSES at day 40: last trade at 40d.
    # naive proxy: cross(38d)/lifetime(40d) = 0.95 — biased to 1. True window: 100d → ~0.38.
    hist = _hist([(0, 0.4), (10, 0.45), (20, 0.5), (30, 0.6), (36, 0.7), (38, 0.93), (39, 0.96),
                  (40, 0.99)])
    frac, ok, proxy = effective_resolution_fraction(hist, end_ts=T0 + 100 * DAY,
                                                    closed_ts=T0 + 40 * DAY)
    assert ok and proxy == "early_close_resolution_time"
    assert frac == pytest.approx(0.38, abs=0.02)


def test_insufficient_history_is_unusable():
    frac, ok, proxy = effective_resolution_fraction(_hist([(0, 0.5), (1, 0.6)]))
    assert not ok and proxy == "insufficient_history"


# ---------------------------------------------------------------- statement→hazard measurement
def test_implied_hazard_inverts_price_correctly():
    lam = implied_hazard(0.5, T0, T0 + 100 * DAY)
    # p = 1 − exp(−λT) with p=0.5 ⇒ λ = ln2 / T
    assert lam == pytest.approx(math.log(2.0) / (100 * DAY))
    assert implied_hazard(0.999, T0, T0 + 100 * DAY) is None   # saturated price unmeasurable
    assert implied_hazard(0.5, T0 + 100 * DAY - 60, T0 + 100 * DAY) is None   # at deadline


def test_statement_hazard_ratio_measures_a_real_drop():
    # steady price 0.5 → statement at day 50 → price decays toward 0.15 (deal hopes crushed)
    deadline = T0 + 100 * DAY
    pts = [(d, 0.5) for d in range(40, 50)] + [(50 + i, 0.5 - 0.05 * i) for i in range(1, 8)]
    hr = statement_hazard_ratio(_hist(pts), T0 + 50 * DAY, deadline)
    assert hr is not None and hr < 0.8                         # measured suppression
    # openness statement: price climbs after → ratio > 1
    pts_up = [(d, 0.4) for d in range(40, 50)] + [(50 + i, 0.4 + 0.04 * i) for i in range(1, 8)]
    hr_up = statement_hazard_ratio(_hist(pts_up), T0 + 50 * DAY, deadline)
    assert hr_up is not None and hr_up > 1.2
    # too few points on one side → no row (never a fabricated measurement)
    assert statement_hazard_ratio(_hist([(49, 0.5), (51, 0.4)]), T0 + 50 * DAY, deadline) is None


def test_fit_pipeline_composes_with_fit_intention_hazard_ratios():
    from swm.world_model_v2.event_time import fit_intention_hazard_ratios
    deadline = T0 + 100 * DAY
    rows = []
    for k in range(6):
        pts = [(d, 0.5) for d in range(40, 50)] + [(50 + i, 0.5 - 0.025 * i) for i in range(1, 8)]
        hr = statement_hazard_ratio(_hist(pts), T0 + 50 * DAY, deadline)
        rows.append({"commitment_level": "committed_to_prevent", "hazard_ratio": hr})
    pack = fit_intention_hazard_ratios(rows)
    med, lo, hi = pack["hazard_ratios"]["committed_to_prevent"]
    assert lo <= med <= hi and med < 1.0                       # measured, pooled toward no-effect


# ---------------------------------------------------------------- vault scoring surfaces
def test_market_baseline_cdf_monotone_and_hits_freeze_price_at_deadline():
    grid = [T0 + k / 10.0 * 100 * DAY for k in range(1, 11)]
    cdf = market_baseline_cdf(0.35, T0, T0 + 100 * DAY, grid)
    assert cdf == sorted(cdf)
    assert cdf[-1] == pytest.approx(0.35, abs=0.01)            # F(deadline) = the frozen price


def test_canonical_bytes_stable_ordering():
    a = canonical_bytes({"b": 1, "a": [2, 3]})
    b = canonical_bytes({"a": [2, 3], "b": 1})
    assert a == b


def test_vault_scorer_crps_prefers_correct_timing():
    """The scoring identity the frozen vault runs on: a system CDF concentrated near the true event
    time beats the flat market baseline in censoring-aware CRPS."""
    from swm.world_model_v2.event_time import crps_first_passage
    span = 100 * DAY
    grid = [T0 + k / 10.0 * span for k in range(1, 11)]
    event = T0 + 0.3 * span
    sharp = [0.05, 0.2, 0.85, 0.9, 0.92, 0.93, 0.94, 0.95, 0.95, 0.95]
    flat = market_baseline_cdf(0.95, T0, T0 + span, grid)
    assert crps_first_passage(grid, sharp, event_ts=event, as_of=T0, horizon_ts=T0 + span) < \
        crps_first_passage(grid, flat, event_ts=event, as_of=T0, horizon_ts=T0 + span)
