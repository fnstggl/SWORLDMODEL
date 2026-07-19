"""Tests for the seven mechanism simulators — each honest + anchored + horizon-aware, verified by construction."""
from swm.api.mechanisms import (MECHANISMS, sim_aggregation, sim_arrival, sim_contest, sim_escalation,
                                sim_persistence, sim_whipcount, simulate_mechanism)


def test_aggregation_integrates_poll_error():
    # a 52% share with 4pt error is a lean, not a lock; a 60% share is near-certain
    lean = sim_aggregation(0.52, share_sd=0.04)
    lock = sim_aggregation(0.60, share_sd=0.04)
    assert 0.6 < lean < 0.8 and lock > 0.98
    assert sim_aggregation(None) == 0.5                       # no data -> base rate
    assert sim_aggregation(0.45, share_sd=0.04) < 0.2         # behind -> unlikely


def test_contest_elo_and_winprob():
    assert abs(sim_contest(rating_diff=0) - 0.5) < 0.03       # equal ratings -> coin flip
    assert sim_contest(rating_diff=200) > sim_contest(rating_diff=50)  # bigger edge -> higher
    assert sim_contest(win_prob=0.73) == 0.73                 # grounded odds pass through


def test_arrival_rises_with_horizon():
    near = sim_arrival(base_rate=0.3, horizon_years=1.0, ref_years=1.0)
    far = sim_arrival(base_rate=0.3, horizon_years=3.0, ref_years=1.0)
    assert far > near                                         # more time -> more likely to have arrived
    assert sim_arrival(base_rate=0.3, horizon_years=1.0) > 0.2


def test_whipcount_counts_votes():
    assert sim_whipcount(committed_yes=60, undecided=0, needed=50) > 0.95     # already have the votes
    assert sim_whipcount(committed_yes=30, undecided=0, needed=50) < 0.05     # cannot reach it
    # 45 committed, 20 undecided leaning 0.5, need 50 -> need 5+ of 20 breaks -> very likely
    mid = sim_whipcount(committed_yes=45, undecided=20, needed=50, lean=0.5)
    assert 0.5 < mid < 1.0


def test_escalation_moves_with_pressure_and_sign():
    up = sim_escalation(base_rate=0.3, pressure=1.2, trend=0.5, push=1.0)
    dn = sim_escalation(base_rate=0.3, pressure=1.2, trend=0.5, push=-1.0)
    assert up > 0.3 > dn                                      # measured pressure pushes by the polarity sign


def test_persistence_decays_with_horizon():
    hold_near = sim_persistence(base_rate=0.9, horizon_years=1.0, ref_years=1.0)
    hold_far = sim_persistence(base_rate=0.9, horizon_years=5.0, ref_years=1.0)
    assert hold_near > hold_far                               # status quo erodes over time
    # complement: P(disruption occurs) is the mirror
    assert abs(sim_persistence(base_rate=0.9, horizon_years=1.0, happens=False) - (1 - hold_near)) < 1e-9


def test_router_dispatches_and_falls_back():
    assert simulate_mechanism("contest", {"rating_diff": 200}) > 0.5
    assert simulate_mechanism("aggregation", {"share": 0.6, "share_sd": 0.04}) > 0.9
    assert simulate_mechanism("nonsense", {}) is None         # unknown -> caller falls back to generic sim
    assert set(MECHANISMS) == {"aggregation", "contest", "diffusion", "arrival", "whipcount",
                               "escalation", "persistence"}


def test_provenance_widening_exp101():
    # invented params may not assert near-certainty: no hard gates, anchored toward the base rate
    g = sim_whipcount(committed_yes=30, undecided=0, needed=50, base_rate=0.4)                  # grounded gate
    i = sim_whipcount(committed_yes=30, undecided=0, needed=50, base_rate=0.4, provenance="invented")
    assert g < 0.05 and 0.05 < i < 0.6 and i > g
    ga = sim_aggregation(0.30, base_rate=0.4)
    ia = sim_aggregation(0.30, base_rate=0.4, provenance="invented")
    assert ia > ga and 0.03 < ia < 0.6
    # quoted sits between grounded and invented; grounded behavior is unchanged by the new tiers
    qa = sim_aggregation(0.30, base_rate=0.4, provenance="quoted")
    assert ga <= qa <= ia
    assert simulate_mechanism("whipcount", {"committed_yes": 60, "needed": 50}) > 0.95          # default grounded
