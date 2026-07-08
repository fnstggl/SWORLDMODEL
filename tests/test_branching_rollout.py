"""Branching-realities rollout — martingale preservation, bimodal futures, pivotal-branch decomposition."""
from swm.simulation.branching_rollout import (BranchingRollout, forward_forecast, martingale_step,
                                              pivotal_branches)
from swm.transition.future_events import events_from_records


def _fed_calendar():
    """Feb CPI forks the belief (hot -0.25 / cool +0.25); the March FOMC resolves ~Bernoulli(belief)."""
    return events_from_records([
        {"name": "cpi", "time": 2.0, "outcomes": [{"label": "hot", "prob": 0.5, "impact": -0.25},
                                                  {"label": "cool", "prob": 0.5, "impact": 0.25}]},
        {"name": "fomc", "time": 4.0, "from_belief": True},
    ])


def test_resolving_event_preserves_the_martingale_mean():
    """Symmetric belief-shifting events must not move the marginal P(event) — it stays at b0 (EXP-033)."""
    res = BranchingRollout(_fed_calendar()).run(0.55, 5.0, n=20000, seed=0)
    assert abs(res["p_event"] - 0.55) < 0.02                   # marginal unchanged...


def test_events_make_the_future_bimodal():
    """...but the DISTRIBUTION is bimodal — mass near 0 and near 1, not piled at the mean."""
    res = BranchingRollout(_fed_calendar()).run(0.55, 5.0, n=20000, seed=0)
    outs = res["_outcomes"]
    near_mean = sum(1 for o in outs if 0.4 < o < 0.6) / len(outs)
    assert near_mean < 0.05                                    # almost nothing at the mean (resolves to 0/1)


def test_pivotal_branch_recovers_the_conditional_fork():
    """The decomposition must recover P(yes | CPI hot) ≈ 0.30 and P(yes | CPI cool) ≈ 0.80."""
    roll = BranchingRollout(_fed_calendar())
    res = roll.run(0.55, 5.0, n=20000, seed=1)
    pivots = pivotal_branches(res, roll.calendar, top=3)
    assert pivots and pivots[0]["event"] == "cpi"              # CPI is THE pivot
    cond = {b["label"]: b["p_event"] for b in pivots[0]["branches"]}
    assert abs(cond["hot"] - 0.30) < 0.04
    assert abs(cond["cool"] - 0.80) < 0.04
    assert pivots[0]["spread"] > 0.4                           # a real fork, not a nudge


def test_no_events_defers_to_persistence():
    """With no calendar events and no vol, the forward forecast is exactly the starting belief (no fork)."""
    cal = events_from_records([])
    res = forward_forecast(0.62, 5.0, cal, n=4000)
    assert abs(res["p_event"] - 0.62) < 1e-6
    assert res["pivotal_branches"] == []
    assert res["irreducible_frac"] == 1.0                      # nothing known to resolve → all irreducible


def test_forward_forecast_reports_reducible_share_of_a_known_pivot():
    res = forward_forecast(0.55, 5.0, _fed_calendar(), n=8000)
    # CPI sharpens 55% -> 30%/80% (reducible by watching it), but the FOMC coin-flip resolution stays
    # irreducible — so ~1/4 of the outcome variance is reducible, the honest split.
    assert 0.15 < res["reducible_frac"] < 0.4
    assert "PIVOT cpi" in res["headline"]


def test_continuous_step_is_pluggable():
    """A drifting continuous step (stand-in for the calibrated transition operator) moves the mean —
    proving the between-event dynamics compose with the discrete branching."""
    cal = events_from_records([])                             # no discrete events; only the continuous term

    def drift_up(belief, dt, rng):
        return belief + 0.02 * dt                             # a calibrated upward drift

    res = forward_forecast(0.50, 5.0, cal, continuous_step=drift_up, n=2000)
    assert res["p_event"] > 0.55                              # the pluggable dynamics moved it forward


def test_surprise_hazard_widens_without_shifting():
    """Unscheduled mean-zero shocks widen the distribution but keep the mean — the irreducible floor."""
    calm = forward_forecast(0.5, 5.0, events_from_records([]), n=8000)
    noisy = forward_forecast(0.5, 5.0, events_from_records([{"hazard": {"rate": 1.0, "shock_sd": 0.08}}]),
                             n=8000)
    assert noisy["sd"] > calm["sd"] + 0.03                    # wider
    assert abs(noisy["p_event"] - 0.5) < 0.02                 # not shifted
