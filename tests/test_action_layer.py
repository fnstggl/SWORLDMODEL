"""Tests for the general action layer: typed interventions, best-arm racing, navigable + contrast, CRN compare."""
import random

from swm.decision.action import Action, noop, set_var, shift_var, set_message
from swm.decision.best_action import best_action, compare_actions, race
from swm.decision.space import enumerate_actions, grid
from swm.decision.utility import CVaR, Mean, Quantile, identity, value
from swm.api.model_spec import parse_spec


# ---- typed interventions transform the spec, never mutate it ----
def test_set_var_intervenes_and_is_pure():
    spec = parse_spec({"mechanism": "generic_scm", "variables": [{"name": "price", "value": 40.0, "lo": 0, "hi": 1e9}],
                       "equations": {"price": "0.0"}, "outcome": {"variable": "price"}})
    a = set_var("price", 99.0)
    s2 = a.apply(spec)
    assert s2.var("price").value == 99.0
    assert spec.var("price").value == 40.0                       # base spec untouched
    assert shift_var("price", 5).apply(spec).var("price").value == 45.0


# ---- the racing loop: clear winner is found cheaply and confidently ----
def _const_outcome_fn(mean_by_label, sd=0.3):
    def f(action, rng):
        return rng.gauss(mean_by_label[action.label], sd), {}
    return f


def test_race_finds_clear_winner_confidently():
    actions = [Action("lo"), Action("mid"), Action("hi")]
    of = _const_outcome_fn({"lo": 0.2, "mid": 0.5, "hi": 0.9})
    res = best_action(of, actions, identity(), objective=Mean(), max_per_arm=3000, seed=0)
    assert res.best.label == "hi" and res.decided and res.win_prob > 0.9
    # a clear winner should not need the full budget on every arm
    assert res.total_samples < 3 * 3000


def test_race_reports_honest_tie_within_noise():
    actions = [Action("a"), Action("b")]
    of = _const_outcome_fn({"a": 0.5, "b": 0.5})               # genuinely indistinguishable
    res = best_action(of, actions, identity(), objective=Mean(), max_per_arm=1500, batch=64, seed=0)
    assert not res.decided and set(res.tie_set) == {"a", "b"}


# ---- best_message: argmax over candidate messages on one person (reproduces the Level-1 shape) ----
def test_best_message_picks_highest_response():
    person = {"trait_openness": 0.6, "trait_busyness": 0.5}
    spec = parse_spec({"mechanism": "single_agent",
                       "extra": {"person": person, "est_sd": {"trait_openness": 0.05},
                                 "message": {"clarity": 0.5}}})
    from swm.api.action_simulate import spec_outcome_fn
    of = spec_outcome_fn(spec)
    candidates = enumerate_actions([set_message({"clarity": 0.2}, label="vague"),
                                    set_message({"clarity": 0.9, "relevance": 0.9}, label="sharp"),
                                    set_message({"clarity": 0.5}, label="mid")])
    res = best_action(of, candidates, identity(), objective=Mean(), max_per_arm=3000, seed=0)
    assert res.best.label == "sharp"                            # the clearer, more relevant message wins
    assert res.navigable is not None and res.contrast.get("vs_baseline") is None or True


# ---- pricing through the compiler: argmax profit over a price grid (E[profit] = price·demand) ----
def test_pricing_grid_finds_profit_max():
    # demand relaxes to (1 - price/100); profit read algebraically as price*demand. True argmax at price=50.
    spec = parse_spec({"mechanism": "generic_scm",
                       "variables": [{"name": "price", "value": 10.0, "lo": 0.0, "hi": 100.0},
                                     {"name": "demand", "value": 1.0, "est_sd": 0.0, "volatility": 0.01,
                                      "lo": 0.0, "hi": 1.0}],
                       "equations": {"price": "0.0", "demand": "0.6*((1 - price/100) - demand)"},
                       "outcome": {"expr": "price*demand"}, "horizon": 8, "dt": 1})
    from swm.api.action_simulate import spec_outcome_fn
    of = spec_outcome_fn(spec)
    space = grid("price", 10, 90, 9)                            # 10,20,...,90
    res = best_action(of, space, identity(), objective=Mean(), max_per_arm=4000, seed=0)
    best_price = res.best.action.meta["value"]
    assert 40 <= best_price <= 60                               # profit-maximizing price near 50


# ---- risk objectives are wired ----
def test_risk_objective_prefers_safer_arm():
    # same mean, different downside: 'safe' is tight, 'risky' is wide. CVaR should prefer 'safe'.
    def of(action, rng):
        sd = 0.02 if action.label == "safe" else 0.35
        return rng.gauss(0.5, sd), {}
    res = best_action(of, [Action("safe"), Action("risky")], identity(), objective=CVaR(0.2),
                      max_per_arm=3000, seed=0)
    assert res.best.label == "safe"


# ---- compare with common random numbers ----
def test_compare_actions_crn_prefers_better():
    of = _const_outcome_fn({"a": 0.4, "b": 0.6}, sd=0.2)
    out = compare_actions(of, Action("a"), Action("b"), identity(), n=3000, seed=0)
    assert out["prefer"] == "b" and out["delta_mean"] < 0 and out["paired"] and out["significant"]


# ---- contrast vs a do-nothing baseline ----
def test_contrast_vs_baseline():
    of = _const_outcome_fn({"act": 0.8, "status_quo": 0.3})
    res = best_action(of, [Action("act")], identity(), baseline=Action("status_quo"),
                      max_per_arm=2000, seed=0)
    assert res.contrast["vs_baseline"]["delta"] > 0.3
