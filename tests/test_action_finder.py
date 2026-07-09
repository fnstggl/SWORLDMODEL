"""The general typed best-action finder: one spine (calibrated world model + best-arm racing), a search
operator matched to each action TYPE (continuous / discrete / generative / structured)."""
import math

from swm.decision.action_finder import (Action, Continuous, DiscreteChoice, GenerativeText, Structured,
                                         find_best_action, world_model)


def test_continuous_finds_the_revenue_optimum():
    # revenue = price * P(sale); P(sale) = sigmoid(3*(1 - price/50)). Optimum is interior (~40-50).
    def p_sale(price):
        return 1 / (1 + math.exp(-3 * (1 - price / 50)))
    r = find_best_action(Continuous("price", 5, 120, steps=15, rounds=4),
                         world_model(p_sale, value_fn=lambda price, s: price * s), seed=0)
    price = r.best.action.value
    assert 25 < price < 60                       # interior optimum, not the boundary
    assert r.best.value > 20                      # beats trivial low/high prices


def test_discrete_picks_the_best_option():
    vendors = {"acme": 0.55, "globex": 0.72, "initech": 0.40}
    r = find_best_action(DiscreteChoice([Action(k, k) for k in vendors]),
                         world_model(lambda v: vendors[v]), seed=0)
    assert r.best.label == "globex"


def test_structured_coordinate_ascent_finds_the_best_config():
    def conv(cfg):
        ch = {"email": 0.1, "ads": 0.05, "referral": 0.25}[cfg["channel"]]
        st = {"plain": 0.02, "story": 0.18}[cfg["style"]]
        return min(0.95, ch + st + 0.05 * cfg["budget"] / 5000)
    r = find_best_action(Structured({"channel": ["email", "ads", "referral"],
                                     "style": ["plain", "story"], "budget": (100, 5000, 4)}, sweeps=3),
                         world_model(conv), seed=0)
    cfg = r.best.action.value
    assert cfg["channel"] == "referral" and cfg["style"] == "story"   # the true optimum fields


def test_generative_selects_the_best_proposal():
    def propose(seed):
        return [Action("short punchy ask" if i % 2 else "long rambling padded wordy text", f"v{i}")
                for i in range(6)]
    r = find_best_action(GenerativeText(propose, rounds=2, k=6),
                         world_model(lambda t: 0.4 if "short" in t else 0.1), seed=0)
    assert "short" in r.best.action.value


def test_world_model_accepts_an_ensemble():
    # score_fn may return a LIST of ensemble samples (predictive uncertainty)
    r = find_best_action(DiscreteChoice([Action("a", "a"), Action("b", "b")]),
                         world_model(lambda v: [0.9, 0.8, 0.85] if v == "a" else [0.2, 0.1, 0.15]), seed=0)
    assert r.best.label == "a"


def test_result_carries_confidence_and_contrast():
    r = find_best_action(DiscreteChoice([Action("hi", "hi"), Action("lo", "lo")]),
                         world_model(lambda v: 0.8 if v == "hi" else 0.2),
                         baseline=Action("lo", "lo"), seed=0)
    assert r.best.label == "hi"
    assert "vs_baseline" in r.contrast and 0.0 <= r.win_prob <= 1.0
