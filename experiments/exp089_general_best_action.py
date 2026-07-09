"""EXP-089 — the GENERAL best-action finder across action TYPES.

argmax_a E[U(outcome) | do(a), context] on one shared spine (a calibrated world model + best-arm racing),
with the search operator matched to the action type. Demonstrates the four common shapes people ask a
best-action product for, each trying MANY actions (not 3-4):

  - CONTINUOUS  (best price to maximize revenue)      -> grid → local refine over the response curve
  - DISCRETE    (best vendor / candidate / channel)   -> enumerate + best-arm race
  - STRUCTURED  (best campaign = channel×budget×style) -> coordinate ascent over the fields
  - GENERATIVE  (best copy)                            -> propose → score → mutate (the message pattern)

The world models here are toy stand-ins; the point is the harness. In production each `score_fn` is a
calibrated per-domain world model (a demand curve, a persuasion model, …) — the search is general and
cheap, the world model is the hard, data-hungry part.

Run:  PYTHONPATH=. python experiments/exp089_general_best_action.py
"""
from __future__ import annotations

import math

from swm.decision.action_finder import (Action, Continuous, DiscreteChoice, GenerativeText, Structured,
                                         find_best_action, world_model)


def main():
    print("=" * 78)
    print("EXP-089  general best-action finder (typed action space + best-arm racing)")
    print("=" * 78)

    # 1) CONTINUOUS — best price to maximize revenue. Demand falls with price; revenue = price × P(sale).
    def p_sale(price):
        return 1 / (1 + math.exp(-3 * (1 - price / 50)))
    r = find_best_action(Continuous("price", 5, 120, steps=15, rounds=4),
                         world_model(p_sale, value_fn=lambda price, s: price * s), seed=0)
    print(f"\n[CONTINUOUS] best price = {r.best.action.value:.2f}   E[revenue] = {r.best.value:.2f}   "
          f"({r.total_samples} sims)")

    # 2) DISCRETE — best vendor by P(on-time delivery); race to confidence.
    vendors = {"acme": 0.55, "globex": 0.72, "initech": 0.40, "umbrella": 0.68}
    r = find_best_action(DiscreteChoice([Action(k, k) for k in vendors]),
                         world_model(lambda v: vendors[v]), baseline=Action("acme", "acme"), seed=0)
    print(f"[DISCRETE]   best vendor = {r.best.label}  P={r.best.value:.3f}  "
          f"decided={r.decided}  win_prob={r.win_prob}")

    # 3) STRUCTURED — best campaign config (channel × style × budget) maximizing P(conversion).
    def conv(cfg):
        ch = {"email": 0.10, "ads": 0.05, "referral": 0.25}[cfg["channel"]]
        st = {"plain": 0.02, "story": 0.18}[cfg["style"]]
        return min(0.95, ch + st + 0.05 * cfg["budget"] / 5000)
    r = find_best_action(Structured({"channel": ["email", "ads", "referral"],
                                     "style": ["plain", "story"], "budget": (100, 5000, 5)}, sweeps=3),
                         world_model(conv), seed=0)
    print(f"[STRUCTURED] best campaign = {r.best.label}   P(conversion)={r.best.value:.3f}")

    # 4) GENERATIVE — best copy: propose variants, score by a (toy) world model, pick the winner.
    def propose(seed):
        variants = ["short, one clear ask", "long rambling wordy padded pitch with no ask",
                    "medium, one number and one ask", "buzzword-heavy synergy leverage disrupt"]
        return [Action(v, f"v{i}") for i, v in enumerate(variants)]
    def copy_score(t):
        return 0.35 if ("ask" in t and len(t.split()) < 8) else 0.25 if "ask" in t else 0.08
    r = find_best_action(GenerativeText(propose, rounds=1, k=4), world_model(copy_score), seed=0)
    print(f"[GENERATIVE] best copy = \"{r.best.action.value}\"   P={r.best.value:.3f}")

    print("\nOne spine (world model + best-arm racing), four search operators matched to the action type. "
          "The search is general; each world model is its own calibration project.")


if __name__ == "__main__":
    main()
