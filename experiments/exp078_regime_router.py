"""EXP-078: the regime router, fit on the real portfolio — decide rich-sim vs baseline per question.

Architecture item #2. Loads the EXP-074 portfolio result, turns each domain into a labeled example (did the
high-fidelity sim beat the baselines?), fits the router (a calibrated logistic over kind + baseline-strength +
irreducibility, shrunk toward a world-knowledge prior so it works with few domains), and shows its routing.

Run: python -m experiments.exp078_regime_router
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.eval.regime_router import RegimeRouter, examples_from_portfolio

PORTFOLIO = "experiments/results/exp074_portfolio_backtest.json"
RESULT = "experiments/results/exp078_regime_router.json"


def run() -> dict:
    port = json.loads(Path(PORTFOLIO).read_text())
    examples = examples_from_portfolio(port)
    router = RegimeRouter().fit(examples)

    probes = [("population", 0.10), ("diffusion", 0.05), ("institution", 0.40),
              ("election", 0.60), ("macro", 0.80), ("referendum", 0.55), ("market", 0.95)]
    routes = {kind: router.route(kind, baseline_strength=bs) for kind, bs in probes}
    res = {"training_examples": examples, "routes": routes}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print("EXP-078  regime router fit on the portfolio (rich-sim vs baseline per question)")
    print("  training examples (kind, fidelity-won?, baseline-strength):")
    for e in examples:
        print(f"    {e['domain']:12s} {e['kind']:11s} won={e['y']}  baseline_strength={e['baseline_strength']}")
    print("  routing decisions:")
    for kind, r in routes.items():
        print(f"    {kind:11s} -> {r['decision']:8s}  P(fidelity wins)={r['p_fidelity_wins']:.2f}  ({r['reason']})")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
