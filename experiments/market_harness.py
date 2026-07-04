"""EXP-012 market harness: fair, no-cheat market comparison + the as-of retrieval question.

Wraps the reusable `swm.eval.market_comparison` around the Manifold fetch (reuses
experiments/manifold_harness for the leakage-controlled snapshot at a fixed lead T). Runs the four
arms the spec names:

  1. no retrieval            : predictor at T, question text only          [agent-swarm file]
  2. raw LLM + as-of retrieval: predictor at T + as-of news context        [BLOCKED-ON-CORPUS]
  3. world model + as-of retr : state model + as-of news context           [BLOCKED-ON-CORPUS]
  4. market @ fixed horizon   : the market's own reconstructed price at T   [live from bet history]

Honest status (unchanged from EXP-006 and confirmed here): the binding constraint is INFORMATION
STALENESS, and the only lever — as-of NEWS retrieval — needs a timestamped news corpus this
environment does not have. The retrieval PLUMBING is built and leakage-tested (swm/retrieval/*,
tests/test_general_swm.py), so arms 2/3 are ready the moment a corpus is wired; they are marked
blocked rather than faked. Arm 1 vs arm 4 reproduces the fair market result through the new module.

Usage:
  python -m experiments.manifold_harness fetch --target 140 --lead-hours 48   # writes data/mf_*.json
  python -m experiments.market_harness score --preds "data/mf_pred_*.json"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from swm.eval.market_comparison import compare
from swm.retrieval.news_context import LiveNewsAdapter

RESULT = "experiments/results/exp012_market.json"


def score(pred_globs):
    truth = json.loads(Path("data/mf_truth.json").read_text())
    preds = {}
    for g in pred_globs:
        for fp in glob.glob(g):
            for p in json.loads(Path(fp).read_text()):
                preds[p["id"]] = min(0.99, max(0.01, p["p_yes"]))
    res = compare(truth, preds)
    print("Fair no-cheat market comparison (market price snapshot at fixed lead T):\n")
    print(f"  {'segment':<28}{'n':>5}{'model_brier':>13}{'market_brier':>14}{'model_wins':>12}")
    for r in res["segments"]:
        print(f"  {r['segment']:<28}{r['n']:>5}{r['model_brier']:>13.4f}{r['market_brier']:>14.4f}"
              f"{r['model_beats_market_frac']:>12.0%}")
    out = {
        "no_retrieval_vs_market": res,
        "raw_llm_plus_asof_retrieval": "BLOCKED-ON-CORPUS: needs a timestamped news source; "
                                       "plumbing ready (swm/retrieval/news_context.py, leakage-tested)",
        "world_model_plus_asof_retrieval": "BLOCKED-ON-CORPUS: same",
        "live_news_adapter_implemented": LiveNewsAdapter.IMPLEMENTED,
        "diagnosis": "binding constraint is information staleness, not reasoning (see EXP-006); "
                     "the market's edge concentrates on markets already information-determined by T; "
                     "on the market-uncertain (information-symmetric) subset the model is near parity.",
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"\n  retrieval arms: BLOCKED-ON-CORPUS (plumbing built + leakage-tested)")
    print(f"  wrote {RESULT}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--preds", nargs="+", required=True)
    a = ap.parse_args()
    score(a.preds)


if __name__ == "__main__":
    main()
