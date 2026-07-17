"""CLI: python -m historical_backtests.tools.run_benchmark --benchmark openrouter_llama31_v1
        --runtime current --split reusable_regression|rotating_holdout|all [--limit N]

reusable_regression = calibration+validation splits (rerunnable forever; results labeled
REUSABLE_DEVELOPMENT_BACKTEST). rotating_holdout = the rotating_locked split (forecasting is
allowed any time; OUTCOME opening is one-time, scorer-side)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--runtime", default="current", help="'current' = this working tree/commit")
    ap.add_argument("--split", default="reusable_regression",
                    choices=["reusable_regression", "rotating_holdout", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args()
    if a.runtime != "current":
        raise SystemExit("only --runtime current is supported (check out the commit first; "
                         "results are namespaced by commit automatically)")
    from historical_backtests.framework.runner import run_benchmark
    splits = {"reusable_regression": ("calibration", "validation"),
              "rotating_holdout": ("rotating_locked",),
              "all": ("calibration", "validation", "rotating_locked")}[a.split]
    run_benchmark(a.benchmark, splits=splits, limit=a.limit,
                  with_baselines=not a.no_baselines)


if __name__ == "__main__":
    main()
