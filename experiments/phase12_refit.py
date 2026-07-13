"""Phase 12 — resumable final-integration refit command (Part S / continuation).

Phase 11 (dynamic recompilation) is absent from the base branch, so the Phase-12 calibrators are PROVISIONAL
(fit on a pre-Phase-11 distribution). Once Phase 11 lands and the corpus is regenerated from the full
maximum-capacity path, run THIS command to re-run the entire fit→select→grade→decompose→evaluate freeze and
re-emit all artifacts + hashes. It is idempotent and safe to re-run.

    PYTHONPATH=. python experiments/phase12_refit.py            # rebuild everything from the current corpus
    PYTHONPATH=. python experiments/phase12_refit.py --regen    # also rebuild the corpus from source captures

Before final production grading, the corpus must be regenerated from post-Phase-11 full-system forecasts (set
`maximum_capacity_available=True` in the corpus builder once every subsystem is on the path).
"""
from __future__ import annotations
import argparse, importlib, sys


STEPS = [
    ("experiments.phase12_corpus", "regen-only"),   # only when --regen
    ("experiments.phase12_calibrate", "always"),
    ("experiments.phase12_grade", "always"),
    ("experiments.phase12_uncertainty", "always"),
    ("experiments.phase12_evaluate", "always"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="rebuild the corpus from source captures first")
    args = ap.parse_args()
    for mod, when in STEPS:
        if when == "regen-only" and not args.regen:
            continue
        print(f"\n=== running {mod} ===")
        m = importlib.import_module(mod)
        m.main()
    print("\nphase12 refit complete. NOTE: still PROVISIONAL until the corpus is regenerated from the "
          "post-Phase-11 maximum-capacity path (set maximum_capacity_available=True).")


if __name__ == "__main__":
    sys.exit(main())
