"""Fit the message-elasticity model on the FULL real CMV persuasion corpus (19,714 labeled outcomes).

This is the calibration step that upgrades the L3 reply evaluator from `grade=unvalidated`
world-knowledge priors to a fit GRADED on held-out real outcomes. Two evaluations run:

  1. backtest_messages — by-PAIR split (a matched pair never straddles train/test), reporting
     ECE / Brier / log-loss / AUC / pair-accuracy vs base rate on held-out real outcomes;
  2. grade_fit — the FittedElasticities production artifact (70/30 split), persisted to
     artifacts/phase13/message_calibration/cmv_fit.json for optimize_message(fit=...).

Honesty note (also stamped into the artifact): the corpus is PERSUASION (CMV delta awards), the
nearest public labeled analogue of "did this message get the desired response". Transporting the
fitted elasticities to cold email is an assumption, recorded as `transport_assumption`; the grade
applies to the fitted domain, and downstream reports must carry both.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from swm.decision.elasticity_fit import grade_fit
from swm.decision.outcome_import import backtest_messages, import_convokit_cmv, to_samples

CORPUS = os.path.join(os.path.dirname(__file__), "..", "..", "data", "phase13_real",
                      "winning-args-corpus", "utterances.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "phase13",
                       "message_calibration")


def main():
    t0 = time.time()
    labeled = import_convokit_cmv(CORPUS)
    n_pos = sum(m.outcome for m in labeled)
    print(f"imported {len(labeled)} labeled challenges ({n_pos} positive) "
          f"in {time.time() - t0:.0f}s", flush=True)
    samples, pairs = to_samples(labeled)                     # lexical encoder over the full corpus
    print(f"encoded {len(samples)} samples in {time.time() - t0:.0f}s", flush=True)

    bt = backtest_messages(samples, pairs, split=0.7)
    print("by-pair backtest:", json.dumps({k: v for k, v in bt.items() if k != "weights"}), flush=True)

    fit = grade_fit(samples, split=0.7, temporal=False)      # corpus import carries no timestamps
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "cmv_fit.json"), "w") as f:
        f.write(fit.to_json())
    with open(os.path.join(OUT_DIR, "cmv_backtest.json"), "w") as f:
        json.dump({
            "dataset": "ConvoKit winning-args (ChangeMyView), Tan et al. 2016",
            "unit": "one challenger argument to an OP",
            "outcome": "OP awarded a delta (real recorded persuasion outcome)",
            "assignment": "observational, matched-pair construction (pair_ids)",
            "n_labeled": len(labeled), "n_positive": n_pos,
            "corpus_sha16": _file_hash16(CORPUS),
            "by_pair_backtest": bt,
            "production_fit_grade": fit.grade,
            "transport_assumption": "persuasion (CMV delta) -> cold-email reply transport is an "
                                    "assumption; the grade applies to the fitted domain",
            "encoder": "lexical (deterministic over the full corpus)",
            "wall_s": round(time.time() - t0, 1),
        }, f, indent=1)
    print(f"fit grade: {fit.grade} | persisted to {OUT_DIR} in {time.time() - t0:.0f}s")


def _file_hash16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


if __name__ == "__main__":
    main()
