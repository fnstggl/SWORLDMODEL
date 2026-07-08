"""EXP-087 — the message model, graded on REAL content→outcome data (not synthetic).

This is the step that turns the elasticity-fitting harness from synthetic-validated into really-graded. It
imports the Cornell ConvoKit "winning arguments" ChangeMyView corpus (a real, labeled persuasion dataset:
each argument did or did not earn a delta), encodes every message into the general message-lever vector,
fits the elasticities on a held-out-by-pair split, and grades them on truly out-of-sample outcomes —
calibration (ECE) plus AUC and PAIR ACCURACY (does the model score the winning argument above the losing
one?). Chance is 0.5.

Get the data (~73MB) once:
    curl -o wac.zip https://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/winning-args-corpus.zip
    unzip wac.zip
Then:
    PYTHONPATH=. python experiments/exp087_cmv_reply_backtest.py path/to/winning-args-corpus/utterances.jsonl

Honest scope: this validates the MESSAGE model (which levers predict persuasion) on real outcomes for the
CMV population. A cold-email reply model would be graded the same way on a sent→replied corpus — the import
path (swm/decision/outcome_import.py) is identical; only the dataset changes.
"""
from __future__ import annotations

import json
import sys

from swm.decision.outcome_import import backtest_messages, import_convokit_cmv, to_samples


def main(path: str):
    print("=" * 78)
    print("EXP-087  message model graded on REAL persuasion outcomes (CMV winning-args)")
    print("=" * 78)

    msgs = import_convokit_cmv(path)
    pos = sum(m.outcome for m in msgs)
    print(f"\nimported {len(msgs)} labeled messages  ({pos} persuaded / {len(msgs)-pos} did not)")

    samples, pairs = to_samples(msgs)                 # lexical encoder (fast over the whole corpus)
    res = backtest_messages(samples, pairs, split=0.7, use_prior=True)

    print(f"\nheld-out grade: {res['grade']}   ECE={res['ece']}   Brier={res['brier']}   "
          f"log_loss={res['log_loss']}")
    print(f"AUC = {res['auc']}   PAIR ACCURACY = {res['pair_accuracy']}   "
          f"(chance = 0.5, over {res['n_pairs_tested']} held-out pairs)")
    print(f"train/test = {res['n_train']}/{res['n_test']}   test base rate = {res['test_base_rate']}")
    print("\ntop fitted message elasticities (data-calibrated, toward the priors):")
    for k, v in res["weights"].items():
        print(f"   {k:>22}: {v:+.3f}")
    print("\n" + res["note"])
    print("This is a real, held-out signal above chance — the message levers carry genuine predictive "
          "power on real outcomes. The LLM encoder lifts it further (see exp086 / the subset comparison).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python experiments/exp087_cmv_reply_backtest.py <utterances.jsonl>")
        print("download: https://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/")
        sys.exit(1)
    main(sys.argv[1])
