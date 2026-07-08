"""The outcome-data import path: import labeled rows -> encode -> fit -> grade on held-out REAL outcomes.

Tested here with a small synthetic-but-realistic labeled corpus (no external download); the same path runs
the real 20k-message CMV corpus in experiments/exp087. Validates: general import, ConvoKit CMV parsing,
sample building, AUC/pair-accuracy, and that a learnable signal is recovered on held-out data.
"""
import json
import math

from swm.decision.outcome_import import (LabeledMessage, _auc, backtest_messages, import_convokit_cmv,
                                          import_rows, to_samples)
from swm.decision.strategy_scorer import MESSAGE_VARS


def test_import_rows_general():
    rows = [{"text": "hi", "replied": True, "pair_id": "p1"},
            {"text": "yo", "replied": 0, "recipient": {"skepticism": 0.9}}]
    msgs = import_rows(rows)
    assert len(msgs) == 2 and msgs[0].outcome == 1 and msgs[1].outcome == 0
    assert msgs[1].recipient_vars["skepticism"] == 0.9


def test_import_convokit_cmv(tmp_path):
    # minimal ConvoKit-shaped utterances: an OP + a winning (delta) and losing reply
    lines = [
        {"id": "t3_op", "root": "t3_op", "text": "change my view on X", "meta": {"success": None}},
        {"id": "t1_a", "root": "t3_op", "text": "here is evidence and a clear point", "meta": {"success": 1, "pair_ids": ["pair1"]}},
        {"id": "t1_b", "root": "t3_op", "text": "you are just wrong idiot", "meta": {"success": 0, "pair_ids": ["pair1"]}},
    ]
    p = tmp_path / "utt.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    msgs = import_convokit_cmv(str(p))
    assert len(msgs) == 2                              # only the two labeled replies
    assert {m.outcome for m in msgs} == {0, 1}
    assert all(m.pair_id == "pair1" for m in msgs)
    assert "change my view" in msgs[0].recipient_context


def test_to_samples_shape():
    msgs = [LabeledMessage("a clear concrete message with a number 5", 1, {"skepticism": 0.5}, pair_id="p"),
            LabeledMessage("vague fluff", 0, {"skepticism": 0.5}, pair_id="p")]
    samples, pairs = to_samples(msgs)
    assert len(samples) == 2 and pairs == ["p", "p"]
    recipient, strat, base, outcome = samples[0]
    assert set(MESSAGE_VARS).issubset(strat.keys()) and 0.0 <= base <= 1.0


def test_auc_ranks_correctly():
    assert _auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0     # perfect separation
    assert abs(_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9   # ties -> chance


def test_backtest_recovers_signal_on_labeled_data():
    # build a labeled corpus where clarity/proof genuinely drive the outcome, then check the held-out
    # backtest beats chance and grades sensibly.
    import random
    rng = random.Random(0)
    rv = {"skepticism": 0.7, "platform_response_norm": 0.5}
    msgs = []
    for i in range(400):
        # "good" messages: concrete number + clear; "bad": vague fluff
        good = rng.random() < 0.5
        text = ("We grew 40% and have 1000 paying users. One clear point." if good
                else "This is some vague fluff with no specifics whatsoever really.")
        # outcome correlated with good-ness (noisy)
        outcome = 1 if (good and rng.random() < 0.75) or (not good and rng.random() < 0.3) else 0
        msgs.append(LabeledMessage(text, outcome, dict(rv), pair_id=f"pair{i//2}"))
    samples, pairs = to_samples(msgs)
    res = backtest_messages(samples, pairs, split=0.7)
    assert "error" not in res
    assert res["auc"] > 0.55                          # recovers real signal above chance
    assert res["grade"] in ("A", "B", "C")
