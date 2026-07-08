"""EXP-074: the data-scaling program, step 1 — the full paired CMV corpus + a learned readout at scale.

EXP-073 showed the best-message ceiling is real (~0.75-0.83) but we were stuck at ~0.66 because 138 examples
is too few to LEARN it. This pulls the FULL Tan-et-al. paired corpus (4,263 matched winner/loser argument
pairs to the same OP — 31x more data) and asks the honest questions:

  A. DOES MORE DATA CLIMB? A data-scaling curve: train the learned readout (pure-python logistic, CPU-only)
     on N pairs (138 -> full) and score PAIRED accuracy (does it rank the winning argument above the losing
     one?) on a FIXED held-out set. If accuracy rises with N, more data is the lever, as predicted.
  B. LEXICAL vs the LLM. The lexical readout (word-overlap-with-OP, evidence, hedging, tone — the features
     Tan et al. used; literature ceiling ~0.65) vs DeepSeek's holistic pairwise judgment ("which argument
     better changes THIS person's mind?") on a subset — the immediate free win, at scale.

Paired accuracy is the honest KPI here (pick the winner of two matched messages; random = 0.50). Leakage-
free: split by PAIR, features from the argument+OP text only. DeepSeek judgments cached (key from env).
Run: DEEPSEEK_API_KEY=... python -m experiments.exp074_cmv_scale
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from swm.transition.readout import LogisticReadout

PAIRS = "experiments/results/exp074/cmv_pairs.json"
DS_CACHE = "experiments/results/exp074/deepseek_pairwise.json"
RESULT = "experiments/results/exp074_cmv_scale.json"

_EVID = re.compile(r"\b(evidence|study|data|source|research|statistic|because|therefore|for example|e\.g\.)\b", re.I)
_HEDGE = re.compile(r"\b(i think|maybe|perhaps|might|could be|arguably|it seems|possibly)\b", re.I)
_ABSOL = re.compile(r"\b(always|never|everyone|no one|obviously|clearly|definitely)\b", re.I)
_HOSTILE = re.compile(r"\b(stupid|idiot|nonsense|ridiculous|dumb|absurd)\b", re.I)
_POLITE = re.compile(r"\b(fair point|good point|i appreciate|you're right|i understand|i see)\b", re.I)
_WORD = re.compile(r"[a-z']+")


def _feats(arg, op):
    """Pure-python lexical features for one argument, incl. its interplay with the OP (Tan et al.: LOW
    word-overlap with the OP tends to win — don't just echo them)."""
    a = arg.lower()
    aw = _WORD.findall(a)
    n = max(1, len(aw))
    opw = set(_WORD.findall(op.lower()))
    aset = set(aw)
    jacc = len(aset & opw) / max(1, len(aset | opw))
    return [
        math.log1p(len(aw)) / 8.0,                        # length
        len(_EVID.findall(a)) / 3.0,                      # evidence markers
        len(_HEDGE.findall(a)) / 3.0,                     # hedging / epistemic humility
        len(_ABSOL.findall(a)) / 3.0,                     # absolutes (hurt)
        (len(_POLITE.findall(a)) - len(_HOSTILE.findall(a)) + 3) / 6.0,   # tone
        a.count("?") / 5.0,                               # questions
        a.count("http") / 2.0,                            # links / sources
        a.count(">") / 5.0,                               # quoting the OP
        jacc,                                             # word overlap with OP (lower wins)
        len(aset) / n,                                    # lexical diversity
    ]


def _paired_acc(model, test):
    """Fraction of test pairs where the winner's score exceeds the loser's."""
    hits = 0
    for p in test:
        if model.predict_proba(_feats(p["pos"], p["op_text"])) > model.predict_proba(_feats(p["neg"], p["op_text"])):
            hits += 1
    return hits / len(test)


def _train(pairs):
    X = [_feats(p["pos"], p["op_text"]) for p in pairs] + [_feats(p["neg"], p["op_text"]) for p in pairs]
    y = [1] * len(pairs) + [0] * len(pairs)
    return LogisticReadout(l2=0.5, epochs=300).fit(X, y)


def _deepseek_pairwise(subset):
    Path(DS_CACHE).parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(Path(DS_CACHE).read_text()) if Path(DS_CACHE).exists() else {}
    todo = [p for p in subset if p["pair"] not in cache]
    if todo and os.environ.get("DEEPSEEK_API_KEY"):
        from swm.api.deepseek_backend import deepseek_chat_fn
        fn = deepseek_chat_fn(system="You judge which argument better changes a specific person's mind. "
                                     "Answer with ONLY 'A' or 'B'.", max_tokens=5)
        import random as _r
        rng = _r.Random(0)
        for k, p in enumerate(todo):
            swap = rng.random() < 0.5                      # randomize order to kill position bias
            a, b = (p["neg"], p["pos"]) if swap else (p["pos"], p["neg"])
            prompt = (f"A person holds this view:\n{p['op_text'][:600]}\n\nArgument A:\n{a[:800]}\n\n"
                      f"Argument B:\n{b[:800]}\n\nWhich argument is more likely to change their mind? Answer A or B.")
            try:
                ans = fn(prompt).strip().upper()[:1]
                winner_is_A = (ans == "A")
                cache[p["pair"]] = 1 if (winner_is_A != swap) else 0   # 1 => picked the true winner (pos)
            except Exception as e:
                print(f"  deepseek stopped at {k}/{len(todo)}: {str(e)[:60]}")
                break
            if k % 20 == 0:
                Path(DS_CACHE).write_text(json.dumps(cache))
        Path(DS_CACHE).write_text(json.dumps(cache))
    return cache


def run():
    pairs = json.loads(Path(PAIRS).read_text())
    cut = int(0.75 * len(pairs))
    train_all, test = pairs[:cut], pairs[cut:]

    # A. data-scaling curve on a fixed held-out set
    curve = {}
    for n in (138, 400, 1000, 2000, len(train_all)):
        m = _train(train_all[:n])
        curve[n] = round(_paired_acc(m, test), 4)
    full_acc = curve[len(train_all)]

    # B. DeepSeek holistic pairwise judgment on a subset of the test set (the immediate LLM win)
    sub = test[:600]
    ds = _deepseek_pairwise(sub)
    ds_scored = [p for p in sub if p["pair"] in ds]
    ds_acc = (sum(ds[p["pair"]] for p in ds_scored) / len(ds_scored)) if ds_scored else None

    out = {"data": "full Tan-et-al. paired CMV corpus (ConvoKit winning-args)", "n_pairs": len(pairs),
           "n_train": len(train_all), "n_test": len(test),
           "A_scaling_curve_paired_acc": {str(k): v for k, v in curve.items()},
           "lexical_readout_full": full_acc,
           "vs_138_scale": round(full_acc - curve[138], 4),
           "literature_baseline_tan2016": 0.65,
           "B_deepseek_pairwise": {"n_scored": len(ds_scored), "paired_acc": round(ds_acc, 4) if ds_acc else None},
           "random_baseline": 0.5}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-074  data-scaling step 1: full paired CMV corpus + learned readout at scale")
    print(f"  {len(pairs)} matched winner/loser pairs (was 138) | train {len(train_all)} / test {len(test)}")
    print("  A. DOES MORE DATA CLIMB? (lexical readout, paired accuracy on fixed held-out set)")
    for k, v in curve.items():
        bar = "#" * int((v - 0.5) * 200)
        print(f"       train n={k:5d}  paired acc = {v:.4f}  {bar}")
    print(f"     -> full-scale {full_acc:.4f} vs 138-scale {curve[138]:.4f}  ({out['vs_138_scale']:+})  "
          f"| literature (Tan 2016) ~0.65 | random 0.50")
    if ds_acc:
        print(f"  B. DEEPSEEK holistic pairwise ('which changes this mind?'): {ds_acc:.4f} on {len(ds_scored)} "
              f"pairs (zero training)")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
