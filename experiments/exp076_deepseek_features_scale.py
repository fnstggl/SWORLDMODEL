"""EXP-076: does DeepSeek RICH-feature extraction + more data climb the product KPI toward 0.83?

The chain so far:
  - EXP-075: with pure LEXICAL features, the product KPI (pick the best of an OP's candidate arguments)
    climbs with data and saturates ~0.63. Lexical features cap it.
  - EXP-073: DeepSeek rich features raised the IN-SAMPLE ceiling to 0.83, but on 64 OPs leave-one-out was
    STUCK at 0.656 — too few examples to LEARN the richer mapping (it overfit).

The open question this resolves: at LARGER scale, do the DeepSeek rich features finally pay off on a
HELD-OUT set — i.e. does lexical+DeepSeek climb above the 0.63 lexical ceiling toward 0.83?

Design: extract DeepSeek's 10 persuasion-grounded scores for each argument on a 950-OP subset (700 train
/ 250 test, deterministic slice matching EXP-075's split), cached & resumable (key from env only). Then
train the learned readout (pure-python logistic, CPU) three ways on the SAME train/test OPs and score
precision@1 on held-out OPs:
    lexical (10)  |  deepseek (10)  |  lexical+deepseek (20)
plus DeepSeek's holistic persuasive_force used directly as the ranker (zero training).

Leakage-free: split by OP. Cache: experiments/results/exp076/deepseek_argfeats.json.
Run: DEEPSEEK_API_KEY=... python -m experiments.exp076_deepseek_features_scale
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from swm.transition.readout import LogisticReadout
from experiments.exp074_cmv_scale import _feats

PEROP = "experiments/results/exp075/cmv_perop.json"
DS_CACHE = "experiments/results/exp076/deepseek_argfeats.json"
RESULT = "experiments/results/exp076_deepseek_features_scale.json"

DS_FEATURES = ["addresses_ops_actual_reasoning", "concrete_evidence_or_examples", "respectful_calm_tone",
               "introduces_new_perspective", "epistemic_humility", "directly_rebuts_key_point",
               "concreteness", "personal_relatability", "reframes_the_issue", "persuasive_force"]


def _subset(ops):
    """450 train OPs + 200 test OPs — deterministic slice aligned to EXP-075's 75/25 split (cut=2288).
    Sized to fully extract within one session so the held-out set is COMPLETELY DeepSeek-scored (a partial
    extraction that never reaches the test OPs silently defaults them to 0.5 and invalidates the eval)."""
    return ops[:450], ops[2288:2488]


def _arg_key(op_id, i):
    return f"{op_id}#{i}"


def _extract(train, test):
    Path(DS_CACHE).parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(Path(DS_CACHE).read_text()) if Path(DS_CACHE).exists() else {}
    # TEST OPs first so a partial run still fully covers the held-out set (a half-scored test set is worse
    # than none — it silently defaults to 0.5 and invalidates the comparison).
    items = [(o, i, a) for o in test + train for i, a in enumerate(o["args"])]
    todo = [(o, i, a) for (o, i, a) in items if _arg_key(o["op_id"], i) not in cache]
    if todo and os.environ.get("DEEPSEEK_API_KEY"):
        from swm.api.deepseek_backend import deepseek_chat_fn
        fn = deepseek_chat_fn(system="You are an expert on what makes an argument change someone's mind. "
                                     "Score only from the texts. Return ONLY JSON.", max_tokens=300)
        for k, (o, i, a) in enumerate(todo):
            prompt = (f"OP's stated view:\n{o['op_text'][:700]}\n\nA challenger's argument:\n{a['text'][:900]}\n\n"
                      f"Rate this argument 0.0-1.0 on each dimension for whether it would change THIS OP's mind:\n"
                      + ", ".join(DS_FEATURES) +
                      '\nReturn ONLY JSON mapping each key to a number, e.g. {"persuasive_force":0.6, ...}')
            for attempt in range(5):                       # retry transient network errors (reset/timeout)
                try:
                    raw = fn(prompt)
                    obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                    cache[_arg_key(o["op_id"], i)] = {f: float(obj.get(f, 0.5)) for f in DS_FEATURES}
                    break
                except Exception as e:
                    if attempt == 4:
                        print(f"  deepseek gave up at {k}/{len(todo)}: {str(e)[:70]}")
                    else:
                        time.sleep(2 ** attempt)
            else:
                continue
            if k % 25 == 0:
                Path(DS_CACHE).write_text(json.dumps(cache))
                print(f"  extracted {k}/{len(todo)}")
        Path(DS_CACHE).write_text(json.dumps(cache))
    return cache


def _lex(o, i, a):
    return _feats(a["text"], o["op_text"])


def _ds(cache, o, i, a):
    d = cache.get(_arg_key(o["op_id"], i), {k: 0.5 for k in DS_FEATURES})
    return [d[k] for k in DS_FEATURES]


def _train(ops, feat):
    X = [feat(o, i, a) for o in ops for i, a in enumerate(o["args"])]
    y = [a["success"] for o in ops for a in o["args"]]
    return LogisticReadout(l2=0.5, epochs=300).fit(X, y)


def _p1(model, ops, feat):
    hits = rand = 0
    for o in ops:
        top = max(range(len(o["args"])), key=lambda i: model.predict_proba(feat(o, i, o["args"][i])))
        hits += o["args"][top]["success"]
        rand += sum(a["success"] for a in o["args"]) / len(o["args"])
    return hits / len(ops), rand / len(ops)


def _p1_direct(ops, score):
    """Rank by a raw per-arg score (no training)."""
    hits = 0
    for o in ops:
        top = max(range(len(o["args"])), key=lambda i: score(o, i, o["args"][i]))
        hits += o["args"][top]["success"]
    return hits / len(ops)


def run():
    ops = json.loads(Path(PEROP).read_text())
    train, test = _subset(ops)
    cache = _extract(train, test)
    n_have = sum(1 for o in train + test for i in range(len(o["args"])) if _arg_key(o["op_id"], i) in cache)
    n_need = sum(len(o["args"]) for o in train + test)
    # GUARD: the held-out comparison is only valid if EVERY test-OP argument is DeepSeek-scored — otherwise
    # unscored args default to 0.5 and the argmax degenerates. Refuse to report a spurious number.
    test_missing = [(o["op_id"], i) for o in test for i in range(len(o["args"]))
                    if _arg_key(o["op_id"], i) not in cache]
    if test_missing:
        print(f"  INCOMPLETE: {len(test_missing)} test-OP args unscored ({n_have}/{n_need} total). "
              f"Re-run to finish extraction (resumes from cache) before trusting the numbers.")
        return {"status": "incomplete", "n_args_scored": n_have, "n_args_needed": n_need,
                "test_args_unscored": len(test_missing)}

    lex = lambda o, i, a: _lex(o, i, a)
    ds = lambda o, i, a: _ds(cache, o, i, a)
    both = lambda o, i, a: _lex(o, i, a) + _ds(cache, o, i, a)

    p_lex, rand = _p1(_train(train, lex), test, lex)
    p_ds, _ = _p1(_train(train, ds), test, ds)
    p_both, _ = _p1(_train(train, both), test, both)
    p_force = _p1_direct(test, lambda o, i, a: cache.get(_arg_key(o["op_id"], i), {}).get("persuasive_force", 0.5))

    out = {"subset": "700 train / 250 test OPs (aligned to EXP-075 split)",
           "n_args_scored": n_have, "n_args_needed": n_need,
           "kpi": "precision@1 on held-out OPs — top-ranked candidate is a delta winner?",
           "random_pick_baseline": round(rand, 4),
           "lexical_only": round(p_lex, 4),
           "deepseek_only": round(p_ds, 4),
           "lexical_plus_deepseek": round(p_both, 4),
           "deepseek_persuasive_force_direct_no_training": round(p_force, 4),
           "best_vs_lexical": round(max(p_ds, p_both, p_force) - p_lex, 4),
           "exp075_lexical_full_scale": 0.6317,
           "exp073_in_sample_ceiling_with_deepseek": 0.828}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-076  DeepSeek rich features + data: does the product KPI climb toward 0.83? (held-out, 250 test OPs)")
    print(f"  args scored by DeepSeek: {n_have}/{n_need}")
    print(f"  random pick baseline:                    {rand:.4f}")
    print(f"  lexical features only (trained):         {p_lex:.4f}")
    print(f"  DeepSeek features only (trained):        {p_ds:.4f}")
    print(f"  lexical + DeepSeek (trained):            {p_both:.4f}   <- the key number")
    print(f"  DeepSeek persuasive_force direct (none): {p_force:.4f}")
    print(f"  -> best over lexical: {out['best_vs_lexical']:+.4f}  | in-sample ceiling (EXP-073) 0.83")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
