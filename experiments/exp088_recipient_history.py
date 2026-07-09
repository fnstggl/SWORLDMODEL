"""EXP-088 — recipient conditioning from history, measured honestly.

Two findings, one negative and one positive, both real:

A. ON CMV (real data): recipient-from-history does NOT improve pair-accuracy, and the honest reason is
   structural — the CMV "winning arguments" PAIR task compares two arguments sent to the SAME OP, so it
   CONTROLS FOR THE RECIPIENT by construction (per-OP success-rate std across OPs is only ~0.14, and both
   members of a pair share the OP, so any per-recipient base rate cancels in the within-pair comparison).
   Recipient conditioning cannot help a metric that fixes the recipient. So the earlier claim that it would
   "break 0.56" was wrong for THIS metric — an honest correction. The conditioning DID de-compress the base
   rate (per-OP base spans ~0.08–0.65), it just can't show up in pair accuracy.

B. THE METHOD IS CORRECT (synthetic validation): when recipients GENUINELY differ (as they do for real cold
   outreach — different people reply at different rates), conditioning on their history recovers the
   recipient effect and improves CROSS-recipient ranking, while a constant-recipient model cannot. This is
   the estimator-validation stance the repo uses elsewhere.

The lesson: recipient conditioning is validated by a MULTI-recipient task, not CMV's controlled pair task.
The real unlock remains a multi-recipient corpus (a sent→replied cold-email set) — the `HistoryStore`
primitive is ready for it.

Run:  PYTHONPATH=. python experiments/exp088_recipient_history.py [path/to/utterances.jsonl]
"""
from __future__ import annotations

import math
import random
import statistics
import sys

from swm.decision.outcome_import import _auc
from swm.decision.recipient_history import HistoryStore
from swm.variables.deep_inference import DeepInferenceEngine


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))


def synthetic_multi_recipient(n_recip=300, per=8, seed=0):
    """Recipients genuinely differ in persuadability, and it shows in their writing history (open, humble
    writers are more persuadable). Each gets `per` messages of varying quality. Show that conditioning on
    history beats a constant recipient at CROSS-recipient ranking."""
    rng = random.Random(seed)
    store = HistoryStore(engine=DeepInferenceEngine())
    truth_base = {}
    rows = []
    OPEN = ["I think maybe I could be wrong about this, good point, fair enough, I appreciate your view",
            "you make a fair point and I might be mistaken, perhaps, I could reconsider"]
    ENTRENCHED = ["This is obviously true, everyone knows it, I will never change my mind, absolutely certain",
                  "definitely correct, no one can argue, the fact is undeniable, always been this way"]
    for r in range(n_recip):
        openness = rng.random()
        # history reflects disposition
        for _ in range(6):
            store.ingest(f"r{r}", rng.choice(OPEN if openness > 0.5 else ENTRENCHED), ts=None)
        base = 0.15 + 0.6 * openness            # persuadability genuinely varies by recipient
        truth_base[r] = base
        for _ in range(per):
            quality = rng.random()               # message quality
            p = _sigmoid(4 * (base - 0.5) + 1.5 * (quality - 0.5))
            rows.append((r, quality, 1 if rng.random() < p else 0))
    return store, rows, truth_base


def synthetic_demo():
    print("[B] SYNTHETIC multi-recipient validation (recipients genuinely differ)")
    store, rows, _ = synthetic_multi_recipient()
    # feature = [quality] ; recipient conditioning adds the history-derived base as a feature
    def fit(feats, y, epochs=300, lr=0.3):
        w = [0.0] * len(feats[0]); b = 0.0; n = len(y)
        for _ in range(epochs):
            gb = 0.0; gw = [0.0] * len(w)
            for f, yi in zip(feats, y):
                p = _sigmoid(b + sum(wi * fi for wi, fi in zip(w, f))); e = p - yi; gb += e
                for j in range(len(w)):
                    gw[j] += e * f[j]
            b -= lr * gb / n
            for j in range(len(w)):
                w[j] -= lr * (gw[j] / n + 0.5 / n * w[j])
        return w, b
    idx = list(range(len(rows))); random.Random(1).shuffle(idx)
    cut = int(0.7 * len(idx)); tr = [rows[i] for i in idx[:cut]]; te = [rows[i] for i in idx[cut:]]

    def run(conditioned):
        def feat(r, q):
            if not conditioned:
                return [q]
            _, base = store.recipient(f"r{r}")
            return [q, base]                      # history-derived recipient base as a feature
        w, b = fit([feat(r, q) for r, q, _ in tr], [o for _, _, o in tr])
        preds = [_sigmoid(b + sum(wi * fi for wi, fi in zip(w, feat(r, q)))) for r, q, _ in te]
        y = [o for _, _, o in te]
        return _auc(y, preds), (min(preds), max(preds))
    for cond in (False, True):
        auc, (lo, hi) = run(cond)
        tag = "history-conditioned" if cond else "constant recipient  "
        print(f"   {tag}: cross-recipient AUC={auc:.3f}  pred range [{lo:.2f},{hi:.2f}]")


def cmv_finding(path):
    import json
    from collections import defaultdict
    from swm.decision.outcome_import import LabeledMessage, to_samples, backtest_messages, CMV_RECIPIENT
    print("[A] CMV real data — recipient conditioning vs constant (pair task CONTROLS for recipient)")
    author_docs = defaultdict(list); root_author = {}; labeled = []
    with open(path) as f:
        for ln in f:
            d = json.loads(ln); au = d.get("user"); ts = d.get("timestamp") or 0
            author_docs[au].append((ts, d.get("text", "")))
            if d.get("id") == d.get("root"):
                root_author[d["id"]] = au
            m = d.get("meta") or {}
            if m.get("success") in (0, 1) and (d.get("text") or "").strip():
                pid = (m.get("pair_ids") or [d.get("root")])[0]
                labeled.append((d.get("root"), d.get("text"), int(m["success"]), pid, ts))
    store = HistoryStore(engine=DeepInferenceEngine())
    ops = {root_author.get(r) for r, *_ in labeled if root_author.get(r)}
    for au in ops:
        for ts, text in author_docs.get(au, []):
            store.ingest(au, text, ts)
    const, hist = [], []
    for root, arg, outcome, pair, ts in labeled:
        au = root_author.get(root)
        const.append(LabeledMessage(arg, outcome, dict(CMV_RECIPIENT), pair_id=pair))
        rv, base = store.recipient(au, now=ts) if au else (dict(CMV_RECIPIENT), None)
        hist.append(LabeledMessage(arg, outcome, rv, pair_id=pair, base=base))
    for tag, msgs in [("constant   ", const), ("history-cond", hist)]:
        s, p = to_samples(msgs); res = backtest_messages(s, p, split=0.7)
        print(f"   {tag}: pair_acc={res['pair_accuracy']}  auc={res['auc']}")
    bases = [m.base for m in hist if m.base is not None]
    print(f"   per-OP base rate now spans [{min(bases):.2f}, {max(bases):.2f}] (was constant) — de-compressed,")
    print("   but it CANCELS within a pair, so it can't move pair-accuracy. Wrong instrument, not wrong method.")


def main():
    print("=" * 80)
    print("EXP-088  recipient conditioning from history (deep_inference)")
    print("=" * 80)
    if len(sys.argv) > 1:
        cmv_finding(sys.argv[1]); print()
    else:
        print("(no CMV path given; skipping the real-data finding — see the module docstring)\n")
    synthetic_demo()


if __name__ == "__main__":
    main()
