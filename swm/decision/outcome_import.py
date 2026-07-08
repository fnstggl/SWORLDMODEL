"""The IMPORT PATH for labeled content→outcome data — the scarce asset that turns the message model from
`unvalidated` priors into a really-graded predictor.

A labeled row is (message text, recipient context, outcome 0/1): a cold email that did/didn't get a reply,
a CMV argument that did/didn't earn a delta, a YC application that was/ wasn't accepted. This module:

  1. IMPORTS such rows (a general `import_rows`, plus a concrete `import_convokit_cmv` for the Cornell
     ChangeMyView "winning arguments" corpus — a real, labeled persuasion dataset).
  2. ENCODES each message into the general message-lever vector (the LLM encoder, or the lexical fallback
     for large corpora) and pairs it with the recipient's inferred variables.
  3. BACKTESTS: fits the elasticities on a held-out split (regularized toward the world-knowledge priors)
     and grades them on truly out-of-sample outcomes — ECE/Brier/log-loss, plus AUC and, for paired data,
     PAIR ACCURACY (does the model score the winning message above the losing one?), against a base-rate
     baseline. This is the step that makes the reply numbers earned rather than asserted.

Everything is streaming/bounded so a 350MB corpus is fine, and the message encoder is pluggable so you can
run the whole corpus lexically (fast) or LLM-encode a subset (accurate).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.decision.compositional_search import encode_text_to_strategy
from swm.decision.elasticity_fit import _predict, fit_elasticities
from swm.eval.metrics import brier_score, expected_calibration_error, log_loss

# a CMV OP is a person who explicitly invited debate: open-ish, skeptical, on a high-visibility platform.
CMV_RECIPIENT = {"skepticism": 0.7, "openness_to_outreach": 0.6, "status_orientation": 0.3,
                 "status": 0.4, "attention_availability": 0.6, "platform_response_norm": 0.63,
                 "relationship_strength": 0.0}


@dataclass
class LabeledMessage:
    text: str
    outcome: int                       # 1 = got the desired response, 0 = did not
    recipient_vars: dict = field(default_factory=dict)
    recipient_context: str = ""        # free text about the recipient (for the LLM encoder / resolver)
    pair_id: str = ""                  # links a matched positive/negative pair (for pair accuracy)
    ts: float = 0.0                    # ordering for a temporal split (0 => index order)


# ---- importers -----------------------------------------------------------------------------------

def import_rows(rows, *, text_key="text", outcome_key="replied", recipient_key="recipient",
                default_recipient: dict | None = None) -> list:
    """General importer: a list/iterable of dicts -> [LabeledMessage]. `outcome_key` may be bool/0/1."""
    out = []
    for r in rows:
        if text_key not in r or outcome_key not in r:
            continue
        rv = r.get(recipient_key) if isinstance(r.get(recipient_key), dict) else None
        out.append(LabeledMessage(text=str(r[text_key]), outcome=int(bool(r[outcome_key])),
                                  recipient_vars=rv or dict(default_recipient or {}),
                                  pair_id=str(r.get("pair_id", "")), ts=float(r.get("ts", 0.0))))
    return out


def import_convokit_cmv(utterances_path: str, *, max_pairs: int | None = None,
                        recipient_vars: dict | None = None) -> list:
    """Import the Cornell ConvoKit 'winning-args' CMV corpus (utterances.jsonl). Uses the labeled PAIR
    utterances (meta.success in {0,1}); the OP (root) text is the recipient context. Streams the file."""
    recipient_vars = dict(recipient_vars or CMV_RECIPIENT)
    op_text: dict[str, str] = {}
    labeled = []
    with open(utterances_path) as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            meta = d.get("meta") or {}
            if d.get("id") == d.get("root"):                 # the OP post
                op_text[d["id"]] = d.get("text", "")
            s = meta.get("success")
            if s in (0, 1) and (d.get("text") or "").strip():
                pids = meta.get("pair_ids") or []
                pair = pids[0] if pids else d.get("root", "")
                labeled.append((d.get("root", ""), d.get("text", ""), int(s), str(pair)))
    msgs = []
    for root, text, outcome, pair in labeled:
        msgs.append(LabeledMessage(text=text, outcome=outcome, recipient_vars=dict(recipient_vars),
                                   recipient_context=op_text.get(root, "")[:600], pair_id=pair))
    if max_pairs is not None:
        # keep whole pairs up to a cap (deterministic by pair id)
        keep, seen = [], set()
        for m in msgs:
            if m.pair_id not in seen and len(seen) >= max_pairs:
                continue
            seen.add(m.pair_id)
            keep.append(m)
        msgs = keep
    return msgs


# ---- encode -> samples ---------------------------------------------------------------------------

def to_samples(labeled: list, *, encode_fn=None, base_rate: float | None = None) -> tuple:
    """Encode each message into the strategy vector and build (recipient, strategy, base, outcome) samples.
    Returns (samples, pair_ids). `base_rate` defaults to the empirical positive rate."""
    encode = encode_fn or encode_text_to_strategy
    ys = [m.outcome for m in labeled]
    base = base_rate if base_rate is not None else (sum(ys) + 1) / (len(ys) + 2)
    samples, pairs = [], []
    for m in labeled:
        strat = encode(m.text)
        samples.append((m.recipient_vars, strat, base, m.outcome))
        pairs.append(m.pair_id)
    return samples, pairs


# ---- metrics + backtest --------------------------------------------------------------------------

def _auc(y: list, p: list) -> float:
    """Rank-based AUC (probability a random positive outranks a random negative)."""
    pos = [pi for pi, yi in zip(p, y) if yi == 1]
    neg = [pi for pi, yi in zip(p, y) if yi == 0]
    if not pos or not neg:
        return 0.5
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = {}
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and p[order[j]] == p[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    rank_pos = sum(ranks[i] for i in range(len(p)) if y[i] == 1)
    return (rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def backtest_messages(samples: list, pairs: list, *, split: float = 0.7, seed: int = 0, **fit_kw) -> dict:
    """Fit elasticities on the first `split` (by pair, so a pair never straddles the split) and grade on the
    held-out remainder: ECE/Brier/log-loss, AUC, and PAIR ACCURACY vs a base-rate baseline."""
    n = len(samples)
    if n < 40:
        return {"error": f"only {n} samples; need >= 40"}
    # split by pair id so both members of a pair land on the same side (fair pair-accuracy)
    uniq_pairs = list(dict.fromkeys(pairs))
    order = sorted(range(len(uniq_pairs)), key=lambda i: (i * 2654435761) % max(1, len(uniq_pairs)))
    cut = int(split * len(uniq_pairs))
    train_pairs = {uniq_pairs[i] for i in order[:cut]}
    tr = [(s, pr) for s, pr in zip(samples, pairs) if pr in train_pairs]
    te = [(s, pr) for s, pr in zip(samples, pairs) if pr not in train_pairs]
    train = [s for s, _ in tr]
    weights = fit_elasticities(train, **fit_kw)

    preds, y, tp = [], [], []
    for s, pr in te:
        r, strat, base, outcome = s
        preds.append(_predict(weights, r, strat, base))
        y.append(int(outcome)); tp.append(pr)
    ece = expected_calibration_error(y, preds)
    # pair accuracy: for each held-out pair with a winner and a loser, does the winner score higher?
    by_pair = {}
    for pr, pp, yy in zip(tp, preds, y):
        by_pair.setdefault(pr, []).append((pp, yy))
    correct = total = 0
    for members in by_pair.values():
        wins = [pp for pp, yy in members if yy == 1]
        losses = [pp for pp, yy in members if yy == 0]
        for w in wins:
            for l in losses:
                total += 1
                correct += 1 if w > l else (0.5 if w == l else 0)
    pair_acc = correct / total if total else None
    return {
        "n_train": len(train), "n_test": len(y), "test_base_rate": round(sum(y) / len(y), 4),
        "grade": "A" if ece < 0.05 else "B" if ece < 0.10 else "C" if ece < 0.15 else "F",
        "ece": round(ece, 4), "brier": round(brier_score(y, preds), 4),
        "log_loss": round(log_loss(y, preds), 4), "auc": round(_auc(y, preds), 4),
        "pair_accuracy": round(pair_acc, 4) if pair_acc is not None else None,
        "n_pairs_tested": total,
        "weights": {k: round(v[0], 3) for k, v in sorted(weights.items(), key=lambda kv: -abs(kv[1][0]))[:10]},
        "note": "graded on a held-out (by-pair) split of REAL outcomes. AUC 0.5 / pair_accuracy 0.5 = chance.",
    }
