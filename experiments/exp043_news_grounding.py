"""EXP-043: grounding drivers in REAL retrieved news vs the LLM gestalt (Part b of the north-star build).

EXP-037's honest finding: the LLM-gestalt "drivers" added nothing over the holistic base rate — because
they were the LLM re-deriving its own judgment in a variable costume, not real external variables. Part b
tests the fix: extract drivers from the REAL as-of news actually attached to each question (dated strictly
before the target — leakage-free) and ask whether *grounded* variables recover signal the gestalt could
not.

Grounded features from the as-of news (no price, no LLM opinion):
  - volume        : log count of news items (salience/activity around the question)
  - result_cue    : fraction of headlines with resolution language (_RESULT regex) — is it resolving?
  - resolution_polarity : signed balance of positive vs negative resolution terms in result-cue headlines
  - recency       : freshness of the latest news relative to the target (closer -> more decisive)
  - source_count  : distinct sources (independent corroboration)

Arms (no-cheat: features are as-of; train/test are the benchmark's chronological splits):
  1. base rate (composite)          — the training marginal ("the crowd's average")
  2. NEWS-GROUNDED readout          — a pooled logistic on the real news features only (no price)
  3. market lean (reference ceiling)— the price itself (what "reading the market" achieves)

Decisive question: does the news-grounded readout beat the base rate (unlike EXP-037's gestalt), and how
far does real grounding close the gap to the price? Writes JSON.
Run: python -m experiments.exp043_news_grounding
"""
from __future__ import annotations

import datetime
import json
import math
import re
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss
from swm.transition.attribution import _RESULT
from swm.transition.readout import LogisticReadout
from experiments.datasets_swm import load

RESULT = "experiments/results/exp043_news_grounding.json"
_POS = re.compile(r"\b(win|wins|won|lead|leads|leading|ahead|approv|pass(?:es|ed)?|surg|rise|rises|"
                  r"gain|gains|beat|beats|clinch|secure|advance|on track|likely)\b", re.I)
_NEG = re.compile(r"\b(lose|loses|lost|trail|trails|behind|reject|fail|fails|failed|drop|drops|fall|"
                  r"falls|decline|miss|misses|eliminat|out of|unlikely|defeat)\b", re.I)


def _news_features(rec):
    news = rec.get("news") or []
    t_target = rec["target"]["t"]
    texts, cues, pos, neg, latest, srcs = [], 0, 0, 0, 0.0, set()
    for nw in news:
        title = (nw.get("title") or "") + " " + (nw.get("description") or "")
        texts.append(title)
        is_cue = bool(_RESULT.search(title))
        cues += int(is_cue)
        if is_cue:
            pos += len(_POS.findall(title)); neg += len(_NEG.findall(title))
        srcs.add(nw.get("source", ""))
        pa = nw.get("published_at")
        if pa:
            try:
                ts = datetime.datetime.fromisoformat(pa.replace("Z", "+00:00")).timestamp()
                latest = max(latest, ts)
            except Exception:
                pass
    nnews = len(news)
    volume = math.log1p(nnews)
    result_cue = cues / nnews if nnews else 0.0
    polarity = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
    recency = 1.0 if not latest else max(0.0, min(1.0, 1.0 - (t_target - latest) / (14 * 86400.0)))
    source_count = math.log1p(len(srcs))
    return [volume, result_cue, polarity, recency, source_count]


def _outcome(rec, horizon=6):
    fut = rec.get("future") or []
    if len(fut) < 1:
        return None
    return int(fut[min(horizon, len(fut)) - 1]["p"] > 0.5)


def _usable(split):
    out = []
    for r in load(split):
        if not (r.get("news") and r.get("target") and r.get("future")):
            continue
        y = _outcome(r)
        if y is None:
            continue
        out.append((_news_features(r), y, r["target"]["p"]))
    return out


def _score(y, p):
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    nf = [(pi, yi) for pi, yi in zip(p, y) if abs(pi - 0.5) > 0.02]
    da = sum(int((pi > 0.5) == (yi == 1)) for pi, yi in nf) / max(1, len(nf))
    my = sum(y) / len(y); mp = sum(p) / len(p)
    num = sum((pi - mp) * (yi - my) for pi, yi in zip(p, y))
    den = (sum((pi - mp) ** 2 for pi in p) * sum((yi - my) ** 2 for yi in y)) ** 0.5
    corr = num / den if den > 1e-9 else 0.0
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4),
            "directional_accuracy": round(da, 4), "corr_with_outcome": round(corr, 4)}


def run():
    tr, te = _usable("train"), _usable("test_kalshi")
    Xtr = [f for f, _, _ in tr]; ytr = [y for _, y, _ in tr]
    yte = [y for _, y, _ in te]
    base = sum(ytr) / len(ytr)

    grounded = LogisticReadout(epochs=400, l2=1.0).fit(Xtr, ytr)
    p_grounded = [grounded.predict_proba(f) for f, _, _ in te]
    p_base = [base] * len(yte)
    p_price = [p for _, _, p in te]

    arms = {"base_rate_composite": _score(yte, p_base),
            "news_grounded": _score(yte, p_grounded),
            "market_lean_reference": _score(yte, p_price)}
    d_base = round(arms["base_rate_composite"]["log_loss"] - arms["news_grounded"]["log_loss"], 4)
    gap_closed = None
    price_ll, base_ll, grd_ll = (arms[k]["log_loss"] for k in
                                 ("market_lean_reference", "base_rate_composite", "news_grounded"))
    if base_ll - price_ll > 1e-9:
        gap_closed = round((base_ll - grd_ll) / (base_ll - price_ll), 3)

    # feature importances (standardized weights) — which real signals carry the grounding
    names = ["volume", "result_cue", "resolution_polarity", "recency", "source_count"]
    imp = sorted(zip(names, grounded.w), key=lambda t: -abs(t[1]))

    out = {"dataset": "kalshi", "n_train": len(tr), "n_test": len(te), "base_rate": round(base, 4),
           "arms": arms, "news_grounded_vs_base_logloss": d_base,
           "grounded_beats_base": d_base > 0, "fraction_of_price_gap_closed": gap_closed,
           "feature_weights": [(n, round(w, 4)) for n, w in imp]}
    print(f"EXP-043 news-grounded drivers vs base rate — Kalshi, train={len(tr)} test={len(te)} "
          f"base rate {base:.3f}")
    for k, v in arms.items():
        print(f"  {k:<26} log_loss {v['log_loss']}  brier {v['brier']}  dir_acc {v['directional_accuracy']}"
              f"  corr {v['corr_with_outcome']}")
    print(f"  -> news-grounded beats base rate: {out['grounded_beats_base']} (Δlog_loss {d_base:+.4f}); "
          f"fraction of price gap closed: {gap_closed}")
    print(f"  -> grounding signal carried by: {', '.join(f'{n}({w:+.2f})' for n, w in imp[:3])}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
