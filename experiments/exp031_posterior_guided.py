"""EXP-031: posterior-guided attribution training (the SWM paper's core recipe).

Trains a FORWARD event attributor on the hindsight attribution labels shipped in SWM-Bench (foresight
learns from hindsight — no outcome leakage in the features, only in the labels, which is the point). Two
questions:
  (Q1) Does hindsight supervise a useful foresight attributor? — measure the forward attributor's
       accuracy at predicting which candidate news the posterior judged causal, on held-out transitions.
  (Q2) Does posterior-guided attribution improve the belief-transition operator? — use the attributor's
       learned "event strength" as a gate on the belief change, and compare to persistence, the EXP-030
       one-shot LLM channel, and a naive learned model with no attribution.

No-cheat: chronological (train before test); features strictly pre-shift. Writes JSON.
Run: python -m experiments.exp031_posterior_guided
"""
from __future__ import annotations

import glob
import json
import random
import statistics
from pathlib import Path

from swm.transition.attribution import ForwardAttributor, news_features, _tok
from swm.transition.belief_dynamics import BeliefTransition, _Ridge, featurize
from experiments.datasets_swm import load

RESULT = "experiments/results/exp031_posterior_guided.json"
FLAT = 0.02


def _load_impacts():
    imp = {}
    paths = glob.glob("data/swm_impact_[0-9]*.json") or glob.glob("experiments/results/exp030_swm/swm_impact.json")
    for fp in paths:
        try:
            rows = json.loads(Path(fp).read_text())
        except Exception:
            continue
        for r in rows:
            if isinstance(r, dict) and "id" in r:
                imp[r["id"]] = float(r.get("impact", 0.0)) * float(r.get("confidence", 1.0))
    return imp


def _cls(d):
    return 1 if d > FLAT else (-1 if d < -FLAT else 0)


def _metrics(rows, pred):
    mae = da = 0
    for r, pd in zip(rows, pred):
        p = r["history"][-1]["p"]; t = r["target"]["p"]
        mae += abs(min(1, max(0, p + pd)) - t)
        da += int(_cls(pd) == _cls(t - p))
    n = len(rows)
    return {"mae": round(mae / n, 4), "da3": round(da / n, 3)}


def _attributor_accuracy(attributor, records):
    """Held-out accuracy of the forward attributor at recovering the posterior causal label."""
    correct = tot = pos = 0
    for r in records:
        qtok = _tok(r.get("question", "") + " " + r.get("description", ""))
        attr = {a["news_idx"]: a.get("score", 0.0) for a in r.get("attributions", [])}
        for i, n in enumerate(r.get("news", []) or []):
            if attributor.model is None:
                continue
            p = attributor.model.predict_proba(news_features(n, qtok))
            lab = 1 if attr.get(i, 0.0) >= 0.5 else 0
            correct += int((p > 0.5) == lab); tot += 1; pos += lab
    return {"accuracy": round(correct / max(1, tot), 3), "base_rate_causal": round(pos / max(1, tot), 3),
            "n_news": tot}


def run():
    imp = _load_impacts()
    train_all = [r for r in load("train") if r.get("history") and r.get("target")]
    test = [r for r in load("test_kalshi") if r.get("history") and r.get("target")]
    rng = random.Random(0); tr = train_all[:]; rng.shuffle(tr); train = tr[:640]
    for i, r in enumerate(test):
        r["_impact"] = imp.get(f"te_{i}", 0.0)
    for i, r in enumerate(train):
        r["_impact"] = imp.get(f"tr_{i}", 0.0)

    # Q1: forward attributor trained on hindsight labels (train), evaluated on the held-out test
    attributor = ForwardAttributor().fit(train_all)              # train on all labeled train transitions
    q1 = _attributor_accuracy(attributor, test)

    # gate scale tuned on train
    imp_fn = lambda r: r.get("_impact", 0.0)
    best_s = min((0.05, 0.1, 0.15, 0.2, 0.25),
                 key=lambda s: sum(abs((r["history"][-1]["p"] + s * r["_impact"]) - r["target"]["p"]) for r in train))

    def persistence(rows): return [0.0] * len(rows)
    def llm_raw(rows): return [best_s * r["_impact"] for r in rows]
    def posterior_gated(rows):
        # learned event-strength gate (from the posterior-guided attributor) x the LLM impact channel
        return [attributor.event_strength(r) * best_s * r["_impact"] for r in rows]

    # naive vs posterior-guided PURELY learned world models (no LLM at inference)
    def wm_features(r, guided):
        base = featurize(r, 0.0)
        if guided:
            return base + [attributor.event_strength(r), attributor.attributed_salience(r)]
        qtok = _tok(r.get("question", "") + " " + r.get("description", ""))
        news = r.get("news", []) or []
        msal = statistics.mean([news_features(n, qtok)[0] for n in news]) if news else 0.0
        return base + [min(1.0, len(news) / 20.0), msal]

    def fit_wm(guided):
        X = [wm_features(r, guided) for r in train]
        y = [r["target"]["p"] - r["history"][-1]["p"] for r in train]
        return _Ridge(l2=1.0).fit(X, y)

    naive_wm = fit_wm(False); guided_wm = fit_wm(True)

    tiers = {
        "persistence": _metrics(test, persistence(test)),
        "llm_impact": _metrics(test, llm_raw(test)),
        "posterior_gated_llm": _metrics(test, posterior_gated(test)),
        "naive_learned": _metrics(test, [naive_wm.predict(wm_features(r, False)) for r in test]),
        "posterior_guided_learned": _metrics(test, [guided_wm.predict(wm_features(r, True)) for r in test]),
    }
    out = {"dataset": "kalshi", "n_test": len(test), "forward_attributor": q1, "gate_scale": best_s,
           "tiers": tiers,
           "guided_beats_naive_da": tiers["posterior_guided_learned"]["da3"] > tiers["naive_learned"]["da3"],
           "attributor_beats_chance": q1["accuracy"] > max(q1["base_rate_causal"], 1 - q1["base_rate_causal"])}
    print(f"EXP-031 posterior-guided attribution — Kalshi, n_test={len(test)}")
    print(f"  Q1 forward attributor: accuracy {q1['accuracy']} (causal base rate {q1['base_rate_causal']}, "
          f"n_news {q1['n_news']}) — {'beats chance' if out['attributor_beats_chance'] else 'at chance'}")
    print(f"  Q2 belief-change tiers (MAE / 3-way DA):")
    for k, v in tiers.items():
        print(f"    {k:<26} MAE {v['mae']}  DA3 {v['da3']}")
    print(f"  posterior-guided learned beats naive learned on DA: {out['guided_beats_naive_da']}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
