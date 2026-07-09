"""EXP-091: the flywheel's second turn — re-run the 660-question backtest with the RE-ARCHITECTED forecaster
(latent-state + time-accurate transitions + base-rate anchor + honest uncertainty) and measure the change.

Tests the thesis directly: does honest calibration + real state, instead of confident confabulation, close the
gap to the crowd? Reports vs the OLD readout and the crowd: log-loss, discrimination (AUC), calibration
(temperature fit on train / scored held-out), extreme-prediction rate, and the coin-flip sanity check.

Run: HF_TOKEN=.. DEEPSEEK_API_KEY=.. python -m experiments.exp091_latent_backtest [max_items]
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.api.latent_forecast import latent_forecast
from swm.api.resilient_llm import resilient_chat_fn
from swm.eval.forecasting_corpus import load_corpus
from swm.eval.metrics import log_loss

PRED = "experiments/results/exp091_latent_predictions.json"
RESULT = "experiments/results/exp091_latent_backtest.json"


def _auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    if not pos or not neg:
        return None
    return round(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)), 4)


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + math.exp(-max(-35, min(35, z))))


def _temperature(ps, ys):
    best_l, best = 1.0, 1e9
    for l in [x / 20 for x in range(2, 31)]:
        ll = log_loss(ys, [_sig(l * _logit(p)) for p in ps])
        if ll < best:
            best_l, best = l, ll
    return best_l


def run(max_items=700, n=3000) -> dict:
    corpus = [it for it in load_corpus() if it.cutoff_clean][:max_items]
    old = {r["qid"]: r for r in json.loads(Path("experiments/results/exp089_predictions.json").read_text())}
    llm = resilient_chat_fn(system="You are a careful superforecaster. Reply with ONLY compact JSON.",
                            max_tokens=700)

    done = {r["qid"]: r for r in json.loads(Path(PRED).read_text())} if Path(PRED).exists() else {}
    rows = list(done.values())
    todo = [it for it in corpus if it.qid not in done]

    def _work(it):
        p, spec = latent_forecast(it.question, it.as_of, it.resolve_ts, llm, n=n)
        if p is None:
            p = it and 0.5
        return {"qid": it.qid, "category": it.category, "outcome": it.outcome, "p_model": p,
                "p_crowd": it.crowd_prob, "kind": spec.kind if spec else None,
                "base_rate": spec.base_rate if spec else None, "question": it.question[:100]}

    with ThreadPoolExecutor(max_workers=10) as ex:
        for fut in as_completed([ex.submit(_work, it) for it in todo]):
            rows.append(fut.result())
            if len(rows) % 25 == 0:
                Path(PRED).write_text(json.dumps(rows))
                print(f"    {len(rows)} scored (hf={llm.calls['hf']} ds={llm.calls['deepseek']} cache={llm.calls['cache']})")
    Path(PRED).write_text(json.dumps(rows))

    ys = [r["outcome"] for r in rows]
    base = sum(ys) / len(ys)
    ps = [r["p_model"] for r in rows]
    old_ps = [old[r["qid"]]["p_model"] for r in rows if r["qid"] in old]
    old_ys = [r["outcome"] for r in rows if r["qid"] in old]
    crowd = [r["p_crowd"] for r in rows]
    # held-out temperature calibration
    half = len(rows) // 2
    lam = _temperature(ps[:half], ys[:half])
    ps_cal = [_sig(lam * _logit(p)) for p in ps]

    def _sk(a, b):
        return round(1 - a / b, 4) if b > 1e-9 else None
    ll_model = log_loss(ys, ps)
    ll_cal = log_loss(ys[half:], ps_cal[half:])
    ll_crowd = log_loss(ys, crowd)
    ll_base = log_loss(ys, [base] * len(ys))
    ll_crowd_te = log_loss(ys[half:], crowd[half:])
    ll_base_te = log_loss(ys[half:], [base] * len(ys[half:]))

    res = {"n": len(rows), "base_rate": round(base, 4),
           "latent": {"log_loss_raw": round(ll_model, 4), "log_loss_calibrated": round(ll_cal, 4),
                      "AUC": _auc(ps, ys), "temperature": round(lam, 3),
                      "frac_extreme": round(sum(1 for p in ps if p > 0.9 or p < 0.1) / len(ps), 3),
                      "skill_cal_vs_crowd": _sk(ll_cal, ll_crowd_te), "skill_cal_vs_base": _sk(ll_cal, ll_base_te)},
           "old_readout": {"log_loss": round(log_loss(old_ys, old_ps), 4), "AUC": _auc(old_ps, old_ys),
                           "frac_extreme": round(sum(1 for p in old_ps if p > 0.9 or p < 0.1) / len(old_ps), 3)},
           "crowd": {"log_loss": round(ll_crowd, 4), "AUC": _auc(crowd, ys)},
           "coinflips": [[round(r["p_model"], 3), r["outcome"]] for r in rows if "coinflip" in r["question"].lower()]}
    # by category (calibrated skill vs crowd)
    res["by_category"] = {}
    for c in sorted({r["category"] for r in rows}):
        sub = [(i, r) for i, r in enumerate(rows) if r["category"] == c]
        if len(sub) >= 5:
            yy = [r["outcome"] for _, r in sub]
            res["by_category"][c] = {"n": len(sub), "auc": _auc([r["p_model"] for _, r in sub], yy),
                                     "ll_model": round(log_loss(yy, [_sig(lam * _logit(r["p_model"])) for _, r in sub]), 3),
                                     "ll_crowd": round(log_loss(yy, [r["p_crowd"] for _, r in sub]), 3)}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    L, O, C = res["latent"], res["old_readout"], res["crowd"]
    print(f"\nEXP-091  re-architected latent forecaster on {res['n']} clean questions")
    print(f"  LOG-LOSS   latent raw {L['log_loss_raw']} / calibrated {L['log_loss_calibrated']}  |  "
          f"OLD readout {O['log_loss']}  |  crowd {C['log_loss']}  |  base {round(ll_base,4)}")
    print(f"  AUC (discrimination)  latent {L['AUC']}  |  old {O['AUC']}  |  crowd {C['AUC']}")
    print(f"  extreme preds  latent {L['frac_extreme']}  vs old {O['frac_extreme']}   | temp {L['temperature']}")
    print(f"  SKILL cal vs crowd {L['skill_cal_vs_crowd']}   vs base {L['skill_cal_vs_base']}")
    print(f"  coin-flips (p, outcome): {res['coinflips']}")
    print("  by category (auc | model_ll vs crowd_ll):")
    for c, a in sorted(res["by_category"].items(), key=lambda kv: -(kv[1]["auc"] or 0)):
        print(f"    {c:12s} n={a['n']:3d}  auc {a['auc']}  ll {a['ll_model']} vs crowd {a['ll_crowd']}")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run(max_items=int(sys.argv[1]) if len(sys.argv) > 1 else 700)
