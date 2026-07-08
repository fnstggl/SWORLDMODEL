"""EXP-090 (Stage 2): the flywheel — ablate architecture choices against aggregate skill on the backtest corpus.

Re-runs the forecaster over the SAME clean questions under different architecture configs (grounding on/off,
compiler-chosen mechanism vs forced calibrated readout, variable count) — reusing the on-disk LLM cache so most
configs cost no new tokens — and, crucially, tests CALIBRATION fixes (temperature scaling) and ENSEMBLES,
fitting each on a train split and scoring on a held-out split (no cheating). The output is the loss surface:
which architecture maximizes log-loss skill vs the crowd, so we keep what wins.

Run: HF_TOKEN=.. DEEPSEEK_API_KEY=.. python -m experiments.exp090_ablation [max_items]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from swm.api.resilient_llm import resilient_chat_fn
from swm.eval.backtest_harness import direct_estimate, forecast_item
from swm.eval.forecasting_corpus import load_corpus
from swm.eval.metrics import log_loss

RESULT = "experiments/results/exp090_ablation.json"
CONFIGS = {                                                    # (kwargs, needs_fresh_compile)
    "grounded_readout":   (dict(force_readout=True, ground=True), False),
    "ungrounded_readout": (dict(force_readout=True, ground=False), False),
    "readout_top3vars":   (dict(force_readout=True, ground=True, max_vars=3), False),
    "compiler_mechanism": (dict(force_readout=False, ground=True), True),
}


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + math.exp(-max(-35, min(35, z))))


def _temperature(ps, ys):
    """Fit λ in p' = sigmoid(λ·logit(p)) minimizing log-loss — the classic overconfidence fix."""
    best_l, best = 1.0, 1e9
    for l in [x / 20 for x in range(4, 31)]:                   # 0.2 .. 1.5
        ll = log_loss(ys, [_sig(l * _logit(p)) for p in ps])
        if ll < best:
            best_l, best = l, ll
    return best_l


def _skill(ll_m, ll_b):
    return round(1 - ll_m / ll_b, 4) if ll_b > 1e-9 else None


def _eval_split(rows, key):
    """Fit temperature on the train half, score raw + calibrated on the held-out half, vs crowd + base."""
    half = len(rows) // 2
    tr, te = rows[:half], rows[half:]
    lam = _temperature([r[key] for r in tr], [r["outcome"] for r in tr])
    yte = [r["outcome"] for r in te]
    base = sum(r["outcome"] for r in tr) / max(1, len(tr))
    ll_raw = log_loss(yte, [r[key] for r in te])
    ll_cal = log_loss(yte, [_sig(lam * _logit(r[key])) for r in te])
    ll_crowd = log_loss(yte, [r["p_crowd"] for r in te])
    ll_base = log_loss(yte, [base] * len(te))
    return {"n_test": len(te), "temperature": round(lam, 3), "ll_raw": round(ll_raw, 4),
            "ll_calibrated": round(ll_cal, 4), "ll_crowd": round(ll_crowd, 4),
            "skill_raw_vs_crowd": _skill(ll_raw, ll_crowd), "skill_cal_vs_crowd": _skill(ll_cal, ll_crowd),
            "skill_cal_vs_base": _skill(ll_cal, ll_base)}


def run(max_items=700, n=2500) -> dict:
    corpus = [it for it in load_corpus() if it.cutoff_clean][:max_items]
    llm_c = resilient_chat_fn(system="You compile questions into runnable structural simulations. Emit ONLY "
                                     "the JSON spec.", max_tokens=1100)
    llm_d = resilient_chat_fn(system="You are a calibrated forecaster. Reply with ONLY compact JSON.",
                              max_tokens=80)

    rows = []
    for i, it in enumerate(corpus):
        row = {"outcome": it.outcome, "p_crowd": it.crowd_prob, "category": it.category}
        ok = True
        for name, (kw, _) in CONFIGS.items():
            p, _ = forecast_item(it, llm_c, n=n, **kw)
            if p is None and name == "grounded_readout":
                ok = False
                break
            row[name] = p
        if not ok:
            continue
        row["direct"] = direct_estimate(it, llm_d)
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"    {len(rows)} items  (cache={llm_c.calls['cache']} hf={llm_c.calls['hf']} ds={llm_c.calls['deepseek']})")

    # fill Nones for non-primary configs with the grounded value (config produced nothing) so evals are fair
    for r in rows:
        for name in list(CONFIGS) + ["direct"]:
            if r.get(name) is None:
                r[name] = r["grounded_readout"]

    ablation = {name: _eval_split(rows, name) for name in CONFIGS}
    ablation["direct_llm"] = _eval_split(rows, "direct")
    # ensembles (mean of logits), fit + score the same way
    for r in rows:
        r["ens_model_direct"] = _sig(0.5 * (_logit(r["grounded_readout"]) + _logit(r["direct"])))
        r["ens_model_crowd"] = _sig(0.5 * (_logit(r["grounded_readout"]) + _logit(r["p_crowd"])))
    ablation["ensemble_model+direct"] = _eval_split(rows, "ens_model_direct")
    ablation["ensemble_model+crowd"] = _eval_split(rows, "ens_model_crowd")

    res = {"n": len(rows), "note": "temperature fit on train half, scored on held-out half; skill vs crowd/base",
           "ablation": ablation}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print(f"\nEXP-090  ablation on {len(rows)} clean questions (held-out scoring):")
    print(f"  {'config':24s} {'temp':>5s} {'skill_cal_vs_crowd':>18s} {'skill_cal_vs_base':>17s}")
    for name, a in sorted(ablation.items(), key=lambda kv: -(kv[1]['skill_cal_vs_crowd'] or -9)):
        print(f"  {name:24s} {a['temperature']:>5.2f} {str(a['skill_cal_vs_crowd']):>18s} "
              f"{str(a['skill_cal_vs_base']):>17s}")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run(max_items=int(sys.argv[1]) if len(sys.argv) > 1 else 700)
