"""EXP-096: firm up the GDELT result on a LARGER conflict corpus, and add the structural-state layer.

The EXP-095 conflict slice was only n=15. This pulls ~140 cutoff-clean geopolitics/conflict questions that name
a country and runs the latent-state simulation in THREE conditions on each:
  - plain     : no social/structural grounding (the base forecaster)
  - gdelt     : + GDELT as-of social state (fast event/flow layer)
  - full      : + GDELT social state AND the structural-state layer (V-Dem institutions + World Bank economy),
                with fragility scaling the event shock (the structural×event coupling)
Compare AUC / calibrated log-loss / skill-vs-crowd to see whether measuring the real state of the social world
— fast flow, then slow structural state — improves forecasts of it, on a sample big enough to trust.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp096_conflict_grounding [max_items]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.api.gdelt_social import GdeltSocialGrounder
from swm.api.inner_crowd import _logit, _sig
from swm.api.latent_forecast import latent_forecast
from swm.api.resilient_llm import resilient_chat_fn
from swm.api.structural_state import StructuralGrounder
from swm.eval.forecasting_corpus import BacktestItem, pull_conflict_corpus
from swm.eval.metrics import log_loss

CORPUS = "experiments/results/conflict_corpus.json"
PRED = "experiments/results/exp096_predictions.json"
RESULT = "experiments/results/exp096_conflict_grounding.json"
CONDS = ("plain", "gdelt", "full")


def _auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    return round(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)), 4) if pos and neg else None


def _temp(ps, ys):
    return min([x / 20 for x in range(2, 31)], key=lambda l: log_loss(ys, [_sig(l * _logit(p)) for p in ps]))


def _score(name, ps, ys, crowd):
    half = len(ps) // 2
    lam = _temp(ps[:half], ys[:half])
    cal = [_sig(lam * _logit(p)) for p in ps]
    base = sum(ys[:half]) / max(1, half)
    ll = log_loss(ys[half:], cal[half:])
    llc = log_loss(ys[half:], crowd[half:])
    llb = log_loss(ys[half:], [base] * len(ys[half:]))
    return {"name": name, "n": len(ps), "auc": _auc(ps, ys), "ll_cal": round(ll, 4),
            "skill_vs_crowd": round(1 - ll / llc, 4), "skill_vs_base": round(1 - ll / llb, 4)}


def run(max_items=200, n=3000) -> dict:
    corpus = ([BacktestItem(**r) for r in json.loads(Path(CORPUS).read_text())] if Path(CORPUS).exists()
              else pull_conflict_corpus())[:max_items]
    llm = resilient_chat_fn(max_tokens=700)
    gdelt = GdeltSocialGrounder(window_days=14)
    struct = StructuralGrounder()

    pr = json.loads(Path(PRED).read_text()) if Path(PRED).exists() else {}
    pr = {q: dict(v) for q, v in pr.items()}
    tasks = [(it, c) for it in corpus for c in CONDS if not (it.qid in pr and c in pr[it.qid])]

    def _one(it, cond):
        sg = gdelt if cond in ("gdelt", "full") else None
        stg = struct if cond == "full" else None
        p, _ = latent_forecast(it.question, it.as_of, it.resolve_ts, llm, n=n,
                               social_grounder=sg, structural_grounder=stg)
        return it.qid, cond, p

    ct = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for fut in as_completed([ex.submit(_one, it, c) for it, c in tasks]):
            qid, cond, p = fut.result()
            pr.setdefault(qid, {})[cond] = p
            ct += 1
            if ct % 45 == 0:
                Path(PRED).write_text(json.dumps(pr))
                print(f"    {ct}/{len(tasks)} (cache={llm.calls.get('cache', 0)}, ds={llm.calls.get('deepseek', 0)})")
    Path(PRED).write_text(json.dumps(pr))

    qids = [it.qid for it in corpus if it.qid in pr and all(pr[it.qid].get(c) is not None for c in CONDS)]
    by = {it.qid: it for it in corpus}
    ys = [by[q].outcome for q in qids]
    crowd = [by[q].crowd_prob for q in qids]
    evals = [_score(c, [pr[q][c] for q in qids], ys, crowd) for c in CONDS]
    evals.append({"name": "REAL_CROWD", "n": len(qids), "auc": _auc(crowd, ys),
                  "ll_cal": round(log_loss(ys, crowd), 4)})
    moved_g = sum(1 for q in qids if abs(pr[q]["gdelt"] - pr[q]["plain"]) > 0.02)
    moved_s = sum(1 for q in qids if abs(pr[q]["full"] - pr[q]["gdelt"]) > 0.02)
    res = {"n": len(qids), "yes_rate": round(sum(ys) / max(1, len(ys)), 3),
           "gdelt_moved_vs_plain": moved_g, "structural_moved_vs_gdelt": moved_s, "variants": evals}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print(f"\nEXP-096  conflict grounding on {len(qids)} questions (YES rate {res['yes_rate']}; "
          f"GDELT moved {moved_g}, structural moved {moved_s})")
    print(f"  {'condition':12s} {'AUC':>7s} {'ll_cal':>7s} {'sk_vs_crowd':>11s}")
    for e in evals:
        print(f"  {e['name']:12s} {str(e['auc']):>7s} {str(e['ll_cal']):>7s} {str(e.get('skill_vs_crowd')):>11s}")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run(max_items=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
