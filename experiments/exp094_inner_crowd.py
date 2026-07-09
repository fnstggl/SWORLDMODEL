"""EXP-094: inner-crowd ensemble on the 660 — does simulating a diverse crowd (not one agent) recover the
discrimination a single LLM pass lacks, and close the gap to the real crowd?

For each question, run the latent-state simulation through 8 diverse forecaster personas (base-rater, bull,
bear, domain-expert, superforecaster, contrarian, historian, quant), each with as-of metric grounding, then
aggregate their calibrated forecasts (mean log-odds, extremized — factor tuned on train). Compare AUC /
log-loss / skill-vs-crowd to the single-pass latent (EXP-091) and the crowd.

Run: HF_TOKEN=.. DEEPSEEK_API_KEY=.. python -m experiments.exp094_inner_crowd [max_items]
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.api.asof_market import CryptoAsofGrounder
from swm.api.inner_crowd import PERSONAS, _logit, _sig, aggregate, inner_crowd
from swm.api.resilient_llm import resilient_chat_fn
from swm.eval.forecasting_corpus import load_corpus
from swm.eval.metrics import log_loss

PRED = "experiments/results/exp094_crowd_predictions.json"
RESULT = "experiments/results/exp094_inner_crowd.json"


def _auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    return round(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)), 4) if pos and neg else None


def _temp(ps, ys):
    return min([x / 20 for x in range(2, 31)], key=lambda l: log_loss(ys, [_sig(l * _logit(p)) for p in ps]))


def _score(name, ps, ys, crowd, cats):
    half = len(ps) // 2
    lam = _temp(ps[:half], ys[:half])
    cal = [_sig(lam * _logit(p)) for p in ps]
    base = sum(ys[:half]) / max(1, half)
    ll = log_loss(ys[half:], cal[half:])
    llc = log_loss(ys[half:], crowd[half:])
    llb = log_loss(ys[half:], [base] * len(ys[half:]))
    out = {"name": name, "auc": _auc(ps, ys), "ll_cal": round(ll, 4),
           "skill_vs_crowd": round(1 - ll / llc, 4), "skill_vs_base": round(1 - ll / llb, 4)}
    for cat in ("crypto", "economy", "election", "sports", "tech"):
        sub = [i for i, c in enumerate(cats) if c == cat]
        if len(sub) >= 8:
            out[f"auc_{cat}"] = _auc([ps[i] for i in sub], [ys[i] for i in sub])
    return out


def run(max_items=700, n=2500) -> dict:
    corpus = [it for it in load_corpus() if it.cutoff_clean][:max_items]
    single = {r["qid"]: r["p_model"] for r in json.loads(Path("experiments/results/exp091_latent_predictions.json").read_text())}
    persona_llms = {name: resilient_chat_fn(system=sp + " Reply with ONLY compact JSON.", max_tokens=700)
                    for name, sp in PERSONAS.items()}
    asof = CryptoAsofGrounder()

    from swm.api.latent_forecast import latent_forecast
    done = {r["qid"]: r["probs"] for r in json.loads(Path(PRED).read_text())} if Path(PRED).exists() else {}
    pr = {q: dict(v) for q, v in done.items()}
    # FLATTEN to (question, persona) tasks for max concurrency
    tasks = [(it, name) for it in corpus for name in PERSONAS
             if not (it.qid in pr and name in pr[it.qid])]

    def _one(it, name):
        p, _ = latent_forecast(it.question, it.as_of, it.resolve_ts, persona_llms[name], n=n,
                               metric_grounder=asof)
        return it.qid, name, p

    done_ct = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for fut in as_completed([ex.submit(_one, it, name) for it, name in tasks]):
            qid, name, p = fut.result()
            pr.setdefault(qid, {})[name] = p
            done_ct += 1
            if done_ct % 150 == 0:
                Path(PRED).write_text(json.dumps([{"qid": q, "probs": v} for q, v in pr.items()]))
                c = persona_llms["bull"].calls
                print(f"    {done_ct}/{len(tasks)} calls (cache={sum(l.calls['cache'] for l in persona_llms.values())})")
    Path(PRED).write_text(json.dumps([{"qid": q, "probs": v} for q, v in pr.items()]))
    qids = [it.qid for it in corpus if it.qid in pr and it.qid in single]
    by = {it.qid: it for it in corpus}
    ys = [by[q].outcome for q in qids]
    crowd = [by[q].crowd_prob for q in qids]
    cats = [by[q].category for q in qids]

    def panel(q):
        return [v for v in pr[q].values() if v is not None]

    # tune the extremization factor on the train half (mean-logodds aggregation)
    half = len(qids) // 2
    best_a, best = 1.0, 1e9
    for a in [x / 10 for x in range(8, 31)]:                    # 0.8 .. 3.0
        agg_tr = [aggregate(panel(qids[i]), extremize=a) for i in range(half)]
        best_a, best = (a, log_loss(ys[:half], agg_tr)) if log_loss(ys[:half], agg_tr) < best else (best_a, best)

    variants = {
        "SINGLE_pass (V0)": [single[q] for q in qids],
        "CROWD_mean_logodds": [aggregate(panel(q)) for q in qids],
        "CROWD_median": [aggregate(panel(q), method="median") for q in qids],
        f"CROWD_extremized(a={best_a})": [aggregate(panel(q), extremize=best_a) for q in qids],
    }
    evals = [_score(name, ps, ys, crowd, cats) for name, ps in variants.items()]
    evals.append({"name": "REAL_CROWD", "auc": _auc(crowd, ys), "ll_cal": round(log_loss(ys, crowd), 4),
                  "auc_crypto": _auc([crowd[i] for i, c in enumerate(cats) if c == "crypto"],
                                     [ys[i] for i, c in enumerate(cats) if c == "crypto"])})
    # panel disagreement diagnostic
    import statistics
    spreads = [statistics.pstdev(panel(q)) for q in qids if len(panel(q)) > 1]
    res = {"n": len(qids), "n_personas": len(PERSONAS), "extremize_factor": best_a,
           "mean_panel_spread": round(sum(spreads) / len(spreads), 3), "variants": evals}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print(f"\nEXP-094  inner-crowd ({len(PERSONAS)} personas) on {len(qids)} questions "
          f"(mean panel disagreement sd={res['mean_panel_spread']}, extremize a={best_a})")
    print(f"  {'variant':30s} {'AUC':>6s} {'ll_cal':>7s} {'sk_vs_crowd':>11s} | crypto econ elec spo tech")
    for e in sorted(evals, key=lambda x: -(x["auc"] or 0)):
        print(f"  {e['name']:30s} {str(e['auc']):>6s} {str(e['ll_cal']):>7s} {str(e.get('skill_vs_crowd')):>11s} | "
              f"{e.get('auc_crypto')} {e.get('auc_economy')} {e.get('auc_election')} {e.get('auc_sports')} {e.get('auc_tech')}")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run(max_items=int(sys.argv[1]) if len(sys.argv) > 1 else 700)
