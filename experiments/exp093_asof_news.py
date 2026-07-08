"""EXP-093: information parity — does as-of NEWS let the simulation match/beat the crowd?

The crowd's edge was information: it had the news at the question's date. This gives the model the SAME
information (GDELT headlines in [as_of−30d, as_of], leakage-free) plus as-of metric grounding, and re-runs the
660-question backtest. If the architecture is sound — same information, but computed by a real simulation
instead of biased instinct — this is where it should close the gap to the market.

Run: HF_TOKEN=.. DEEPSEEK_API_KEY=.. python -m experiments.exp093_asof_news [max_items]
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.api.asof_market import CryptoAsofGrounder
from swm.api.latent_forecast import latent_forecast
from swm.api.resilient_llm import resilient_chat_fn
from swm.eval.forecasting_corpus import load_corpus
from swm.eval.metrics import log_loss
from swm.retrieval.asof_news import asof_headlines

PRED = "experiments/results/exp093_news_predictions.json"
NEWS = "data/asof_news_cache.json"
RESULT = "experiments/results/exp093_asof_news.json"


def _auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    return round(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)), 4) if pos and neg else None


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + math.exp(-max(-35, min(35, z))))


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
    for cat in ("crypto", "economy", "election", "sports"):
        sub = [i for i, c in enumerate(cats) if c == cat]
        if len(sub) >= 8:
            out[f"auc_{cat}"] = _auc([ps[i] for i in sub], [ys[i] for i in sub])
    return out


def run(max_items=700, n=3000) -> dict:
    corpus = [it for it in load_corpus() if it.cutoff_clean][:max_items]
    base_latent = {r["qid"]: r["p_model"] for r in json.loads(Path("experiments/results/exp091_latent_predictions.json").read_text())}
    llm = resilient_chat_fn(system="You are a careful superforecaster. Reply with ONLY compact JSON.", max_tokens=700)
    asof = CryptoAsofGrounder()
    news_cache = json.loads(Path(NEWS).read_text()) if Path(NEWS).exists() else {}
    done = {r["qid"]: r for r in json.loads(Path(PRED).read_text())} if Path(PRED).exists() else {}
    todo = [it for it in corpus if it.qid not in done]

    def _work(it):
        heads = news_cache.get(it.qid)
        if heads is None:
            heads = asof_headlines(it.question, it.as_of)
        p, spec = latent_forecast(it.question, it.as_of, it.resolve_ts, llm, n=n, metric_grounder=asof, news=heads)
        return {"qid": it.qid, "p": p if p is not None else 0.5, "n_news": len(heads or []), "_heads": heads}

    rows = list(done.values())
    with ThreadPoolExecutor(max_workers=10) as ex:
        for fut in as_completed([ex.submit(_work, it) for it in todo]):
            r = fut.result()
            news_cache[r["qid"]] = r.pop("_heads") or []
            rows.append(r)
            if len(rows) % 25 == 0:
                Path(PRED).write_text(json.dumps(rows))
                Path(NEWS).write_text(json.dumps(news_cache))
                print(f"    {len(rows)} scored (news hits: {sum(1 for x in rows if x['n_news']>0)})")
    Path(PRED).write_text(json.dumps(rows))
    Path(NEWS).write_text(json.dumps(news_cache))

    news_p = {r["qid"]: r["p"] for r in rows}
    qids = [it.qid for it in corpus if it.qid in news_p and it.qid in base_latent]
    by = {it.qid: it for it in corpus}
    ys = [by[q].outcome for q in qids]
    crowd = [by[q].crowd_prob for q in qids]
    cats = [by[q].category for q in qids]
    evals = [_score("NEWS+grounded", [news_p[q] for q in qids], ys, crowd, cats),
             _score("no-news latent (V0)", [base_latent[q] for q in qids], ys, crowd, cats)]
    evals.append({"name": "CROWD", "auc": _auc(crowd, ys), "ll_cal": round(log_loss(ys, crowd), 4),
                  "auc_crypto": _auc([crowd[i] for i, c in enumerate(cats) if c == "crypto"],
                                     [ys[i] for i, c in enumerate(cats) if c == "crypto"])})
    res = {"n": len(qids), "n_with_news": sum(1 for q in qids if news_cache.get(q)), "variants": evals}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print(f"\nEXP-093  as-of news (information parity) on {len(qids)} questions "
          f"({res['n_with_news']} with as-of news)")
    print(f"  {'variant':22s} {'AUC':>6s} {'ll_cal':>7s} {'sk_vs_crowd':>11s} | cr{'yp':>4s} eco elec spo")
    for e in evals:
        print(f"  {e['name']:22s} {str(e['auc']):>6s} {str(e['ll_cal']):>7s} {str(e.get('skill_vs_crowd')):>11s} | "
              f"{e.get('auc_crypto')} {e.get('auc_economy')} {e.get('auc_election')} {e.get('auc_sports')}")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run(max_items=int(sys.argv[1]) if len(sys.argv) > 1 else 700)
