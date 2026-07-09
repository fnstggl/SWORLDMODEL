"""EXP-092: full experimentation run — try many first-principles approaches on the 660 questions, keep winners.

Deconstructing how events actually resolve, and testing each lever:
  V0  baseline latent (EXP-091).
  V1  AS-OF grounded metric: for a crypto price question, MEASURE the price known ON THE QUESTION'S DATE + the
      asset's realised volatility (leakage-free, Coinbase historical), trust=high — the sim of the real path.
  DIR the direct-LLM estimate (a second, independent view).
  ENS log-odds ensembles: latent+direct, grounded+direct, grounded+crowd (a deployable product mode).

Reports held-out calibrated log-loss + AUC + skill-vs-crowd, overall and per category (esp. crypto/economy —
the modelable, groundable slice), so we can see whether a REAL simulation with the same information the market
has actually beats the market where we can ground it.

Run: HF_TOKEN=.. DEEPSEEK_API_KEY=.. python -m experiments.exp092_experiments [max_items]
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

RESULT = "experiments/results/exp092_experiments.json"
GPRED = "experiments/results/exp092_grounded_predictions.json"


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
    best_l, best = 1.0, 1e9
    for l in [x / 20 for x in range(2, 31)]:
        ll = log_loss(ys, [_sig(l * _logit(p)) for p in ps])
        if ll < best:
            best_l, best = l, ll
    return best_l


def _ens(a, b, wa=0.5):
    return _sig(wa * _logit(a) + (1 - wa) * _logit(b))


def evaluate(name, ps, ys, crowd, idx_cat):
    half = len(ps) // 2
    lam = _temp(ps[:half], ys[:half])
    te = slice(half, None)
    cal = [_sig(lam * _logit(p)) for p in ps]
    base = sum(ys[:half]) / max(1, half)
    ll_cal = log_loss(ys[te], cal[half:])
    ll_crowd = log_loss(ys[te], crowd[half:])
    ll_base = log_loss(ys[te], [base] * len(ys[te]))
    out = {"name": name, "auc": _auc(ps, ys), "ll_cal": round(ll_cal, 4),
           "skill_vs_crowd": round(1 - ll_cal / ll_crowd, 4) if ll_crowd > 0 else None,
           "skill_vs_base": round(1 - ll_cal / ll_base, 4) if ll_base > 0 else None, "temp": round(lam, 3)}
    # crypto + economy slice (the groundable/modelable domains)
    for cat in ("crypto", "economy"):
        sub = [i for i, c in enumerate(idx_cat) if c == cat]
        if len(sub) >= 8:
            yy = [ys[i] for i in sub]
            out[f"auc_{cat}"] = _auc([ps[i] for i in sub], yy)
            out[f"ll_{cat}"] = round(log_loss(yy, [_sig(lam * _logit(ps[i])) for i in sub]), 3)
            out[f"crowd_ll_{cat}"] = round(log_loss(yy, [crowd[i] for i in sub]), 3)
    return out


def run(max_items=700, n=3000) -> dict:
    corpus = [it for it in load_corpus() if it.cutoff_clean][:max_items]
    by_qid = {it.qid: it for it in corpus}
    latent0 = {r["qid"]: r for r in json.loads(Path("experiments/results/exp091_latent_predictions.json").read_text())}
    direct = {r["qid"]: r["p_direct"] for r in json.loads(Path("experiments/results/exp089_predictions.json").read_text())
              if r.get("p_direct") is not None}
    llm = resilient_chat_fn(system="You are a careful superforecaster. Reply with ONLY compact JSON.",
                            max_tokens=700)
    asof = CryptoAsofGrounder()

    # V1: re-run latent WITH as-of metric grounding (LLM specs cache-served; crypto price fetched as-of)
    done = {r["qid"]: r for r in json.loads(Path(GPRED).read_text())} if Path(GPRED).exists() else {}
    todo = [it for it in corpus if it.qid not in done]

    def _work(it):
        p, spec = latent_forecast(it.question, it.as_of, it.resolve_ts, llm, n=n, metric_grounder=asof)
        return {"qid": it.qid, "p": p if p is not None else 0.5,
                "grounded": bool((spec.raw or {}).get("_grounded")) if spec else False}
    grows = list(done.values())
    if todo:
        with ThreadPoolExecutor(max_workers=10) as ex:
            for fut in as_completed([ex.submit(_work, it) for it in todo]):
                grows.append(fut.result())
                if len(grows) % 50 == 0:
                    Path(GPRED).write_text(json.dumps(grows))
        Path(GPRED).write_text(json.dumps(grows))
    grounded = {r["qid"]: r["p"] for r in grows}
    n_grounded = sum(1 for r in grows if r.get("grounded"))

    # assemble aligned vectors over the items we have everything for
    qids = [it.qid for it in corpus if it.qid in latent0 and it.qid in grounded and it.qid in direct]
    ys = [by_qid[q].outcome for q in qids]
    crowd = [by_qid[q].crowd_prob for q in qids]
    cats = [by_qid[q].category for q in qids]
    v0 = [latent0[q]["p_model"] for q in qids]
    v1 = [grounded[q] for q in qids]
    dr = [direct[q] for q in qids]

    variants = {
        "V0_latent_baseline": v0,
        "V1_asof_grounded_metric": v1,
        "DIRECT_llm": dr,
        "ENS_latent+direct": [_ens(a, b) for a, b in zip(v0, dr)],
        "ENS_grounded+direct": [_ens(a, b) for a, b in zip(v1, dr)],
        "ENS_grounded+crowd": [_ens(a, b) for a, b in zip(v1, crowd)],
    }
    evals = [evaluate(name, ps, ys, crowd, cats) for name, ps in variants.items()]
    evals.append({"name": "CROWD", "auc": _auc(crowd, ys), "ll_cal": round(log_loss(ys, crowd), 4),
                  "auc_crypto": _auc([crowd[i] for i, c in enumerate(cats) if c == "crypto"],
                                     [ys[i] for i, c in enumerate(cats) if c == "crypto"])})

    res = {"n": len(qids), "n_crypto_asof_grounded": n_grounded, "variants": evals}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print(f"EXP-092  experimentation on {len(qids)} questions ({n_grounded} crypto metric as-of grounded)")
    print(f"  {'variant':24s} {'AUC':>6s} {'ll_cal':>7s} {'sk_vs_crowd':>11s} | {'crypto:AUC':>10s} {'ll':>5s} {'crowdll':>7s}")
    for e in sorted(evals, key=lambda x: -(x["auc"] or 0)):
        print(f"  {e['name']:24s} {str(e['auc']):>6s} {str(e['ll_cal']):>7s} {str(e.get('skill_vs_crowd')):>11s} | "
              f"{str(e.get('auc_crypto')):>10s} {str(e.get('ll_crypto')):>5s} {str(e.get('crowd_ll_crypto')):>7s}")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run(max_items=int(sys.argv[1]) if len(sys.argv) > 1 else 700)
