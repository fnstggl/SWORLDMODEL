"""Higgs diffusion benchmark — Reference World D: network + temporal rollout, leak-free (item 3).

Time-forward design on the pre-announcement rumor period (activity starts 2012-07-01; the announcement
regime break is ~hour 80): TRAIN cohort = exposed inactive users at t0+24h, labeled on (24h,48h];
EVAL cohort = exposed inactive users at t0+48h, labeled on (48h,72h]. Features strictly as-of each cutoff.

Arms (identical eval cohort):
  H0 base rate (train-window activation rate)
  H1 fitted logistic on as-of exposure features (active followees, degree, exposure frac, recency)
  V2 MAX-CAPACITY: event-driven contagion world — fitted per-exposure hazard q, per-particle latent q,
     within-window in-sample contagion rollout (new activations add exposure)
  Ablations: no network (population hazard) | no rollout (exposure frozen at t0) | no latent (point q)
LLM arms: STRUCTURALLY ABSENT — SNAP Higgs carries no message content (omission logged, not faked).

Metrics: Brier / logloss / AUROC / PR-AUC, paired bootstrap CIs. No API cost; pure compute.
Run: PYTHONPATH=. python -m experiments.wmv2_higgs_run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_higgs.json"


def _metrics(rows, preds):
    pr = list(zip(preds, [r["y"] for r in rows]))
    n = len(pr)
    brier = sum((p - y) ** 2 for p, y in pr) / n
    ll = -sum(y * math.log(max(1e-6, p)) + (1 - y) * math.log(max(1e-6, 1 - p)) for p, y in pr) / n
    pos = [p for p, y in pr if y == 1]
    neg = [p for p, y in pr if y == 0]
    auroc = (sum(1 for a in pos for c in neg if a > c) + 0.5 * sum(1 for a in pos for c in neg if a == c)) \
        / max(1, len(pos) * len(neg)) if pos and neg else None
    ap = None
    if pos:
        ranked = sorted(pr, key=lambda t: -t[0])
        tp, ap = 0, 0.0
        for i, (_, y) in enumerate(ranked, 1):
            if y == 1:
                tp += 1
                ap += tp / i
        ap /= len(pos)
    return {"brier": round(brier, 5), "logloss": round(ll, 4),
            "auroc": round(auroc, 3) if auroc else None, "pr_auc": round(ap, 3) if ap else None,
            "real_rate": round(sum(y for _, y in pr) / n, 4),
            "pred_rate": round(sum(p for p, _ in pr) / n, 4), "n": n}


def _paired(rows, pa, pb, n_boot=1000, seed=5):
    d = [(a - r["y"]) ** 2 - (b - r["y"]) ** 2 for a, b, r in zip(pa, pb, rows)]
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n_boot))
    return {"mean": round(sum(d) / len(d), 6), "ci95": [round(bs[25], 6), round(bs[-26], 6)], "n": len(d)}


def run(n_sample, particles):
    from swm.world_model_v2.reference.higgs import (build_cohort, exposure_snapshot, feats_full,
                                                    fit_logistic, fit_q, load_activation_times,
                                                    sample_subgraph_edges, v2_contagion_predict)
    t0w = time.time()
    print("parsing activity …", flush=True)
    first, t_min = load_activation_times()
    H = 3600.0
    cut_tr, cut_ev, win = t_min + 24 * H, t_min + 48 * H, 24 * H
    print(f"users_ever_active={len(first)}; streaming follower graph (2 cutoffs) …", flush=True)
    snap = exposure_snapshot(first, [cut_tr, cut_ev])
    train = build_cohort(first, snap, cut_tr, win, n_sample=n_sample, seed=13)
    test = build_cohort(first, snap, cut_ev, win, n_sample=n_sample, seed=17)
    r_tr = sum(r["y"] for r in train) / max(1, len(train))
    print(f"train n={len(train)} rate={r_tr:.4f} | test n={len(test)} "
          f"rate={sum(r['y'] for r in test)/max(1,len(test)):.4f}", flush=True)

    h0 = [max(1e-4, r_tr)] * len(test)
    pred_h1, coef = fit_logistic(train, feats_full)
    h1 = [pred_h1(r) for r in test]
    q = fit_q(train, win)
    print(f"fitted: logistic w={coef['w']} | q={q:.5f}/exposure/day", flush=True)
    print("collecting in-sample subgraph for rollout …", flush=True)
    fol = sample_subgraph_edges([r["u"] for r in test])
    n_sub_edges = sum(len(v) for v in fol.values())
    print(f"sample-subgraph followers-of edges: {n_sub_edges}", flush=True)

    t1 = time.time()
    v2 = v2_contagion_predict(test, q, win, fol, n_particles=particles, seed=7, base_rate=r_tr)
    v2_lat = time.time() - t1
    v2_noroll = v2_contagion_predict(test, q, win, fol, n_particles=particles, seed=7, rollout=False,
                                     base_rate=r_tr)
    v2_nonet = v2_contagion_predict(test, q, win, fol, n_particles=particles, seed=7, network=False,
                                    base_rate=r_tr)
    v2_nolat = v2_contagion_predict(test, q, win, fol, n_particles=particles, seed=7, latent=False,
                                    base_rate=r_tr)

    arms = {"H0_base": h0, "H1_logistic": h1, "V2_max": v2, "V2_no_rollout": v2_noroll,
            "V2_no_network": v2_nonet, "V2_no_latent": v2_nolat}
    detail = {a: _metrics(test, p) for a, p in arms.items()}
    paired = {"V2_vs_H1": _paired(test, v2, h1), "V2_vs_H0": _paired(test, v2, h0),
              "V2_vs_no_rollout": _paired(test, v2, v2_noroll),
              "V2_vs_no_network": _paired(test, v2, v2_nonet),
              "V2_vs_no_latent": _paired(test, v2, v2_nolat),
              "H1_vs_H0_network_features": _paired(test, h1, h0)}
    out = {"cohorts": {"train_cutoff_h": 24, "eval_cutoff_h": 48, "window_h": 24,
                       "note": "both windows pre-announcement (regime break ~h80)"},
           "detail": detail, "paired": paired, "fitted": {"logistic": coef, "q_per_exposure_day": q},
           "rollout_scope": {"in_sample_edges": n_sub_edges,
                             "note": "within-window contagion simulated over the sampled cohort subgraph "
                                     "only — out-of-sample mid-window activations are not simulated "
                                     "(honest scope bound)"},
           "llm_arms": "STRUCTURALLY ABSENT — no message content in SNAP Higgs; nothing to interpret",
           "_meta": {"n_sample": n_sample, "particles": particles,
                     "v2_latency_s": round(v2_lat, 1), "runtime_s": round(time.time() - t0w, 1),
                     "llm_calls": 0, "est_cost_usd": 0.0}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    for a, m in detail.items():
        print(f"  {a:14s} brier={m['brier']} logloss={m['logloss']} auroc={m['auroc']} "
              f"pr_auc={m['pr_auc']} pred_rate={m['pred_rate']}")
    for k, v in paired.items():
        print(f"  {k}: Δ={v['mean']:+.6f} CI{v['ci95']}")
    print(f"wrote {RESULT} ({out['_meta']['runtime_s']}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=4000)
    ap.add_argument("--particles", type=int, default=30)
    a = ap.parse_args()
    run(a.n_sample, a.particles)
