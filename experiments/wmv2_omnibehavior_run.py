"""OmniBehavior longitudinal persistence benchmark — Reference World C, REPAIRED, leak-free (item 2).

Repairs (both invalidities of the earlier harness):
  * targets = PASSIVE exposures only (Video/Live/Search) — E-commerce/CS/Ad events are action RECORDS
    (rate≈1.0 by construction; predicting them is a type lookup, not behavior prediction)
  * targets sampled UNIFORMLY over each user's chronological TEST segment (per-user 70/30 time split)

The structural question: does PERSISTENT engagement state (bursts across the real event sequence) add
held-out value? The momentum lift is measured on train FIRST; if ≈1, persistence is structurally
unexercised and the report says so.

Arms (identical targets, identical history evidence):
  P0 global rate | P1 per-user rate (persistence-free baseline) | P2 fitted metadata+momentum logistic
  P3 grounded direct LLM (full history prefix) | P4 true 3-call ensemble (temp 0.7, separate calls)
  V5 MAX-CAPACITY V2: fitted policy over metadata + momentum persistence state + LLM interpretation dims,
     particle latent-responsiveness readout
Ablations of V5: no persistence (momentum state zeroed, policy refit) | no interpretation (policy refit) |
no latent (point readout).

Metrics: Brier / logloss / AUROC / PR-AUC / hyperactivity delta; paired bootstrap CIs; cost + latency.
Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_omnibehavior_run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_omnibehavior_v2.json"
DS_IN, DS_OUT = 0.27e-6, 1.10e-6


def _metrics(rows, key):
    pr = [(min(1, max(0, r[key])), r["y"]) for r in rows if r.get(key) is not None]
    if not pr:
        return {}
    n = len(pr)
    brier = sum((p - y) ** 2 for p, y in pr) / n
    ll = -sum(y * math.log(min(1 - 1e-6, max(1e-6, p))) + (1 - y) * math.log(min(1 - 1e-6, max(1e-6, 1 - p)))
              for p, y in pr) / n
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
    return {"brier": round(brier, 4), "logloss": round(ll, 4),
            "auroc": round(auroc, 3) if auroc is not None else None,
            "pr_auc": round(ap, 3) if ap is not None else None,
            "real_rate": round(sum(y for _, y in pr) / n, 3),
            "pred_rate": round(sum(p for p, _ in pr) / n, 3),
            "hyperactivity_delta": round(sum(p for p, _ in pr) / n - sum(y for _, y in pr) / n, 3),
            "n": n}


def _paired(rows, k1, k2, n_boot=1000, seed=5):
    d = [(r[k1] - r["y"]) ** 2 - (r[k2] - r["y"]) ** 2
         for r in rows if r.get(k1) is not None and r.get(k2) is not None]
    if len(d) < 5:
        return None
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n_boot))
    return {"mean": round(sum(d) / len(d), 5), "ci95": [round(bs[25], 5), round(bs[-26], 5)], "n": len(d)}


def run(n_users, targets_per_user, fit_n, particles):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.grounding import parse_json
    from swm.eval.omnibehavior_eval import _PROMPT, _summ, _exposure, download_users
    from swm.world_model_v2.actor_cognition import fit_action_policy, interpret
    from swm.world_model_v2.reference.omnibehavior import (PASSIVE, acted, fit_stats, item_features,
                                                           split_user, user_events, v2_engagement_predict)
    t0 = time.time()
    meter = {"calls": 0, "tokens": 0}
    llm = default_chat_fn(system="You simulate ONE real platform user. Reply ONLY compact JSON.",
                          max_tokens=80, temperature=0.3)
    llm_hot = default_chat_fn(system="You simulate ONE real platform user. Reply ONLY compact JSON.",
                              max_tokens=80, temperature=0.7)
    llm_json = default_chat_fn(system="You are the actor described. Reply ONLY compact JSON.",
                               max_tokens=160, temperature=0.3)
    if llm is None:
        raise SystemExit("needs DEEPSEEK_API_KEY")

    def call(fn, prompt):
        txt = fn(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
        return txt

    # mid-size users carry the mixed-rate behavior (smallest users are degenerate 0/1)
    paths = download_users(n_users, max_bytes=6_000_000, cache_dir="data/omnibehavior")
    paths = [p for p in paths]
    users = {}
    for p in paths:
        for uid, u in json.load(open(p)).items():
            evs = user_events(u)
            pas = [e for e in evs if e.get("type") in PASSIVE]
            k = sum(1 for e in pas if acted(e))
            if len(pas) >= 60 and 0.02 <= k / len(pas) <= 0.6:           # mixed-rate users only
                users[uid] = {"profile": u.get("user_profile", ""), "events": evs}
    print(f"eligible mixed-rate users: {len(users)}", flush=True)

    train_by, test_targets = {}, []
    rng = random.Random(13)
    for uid, u in users.items():
        tr, te = split_user(u["events"])
        train_by[uid] = tr
        pas_idx = [i for i, e in enumerate(te) if e.get("type") in PASSIVE]
        rng.shuffle(pas_idx)                                  # UNIFORM chronological-segment sampling
        for i in sorted(pas_idx[:targets_per_user]):
            test_targets.append({"uid": uid, "profile": u["profile"], "test_events": te, "idx": i,
                                 "target": te[i], "prior": tr + te[:i], "y": int(acted(te[i]))})
    stats = fit_stats(train_by)
    print(f"targets={len(test_targets)} global_rate={stats['global_rate']:.3f} "
          f"momentum_lift={stats['momentum_lift']:.2f} (hot n={stats['momentum_n']['hot']}, "
          f"cold n={stats['momentum_n']['cold']})", flush=True)
    persistence_exercised = stats["momentum_lift"] > 1.15 or stats["momentum_lift"] < 0.87

    def interp_of(uid, profile, target):
        c = {k: v for k, v in (target.get("context") or {}).items() if v not in (None, "", [])}
        return interpret(lambda p: call(llm_json, p),
                         actor=f"platform user {uid}: {str(profile)[:300]}",
                         channel=f"short-video platform ({target.get('type', '?')})",
                         context="- you scroll past most content; acting is rare",
                         content=json.dumps(c, ensure_ascii=False)[:700])

    # ---- fit the calibration layers on TRAIN passive events (leak-free; interp on a subsample) ----
    fit_events = []
    for uid, tr in train_by.items():
        pas = [(i, e) for i, e in enumerate(tr) if e.get("type") in PASSIVE]
        for i, e in pas[-max(3, fit_n // max(1, len(train_by))):]:
            fit_events.append({"uid": uid, "profile": users[uid]["profile"], "target": e,
                               "prior": tr[:i], "y": int(acted(e))})
    fit_events = fit_events[:fit_n]
    S = {"full": [], "nointerp": [], "nopersist": []}
    for i, fe in enumerate(fit_events):
        pas_prior = [e for e in fe["prior"] if e.get("type") in PASSIVE]
        base = stats["user_rate"].get(fe["uid"], stats["global_rate"])
        f_meta = item_features(fe["uid"], fe["target"], pas_prior, stats)
        it = interp_of(fe["uid"], fe["profile"], fe["target"])
        ifeats = it.features() if it else [0.5] * 18
        S["full"].append((f_meta + ifeats, base, fe["y"]))
        S["nointerp"].append((f_meta, base, fe["y"]))
        f_nop = list(f_meta)
        f_nop[2] = 0.0
        S["nopersist"].append((f_nop + ifeats, base, fe["y"]))
        if i % 25 == 0:
            print(f"  [fit] {i}/{len(fit_events)} calls={meter['calls']}", flush=True)
    pols = {k: fit_action_policy(v) for k, v in S.items()}
    print(f"fitted policies (n={len(fit_events)}): momentum_w={pols['full'].w[2]:+.3f} "
          f"anchor={pols['full'].w_anchor:.2f}", flush=True)

    # ---- evaluate all arms on identical targets ----
    rows, lat = [], {"direct_s": 0.0, "v2_s": 0.0}
    for i, t in enumerate(test_targets):
        pas_prior = [e for e in t["prior"] if e.get("type") in PASSIVE]
        base = stats["user_rate"].get(t["uid"], stats["global_rate"])
        row = {"y": t["y"], "uid": t["uid"]}
        row["P0"] = stats["global_rate"]
        row["P1"] = base
        f_meta = item_features(t["uid"], t["target"], pas_prior, stats)
        row["P2"] = pols["nointerp"].p_engage(f_meta, base)
        hist = "\n".join(_summ(e) for e in t["prior"][-12:])
        prompt = _PROMPT.format(profile=t["profile"], history=hist, exposure=_exposure(t["target"]))
        tt = time.time()
        r3 = parse_json(call(llm, prompt)) or {}
        try:
            row["P3"] = min(0.97, max(0.01, float(r3.get("p"))))
        except (TypeError, ValueError):
            row["P3"] = None
        ens = []
        for _ in range(3):                                    # TRUE separate calls, temp 0.7
            re_ = parse_json(call(llm_hot, prompt)) or {}
            try:
                ens.append(min(0.97, max(0.01, float(re_.get("p")))))
            except (TypeError, ValueError):
                continue
        row["P4"] = sum(ens) / len(ens) if ens else None
        lat["direct_s"] += time.time() - tt
        tt = time.time()
        it = interp_of(t["uid"], t["profile"], t["target"])
        ifeats = it.features() if it else [0.5] * 18
        row["V5"] = v2_engagement_predict(f_meta + ifeats, base, pols["full"],
                                          latent=True, n_particles=particles, seed=i)["p"]
        row["V5_no_latent"] = v2_engagement_predict(f_meta + ifeats, base, pols["full"],
                                                    latent=False, seed=i)["p"]
        f_nop = list(f_meta)
        f_nop[2] = 0.0
        row["V5_no_persist"] = v2_engagement_predict(f_nop + ifeats, base, pols["nopersist"],
                                                     latent=True, n_particles=particles, seed=i)["p"]
        row["V5_no_interp"] = v2_engagement_predict(f_meta, base, pols["nointerp"],
                                                    latent=True, n_particles=particles, seed=i)["p"]
        lat["v2_s"] += time.time() - tt
        rows.append(row)
        if i % 20 == 0:
            print(f"  [eval] {i}/{len(test_targets)} calls={meter['calls']}", flush=True)

    ARMS = ["P0", "P1", "P2", "P3", "P4", "V5", "V5_no_persist", "V5_no_interp", "V5_no_latent"]
    detail = {a: _metrics(rows, a) for a in ARMS}
    paired = {"V5_vs_P1": _paired(rows, "V5", "P1"), "V5_vs_P2": _paired(rows, "V5", "P2"),
              "V5_vs_P3": _paired(rows, "V5", "P3"), "V5_vs_P4": _paired(rows, "V5", "P4"),
              "V5_vs_no_persist": _paired(rows, "V5", "V5_no_persist"),
              "V5_vs_no_interp": _paired(rows, "V5", "V5_no_interp"),
              "V5_vs_no_latent": _paired(rows, "V5", "V5_no_latent"),
              "P2_vs_P1_momentum_feature": _paired(rows, "P2", "P1")}
    out = {"n_targets": len(rows), "n_users": len(users), "detail": detail, "paired": paired,
           "momentum": {"lift": round(stats["momentum_lift"], 3), "p_hot": round(stats["p_hot"], 4),
                        "p_cold": round(stats["p_cold"], 4), "n": stats["momentum_n"],
                        "persistence_structurally_exercised": persistence_exercised},
           "policy_momentum_weight": pols["full"].w[2],
           "repair_note": ("targets=passive exposures only, uniform test-segment sampling; "
                           "action-record types (rate≈1.0) excluded as targets, kept as history"),
           "_meta": {"llm_calls": meter["calls"], "llm_tokens_est": meter["tokens"],
                     "est_cost_usd": round(meter["tokens"] * (DS_IN + DS_OUT) / 2, 4),
                     "model_name": "deepseek-chat (DeepSeek V3)",
                     "runtime_s": round(time.time() - t0, 1),
                     "latency_s": {k: round(v, 1) for k, v in lat.items()},
                     "license_note": "OmniBehavior CC-BY-NC-SA 4.0 — benchmark/research use only"}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nmomentum_lift={stats['momentum_lift']:.2f} exercised={persistence_exercised}")
    for a in ARMS:
        m = detail[a]
        if m:
            print(f"  {a:14s} brier={m['brier']} logloss={m['logloss']} auroc={m['auroc']} "
                  f"pr_auc={m['pr_auc']} hyper={m['hyperactivity_delta']}")
    for k, v in paired.items():
        if v:
            print(f"  {k}: Δ={v['mean']:+.5f} CI{v['ci95']}")
    print(f"wrote {RESULT} (calls={meter['calls']}, ~${out['_meta']['est_cost_usd']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=25)
    ap.add_argument("--targets-per-user", type=int, default=12)
    ap.add_argument("--fit-n", type=int, default=150)
    ap.add_argument("--particles", type=int, default=32)
    a = ap.parse_args()
    run(a.n_users, a.targets_per_user, a.fit_n, a.particles)
