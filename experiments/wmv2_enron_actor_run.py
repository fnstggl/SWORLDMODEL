"""Structured actor-cognition ablation C0–C6 — Reference World A (Enron), leak-free, paired.

The scalar-content arm of the previous round (E10) is DEMOTED to baseline C0: exact message meaning entered
it through one scalar (reply_propensity). This ladder tests the structured replacement:

  C0  scalar reply_propensity policy (previous "max-capacity" E10) — the bottleneck under test
  C1  structured interpretation only (typed semantic features → TRAIN-fitted calibration layer, closed form)
  C2  C1 + typed action distribution through the event world (reply_now/reply_later/clarify/delegate/ignore)
  C3  C2 + sampled coherent hidden actor state (attention/responsiveness/obligation_sensitivity, correlated)
  C4  C3 + dynamic attention (mean-reverting workload/time-of-day process)
  C5  C4 + relationship state (history-inferred strength; bounded terminal transition)
  C6  MAXIMUM STRUCTURED ACTOR MODEL (all + interruption hazard + obligation coupling)

against E0 base rate, E1 fitted metadata, E2 text BoW, E3 grounded direct DeepSeek, E4 direct ensemble —
every arm on IDENTICAL held-out rows. ONE interpretation call per example is shared across C1–C6 (memoized),
so ladder differences isolate mechanism, not prompt luck. The calibration layer is fitted on a TRAIN
subsample's interpretations (leak-free). Metrics: Brier/logloss/AUROC/PR-AUC/ECE @7d + paired bootstrap CIs.

Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_enron_actor_run --limit 60000 --llm-n 120
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_enron_actor_ladder.json"
FORENSIC = "experiments/results/wmv2_enron_actor_forensic.json"
BUCKS = (1.0, 3.0, 7.0, 14.0)
DS_IN, DS_OUT = 0.27e-6, 1.10e-6


def _metrics(rows, key, b):
    pr = [(min(1, max(0, r[key][b])), r["y"][b]) for r in rows if r.get(key)]
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
    # PR-AUC (average precision)
    ap = None
    if pos:
        ranked = sorted(pr, key=lambda t: -t[0])
        tp = 0
        ap = 0.0
        for i, (_, y) in enumerate(ranked, 1):
            if y == 1:
                tp += 1
                ap += tp / i
        ap /= len(pos)
    ece, bins = 0.0, [[] for _ in range(10)]
    for p, y in pr:
        bins[min(9, int(p * 10))].append((p, y))
    for bn in bins:
        if bn:
            ece += len(bn) / n * abs(sum(p for p, _ in bn) / len(bn) - sum(y for _, y in bn) / len(bn))
    return {"brier": round(brier, 4), "logloss": round(ll, 4),
            "auroc": (round(auroc, 3) if auroc is not None else None),
            "pr_auc": (round(ap, 3) if ap is not None else None), "ece": round(ece, 4),
            "base_rate": round(sum(y for _, y in pr) / n, 3), "n": n}


def _paired(rows, k1, k2, b, n_boot=1000, seed=5):
    d = [(r[k1][b] - r["y"][b]) ** 2 - (r[k2][b] - r["y"][b]) ** 2
         for r in rows if r.get(k1) and r.get(k2)]
    if len(d) < 5:
        return None
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n_boot))
    return {"mean": round(sum(d) / len(d), 5), "ci95": [round(bs[25], 5), round(bs[-26], 5)], "n": len(d)}


def run(limit, llm_n, n_particles, train_fit_n):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.grounding import parse_json
    from swm.eval.response_datasets import load_enron_reply_delay
    from swm.world_model_v2.actor_cognition import fit_action_policy
    from swm.world_model_v2.reference.enron import (_CONTENT_PROMPT, build_examples, fit_hetero_sd,
                                                    fit_mechanisms, fit_text_baseline, interpret_message,
                                                    splits, text_baseline_p, v2_predict, v2_predict_actor)
    t0 = time.time()
    records = load_enron_reply_delay(os.path.join("data/enron/maildir"), limit_messages=limit)
    exs = build_examples(records)
    train, test_seen, test_new = splits(exs)
    fm = fit_mechanisms(train)
    tb = fit_text_baseline(train)
    hetero_sd = fit_hetero_sd(train)
    meter = {"calls": 0, "tokens": 0}
    _raw = default_chat_fn(system="You are the actor described. Reply ONLY compact JSON.",
                           max_tokens=160, temperature=0.3)
    _scalar_raw = default_chat_fn(system="You are the email recipient. Reply ONLY compact JSON.",
                                  max_tokens=60, temperature=0.3)
    _memo = {}

    def metered(fn, prompt):
        key = (id(fn), prompt)
        if key not in _memo:
            txt = fn(prompt)
            meter["calls"] += 1
            meter["tokens"] += (len(prompt) + len(txt or "")) // 4
            _memo[key] = txt
        return _memo[key]

    chat_interp = (lambda p: metered(_raw, p)) if _raw else None
    chat_scalar = (lambda p: metered(_scalar_raw, p)) if _scalar_raw else None
    print(f"train={len(train)} test_time={len(test_seen)} test_person={len(test_new)} "
          f"hetero_sd={hetero_sd:.3f} chat={'on' if chat_interp else 'OFF'}", flush=True)
    if chat_interp is None:
        raise SystemExit("no LLM backend — the structured ladder needs DEEPSEEK_API_KEY")

    # ---- PHASE 1: fit the calibration layer on TRAIN interpretations (leak-free) ----
    fit_rows = train[-train_fit_n:]
    samples, t_fit = [], time.time()
    for i, ex in enumerate(fit_rows):
        it = interpret_message(ex, chat_interp, meter=None)
        if it is not None:
            samples.append((it.features(), fm.base_p(ex), 1 if ex.replied else 0))
        if i % 40 == 0:
            print(f"  [fit] {i}/{len(fit_rows)} calls={meter['calls']}", flush=True)
    pol = fit_action_policy(samples)
    print(f"fitted policy on n={pol.n_train} (abstained={len(fit_rows) - len(samples)}); "
          f"w_anchor={pol.w_anchor} b={pol.b}\n  w={pol.w}", flush=True)

    def hz_cum(p_base, j):
        return p_base * (sum(fm.hazard[:j + 2]) / max(1e-6, sum(fm.hazard)))

    def e3_direct(ex):
        pr = parse_json(metered(_scalar_raw, _CONTENT_PROMPT.format(
            recipient=ex.recipient, sender=ex.sender, pair_n=ex.feats["pair_n"],
            pair_rate=ex.feats["pair_rate"], inbox_7d=ex.feats["inbox_7d"],
            subject=ex.subject[:200], body=ex.body[:1500]))) or {}
        try:
            return min(0.97, max(0.01, float(pr["reply_propensity"])))
        except (KeyError, TypeError, ValueError):
            return fm.global_rate

    # ---- PHASE 2: identical held-out rows, all arms ----
    def eval_split(test, tag, cap):
        rows, lat = [], {"interp_s": 0.0, "sim_s": 0.0}
        for i, ex in enumerate(test[:cap]):
            y = {b: (1.0 if (ex.replied and ex.delay_days is not None and ex.delay_days <= b) else 0.0)
                 for b in BUCKS}
            row = {"y": y, "recipient": ex.recipient}
            row["E0"] = {b: min(0.97, max(0.02, fm.global_rate)) for b in BUCKS}
            row["E1"] = {b: hz_cum(fm.base_p(ex), j) for j, b in enumerate(BUCKS)}
            row["E2"] = {b: hz_cum(text_baseline_p(ex, tb), j) for j, b in enumerate(BUCKS)}
            row["E3"] = {b: hz_cum(e3_direct(ex), j) for j, b in enumerate(BUCKS)}
            # E4: call-matched ensemble — 3 scalar reads with small prompt jitter
            vals = []
            for k in (0, 7, 13):
                pr4 = parse_json(metered(_scalar_raw, _CONTENT_PROMPT.format(
                    recipient=ex.recipient, sender=ex.sender, pair_n=ex.feats["pair_n"],
                    pair_rate=ex.feats["pair_rate"], inbox_7d=ex.feats["inbox_7d"],
                    subject=ex.subject[:200 - k], body=ex.body[:1500]))) or {}
                try:
                    vals.append(min(0.97, max(0.01, float(pr4["reply_propensity"]))))
                except (KeyError, TypeError, ValueError):
                    vals.append(fm.global_rate)
            row["E4"] = {b: hz_cum(sum(vals) / len(vals), j) for j, b in enumerate(BUCKS)}
            # C0: the scalar-content world (previous round's E10)
            o0 = v2_predict(ex, fm, n_particles=24, seed=i, content_fn=chat_scalar, meter=None)
            row["C0"] = {b: o0["p_by"].get(b, o0["p14"]) for b in BUCKS}
            # ONE interpretation, shared by C1–C6
            t_i = time.time()
            interp = interpret_message(ex, chat_interp, meter=None)
            lat["interp_s"] += time.time() - t_i
            p_c1 = pol.p_engage(interp.features(), fm.base_p(ex)) if interp else fm.base_p(ex)
            row["C1"] = {b: hz_cum(p_c1, j) for j, b in enumerate(BUCKS)}
            row["_interp"] = interp.as_dict() if interp else None
            t_s = time.time()
            for lvl in (2, 3, 4, 5, 6):
                o = v2_predict_actor(ex, fm, interp, pol, level=lvl, n_particles=n_particles,
                                     seed=i, hetero_sd=hetero_sd)
                row[f"C{lvl}"] = {b: o["p_by"].get(b, o["p14"]) for b in BUCKS}
                if lvl == 6:
                    row["_c6"] = {"p_engage": round(o["p_engage"], 4),
                                  "action_dist": {k: round(v, 4) for k, v in o["action_dist"].items()},
                                  "terminal_actions": o["terminal_actions"][:12]}
            lat["sim_s"] += time.time() - t_s
            rows.append(row)
            if i % 20 == 0:
                print(f"  [{tag}] {i}/{min(cap, len(test))} calls={meter['calls']}", flush=True)
        return rows, lat

    ARMS = ["E0", "E1", "E2", "E3", "E4", "C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    out = {}
    for tag, test, cap in (("time_forward", test_seen, llm_n), ("person_disjoint", test_new, llm_n // 2)):
        rows, lat = eval_split(test, tag, cap)
        detail = {a: _metrics(rows, a, 7.0) for a in ARMS}
        raw = {a: {f"brier@{int(b)}d": _metrics(rows, a, b).get("brier") for b in BUCKS} for a in ARMS}
        pairs = {f"{a}_vs_E1": _paired(rows, a, "E1", 7.0) for a in ("C0", "C1", "C6")}
        pairs.update({"C6_vs_C0": _paired(rows, "C6", "C0", 7.0),
                      "C6_vs_E3": _paired(rows, "C6", "E3", 7.0),
                      "C1_vs_E2": _paired(rows, "C1", "E2", 7.0)})
        ladder = {f"C{l}_vs_C{l - 1}": _paired(rows, f"C{l}", f"C{l - 1}", 7.0) for l in (2, 3, 4, 5, 6)}
        out[tag] = {"n": len(rows), "detail@7d": detail, "raw_brier": raw,
                    "paired": pairs, "ladder": ladder,
                    "latency_s": {k: round(v, 1) for k, v in lat.items()},
                    "rows_forensic": rows[:20] if tag == "time_forward" else None}
        print(f"\n== {tag} n={len(rows)} ==")
        for a in ARMS:
            m = detail[a]
            if m:
                print(f"  {a:3s} @7d: brier={m['brier']} logloss={m['logloss']} auroc={m['auroc']} "
                      f"pr_auc={m['pr_auc']} ece={m['ece']}")
        for k, v in {**pairs, **ladder}.items():
            if v:
                print(f"  {k}: Δ={v['mean']:+.5f} CI{v['ci95']}")

    forensic = out["time_forward"].pop("rows_forensic")
    out["_meta"] = {"llm_calls": meter["calls"], "llm_tokens_est": meter["tokens"],
                    "est_cost_usd": round(meter["tokens"] * (DS_IN + DS_OUT) / 2, 4),
                    "model_name": "deepseek-chat (DeepSeek V3)",
                    "runtime_s": round(time.time() - t0, 1), "n_particles": n_particles,
                    "hetero_sd_fitted": round(hetero_sd, 4),
                    "policy": {"w": pol.w, "w_anchor": pol.w_anchor, "b": pol.b, "n_train": pol.n_train,
                               "status": pol.status},
                    "arm_legend": {
                        "C0": "V2_SCALAR_CONTENT (prev round's E10 — scalar bottleneck, DEMOTED to baseline)",
                        "C1": "structured interpretation → fitted layer (closed form)",
                        "C2": "+typed actions in the event world", "C3": "+sampled hidden actor state",
                        "C4": "+dynamic attention", "C5": "+relationship state",
                        "C6": "MAXIMUM STRUCTURED ACTOR MODEL"}}
    out["person_disjoint"].pop("rows_forensic", None)
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    Path(FORENSIC).write_text(json.dumps({"note": "first 20 time-forward rows, all arms + interpretation",
                                          "rows": forensic}, indent=1, default=str))
    print(f"\nwrote {RESULT} + {FORENSIC} (llm calls={meter['calls']}, ~${out['_meta']['est_cost_usd']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--llm-n", type=int, default=120)
    ap.add_argument("--particles", type=int, default=48)
    ap.add_argument("--train-fit-n", type=int, default=300)
    a = ap.parse_args()
    run(a.limit, a.llm_n, a.particles, a.train_fit_n)
