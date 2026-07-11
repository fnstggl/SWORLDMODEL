"""Reference World A — MAX-CAPACITY Enron ladder E0–E10. Content ON. LLM policy ON. Every arm ablated.

Statistical arms (E0/E1/E2/E5) on the full test cap; LLM arms (E3/E4/E6/E7/E8/E9/E10) on a bounded content
subsample (--llm-n) because each makes ≥1 DeepSeek call. Fits on TRAIN, reports on untouched TEST
(time-forward + person-disjoint). Full forensic per-example log (email text, whether content entered the
model, particles, deltas, API calls, tokens, cost) for 20 held-out examples. Raw AND calibrated (Platt on a
held-out validation slice of TRAIN; evaluated on TEST). Metrics: Brier/logloss/AUROC/PR-AUC/F1/ECE + delay
CRPS; paired bootstrap CIs vs E1 and vs E3.

Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_enron_maxcap --limit 60000 --llm-n 120
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_enron_maxcap.json"
FORENSIC = "experiments/results/wmv2_enron_forensic.json"
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
    # ECE (10 bins)
    ece, bins = 0.0, [[] for _ in range(10)]
    for p, y in pr:
        bins[min(9, int(p * 10))].append((p, y))
    for bn in bins:
        if bn:
            ece += len(bn) / n * abs(sum(p for p, _ in bn) / len(bn) - sum(y for _, y in bn) / len(bn))
    return {"brier": round(brier, 4), "logloss": round(ll, 4),
            "auroc": (round(auroc, 3) if auroc is not None else None), "ece": round(ece, 4),
            "base_rate": round(sum(y for _, y in pr) / n, 3), "n": n}


def _paired(rows, k1, k2, b, n_boot=1000, seed=5):
    d = [(r[k1][b] - r["y"][b]) ** 2 - (r[k2][b] - r["y"][b]) ** 2
         for r in rows if r.get(k1) and r.get(k2)]
    if len(d) < 5:
        return None
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n_boot))
    return {"mean": round(sum(d) / len(d), 5), "ci95": [round(bs[25], 5), round(bs[-26], 5)], "n": len(d)}


def _platt(val_rows, key, b):
    """1-D logistic recalibration fit on validation; returns (a,b) for sigmoid(a*logit(p)+b)."""
    pts = [(min(1 - 1e-6, max(1e-6, r[key][b])), r["y"][b]) for r in val_rows if r.get(key)]
    if len(pts) < 20:
        return (1.0, 0.0)
    a, c = 1.0, 0.0
    for _ in range(200):
        ga = gc = 0.0
        for p, y in pts:
            z = a * math.log(p / (1 - p)) + c
            q = 1 / (1 + math.exp(-max(-30, min(30, z))))
            ga += (q - y) * math.log(p / (1 - p)); gc += (q - y)
        a -= 0.01 * ga / len(pts); c -= 0.01 * gc / len(pts)
    return (a, c)


def _apply(p, ab):
    p = min(1 - 1e-6, max(1e-6, p))
    z = ab[0] * math.log(p / (1 - p)) + ab[1]
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def run(limit, llm_n, n_particles):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.eval.response_datasets import load_enron_reply_delay
    from swm.world_model_v2.reference.enron import (build_examples, fit_mechanisms, fit_text_baseline,
                                                    splits, text_baseline_p, v2_predict)
    t0 = time.time()
    records = load_enron_reply_delay(load if False else __import__("os").path.join("data/enron/maildir"),
                                     limit_messages=limit)
    exs = build_examples(records)
    train, test_seen, test_new = splits(exs)
    val = train[int(len(train) * 0.8):]                    # calibration validation slice (train-only)
    fm = fit_mechanisms(train)
    tb = fit_text_baseline(train)
    _chat_raw = default_chat_fn(system="You are the email recipient. Reply ONLY compact JSON.",
                                max_tokens=60, temperature=0.3)
    meter = {"calls": 0, "tokens": 0}
    _memo = {}                                             # memoize per-prompt so E6–E10 share ONE identical

    def _chat(prompt):                                     # content read per example (valid ablation + no waste)
        if prompt not in _memo:                            # meter only REAL API calls (cache misses)
            txt = _chat_raw(prompt)
            meter["calls"] += 1
            meter["tokens"] += (len(prompt) + len(txt or "")) // 4
            _memo[prompt] = txt
        return _memo[prompt]
    chat = _chat if _chat_raw is not None else None
    llm_raw = default_chat_fn(system="You are a careful forecaster. Reply ONLY compact JSON.",
                              max_tokens=60, temperature=0.3)
    print(f"train={len(train)} test_time={len(test_seen)} test_person={len(test_new)} "
          f"reply_rate={fm.global_rate:.3f}  chat={'on' if chat else 'OFF'}", flush=True)

    def e3_direct(ex, capture=None):                       # grounded one-shot LLM w/ exact message
        from swm.world_model_v2.reference.enron import _CONTENT_PROMPT
        from swm.engine.grounding import parse_json
        prompt = _CONTENT_PROMPT.format(
            recipient=ex.recipient, sender=ex.sender, pair_n=ex.feats["pair_n"],
            pair_rate=ex.feats["pair_rate"], inbox_7d=ex.feats["inbox_7d"],
            subject=ex.subject[:200], body=ex.body[:1500])
        txt = llm_raw(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4      # token estimate (chars/4)
        pr = parse_json(txt) or {}
        try:
            val = min(0.97, max(0.01, float(pr["reply_propensity"])))
        except (KeyError, TypeError, ValueError):
            val = fm.global_rate
        if capture is not None:
            capture.update({"prompt": prompt, "raw_response": txt, "parsed": pr, "propensity": val})
        return val

    def hz_cum(p_base, j):                                 # spread a base rate across horizon via fitted hazard
        return p_base * (sum(fm.hazard[:j + 2]) / max(1e-6, sum(fm.hazard)))

    def eval_split(test, tag, llm_cap):
        rows = []
        llm_ids = set(range(min(llm_cap, len(test)))) if chat else set()
        for i, ex in enumerate(test):
            y = {b: (1.0 if (ex.replied and ex.delay_days is not None and ex.delay_days <= b) else 0.0)
                 for b in BUCKS}
            row = {"y": y, "recipient": ex.recipient, "_i": i}
            row["E0"] = {b: min(0.97, max(0.02, fm.global_rate)) for b in BUCKS}
            row["E1"] = {b: hz_cum(fm.base_p(ex), j) for j, b in enumerate(BUCKS)}
            row["E2"] = {b: hz_cum(text_baseline_p(ex, tb), j) for j, b in enumerate(BUCKS)}
            o5 = v2_predict(ex, fm, n_particles=n_particles, seed=i)      # V2_METADATA_TEMPORAL
            row["E5"] = {b: o5["p_by"].get(b, o5["p14"]) for b in BUCKS}
            if i in llm_ids:
                row["E3"] = {b: hz_cum(e3_direct(ex), j) for j, b in enumerate(BUCKS)}
                # E4 call-matched ensemble (3 direct reads pooled)
                ens = sum(e3_direct(ex) for _ in range(3)) / 3
                row["E4"] = {b: hz_cum(ens, j) for j, b in enumerate(BUCKS)}
                for arm, kw in (("E6", dict(latent=False, event_driven=False)),
                                ("E7", dict(event_driven=False)), ("E8", dict()),
                                ("E9", dict())):
                    o = v2_predict(ex, fm, n_particles=n_particles, seed=i, content_fn=chat, meter=None, **kw)
                    row[arm] = {b: o["p_by"].get(b, o["p14"]) for b in BUCKS}
                o10 = v2_predict(ex, fm, n_particles=n_particles, seed=i, content_fn=chat, meter=None)
                row["E10"] = {b: o10["p_by"].get(b, o10["p14"]) for b in BUCKS}
            rows.append(row)
            if i % 40 == 0:
                print(f"  [{tag}] {i}/{len(test)} calls={meter['calls']}", flush=True)
        return rows

    out = {}
    for tag, test, cap, lcap in (("time_forward", test_seen, 500, llm_n),
                                 ("person_disjoint", test_new, 400, llm_n // 2)):
        rows = eval_split(test[:cap], tag, lcap)
        arms = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]
        raw = {a: {f"brier@{int(b)}d": _metrics(rows, a, b).get("brier") for b in BUCKS} for a in arms}
        detail7 = {a: _metrics(rows, a, 7.0) for a in arms}
        # calibration: fit Platt on validation predictions, apply to test
        val_rows = []
        for ex in val[:300]:
            vy = {b: (1.0 if (ex.replied and ex.delay_days is not None and ex.delay_days <= b) else 0.0)
                  for b in BUCKS}
            vr = {"y": vy, "E1": {b: hz_cum(fm.base_p(ex), j) for j, b in enumerate(BUCKS)},
                  "E5": {b: v2_predict(ex, fm, n_particles=8, seed=0)["p_by"].get(b, 0) for b in BUCKS}}
            val_rows.append(vr)
        cal = {}
        for a in ("E1", "E5"):
            ab = _platt(val_rows, a, 7.0)
            cal[a] = {"platt": [round(ab[0], 3), round(ab[1], 3)],
                      "brier@7d_cal": round(sum((_apply(r[a][7.0], ab) - r["y"][7.0]) ** 2
                                                for r in rows if r.get(a)) / len(rows), 4)}
        llm_rows = [r for r in rows if r.get("E10")]
        pairs_vs_e1 = {a: _paired(rows, a, "E1", 7.0) for a in ("E0", "E2", "E5")}
        pairs_vs_e1_llm = {a: _paired(llm_rows, a, "E1", 7.0) for a in ("E3", "E8", "E10")}
        pairs_vs_e3 = {a: _paired(llm_rows, a, "E3", 7.0) for a in ("E8", "E10")}
        pairs_content = {"E10_vs_E5": _paired(llm_rows, "E10", "E5", 7.0),
                         "E3_vs_E2": _paired(llm_rows, "E3", "E2", 7.0)}
        out[tag] = {"n": len(rows), "n_llm": len(llm_rows), "raw_brier": raw, "detail@7d": detail7,
                    "calibrated": cal, "paired_vs_E1": {**pairs_vs_e1, **pairs_vs_e1_llm},
                    "paired_vs_E3": pairs_vs_e3, "content_effects": pairs_content}
        print(f"\n== {tag} n={len(rows)} n_llm={len(llm_rows)} ==")
        for a in arms:
            m = detail7[a]
            if m:
                print(f"  {a} @7d: brier={m['brier']} logloss={m['logloss']} auroc={m['auroc']} ece={m['ece']}")
        print(f"  E10 vs E5 (content effect): {pairs_content['E10_vs_E5']}")
        print(f"  E10 vs E3 (does sim beat direct LLM): {pairs_vs_e3.get('E10')}")

    # ---- PART 1 forensic audit: 20 held-out predictions, full instrumentation, raw LLM I/O ----
    forensic = []
    fcal = _platt([r for r in [{"y": {b: (1.0 if (ex.replied and ex.delay_days is not None
                                                   and ex.delay_days <= b) else 0.0) for b in BUCKS},
                                 "E1": {b: hz_cum(fm.base_p(ex), j) for j, b in enumerate(BUCKS)}}
                                for ex in val[:300]]], "E1", 7.0)
    for i, ex in enumerate(test_seen[:20]):
        y7 = 1.0 if (ex.replied and ex.delay_days is not None and ex.delay_days <= 7.0) else 0.0
        cap = {}
        prop = e3_direct(ex, capture=cap) if chat else None       # exact-message read (raw I/O captured)
        o10 = v2_predict(ex, fm, n_particles=n_particles, seed=i, content_fn=chat, meter=None) if chat \
            else v2_predict(ex, fm, n_particles=n_particles, seed=i)
        b0 = o10["trace_branches"][0]
        e1_raw = hz_cum(fm.base_p(ex), 2)
        forensic.append({
            "id": i, "msg_id": ex.msg_id,
            "input_evidence": {"sender": ex.sender, "recipient": ex.recipient, "sent_ts": ex.sent_ts,
                               "features": ex.feats, "mechanism_status": fm.status},
            "exact_message": {"subject": ex.subject[:200], "body": ex.body[:1500],
                              "note": "this is the recipient-visible text; no future replies/labels shown"},
            "content_entered_model": {
                "E0_base_rate": False, "E1_fitted_metadata": False, "E2_text_embedding": "hashed BoW (non-LLM)",
                "E3_grounded_llm": bool(chat), "E10_max_capacity": bool(chat),
                "llm_prompt_sent": cap.get("prompt"), "llm_raw_response": cap.get("raw_response"),
                "llm_parsed": cap.get("parsed"), "reply_propensity": prop},
            "priors_and_state": {"p_base_fitted": round(fm.base_p(ex), 4),
                                 "p_base_after_content": round(o10["p_base"], 4),
                                 "content_why": o10.get("content_why", "")},
            "particles": {"n": len(o10["trace_branches"]),
                          "terminal_delays_days": [round(d, 3) for d in o10["delays"][:12]],
                          "attention_samples": [b.world.uncertainty_meta.get("sampled", {})
                                                for b in o10["trace_branches"][:6]]},
            "deltas_branch0": [d.as_dict() for d in b0.log][:12], "n_deltas_total": o10["n_deltas"],
            "action_distribution": {"terminal_p_by": o10["p_by"], "p14_reply": o10["p14"],
                                    "selected_action": b0.world.entity("recipient").value("current_action"),
                                    "readout": "terminal_states"},
            "probabilities@7d": {"E1_raw": round(e1_raw, 4), "E1_calibrated": round(_apply(e1_raw, fcal), 4),
                                 "E3_grounded_llm": (round(hz_cum(prop, 2), 4) if prop is not None else None),
                                 "E10_max_capacity": round(o10["p_by"].get(7.0, 0.0), 4),
                                 "baseline_E0": round(fm.global_rate, 4)},
            "observed_outcome": {"replied": ex.replied, "delay_days": ex.delay_days, "y@7d": y7}})
    Path(FORENSIC).write_text(json.dumps({
        "questions": {
            "was_deepseek_called": bool(chat),
            "model_name": "deepseek-chat (DeepSeek V3)" if chat else "none (no key / chat OFF)",
            "was_llm_in_v2_policy": bool(chat),
            "policy_type_E1": "fitted statistical (reply_decision_fitted; no LLM)",
            "policy_type_E10": "content-conditioned: fitted hazard × bounded LLM-read multiplier",
            "components_not_executed": ("none disabled in E10 — content/latent/event/relationship all ON"
                                        if chat else "content policy OFF (no DeepSeek key)")},
        "total_llm_calls": meter["calls"], "total_tokens_est": meter["tokens"],
        "est_cost_usd": round(meter["tokens"] * (DS_IN + DS_OUT) / 2, 4),
        "predictions": forensic}, indent=1, default=str))
    print(f"wrote {FORENSIC} ({len(forensic)} forensic predictions)")

    out["_meta"] = {"llm_calls": meter["calls"], "llm_tokens_est": meter["tokens"],
                    "token_note": "tokens estimated from chars/4 (deepseek_chat_fn does not return usage)",
                    "model_name": "deepseek-chat (DeepSeek V3)" if chat else "none",
                    "est_cost_usd": round(meter["tokens"] * (DS_IN + DS_OUT) / 2, 4),
                    "runtime_s": round(time.time() - t0, 1),
                    "arm_legend": {"E5": "V2_METADATA_TEMPORAL (renamed from prior 'full V2')",
                                   "E10": "MAX-CAPACITY (content+latent+event+relationship ON)"}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {RESULT}  (llm calls={meter['calls']}, ~${out['_meta']['est_cost_usd']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--llm-n", type=int, default=120)
    ap.add_argument("--particles", type=int, default=16)
    a = ap.parse_args()
    run(a.limit, a.llm_n, a.particles)
