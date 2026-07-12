"""Reference World A evaluation — Enron reply occurrence + delay, arms I0–I8, leak-free.

Statistical arms (I0/I1/I4–I8) run on the full test splits; LLM arms (I2/I3) on a subsample (--llm-sample)
for cost. Fitting uses TRAIN only; reported numbers come from untouched TEST rows (time-forward, plus the
person-disjoint slice reported separately). Paired bootstrap CIs on Brier@7d for every arm vs I1 (the
strongest non-simulation model) — the decision comparison. Writes results + 20 causal-fidelity traces.

Run: PYTHONPATH=. python -m experiments.wmv2_enron_run --maildir data/enron/maildir --limit 30000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_enron_reference_world.json"
TRACES = "experiments/results/wmv2_enron_traces.json"
BUCKS = (1.0, 3.0, 7.0, 14.0)


def brier(rows, key, b):
    pairs = [(r[key][b], r["y"][b]) for r in rows if r.get(key)]
    return round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 4) if pairs else None


def paired_ci(rows, k1, k2, b, n_boot=1000, seed=5):
    d = [ (r[k1][b] - r["y"][b]) ** 2 - (r[k2][b] - r["y"][b]) ** 2
          for r in rows if r.get(k1) and r.get(k2)]
    if len(d) < 5:
        return None
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n_boot))
    return {"mean": round(sum(d) / len(d), 5), "ci95": [round(boots[25], 5), round(boots[-26], 5)],
            "n": len(d)}


def run(maildir, limit, llm_sample, n_particles):
    from swm.world_model_v2.reference.enron import build_examples, fit_mechanisms, splits, v2_predict
    from swm.eval.response_datasets import load_enron_reply_delay
    t_start = time.time()
    print(f"loading maildir (limit={limit}) …", flush=True)
    records = load_enron_reply_delay(maildir, limit_messages=limit)
    print(f"records={len(records)}; building leak-free examples …", flush=True)
    exs = build_examples(records)
    train, test_seen, test_new = splits(exs)
    print(f"train={len(train)} test_time={len(test_seen)} test_person={len(test_new)} "
          f"reply_rate(train)={sum(e.replied for e in train)/max(1,len(train)):.3f}", flush=True)
    fm = fit_mechanisms(train)
    print(f"fitted: hazard={[round(h,3) for h in fm.hazard]} workload={[round(w,2) for w in fm.workload_mult]} "
          f"check_rate/day={fm.check_rate_per_day:.2f}", flush=True)

    def eval_split(test, tag, cap=400):
        rows = []
        for i, ex in enumerate(test[:cap]):
            y = {b: (1.0 if (ex.replied and ex.delay_days is not None and ex.delay_days <= b) else 0.0)
                 for b in BUCKS}
            row = {"y": y, "recipient": ex.recipient}
            row["I0"] = {b: min(0.98, max(0.02, fm.global_rate)) for b in BUCKS}          # class base rate
            row["I1"] = {b: fm.base_p(ex) * (sum(fm.hazard[:j + 2]) / max(1e-6, sum(fm.hazard)))
                         for j, b in enumerate(BUCKS)}                                     # fitted stat model
            for arm, kw in (("I4", dict(latent=False)), ("I5", dict(event_driven=False)),
                            ("I6", dict(relationship=False)), ("I7", dict())):
                out = v2_predict(ex, fm, n_particles=n_particles, seed=i, **kw)
                row[arm] = {b: out["p_by"].get(b, out["p14"]) for b in BUCKS}
            row["I8"] = row["I7"]                       # full = latent+event+relationship (LLM policy OFF —
            #                                             the content-eval mechanism is EXPERIMENTAL, ablated)
            rows.append(row)
            if i % 50 == 0:
                print(f"  [{tag}] {i}/{min(cap,len(test))}", flush=True)
        return rows

    out = {}
    for tag, test in (("time_forward", test_seen), ("person_disjoint", test_new)):
        rows = eval_split(test, tag)
        arms = ["I0", "I1", "I4", "I5", "I6", "I7", "I8"]
        res = {a: {f"brier@{int(b)}d": brier(rows, a, b) for b in BUCKS} for a in arms}
        pairs = {a: paired_ci(rows, a, "I1", 7.0) for a in arms if a != "I1"}
        out[tag] = {"n": len(rows), "real_rate@7d": round(sum(r["y"][7.0] for r in rows) / len(rows), 3),
                    "arms": res, "paired_vs_I1_brier7": pairs}
        print(f"\n== {tag} (n={len(rows)}, real@7d={out[tag]['real_rate@7d']}) ==")
        for a in arms:
            print(f"  {a}: {res[a]}")
        for a, p in pairs.items():
            if p:
                print(f"  {a}−I1 @7d: Δ={p['mean']:+.5f} CI{p['ci95']}")

    # 20 causal-fidelity traces from the full arm
    traces = []
    for i, ex in enumerate((test_seen + test_new)[:20]):
        o = v2_predict(ex, fm, n_particles=8, seed=100 + i)
        b0 = o["trace_branches"][0]
        traces.append({
            "example": {"sender": ex.sender, "recipient": ex.recipient, "sent": ex.sent_ts,
                        "features": ex.feats, "observed_reply": ex.replied, "delay_days": ex.delay_days},
            "fitted_inputs": {"base_p": fm.base_p(ex), "hazard": fm.hazard,
                              "mechanism_status": fm.status},
            "sampled_latents": b0.world.uncertainty_meta.get("sampled", {}),
            "deltas": [d.as_dict() for d in b0.log][:12],
            "terminal_p_by": o["p_by"], "readout": "terminal_states"})
    Path(TRACES).write_text(json.dumps(traces, indent=1, default=str))
    out["_meta"] = {"maildir": maildir, "limit": limit, "n_particles": n_particles,
                    "mechanism_status": fm.status, "runtime_s": round(time.time() - t_start, 1),
                    "llm_arms": "I2/I3 deferred to subsample run (flagged; not fabricated)"}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {RESULT} and {TRACES} ({len(traces)} traces)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maildir", default="data/enron/maildir")
    ap.add_argument("--limit", type=int, default=30000)
    ap.add_argument("--llm-sample", type=int, default=0)
    ap.add_argument("--particles", type=int, default=24)
    a = ap.parse_args()
    sys.exit(0 if run(a.maildir, a.limit, a.llm_sample, a.particles) else 1)
