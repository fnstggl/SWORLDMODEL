"""QUARANTINED — this harness's "V2 arm" is a crowd-anchored RESCALING, NOT the simulator. Any claim
about world-model-v2 accuracy must run `unified_runtime.simulate_world` (see docs/WMV2_CANONICAL_PATH.md
and EXP-105); results from this file must never be reported as simulator performance.

ForecastBench-class V2-supported subset — resolved-market corpus, leak-free, paired (item 5).

Corpus: 661 cutoff-clean RESOLVED binary questions (Manifold/Polymarket) with the CROWD probability
reconstructed at the same as-of the model forecasts from. V2-SUPPORTED SUBSET: deadline-event questions
("will X happen by <resolve date>") where the temporal hazard mechanism structurally applies — the one V2
layer that is defensibly parameterized here WITHOUT an as-of retrieval pipeline. Coverage is reported;
everything else is logged as unsupported rather than faked.

Immutable split: shuffle(seed 13) → 50% train (fit Platt + the hazard-time exponent) / 50% test.

Arms (identical questions):
  F0 base rate (train)         F1 CROWD at as-of (the bar)         F2 crowd + Platt (fitted train)
  F3 grounded direct LLM (question text only — no retrieval; its evidence poverty is the point)
  F4 TRUE 3-call ensemble      V2 crowd-anchored temporal world: hazard-consistent time rescaling
     p_v2 = 1-(1-p_crowd_cal)^(g(elapsed_frac)) with g fitted on train — the deadline mechanism
Ablations: V2_no_temporal (= F2, the nesting) | V2_no_crowd_anchor (hazard on base rate only).
Institutional/population kernels: UNSUPPORTED here (no defensible whip/poll evidence at as-of) — logged.

Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_forecastbench_run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_forecastbench_subset.json"
DS_IN, DS_OUT = 0.27e-6, 1.10e-6
DEADLINE_PAT = re.compile(r"\bby\b|\bbefore\b|\buntil\b|\bin 20\d\d\b|\bthis (year|month|week)\b", re.I)


def _metrics(rows, key):
    pr = [(min(1, max(0, r[key])), r["y"]) for r in rows if r.get(key) is not None]
    if not pr:
        return {}
    n = len(pr)
    brier = sum((p - y) ** 2 for p, y in pr) / n
    ll = -sum(y * math.log(max(1e-6, p)) + (1 - y) * math.log(max(1e-6, 1 - p)) for p, y in pr) / n
    pos = [p for p, y in pr if y == 1]
    neg = [p for p, y in pr if y == 0]
    auroc = (sum(1 for a in pos for c in neg if a > c) + 0.5 * sum(1 for a in pos for c in neg if a == c)) \
        / max(1, len(pos) * len(neg)) if pos and neg else None
    return {"brier": round(brier, 4), "logloss": round(ll, 4),
            "auroc": round(auroc, 3) if auroc is not None else None, "n": n}


def _paired(rows, k1, k2, n_boot=1000, seed=5):
    d = [(r[k1] - r["y"]) ** 2 - (r[k2] - r["y"]) ** 2
         for r in rows if r.get(k1) is not None and r.get(k2) is not None]
    if len(d) < 5:
        return None
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n_boot))
    return {"mean": round(sum(d) / len(d), 5), "ci95": [round(bs[25], 5), round(bs[-26], 5)], "n": len(d)}


def _platt_fit(pairs, iters=300, lr=0.05):
    a, c = 1.0, 0.0
    for _ in range(iters):
        ga = gc = 0.0
        for p, y in pairs:
            p = min(1 - 1e-6, max(1e-6, p))
            z = a * math.log(p / (1 - p)) + c
            q = 1 / (1 + math.exp(-max(-30, min(30, z))))
            ga += (q - y) * math.log(p / (1 - p))
            gc += (q - y)
        a -= lr * ga / len(pairs)
        c -= lr * gc / len(pairs)
    return a, c


def _platt(p, ab):
    p = min(1 - 1e-6, max(1e-6, p))
    z = ab[0] * math.log(p / (1 - p)) + ab[1]
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def run(n_llm):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.grounding import parse_json
    from swm.eval.forecasting_corpus import load_corpus
    t0 = time.time()
    meter = {"calls": 0, "tokens": 0}
    llm = default_chat_fn(system="You are a careful forecaster. Reply ONLY compact JSON.",
                          max_tokens=120, temperature=0.3)
    llm_hot = default_chat_fn(system="You are a careful forecaster. Reply ONLY compact JSON.",
                              max_tokens=120, temperature=0.7)

    def call(fn, prompt):
        txt = fn(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
        return txt

    corpus = [i for i in load_corpus() if i.cutoff_clean and i.crowd_prob is not None]
    subset = [i for i in corpus if DEADLINE_PAT.search(i.question or "")
              and i.resolve_ts and i.as_of and i.resolve_ts > i.as_of]
    coverage = len(subset) / max(1, len(corpus))
    rng = random.Random(13)
    rng.shuffle(subset)
    cut = len(subset) // 2
    train, test = subset[:cut], subset[cut:]
    test = test[:n_llm] if n_llm else test
    base = sum(i.outcome for i in train) / max(1, len(train))
    print(f"corpus={len(corpus)} V2-supported deadline subset={len(subset)} (coverage {coverage:.0%}) "
          f"train={len(train)} test={len(test)} base={base:.3f}", flush=True)

    # fitted layers (train only): Platt on crowd; hazard-time exponent g on the calibrated crowd
    ab = _platt_fit([(i.crowd_prob, i.outcome) for i in train])

    def elapsed_frac(i):
        # as_of sits at AS_OF_FRAC of market life; elapsed fraction of the REMAINING window drives the
        # hazard rescale. Use as_of/resolve span (observed timestamps only).
        try:
            return min(0.95, max(0.05, i.as_of / max(1.0, i.resolve_ts)))
        except (TypeError, ZeroDivisionError):
            return 0.5

    def v2_p(i, g, anchor=True):
        p0 = _platt(i.crowd_prob, ab) if anchor else base
        # hazard-consistent deadline mechanism: treat p0 as P(event by deadline | info at as-of); the
        # exponent g reweights for remaining-time survival (g=1 → identity; fitted on train)
        rem = 1.0 - elapsed_frac(i)
        return min(0.99, max(0.01, 1.0 - (1.0 - p0) ** (g ** (1.0 - rem))))

    best_g, best_b = 1.0, None
    for g in (0.6, 0.8, 1.0, 1.25, 1.6, 2.0):
        b = sum((v2_p(i, g) - i.outcome) ** 2 for i in train) / len(train)
        if best_b is None or b < best_b:
            best_b, best_g = b, g
    print(f"fitted: platt={[round(x, 3) for x in ab]} hazard_g={best_g}", flush=True)

    rows = []
    for k, i in enumerate(test):
        row = {"y": i.outcome, "F0": base, "F1": i.crowd_prob, "F2": _platt(i.crowd_prob, ab),
               "V2": v2_p(i, best_g), "V2_no_temporal": _platt(i.crowd_prob, ab),
               "V2_no_crowd_anchor": v2_p(i, best_g, anchor=False)}
        if llm is not None:
            q = (f"Question (resolved yes/no after this date; you know NOTHING after the as-of date):\n"
                 f"{i.question}\nReturn ONLY JSON: {{\"p\": <0..1 probability of YES>}}")
            r3 = parse_json(call(llm, q)) or {}
            try:
                row["F3"] = min(0.99, max(0.01, float(r3.get("p"))))
            except (TypeError, ValueError):
                row["F3"] = None
            ens = []
            for _ in range(3):
                r4 = parse_json(call(llm_hot, q)) or {}
                try:
                    ens.append(min(0.99, max(0.01, float(r4.get("p")))))
                except (TypeError, ValueError):
                    continue
            row["F4"] = sum(ens) / len(ens) if ens else None
        rows.append(row)
        if k % 20 == 0:
            print(f"  [eval] {k}/{len(test)} calls={meter['calls']}", flush=True)

    ARMS = ["F0", "F1", "F2", "F3", "F4", "V2", "V2_no_temporal", "V2_no_crowd_anchor"]
    detail = {a: _metrics(rows, a) for a in ARMS}
    paired = {"V2_vs_F1_crowd": _paired(rows, "V2", "F1"), "V2_vs_F2_platt": _paired(rows, "V2", "F2"),
              "V2_vs_F3_direct": _paired(rows, "V2", "F3"), "V2_vs_F4_ens": _paired(rows, "V2", "F4"),
              "V2_temporal_effect": _paired(rows, "V2", "V2_no_temporal"),
              "crowd_anchor_effect": _paired(rows, "V2", "V2_no_crowd_anchor"),
              "F2_vs_F1_calibration": _paired(rows, "F2", "F1")}
    out = {"n_test": len(rows), "subset_coverage_of_corpus": round(coverage, 3),
           "detail": detail, "paired": paired,
           "unsupported_logged": {
               "institutional_kernels": "no defensible whip/poll evidence at as-of in this corpus",
               "population/diffusion": "no population ground truth attached to these questions",
               "retrieval-grounded direct": "corpus carries no as-of evidence snapshots; F3/F4 are "
                                            "question-text-only (their evidence poverty is reported, "
                                            "not hidden)"},
           "_meta": {"llm_calls": meter["calls"], "llm_tokens_est": meter["tokens"],
                     "est_cost_usd": round(meter["tokens"] * (DS_IN + DS_OUT) / 2, 4),
                     "model_name": "deepseek-chat (DeepSeek V3)" if llm else "none",
                     "runtime_s": round(time.time() - t0, 1)}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    for a in ARMS:
        m = detail[a]
        if m:
            print(f"  {a:19s} brier={m['brier']} logloss={m['logloss']} auroc={m['auroc']} n={m['n']}")
    for k2, v in paired.items():
        if v:
            print(f"  {k2}: Δ={v['mean']:+.5f} CI{v['ci95']}")
    print(f"wrote {RESULT} (calls={meter['calls']}, ~${out['_meta']['est_cost_usd']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-llm", type=int, default=120)
    a = ap.parse_args()
    run(a.n_llm)
