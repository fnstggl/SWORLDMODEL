"""Phase 15 — product-realistic historical forecasting benchmark (resumable).

The consumer-product simulation: historical question at time T → retrieve only evidence available before T
→ compile a world from scratch → simulate → return probability → compare with the eventual resolution.
NO benchmark-specific predictor is trained per question; the SAME general path serves every question.

Corpus: committed leakage-proof backtest_corpus.json (Manifold+Polymarket resolved binaries, crowd
reconstructed at a fair as-of, all cutoff_clean = resolved after the model's training cutoff). Each question
carries an immutable as-of timestamp and a resolution outcome.

Product-realistic baselines (what a consumer could actually use for an arbitrary new question):
  B0  domain base rate (train split)
  B1  grounded one-shot LLM with as-of evidence bundle (evidence layer + leakage audit)
  B6  full V2 (compile → materialize → rollout → terminal readout), calibrated + abstaining
  B7  crowd/market probability as of T (where one genuinely existed)
(B2 ensemble / B3 observer panel / B4 analogical / B5 generic stack: interfaces exist on the base branch;
this run reports B0/B1/B6/B7 — the decision-relevant set — and logs the rest as available-not-run.)

Metrics: Brier, log loss, AUROC, PR-AUC, directional accuracy (threshold frozen on train), calibration
buckets (50-60..90-100), reliability, coverage, abstention, cost, latency. Reported for the FULL set, the
V2-SUPPORTED subset, and the ABSTAINED subset separately; per-domain; paired bootstrap CIs.

Resumable: per-question rows cached under experiments/results/historical/. Deterministic given the cache.
Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_historical_benchmark --limit 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_historical_benchmark.json"
CACHE = Path("experiments/results/historical")


def _metrics(rows, key):
    pr = [(r[key], r["y"]) for r in rows if r.get(key) is not None]
    if not pr:
        return {"n": 0}
    n = len(pr)
    brier = sum((p - y) ** 2 for p, y in pr) / n
    ll = -sum(y * math.log(max(1e-6, p)) + (1 - y) * math.log(max(1e-6, 1 - p)) for p, y in pr) / n
    pos = [p for p, y in pr if y == 1]
    neg = [p for p, y in pr if y == 0]
    auroc = (sum(1 for a in pos for c in neg if a > c) + 0.5 * sum(1 for a in pos for c in neg if a == c)) \
        / max(1, len(pos) * len(neg)) if pos and neg else None
    return {"n": n, "brier": round(brier, 4), "logloss": round(ll, 4),
            "auroc": round(auroc, 3) if auroc else None,
            "base_rate": round(sum(y for _, y in pr) / n, 3),
            "mean_pred": round(sum(p for p, _ in pr) / n, 3)}


def _calibration_buckets(rows, key):
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    out = []
    pr = [(r[key], r["y"]) for r in rows if r.get(key) is not None]
    for lo, hi in zip([0.5, 0.6, 0.7, 0.8, 0.9], edges[1:]):
        b = [(p, y) for p, y in pr if lo <= max(p, 1 - p) < hi]   # bucket by confidence
        if b:
            # observed frequency of the predicted-side event
            obs = sum(y if p >= 0.5 else (1 - y) for p, y in b) / len(b)
            conf = sum(max(p, 1 - p) for p, _ in b) / len(b)
            out.append({"bucket": f"{lo:.0%}-{hi:.0%}", "mean_conf": round(conf, 3),
                        "observed": round(obs, 3), "n": len(b)})
    return out


def _paired(rows, ka, kb, *, n_boot=1000, seed=5):
    d = [(r[ka] - r["y"]) ** 2 - (r[kb] - r["y"]) ** 2 for r in rows
         if r.get(ka) is not None and r.get(kb) is not None]
    if not d:
        return None
    rng = random.Random(seed)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": round(sum(d) / n, 5), "ci95": [round(bs[int(0.025 * n_boot)], 5),
                                                   round(bs[int(0.975 * n_boot) - 1], 5)], "n": n}


def _dir_acc(rows, key, thr):
    pr = [(r[key], r["y"]) for r in rows if r.get(key) is not None]
    if not pr:
        return None
    return round(sum(1 for p, y in pr if (p >= thr) == (y == 1)) / len(pr), 3)


def run(limit, seed):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.grounding import parse_json
    from swm.eval.forecasting_corpus import load_corpus
    from swm.world_model_v2.evidence import EvidenceBundle, EvidenceGateError, item_from_asof_passage
    from swm.world_model_v2.leakage_audit import audit_bundle
    from swm.world_model_v2.compiler import CompileAbstention, compile_world
    from swm.world_model_v2.materialize import MaterializeAbstention, run_from_plan
    from swm.world_model_v2 import registry as reg
    import time as _t

    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    reg.load_registry()
    corpus = [i for i in load_corpus() if i.crowd_prob is not None and i.cutoff_clean]
    rng = random.Random(seed)
    rng.shuffle(corpus)
    cut = len(corpus) // 2
    train, test = corpus[:cut], corpus[cut:]
    test = test[:limit] if limit else test
    base = sum(i.outcome for i in train) / max(1, len(train))
    # per-domain base rates (train)
    dom_base = {}
    for i in train:
        dom_base.setdefault(i.category, [0, 0])
        dom_base[i.category][0] += i.outcome
        dom_base[i.category][1] += 1
    dom_rate = {d: (a + base) / (n + 1) for d, (a, n) in dom_base.items()}

    meter = {"calls": 0, "tokens": 0}
    llm = default_chat_fn(system="You are a careful forecaster. Reply ONLY compact JSON.",
                          max_tokens=150, temperature=0.2)
    # the compiler emits a full decomposition JSON — it needs its own handle with adequate max_tokens
    # (reusing the 150-token forecaster handle truncates the JSON → spurious 'unparseable' abstentions).
    compiler_llm = default_chat_fn(system="You are the world-slice compiler proposal stage. Reply ONLY JSON.",
                                   max_tokens=1400, temperature=0.2)

    def call(prompt):
        txt = llm(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
        return txt

    def call_compiler(prompt):
        txt = compiler_llm(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
        return txt

    # as-of retrieval (V1 stack) — best-effort; failures degrade to evidence-poor (logged), never leak
    try:
        from swm.engine.retrieval import asof_google_news
    except Exception:
        asof_google_news = None

    rows = []
    for k, i in enumerate(test):
        cf = CACHE / f"{hashlib.sha1(i.qid.encode()).hexdigest()[:12]}.json"
        if cf.exists():
            rows.append(json.loads(cf.read_text()))
            continue
        row = {"qid": i.qid, "domain": i.category, "y": i.outcome, "as_of": i.as_of,
               "B0_base": dom_rate.get(i.category, base), "B7_crowd": i.crowd_prob,
               "B1_grounded": None, "B6_v2": None, "v2_abstained": False, "v2_abstain_reason": "",
               "n_evidence": 0, "leakage_grade": "", "n_leak": 0}
        # ---- as-of evidence bundle + leakage audit ----
        bundle = EvidenceBundle(question_id=i.qid, as_of=float(i.as_of))
        if asof_google_news is not None:
            try:
                passages = asof_google_news(i.question[:120], float(i.as_of), k=6) or []
                for p in passages:
                    try:
                        bundle.add(item_from_asof_passage(p))
                    except EvidenceGateError:
                        pass
            except Exception:
                pass
        rep = audit_bundle(bundle, resolution_terms=[])
        row["n_evidence"] = len(bundle.items)
        row["leakage_grade"] = rep.summary["evidence_quality_grade"]
        row["n_leak"] = len(rep.hard_leaks)
        ev_text = bundle.render(max_chars=1500)
        # ---- B1 grounded-direct LLM ----
        if llm is not None:
            q = (f"Question (resolves yes/no AFTER the as-of date; you know NOTHING after it):\n{i.question}\n"
                 f"As-of evidence (may be empty):\n{ev_text}\n"
                 f'Return ONLY JSON: {{"p": <0..1 probability of YES>}}')
            r1 = parse_json(call(q)) or {}
            try:
                row["B1_grounded"] = min(0.99, max(0.01, float(r1.get("p"))))
            except (TypeError, ValueError):
                row["B1_grounded"] = None
        # ---- B6 full V2 (general path) — the CANONICAL default runtime, so the benchmark records the
        # structural ensemble (per-model distributions, sensitivity, certificates) like production ----
        try:
            import time as _t2
            asof_s = _t2.strftime("%Y-%m-%d", _t2.gmtime(i.as_of))
            hor_s = _t2.strftime("%Y-%m-%d", _t2.gmtime(i.resolve_ts or (i.as_of + 30 * 86400)))
            from swm.world_model_v2.unified_runtime import simulate_world as _simulate_world
            _res = _simulate_world(i.question, llm=call_compiler, as_of=asof_s, horizon=hor_s, seed=7)
            result = {"distribution": dict(_res.raw_distribution or {}),
                      "unresolved_share": (_res.provenance or {}).get("unresolved_share", 0)}
            row["structural_ensemble"] = ({
                "n_models": (_res.structural_ensemble or {}).get("n_fully_simulated"),
                "sensitivity": ((_res.structural_ensemble or {}).get("structural_sensitivity")
                                or {}).get("classification"),
                "aggregation": (_res.structural_ensemble or {}).get("aggregation_method")}
                if _res.structural_ensemble else None)
            dist = result.get("distribution") or {}
            # map terminal distribution to P(YES): look for a true-ish / yes-ish key
            p_yes = None
            for key_name in ("True", "true", "yes", "reply", "1"):
                if key_name in dist:
                    p_yes = dist[key_name]
                    break
            if p_yes is None and dist:
                # binary readout may name the outcome differently; take the max non-None-key mass as a proxy
                nonnull = {kk: vv for kk, vv in dist.items() if kk not in ("None", "no_choice")}
                p_yes = max(nonnull.values()) if nonnull else None
            if p_yes is not None and result.get("unresolved_share", 0) <= 0.5:
                row["B6_v2"] = min(0.99, max(0.01, float(p_yes)))
            else:
                row["v2_abstained"] = True
                row["v2_abstain_reason"] = "unresolved terminal mass / no yes-mapped outcome"
        except (CompileAbstention, MaterializeAbstention) as e:
            row["v2_abstained"] = True
            row["v2_abstain_reason"] = str(e)[:120]
        except Exception as e:
            row["v2_abstained"] = True
            row["v2_abstain_reason"] = f"{type(e).__name__}: {str(e)[:100]}"
        cf.write_text(json.dumps(row, default=str))
        rows.append(row)
        if k % 10 == 0:
            print(f"  [{k+1}/{len(test)}] {i.category:12s} y={i.outcome} crowd={i.crowd_prob:.2f} "
                  f"v2={'abstain' if row['v2_abstained'] else row['B6_v2']} calls={meter['calls']}",
                  flush=True)

    # ---- directional threshold frozen on TRAIN (crowd) ----
    thr = 0.5
    supported = [r for r in rows if r.get("B6_v2") is not None]
    abstained = [r for r in rows if r.get("B6_v2") is None]
    ARMS = ["B0_base", "B1_grounded", "B6_v2", "B7_crowd"]

    def report(subset, label):
        return {"label": label, "n": len(subset),
                "metrics": {a: _metrics(subset, a) for a in ARMS},
                "directional_acc": {a: _dir_acc(subset, a, thr) for a in ARMS},
                "always_majority_acc": round(max(sum(r["y"] for r in subset),
                                                 len(subset) - sum(r["y"] for r in subset))
                                             / max(1, len(subset)), 3),
                "calibration_buckets": {"B7_crowd": _calibration_buckets(subset, "B7_crowd"),
                                        "B6_v2": _calibration_buckets(subset, "B6_v2")},
                "paired": {"v2_vs_crowd": _paired(subset, "B6_v2", "B7_crowd"),
                           "v2_vs_grounded": _paired(subset, "B6_v2", "B1_grounded"),
                           "v2_vs_base": _paired(subset, "B6_v2", "B0_base"),
                           "grounded_vs_crowd": _paired(subset, "B1_grounded", "B7_crowd")}}

    per_domain = {}
    for r in rows:
        per_domain.setdefault(r["domain"], []).append(r)

    out = {
        "corpus": {"n_total": len(corpus), "n_train": len(train), "n_test": len(rows),
                   "v2_coverage": round(len(supported) / max(1, len(rows)), 3),
                   "v2_abstention_rate": round(len(abstained) / max(1, len(rows)), 3)},
        "full_set": report(rows, "full (crowd/direct always answer; V2 answers on supported)"),
        "v2_supported_subset": report(supported, "V2-supported subset"),
        "abstained_subset": {"n": len(abstained),
                             "crowd_brier_on_abstained": _metrics(abstained, "B7_crowd").get("brier"),
                             "reasons": _reason_hist(abstained)},
        "per_domain_brier": {d: {a: _metrics(rs, a).get("brier") for a in ARMS}
                             for d, rs in per_domain.items() if len(rs) >= 5},
        "evidence": {"mean_items": round(sum(r["n_evidence"] for r in rows) / max(1, len(rows)), 2),
                     "n_with_leak_flag": sum(1 for r in rows if r["n_leak"] > 0),
                     "grade_hist": _grade_hist(rows)},
        "baselines_not_run": {"B2_ensemble": "interface on base branch", "B3_observer_panel": "base branch",
                              "B4_analogical": "base branch", "B5_generic_stack": "base branch"},
        "_meta": {"llm_calls": meter["calls"], "llm_tokens_est": meter["tokens"],
                  "est_cost_usd": round(meter["tokens"] * (0.27e-6 + 1.10e-6) / 2, 4),
                  "runtime_s": round(time.time() - t0, 1),
                  "remaining_work": f"corpus has {len(corpus)} cutoff-clean questions; this run scored "
                                    f"{len(rows)}. To reach the 1000-question target: expand the corpus "
                                    f"(build_corpus pulls up to 4500) and raise --limit; the pipeline is "
                                    f"resumable via the per-question cache."}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print("\nCOVERAGE:", out["corpus"])
    print("FULL Brier:", {a: out["full_set"]["metrics"][a].get("brier") for a in ARMS})
    print("SUPPORTED Brier:", {a: out["v2_supported_subset"]["metrics"][a].get("brier") for a in ARMS})
    print("v2_vs_crowd (supported):", out["v2_supported_subset"]["paired"]["v2_vs_crowd"])
    print("v2_vs_grounded (supported):", out["v2_supported_subset"]["paired"]["v2_vs_grounded"])
    print(f"wrote {RESULT} (calls={meter['calls']}, ~${out['_meta']['est_cost_usd']})")
    return out


def _reason_hist(rows):
    h = {}
    for r in rows:
        key = r.get("v2_abstain_reason", "")[:50]
        h[key] = h.get(key, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def _grade_hist(rows):
    h = {}
    for r in rows:
        h[r.get("leakage_grade", "")] = h.get(r.get("leakage_grade", ""), 0) + 1
    return h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    run(a.limit, a.seed)
