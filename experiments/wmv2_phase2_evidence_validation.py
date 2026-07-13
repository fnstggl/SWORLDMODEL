"""Phase 2 evidence — held-out end-to-end validation (real DeepSeek + LIVE Google News RSS).

Runs the full evidence-conditioned path (compile → typed requirements → live paired-RSS retrieval →
temporal verification → claims → entities → dependence → contradictions → visibility → leakage → immutable
bundle → evidence-conditioned recompile → materialization → rollout) over the held-out question bank, and
measures the Phase-2 acceptance gates:

  requirements_produced      = 1.00   every question yields typed evidence requirements
  nonempty_bundle            ≥ 0.90   a real evidence bundle where public evidence should exist
  paired_rss_share           ≥ 0.95   historical RSS queries carrying BOTH after: and before:
  paired_rss_violation       = 0.00   production historical queries missing an operator
  raw_persisted              = 1.00   raw RSS responses persisted (by hash)
  plan_diff_persisted        = 1.00   pre/post plan diff recorded
  material_change            ≥ 0.50   evidence causes a structural plan (or terminal) change
  forecast_abstention        = 0.00   weak evidence never blocks a forecast
  execution_failure          < 0.10   engineering failures only, taxonomy'd

Resumable per-question cache; metered (LLM calls + HTTP). Bundles persisted under
experiments/results/phase2_bundles/.  Run:
  DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_phase2_evidence_validation --limit 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from experiments.wmv2_compiler_generality import QUESTIONS, _expand_to_100

RESULT = "experiments/results/wmv2_phase2_evidence_validation.json"
CACHE = Path("experiments/results/phase2_evidence_validation")

#: domains about PRIVATE/internal matters where public contemporaneous evidence often should NOT exist —
#: a 0-doc bundle there is CORRECT, not a retrieval failure. Public-event domains carry the nonempty gate.
PRIVATE_DOMAINS = {"messaging", "best_action", "organizational_decision"}


def _digest(q, as_of):
    return hashlib.sha1(f"ph2|{q}|{as_of}".encode()).hexdigest()[:12]


def run(limit, lookback_days, verbose=True):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.world_model_v2 import registry as reg
    from swm.world_model_v2.evidence_connectors import RawContentStore, paired_dates_ok
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig
    from swm.world_model_v2.evidence_pipeline import simulate_with_evidence

    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    reg.load_registry()
    questions = _expand_to_100(QUESTIONS)
    questions = questions[:limit] if limit else questions
    meter = {"calls": 0, "tokens": 0}
    llm0 = default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)
    if llm0 is None:
        raise SystemExit("needs DEEPSEEK_API_KEY")

    def llm(prompt):
        txt = llm0(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
        return txt

    store = RawContentStore()
    cfg = OrchestratorConfig(lookback_days=lookback_days, verify_online=False, use_wikipedia=False,
                             max_items_per_query=8, max_requirements_retrieved=3, max_claim_docs=6)
    rows = []
    for i, (domain, q, as_of, horizon) in enumerate(questions):
        cache_f = CACHE / f"{_digest(q, as_of)}.json"
        if cache_f.exists():
            rows.append(json.loads(cache_f.read_text()))
            continue
        rec = {"domain": domain, "question": q, "as_of": as_of, "horizon": horizon,
               "requirements_produced": False, "n_requirements": 0, "n_documents": 0, "n_included_claims": 0,
               "n_independent_sources": 0, "n_contradictions": 0, "n_paired_rss": 0, "n_rss_traces": 0,
               "paired_rss_ok": True, "paired_rss_violation": False, "raw_persisted": True,
               "structural_changes": 0, "lean_only": None, "plan_hash_changed": None,
               "material_change": False, "simulation_status": "", "support_grade": "", "has_forecast": False,
               "raw_probability": None, "bundle_hash": "", "failure_taxonomy": "", "evidence_stage": "",
               "error": ""}
        t_q = time.time()
        try:
            res, art = simulate_with_evidence(q, llm=llm, as_of=as_of, horizon=horizon, config=cfg,
                                              store=store, seed=7)
            rec["simulation_status"] = res.simulation_status
            rec["failure_taxonomy"] = res.failure_taxonomy
            rec["evidence_stage"] = art.get("stage", "")
            rec["support_grade"] = res.support_grade
            rec["has_forecast"] = res.has_forecast() and bool(res.raw_distribution)
            rec["raw_probability"] = res.raw_probability
            b = art.get("bundle")
            if b is not None:
                rec["requirements_produced"] = art.get("n_requirements", 0) > 0
                rec["n_requirements"] = art.get("n_requirements", 0)
                rec["n_documents"] = len(b.documents)
                rec["n_included_claims"] = len(b.included_claim_ids)
                rec["n_independent_sources"] = b.evidence_uncertainty.get("n_independent_sources", 0)
                rec["n_contradictions"] = len(b.contradiction_graph)
                rec["bundle_hash"] = b.bundle_hash()
                rss = [t for t in b.retrieval_traces if t["connector_id"] == "google_news_rss"]
                rec["n_rss_traces"] = len(rss)
                paired = [t for t in rss if paired_dates_ok(t.get("after_date", ""), t.get("before_date", ""))
                          and "after:" in t["logical_query"] and "before:" in t["logical_query"]]
                rec["n_paired_rss"] = len(paired)
                rec["paired_rss_ok"] = (len(paired) == len(rss)) if rss else True
                rec["paired_rss_violation"] = any("before:" in t["logical_query"] and "after:" not in
                                                  t["logical_query"] for t in rss)
                rec["raw_persisted"] = all(t.get("raw_content_hash") for t in rss
                                           if t.get("connector_status") in ("ok", "zero_results"))
                b.persist()
            diff = art.get("plan_diff")
            if diff is not None and hasattr(diff, "n_structural_changes"):
                rec["structural_changes"] = diff.n_structural_changes
                rec["lean_only"] = diff.lean_only
            rec["plan_hash_changed"] = art.get("pre_plan_hash") != art.get("post_plan_hash")
            # a MATERIAL change = the recompile made real structural plan changes grounded in evidence
            # (adding the observation mechanism alone does not count).
            rec["material_change"] = bool(rec["structural_changes"] > 0)
        except Exception as e:  # noqa: BLE001 — record honestly
            rec["simulation_status"] = "execution_failed"
            rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        rec["latency_s"] = round(time.time() - t_q, 2)
        cache_f.write_text(json.dumps(rec, indent=1, default=str))
        rows.append(rec)
        if verbose:
            print(f"  [{i+1}/{len(questions)}] {domain:20s} docs={rec['n_documents']} "
                  f"claims={rec['n_included_claims']} Δstruct={rec['structural_changes']} "
                  f"mat={rec['material_change']} status={rec['simulation_status']}", flush=True)

    # ---- aggregate gates ----
    n = len(rows)
    def rate(pred, over=None):
        over = over if over is not None else rows
        return round(sum(1 for r in over if pred(r)) / max(1, len(over)), 4)

    rss_rows = [r for r in rows if r["n_rss_traces"] > 0]
    public_rows = [r for r in rows if r["domain"] not in PRIVATE_DOMAINS]
    ev_rows = [r for r in rows if r["n_included_claims"] > 0]
    gates = {
        "requirements_produced": {"value": rate(lambda r: r["requirements_produced"]), "th": 1.0, "op": "=="},
        # nonempty bundle measured only where PUBLIC evidence should plausibly exist (private domains excluded)
        "nonempty_bundle_public_rate": {"value": rate(lambda r: r["n_documents"] > 0, public_rows), "th": 0.90, "op": ">="},
        "paired_rss_share": {"value": rate(lambda r: r["paired_rss_ok"], rss_rows), "th": 0.95, "op": ">="},
        "paired_rss_violation_rate": {"value": rate(lambda r: r["paired_rss_violation"]), "th": 0.0, "op": "=="},
        "raw_persisted_rate": {"value": rate(lambda r: r["raw_persisted"], rss_rows), "th": 1.0, "op": "=="},
        # material change measured where evidence was actually admitted (a 0-evidence question can't change)
        "material_change_rate": {"value": rate(lambda r: r["material_change"], ev_rows), "th": 0.50, "op": ">="},
        "forecast_abstention_rate": {"value": rate(lambda r: r["simulation_status"].startswith("completed") and not r["has_forecast"]), "th": 0.0, "op": "=="},
        "execution_failure_rate": {"value": rate(lambda r: r["simulation_status"] == "execution_failed"), "th": 0.10, "op": "<"},
        "lean_only_rate_when_evidence": {"value": rate(lambda r: r["lean_only"] is True, [r for r in rows if r["n_included_claims"] > 0]), "th": 0.5, "op": "<"},
    }
    for g in gates.values():
        v, th, op = g["value"], g["th"], g["op"]
        g["passed"] = (v >= th) if op == ">=" else (v < th) if op == "<" else abs(v - th) < 1e-9
    total_paired = sum(r["n_paired_rss"] for r in rows)
    tax_hist = {}
    for r in rows:
        if r["simulation_status"] == "execution_failed":
            key = f"{r.get('failure_taxonomy', '')}@{r.get('evidence_stage', '')}"
            tax_hist[key] = tax_hist.get(key, 0) + 1
    by_domain = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    out = {"n_questions": n, "n_domains": len(by_domain), "gates": gates,
           "all_gates_passed": all(g["passed"] for g in gates.values()),
           "total_paired_rss_queries": total_paired,
           "failure_taxonomy_histogram": tax_hist,
           "mean_docs": round(sum(r["n_documents"] for r in rows) / max(1, n), 2),
           "mean_included_claims": round(sum(r["n_included_claims"] for r in rows) / max(1, n), 2),
           "mean_structural_changes": round(sum(r["structural_changes"] for r in rows) / max(1, n), 2),
           "per_domain": {d: {"n": len(rs), "mean_docs": round(sum(r["n_documents"] for r in rs) / len(rs), 1),
                              "material_rate": rate(lambda r: r["material_change"], rs)}
                          for d, rs in sorted(by_domain.items())},
           "forensic_examples": [r for r in rows if r["material_change"] and r["n_included_claims"] > 0][:4],
           "_meta": {"llm_calls": meter["calls"], "est_cost_usd": round(meter["tokens"] * (0.27e-6 + 1.10e-6) / 2, 4),
                     "runtime_s": round(time.time() - t0, 1), "lookback_days": lookback_days,
                     "model": "deepseek-chat + live Google News RSS"}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print("\n=== PHASE 2 GATES ===")
    for name, g in gates.items():
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {name:32s} {g['value']} {g['op']} {g['th']}")
    print(f"\ntotal paired RSS queries: {total_paired} | all gates: {out['all_gates_passed']}")
    print(f"wrote {RESULT} (calls={meter['calls']}, ~${out['_meta']['est_cost_usd']}, {out['_meta']['runtime_s']}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lookback-days", type=int, default=150)
    a = ap.parse_args()
    run(a.limit, a.lookback_days)
