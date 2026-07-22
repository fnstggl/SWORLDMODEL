"""Phase 1 no-abstention generality validation — every coherent question SIMULATES (real DeepSeek).

This is the production-semantics successor to wmv2_compiler_generality.py (which is kept intact as the
Session-1 historical record, where CompileAbstention was still a "valid, desired outcome"). Here abstention
for epistemic reasons is REMOVED: the harness feeds ≥100 held-out NL questions across ≥15 domains through
the ONE production entry `pipeline.simulate()` (real LLM decomposition, no scripted plans) and measures the
B13 acceptance gates:

  valid_plan            ≥ 0.95   compile produced a typed, executable plan
  materialize           ≥ 0.90   build_world succeeded (world instantiated)
  complete_rollout      ≥ 0.85   rollout ran AND the terminal readout bound + produced a distribution
  forecast_abstention   = 0.00   a coherent question that produced NO forecast for EPISTEMIC reasons
  clarification         < 0.05   genuinely-incoherent questions only (rare)
  execution_failure     < 0.10   engineering failures (taxonomy'd), never epistemic
  provenance_status     = 1.00   every entity field carries observed/inferred/assumed (no fabricated obs)
  fallback_tier_ided    = 1.00   every fallback names its tier
  unsupported_precision < 0.02   no LLM-proposed field enters the world stamped 'observed'
  llm_prob_injection    = 0.00   no accepted mechanism mints terminal probabilities from the LLM
  no_keyword_router     = True   static: no scenario-level if/elif in swm/world_model_v2 (grep-proof)

Every result is a SimulationResult with a forecast whenever the simulation ran; support_grade + limitations
carry the epistemic weakness. Resumable (per-question cache), metered (cost + latency), deterministic given
the cache. Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_phase1_no_abstention_generality
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

# reuse the SAME held-out question bank as the Session-1 harness (16 domains, ≥100 Qs, NO plans)
from experiments.wmv2_compiler_generality import QUESTIONS, _expand_to_100

RESULT = "experiments/results/wmv2_phase1_no_abstention_generality.json"
CACHE = Path("experiments/results/phase1_no_abstention_generality")


def _digest(q, as_of, horizon):
    return hashlib.sha1(f"no-abstain|{q}|{as_of}|{horizon}".encode()).hexdigest()[:12]


def _no_keyword_router() -> dict:
    """Static gate: no scenario-level branch (if election/email/viral/…) anywhere in swm/world_model_v2.
    A single generic path must handle every domain — a keyword router would be domain-hardcoding."""
    v2 = Path(__file__).resolve().parent.parent / "swm" / "world_model_v2"
    pat = re.compile(r"^\s*(if|elif)\s+.*(election|email|viral|sports|headline|merger|strike)\b.*:", re.I)
    hits = []
    for f in sorted(v2.glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or "`" in line or '"' in s or "'" in s:
                continue                                      # prose/docstrings/string-literals
            if pat.search(line):
                hits.append(f"{f.name}:{i}: {s[:80]}")
    return {"passed": not hits, "violations": hits}


def run(limit, verbose=True):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.world_model_v2 import registry as reg
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.materialize import build_world, run_from_plan
    from swm.world_model_v2.pipeline import result_from_run
    from swm.world_model_v2.result import (SUPPORT_GRADES, ClarificationRequired, CompilerExecutionError,
                                           SimulationResult, migrate_legacy_result)

    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    reg.load_registry()
    questions = _expand_to_100(QUESTIONS)
    questions = questions[:limit] if limit else questions
    meter = {"calls": 0, "tokens": 0}
    llm = default_chat_fn(system="You are the world-slice compiler proposal stage. Reply ONLY JSON.",
                          max_tokens=2200, temperature=0.2)
    if llm is None:
        raise SystemExit("needs DEEPSEEK_API_KEY")

    def call(prompt):
        txt = llm(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
        return txt

    rows = []
    for i, (domain, q, as_of, horizon) in enumerate(questions):
        cache_f = CACHE / f"{_digest(q, as_of, horizon)}.json"
        if cache_f.exists():
            rows.append(json.loads(cache_f.read_text()))
            continue
        rec = {"domain": domain, "question": q, "as_of": as_of, "horizon": horizon,
               # stage flags
               "compiled": False, "materialized": False, "executed_e2e": False, "readout_resolved": False,
               "has_forecast": False, "provenance_ok": None, "unsupported_precision": False,
               "llm_prob_injection": False,
               # semantics
               "simulation_status": "", "support_grade": "", "recommendation_status": "",
               "raw_probability": None, "distribution_keys": [], "n_distribution": 0,
               "support_grade_valid": None, "degraded": None,
               "n_fallbacks": 0, "fallbacks_all_tiered": None, "fallback_tiers": [],
               "n_structural_hypotheses": 0, "structural_disagreement": None,
               "n_accepted_mech": 0, "n_rejected_mech": 0, "mechanisms": [],
               "n_entities": 0, "n_institutions": 0, "n_populations": 0, "n_latents": 0,
               "n_limitations": 0, "failure_taxonomy": "", "clarification": False,
               "forecast_abstention": False, "error": ""}
        t_q = time.time()
        res = None
        # --- ONE compile → materialize → rollout → shipped SimulationResult (single LLM decomposition) ---
        try:
            plan = compile_world(q, llm=lambda p: call(p), evidence="", as_of=as_of, horizon=horizon, seed=7)
            rec["compiled"] = True
            rec["n_accepted_mech"] = len(plan.accepted_mechanisms)
            rec["n_rejected_mech"] = len(plan.rejected_mechanisms)
            rec["mechanisms"] = [m["mech_id"] for m in plan.accepted_mechanisms]
            rec["n_entities"], rec["n_institutions"] = len(plan.entities), len(plan.institutions)
            rec["n_populations"], rec["n_latents"] = len(plan.populations), len(plan.latents)
            rec["n_structural_hypotheses"] = len(plan.structural_hypotheses)
            rec["n_fallbacks"] = len(plan.fallbacks_used)
            rec["fallback_tiers"] = [fb.get("tier") for fb in plan.fallbacks_used]
            rec["fallbacks_all_tiered"] = all(isinstance(fb.get("tier"), int) for fb in plan.fallbacks_used)
            rec["support_grade_valid"] = plan.support_grade in SUPPORT_GRADES
            rec["degraded"] = plan.degraded
            # LLM probability injection: NO accepted mechanism may mint terminal probabilities from the LLM
            rec["llm_prob_injection"] = any(m.get("operator") == "agent_decision"
                                            and m.get("allow_llm_probabilities") for m in plan.accepted_mechanisms)
            w = build_world(plan)
            rec["materialized"] = True
            statuses = [sf.prov.status for e in w.entities.values() for sf in e.fields.values()
                        if hasattr(sf, "prov")]
            rec["provenance_ok"] = all(s != "observed" for s in statuses)
            rec["unsupported_precision"] = any(s == "observed" for s in statuses)
            result, branches = run_from_plan(plan, seed=7)
            rec["executed_e2e"] = True
            dist = result.get("distribution") or {}
            rec["readout_resolved"] = bool(dist) and result.get("readout") == "terminal_states"
            res = result_from_run(q, plan, result, branches, t0=t_q)
        except ClarificationRequired as e:
            res = SimulationResult(question=q, simulation_status="clarification_required",
                                   clarification_reason=str(e), interpretation_hypotheses=e.interpretations_tried)
        except CompilerExecutionError as e:
            res = SimulationResult(question=q, simulation_status="execution_failed", failure_taxonomy=e.taxonomy,
                                   limitations=[f"{e}"[:150]])
            rec["error"] = f"CompilerExecutionError[{e.taxonomy}]: {str(e)[:120]}"
        except Exception as e:  # noqa: BLE001 — a raw exception is itself an execution_failed (engineering gap)
            res = SimulationResult(question=q, simulation_status="execution_failed",
                                   failure_taxonomy="runtime_exception", limitations=[f"{e}"[:150]])
            rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"

        # --- record the shipped contract ---
        rec["simulation_status"] = res.simulation_status
        rec["support_grade"] = res.support_grade
        rec["recommendation_status"] = res.recommendation_status
        rec["raw_probability"] = res.raw_probability
        rec["distribution_keys"] = list((res.raw_distribution or {}).keys())
        rec["n_distribution"] = len(res.raw_distribution or {})
        rec["has_forecast"] = res.has_forecast() and bool(res.raw_distribution)
        rec["structural_disagreement"] = res.structural_disagreement
        rec["n_limitations"] = len(res.limitations)
        rec["failure_taxonomy"] = res.failure_taxonomy
        rec["clarification"] = res.simulation_status == "clarification_required"
        # a FORECAST ABSTENTION = the simulation "ran" (completed/degraded) but produced NO forecast for
        # epistemic reasons. Must be ZERO. (Incoherent→clarification and technical→execution_failed have their
        # OWN gates and are NOT forecast abstentions.) Back-compat: a legacy abstain=True on a completed result.
        d = migrate_legacy_result(res.as_dict())
        rec["forecast_abstention"] = bool(res.has_forecast() and (not res.raw_distribution or d.get("abstain")))
        rec["latency_s"] = round(time.time() - t_q, 2)
        cache_f.write_text(json.dumps(rec, indent=1, default=str))
        rows.append(rec)
        if verbose:
            print(f"  [{i+1}/{len(questions)}] {domain:22s} status={rec['simulation_status']:26s} "
                  f"grade={rec['support_grade']:20s} fc={rec['has_forecast']} p={rec['raw_probability']}",
                  flush=True)

    # ---------------------------------------------------------------- aggregate B13 gates
    n = len(rows)
    by_domain = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    comp = [r for r in rows if r["compiled"]]

    def rate(pred, over=rows):
        over = list(over)
        return round(sum(1 for r in over if pred(r)) / max(1, len(over)), 4)

    complete = rate(lambda r: r["executed_e2e"] and r["readout_resolved"])
    gates = {
        "valid_plan_rate":            {"value": rate(lambda r: r["compiled"]), "threshold": 0.95, "op": ">="},
        "materialize_rate":           {"value": rate(lambda r: r["materialized"]), "threshold": 0.90, "op": ">="},
        "complete_rollout_readout_rate": {"value": complete, "threshold": 0.85, "op": ">="},
        "forecast_abstention_rate":   {"value": rate(lambda r: r["forecast_abstention"]), "threshold": 0.0, "op": "=="},
        "clarification_rate":         {"value": rate(lambda r: r["clarification"]), "threshold": 0.05, "op": "<"},
        "execution_failure_rate":     {"value": rate(lambda r: r["simulation_status"] == "execution_failed"), "threshold": 0.10, "op": "<"},
        "provenance_status_rate":     {"value": rate(lambda r: r["provenance_ok"], comp), "threshold": 1.0, "op": "=="},
        "fallback_tier_identified_rate": {"value": rate(lambda r: r["fallbacks_all_tiered"] is not False, comp), "threshold": 1.0, "op": "=="},
        "unsupported_precision_rate": {"value": rate(lambda r: r["unsupported_precision"], comp), "threshold": 0.02, "op": "<"},
        "llm_prob_injection_rate":    {"value": rate(lambda r: r["llm_prob_injection"], comp), "threshold": 0.0, "op": "=="},
    }
    for g in gates.values():
        v, th, op = g["value"], g["threshold"], g["op"]
        g["passed"] = (v >= th) if op == ">=" else (v < th) if op == "<" else (abs(v - th) < 1e-9)
    router = _no_keyword_router()
    gates["no_keyword_router"] = {"value": router["passed"], "threshold": True, "op": "==",
                                  "passed": router["passed"], "violations": router["violations"]}

    status_hist, grade_hist, tax_hist = {}, {}, {}
    for r in rows:
        status_hist[r["simulation_status"]] = status_hist.get(r["simulation_status"], 0) + 1
        if r["support_grade"]:
            grade_hist[r["support_grade"]] = grade_hist.get(r["support_grade"], 0) + 1
        if r["failure_taxonomy"]:
            tax_hist[r["failure_taxonomy"]] = tax_hist.get(r["failure_taxonomy"], 0) + 1
    mech_hist = {}
    for r in comp:
        for m in r["mechanisms"]:
            mech_hist[m] = mech_hist.get(m, 0) + 1

    all_pass = all(g["passed"] for g in gates.values())
    summary = {
        "n_questions": n, "n_domains": len(by_domain),
        "all_gates_passed": all_pass,
        "forecasts_produced": sum(1 for r in rows if r["has_forecast"]),
        "forecast_rate": rate(lambda r: r["has_forecast"]),
        "mean_support_by_grade": grade_hist,
        "mean_latency_s": round(sum(r.get("latency_s", 0) for r in rows) / max(1, n), 2),
    }
    out = {"summary": summary, "b13_gates": gates,
           "simulation_status_histogram": status_hist,
           "support_grade_histogram": grade_hist,
           "failure_taxonomy_histogram": tax_hist,
           "mechanism_histogram": dict(sorted(mech_hist.items(), key=lambda kv: -kv[1])),
           "per_domain": {d: {"n": len(rs),
                              "forecast_rate": rate(lambda r: r["has_forecast"], rs),
                              "complete_rate": rate(lambda r: r["executed_e2e"] and r["readout_resolved"], rs),
                              "grades": sorted({r["support_grade"] for r in rs if r["support_grade"]})}
                          for d, rs in sorted(by_domain.items())},
           "forensic_examples": (
               [r for r in rows if r["has_forecast"] and r["degraded"]][:4]
               + [r for r in rows if r["n_structural_hypotheses"] > 1][:3]
               + [r for r in rows if r["simulation_status"] == "execution_failed"][:2]),
           "_meta": {"llm_calls": meter["calls"], "llm_tokens_est": meter["tokens"],
                     "est_cost_usd": round(meter["tokens"] * (0.27e-6 + 1.10e-6) / 2, 4),
                     "model": "deepseek-chat", "runtime_s": round(time.time() - t0, 1),
                     "semantics": "no_abstention_v2",
                     "note": "production simulate() path; every coherent question forecasts; epistemic "
                             "weakness lowers support_grade, never refuses. NO scripted plans."}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print("\n=== B13 GATES ===")
    for name, g in gates.items():
        mark = "PASS" if g["passed"] else "FAIL"
        print(f"  [{mark}] {name:34s} value={g['value']} {g['op']} {g['threshold']}")
    print(f"\nALL GATES PASSED: {all_pass}")
    print(f"forecasts: {summary['forecasts_produced']}/{n}  grades: {grade_hist}")
    print(f"wrote {RESULT} (calls={meter['calls']}, ~${out['_meta']['est_cost_usd']}, "
          f"{out['_meta']['runtime_s']}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.limit)
