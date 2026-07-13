"""Phase 1 ablations (B12) — what each component contributes to the no-abstention guarantee.

Phase 1's claim is GENERALITY, not accuracy against ground truth (these held-out questions have no resolved
outcome), so the ablations measure how the B13 gates and the terminal DISPERSION degrade when a load-bearing
component is removed. To keep cost down, each question is compiled ONCE (one LLM decomposition) and the
compiled plan is transformed per ablation (deep-copied, no extra LLM calls) before rollout.

Ablations (each vs. the full compiler):
  full_compiler          the production pipeline (baseline)
  no_fallback_hierarchy  drop the generic_outcome_prior resolver + resolve_outcome event → the readout is
                         no longer guaranteed to be written. Isolates the mechanism that GUARANTEES every
                         coherent question forecasts (the core no-abstention lever).
  no_readout_repair      re-point a repaired readout back at its original unbindable target → isolates the
                         readout-repair that guarantees the terminal reads out.
  no_structural_hyps     drop competing structural hypotheses → single lean path. Isolates the component that
                         produces structural disagreement + wider posterior dispersion under uncertainty.
  no_sensitivity_margin  give every component explicit fidelity (no marginalization) → isolates the fidelity
                         planner (expected: same forecast, different compute).

Metrics per ablation: forecast_rate (produced a probability), complete_rate (rollout + bound readout),
execution_failure_rate, mean terminal dispersion (entropy of the binary distribution), mean structural
disagreement entropy. Resumable via the shared compile cache. Metered.
Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_phase1_ablations --k 24
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

from experiments.wmv2_compiler_generality import QUESTIONS, _expand_to_100

RESULT = "experiments/results/wmv2_phase1_ablations.json"
CACHE = Path("experiments/results/phase1_ablations")
ABLATIONS = ("full_compiler", "no_fallback_hierarchy", "no_readout_repair",
             "no_structural_hyps", "no_sensitivity_margin")


def _digest(q, as_of, horizon):
    return hashlib.sha1(f"ablate|{q}|{as_of}|{horizon}".encode()).hexdigest()[:12]


def _entropy(dist):
    ps = [p for p in (dist or {}).values() if p > 0]
    return round(-sum(p * math.log(p, 2) for p in ps), 4) if ps else 0.0


def _ablate(plan, mode):
    """Return a deep-copied plan with one component removed. No LLM calls."""
    p = copy.deepcopy(plan)
    if mode == "no_fallback_hierarchy":
        p.accepted_mechanisms = [m for m in p.accepted_mechanisms if m["mech_id"] != "generic_outcome_prior"]
        p.scheduled_events = [e for e in p.scheduled_events if e.get("etype") != "resolve_outcome"]
        p.fallbacks_used = []
    elif mode == "no_readout_repair":
        if (p.provenance or {}).get("readout_repaired"):
            # re-point the contract readout at an entity.field nothing writes (the pre-repair failure mode)
            from swm.world_model_v2.compiler import _make_readout
            p.outcome_contract.readout_var = "ghost_unbindable.field"
            p.outcome_contract.readout = _make_readout("ghost_unbindable.field")
    elif mode == "no_structural_hyps":
        p.structural_hypotheses = []
    elif mode == "no_sensitivity_margin":
        fp = dict(p.fidelity_plan or {})
        fp["explicit"] = list(fp.get("explicit", [])) + list(fp.get("marginalized_with_uncertainty", []))
        fp["marginalized_with_uncertainty"] = []
        p.fidelity_plan = fp
    return p


def run(k, verbose=True):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.world_model_v2 import registry as reg
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.materialize import run_from_plan
    from swm.world_model_v2.pipeline import result_from_run
    from swm.world_model_v2.result import CompilerExecutionError

    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    reg.load_registry()
    questions = _expand_to_100(QUESTIONS)[:k]
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

    per_ablation = {a: [] for a in ABLATIONS}
    for i, (domain, q, as_of, horizon) in enumerate(questions):
        cache_f = CACHE / f"{_digest(q, as_of, horizon)}.json"
        if cache_f.exists():
            rec = json.loads(cache_f.read_text())
        else:
            rec = {"domain": domain, "question": q, "ablations": {}}
            try:
                plan = compile_world(q, llm=lambda p: call(p), evidence="", as_of=as_of, horizon=horizon, seed=7)
            except Exception as e:  # noqa: BLE001
                rec["compile_error"] = f"{type(e).__name__}: {str(e)[:120]}"
                plan = None
            if plan is not None:
                for mode in ABLATIONS:
                    ap = _ablate(plan, mode)
                    r = {"forecast": False, "complete": False, "execution_failed": False,
                         "dispersion": 0.0, "struct_entropy": 0.0, "status": ""}
                    try:
                        result, branches = run_from_plan(ap, seed=7)
                        res = result_from_run(q, ap, result, branches, t0=time.time())
                        dist = res.raw_distribution or {}
                        r["status"] = res.simulation_status
                        r["forecast"] = res.has_forecast() and bool(dist) and res.raw_probability is not None
                        r["complete"] = result.get("readout") == "terminal_states" and bool(result.get("distribution"))
                        r["dispersion"] = _entropy(dist)
                        r["struct_entropy"] = _entropy(res.structural_disagreement or {})
                    except CompilerExecutionError as e:
                        r["execution_failed"] = True
                        r["status"] = f"execution_failed:{e.taxonomy}"
                    except Exception as e:  # noqa: BLE001
                        r["execution_failed"] = True
                        r["status"] = f"error:{type(e).__name__}"
                    rec["ablations"][mode] = r
            cache_f.write_text(json.dumps(rec, indent=1, default=str))
        for mode in ABLATIONS:
            if mode in rec.get("ablations", {}):
                per_ablation[mode].append(rec["ablations"][mode])
        if verbose:
            fullr = rec.get("ablations", {}).get("full_compiler", {})
            print(f"  [{i+1}/{len(questions)}] {domain:20s} full: fc={fullr.get('forecast')} "
                  f"disp={fullr.get('dispersion')}", flush=True)

    def agg(rows):
        n = max(1, len(rows))
        return {"n": len(rows),
                "forecast_rate": round(sum(r["forecast"] for r in rows) / n, 3),
                "complete_rate": round(sum(r["complete"] for r in rows) / n, 3),
                "execution_failure_rate": round(sum(r["execution_failed"] for r in rows) / n, 3),
                "mean_dispersion": round(sum(r["dispersion"] for r in rows) / n, 3),
                "mean_struct_entropy": round(sum(r["struct_entropy"] for r in rows) / n, 3)}

    summary = {a: agg(per_ablation[a]) for a in ABLATIONS}
    # contribution = full minus ablation on the key gates
    full = summary["full_compiler"]
    contributions = {}
    for a in ABLATIONS:
        if a == "full_compiler":
            continue
        s = summary[a]
        contributions[a] = {
            "forecast_rate_drop": round(full["forecast_rate"] - s["forecast_rate"], 3),
            "complete_rate_drop": round(full["complete_rate"] - s["complete_rate"], 3),
            "execution_failure_increase": round(s["execution_failure_rate"] - full["execution_failure_rate"], 3),
            "dispersion_change": round(s["mean_dispersion"] - full["mean_dispersion"], 3),
            "struct_entropy_change": round(s["mean_struct_entropy"] - full["mean_struct_entropy"], 3)}

    out = {"summary": summary, "contributions_vs_full": contributions,
           "interpretation": {
               "no_fallback_hierarchy": "expected: forecast/complete rate COLLAPSES — the fallback resolver "
                                        "is what guarantees every coherent question forecasts (core no-abstention lever).",
               "no_readout_repair": "expected: execution_failure INCREASES on questions whose LLM readout "
                                    "was unbindable — repair is what guarantees the terminal reads out.",
               "no_structural_hyps": "expected: structural entropy → 0 and dispersion narrows — the component "
                                     "that represents structural uncertainty as competing particles.",
               "no_sensitivity_margin": "expected: negligible forecast change — fidelity planning affects "
                                       "compute allocation, not whether a forecast is produced."},
           "_meta": {"k_questions": k, "llm_calls": meter["calls"],
                     "est_cost_usd": round(meter["tokens"] * (0.27e-6 + 1.10e-6) / 2, 4),
                     "model": "deepseek-chat", "runtime_s": round(time.time() - t0, 1),
                     "note": "each question compiled ONCE; ablations are plan transforms (no extra LLM calls)"}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print("\n=== ABLATION SUMMARY (vs full_compiler) ===")
    for a in ABLATIONS:
        s = summary[a]
        print(f"  {a:24s} forecast={s['forecast_rate']} complete={s['complete_rate']} "
              f"exec_fail={s['execution_failure_rate']} disp={s['mean_dispersion']} "
              f"struct_H={s['mean_struct_entropy']}")
    print(f"\nwrote {RESULT} (calls={meter['calls']}, ~${out['_meta']['est_cost_usd']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=24)
    a = ap.parse_args()
    run(a.k)
