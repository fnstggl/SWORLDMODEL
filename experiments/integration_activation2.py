"""Integration activation measurement v2 — EXECUTION level (gates 13-19 rescoring).

v1 (integration_activation.py) measured compiler EMISSION only. This measures what the mandate's standard
actually demands, on the same independently-labeled corpus:

  * EXECUTION recall     — did the phase's operator produce >=1 StateDelta when the phase is REQUIRED?
  * false EXECUTION      — did it produce StateDeltas when NOT required (after the relevance gate)?
  * matched causal ablation — for each required phase on each question, rerun the SAME plan/seed with only
    that phase's requirement forced off (its synthesis removed); does the terminal P(affirmative) change?

Pipeline per question: real compiler (no evidence — the activation chain under test is compiler→execution)
→ rule normalization → relevance gate + activation synthesis → run_from_plan → per-operator StateDelta
census → per-phase ablation reruns. run_from_plan uses no LLM (fitted/uniform policies), so ablations are
exactly matched (same seed, same base plan JSON).
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path

from experiments.integration_corpus import QUESTIONS, PHASE_FLAGS

OUT = Path("experiments/results/integration")
ART = OUT / "activation2.json"

#: operator → phase flag (execution census)
_OP_PHASE = {
    "production_actor_policy": "p4", "agent_decision": "p4", "fitted_decision": "p4",
    "actor_action_aggregation": "p4",
    "behavioral_mechanism": "p6", "feature_hazard": "p6",
    "nonlinear_state_step": "p7", "nonlinear_mechanism": "p7", "nonlinear_contagion": "p7",
    "population_aggregation": "p9pop", "network_diffusion": "p9net",
    "institutional_decision": "p10", "institutional_vote": "p10", "institution_action": "p10",
}
_PHASE_KEY = {"p4": "phase4_actor_policy", "p6": "phase6_registry", "p7": "phase7_nonlinear",
              "p9pop": "phase9_populations", "p9net": "phase9_networks", "p10": "phase10_institutions"}
ABL_EPS = 0.02          # a terminal shift below this is treated as no causal effect


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)


def _p_affirmative(res, plan):
    dist = res.get("distribution") or {}
    opts = list(plan.outcome_contract.options)
    aff = str(opts[0]) if opts else "True"
    p = dist.get(aff)
    if p is None:                                            # truthy canonicalization fallback
        p = sum(v for k, v in dist.items() if str(k).lower() in (aff.lower(), "true", "yes"))
    return float(p or 0.0)


def _execute(plan, seed=3):
    """Normalize → gate → synthesize → run. Returns (per-phase delta counts, p_affirmative, requirements)."""
    from swm.world_model_v2.integration_completion import normalize_institution_rules
    from swm.world_model_v2.activation_synthesis import phase_requirements, synthesize_activation
    from swm.world_model_v2.materialize import run_from_plan
    normalize_institution_rules(plan)
    req = phase_requirements(plan)
    synthesize_activation(plan, req)
    res, branches = run_from_plan(plan, llm=None, seed=seed)
    counts = {f: 0 for f in _PHASE_KEY}
    for b in branches:
        for d in b.log:
            f = _OP_PHASE.get(d.operator)
            if f:
                counts[f] += 1
    return counts, _p_affirmative(res, plan), {k: v["required"] for k, v in req.items()}


def _ablate(plan_factory, flag, seed=3):
    """Rerun with ONLY this phase's requirement forced off — the matched ablation arm."""
    from swm.world_model_v2.integration_completion import normalize_institution_rules
    from swm.world_model_v2.activation_synthesis import phase_requirements, synthesize_activation
    from swm.world_model_v2.materialize import run_from_plan
    plan = plan_factory()
    normalize_institution_rules(plan)
    req = phase_requirements(plan)
    req[_PHASE_KEY[flag]] = {"required": False, "why": "matched ablation arm"}
    synthesize_activation(plan, req)
    res, _ = run_from_plan(plan, llm=None, seed=seed)
    return _p_affirmative(res, plan)


def run(limit=None):
    from swm.world_model_v2.compiler import compile_world
    OUT.mkdir(parents=True, exist_ok=True)
    llm = _make_llm()
    if llm is None:
        print("no llm"); return
    existing = {}
    if ART.exists():
        for r in json.loads(ART.read_text()).get("rows", []):
            if not r.get("error"):
                existing[r["qid"]] = r
    rows = []
    qs = QUESTIONS[:limit] if limit else QUESTIONS
    for qid, q, as_of, horizon, flags in qs:
        if qid in existing:
            rows.append(existing[qid]); continue
        rec = {"qid": qid, "required_labels": sorted(flags)}
        try:
            base_plan = compile_world(q, llm=llm, evidence="", as_of=as_of, horizon=horizon, seed=0)
            factory = lambda: copy.deepcopy(base_plan)                       # noqa: E731 — matched arms
            counts, p_full, gate = _execute(factory())
            rec.update({"executed": {f: c for f, c in counts.items() if c > 0},
                        "gate_required": {f: gate[_PHASE_KEY[f]] for f in _PHASE_KEY},
                        "p_full": round(p_full, 4)})
            abls = {}
            for f in _PHASE_KEY:
                if counts.get(f, 0) > 0:                     # ablate only phases that actually executed
                    p_abl = _ablate(factory, f)
                    abls[f] = {"p_ablated": round(p_abl, 4),
                               "delta": round(abs(p_full - p_abl), 4),
                               "causal_effect": abs(p_full - p_abl) >= ABL_EPS}
            rec["ablations"] = abls
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:160]
        rows.append(rec)
        ART.write_text(json.dumps({"rows": rows}, indent=2))
        ex = rec.get("executed", {})
        abl_s = ",".join(f"{k}:{v['delta']}" for k, v in rec.get("ablations", {}).items())
        print(f"{qid:16s} req={','.join(rec['required_labels']) or '-':18s} "
              f"exec={','.join(sorted(ex)) or '-':22s} abl={{{abl_s}}}")
    _aggregate(rows)


def _aggregate(rows):
    ok = [r for r in rows if not r.get("error")]
    per_phase = {}
    for f in PHASE_FLAGS:
        if f == "p11":
            continue
        req = [r for r in ok if f in r["required_labels"]]
        notreq = [r for r in ok if f not in r["required_labels"]]
        tp = sum(1 for r in req if r.get("executed", {}).get(f))
        fp = sum(1 for r in notreq if r.get("executed", {}).get(f))
        ce_rows = [r for r in req if r.get("ablations", {}).get(f)]
        ce = sum(1 for r in ce_rows if r["ablations"][f]["causal_effect"])
        per_phase[f] = {
            "n_required": len(req), "executed_when_required": tp,
            "execution_recall": round(tp / len(req), 3) if req else None,
            "n_not_required": len(notreq), "false_execution": fp,
            "false_execution_rate": round(fp / len(notreq), 3) if notreq else None,
            "n_ablated": len(ce_rows), "causal_effect_count": ce,
            "causal_effect_rate": round(ce / len(ce_rows), 3) if ce_rows else None}
    agg = {"n_scored": len(ok), "n_errors": len(rows) - len(ok),
           "per_phase_execution": per_phase,
           "note": "EXECUTION-level: recall = >=1 StateDelta when required; false = StateDeltas when not "
                   "required (post-gate); causal effect = matched ablation shifts terminal by >= "
                   f"{ABL_EPS}. Deterministic rollout (no LLM), matched seeds."}
    payload = json.loads(ART.read_text()); payload["aggregate"] = agg
    ART.write_text(json.dumps(payload, indent=2))
    print("\nAGGREGATE:", json.dumps(agg, indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
