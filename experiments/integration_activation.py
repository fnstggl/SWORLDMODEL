"""Integration activation measurement (Parts L/M).

Compiles each relevance-labeled corpus question through the REAL compiler and measures, per phase, whether the
compiler EMITS the typed spec the runtime needs — scored against the INDEPENDENT relevance labels (recall =
emitted│required, false-activation = emitted│not-required). Also measures the Phase-10 rule-kind normalization
effect (executable institution rules before/after) — the one verifiable execution-level fix this run lands.

This is honest about the pipeline STAGE it measures: EMISSION + Phase-10 executability. Full execute+StateDelta+
matched-ablation activation for P6/P7/P9/P10 to the >=95% gates is NOT achieved here and is reported as a
failed gate with a continuation path (see the validation doc).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

from experiments.integration_corpus import QUESTIONS, PHASE_FLAGS

OUT = Path("experiments/results/integration")
ART = OUT / "activation.json"


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)


def _emission(plan):
    """What the compiler emitted, mapped to the phase flags."""
    ops = " ".join(str(m.get("operator", "")) for m in (getattr(plan, "accepted_mechanisms", []) or [])
                   if isinstance(m, dict))
    return {
        "p4": bool(getattr(plan, "actor_decisions", [])) or any(k in ops for k in ("decision", "policy", "actor")),
        "p6": any(k in ops for k in ("mechanism", "contagion", "diffusion", "belief_update", "resource_update")),
        "p7": "nonlinear" in ops,
        "p9pop": bool(getattr(plan, "populations", [])),
        "p9net": bool(getattr(plan, "relations", [])),
        "p10": bool(getattr(plan, "institutions", [])),
        "p11": True,  # controller runs on every execution; trigger firing is measured separately (not here)
    }


def run(limit=None):
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.integration_completion import executable_rule_count, normalize_institution_rules
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
        rec = {"qid": qid, "required": sorted(flags)}
        try:
            plan = compile_world(q, llm=llm, evidence="", as_of=as_of, horizon=horizon, seed=0)
            em = _emission(plan)
            exec_before = executable_rule_count(plan)
            norm = normalize_institution_rules(plan)
            exec_after = executable_rule_count(plan)
            rec.update({"emitted": em, "n_institutions": len(getattr(plan, "institutions", []) or []),
                        "n_populations": len(getattr(plan, "populations", []) or []),
                        "n_relations": len(getattr(plan, "relations", []) or []),
                        "exec_rules_before": exec_before, "exec_rules_after": exec_after,
                        "rule_normalization": norm})
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:140]
        rows.append(rec)
        ART.write_text(json.dumps({"rows": rows}, indent=2))
        em = rec.get("emitted", {})
        print(f"{qid:16s} req={','.join(rec['required']) or '-':20s} "
              f"emit={','.join(k for k in PHASE_FLAGS if em.get(k)) or '-':22s} "
              f"instRules {rec.get('exec_rules_before')}->{rec.get('exec_rules_after')}")
    _aggregate(rows)


def _aggregate(rows):
    ok = [r for r in rows if r.get("emitted")]
    per_phase = {}
    for ph in PHASE_FLAGS:
        if ph == "p11":
            continue
        req = [r for r in ok if ph in r["required"]]
        notreq = [r for r in ok if ph not in r["required"]]
        tp = sum(1 for r in req if r["emitted"].get(ph))
        fp = sum(1 for r in notreq if r["emitted"].get(ph))
        per_phase[ph] = {
            "n_required": len(req), "emitted_when_required": tp,
            "recall": round(tp / len(req), 3) if req else None,
            "n_not_required": len(notreq), "false_emit": fp,
            "false_activation_rate": round(fp / len(notreq), 3) if notreq else None}
    # phase-10 executability fix effect
    inst_rows = [r for r in ok if r.get("n_institutions", 0) > 0]
    exec_fix = {
        "n_institution_rows": len(inst_rows),
        "rows_executable_before": sum(1 for r in inst_rows if r.get("exec_rules_before", 0) > 0),
        "rows_executable_after": sum(1 for r in inst_rows if r.get("exec_rules_after", 0) > 0),
        "total_exec_rules_before": sum(r.get("exec_rules_before", 0) for r in inst_rows),
        "total_exec_rules_after": sum(r.get("exec_rules_after", 0) for r in inst_rows)}
    agg = {"n_scored": len(ok), "per_phase_emission": per_phase, "phase10_executability_fix": exec_fix,
           "note": "Measures compiler EMISSION recall/precision + the Phase-10 rule-executability fix. Full "
                   "execute+StateDelta+ablation activation to the >=95% gates is NOT claimed here."}
    payload = json.loads(ART.read_text()); payload["aggregate"] = agg
    ART.write_text(json.dumps(payload, indent=2))
    print("\nAGGREGATE:", json.dumps(agg, indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
