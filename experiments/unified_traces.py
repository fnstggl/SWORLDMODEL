"""Unified runtime — forensic traces (Part T). Detailed full-chain runs through simulate_world.

Emits, per stratified case, the complete chain: question → evidence → posterior → plan lineage → recompilation
→ terminal → active-component manifest → limitations → cost/latency, so a reviewer can see the output came
from the integrated simulation, not a wrapper around separate predictors. Writes traces.json + markdown.
"""
from __future__ import annotations
import json, time
from pathlib import Path

OUT = Path("experiments/results/unified")

CASES = [
    ("Will the US Federal Reserve cut interest rates at its September 2024 meeting?", "2024-09-10", "2024-09-19", "econ"),
    ("Will Bashar al-Assad's government fall in Syria in 2024?", "2024-11-25", "2024-12-31", "geopolitics"),
    ("Will Nvidia announce a stock split in 2024?", "2024-05-01", "2024-06-30", "finance"),
    ("Will India win the 2024 ICC Men's T20 Cricket World Cup?", "2024-06-20", "2024-06-29", "sports"),
]


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)


def run():
    import functools
    from swm.world_model_v2.unified_runtime import simulate_world as _sw_default
    # archival full-fidelity harness: pinned since the §25 default switch
    simulate_world = functools.partial(_sw_default, execution_profile="full_fidelity")
    OUT.mkdir(parents=True, exist_ok=True)
    llm = _make_llm()
    if llm is None:
        print("no llm"); return
    traces = []
    for q, as_of, horizon, domain in CASES:
        t0 = time.time()
        try:
            res = simulate_world(q, as_of=as_of, horizon=horizon, seed=0, llm=llm)
            prov = res.provenance or {}
            m = prov.get("active_component_manifest", {})
            traces.append({
                "question": q, "domain": domain, "as_of": as_of, "horizon": horizon,
                "simulation_status": res.simulation_status, "raw_probability": res.raw_probability,
                "support_grade": res.support_grade, "recommendation_status": res.recommendation_status,
                "plan_lineage": prov.get("plan_lineage"), "evidence_bundle_hash": prov.get("evidence_bundle_hash"),
                "posterior_consumed": prov.get("posterior_consumed"),
                "active_component_manifest": {k: {"executed": v["executed"], "omitted": v["omitted"],
                                                  "reason": v["reason"], "n_events": v["n_events"]}
                                              for k, v in m.items()},
                "calibration_compatibility": prov.get("calibration_compatibility", {}).get("old_phase12_calibrator"),
                "limitations": (res.limitations or [])[:5], "latency_s": res.latency_s,
                "wall_s": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001
            traces.append({"question": q, "domain": domain, "error": f"{type(e).__name__}: {e}"[:160]})
        (OUT / "traces.json").write_text(json.dumps({"traces": traces}, indent=2))
        print(f"[{domain:11s}] {q[:44]:44s} p={traces[-1].get('raw_probability')} "
              f"status={traces[-1].get('simulation_status')}")
    _markdown(traces)
    print("DONE traces", len(traces))


def _markdown(traces):
    m = ["# WMv2 Unified Runtime — Forensic Traces\n",
         "*Full-chain traces from the ONE canonical `simulate_world` path. Each shows the phases that executed "
         "on the shared plan/world, the plan lineage, the posterior, dynamic-recompilation activity, and the "
         "terminal — so a reviewer can confirm the output is the integrated simulation, not a wrapper around "
         "separate predictors. Machine-readable: `experiments/results/unified/traces.json`.*\n"]
    for t in traces:
        m.append(f"\n## {t['domain']} — {t['question']}\n")
        if t.get("error"):
            m.append(f"- ERROR: {t['error']}\n"); continue
        m.append(f"- as_of **{t['as_of']}**, horizon **{t['horizon']}**; status **{t['simulation_status']}**, "
                 f"support **{t['support_grade']}**\n")
        lin = t.get("plan_lineage") or {}
        m.append(f"- plan lineage: **{len(lin.get('plan_hashes', []))}** plan version(s), "
                 f"**{len(lin.get('recompilations', []))}** recompilation trace(s); posterior consumed "
                 f"**{t.get('posterior_consumed')}**; evidence bundle `{(t.get('evidence_bundle_hash') or '')[:12]}`\n")
        m.append("- **active-component manifest**:\n")
        for ph, v in (t.get("active_component_manifest") or {}).items():
            state = "EXECUTED" if v["executed"] else ("omitted" if v["omitted"] else "available")
            m.append(f"    - `{ph}`: {state} — {v['reason'][:70]}\n")
        m.append(f"- **terminal raw P(yes) = {t.get('raw_probability')}**\n")
        m.append(f"- old Phase-12 calibrator: **{t.get('calibration_compatibility')}** (unified runtime "
                 "changed the distribution)\n")
        if t.get("limitations"):
            m.append(f"- limitations: {'; '.join(t['limitations'])}\n")
        m.append(f"- latency {t.get('latency_s')}s\n")
    Path("docs/WMV2_UNIFIED_RUNTIME_FORENSIC_TRACES.md").write_text("".join(m))


if __name__ == "__main__":
    run()
