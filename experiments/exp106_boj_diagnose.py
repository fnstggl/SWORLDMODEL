"""EXP-106: diagnose BoJ's empty-census/p=None. Run BoJ once through simulate_world with MAXIMAL capture —
status, failure taxonomy, limitations, phase-8 persistence manifest, operator census, event_time,
fallbacks, phase_integration_failures — and surface any raw exception. Answers: was it execution_failed
(an exception in the rollout) or completed-with-empty-census, and WHY.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp106_boj_diagnose
"""
from __future__ import annotations
import dataclasses
import json
import traceback
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input

QID = "7279494c-a775-5a57-a5f2-ac22252fb286"
OUT = Path("experiments/results/exp106_boj_diagnose.json")


def run():
    from swm.api.deepseek_backend import default_chat_fn
    import functools
    from swm.world_model_v2.unified_runtime import simulate_world as _sw_default
    # archival full-fidelity harness: pinned since the §25 default switch
    simulate_world = functools.partial(_sw_default, execution_profile="full_fidelity")
    rows = {r["question_id"]: r for r in fetch_btf3()}
    q = _forecast_input(rows[QID])
    llm0 = default_chat_fn(system="Reply ONLY JSON.", max_tokens=8000, temperature=0.2)
    n = {"c": 0}

    def llm(p):
        n["c"] += 1
        return llm0(p)

    rec = {"qid": QID, "question": q["question"]}
    try:
        res = simulate_world(q["question"], as_of=str(q["present_date"])[:10],
                             horizon=str(q["expected_resolution_date"])[:10], llm=llm, seed=0)
        d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
        prov = d.get("provenance") or {}
        acm = prov.get("active_component_manifest") or {}
        rec.update({
            "n_llm_calls": n["c"],
            "simulation_status": d.get("simulation_status"),
            "failure_taxonomy": d.get("failure_taxonomy"),
            "raw_probability": d.get("raw_probability"),
            "calibrated_probability": d.get("calibrated_probability"),
            "limitations": d.get("limitations"),
            "census_ops": sorted((prov.get("operator_delta_census") or {}).keys()),
            "operator_delta_census": prov.get("operator_delta_census"),
            "execution_degraded_fallback": prov.get("execution_degraded_fallback"),
            "evidence_sufficiency": prov.get("evidence_sufficiency"),
            "fully_integrated": prov.get("fully_integrated"),
            "phase_integration_failures": prov.get("phase_integration_failures"),
            "phase8_persistence": acm.get("phase8_persistence"),
            "phase10_institutions": acm.get("phase10_institutions"),
            "readout_var": prov.get("readout_var"), "readout_repaired": prov.get("readout_repaired"),
            "event_time": {k: (prov.get("event_time") or {}).get(k)
                           for k in ("n_absorbed", "n_particles", "p_censored")},
            "fallbacks_used": d.get("fallbacks_used"),
        })
        print(json.dumps({k: rec[k] for k in ("n_llm_calls", "simulation_status", "failure_taxonomy",
              "raw_probability", "census_ops", "execution_degraded_fallback", "phase8_persistence",
              "limitations", "event_time")}, indent=1, default=str))
    except Exception as e:  # noqa: BLE001
        rec["EXCEPTION"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        print("RAW EXCEPTION from simulate_world:\n", rec["traceback"])
    OUT.write_text(json.dumps(rec, indent=1, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run()
