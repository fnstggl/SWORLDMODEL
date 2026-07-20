"""EXP-104: the 5 BTF-3 questions through the REAL top entry `unified_runtime.simulate_world` — not the
bare `pipeline.simulate` EXP-102 used. simulate_world is what swm/facade.py and every other backtest
harness calls; it adds Phase-10 institution normalization, fidelity expansion, the scheduled-reality
(calendar) layer, resolution-criterion parsing and mode graphs before/around the rollout. Adequate token
budget (8000) so the decomposition is not truncated (EXP-102 used 2200 -> all 5 truncated before emitting
mechanisms). Same leakage quarantine as EXP-101/102; answers join only at scoring.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp104_btf3_simulate_world
"""
from __future__ import annotations
import dataclasses
import json
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
from experiments.exp102_btf3_wmv2_full import QIDS

TRACES = Path("experiments/results/exp104_simulate_world_traces.json")
SUMMARY = Path("experiments/results/exp104_simulate_world.json")
CKPT_DIR = Path("experiments/results/exp104_checkpoints")            # per-question checkpoints (resume-safe)


def run(tag: str = "") -> dict:
    from swm.api.deepseek_backend import default_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world

    ck = CKPT_DIR if not tag else CKPT_DIR.with_name(f"exp104_checkpoints_{tag}")
    ck.mkdir(parents=True, exist_ok=True)
    rows = {r["question_id"]: r for r in fetch_btf3()}
    base = default_chat_fn(system="Reply ONLY JSON.", max_tokens=8000, temperature=0.2)
    traces, results = [], []

    for qid in QIDS:
        cpath = ck / f"{qid}.json"
        if cpath.exists():                                          # RESUME: skip already-simulated questions
            saved = json.loads(cpath.read_text())
            traces.append(saved["trace"])
            results.append(saved["result"])
            print(f"  {qid[:8]}  [resumed from checkpoint]  p={saved['result'].get('p_used')}")
            continue
        q = _forecast_input(rows[qid])
        calls = []

        def llm(prompt, _c=calls):
            reply = base(prompt)
            _c.append({"i": len(_c), "prompt_chars": len(prompt), "prompt": prompt[:6000], "reply": reply})
            return reply

        res = simulate_world(q["question"], as_of=str(q["present_date"])[:10],
                             horizon=str(q["expected_resolution_date"])[:10],
                             prebuilt_bundle=None, llm=llm, seed=0)
        d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
        census = (d.get("provenance") or {}).get("operator_delta_census")
        trace = {"qid": qid, "question": q["question"], "n_llm_calls": len(calls),
                 "status": d.get("simulation_status"), "p_raw": d.get("raw_probability"),
                 "p_cal": d.get("calibrated_probability"),
                 "operator_delta_census": census, "fallbacks_used": d.get("fallbacks_used"),
                 "mechanism_tiers": d.get("mechanism_tiers"), "support_grade": d.get("support_grade"),
                 "llm_calls": calls, "simulation_result": d}
        p = d.get("calibrated_probability") if d.get("calibrated_probability") is not None else d.get("raw_probability")
        result = {"qid": qid, "question": q["question"][:110], "status": d.get("simulation_status"),
                  "p_used": p, "n_llm_calls": len(calls),
                  "census_ops": sorted((census or {}).keys()) if census else []}
        # CHECKPOINT this question immediately (crash-safe, incremental, verifiable)
        cpath.write_text(json.dumps({"trace": trace, "result": result}, default=str))
        traces.append(trace)
        results.append(result)
        print(f"  {qid[:8]}  status={d.get('simulation_status')}  p={p}  calls={len(calls)}  "
              f"ops={sorted((census or {}).keys()) if census else []}  [checkpointed]")

    for r in results:
        r["outcome"] = int(rows[r["qid"]]["resolution"])
        sota = rows[r["qid"]].get("sota_forecast_probability")
        r["p_sota"] = None if sota is None else round(float(sota) / 100.0, 4)
        p = r["p_used"]
        r["brier"] = None if p is None else round((p - r["outcome"]) ** 2, 4)
        r["correct_at_0.5"] = None if p is None else bool((p > 0.5) == r["outcome"])
    scored = [r for r in results if r["brier"] is not None]
    summary = {"n": len(results), "n_scored": len(scored),
               "brier": round(sum(r["brier"] for r in scored) / len(scored), 4) if scored else None,
               "accuracy_at_0.5": round(sum(r["correct_at_0.5"] for r in scored) / len(scored), 4) if scored else None,
               "sota_brier_same_qs": round(sum((r["p_sota"] - r["outcome"]) ** 2 for r in scored
                                               if r["p_sota"] is not None)
                                           / max(1, len([r for r in scored if r["p_sota"] is not None])), 4),
               "results": results}
    TRACES.write_text(json.dumps(traces, indent=1, default=str))
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    return summary


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "")
