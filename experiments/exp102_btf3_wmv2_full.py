"""EXP-102: the FULL world-model-v2 simulator on 5 BTF-3 pastcasting questions — top to bottom, traced.

EXP-101 ran the thin mechanism layer (one LLM call -> params -> kernel integral). This runs the canonical
production entry `swm.world_model_v2.pipeline.simulate` — compiler decomposition (outcome contract, actors,
latent variables, mechanism plan) -> operator execution over particles -> readout -> calibration — on the
same questions, with EVERY LLM call (prompt + reply) captured verbatim, so the under-the-hood claim is
inspectable rather than asserted. Same leakage quarantine as EXP-101 (allowlisted fields only; the
`evidence` string is the benchmark's own as-of background + resolution criteria; answers join at scoring).

n=5 is a trace/anatomy run, not a benchmark claim.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp102_btf3_wmv2_full
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input

QIDS = ["7279494c-a775-5a57-a5f2-ac22252fb286",   # BoJ June rate hike (whipcount in EXP-101)
        "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6",   # visionOS 27 at WWDC (arrival)
        "741b4bed-7502-5cd2-9cbe-949fbc70f857",   # Matthew Wale PM (contest)
        "017e64ef-7354-56c4-8a4d-e27121bc639a",   # Hormuz tanker transits >=50/day (diffusion)
        "cfb43147-d9d2-5bd9-903f-f449e9a5aecf"]   # Banxico unanimous decision (aggregation)

TRACES = Path("experiments/results/exp102_wmv2_full_traces.json")
SUMMARY = Path("experiments/results/exp102_wmv2_full.json")


def run() -> dict:
    from swm.api.deepseek_backend import default_chat_fn
    from swm.world_model_v2.pipeline import simulate

    rows = {r["question_id"]: r for r in fetch_btf3()}
    base_llm = default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)
    traces, results = [], []

    for qid in QIDS:
        q = _forecast_input(rows[qid])
        evidence = (f"Resolution criteria: {q['resolution_criteria']}\n\n"
                    f"Background (as of {str(q['present_date'])[:10]}): {q['background']}")
        calls = []

        def llm(prompt, _calls=calls):
            reply = base_llm(prompt)
            _calls.append({"i": len(_calls), "prompt_chars": len(prompt), "prompt": prompt[:8000],
                           "reply": reply})
            return reply

        res = simulate(q["question"], llm=llm, evidence=evidence,
                       as_of=str(q["present_date"])[:10],
                       horizon=str(q["expected_resolution_date"])[:10], seed=0)
        d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
        traces.append({"qid": qid, "question": q["question"], "as_of": str(q["present_date"])[:10],
                       "horizon": str(q["expected_resolution_date"])[:10],
                       "evidence_chars": len(evidence), "n_llm_calls": len(calls),
                       "llm_calls": calls, "simulation_result": d})
        results.append({"qid": qid, "question": q["question"][:110],
                        "status": res.simulation_status,
                        "p_raw": res.raw_probability, "p_cal": res.calibrated_probability,
                        "n_llm_calls": len(calls), "latency_s": res.latency_s})
        print(f"  {qid[:8]}  status={res.simulation_status}  p_raw={res.raw_probability}  "
              f"p_cal={res.calibrated_probability}  llm_calls={len(calls)}  {res.latency_s}s")

    # ---- scoring only from here on ----
    for r in results:
        r["outcome"] = int(rows[r["qid"]]["resolution"])
        sota = rows[r["qid"]].get("sota_forecast_probability")
        r["p_sota"] = None if sota is None else round(float(sota) / 100.0, 4)
        p = r["p_cal"] if r["p_cal"] is not None else r["p_raw"]
        r["p_used"] = p
        r["brier"] = None if p is None else round((p - r["outcome"]) ** 2, 4)
        r["correct_at_0.5"] = None if p is None else bool((p > 0.5) == r["outcome"])
    scored = [r for r in results if r["brier"] is not None]
    summary = {"n": len(results), "n_scored": len(scored),
               "brier": round(sum(r["brier"] for r in scored) / len(scored), 4) if scored else None,
               "accuracy_at_0.5": (round(sum(r["correct_at_0.5"] for r in scored) / len(scored), 4)
                                   if scored else None),
               "sota_brier_same_qs": round(sum((r["p_sota"] - r["outcome"]) ** 2 for r in scored
                                               if r["p_sota"] is not None)
                                           / max(1, len([r for r in scored if r["p_sota"] is not None])), 4),
               "results": results}
    TRACES.write_text(json.dumps(traces, indent=1, default=str))
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    print(f"wrote {SUMMARY} and full traces to {TRACES}")
    return summary


if __name__ == "__main__":
    run()
