"""EXP-105: the FULL unified world-model-v2 runtime on 5 FRESH BTF-3 pastcasting questions — traced.

EXP-102 ran `pipeline.simulate` — which the audit exposed as the BARE inner path (no calendar layer, no
institution normalization, no activation synthesis), with a 2200-token budget that truncated every
decomposition. This experiment runs the REAL top entry, `unified_runtime.simulate_world`, after the
validation-gate removal (run-everything principle, general-path hardening, truncation recovery,
recurring-events calendar, actor knowledge scoping), on 5 BTF-3 questions DISJOINT from EXP-102's.

Leakage protocol (same as EXP-101/102, enforced in code):
  * forecaster sees ONLY ALLOWED_FIELDS; resolutions/SOTA join at scoring; hard assert on forbidden keys;
  * `phase2_evidence` is DROPPED via execution_policy — no live retrieval can reach past the as_of date;
    the caller-supplied `evidence` is the benchmark's own as-of background + resolution criteria;
  * every LLM call (prompt + reply) captured verbatim for the under-the-hood anatomy.

n=5 is a trace/anatomy run, not a benchmark claim.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp105_btf3_simulate_world
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input

#: EXP-102's five (excluded so this run is on FRESH questions)
EXP102_QIDS = {"7279494c-a775-5a57-a5f2-ac22252fb286", "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6",
               "741b4bed-7502-5cd2-9cbe-949fbc70f857", "017e64ef-7354-56c4-8a4d-e27121bc639a",
               "cfb43147-d9d2-5bd9-903f-f449e9a5aecf"}

SAMPLE_IDS = Path("experiments/results/exp101_btf3_sample_ids.json")
TRACES = Path("experiments/results/exp105_simulate_world_traces.json")
SUMMARY = Path("experiments/results/exp105_simulate_world.json")


def pick_fresh_qids(rows, n=5) -> list:
    """Deterministic: the first n ids of the committed EXP-101 sample not used by EXP-102 (paired history
    stays intact; no cherry-picking)."""
    committed = json.loads(SAMPLE_IDS.read_text()) if SAMPLE_IDS.exists() else sorted(
        r["question_id"] for r in rows)
    return [qid for qid in committed if qid not in EXP102_QIDS][:n]


def run() -> dict:
    from swm.api.deepseek_backend import default_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world

    rows = {r["question_id"]: r for r in fetch_btf3()}
    qids = pick_fresh_qids(list(rows.values()))
    # 3600 tokens: enough for an untruncated decomposition with the reordered schema, while staying
    # inside the backend's 120s HTTP timeout (6000 tokens exceeds it and livelocks on retries); any
    # residual clipping is handled by the compiler's truncation-recovery continuation call.
    base_llm = default_chat_fn(system="Reply ONLY JSON.", max_tokens=3600, temperature=0.2)
    traces, results = [], []

    for qid in qids:
        q = _forecast_input(rows[qid])
        evidence = (f"Resolution criteria: {q['resolution_criteria']}\n\n"
                    f"Background (as of {str(q['present_date'])[:10]}): {q['background']}")
        calls = []

        def llm(prompt, _calls=calls):
            reply = base_llm(prompt)
            _calls.append({"i": len(_calls), "prompt_chars": len(prompt), "prompt": prompt[:8000],
                           "reply": reply})
            return reply

        res = simulate_world(q["question"], llm=llm, evidence=evidence,
                             as_of=str(q["present_date"])[:10],
                             horizon=str(q["expected_resolution_date"])[:10], seed=0,
                             execution_policy={"drop_phases": ["phase2_evidence"]})
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
              f"p_cal={res.calibrated_probability}  llm_calls={len(calls)}  {res.latency_s}s",
              flush=True)
        TRACES.write_text(json.dumps(traces, indent=1, default=str))   # checkpoint per question

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
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    print(f"wrote {SUMMARY} and full traces to {TRACES}")
    return summary


if __name__ == "__main__":
    run()
