"""EXP-105: re-run BoJ + visionOS through the FIXED simulate_world (evidence-sufficiency gate + retry,
recurrence-aware scheduled-facts calendar), scored against the EXP-104 baseline. Two questions only,
fresh (post-fix). Captures p, operator census, evidence_sufficiency, and the scheduled-reality lineage so
we can see the calendar fact actually move the forecast.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp105_rerun_boj_visionos
"""
from __future__ import annotations
import dataclasses
import json
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input

QIDS = ["7279494c-a775-5a57-a5f2-ac22252fb286",   # BoJ June hike (EXP-104: 0.057, actual YES)
        "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6"]   # visionOS 27 @ WWDC (EXP-104: 0.493, actual YES)
BASELINE = {"7279494c-a775-5a57-a5f2-ac22252fb286": 0.0566,
            "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6": 0.493}
CKPT = Path("experiments/results/exp105_checkpoints")
OUT = Path("experiments/results/exp105_rerun.json")


def run():
    from swm.api.deepseek_backend import default_chat_fn
    import functools
    from swm.world_model_v2.unified_runtime import simulate_world as _sw_default
    # archival full-fidelity harness: pinned since the §25 default switch
    simulate_world = functools.partial(_sw_default, execution_profile="full_fidelity")
    CKPT.mkdir(parents=True, exist_ok=True)
    rows = {r["question_id"]: r for r in fetch_btf3()}
    base = default_chat_fn(system="Reply ONLY JSON.", max_tokens=8000, temperature=0.2)
    results = []
    for qid in QIDS:
        cp = CKPT / f"{qid}.json"
        if cp.exists():
            r = json.loads(cp.read_text()); results.append(r)
            print(f"  {qid[:8]} [resumed] p={r['p_used']}"); continue
        q = _forecast_input(rows[qid])
        calls = []

        def llm(p, _c=calls):
            _c.append(1); return base(p)

        res = simulate_world(q["question"], as_of=str(q["present_date"])[:10],
                             horizon=str(q["expected_resolution_date"])[:10], llm=llm, seed=0)
        d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
        prov = d.get("provenance") or {}
        p = d.get("calibrated_probability") if d.get("calibrated_probability") is not None else d.get("raw_probability")
        r = {"qid": qid, "question": q["question"][:90], "p_used": p, "n_llm_calls": len(calls),
             "outcome": int(rows[qid]["resolution"]),
             "p_sota": (None if rows[qid].get("sota_forecast_probability") is None
                        else round(float(rows[qid]["sota_forecast_probability"]) / 100.0, 4)),
             "baseline_exp104": BASELINE.get(qid),
             "evidence_sufficiency": prov.get("evidence_sufficiency"),
             "scheduled_reality": (d.get("provenance", {}) or {}).get("scheduled_reality")
                                  or _lineage_sched(d),
             "census_ops": sorted((prov.get("operator_delta_census") or {}).keys())}
        r["brier_new"] = None if p is None else round((p - r["outcome"]) ** 2, 4)
        r["brier_baseline"] = round((BASELINE.get(qid, 0.5) - r["outcome"]) ** 2, 4)
        cp.write_text(json.dumps(r, default=str))
        results.append(r)
        print(f"  {qid[:8]} p={p} (was {BASELINE.get(qid)})  brier {r['brier_new']} (was {r['brier_baseline']})  "
              f"starved={(r['evidence_sufficiency'] or {}).get('starved')}  calls={len(calls)}")
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print("\n=== EXP-105 (fixed path) vs EXP-104 baseline ===")
    for r in results:
        print(f"  {r['question'][:55]:55s} new={r['p_used']} was={r['baseline_exp104']} "
              f"actual={r['outcome']} sota={r['p_sota']}")
    print(f"wrote {OUT}")


def _lineage_sched(d):
    return None


if __name__ == "__main__":
    run()
