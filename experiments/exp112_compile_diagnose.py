"""EXP-112: WHY does the ensemble compiler reject every candidate? (Step 1 of the compiler-collapse fix)

The frozen-5 audit showed 4/5 questions collapse at ensemble compilation (execution_failed /
invalid_execution_plan, zero executable models). This dumps, per candidate, the promotion_status +
promotion_reason + executability verdict for each failing question, so the rejection cause is inspected —
not guessed. The failing result already carries provenance.structural_ensemble_generation (ens.as_dict()),
whose `candidates` list records each disposition.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp112_compile_diagnose [label ...]
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input

QIDS = {
    "BoJ": "7279494c-a775-5a57-a5f2-ac22252fb286",
    "visionOS": "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6",
    "Wale": "741b4bed-7502-5cd2-9cbe-949fbc70f857",
    "Banxico": "cfb43147-d9d2-5bd9-903f-f449e9a5aecf",
    "Knesset": "0851f82c-aabd-57f0-abbb-4a23f99963c2",
}
OUT = Path("experiments/results/exp112_compile_diagnose.json")


def diagnose(label, qid, rows, llm):
    from swm.world_model_v2.unified_runtime import simulate_world
    q = _forecast_input(rows[qid])
    as_of, horizon = str(q["present_date"])[:10], str(q["expected_resolution_date"])[:10]
    res = simulate_world(q["question"], as_of=as_of, horizon=horizon, llm=llm, seed=0)
    d = res.__dict__
    prov = d.get("provenance") or {}
    gen = prov.get("structural_ensemble_generation") or {}
    cands = gen.get("candidates") or []
    rows_out = []
    for c in cands:
        val = c.get("validation") or {}
        rows_out.append({
            "model_id": c.get("model_id"), "role": c.get("generation_role"),
            "status": c.get("promotion_status"), "reason": (c.get("promotion_reason") or "")[:200],
            "executable": val.get("executable"), "why_nonexec": (val.get("why") or "")[:200],
            "n_decisive_actors": len(c.get("decisive_actors") or []),
            "n_decisive_mechanisms": len(c.get("decisive_mechanisms") or []),
        })
    rec = {
        "label": label, "qid": qid, "question": q["question"][:90],
        "final_status": d.get("simulation_status"),
        "final_taxonomy": d.get("failure_taxonomy"),
        "grounded_floor_used": bool((prov.get("grounded_outside_view_fallback") or {}).get("used")),
        "n_candidates": len(cands),
        "status_counts": dict(Counter(c["status"] for c in rows_out)),
        "reason_samples": rows_out,
        "stopping_reason": gen.get("stopping_reason"),
    }
    print(f"\n=== {label} ({qid[:8]}) final={rec['final_status']}/{rec['final_taxonomy']} "
          f"floor={rec['grounded_floor_used']} n_cand={rec['n_candidates']} ===")
    print(f"  status_counts: {rec['status_counts']}  stopping_reason: {rec['stopping_reason']}")
    for r in rows_out:
        print(f"  - {r['status']:10s} exec={r['executable']} | {r['reason'][:120]}")
        if r["why_nonexec"]:
            print(f"      why_nonexec: {r['why_nonexec']}")
    return rec


def run(labels):
    from swm.api.deepseek_backend import default_chat_fn
    rows = {r["question_id"]: r for r in fetch_btf3()}
    llm = default_chat_fn(system="Reply ONLY JSON.", max_tokens=8000, temperature=0.2)
    out = []
    for label in labels:
        try:
            out.append(diagnose(label, QIDS[label], rows, llm))
        except Exception as e:  # noqa: BLE001
            out.append({"label": label, "error": f"{type(e).__name__}: {e}"[:200]})
            print(f"  {label} ERROR: {type(e).__name__}: {e}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {OUT}")
    return out


if __name__ == "__main__":
    run(sys.argv[1:] or ["BoJ", "Knesset"])
