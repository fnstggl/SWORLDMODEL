"""EXP-107: the POST-PR-#127 full-fidelity baseline — the 5 frozen BTF-3 questions of EXP-102/104
through `unified_runtime.simulate_world(..., execution_profile="full_fidelity")`.

The previous five-question full-fidelity runs (EXP-104, EXP-105) predate the PR #124/#125/#126
reconciliation merged by PR #127 (merge commit recorded in the run manifest), so they are not a valid
comparison baseline for the lean-adaptive runtime. This experiment produces the ONE fresh
post-#127 baseline the lean end-to-end comparison (EXP-108) is scored against.

Frozen inputs (never reconstructed from titles): the exact BTF-3 rows of EXP-102 `QIDS` —
BoJ June-2026 hike, visionOS 27 at WWDC-2026, Matthew Wale PM, Hormuz tanker transits, Banxico
unanimous vote — with the benchmark's own question / resolution_criteria / background / present_date /
expected_resolution_date fields, via the EXP-101 leakage quarantine (allowlisted fields only; hard
assert on forbidden keys; resolutions + SOTA join at scoring time only).

Leakage protocol (same as EXP-105, enforced in code):
  * `phase2_evidence` dropped via execution_policy — live retrieval cannot reach past as_of;
    the caller-supplied `evidence` is the benchmark's own as-of background + resolution criteria;
  * forecaster sees ONLY ALLOWED_FIELDS; answers join at scoring.

Full-fidelity configuration (exact EXP-105 conventions):
  * model deepseek-v4-flash, temperature 0.2, max_tokens 3600 (8000 exceeds the 120s HTTP timeout
    and livelocks — EXP-105 finding), system "Reply ONLY JSON.";
  * SWM_ACTOR_MAX_CALLS lifted — FULL LLM actor cognition, no numeric-actor substitute;
  * default structural ensemble, default particle budgets, seed 0, serial branches.

Metering: every backend call's real provider usage block (prompt/completion/cached tokens, latency)
is recorded via the backend's usage_sink — token/cost manifests contain provider numbers, never
estimates. Per-question checkpoints under experiments/results/exp107_checkpoints/ are crash-safe and
resumable; a completed question is NEVER re-run.

Run: DEEPSEEK_API_KEY=..  python -m experiments.exp107_btf3_full_fidelity_post127 <i>   # worker: question i
     DEEPSEEK_API_KEY=..  python -m experiments.exp107_btf3_full_fidelity_post127       # merge + score
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
from experiments.exp102_btf3_wmv2_full import QIDS

CKPT_DIR = Path("experiments/results/exp107_checkpoints")
SUMMARY = Path("experiments/results/exp107_full_fidelity.json")

MODEL = "deepseek-v4-flash"
MAX_TOKENS = 3600
TEMPERATURE = 0.2
SEED = 0


def _merge_commit() -> str:
    """The PR #127 merge commit this baseline runs on top of (recorded, never asserted from memory)."""
    try:
        out = subprocess.run(["git", "log", "--format=%H %s", "-200"], capture_output=True,
                             text=True, check=True).stdout
        return next((ln.split()[0] for ln in out.splitlines() if "#127" in ln), "")
    except Exception:  # noqa: BLE001
        return ""


def _extract_metrics(d: dict, calls: list, wall_s: float) -> dict:
    """Every §23 metric the result surfaces, extracted from provenance — never invented."""
    prov = d.get("provenance") or {}
    census = prov.get("operator_delta_census") or {}
    cost_man = prov.get("ensemble_cost_manifest") or {}
    ens = prov.get("structural_ensemble_generation") or {}
    sim_man = ens.get("simulation_manifest") or {}
    usage = [c.get("usage") or {} for c in calls]
    return {
        "status": d.get("simulation_status"),
        "p_raw": d.get("raw_probability"), "p_cal": d.get("calibrated_probability"),
        "n_llm_calls": len(calls),
        "llm_calls_by_stage": cost_man.get("llm_calls_by_stage"),
        "cache_hits_by_stage": cost_man.get("cache_hits_by_stage"),
        "input_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("completion_tokens", 0) for u in usage),
        "provider_cache_hit_tokens": sum(u.get("prompt_cache_hit_tokens", 0) for u in usage),
        "provider_cache_miss_tokens": sum(u.get("prompt_cache_miss_tokens", 0) for u in usage),
        "wall_clock_s": round(wall_s, 1), "latency_s": d.get("latency_s"),
        "structural_models_generated": len(ens.get("candidates", []) or []) or None,
        "structural_models_simulated": len(sim_man) or None,
        "particles_by_model": {m: v.get("final_particles") for m, v in sim_man.items()} or None,
        "operator_delta_census": census, "census_ops": sorted(census.keys()),
        "outcome_pathway": prov.get("outcome_pathway"),
        "support_grade": d.get("support_grade"),
        "limitations": (d.get("limitations") or [])[:12],
        "failure_taxonomy": d.get("failure_taxonomy"),
        "truncation": [c["i"] for c in calls
                       if (c.get("usage") or {}).get("completion_tokens", 0) >= MAX_TOKENS][:40],
    }


def run_worker(i: int) -> dict:
    from datetime import datetime, timezone

    from experiments.btf3_frozen_bundle import frozen_background_bundle
    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world

    os.environ.setdefault("SWM_ACTOR_MAX_CALLS", "1000000")   # FULL actor cognition (EXP-105 convention)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    qid = QIDS[i]
    cpath = CKPT_DIR / f"{qid}.json"
    if cpath.exists():
        print(f"worker {i}: {qid[:8]} already checkpointed — never re-run")
        return json.loads(cpath.read_text())["metrics"]

    rows = {r["question_id"]: r for r in fetch_btf3()}
    q = _forecast_input(rows[qid])
    evidence = (f"Resolution criteria: {q['resolution_criteria']}\n\n"
                f"Background (as of {str(q['present_date'])[:10]}): {q['background']}")
    # sealed-replay injection: the benchmark's own as-of background as a FROZEN bundle, so the
    # Phase-3 posterior is evidence-updated exactly as production would be (post-#127, a run
    # without a bundle has no §NAP-admissible fallback when a world cannot bind its outcome —
    # the pre-#127 harnesses only needed the conditioning text). Same bundle in BOTH arms.
    as_of_ts = datetime.fromisoformat(str(q["present_date"]).split(".")[0]) \
        .replace(tzinfo=timezone.utc).timestamp()
    bundle = frozen_background_bundle(q["question"], as_of_ts=as_of_ts,
                                      background=q["background"],
                                      resolution_criteria=q["resolution_criteria"], seed=SEED)

    calls = []
    pending_usage = []
    base_llm = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=MAX_TOKENS,
                                temperature=TEMPERATURE, usage_sink=pending_usage.append)

    def llm(prompt, _c=calls, _u=pending_usage):
        t = time.time()
        reply = base_llm(prompt)
        _c.append({"i": len(_c), "prompt_chars": len(prompt), "reply_chars": len(reply or ""),
                   "latency_s": round(time.time() - t, 3),
                   "usage": (_u.pop() if _u else None),
                   "prompt_head": prompt[:1500], "reply_head": (reply or "")[:1500]})
        return reply

    t0 = time.time()
    res = simulate_world(q["question"], llm=llm, evidence=evidence,
                         as_of=str(q["present_date"])[:10],
                         horizon=str(q["expected_resolution_date"])[:10], seed=SEED,
                         prebuilt_bundle=bundle,
                         execution_profile="full_fidelity")
    wall = time.time() - t0
    d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
    metrics = {"qid": qid, "question": q["question"][:140], "as_of": str(q["present_date"])[:10],
               "horizon": str(q["expected_resolution_date"])[:10],
               "profile": "full_fidelity", "model": MODEL, "max_tokens": MAX_TOKENS,
               "temperature": TEMPERATURE, "seed": SEED,
               **_extract_metrics(d, calls, wall)}
    cpath.write_text(json.dumps({"metrics": metrics, "llm_calls": calls,
                                 "simulation_result": d}, default=str))
    print(f"worker {i}: {qid[:8]} status={metrics['status']} p={metrics['p_cal'] or metrics['p_raw']} "
          f"calls={metrics['n_llm_calls']} wall={metrics['wall_clock_s']}s [checkpointed]")
    return metrics


def merge_and_score() -> dict:
    rows = {r["question_id"]: r for r in fetch_btf3()}
    results = []
    for qid in QIDS:
        cpath = CKPT_DIR / f"{qid}.json"
        if not cpath.exists():
            print(f"MISSING checkpoint for {qid[:8]} — run its worker first")
            continue
        m = json.loads(cpath.read_text())["metrics"]
        results.append(m)
    for r in results:
        row = rows[r["qid"]]
        r["outcome"] = int(row["resolution"])
        sota = row.get("sota_forecast_probability")
        r["p_sota"] = None if sota is None else round(float(sota) / 100.0, 4)
        p = r["p_cal"] if r["p_cal"] is not None else r["p_raw"]
        r["p_used"] = p
        r["brier"] = None if p is None else round((p - r["outcome"]) ** 2, 4)
        r["correct_at_0.5"] = None if p is None else bool((p > 0.5) == r["outcome"])
    scored = [r for r in results if r["brier"] is not None]
    summary = {
        "experiment": "EXP-107 post-PR-127 full-fidelity baseline",
        "pr127_merge_commit": _merge_commit(),
        "n": len(results), "n_scored": len(scored),
        "brier": round(sum(r["brier"] for r in scored) / len(scored), 4) if scored else None,
        "accuracy_at_0.5": round(sum(r["correct_at_0.5"] for r in scored) / len(scored), 4)
        if scored else None,
        "total_llm_calls": sum(r["n_llm_calls"] for r in results),
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "total_provider_cache_hit_tokens": sum(r["provider_cache_hit_tokens"] for r in results),
        "total_wall_clock_s": round(sum(r["wall_clock_s"] for r in results), 1),
        "results": [{k: v for k, v in r.items()
                     if k not in ("operator_delta_census", "outcome_pathway")} for r in results]}
    SUMMARY.write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_worker(int(sys.argv[1]))
    else:
        merge_and_score()
