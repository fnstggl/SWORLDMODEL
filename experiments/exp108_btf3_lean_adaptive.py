"""EXP-108: the lean-adaptive arm — the SAME five frozen BTF-3 questions as EXP-107, through
`unified_runtime.simulate_world(..., execution_profile="lean_adaptive")`.

The one-to-one comparison against the EXP-107 post-#127 full-fidelity baseline: same frozen rows,
same leakage protocol (sealed-replay frozen-background bundle; no live retrieval), same model and
provider family for consequential actor calls (deepseek-v4-flash, temperature 0.2, max_tokens
3600), same as_of/horizon/seed, same maximum particle budgets (lean may stop earlier ONLY under
the recorded §17 stopping conditions), full LLM actor cognition — no numeric actors anywhere.

Lean is ALLOWED fewer actual calls (cohort sharing, decision-equivalence cache, one-call
cognition, duplicate suppression, reversal-triggered structural models, progressive particles);
it is NOT allowed different semantics. Every §23 metric is captured per question, plus the lean
manifests (cohorts, decision contexts, cache, consequence cache, structural, particle stopping,
stability, avoided-call reasons).

Run: DEEPSEEK_API_KEY=..  python -m experiments.exp108_btf3_lean_adaptive <i>   # worker: question i
     DEEPSEEK_API_KEY=..  python -m experiments.exp108_btf3_lean_adaptive       # merge + score
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
from experiments.exp102_btf3_wmv2_full import QIDS
from experiments.exp107_btf3_full_fidelity_post127 import (MODEL, MAX_TOKENS, TEMPERATURE, SEED,
                                                           _extract_metrics, _merge_commit)


SCOREABLE_STATUSES = ("completed", "completed_with_degradation", "partially_resolved",
                      "under_modeled", "unresolved", "truncated", "temporally_truncated")


def usable_probability(m: dict):
    """Forecast availability is separate from grounding (forecast_recovery contract): score the
    probability whenever the run served one — the status and grounding_grade are reported
    alongside, never used to erase the forecast. Only malformed/failed runs (no probability at
    all) stay unscored."""
    if m.get("status") not in SCOREABLE_STATUSES:
        return None
    return m.get("p_cal") if m.get("p_cal") is not None else m.get("p_raw")


CKPT_DIR = Path("experiments/results/exp108_checkpoints")
SUMMARY = Path("experiments/results/exp108_lean_adaptive.json")


def _lean_metrics(d: dict) -> dict:
    """The lean-specific §23 metrics, straight from the manifests — measured, never projected."""
    lean = (d.get("provenance") or {}).get("lean") or {}
    ctl = lean.get("controller") or {}
    cache = ctl.get("decision_cache") or {}
    avoided = ctl.get("avoided_calls") or {}
    cohorts = ctl.get("cohorts") or {}
    prompts = ctl.get("prompts") or {}
    ccache = ctl.get("consequence_cache") or {}
    stopping = lean.get("particle_stopping") or []
    return {
        "one_call_successes": ctl.get("one_call_successes"),
        "escalations": ctl.get("escalations_by_reason"),
        "unique_decision_contexts": cache.get("unique_decision_contexts"),
        "decision_cache_hits": cache.get("hits"),
        "invalidated_cache_hits": cache.get("invalidated_hits"),
        "largest_context_reuse": cache.get("largest_context_reuse"),
        "actor_calls_avoided_total": avoided.get("avoided_calls_total"),
        "actor_calls_avoided_by_reason": avoided.get("avoided_by_reason"),
        "execution_classifications": avoided.get("execution_classifications"),
        "cohort_actors": {a: {k: c.get(k) for k in
                              ("n_cohorts", "generated", "collapsed_paraphrases", "expanded",
                               "under_modeled")}
                          for a, c in (cohorts.get("actors") or {}).items()},
        "largest_cohorts": cohorts.get("largest_cohorts"),
        "under_modeled_actors": cohorts.get("under_modeled_actors"),
        "prompt_chars_sent": prompts.get("prompt_chars_sent"),
        "prompt_chars_saved": prompts.get("chars_saved_vs_full_rerender"),
        "consequence_compile_calls": (ccache or {}).get("compile_calls"),
        "consequence_cache_reuses": (ccache or {}).get("reuses"),
        "structural": lean.get("structural"),
        "particle_stopping": [{k: s.get(k) for k in
                               ("model_id", "n_full_budget", "n_executed", "stopped_early",
                                "stop_reason", "particles_avoided", "forced_full_reasons")}
                              for s in stopping],
        "stability_signals": lean.get("stability_signals"),
        "stability_replicate": lean.get("stability_replicate"),
        "frontier_skips": ctl.get("frontier_skips"),
    }


def run_worker(i: int) -> dict:
    from datetime import datetime, timezone

    from experiments.btf3_frozen_bundle import frozen_background_bundle
    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world

    os.environ.setdefault("SWM_ACTOR_MAX_CALLS", "1000000")   # full actor cognition, both arms
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
                         execution_profile="lean_adaptive")
    wall = time.time() - t0
    d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
    metrics = {"qid": qid, "question": q["question"][:140], "as_of": str(q["present_date"])[:10],
               "horizon": str(q["expected_resolution_date"])[:10],
               "profile": "lean_adaptive", "model": MODEL, "max_tokens": MAX_TOKENS,
               "temperature": TEMPERATURE, "seed": SEED,
               **_extract_metrics(d, calls, wall), "lean": _lean_metrics(d)}
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
        results.append(json.loads(cpath.read_text())["metrics"])
    for r in results:
        row = rows[r["qid"]]
        r["outcome"] = int(row["resolution"])
        sota = row.get("sota_forecast_probability")
        r["p_sota"] = None if sota is None else round(float(sota) / 100.0, 4)
        p = usable_probability(r)
        r["p_used"] = p
        r["brier"] = None if p is None else round((p - r["outcome"]) ** 2, 4)
        r["correct_at_0.5"] = None if p is None else bool((p > 0.5) == r["outcome"])
    scored = [r for r in results if r["brier"] is not None]
    summary = {
        "experiment": "EXP-108 lean-adaptive arm (vs EXP-107 full-fidelity baseline)",
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
