"""EXP-111: the SINGLE live Lean V2 pastcasting run — Banxico unanimous June 2026 vote.

Protocol (identical to the completed EXP-107/108 arms for this question):
    * the EXACT frozen BTF-3 row (qid cfb43147…), its evidence/resolution criterion verbatim;
    * exact as_of / horizon / seed 0; sealed-replay frozen-background bundle (no retrieval);
    * same consequential model family: deepseek-v4-flash, temperature 0.2, max_tokens 3600;
    * canonical `unified_runtime.simulate_world(..., execution_profile="lean_v2")`;
    * ONE foreground process; NO background monitors; NO automatic relaunches.

RESOURCE GUARD (benchmark-scoped, enforced through the production ConsumerComputeBudget so a
trip finalizes gracefully with the partial trace preserved — never a relaunch):
    target: 1-5 min wall, ideally 10-35 external calls
    hard benchmark stop: 10 min wall or 80 external calls -> report FAILED consumer target.

The known outcome (YES) and the stored comparison probabilities (~0.729 full fidelity,
~0.769 Lean V1) are NEVER given to the simulation — they are read from the stored artifacts
AFTER the run, for evaluation only.

Run: DEEPSEEK_API_KEY=..  python -m experiments.exp111_btf3_lean_v2_banxico
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
from experiments.exp107_btf3_full_fidelity_post127 import (MODEL, MAX_TOKENS, TEMPERATURE,
                                                           SEED)

BANXICO_QID = "cfb43147-d9d2-5bd9-903f-f449e9a5aecf"
OUT = Path("experiments/results/exp111_lean_v2_banxico.json")

#: benchmark guard (NOT the production default budget, which stays ~4x liberal)
GUARD_WALL_S = 600.0            # 10 minutes hard benchmark stop
GUARD_CALLS = 80

PRICE = {"input_per_m_cache_miss": 0.27, "input_per_m_cache_hit": 0.07,
         "output_per_m": 1.10}   # same recorded assumptions as exp109


def run() -> dict:
    from datetime import datetime, timezone

    from experiments.btf3_frozen_bundle import frozen_background_bundle
    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world

    if OUT.exists():
        print("EXP-111 result already exists — never re-run (delete the artifact to redo)")
        return json.loads(OUT.read_text())

    rows = {r["question_id"]: r for r in fetch_btf3()}
    q = _forecast_input(rows[BANXICO_QID])
    evidence = (f"Resolution criteria: {q['resolution_criteria']}\n\n"
                f"Background (as of {str(q['present_date'])[:10]}): {q['background']}")
    as_of_ts = datetime.fromisoformat(str(q["present_date"]).split(".")[0]) \
        .replace(tzinfo=timezone.utc).timestamp()
    bundle = frozen_background_bundle(q["question"], as_of_ts=as_of_ts,
                                     background=q["background"],
                                     resolution_criteria=q["resolution_criteria"], seed=SEED)

    calls, pending_usage = [], []
    base_llm = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=MAX_TOKENS,
                                temperature=TEMPERATURE, usage_sink=pending_usage.append)

    def llm(prompt, _c=calls, _u=pending_usage):
        t = time.time()
        reply = base_llm(prompt)
        _c.append({"i": len(_c), "prompt_chars": len(prompt),
                   "reply_chars": len(reply or ""),
                   "latency_s": round(time.time() - t, 3),
                   "usage": (_u.pop() if _u else None),
                   "prompt_head": prompt[:1200], "reply_head": (reply or "")[:1200]})
        return reply

    t0 = time.time()
    res = simulate_world(
        q["question"], llm=llm, evidence=evidence,
        as_of=str(q["present_date"])[:10],
        horizon=str(q["expected_resolution_date"])[:10], seed=SEED,
        prebuilt_bundle=bundle,
        execution_policy={"lean_v2": {
            "budget": {"max_wall_s": GUARD_WALL_S, "max_calls": GUARD_CALLS},
            "backend_fingerprint": MODEL,
            "persistent_cache": True,
            "persistent_cache_dir": "experiments/results/exp111_compile_cache",
            "max_workers": 6}},
        execution_profile="lean_v2")
    wall = time.time() - t0

    d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
    usage_in = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in calls)
    usage_out = sum((c.get("usage") or {}).get("completion_tokens", 0) for c in calls)
    cache_hit = sum((c.get("usage") or {}).get("prompt_cache_hit_tokens", 0) for c in calls)
    cost = round((usage_in - cache_hit) / 1e6 * PRICE["input_per_m_cache_miss"]
                 + cache_hit / 1e6 * PRICE["input_per_m_cache_hit"]
                 + usage_out / 1e6 * PRICE["output_per_m"], 4)
    lv2 = (d.get("provenance") or {}).get("lean_v2") or {}
    eng = lv2.get("engine_primary") or {}
    dec = (eng.get("decisions") or {})
    bud = lv2.get("budget") or {}
    metrics = {
        "qid": BANXICO_QID, "question": q["question"][:140],
        "as_of": str(q["present_date"])[:10],
        "horizon": str(q["expected_resolution_date"])[:10],
        "profile": "lean_v2", "model": MODEL, "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE, "seed": SEED,
        "status": d.get("simulation_status"),
        "p_raw": d.get("raw_probability"), "p_cal": d.get("calibrated_probability"),
        "probability_source": d.get("probability_source"),
        "grounding_grade": d.get("grounding_grade"),
        "confidence": d.get("confidence"),
        "unresolved_mass": d.get("unresolved_mass"),
        "uncertainty_interval": d.get("uncertainty_interval"),
        "weight_sensitive": d.get("weight_sensitive"),
        "n_llm_calls": len(calls),
        "calls_by_stage": (bud.get("by_stage") or {}),
        "actor_calls": ((bud.get("by_stage") or {}).get("actor_decision") or {}).get("calls"),
        "unique_decision_contexts": dec.get("unique_decision_contexts"),
        "decision_reuses": dec.get("reuses"),
        "actors_pruned": (lv2.get("slice") or {}).get("n_pruned"),
        "aliases_merged": (lv2.get("slice") or {}).get("n_merged"),
        "structural_models": bud.get("structural_models"),
        "challenger": lv2.get("challenger"),
        "weighted_nodes_executed": (eng.get("coalescer") or {}).get("executed_unique_nodes"),
        "branches_merged": (eng.get("coalescer") or {}).get("merges"),
        "consequence_template_hits": (lv2.get("consequences") or {}).get("template_hits"),
        "novel_consequence_compiles": bud.get("novel_consequence_compiles"),
        "deliberations": len(eng.get("deliberations") or []),
        "stability_reruns": 0,
        "input_tokens": usage_in, "output_tokens": usage_out,
        "provider_cache_hit_tokens": cache_hit, "cost_usd": cost,
        "wall_clock_s": round(wall, 1),
        "preflight_verdict": (lv2.get("preflight") or {}).get("verdict"),
        "budget_exhausted_dimension": bud.get("exhausted_dimension"),
        "guard": {"wall_cap_s": GUARD_WALL_S, "call_cap": GUARD_CALLS,
                  "passed_hard_stop": wall <= GUARD_WALL_S and len(calls) <= GUARD_CALLS,
                  "hit_consumer_target": wall <= 300 and 10 <= len(calls) <= 35}}

    # -------- evaluation AFTER the run (outcome/known forecasts only enter here) ---------
    outcome = int(rows[BANXICO_QID]["resolution"])
    stored = _stored_comparators()
    evaluation = {"outcome": outcome,
                  "full_fidelity_p": stored.get("ff_p"),
                  "lean_v1_p": stored.get("l1_p"),
                  "lean_v2_p": metrics["p_raw"],
                  "brier": {
                      "full_fidelity": _brier(stored.get("ff_p"), outcome),
                      "lean_v1": _brier(stored.get("l1_p"), outcome),
                      "lean_v2": _brier(metrics["p_raw"], outcome)},
                  "correct_side": {
                      "full_fidelity": _side(stored.get("ff_p"), outcome),
                      "lean_v1": _side(stored.get("l1_p"), outcome),
                      "lean_v2": _side(metrics["p_raw"], outcome)},
                  "lean_v1_wall_s": stored.get("l1_wall_s"),
                  "lean_v1_calls": stored.get("l1_calls"),
                  "divergence_gt_0.10_from_both": (
                      metrics["p_raw"] is not None and stored.get("ff_p") is not None
                      and stored.get("l1_p") is not None
                      and abs(metrics["p_raw"] - stored["ff_p"]) > 0.10
                      and abs(metrics["p_raw"] - stored["l1_p"]) > 0.10)}

    out = {"experiment": "EXP-111 Lean V2 single live pastcast (Banxico unanimity)",
           "metrics": metrics, "evaluation": evaluation,
           "llm_calls": calls, "simulation_result": d}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"metrics": {k: v for k, v in metrics.items()
                                  if k not in ("calls_by_stage", "challenger")},
                      "evaluation": evaluation}, indent=1, default=str))
    if not metrics["guard"]["passed_hard_stop"]:
        print("LEAN V2 FAILED THE CONSUMER TARGET (hard benchmark stop exceeded) — partial "
              "trace preserved; NOT relaunching.")
    return out


def _brier(p, outcome):
    return None if p is None else round((float(p) - outcome) ** 2, 4)


def _side(p, outcome):
    return None if p is None else bool((float(p) > 0.5) == bool(outcome))


def _stored_comparators() -> dict:
    """The stored Banxico comparison values — read from the committed benchmark artifacts
    (never re-run, never passed to the model)."""
    out = {}
    try:
        comp = json.loads(Path("experiments/results/exp109_comparison.json").read_text())
        row = next(r for r in comp["per_question"] if r["qid"] == BANXICO_QID)
        out["ff_p"] = (row.get("full_fidelity") or {}).get("prediction")
        out["l1_p"] = (row.get("lean_adaptive") or {}).get("prediction")
        out["l1_wall_s"] = (row.get("lean_adaptive") or {}).get("wall_clock_s")
        out["l1_calls"] = (row.get("lean_adaptive") or {}).get("llm_calls")
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:160]
    return out


if __name__ == "__main__":
    run()
