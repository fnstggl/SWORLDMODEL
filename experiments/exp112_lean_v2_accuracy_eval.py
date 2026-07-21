"""EXP-112: the Lean V2 accuracy-architecture five-question evaluation.

ONE question at a time (Banxico → BoJ → visionOS → Wale → Hormuz), each through the canonical
`unified_runtime.simulate_world(..., execution_profile="lean_v2")` with the exact frozen BTF-3
row, evidence, resolution criterion, as_of, horizon, sealed-replay bundle, and the deepseek
consequential-actor family. No numeric actors.

LEAKAGE: the actual outcome and the stored Lean V1 / full-fidelity predictions are NEVER passed
to any research/compilation/actor/weighting/calibration/simulation stage. They are read from the
committed artifacts ONLY after this arm's forecast is frozen and checkpointed, for the comparison.

GUARD (per question): target 1-5 min / 10-35 calls; hard stop 10 min / 80 calls, enforced through
the production ConsumerComputeBudget so a trip finalizes the best labeled forecast — never a
relaunch. One process, foreground, no monitors.

Run one worker at a time:
    python -m experiments.exp112_lean_v2_accuracy_eval <i>     # i in 0..4, Banxico=0
    python -m experiments.exp112_lean_v2_accuracy_eval          # merge + final comparison
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
from experiments.exp102_btf3_wmv2_full import QIDS
from experiments.exp107_btf3_full_fidelity_post127 import (MODEL, MAX_TOKENS, TEMPERATURE, SEED)

# canonical evaluation order (Banxico first)
ORDER = ["cfb43147-d9d2-5bd9-903f-f449e9a5aecf",   # 0 Banxico
         "7279494c-a775-5a57-a5f2-ac22252fb286",   # 1 BoJ
         "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6",   # 2 visionOS
         "741b4bed-7502-5cd2-9cbe-949fbc70f857",   # 3 Wale
         "017e64ef-7354-56c4-8a4d-e27121bc639a"]   # 4 Hormuz
NAMES = {ORDER[0]: "Banxico", ORDER[1]: "BoJ", ORDER[2]: "visionOS", ORDER[3]: "Wale",
         ORDER[4]: "Hormuz"}

CKPT = Path("experiments/results/exp112_checkpoints")
SUMMARY = Path("experiments/results/exp112_lean_v2_eval.json")
GUARD_WALL_S, GUARD_CALLS = 600.0, 80
PRICE = {"input_per_m_cache_miss": 0.27, "input_per_m_cache_hit": 0.07, "output_per_m": 1.10}


def run_worker(i: int) -> dict:
    from datetime import datetime, timezone

    from experiments.btf3_frozen_bundle import frozen_background_bundle
    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world

    qid = ORDER[i]
    CKPT.mkdir(parents=True, exist_ok=True)
    cpath = CKPT / f"{qid}.json"
    if cpath.exists():
        print(f"{NAMES[qid]} already checkpointed — never re-run (delete to redo)")
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
    calls, pending = [], []
    base = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE, usage_sink=pending.append)

    def llm(prompt, _c=calls, _u=pending):
        t = time.time()
        reply = base(prompt)
        _c.append({"i": len(_c), "prompt_chars": len(prompt), "reply_chars": len(reply or ""),
                   "latency_s": round(time.time() - t, 3), "usage": (_u.pop() if _u else None)})
        return reply

    t0 = time.time()
    res = simulate_world(
        q["question"], llm=llm, evidence=evidence, as_of=str(q["present_date"])[:10],
        horizon=str(q["expected_resolution_date"])[:10], seed=SEED, prebuilt_bundle=bundle,
        execution_policy={"lean_v2": {
            "budget": {"max_wall_s": GUARD_WALL_S, "max_calls": GUARD_CALLS},
            "backend_fingerprint": MODEL, "persistent_cache": True,
            "persistent_cache_dir": f"experiments/results/exp112_cache/{NAMES[qid].lower()}",
            "qid": qid, "max_workers": 6}},
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
    metrics = _extract(qid, q, d, calls, wall, usage_in, usage_out, cache_hit, cost, lv2)
    # FREEZE before joining any outcome
    cpath.write_text(json.dumps({"metrics": metrics, "simulation_result": d,
                                 "n_calls": len(calls)}, default=str))
    _print_human_report(qid, metrics, lv2, d)
    _compare_after_freeze(qid, metrics, rows)
    cpath.write_text(json.dumps({"metrics": metrics, "simulation_result": d,
                                 "n_calls": len(calls)}, default=str))
    guard = metrics["guard"]
    if not guard["passed_hard_stop"]:
        print(f"\n{NAMES[qid]}: FAILED consumer hard stop ({wall:.0f}s / {len(calls)} calls) — "
              f"partial trace preserved; NOT relaunching.")
    return metrics


def _extract(qid, q, d, calls, wall, ui, uo, ch, cost, lv2) -> dict:
    eng = lv2.get("engine_primary") or {}
    dec = eng.get("decisions") or {}
    bud = lv2.get("budget") or {}
    fd = lv2.get("forecast_decomposition") or {}
    return {
        "qid": qid, "name": NAMES[qid], "question": q["question"][:140],
        "as_of": str(q["present_date"])[:10], "horizon": str(q["expected_resolution_date"])[:10],
        "profile": "lean_v2", "model": MODEL, "seed": SEED,
        "status": d.get("simulation_status"), "p_raw": d.get("raw_probability"),
        "p_cal": d.get("calibrated_probability"),
        "probability_source": d.get("probability_source"),
        "grounding_grade": d.get("grounding_grade"), "confidence": d.get("confidence"),
        "unresolved_mass": d.get("unresolved_mass"),
        "uncertainty_interval": d.get("uncertainty_interval"),
        "weight_sensitive": d.get("weight_sensitive"),
        "grounded_prior": fd.get("grounded_prior", {}).get("p"),
        "grounded_prior_n": fd.get("grounded_prior", {}).get("n"),
        "simulation_conditional": fd.get("simulation_conditional", {}).get("p"),
        "combined": fd.get("combined"), "combine_method": fd.get("method"),
        "combiner_available": fd.get("combiner_available"),
        "unresolved_by_cause": (lv2.get("unresolved") or {}).get("by_cause"),
        "shared_conditions": [c["combo"] for c in (lv2.get("shared_condition_worlds") or [])][:6],
        "n_actor_states": {a: len(s) for a, s in (lv2.get("actor_states") or {}).items()},
        "unknown_state_mass": lv2.get("unknown_state_mass"),
        "dependence_sensitive": eng.get("dependence_sensitive"),
        "dependence_range": eng.get("dependence_range"),
        "challenger": lv2.get("challenger"),
        "actors_pruned": (lv2.get("slice") or {}).get("n_pruned"),
        "aliases_merged": (lv2.get("slice") or {}).get("n_merged"),
        "n_llm_calls": len(calls), "calls_by_stage": bud.get("by_stage"),
        "actor_calls": ((bud.get("by_stage") or {}).get("actor_decision") or {}).get("calls"),
        "unique_decision_contexts": dec.get("unique_decision_contexts"),
        "decision_reuses": dec.get("reuses"),
        "weighted_nodes_executed": (eng.get("coalescer") or {}).get("executed_unique_nodes"),
        "branches_merged": (eng.get("coalescer") or {}).get("merges"),
        "deliberations": len(eng.get("deliberations") or []),
        "input_tokens": ui, "output_tokens": uo, "provider_cache_hit_tokens": ch,
        "cost_usd": cost, "wall_clock_s": round(wall, 1),
        "preflight_verdict": (lv2.get("preflight") or {}).get("verdict"),
        "trace_dir": lv2.get("trace_dir"),
        "guard": {"wall_cap_s": GUARD_WALL_S, "call_cap": GUARD_CALLS,
                  "passed_hard_stop": wall <= GUARD_WALL_S and len(calls) <= GUARD_CALLS,
                  "hit_consumer_target": wall <= 300 and 10 <= len(calls) <= 35}}


def _stored(qid: str) -> dict:
    """Stored FF / Lean V1 comparators — read from committed artifacts, AFTER freeze only."""
    out = {}
    try:
        comp = json.loads(Path("experiments/results/exp109_comparison.json").read_text())
        row = next(r for r in comp["per_question"] if r["qid"] == qid)
        out["ff_p"] = (row.get("full_fidelity") or {}).get("prediction")
        out["l1_p"] = (row.get("lean_adaptive") or {}).get("prediction")
        out["l1_wall_s"] = (row.get("lean_adaptive") or {}).get("wall_clock_s")
        out["l1_calls"] = (row.get("lean_adaptive") or {}).get("llm_calls")
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:120]
    # previous Lean V2 (exp111 Banxico) when applicable
    if qid == ORDER[0] and Path("experiments/results/exp111_lean_v2_banxico.json").exists():
        try:
            out["prev_lean_v2_p"] = json.loads(Path(
                "experiments/results/exp111_lean_v2_banxico.json").read_text()
            )["metrics"].get("p_raw")
        except Exception:  # noqa: BLE001
            pass
    return out


def _brier(p, o):
    return None if p is None else round((float(p) - o) ** 2, 4)


def _compare_after_freeze(qid, metrics, rows):
    outcome = int(rows[qid]["resolution"])
    st = _stored(qid)
    p = metrics["p_raw"]
    print(f"\n=== {NAMES[qid]} comparison (joined AFTER freeze) ===")
    print(f"  outcome: {outcome}")
    print(f"  full-fidelity: {st.get('ff_p')}  (Brier {_brier(st.get('ff_p'), outcome)})")
    print(f"  Lean V1:       {st.get('l1_p')}  (Brier {_brier(st.get('l1_p'), outcome)}) "
          f"[{st.get('l1_calls')} calls / {st.get('l1_wall_s')}s]")
    if st.get("prev_lean_v2_p") is not None:
        print(f"  prev Lean V2:  {st.get('prev_lean_v2_p')}  "
              f"(Brier {_brier(st.get('prev_lean_v2_p'), outcome)})")
    print(f"  Lean V2 (new): {p}  (Brier {_brier(p, outcome)})  "
          f"[{metrics['n_llm_calls']} calls / {metrics['wall_clock_s']}s]")
    metrics["evaluation"] = {
        "outcome": outcome, "ff_p": st.get("ff_p"), "l1_p": st.get("l1_p"),
        "prev_lean_v2_p": st.get("prev_lean_v2_p"), "lean_v2_p": p,
        "brier": {"full_fidelity": _brier(st.get("ff_p"), outcome),
                  "lean_v1": _brier(st.get("l1_p"), outcome),
                  "lean_v2": _brier(p, outcome)},
        "lean_v2_closer_than_v1": (_brier(p, outcome) is not None
                                   and _brier(st.get("l1_p"), outcome) is not None
                                   and _brier(p, outcome) < _brier(st.get("l1_p"), outcome)),
        "correct_side": None if p is None else bool((p > 0.5) == bool(outcome)),
        "l1_wall_s": st.get("l1_wall_s"), "l1_calls": st.get("l1_calls")}


def _print_human_report(qid, m, lv2, d):
    print(f"\n{'=' * 70}\n{NAMES[qid]} — Lean V2 accuracy run\n{'=' * 70}")
    print(f"status={m['status']}  p={m['p_raw']}  source={m['probability_source']}  "
          f"grounding={m['grounding_grade']}  confidence={m['confidence']}")
    print(f"grounded prior {m['grounded_prior']} (n={m['grounded_prior_n']}) | "
          f"simulation-conditional {m['simulation_conditional']} | "
          f"combined {m['combined']} via {m['combine_method']}")
    print(f"unresolved by cause: {m['unresolved_by_cause']}")
    print(f"shared conditions: {m['shared_conditions']}")
    print(f"actor states: {m['n_actor_states']} | unknown-state mass: {m['unknown_state_mass']}")
    print(f"dependence_sensitive={m['dependence_sensitive']} range={m['dependence_range']} | "
          f"weight_sensitive={m['weight_sensitive']}")
    print(f"challenger: {(m['challenger'] or {}).get('triggered')} | "
          f"pruned={m['actors_pruned']} merged={m['aliases_merged']}")
    print(f"calls={m['n_llm_calls']} (actor {m['actor_calls']}, {m['unique_decision_contexts']} "
          f"unique ctx, {m['decision_reuses']} reuses) | nodes={m['weighted_nodes_executed']} | "
          f"cost=${m['cost_usd']} | wall={m['wall_clock_s']}s")
    for lim in (d.get("limitations") or [])[:8]:
        print(f"  - {lim}")


def merge_and_score() -> dict:
    rows = {r["question_id"]: r for r in fetch_btf3()}
    results = []
    for qid in ORDER:
        cp = CKPT / f"{qid}.json"
        if cp.exists():
            results.append(json.loads(cp.read_text())["metrics"])
    scored = [r for r in results if r.get("evaluation", {}).get("brier", {}).get("lean_v2")
              is not None]
    def _mean(key, arm):
        vals = [r["evaluation"]["brier"][arm] for r in results
                if r.get("evaluation", {}).get("brier", {}).get(arm) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    summary = {
        "experiment": "EXP-112 Lean V2 accuracy-architecture five-question evaluation",
        "n": len(results), "n_scored": len(scored),
        "mean_brier": {"lean_v2": _mean("brier", "lean_v2"),
                       "lean_v1": _mean("brier", "lean_v1"),
                       "full_fidelity": _mean("brier", "full_fidelity")},
        "lean_v2_total_calls": sum(r["n_llm_calls"] for r in results),
        "lean_v2_avg_calls": round(sum(r["n_llm_calls"] for r in results)
                                   / max(1, len(results)), 1),
        "lean_v2_total_wall_s": round(sum(r["wall_clock_s"] for r in results), 1),
        "lean_v2_avg_wall_s": round(sum(r["wall_clock_s"] for r in results)
                                    / max(1, len(results)), 1),
        "lean_v2_total_cost": round(sum(r["cost_usd"] for r in results), 4),
        "per_question": [{k: r.get(k) for k in
                          ("name", "status", "p_raw", "grounded_prior", "simulation_conditional",
                           "combined", "combiner_available", "n_llm_calls", "wall_clock_s",
                           "evaluation")} for r in results]}
    SUMMARY.write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, indent=1))
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_worker(int(sys.argv[1]))
    else:
        merge_and_score()
