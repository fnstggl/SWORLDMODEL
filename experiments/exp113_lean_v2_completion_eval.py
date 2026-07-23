"""EXP-113: the Lean V2 SIMULATION-COMPLETION five-question evaluation.

The same 5 frozen BTF-3 questions as EXP-112 (Banxico → BoJ → visionOS → Wale → Hormuz),
sequentially, through the canonical `unified_runtime.simulate_world(..., "lean_v2")` — now
carrying the simulation-completion architecture: actor-state completeness invariant +
recovery ladder, readiness gate with terminal round-trip proof, missing-mechanism recovery,
bounded residuals instead of unknown-state worlds, deadline-forced completion, and the
SimulationCompletionAudit acceptance targets.

WHAT THIS MEASURES (§completion): how much world mass the simulation now RESOLVES to the
exact terminal outcome per question, the simulation-only conditional probability, and the
required forecast-decomposition table — prior_forecast vs simulation_forecast vs
headline_forecast, all labeled. No combiner calibration, no BTF-3 outcome training.

LEAKAGE: the actual outcome and stored Lean V1 / full-fidelity predictions are NEVER passed
to any research/compilation/actor/weighting/simulation stage. They are read from committed
artifacts ONLY after this arm's forecast is frozen and checkpointed.

GUARD (per question): target 1-5 min / 10-50 calls; HARD stop 12 min / 100 calls, enforced
through the production ConsumerComputeBudget — a trip finalizes the best labeled forecast,
never a relaunch. One process, foreground, sequential.

Run one worker at a time, then merge:
    python -m experiments.exp113_lean_v2_completion_eval <i>   # i in 0..4, Banxico=0
    python -m experiments.exp113_lean_v2_completion_eval        # merge + table + report
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
from experiments.exp107_btf3_full_fidelity_post127 import (MODEL, MAX_TOKENS, TEMPERATURE,
                                                           SEED)

ORDER = ["cfb43147-d9d2-5bd9-903f-f449e9a5aecf",   # 0 Banxico
         "7279494c-a775-5a57-a5f2-ac22252fb286",   # 1 BoJ
         "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6",   # 2 visionOS
         "741b4bed-7502-5cd2-9cbe-949fbc70f857",   # 3 Wale
         "017e64ef-7354-56c4-8a4d-e27121bc639a"]   # 4 Hormuz
NAMES = {ORDER[0]: "Banxico", ORDER[1]: "BoJ", ORDER[2]: "visionOS", ORDER[3]: "Wale",
         ORDER[4]: "Hormuz"}

CKPT = Path("experiments/results/exp113_checkpoints")
SUMMARY = Path("experiments/results/exp113_completion_eval.json")
REPORT = Path("experiments/results/exp113_completion_eval.md")
GUARD_WALL_S, GUARD_CALLS = 720.0, 100    # per-question HARD maximum: 12 min / 100 calls
TARGET_WALL_S, TARGET_CALLS = 300.0, 50   # consumer target: 1-5 min / 10-50 calls
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
                                      resolution_criteria=q["resolution_criteria"],
                                      seed=SEED)
    calls, pending = [], []
    base = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE, usage_sink=pending.append)

    def llm(prompt, _c=calls, _u=pending):
        t = time.time()
        reply = base(prompt)
        _c.append({"i": len(_c), "prompt_chars": len(prompt),
                   "reply_chars": len(reply or ""),
                   "latency_s": round(time.time() - t, 3),
                   "usage": (_u.pop() if _u else None)})
        return reply

    t0 = time.time()
    res = simulate_world(
        q["question"], llm=llm, evidence=evidence, as_of=str(q["present_date"])[:10],
        horizon=str(q["expected_resolution_date"])[:10], seed=SEED, prebuilt_bundle=bundle,
        execution_policy={"lean_v2": {
            "budget": {"max_wall_s": GUARD_WALL_S, "max_calls": GUARD_CALLS},
            "backend_fingerprint": MODEL, "persistent_cache": True,
            "persistent_cache_dir": f"experiments/results/exp113_cache/{NAMES[qid].lower()}",
            "qid": f"{qid}-completion", "max_workers": 6}},
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
    _print_under_the_hood(qid, metrics, lv2, d)
    _compare_after_freeze(qid, metrics, rows)
    cpath.write_text(json.dumps({"metrics": metrics, "simulation_result": d,
                                 "n_calls": len(calls)}, default=str))
    guard = metrics["guard"]
    if not guard["passed_hard_stop"]:
        print(f"\n{NAMES[qid]}: FAILED the 12min/100-call hard stop ({wall:.0f}s / "
              f"{len(calls)} calls) — partial trace preserved; NOT relaunching.")
    return metrics


def _extract(qid, q, d, calls, wall, ui, uo, ch, cost, lv2) -> dict:
    eng = lv2.get("engine_primary") or {}
    dec = eng.get("decisions") or {}
    bud = lv2.get("budget") or {}
    fd = lv2.get("forecast_decomposition") or {}
    ca = lv2.get("completion_audit") or {}
    rec = lv2.get("state_recovery") or {}
    mech = lv2.get("mechanism_recovery") or {}
    ready = lv2.get("readiness") or {}
    return {
        "qid": qid, "name": NAMES[qid], "question": q["question"][:140],
        "as_of": str(q["present_date"])[:10],
        "horizon": str(q["expected_resolution_date"])[:10],
        "profile": "lean_v2", "model": MODEL, "seed": SEED,
        "status": d.get("simulation_status"),
        # §separation: the three labeled forecasts
        "prior_forecast": fd.get("prior_forecast"),
        "grounded_prior_n": (fd.get("grounded_prior") or {}).get("n"),
        "simulation_forecast": fd.get("simulation_forecast"),
        "headline_forecast": fd.get("headline_forecast", d.get("raw_probability")),
        "headline_source": fd.get("headline_source", d.get("probability_source")),
        "p_raw": d.get("raw_probability"),
        "probability_source": d.get("probability_source"),
        "resolved_simulation_mass": fd.get("resolved_simulation_mass"),
        "unresolved_mass_by_cause": fd.get("unresolved_mass_by_cause"),
        "simulation_probability_bounds": fd.get("simulation_probability_bounds"),
        "residual_bound": fd.get("residual_bound"),
        "uncertainty_interval": d.get("uncertainty_interval"),
        "weight_sensitive": d.get("weight_sensitive"),
        "grounding_grade": d.get("grounding_grade"), "confidence": d.get("confidence"),
        # completion machinery
        "completion_acceptance": ca.get("acceptance"),
        "completion_rounds": (ca.get("engine_completion_rounds") or {}).get("rounds"),
        "readiness_verdict": ready.get("verdict"),
        "round_trip_ok": (ready.get("round_trip") or {}).get("ok"),
        "state_recovery": {a: {"initial": r.get("initial_state_count"),
                               "final": r.get("final_state_count"),
                               "source": r.get("final_source"),
                               "residual": r.get("residual_r"),
                               "attempts": len(r.get("attempts") or [])}
                           for a, r in (rec.get("actors") or {}).items()},
        "reversal_search": rec.get("reversal_search"),
        "mechanism_attempted": bool(mech.get("attempts")),
        "mechanism_validated": mech.get("validated"),
        "mechanism_failure_proof": (mech.get("failure_proof") or "")[:200],
        # world structure
        "n_actor_states": {a: len(s) for a, s in (lv2.get("actor_states") or {}).items()},
        "actor_residual_bounds": eng.get("actor_residual_bounds"),
        "joint_residual_bound": eng.get("joint_residual_bound"),
        "shared_conditions": [c["combo"] for c in
                              (lv2.get("shared_condition_worlds") or [])][:6],
        "waves": eng.get("waves"),
        "weighted_nodes_executed": (eng.get("coalescer") or {}).get(
            "executed_unique_nodes"),
        "branches_merged": (eng.get("coalescer") or {}).get("merges"),
        "decision_trace_sample": (eng.get("decision_trace") or [])[:60],
        "unique_decision_contexts": dec.get("unique_decision_contexts"),
        "decision_reuses": dec.get("reuses"),
        "dependence_sensitive": eng.get("dependence_sensitive"),
        "challenger_triggered": (lv2.get("challenger") or {}).get("triggered"),
        "preflight_verdict": (lv2.get("preflight") or {}).get("verdict"),
        # cost
        "n_llm_calls": len(calls),
        "actor_calls": ((bud.get("by_stage") or {}).get("actor_decision") or {}).get(
            "calls"),
        "calls_by_stage": {k: v.get("calls") for k, v in
                           (bud.get("by_stage") or {}).items()},
        "input_tokens": ui, "output_tokens": uo, "provider_cache_hit_tokens": ch,
        "cost_usd": cost, "wall_clock_s": round(wall, 1),
        "trace_dir": lv2.get("trace_dir"),
        "guard": {"wall_cap_s": GUARD_WALL_S, "call_cap": GUARD_CALLS,
                  "passed_hard_stop": wall <= GUARD_WALL_S and len(calls) <= GUARD_CALLS,
                  "hit_consumer_target": wall <= TARGET_WALL_S
                  and len(calls) <= TARGET_CALLS}}


def _stored(qid: str) -> dict:
    """Stored comparators — read from committed artifacts, AFTER freeze only."""
    out = {}
    try:
        comp = json.loads(Path("experiments/results/exp109_comparison.json").read_text())
        row = next(r for r in comp["per_question"] if r["qid"] == qid)
        out["ff_p"] = (row.get("full_fidelity") or {}).get("prediction")
        out["l1_p"] = (row.get("lean_adaptive") or {}).get("prediction")
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:120]
    try:
        prev = json.loads((Path("experiments/results/exp112_checkpoints")
                           / f"{qid}.json").read_text())["metrics"]
        out["exp112_p"] = prev.get("p_raw")
        out["exp112_prior"] = prev.get("grounded_prior")
    except Exception:  # noqa: BLE001
        pass
    return out


def _brier(p, o):
    return None if p is None else round((float(p) - o) ** 2, 4)


def _compare_after_freeze(qid, metrics, rows):
    outcome = int(rows[qid]["resolution"])
    st = _stored(qid)
    p = metrics["headline_forecast"]
    print(f"\n=== {NAMES[qid]} comparison (joined AFTER freeze) ===")
    print(f"  outcome: {outcome}")
    print(f"  full-fidelity: {st.get('ff_p')}  (Brier {_brier(st.get('ff_p'), outcome)})")
    print(f"  Lean V1:       {st.get('l1_p')}  (Brier {_brier(st.get('l1_p'), outcome)})")
    print(f"  exp112 LeanV2: {st.get('exp112_p')}  "
          f"(Brier {_brier(st.get('exp112_p'), outcome)})")
    print(f"  exp113 (new):  {p}  (Brier {_brier(p, outcome)})  "
          f"[{metrics['n_llm_calls']} calls / {metrics['wall_clock_s']}s]")
    metrics["evaluation"] = {
        "outcome": outcome, "ff_p": st.get("ff_p"), "l1_p": st.get("l1_p"),
        "exp112_p": st.get("exp112_p"), "exp113_p": p,
        "brier": {"full_fidelity": _brier(st.get("ff_p"), outcome),
                  "lean_v1": _brier(st.get("l1_p"), outcome),
                  "exp112_lean_v2": _brier(st.get("exp112_p"), outcome),
                  "exp113_lean_v2": _brier(p, outcome)},
        "sim_forecast_brier": _brier(metrics.get("simulation_forecast"), outcome),
        "prior_brier": _brier(metrics.get("prior_forecast"), outcome),
        "sim_moved_toward_outcome": (
            None if metrics.get("simulation_forecast") is None
            or metrics.get("prior_forecast") is None else
            bool(abs(float(metrics["simulation_forecast"]) - outcome)
                 < abs(float(metrics["prior_forecast"]) - outcome))),
        "correct_side": None if p is None else bool((p > 0.5) == bool(outcome))}


def _print_under_the_hood(qid, m, lv2, d):
    print(f"\n{'=' * 74}\n{NAMES[qid]} — Lean V2 SIMULATION-COMPLETION run (under the hood)"
          f"\n{'=' * 74}")
    print(f"status={m['status']}  readiness={m['readiness_verdict']} "
          f"(round-trip ok={m['round_trip_ok']})  preflight={m['preflight_verdict']}")
    print(f"PRIOR_FORECAST={m['prior_forecast']} (n={m['grounded_prior_n']})  |  "
          f"SIMULATION_FORECAST={m['simulation_forecast']}  |  "
          f"HEADLINE={m['headline_forecast']} via {m['headline_source']}")
    print(f"resolved simulation mass: {m['resolved_simulation_mass']}  |  "
          f"unresolved by cause: {m['unresolved_mass_by_cause']}")
    print(f"simulation bounds: {m['simulation_probability_bounds']} "
          f"(residual bound {m['residual_bound']})")
    print(f"completion acceptance: {m['completion_acceptance']}")
    print("state recovery (per consequential actor):")
    for a, r in (m["state_recovery"] or {}).items():
        print(f"  - {a}: {r['initial']}→{r['final']} states (source={r['source']}, "
              f"residual={r['residual']}, ladder attempts={r['attempts']})")
    print(f"reversal search: {m['reversal_search']}")
    if m["mechanism_attempted"]:
        print(f"mechanism: validated={m['mechanism_validated']}"
              + (f"  failure_proof={m['mechanism_failure_proof']}"
                 if m['mechanism_failure_proof'] else ""))
    print(f"shared-condition worlds: {m['shared_conditions']}")
    print(f"actor states: {m['n_actor_states']}  residuals: {m['actor_residual_bounds']} "
          f"(joint {m['joint_residual_bound']})")
    print(f"waves={m['waves']}  nodes={m['weighted_nodes_executed']} "
          f"(merged {m['branches_merged']})  completion rounds="
          f"{len(m['completion_rounds'] or [])}")
    print("decisions simulated (first 20):")
    for t in (m["decision_trace_sample"] or [])[:20]:
        print(f"  - day {t.get('day')} {t.get('actor')} [{t.get('variant')}] "
              f"{t.get('act_or_wait')}: {t.get('chosen') or t.get('vote_option') or '—'}"
              + (f" vote={t.get('vote_option')}" if t.get('vote_option') else ""))
    print(f"calls={m['n_llm_calls']} (actor {m['actor_calls']}; by stage "
          f"{m['calls_by_stage']})  wall={m['wall_clock_s']}s  cost=${m['cost_usd']}")
    for lim in (d.get("limitations") or [])[:8]:
        print(f"  - {lim}")


def _table(results) -> str:
    """§the required forecast-decomposition table."""
    L = ["| Question | Prior (n) | Sim-conditional | Resolved mass | Headline | Source | "
         "Final−Prior | Outcome | Brier (headline) | Brier (sim) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        ev = r.get("evaluation") or {}
        pf, sf, hf = r.get("prior_forecast"), r.get("simulation_forecast"), \
            r.get("headline_forecast")
        delta = (None if hf is None or pf is None
                 else round(float(hf) - float(pf), 4))
        L.append(
            f"| {r['name']} | {pf} ({r.get('grounded_prior_n')}) | {sf} | "
            f"{r.get('resolved_simulation_mass')} | {hf} | {r.get('headline_source')} | "
            f"{delta} | {ev.get('outcome')} | "
            f"{(ev.get('brier') or {}).get('exp113_lean_v2')} | "
            f"{ev.get('sim_forecast_brier')} |")
    return "\n".join(L)


def merge_and_score() -> dict:
    results = []
    for qid in ORDER:
        cp = CKPT / f"{qid}.json"
        if cp.exists():
            results.append(json.loads(cp.read_text())["metrics"])

    def _mean(arm):
        vals = [(r.get("evaluation") or {}).get("brier", {}).get(arm) for r in results]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    sim_briers = [(r.get("evaluation") or {}).get("sim_forecast_brier") for r in results]
    sim_briers = [v for v in sim_briers if v is not None]
    moved = [bool((r.get("evaluation") or {}).get("sim_moved_toward_outcome"))
             for r in results
             if (r.get("evaluation") or {}).get("sim_moved_toward_outcome") is not None]
    acc = [r.get("completion_acceptance") or {} for r in results]
    summary = {
        "experiment": "EXP-113 Lean V2 simulation-completion five-question evaluation",
        "n": len(results),
        "mean_brier": {"exp113_lean_v2": _mean("exp113_lean_v2"),
                       "exp112_lean_v2": _mean("exp112_lean_v2"),
                       "lean_v1": _mean("lean_v1"),
                       "full_fidelity": _mean("full_fidelity")},
        "mean_sim_forecast_brier": (round(sum(sim_briers) / len(sim_briers), 4)
                                    if sim_briers else None),
        "sim_moved_toward_outcome_count": f"{sum(moved)}/{len(moved)}",
        "mean_resolved_mass": round(sum(float(r.get("resolved_simulation_mass") or 0)
                                        for r in results) / max(1, len(results)), 4),
        "acceptance_all_ok_count":
            f"{sum(1 for a in acc if a.get('all_ok'))}/{len(acc)}",
        "resolved_target_met_count":
            f"{sum(1 for a in acc if a.get('resolved_target_met'))}/{len(acc)}",
        "unknown_state_zero_count":
            f"{sum(1 for a in acc if a.get('terminal_unknown_state_ok'))}/{len(acc)}",
        "total_calls": sum(r["n_llm_calls"] for r in results),
        "total_wall_s": round(sum(r["wall_clock_s"] for r in results), 1),
        "total_cost": round(sum(r["cost_usd"] for r in results), 4),
        "guard_passed_all": all(r["guard"]["passed_hard_stop"] for r in results),
        "per_question": [{k: r.get(k) for k in
                          ("name", "status", "prior_forecast", "simulation_forecast",
                           "headline_forecast", "headline_source",
                           "resolved_simulation_mass", "unresolved_mass_by_cause",
                           "simulation_probability_bounds", "completion_acceptance",
                           "readiness_verdict", "n_llm_calls", "wall_clock_s",
                           "evaluation")} for r in results]}
    SUMMARY.write_text(json.dumps(summary, indent=1, default=str))
    REPORT.write_text(_render_report(results, summary))
    print(_table(results))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, indent=1))
    return summary


def _render_report(results, summary) -> str:
    L = ["# EXP-113 — Lean V2 simulation-completion evaluation\n",
         "The same 5 frozen BTF-3 questions as EXP-112, rerun through the canonical "
         "lean_v2 profile after the simulation-completion fix. Prior and simulation are "
         "fully separate; the headline is the mass-based recovery blend; no combiner was "
         "calibrated and no BTF-3 outcome was trained on.\n",
         "## Forecast decomposition (the required table)\n", _table(results), "",
         f"- mean Brier (headline): {summary['mean_brier']['exp113_lean_v2']} "
         f"(exp112: {summary['mean_brier']['exp112_lean_v2']}, "
         f"Lean V1: {summary['mean_brier']['lean_v1']}, "
         f"full-fidelity: {summary['mean_brier']['full_fidelity']})",
         f"- mean Brier (simulation-only forecast): "
         f"{summary['mean_sim_forecast_brier']}",
         f"- simulation moved the forecast toward the outcome (vs prior): "
         f"{summary['sim_moved_toward_outcome_count']}",
         f"- mean resolved simulation mass: {summary['mean_resolved_mass']}",
         f"- completion acceptance all-ok: {summary['acceptance_all_ok_count']} | "
         f"resolved≥80% target met: {summary['resolved_target_met_count']} | "
         f"unknown-state mass zero: {summary['unknown_state_zero_count']}",
         f"- totals: {summary['total_calls']} calls, {summary['total_wall_s']}s, "
         f"${summary['total_cost']} | 12min/100-call guard passed on every question: "
         f"{summary['guard_passed_all']}\n",
         "## Per-question: exactly what simulated under the hood\n"]
    for r in results:
        ev = r.get("evaluation") or {}
        L.append(f"### {r['name']} — headline {r.get('headline_forecast')} "
                 f"({r.get('headline_source')}), outcome {ev.get('outcome')}\n")
        L.append(f"- readiness: **{r.get('readiness_verdict')}** (terminal round-trip "
                 f"ok={r.get('round_trip_ok')}); status **{r.get('status')}**")
        L.append(f"- prior_forecast {r.get('prior_forecast')} "
                 f"(counted n={r.get('grounded_prior_n')}) | simulation_forecast "
                 f"{r.get('simulation_forecast')} on resolved mass "
                 f"{r.get('resolved_simulation_mass')} | bounds "
                 f"{r.get('simulation_probability_bounds')} (residual "
                 f"{r.get('residual_bound')})")
        L.append(f"- unresolved by cause: {r.get('unresolved_mass_by_cause')}")
        rec = r.get("state_recovery") or {}
        if rec:
            L.append("- actor-state completeness: "
                     + "; ".join(f"{a}: {v['initial']}→{v['final']} ({v['source']}, "
                                 f"r={v['residual']})" for a, v in rec.items()))
        rs = r.get("reversal_search") or {}
        L.append(f"- reversal-state search: ran={rs.get('ran')}, "
                 f"added={rs.get('added', 0)}")
        if r.get("mechanism_attempted"):
            L.append(f"- mechanism recovery: validated={r.get('mechanism_validated')}"
                     + (f"; failure proof: {r.get('mechanism_failure_proof')}"
                        if r.get("mechanism_failure_proof") else ""))
        L.append(f"- world: shared conditions {r.get('shared_conditions')}; actor states "
                 f"{r.get('n_actor_states')}; waves {r.get('waves')}; weighted nodes "
                 f"{r.get('weighted_nodes_executed')} (merged {r.get('branches_merged')}); "
                 f"completion rounds {len(r.get('completion_rounds') or [])}")
        L.append(f"- decisions: {r.get('unique_decision_contexts')} unique contexts, "
                 f"{r.get('decision_reuses')} reuses, actor calls {r.get('actor_calls')}")
        trace = r.get("decision_trace_sample") or []
        if trace:
            L.append("- first simulated decisions:")
            for t in trace[:12]:
                L.append(f"    - day {t.get('day')} — {t.get('actor')} "
                         f"[{t.get('variant')}] {t.get('act_or_wait')}: "
                         f"{t.get('chosen') or '—'}"
                         + (f" (vote {t.get('vote_option')})"
                            if t.get('vote_option') else ""))
        L.append(f"- acceptance: {r.get('completion_acceptance')}")
        L.append(f"- Brier: headline "
                 f"{(ev.get('brier') or {}).get('exp113_lean_v2')} | simulation-only "
                 f"{ev.get('sim_forecast_brier')} | prior {ev.get('prior_brier')} | "
                 f"sim moved toward outcome: {ev.get('sim_moved_toward_outcome')}")
        L.append(f"- cost: {r.get('n_llm_calls')} calls, {r.get('wall_clock_s')}s, "
                 f"${r.get('cost_usd')}\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_worker(int(sys.argv[1]))
    else:
        merge_and_score()
