"""EXP-110: the frozen five-question FULL-ACTOR pass (post p=None fix), single run, no mean-of-K.

The mandated proof-of-life + architecture audit on the five frozen questions (BoJ, visionOS, Wale,
Hormuz, Banxico) via the canonical unified_runtime.simulate_world with FULL LLM actors. This is NOT an
accuracy claim (n=5, single run) — it verifies the rich architecture now runs end-to-end and never
discards a grounded forecast, and profiles where the cost goes.

Per question it answers the 8 audit questions (proof-of-life, not a score):
  A1 grounded outside-view prior — value / reference class / stage / status-quo (standalone estimator probe)
  A2 evidence — did as-of evidence reach the posterior (n_effective_observations), or evidence-starved?
  A3 structure execution — the operator census: did named actors/institutions write terminal state, or
     did it fall to a broad prior / no bound outcome? (the EXP-105 "nobody voted" check)
  A4 simulation_status — completed / degraded / under_modeled / partially_resolved / unresolved + why
  A5 grounded_outside_view_fallback — did the fix serve the grounded prior because the rollout was
     under-modeled? (never a silent None)
  A6 structural ensemble — models promoted, material disagreement (headline suppressed)?
  A7 forecast vs baselines — model p vs FutureSearch SOTA vs actual outcome (scoring join, footer only)
  A8 cost — total calls + wall time + calls_by_stage: accidental loop or normal architecture cost?

Per-question checkpoint (resumable). Leakage-quarantined: only allowlisted as-of fields reach the
forecaster; the resolution/SOTA join happens only in the scoring footer.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp110_frozen5_full_actor [only_index]
"""
from __future__ import annotations
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input, _ts

# the frozen five (EXP-102's five; committed in exp105_btf3_simulate_world.EXP102_QIDS)
FROZEN5 = [
    ("BoJ",     "7279494c-a775-5a57-a5f2-ac22252fb286"),
    ("visionOS","5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6"),
    ("Wale",    "741b4bed-7502-5cd2-9cbe-949fbc70f857"),
    ("Hormuz",  "017e64ef-7354-56c4-8a4d-e27121bc639a"),
    ("Banxico", "cfb43147-d9d2-5bd9-903f-f449e9a5aecf"),
]
CKPT = Path("experiments/results/exp110_checkpoints")
OUT = Path("experiments/results/exp110_frozen5_full_actor.json")


class TimingLLM:
    def __init__(self, llm):
        self._llm, self.durations = llm, []

    def __call__(self, prompt, *a, **k):
        t = time.time()
        try:
            return self._llm(prompt, *a, **k)
        finally:
            self.durations.append(time.time() - t)

    def profile(self):
        d = self.durations
        return {"true_backend_calls": len(d), "llm_wall_s_total": round(sum(d), 1),
                "call_s_mean": round(statistics.mean(d), 2) if d else None,
                "call_s_median": round(statistics.median(d), 2) if d else None,
                "call_s_max": round(max(d), 2) if d else None}


def _prior_probe(question, as_of, horizon_days, llm):
    """A1: the standalone deadline-aware grounded prior for this question (independent of the rollout)."""
    try:
        from swm.world_model_v2.phase3_priors import estimate_reference_base_rate
        est = estimate_reference_base_rate(question, llm=llm, as_of=as_of, horizon_days=horizon_days)
        return {k: est.get(k) for k in ("reference_class", "stage", "base_rate", "status_quo",
                                        "is_recurrence", "transport_risk", "evidence_quality")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:120]}


def _census(d):
    """A3: operator census merged across the promoted models (what actually wrote terminal state)."""
    prov = d.get("provenance") or {}
    out = {}
    for m, pv in (prov.get("per_model_provenance") or {}).items():
        for op, rec in ((pv or {}).get("operator_delta_census") or {}).items():
            out[op] = out.get(op, 0) + int((rec or {}).get("n_deltas", 0) or 0)
    # single-model / general path census
    for op, rec in (prov.get("operator_delta_census") or {}).items():
        out[op] = out.get(op, 0) + int((rec or {}).get("n_deltas", 0) or 0)
    return out


def _one(label, qid, rows, llm_factory):
    cp = CKPT / f"{qid}.json"
    if cp.exists():
        return json.loads(cp.read_text())
    from swm.world_model_v2.unified_runtime import simulate_world
    row = rows[qid]
    q = _forecast_input(row)
    as_of, resolve = _ts(q["present_date"]), _ts(q["expected_resolution_date"])
    if resolve <= as_of:
        resolve = as_of + 30 * 86400
    hz = round((resolve - as_of) / 86400, 1)
    llm = TimingLLM(llm_factory())
    print(f"  [{label}] start  hz={hz}d  {q['question'][:70]}")
    t0 = time.time()
    try:
        prior = _prior_probe(q["question"], str(q["present_date"])[:10], hz, llm)     # A1
        res = simulate_world(q["question"], as_of=str(q["present_date"])[:10],
                             horizon=str(q["expected_resolution_date"])[:10], llm=llm, seed=0)
        wall = round(time.time() - t0, 1)
        d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
        prov = d.get("provenance") or {}
        ens = d.get("structural_ensemble") or {}
        cost = ens.get("cost_manifest") or {}
        p = d.get("calibrated_probability")
        if p is None:
            p = d.get("raw_probability")
        rec = {
            "label": label, "qid": qid, "question": q["question"][:110], "horizon_days": hz,
            "A1_grounded_prior": prior,
            "A2_evidence_sufficiency": prov.get("evidence_sufficiency"),
            "A2_posterior_consumed": prov.get("posterior_consumed"),
            "A3_operator_census": _census(d),
            "A4_status": d.get("simulation_status"),
            "A4_has_forecast": bool(res.has_forecast()),
            "A4_under_modeled_subtypes": d.get("under_modeled_subtypes"),
            "A5_grounded_outside_view_fallback": prov.get("grounded_outside_view_fallback"),
            "A6_n_models": len(ens.get("model_distributions") or {}),
            "A6_aggregation": ens.get("aggregation_method"),
            "A6_structural_sensitivity": (ens.get("structural_sensitivity") or {}).get("classification"),
            # WSim: the three transparent numbers + the surviving-world audit (Step 1 + Step 6/7)
            "W_outside_view": (ens.get("simulation_forecast_report") or {}).get("outside_view_forecast"),
            "W_simulation_derived": (ens.get("simulation_forecast_report") or {}).get("simulation_derived_forecast"),
            "W_final_combined": (ens.get("simulation_forecast_report") or {}).get("final_combined_forecast"),
            "W_final_source": ((ens.get("simulation_forecast_report") or {}).get("final_selected") or {}).get("source"),
            "W_final_reason": ((ens.get("simulation_forecast_report") or {}).get("final_selected") or {}).get("reason"),
            "W_confidence_alpha": ((ens.get("simulation_forecast_report") or {}).get("simulation_confidence") or {}).get("alpha"),
            "W_n_worlds_valid": (ens.get("simulation_forecast_report") or {}).get("n_worlds_valid"),
            "W_world_weights": ((ens.get("simulation_forecast_report") or {}).get("weighted_simulation_forecast") or {}).get("world_weights"),
            "W_spread": ((ens.get("simulation_forecast_report") or {}).get("weighted_simulation_forecast") or {}).get("spread"),
            "W_rejected_worlds": (ens.get("simulation_forecast_report") or {}).get("rejected_worlds"),
            "W_merge_records": (ens.get("simulation_forecast_report") or {}).get("merge_records"),
            "W_interpretation_hypotheses": d.get("interpretation_hypotheses"),
            "p": p, "raw_distribution": d.get("raw_distribution"),
            "A8_wall_s": wall, "A8_timing_wrapper": llm.profile(),
            "A8_ledger_total_calls": cost.get("total_llm_calls"),
            "A8_calls_by_stage": cost.get("llm_calls_by_stage"),
            "A8_calls_by_model": cost.get("llm_calls_by_model"),
            "A8_incremental_multiplier": cost.get("incremental_call_multiplier"),
            "limitations": (d.get("limitations") or [])[:5],
        }
    except Exception as e:  # noqa: BLE001
        rec = {"label": label, "qid": qid, "p": None, "A8_wall_s": round(time.time() - t0, 1),
               "error": f"{type(e).__name__}: {e}"[:200]}
    CKPT.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(rec, indent=1, default=str))
    print(f"  [{label}] done p={rec.get('p')} status={rec.get('A4_status')} "
          f"calls={rec.get('A8_ledger_total_calls')} wall={rec.get('A8_wall_s')}s")
    return rec


def run(only_index=None):
    from swm.api.deepseek_backend import default_chat_fn
    rows = {r["question_id"]: r for r in fetch_btf3()}
    factory = lambda: default_chat_fn(system="Reply ONLY JSON.", max_tokens=8000, temperature=0.2)
    todo = FROZEN5 if only_index is None else [FROZEN5[only_index]]
    results = [_one(label, qid, rows, factory) for label, qid in todo]

    # ---- scoring join (A7; footer only) ----
    for r in results:
        row = rows[r["qid"]]
        r["A7_outcome"] = int(row["resolution"])
        r["A7_sota"] = None if row.get("sota_forecast_probability") is None \
            else round(float(row["sota_forecast_probability"]) / 100.0, 4)
    CKPT.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n": len(results), "fidelity": "full_llm_actors_single_run",
                               "results": results}, indent=1, default=str))
    print("\n=== EXP-110 frozen-5 full-actor audit (WSim) ===")
    print("  label     | outside | sim    | combined | source            | n_worlds | SOTA | outcome")
    for r in results:
        print(f"  {r['label']:9s} | {str(r.get('W_outside_view')):7s} | "
              f"{str(r.get('W_simulation_derived')):6s} | {str(r.get('W_final_combined')):8s} | "
              f"{str(r.get('W_final_source')):17s} | {str(r.get('W_n_worlds_valid')):8s} | "
              f"{r.get('A7_sota')} | {r.get('A7_outcome')}")
    genuine = [r for r in results if r.get('W_final_source') == 'weighted_simulation']
    print(f"  → {len(genuine)}/{len(results)} final numbers came from ACTOR SIMULATION (weighted_simulation)")
    print(f"  wrote {OUT}")
    return results


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
