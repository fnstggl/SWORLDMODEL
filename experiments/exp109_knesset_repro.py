"""EXP-109: Knesset p=None reproduction + per-stage timing/call profile, on the FIXED runtime.

Reproduces the malformed Knesset run (qid 0851f82c…, prop_check.json: p=None, status=under_modeled,
189 calls, 1707 s) AFTER the ensemble-level grounded-outside-view fix. Two deliverables, one run:

  STEP 1 (profile the 28 min): where did the calls/time go — an accidental loop / repeated validation,
          or the architecture's normal cost? Answered from the runtime's own CallLedger
          (structural_ensemble.cost_manifest.llm_calls_by_stage / _by_model) plus a call-timing wrapper
          that records the TRUE backend-call count and wall-time-in-LLM (the ledger misses any call made
          outside a metered stage — the gap between the two IS the actor-rollout cost).
  STEP 4 (verify the fix): the run must now return a NUMBER (~5%) with an under-modeled WARNING and a
          provenance.grounded_outside_view_fallback record — never a silent None.

Full LLM actors (default; NOT SWM_LLM_ACTORS=off) so the 28-min cost is reproduced and profiled.
Single run, no mean-of-K (per the mandate). Leakage-quarantined: only allowlisted as-of fields reach
the forecaster; the resolution/SOTA join happens only in the printed footer.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp109_knesset_repro
"""
from __future__ import annotations
import dataclasses
import json
import statistics
import time
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input, _ts

QID = "0851f82c-aabd-57f0-abbb-4a23f99963c2"
OUT = Path("experiments/results/exp109_knesset_repro.json")


class TimingLLM:
    """`llm(prompt) -> str` pass-through recording the TRUE backend-call count + per-call wall time.
    The runtime's CallLedger only counts metered stages; this counts EVERY call, so total_calls here vs.
    cost_manifest.total_llm_calls reveals how much of the cost is the (unmetered) actor rollout."""

    def __init__(self, llm):
        self._llm = llm
        self.durations = []

    def __call__(self, prompt, *a, **k):
        t = time.time()
        try:
            return self._llm(prompt, *a, **k)
        finally:
            self.durations.append(time.time() - t)

    def profile(self) -> dict:
        d = self.durations
        return {"true_backend_calls": len(d),
                "llm_wall_s_total": round(sum(d), 1),
                "call_s_mean": round(statistics.mean(d), 2) if d else None,
                "call_s_median": round(statistics.median(d), 2) if d else None,
                "call_s_p90": round(sorted(d)[int(0.9 * len(d))], 2) if len(d) >= 10 else None,
                "call_s_max": round(max(d), 2) if d else None}


def run() -> dict:
    from swm.world_model_v2.unified_runtime import simulate_world
    from swm.api.deepseek_backend import default_chat_fn

    rows = {r["question_id"]: r for r in fetch_btf3()}
    row = rows[QID]
    q = _forecast_input(row)                                  # allowlisted fields ONLY
    as_of, resolve = _ts(q["present_date"]), _ts(q["expected_resolution_date"])
    horizon_days = round((resolve - as_of) / 86400, 1)

    base = default_chat_fn(system="Reply ONLY JSON.", max_tokens=8000, temperature=0.2)
    llm = TimingLLM(base)

    print(f"EXP-109  Knesset repro (full LLM actors, single run)\n  Q: {q['question'][:100]}")
    print(f"  as_of {str(q['present_date'])[:10]}  horizon {horizon_days}d")
    t0 = time.time()
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
        "qid": QID, "question": q["question"][:110], "horizon_days": horizon_days,
        # ---- STEP 4: the fix ----
        "p": p, "status": d.get("simulation_status"),
        "has_forecast": bool(res.has_forecast()),
        "raw_distribution": d.get("raw_distribution"),
        "grounded_outside_view_fallback": prov.get("grounded_outside_view_fallback"),
        "under_modeled_subtypes": d.get("under_modeled_subtypes"),
        "under_modeled_components": [c.get("component") for c in (d.get("under_modeled_components") or [])][:6],
        "limitations": (d.get("limitations") or [])[:6],
        # ---- STEP 1: the profile ----
        "wall_s": wall,
        "timing_wrapper": llm.profile(),
        "ledger_total_llm_calls": cost.get("total_llm_calls"),
        "llm_calls_by_stage": cost.get("llm_calls_by_stage"),
        "llm_calls_by_model": cost.get("llm_calls_by_model"),
        "cache_hits_by_stage": cost.get("cache_hits_by_stage"),
        "incremental_call_multiplier": cost.get("incremental_call_multiplier"),
        "single_model_equivalent_llm_calls": cost.get("single_model_equivalent_llm_calls"),
        "n_models_promoted": len(ens.get("model_distributions") or {}),
        "aggregation_method": ens.get("aggregation_method"),
        "evidence_sufficiency": prov.get("evidence_sufficiency"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, default=str))

    print(f"\n  --- STEP 4 (fix) ---")
    print(f"  p={rec['p']}  status={rec['status']}  has_forecast={rec['has_forecast']}")
    gof = rec["grounded_outside_view_fallback"]
    print(f"  grounded_outside_view_fallback: {json.dumps(gof)[:300] if gof else None}")
    print(f"  --- STEP 1 (profile) ---")
    print(f"  wall {wall}s   timing_wrapper {json.dumps(rec['timing_wrapper'])}")
    print(f"  ledger total_llm_calls={rec['ledger_total_llm_calls']}  by_stage={json.dumps(rec['llm_calls_by_stage'])}")
    print(f"  n_models={rec['n_models_promoted']}  agg={rec['aggregation_method']}")
    # ---- scoring join (footer only) ----
    print(f"\n  [scoring] outcome={int(row['resolution'])}  SOTA={row.get('sota_forecast_probability')}%  "
          f"model_p={rec['p']}")
    print(f"  wrote {OUT}")
    return rec


if __name__ == "__main__":
    run()
