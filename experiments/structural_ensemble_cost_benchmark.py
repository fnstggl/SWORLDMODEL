"""Honest cost benchmark for the structural ensemble (ensemble contract Section 27).

Arms (identical question/seed/backend; a deterministic scripted backend so call counts are EXACT):
  A  single_model_ablation   — the explicit pre-ensemble path
  B  default_ensemble        — the new production default
  C  ensemble_cache_off      — identical-call caching disabled (cost-only knob)
  D  ensemble_no_pilot_reuse — pilot particles discarded, full budget rerun (cost-only knob)
  E  ensemble_max_capacity   — maximum-capacity generation ceiling

Measured per arm: LLM calls by stage (generation/critic/compile/conditioning/actor), total calls,
prompt/response chars (token proxy — no provider pricing is fabricated), wall-clock, pilot particles,
full particles, cache hits, % pilot computation reused, models fully simulated, and the OBSERVED
incremental multiplier vs arm A. Survivor budgets are never reduced to flatter the report; the
multiplier is whatever it is.

Run: PYTHONPATH=. python experiments/structural_ensemble_cost_benchmark.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

OUT = ROOT / "artifacts" / "structural_ensemble"

QUESTION = "Will the initiative be approved?"
POLICY_BASE = {"drop_phases": ["phase2_evidence", "event_time"]}   # hermetic (no live retrieval)


class CountingLLM:
    """Wraps the scripted ensemble backend with exact backend-call accounting."""

    def __init__(self):
        from test_structural_ensemble import four_way_llm
        self.inner = four_way_llm()
        self.n = 0
        self.prompt_chars = 0
        self.response_chars = 0

    def __call__(self, prompt):
        self.n += 1
        self.prompt_chars += len(prompt)
        out = self.inner(prompt)
        self.response_chars += len(out or "")
        return out


def _run(arm: str, policy: dict, compute_budget=None) -> dict:
    from swm.world_model_v2.unified_runtime import simulate_world
    llm = CountingLLM()
    t0 = time.time()
    res = simulate_world(QUESTION, as_of="2025-06-01", horizon="2025-09-01", llm=llm, seed=3,
                         execution_policy=policy, compute_budget=compute_budget)
    dt = round(time.time() - t0, 3)
    se = res.structural_ensemble or {}
    cm = se.get("cost_manifest") or {}
    sim = se.get("simulation_manifest") or {}
    pilot = sum(v.get("pilot_particles", 0) for v in sim.values())
    full = sum(v.get("final_particles", 0) for v in sim.values())
    if not se:                                   # single-model ablation: one plan's full budget
        full = int((res.provenance or {}).get("n_particles") or 0)
    reused = sum(v.get("pilot_particles", 0) for v in sim.values()
                 if v.get("pilot_reused_as_prefix"))
    return {"arm": arm, "status": res.simulation_status,
            "backend_llm_calls": llm.n,
            "prompt_chars": llm.prompt_chars, "response_chars": llm.response_chars,
            "llm_calls_by_stage": cm.get("llm_calls_by_stage", {}),
            "cache_hits_by_stage": cm.get("cache_hits_by_stage", {}),
            "total_cache_hits": cm.get("total_cache_hits", 0),
            "n_models_fully_simulated": se.get("n_fully_simulated", 0 if se else 1),
            "pilot_particles": pilot, "full_particles": full,
            "pilot_particles_reused_pct": round(100.0 * reused / pilot, 1) if pilot else None,
            "wall_clock_s": dt,
            "provider_cost_note": "scripted deterministic backend — call/char counts are exact; no "
                                  "provider pricing is fabricated (see forensics for live-call counts)"}


def main():
    rows = [
        _run("A_single_model_ablation",
             {**POLICY_BASE, "structural_mode": "single_structural_model"}),
        _run("B_default_ensemble", dict(POLICY_BASE)),
        _run("C_ensemble_cache_off", {**POLICY_BASE, "cache_mode": "off"}),
        _run("D_ensemble_no_pilot_reuse", {**POLICY_BASE, "pilot_reuse": "off"}),
        _run("E_ensemble_max_capacity", dict(POLICY_BASE), compute_budget="maximum_capacity"),
    ]
    base_calls = rows[0]["backend_llm_calls"] or 1
    base_particles = rows[0]["full_particles"] or rows[0]["pilot_particles"] or 1
    for r in rows:
        r["llm_call_multiplier_vs_single"] = round(r["backend_llm_calls"] / base_calls, 2)
        r["particle_multiplier_vs_single"] = round(
            (r["full_particles"] or r["pilot_particles"]) / base_particles, 2)
    savings = {
        "identical_call_caching": rows[2]["backend_llm_calls"] - rows[1]["backend_llm_calls"],
        "pilot_reuse_particles": (rows[3]["full_particles"] + rows[3]["pilot_particles"]) -
                                 rows[1]["full_particles"],
        "evidence_sharing": "evidence gathered once per run in every ensemble arm (union of recon "
                            "requirements; hermetic arms drop retrieval identically)",
        "deduplication": "duplicate candidates merge before pilot/full simulation "
                         f"(arm B merged {rows[1]['n_models_fully_simulated']} of 4 initial "
                         "candidates into full simulation)",
        "note": "the observed multiplier depends on how many models survive and how many actor calls "
                "they require — it is reported, not promised",
    }
    report = {"schema_version": "structural_ensemble.cost_benchmark.v1",
              "question": QUESTION, "arms": rows, "observed_savings": savings}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cost_benchmark.json"
    path.write_text(json.dumps(report, indent=1))
    hdr = f"{'arm':28s} {'calls':>6s} {'xLLM':>6s} {'pilot':>6s} {'full':>6s} {'xPart':>6s} " \
          f"{'hits':>5s} {'reuse%':>7s} {'wall_s':>7s}"
    print(hdr)
    for r in rows:
        print(f"{r['arm']:28s} {r['backend_llm_calls']:6d} {r['llm_call_multiplier_vs_single']:6.2f} "
              f"{r['pilot_particles']:6d} {r['full_particles']:6d} "
              f"{r['particle_multiplier_vs_single']:6.2f} {r['total_cache_hits']:5d} "
              f"{str(r['pilot_particles_reused_pct']):>7s} {r['wall_clock_s']:7.2f}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
