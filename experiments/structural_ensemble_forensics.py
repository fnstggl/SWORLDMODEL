"""LIVE LLM-backed forensic runs of the default structural-ensemble runtime (Section 26).

Five structurally different cases through the REAL canonical runtime with the repo's configured
production backend (ResilientLLM — DeepSeek-V3 via HF router / DeepSeek direct, on-disk cached,
temperature 0). Nothing is stubbed: real independent generation calls, real critics, real Stage-B
compilation, real shared evidence retrieval (live as-of connectors), real pilots and full per-model
simulations through the persistence funnel, real qualitative actors where plans declare decisions.

Per case the artifact saves: every generation/critic/compile prompt+response (bounded), initial
candidates, evidence requirements + shared-evidence manifest, executable-plan structural signatures,
critic traces, omission findings, dedup decisions, pilot results, promotion decisions, final particle
counts, per-model results + operator-delta census (the execution-trace proof that each model ran the
production funnel), the aggregate, sensitivity classification, reversal conditions, structural VOI,
cache use, exact call counts, wall-clock, and unresolved limitations.

ARCHITECTURE PROBES ONLY — these runs demonstrate the execution path and its cost, never predictive
accuracy.

Run: PYTHONPATH=. python experiments/structural_ensemble_forensics.py [--case N] [--max-tokens 2400]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts" / "structural_ensemble" / "forensics"

CASES = [
    {"case_id": "founder_launch",
     "question": ("Should Mara launch her indie meal-planning app publicly next month, and will the "
                  "launch reach 1,000 active users within 60 days of launch?"),
     "intervention": "Mara announces a public launch on Product Hunt and her waitlist next month",
     "as_of": "2026-07-10", "horizon": "2026-10-01",
     "why": "distribution, product readiness, trust and competition are plausible competing structures"},
    {"case_id": "partnership_negotiation",
     "question": ("Will the regional grocery chain Hartfield's sign the co-branding partnership with "
                  "the local bakery collective before the end of Q3 2026?"),
     "intervention": "the bakery collective sends a revised revenue-share proposal to Hartfield's CFO",
     "as_of": "2026-07-10", "horizon": "2026-09-30",
     "why": "route, authority, incentives and relationship history could each dominate"},
    {"case_id": "regulatory_institution",
     "question": ("Will the FAA grant the beyond-visual-line-of-sight drone delivery waiver requested "
                  "by a mid-size logistics operator before the end of 2026?"),
     "intervention": "",
     "as_of": "2026-07-10", "horizon": "2026-12-31",
     "why": "institutional procedure may matter more than public opinion"},
    {"case_id": "personal_reaction",
     "question": "How will Jonah react if I cancel our long-planned weekend hiking trip two days before?",
     "intervention": "",
     "as_of": "2026-07-10", "horizon": "",
     "user_context": {"individual": {
         "person_id": "jonah", "relationship": "close friend since college",
         "history": ["we planned this trip for three months",
                     "I already rescheduled one dinner on him this spring",
                     "he organized the gear list and booked the campsite"],
         "stimulus": "Hey — I'm really sorry, but I have to cancel next weekend's trip. "
                     "A work deadline moved onto that Friday.",
         "n_hypotheses": 3, "samples_per_hypothesis": 2}},
     "why": "relationship, attention, interpretation and external obligations differ across frames"},
    {"case_id": "hybrid_physical_market",
     "question": ("Will the coastal microgrid operator complete its battery-storage expansion before "
                  "the 2026 hurricane season peak (September 10), given supplier lead times, permit "
                  "queues and interconnection constraints?"),
     "intervention": "",
     "as_of": "2026-07-10", "horizon": "2026-09-10",
     "why": "physical, operational and market constraints matter alongside actors"},
]


class RecordingLLM:
    """Production backend + full prompt/response tap (bounded) + exact counters.

    Every call is ALSO streamed to a per-case partial JSONL as it happens, so a run killed at a
    wall-clock cap still leaves its complete prompt/response trace on disk (honest partial evidence,
    never lost work)."""

    def __init__(self, max_tokens=2400, stream_path: Path = None):
        from swm.api.resilient_llm import ResilientLLM
        self.inner = ResilientLLM(max_tokens=max_tokens)
        self.calls = []
        self.t_llm = 0.0
        self.stream_path = stream_path

    def __call__(self, prompt):
        t0 = time.time()
        out = self.inner(prompt)
        dt = time.time() - t0
        self.t_llm += dt
        row = {"i": len(self.calls), "prompt": prompt[:4000],
               "response": (out or "")[:4000], "latency_s": round(dt, 2),
               "prompt_chars": len(prompt), "response_chars": len(out or ""),
               "provider_counters": dict(self.inner.calls)}
        self.calls.append(row)
        if self.stream_path is not None:
            try:
                with self.stream_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass
        return out

    def counters(self):
        return dict(self.inner.calls)


def _model_rows(se, res):
    rows = []
    per_prov = (res.provenance or {}).get("per_model_provenance") or {}
    for m in se.get("models", []):
        prov = per_prov.get(m.get("model_id")) or {}
        # tolerant access: full-route rows carry plan hashes/pilots; personal-reaction frame rows
        # carry the frame subset — both are archived as-is
        rows.append({**{k: m.get(k) for k in ("model_id", "generation_role", "causal_thesis",
                                              "plan_hash", "schema_hash", "support_class",
                                              "promotion_status", "promotion_reason",
                                              "pilot_particles", "final_particles", "plan_lineage",
                                              "pilot_result", "prediction", "validation",
                                              "decisive_actors", "decisive_constraints")},
                     "operator_delta_census": prov.get("operator_delta_census"),
                     "n_particles_executed": prov.get("n_particles"),
                     "actor_policy_report": prov.get("actor_policy_report"),
                     "phase8": prov.get("phase8")})
    return rows


def run_case(case: dict, max_tokens: int) -> dict:
    from swm.world_model_v2.unified_runtime import simulate_world
    stream = OUT / f"partial_{case['case_id']}.calls.jsonl"
    try:                                       # each attempt streams its own complete sequence
        stream.unlink()
    except OSError:
        pass
    llm = RecordingLLM(max_tokens=max_tokens, stream_path=stream)
    t0 = time.time()
    err = ""
    try:
        res = simulate_world(case["question"], as_of=case["as_of"], horizon=case.get("horizon", ""),
                             intervention=case.get("intervention", ""), llm=llm, seed=11,
                             user_context=case.get("user_context"))
    except Exception as e:  # noqa: BLE001 — the artifact records the failure verbatim
        res, err = None, f"{type(e).__name__}: {e}"
    wall = round(time.time() - t0, 1)
    se = (res.structural_ensemble if res is not None else None) or {}
    art = {
        "schema_version": "structural_ensemble.forensics.v1",
        "case": {k: v for k, v in case.items() if k != "user_context"},
        "backend": {"provider_counters": llm.counters(), "max_tokens": max_tokens,
                    "temperature": 0.0, "model": "DeepSeek-V3 (HF router / direct, cached)"},
        "wall_clock_s": wall, "llm_wall_clock_s": round(llm.t_llm, 1),
        "n_llm_calls_recorded": len(llm.calls),
        "harness_error": err,
        "result": None if res is None else {
            "simulation_status": res.simulation_status,
            "support_grade": res.support_grade,
            "raw_distribution": res.raw_distribution,
            "structural_disagreement": res.structural_disagreement,
            "limitations": res.limitations,
            "latency_s": res.latency_s},
        "ensemble": None if not se else {
            "ensemble_id": se.get("ensemble_id"),
            "generation_policy": se.get("generation_policy"),
            "n_independent_generation_calls": se.get("n_independent_generation_calls"),
            "n_initial_candidates": se.get("n_initial_candidates"),
            "n_expansion_candidates": se.get("n_expansion_candidates"),
            "n_rejected": se.get("n_rejected"), "n_repaired": se.get("n_repaired"),
            "n_merged": se.get("n_merged"),
            "n_pilot_simulated": se.get("n_pilot_simulated"),
            "n_fully_simulated": se.get("n_fully_simulated"),
            "generation_manifest": se.get("generation_manifest"),
            "critic_manifest": se.get("critic_manifest"),
            "merge_manifest": se.get("merge_manifest"),
            "simulation_manifest": se.get("simulation_manifest"),
            "shared_evidence_bundle_hash": se.get("shared_evidence_bundle_hash"),
            "structural_coverage": se.get("structural_coverage"),
            "unresolved_alternatives": se.get("unresolved_alternatives"),
            "convergence_certificate": se.get("convergence_certificate"),
            "stopping_reason": se.get("stopping_reason"),
            "aggregation_method": se.get("aggregation_method"),
            "equal_weight_mixture": se.get("equal_weight_mixture"),
            "robust_range": se.get("robust_range"),
            "structural_sensitivity": se.get("structural_sensitivity"),
            "reversal_conditions": se.get("reversal_conditions"),
            "structural_value_of_information": se.get("structural_value_of_information"),
            "cost_manifest": se.get("cost_manifest"),
            "human_summary": se.get("human_summary"),
            "models": _model_rows(se, res)},
        "llm_calls": llm.calls,
    }
    return art


def rebuild_summary() -> dict:
    """Combined summary from EVERY case artifact on disk (single-case invocations may run in
    parallel; the summary is always the union, never one invocation's slice)."""
    rows = []
    for p in sorted(OUT.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            art = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        ens = art.get("ensemble") or {}
        rows.append({"case_id": (art.get("case") or {}).get("case_id", p.stem),
                     "status": (art.get("result") or {}).get("simulation_status", "harness_error"),
                     "n_independent_generation_calls": ens.get("n_independent_generation_calls"),
                     "n_fully_simulated": ens.get("n_fully_simulated"),
                     "sensitivity": (ens.get("structural_sensitivity") or {}).get("classification"),
                     "n_llm_calls": art.get("n_llm_calls_recorded"),
                     "provider_counters": (art.get("backend") or {}).get("provider_counters"),
                     "wall_clock_s": art.get("wall_clock_s"),
                     "error": str(art.get("harness_error", ""))[:160]})
    summary = {"schema_version": "structural_ensemble.forensics.summary.v1",
               "note": "architecture probes — no predictive-accuracy claim",
               "cases": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=None, help="run one case index (0-4)")
    ap.add_argument("--max-tokens", type=int, default=2400)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        s = rebuild_summary()
        print(json.dumps(s["cases"], indent=1))
        return 0
    cases = CASES if args.case is None else [CASES[args.case]]
    for case in cases:
        print(f"=== {case['case_id']} ===", flush=True)
        art = run_case(case, args.max_tokens)
        path = OUT / f"{case['case_id']}.json"
        path.write_text(json.dumps(art, indent=1, default=str))
        summary = rebuild_summary()
        print(json.dumps(summary["cases"][-1] if summary["cases"] else {}), flush=True)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
