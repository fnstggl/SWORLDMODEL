"""Production-path forensic traces for actor-mediated causal execution (deliverable 14).

Four traces, each through the EXACT public production route (no toy path, no parallel
simulator), with the live LLM backend when available:

  T1  public-statement coalition cascade      → simulate_world()
  T2  private interpersonal communication     → simulate_individual_reaction()
  T3  institutional decision                  → simulate_world() (institution + vote)
  T4  Phase-13 matched counterfactual         → phase13.recommend_action()

Each trace records: the route entered, the run classification + epistemic contract, the
semantic-event cascade, actor decisions with hypothesis provenance, demoted scalar writes,
approximation manifests, and cost/latency — written to
experiments/results/actor_mediated/forensics/.

Usage: PYTHONPATH=. python experiments/actor_mediated_forensics.py [--trace=T1,T2,T3,T4]
Requires DEEPSEEK_API_KEY for the live arm; falls back to recording the absence honestly.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("experiments/results/actor_mediated/forensics")
AS_OF = "2026-07-18"


def _llm():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return None
    from swm.api.deepseek_backend import deepseek_chat_fn
    return deepseek_chat_fn(max_tokens=1400)


def _res_trace(res) -> dict:
    prov = res.provenance or {}
    am = prov.get("actor_mediated") or {}
    return {
        "simulation_status": res.simulation_status,
        "support_grade": res.support_grade,
        "raw_probability": res.raw_probability,
        "raw_distribution": res.raw_distribution,
        "run_classification": prov.get("run_classification"),
        "epistemic_contract": prov.get("epistemic_contract"),
        "joint_world": prov.get("joint_world"),
        "actor_policy_mode": prov.get("actor_policy_mode"),
        "event_cascade": am.get("event_cascade"),
        "n_semantic_events": am.get("n_semantic_events"),
        "demoted_scalar_writes": am.get("demoted_scalar_writes"),
        "n_approximations": am.get("n_approximations"),
        "actor_decisions": {
            aid: {"raw": d.get("raw_qualitative_simulation_distribution"),
                  "n_branches": d.get("n_qualitative_branches"),
                  "n_numeric_fallbacks": d.get("n_excluded_numeric_fallbacks"),
                  "calibration_status": d.get("calibration_status"),
                  "cluster_version": d.get("cluster_version"),
                  "sample_rows": [
                      {k: r.get(k) for k in ("branch_id", "hypothesis_id", "cluster",
                                             "decision_source", "novel_action_unmodeled")}
                      for r in (d.get("rows") or [])[:6]]}
            for aid, d in (prov.get("actor_decision_distributions") or {}).items()},
        "operator_delta_census": prov.get("operator_delta_census"),
        "manifest_phase2_evidence": (prov.get("active_component_manifest") or {}).get(
            "phase2_evidence"),
        "limitations": res.limitations,
        "latency_s": res.latency_s,
    }


def t1_public_statement_cascade(llm) -> dict:
    from swm.world_model_v2.unified_runtime import simulate_world
    q = ("Will the three-party governing coalition in a parliamentary system publicly hold "
         "together through September 1, 2026, after the senior partner's leader publicly "
         "ruled out the junior partners' demanded budget concessions on July 17, 2026?")
    res = simulate_world(q, as_of=AS_OF, horizon="2026-09-01", llm=llm, seed=7,
                     compute_budget={"n_particles": 12})
    return {"route": "swm.world_model_v2.unified_runtime.simulate_world",
            "question": q, **_res_trace(res)}


def t2_private_communication(llm) -> dict:
    from swm.world_model_v2.individual_reaction import simulate_individual_reaction
    out = simulate_individual_reaction(
        person_id="jordan",
        stimulus=("Hey — I know I said I'd co-present at Friday's board review, but I need "
                  "to pull out and let you run it alone. I'll brief you Thursday night."),
        context={"role": "senior product manager and longtime colleague",
                 "your_role": "peer engineering lead",
                 "relationship": "colleague",
                 "history": [
                     "Last month Jordan covered for you on short notice and said 'you owe me one'.",
                     "Jordan has twice mentioned being overloaded this quarter.",
                     "You and Jordan present the quarterly review together every quarter."]},
        llm=llm, n_hypotheses=3, samples_per_hypothesis=2, seed=11)
    return {"route": "swm.world_model_v2.individual_reaction.simulate_individual_reaction",
            "result": {k: v for k, v in out.items() if k != "rows"},
            "sample_rows": [
                {k: r.get(k) for k in ("hypothesis_id", "observable_response", "cluster",
                                       "novel_action_unmodeled", "decision_source")}
                for r in (out.get("rows") or [])[:6]]}


def t3_institutional_decision(llm) -> dict:
    from swm.world_model_v2.unified_runtime import simulate_world
    q = ("Will a nine-member university faculty senate, which requires a simple majority "
         "of members present with a quorum of six, approve the proposed switch to "
         "semester-long course scheduling at its August 2026 meeting, given that the "
         "provost publicly endorsed the switch and two department chairs on the senate "
         "have publicly opposed it?")
    res = simulate_world(q, as_of=AS_OF, horizon="2026-09-15", llm=llm, seed=13,
                     compute_budget={"n_particles": 12})
    return {"route": "swm.world_model_v2.unified_runtime.simulate_world (institutional)",
            "question": q, **_res_trace(res)}


def t4_phase13_matched_counterfactual(llm) -> dict:
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.phase13 import recommend_action
    from swm.world_model_v2.phase13.contracts import DecisionProblem, Stakeholder, UtilitySpec
    q = ("Will the wavering co-founder agree to stay through the product launch in "
         "September 2026?")
    plan = compile_world(q, llm=llm, evidence="", as_of=AS_OF, horizon="2026-09-30", seed=5)
    entity_ids = [str(e.get("id")) for e in (plan.entities or []) if isinstance(e, dict)]
    maker = next((e for e in entity_ids if "ceo" in e.lower() or "founder" in e.lower()
                  and "co" not in e.lower()), entity_ids[0] if entity_ids else "user")

    def stays_utility(outcome: dict) -> float:
        r = str(outcome.get("readout", "")).lower()
        if r in ("true", "yes", "1", "stays"):
            return 1.0
        qs = outcome.get("quantities") or {}
        for name, v in qs.items():
            if name.startswith(("pathway_progress:", "outcome")) and isinstance(v, (int, float)):
                return float(v)
        return 0.0

    problem = DecisionProblem(
        decision_id="forensic_t4_cofounder",
        decision_maker=maker,
        authority=["communicate", "allocate", "coordinate", "schedule", "guarantee",
                   "preserve_option", "publish", "final_decision"],
        controllable_resources={"retention_budget": 1.0},
        context="What should the CEO do to keep the wavering co-founder through launch?",
        as_of=AS_OF + "T00:00:00Z", horizon="2026-09-30T00:00:00Z",
        utility=UtilitySpec(stakeholders=[Stakeholder("ceo", utility_fn=stays_utility)],
                            provenance="user_supplied"),
        candidate_actions=[], generated_action_permission=True, human_approval_required=True)
    t0 = time.time()
    dr = recommend_action(problem, plan, budget="diagnostic", seed=5, n_particles=8, llm=llm)
    d = dr.as_dict() if hasattr(dr, "as_dict") else {
        k: getattr(dr, k, None) for k in ("recommendation", "abstention", "ranking",
                                          "runtime_fingerprint", "diagnostics")}
    return {"route": "swm.world_model_v2.phase13.recommend_action",
            "question": q, "wall_s": round(time.time() - t0, 1),
            "n_evaluated": len(getattr(dr, "evaluated", []) or []),
            "result": json.loads(json.dumps(d, default=str))}


def main():
    only = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--trace=")),
                ["T1", "T2", "T3", "T4"])
    llm = _llm()
    OUT.mkdir(parents=True, exist_ok=True)
    header = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "llm_backend": "deepseek (env key)" if llm else "ABSENT — numeric fallback path",
              "as_of": AS_OF}
    traces = {"T1": t1_public_statement_cascade, "T2": t2_private_communication,
              "T3": t3_institutional_decision, "T4": t4_phase13_matched_counterfactual}
    for tid in only:
        fn = traces[tid]
        print(f"== {tid} ==", flush=True)
        t0 = time.time()
        try:
            row = fn(llm)
        except Exception as e:  # noqa: BLE001 — a failed trace is a recorded failure
            import traceback
            row = {"error": f"{type(e).__name__}: {e}"[:400],
                   "traceback": traceback.format_exc()[-2000:]}
        row["_meta"] = {**header, "trace_wall_s": round(time.time() - t0, 1)}
        path = OUT / f"trace_{tid}.json"
        path.write_text(json.dumps(row, indent=1, default=str))
        print(f"wrote {path} ({row['_meta']['trace_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
