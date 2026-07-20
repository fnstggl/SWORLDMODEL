"""EXP-103 probe: does the compiler build EXECUTABLE structure (not just metadata) when the decomposition
is NOT truncated? Compiles the 5 BTF-3 questions with a real token budget + JSON retry and dumps, per
question: whether the reply was truncated, the accepted EXECUTABLE mechanisms (operators), the required
causal processes, scheduled events, and whether structure->outcome operators (institutional_decision,
institutional_vote, actor_action_aggregation, population_aggregation) made it into the plan.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp103_compile_probe
"""
from __future__ import annotations
import json
from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
from experiments.exp102_btf3_wmv2_full import QIDS

BIND_OPS = {"institutional_decision", "institutional_vote", "actor_action_aggregation",
            "population_aggregation", "aggregate_outcome_mechanism", "production_actor_policy",
            "scheduled_fact"}


def run():
    from swm.api.deepseek_backend import default_chat_fn
    from swm.world_model_v2.compiler import compile_world
    rows = {r["question_id"]: r for r in fetch_btf3()}
    base = default_chat_fn(system="Reply ONLY JSON.", max_tokens=8000, temperature=0.2)
    out = []
    for qid in QIDS:
        q = _forecast_input(rows[qid])
        ev = (f"Resolution criteria: {q['resolution_criteria']}\n\nBackground (as of "
              f"{str(q['present_date'])[:10]}): {q['background']}")
        seen = {}

        def llm(p, _s=seen):
            r = base(p)
            _s["last"] = r
            return r

        try:
            plan = compile_world(q["question"], llm=llm, evidence=ev, as_of=str(q["present_date"])[:10],
                                 horizon=str(q["expected_resolution_date"])[:10], seed=0)
            ops = [m.get("operator") for m in plan.accepted_mechanisms]
            rec = {"qid": qid[:8], "q": q["question"][:60],
                   "reply_chars": len(seen.get("last", "")),
                   "processes": [c["process"] for c in plan.mechanism_choices],
                   "accepted_operators": ops,
                   "binding_ops_present": sorted(set(ops) & BIND_OPS),
                   "n_scheduled_events": len(plan.scheduled_events),
                   "n_institutions": len(plan.institutions), "n_entities": len(plan.entities),
                   "n_structural_hyp": len(plan.structural_hypotheses),
                   "support_grade": plan.support_grade}
        except Exception as e:
            rec = {"qid": qid[:8], "error": f"{type(e).__name__}: {e}"[:200]}
        out.append(rec)
        print(json.dumps(rec, indent=1))
    open("experiments/results/exp103_compile_probe.json", "w").write(json.dumps(out, indent=1))
    print("\nSUMMARY binding-ops present per q:",
          {r.get("qid"): r.get("binding_ops_present") for r in out})


if __name__ == "__main__":
    run()
