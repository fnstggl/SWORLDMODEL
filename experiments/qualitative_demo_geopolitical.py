"""Demonstration 1 — multi-actor geopolitical simulation under persistent qualitative actors.

A frozen Russia–Ukraine-style ceasefire scenario (hand-constructed world, evidence-style stances
as of the frozen date — no post-date knowledge is supplied): three actors, declared pathway
processes, two decision rounds, six independent world branches. Mode D
(persistent_qualitative_llm_policy) gives each branch its own qualitative hidden-state
hypothesis of each Tier-1 actor; every branch's actor independently interprets, decides, and
executes — pathway quantities move per branch — and the run ends with counted raw action
distributions per actor plus the numeric-arm comparison on the identical scenario.

    DEEPSEEK_API_KEY=… PYTHONPATH=. python experiments/qualitative_demo_geopolitical.py
"""
from __future__ import annotations

import calendar
import json
import time as _time
from pathlib import Path

from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase4_execution import ActorPolicyRuntime
from swm.world_model_v2.qualitative_actor import (
    QualitativeActorPolicyRuntime, QualitativeConfig, QualitativeDecisionEngine,
    aggregate_actor_decisions, load_actor_state,
)
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

AS_OF = float(calendar.timegm(_time.strptime("2026-09-13", "%Y-%m-%d")))
N_BRANCHES = 6
ROUNDS = 2
RESULTS = Path("experiments/results")

STANCES = {
    "Vladimir_Putin": [
        {"actor": "Vladimir_Putin", "commitment_level": "committed_to_prevent",
         "pathway": "cooperative_agreement", "reliability": "high", "capability": "high",
         "quote": "the objectives of the special military operation will be achieved"},
        {"actor": "Vladimir_Putin", "commitment_level": "actively_pursuing",
         "pathway": "unilateral_action", "target_mode": "russian_victory",
         "reliability": "high", "capability": "high"}],
    "Volodymyr_Zelenskyy": [
        {"actor": "Volodymyr_Zelenskyy", "commitment_level": "committed_to_prevent",
         "pathway": "unilateral_action", "target_mode": "russian_victory",
         "reliability": "high", "capability": "medium",
         "quote": "Ukraine will never accept the loss of its territory"},
        {"actor": "Volodymyr_Zelenskyy", "commitment_level": "conditionally_opposed",
         "pathway": "cooperative_agreement", "reliability": "medium", "capability": "medium"}],
}
EVIDENCE = [
    ("2026-09-10", "Front lines have moved little in three months; attrition on both sides is heavy."),
    ("2026-09-11", "The US president publicly pressed both sides toward a ceasefire framework."),
    ("2026-09-12", "European capitals debated the durability of military funding into next year."),
    ("2026-09-12", "Russian officials repeated that operation objectives stand; Kyiv repeated that territory is not negotiable."),
]
DECISIONS = {
    "Vladimir_Putin": {"situation": "A ceasefire framework is circulating with US backing; your "
                                    "military advises the front is stable but costly.",
                       "candidate_actions": ["escalate", "hold_position", "delay",
                                             "counteroffer", "seek_mediator", "mobilize"]},
    "Volodymyr_Zelenskyy": {"situation": "The US-backed framework implies freezing current lines; "
                                          "your ammunition pipeline depends on allied politics.",
                            "candidate_actions": ["reject", "counteroffer", "hold_position",
                                                  "seek_mediator", "mobilize", "delay"]},
}


def build_world(branch_id: str) -> WorldState:
    w = WorldState("geo_demo", branch_id, SimulationClock(AS_OF, AS_OF),
                   network=RelationGraph(), information=InformationLedger())
    for aid, role, cap in (("Vladimir_Putin", "president_of_russia", 0.82),
                           ("Volodymyr_Zelenskyy", "president_of_ukraine", 0.61),
                           ("Donald_Trump", "president_of_us", 0.75)):
        e = Entity(aid)
        e.set("roles", F([role], status="observed"))
        e.set("resources", F(cap, status="derived"), key="capacity")
        e.set("past_actions", F([], status="observed"))
        if aid in STANCES:
            e.set("stances", F(STANCES[aid], status="derived"))
        w.entities[aid] = e
    w.network.add("Donald_Trump", "communicates_with", "Vladimir_Putin")
    w.network.add("Donald_Trump", "communicates_with", "Volodymyr_Zelenskyy")
    w.network.add("Vladimir_Putin", "communicates_with", "Volodymyr_Zelenskyy")
    for i, (day, text) in enumerate(EVIDENCE):
        ts = float(calendar.timegm(_time.strptime(day, "%Y-%m-%d")))
        w.information.publish(InformationItem(f"ev_{i}", text, source="public_reporting",
                                              created_at=ts))
        for aid in w.entities:
            w.information.expose(aid, f"ev_{i}", ts)
    register_quantity_type("pathway_progress", units="process_state")
    register_quantity_type("mode_progress", units="process_state")
    for name, v in (("pathway_progress:cooperative_agreement", 0.30),
                    ("pathway_progress:unilateral_action", 0.50),
                    ("mode_progress:unilateral_action:russian_victory", 0.50),
                    ("mode_progress:unilateral_action:ukrainian_victory", 0.50)):
        w.quantities[name] = Quantity(name=name, qtype=name.split(":", 1)[0], value=v,
                                      timestamp=AS_OF)
    return w


def run_mode_d(llm, hypo_llm):
    cfg = QualitativeConfig(llm=llm, hypothesis_llm=hypo_llm, n_hypotheses=3,
                            max_llm_calls=8 + 4 * N_BRANCHES * ROUNDS)
    rt = QualitativeActorPolicyRuntime(QualitativeDecisionEngine(cfg),
                                       mode="persistent_qualitative_llm_policy")
    worlds = [build_world(f"b{i:03d}") for i in range(N_BRANCHES)]
    log = []
    for rnd in range(ROUNDS):
        for actor_id, decision in DECISIONS.items():
            for i, w in enumerate(worlds):
                w.clock.advance_to(AS_OF + (rnd + 1) * 7 * 86400.0)
                sel, post, tr = rt.decide(None, [w], actor_id,
                                          decision=dict(decision),
                                          seed=1000 * rnd + 10 * i)
                rt.execute(w, sel, post, tr, seed=1000 * rnd + 10 * i)
                q = post.provenance["qualitative"]
                log.append({"round": rnd + 1, "branch": w.branch_id, "actor": actor_id,
                            "hypothesis": q.get("hypothesis_id"),
                            "chosen": sel.action_name,
                            "resolution": q.get("resolution"),
                            "novel_unmodeled": q.get("novel_action_unmodeled"),
                            "decision_source": q.get("decision_source"),
                            "summary": q.get("decision_summary", "")[:160],
                            "cooperative_progress": round(float(
                                w.quantities["pathway_progress:cooperative_agreement"].value), 3)})
                print(f"r{rnd + 1} {w.branch_id} {actor_id[:16]:16s} "
                      f"[{q.get('hypothesis_id', '')[:28]:28s}] -> {sel.action_name:14s} "
                      f"coop={log[-1]['cooperative_progress']}", flush=True)
    agg = aggregate_actor_decisions(rt.decision_records)
    states = {aid: (load_actor_state(worlds[0], aid).as_dict()
                    if load_actor_state(worlds[0], aid) else None)
              for aid in DECISIONS}
    return {"decision_log": log, "aggregated": {
                aid: {k: v for k, v in agg[aid].items() if k != "rows"} for aid in agg},
            "example_final_state_branch0": states,
            "llm_calls": rt.engine.calls_used()}


def run_numeric_arm():
    rt = ActorPolicyRuntime()
    out = {}
    for actor_id, decision in DECISIONS.items():
        w = build_world("num")
        _, post, tr = rt.decide(None, [w], actor_id, decision=dict(decision), seed=7)
        name_of = {a["action_id"]: a["action_name"] for a in tr.candidate_actions}
        dist = {}
        for aid_, p in post.action_probabilities.items():
            dist[name_of.get(aid_, aid_)] = round(dist.get(name_of.get(aid_, aid_), 0.0) + p, 4)
        out[actor_id] = dict(sorted(dist.items(), key=lambda kv: -kv[1]))
    return out


def main():
    from swm.api.deepseek_backend import deepseek_chat_fn
    llm = deepseek_chat_fn(temperature=0.9, max_tokens=2000)
    hypo = deepseek_chat_fn(temperature=0.8, max_tokens=2000)
    t0 = _time.time()
    d = run_mode_d(llm, hypo)
    d["numeric_arm_same_scenario"] = run_numeric_arm()
    d["wall_s"] = round(_time.time() - t0, 1)
    d["schema_version"] = "qualitative.demo.geopolitical.v1"
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "qualitative_demo_geopolitical.json"
    path.write_text(json.dumps(d, indent=1, default=str))
    print("\nAGGREGATED (counted across branches):")
    for aid, row in d["aggregated"].items():
        print(f"  {aid}: {json.dumps(row['raw_qualitative_simulation_distribution'])} "
              f"[{row['calibration_status']}]")
    print("NUMERIC ARM (same scenario):")
    for aid, dist in d["numeric_arm_same_scenario"].items():
        print(f"  {aid}: {json.dumps(dist)}")
    print(f"llm_calls={d['llm_calls']} wall={d['wall_s']}s\nwrote {path}")


if __name__ == "__main__":
    main()
