"""Three-arm architectural benchmark: scalar/pathway vs one-hop vs full recursive
actor-mediated execution — the SAME scenario, the SAME production operators, the SAME particle
set; only the propagation regime differs.

    scalar     SWM_ACTOR_PROPAGATION=off  — legacy ontology pathway coefficients + the narrow
               single-target reaction path (the pre-phase behavior, stamped)
    one_hop    depth budget 1             — recipients react once; their reactions propagate
               no further
    recursive  default (depth 4)          — full cascade through the canonical queue

This is an ARCHITECTURE benchmark: it demonstrates that the three regimes produce genuinely
different causal structure (who acted, what moved the process, what the institution aggregated)
under identical inputs, and reports cost. It does NOT claim predictive superiority — that
requires resolved real-world outcomes (see the sealed external-benchmark track).

Usage:
    PYTHONPATH=. python experiments/actor_mediated_three_arm.py            # scripted mock actors
    PYTHONPATH=. python experiments/actor_mediated_three_arm.py --real    # live LLM actors
Writes experiments/results/actor_mediated/three_arm_report.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swm.world_model_v2.actor_propagation import PropagationBudget, SemanticPropagationEngine
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.init_state import InitialStateModel
from swm.world_model_v2.institutions import Rule, RuleSystem
from swm.world_model_v2.joint_world import JointWorldHypothesizer, attach_joint_hypotheses
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase4_execution import ProductionActorPolicyOperator
from swm.world_model_v2.qualitative_actor import (
    QualitativeActorPolicyRuntime, QualitativeConfig, QualitativeDecisionEngine,
    aggregate_actor_decisions,
)
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import InstitutionalVoteOperator

T0 = 1_752_800_000.0            # 2026-07-18
OUT = Path("experiments/results/actor_mediated")

SITUATION = ("The leader must publicly state whether the bloc holds its negotiating position "
             "or concedes; three members then decide their own public stances; the bloc's "
             "council votes five days later.")


def scripted_llm(prompt: str) -> str:
    """Deterministic mock actors for the offline arm comparison (hypothesis-sensitive)."""
    if "You ARE leader_a" in prompt:
        d = {"act_or_wait": "act", "chosen_action": "hold_position", "target": "",
             "observability": "public", "intended_effect": "We stay the course — no concessions."}
    elif "You ARE member_" in prompt:
        me = next(m for m in ("member_b", "member_c", "member_d") if f"You ARE {m}" in prompt)
        doubting = ("private_doubt" in prompt or "depleted" in prompt)
        observed_defection = "public opposition" in prompt or "break with" in prompt
        if doubting and (observed_defection or me == "member_b"):
            d = {"act_or_wait": "act", "chosen_action": "oppose", "target": "bloc_council",
                 "observability": "public", "intended_effect": "publicly break with the leader"}
        else:
            d = {"act_or_wait": "act", "chosen_action": "support", "target": "bloc_council",
                 "observability": "public", "intended_effect": "back the leader"}
    else:
        d = {"act_or_wait": "wait", "chosen_action": "wait", "target": "",
             "observability": "private", "intended_effect": ""}
    return json.dumps({"schema_version": "qualitative.actor.v1",
                       "decision": {"timing": "immediate", **d},
                       "decision_summary": d["intended_effect"],
                       "novel_action_proposal": {"present": False},
                       "situation_interpretation": {"what_changed": "the statement",
                                                    "why_it_matters": "coalition unity"},
                       "actor_state_update": {"important_memories": ["the statement"]}})


def base_world() -> WorldState:
    w = WorldState("w", "root", SimulationClock(T0, T0), network=RelationGraph(),
                   information=InformationLedger())
    for aid in ("leader_a", "member_b", "member_c", "member_d"):
        e = Entity(aid)
        e.set("roles", F(["bloc member"], status="observed"))
        e.set("past_actions", F([], status="observed"))
        w.entities[aid] = e
        if aid != "leader_a":
            w.network.add("leader_a", "influences", aid)
    w.network.add("member_b", "communicates_with", "member_c")
    w.institutions["bloc_council"] = RuleSystem("bloc_council", [
        Rule("bloc_council:0", "decision_right",
             {"holders": ["member_b", "member_c", "member_d"],
              "actions": ["support", "oppose"]})])
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    register_quantity_type("pathway_progress", units="process_state")
    w.quantities["pathway_progress:cooperative_agreement"] = Quantity(
        name="pathway_progress:cooperative_agreement", qtype="pathway_progress", value=0.5,
        timestamp=T0)
    return w


def build_queue(_world) -> EventQueue:
    q = EventQueue(horizon_ts=T0 + 30 * 86400)
    q.schedule(Event(ts=T0 + 3600, etype="decision_opportunity", participants=["leader_a"],
                     payload={"situation": SITUATION,
                              "candidate_actions": [
                                  {"name": "hold_position",
                                   "observability": {"default": "public"},
                                   "parameters": {"content": "We stay the course. There will "
                                                             "be no concessions."}},
                                  {"name": "concede",
                                   "observability": {"default": "public"}},
                                  {"name": "wait"}]},
                     source="scheduled"))
    q.schedule(Event(ts=T0 + 5 * 86400, etype="collective_vote",
                     participants=["member_b", "member_c", "member_d"],
                     payload={"threshold": 0.5, "outcome_var": "bloc_holds"},
                     source="scheduled"))
    return q


def run_arm(arm: str, *, llm, n_particles: int, seed: int = 11) -> dict:
    budget = {"scalar": None,
              "one_hop": PropagationBudget.one_hop(),
              "recursive": PropagationBudget()}[arm]
    t0 = time.time()
    init = InitialStateModel(base_world=base_world())
    attach_joint_hypotheses(init, JointWorldHypothesizer(None, k=3).generate(
        question="does the bloc hold?"))
    worlds = init.sample_particles(n_particles, seed=seed)
    cfg = QualitativeConfig(llm=llm, llm_hypotheses=False, n_hypotheses=3,
                            max_llm_calls=40 * n_particles)
    engine = QualitativeDecisionEngine(cfg)
    prop = SemanticPropagationEngine(budget=budget) if budget is not None else None
    runtime = QualitativeActorPolicyRuntime(engine, mode="persistent_qualitative_llm_policy",
                                            **({"propagation": prop} if prop else {}))
    op = ProductionActorPolicyOperator(runtime=runtime)
    ops = [op, InstitutionalVoteOperator()]
    eng = RolloutEngine(operators=ops)
    holds, cascades, demotions, reconsiderations = 0, [], 0, 0
    for i, w in enumerate(worlds):
        if arm == "scalar":
            w.uncertainty_meta["actor_propagation"] = False
        eng.run_branch(w, build_queue(w), seed=seed * 7919 + i, max_events=120)
        q = w.quantities.get("bloc_holds")
        holds += 1 if (q is not None and bool(q.value)) else 0
        c = w.uncertainty_meta.get("event_cascade") or {}
        cascades.append({"scheduled": c.get("scheduled", 0),
                         "depth": c.get("max_depth_reached", 0),
                         "quiescence": c.get("quiescence", "")})
        demotions += len(w.uncertainty_meta.get("demoted_scalar_writes") or [])
        reconsiderations += c.get("scheduled", 0)
    dists = aggregate_actor_decisions(runtime.decision_records)
    return {
        "arm": arm, "n_particles": n_particles,
        "p_bloc_holds": round(holds / n_particles, 4),
        "n_reconsiderations": reconsiderations,
        "max_depth": max((c["depth"] for c in cascades), default=0),
        "n_demoted_scalar_writes": demotions,
        "llm_calls": engine.calls_used(),
        "wall_s": round(time.time() - t0, 1),
        "actor_distributions": {aid: {"raw": d["raw_qualitative_simulation_distribution"],
                                      "n": d["n_qualitative_branches"],
                                      "calibration": d["calibration_status"]}
                                for aid, d in dists.items()},
        "quiescence_reasons": sorted({c["quiescence"] for c in cascades if c["quiescence"]}),
    }


def main():
    real = "--real" in sys.argv
    n = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--n=")), "12" if not real
                 else "6"))
    if real:
        from swm.api.deepseek_backend import deepseek_chat_fn
        llm = deepseek_chat_fn(max_tokens=1200)
    else:
        llm = scripted_llm
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "mode": "real_llm" if real else "scripted_mock",
              "scenario": SITUATION,
              "claim_scope": "architectural difference under identical inputs — NOT a "
                             "predictive-accuracy claim (no resolved outcome exists for a "
                             "synthetic scenario)",
              "arms": {}}
    for arm in ("scalar", "one_hop", "recursive"):
        print(f"== arm {arm} ==", flush=True)
        report["arms"][arm] = run_arm(arm, llm=llm, n_particles=n)
        print(json.dumps({k: v for k, v in report["arms"][arm].items()
                          if k != "actor_distributions"}, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_real" if real else ""
    path = OUT / f"three_arm_report{suffix}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
