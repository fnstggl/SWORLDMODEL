"""Single-individual reaction simulation — the same qualitative actor architecture, focused on
one person.

Route for questions like "how will this person react if I skip dinner?", "how will my manager
interpret this message?", "will this customer accept the offer?". The target individual is
AUTOMATICALLY Tier 1 (reaction_is_the_question) — no formal stances, institutional capacity,
network degree, or world-model goals are required. The system builds several qualitative
hidden-state hypotheses for the person from the supplied relationship history, shows them the
exact stimulus as they would experience it, lets the LLM inhabit each hypothesis to interpret,
react internally, and choose an observable response, then aggregates the independently chosen
responses across particles and samples into a raw empirical distribution (externally calibrated
where history exists; labeled ``unvalidated`` otherwise). This is not a one-off prompt: it runs
`QualitativeActorPolicyRuntime` on a real (miniature) world through the standard decide/execute
path, so information boundaries, feasibility, execution, and provenance all apply."""
from __future__ import annotations

import time as _time

from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.qualitative_actor import (
    ActionClusterer, ActorPolicyCalibrator, QualitativeActorPolicyRuntime, QualitativeConfig,
    QualitativeDecisionEngine, aggregate_actor_decisions,
)
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

#: default observable-response menu: the messaging slice of the shared action ontology
DEFAULT_RESPONSE_ACTIONS = ("reply_now", "reply_later", "acknowledge", "clarify", "ignore")


def _mini_world(person_id: str, counterpart_id: str, context: dict, stimulus: str,
                channel: str, now: float, branch_id: str) -> WorldState:
    w = WorldState("individual", branch_id, SimulationClock(now, now),
                   network=RelationGraph(), information=InformationLedger())
    person = Entity(person_id)
    person.set("roles", F([str(context.get("role", "person"))], status="observed"))
    if context.get("goals"):
        person.set("goals", F([str(g) for g in context["goals"]], status="inferred"))
    person.set("memory", F([str(m)[:300] for m in (context.get("history") or [])][:12],
                           status="observed"))
    person.set("past_actions", F([], status="observed"))
    counterpart = Entity(counterpart_id)
    counterpart.set("roles", F([str(context.get("your_role", "counterpart"))], status="observed"))
    w.entities = {person_id: person, counterpart_id: counterpart}
    w.network.add(counterpart_id, "communicates_with", person_id)
    # relationship history reaches the person as their OWN observed information (leakage-safe)
    for i, item in enumerate((context.get("history") or [])[:12]):
        iid = f"history_{i}"
        w.information.publish(InformationItem(iid, str(item)[:400],
                                              source=str(context.get("relationship",
                                                                     counterpart_id)),
                                              created_at=now - 86400.0 * (len(context.get("history") or []) - i)))
        w.information.expose(person_id, iid, now - 86400.0 * (len(context.get("history") or []) - i))
    w.information.publish(InformationItem("stimulus", str(stimulus)[:800],
                                          kind=channel or "message",
                                          source=counterpart_id, created_at=now))
    w.information.expose(person_id, "stimulus", now)
    return w


def simulate_individual_reaction(*, person_id: str, stimulus: str, context: dict | None = None,
                                 llm=None, counterpart_id: str = "you", channel: str = "message",
                                 n_hypotheses: int = 3, samples_per_hypothesis: int = 2,
                                 response_actions=DEFAULT_RESPONSE_ACTIONS, seed: int = 0,
                                 as_of: float | None = None, config: QualitativeConfig | None = None,
                                 calibrator: ActorPolicyCalibrator | None = None,
                                 scenario_schema=None) -> dict:
    """Simulate one person's reaction to one exact stimulus.

    ``context`` may supply: role, your_role, relationship (how the person labels the
    counterpart), history (list of prior interactions, oldest first), goals. Returns the full
    artifact: per-sample rows (hypothesis inhabited, interpretation, internal reaction,
    observable response, novel/unmodeled flags), the raw empirical response distribution, and
    the calibrated-or-unvalidated distribution."""
    context = context or {}
    now = float(as_of if as_of is not None else _time.time())
    cfg = config or QualitativeConfig(llm=llm, n_hypotheses=n_hypotheses,
                                      max_llm_calls=4 * n_hypotheses * samples_per_hypothesis)
    cfg.persistent = True
    engine = QualitativeDecisionEngine(cfg)
    runtime = QualitativeActorPolicyRuntime(
        engine, mode="persistent_qualitative_llm_policy",
        tiers={person_id: {"tier": 1, "reasons": ["reaction_is_the_question"],
                           "selector": "individual-mode"}})
    situation = (f"You just received this via {channel} from "
                 f"{context.get('relationship', counterpart_id)}: \"{str(stimulus)[:500]}\"")
    decision = {"situation": situation,
                "candidate_actions": [{"name": a, "target": counterpart_id}
                                      for a in response_actions]}
    outcomes, total = [], max(1, n_hypotheses) * max(1, samples_per_hypothesis)
    for i in range(total):
        world = _mini_world(person_id, counterpart_id, context, stimulus, channel, now,
                            branch_id=f"b{i:03d}")
        if scenario_schema is not None:
            # generated actor-mediated mode for the individual route: the reply becomes a
            # scenario-typed semantic event instead of a fixed-catalog communication; without
            # a schema, the runtime STAMPS its fixed-v1 degradation on the report (never
            # silent)
            import copy as _copy
            world.scenario_schema = _copy.deepcopy(scenario_schema)
        selected, posterior, trace = runtime.decide(
            None, [world], person_id, decision=dict(decision), seed=seed * 7919 + i)
        runtime.execute(world, selected, posterior, trace, seed=seed * 7919 + i)
        outcomes.append((posterior, trace))
    agg = aggregate_actor_decisions(outcomes, clusterer=ActionClusterer(),
                                    calibrator=calibrator or
                                    ActorPolicyCalibrator.from_file())
    result = agg.get(person_id, {"raw_qualitative_simulation_distribution": {},
                                 "calibrated_distribution": {},
                                 "calibration_status": "unvalidated", "rows": []})
    samples = []
    for posterior, trace in outcomes:
        q = (posterior.provenance or {}).get("qualitative") or {}
        chosen = next((a for a in trace.candidate_actions
                       if a.get("action_id") == trace.sampled_action_id), {})
        samples.append({
            "hypothesis_id": q.get("hypothesis_id", ""),
            "decision_source": q.get("decision_source", "numeric_fallback"),
            "interpretation": q.get("situation_interpretation", {}),
            "internal_reaction": q.get("internal_reaction", ""),
            "observable_response": chosen.get("action_name", ""),
            "target": (chosen.get("target") or {}).get("target_id", ""),
            "decision_summary": q.get("decision_summary", ""),
            "novel_action_unmodeled": bool(q.get("novel_action_unmodeled")),
            "trace_id": trace.trace_id,
        })
    return {
        "schema_version": "individual.reaction.v1",
        "person_id": person_id, "stimulus": str(stimulus)[:800], "channel": channel,
        "n_hypotheses": n_hypotheses, "samples_per_hypothesis": samples_per_hypothesis,
        "samples": samples,
        "raw_qualitative_simulation_distribution":
            result["raw_qualitative_simulation_distribution"],
        "calibrated_distribution": result["calibrated_distribution"],
        "calibration_status": result["calibration_status"],
        "n_excluded_numeric_fallbacks": result.get("n_excluded_numeric_fallbacks", 0),
        "consequence_report": runtime.consequence_report,
        "llm_calls": engine.calls_used(),
        "provenance": {"as_of": now, "tier_rule": "reaction_is_the_question → Tier 1",
                       "runtime": "QualitativeActorPolicyRuntime",
                       "aggregation": "branch-selection counting (cluster-1.0)"},
    }
