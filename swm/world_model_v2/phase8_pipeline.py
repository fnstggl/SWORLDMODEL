"""Phase 8 — the universal persistence entry + shared-world execution + causal-ablation harness.

Two shared-world paths, both through the EXISTING architecture (never a bypass longitudinal predictor):

  1. ``simulate_with_persistence(question, as_of, horizon, ...)`` — the universal path: compile → build the
     WorldState → ingest real history into the event log → replay the sequential filters → materialize the
     posteriors into WorldState fields → run the standard rollout (with the persistence operators enabled)
     → terminal readout. Persistent state reaches execution through the same ``materialize`` / ``rollout``
     machinery every other phase uses.

  2. ``materialize_and_decide(world, actor_id, posteriors, candidate_actions, ...)`` — the actor-decision
     path: materialize persistent posteriors into WorldState, build the ACTOR VIEW with the real
     ``ActorViewBuilder``, and run the real ``ActorPolicyModel`` so persistent state changes the calibrated
     action distribution. This is the "full shared-world longitudinal execution" arm the validation grades.

``run_history_ablation`` runs any of the above twice — full history vs removed/altered — and reports whether
execution actually changed (materialized field, actor view, action distribution, terminal readout). A
component whose ablation changes nothing is flagged ornamental (the anti-scaffolding gate).
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

from swm.world_model_v2.phase8_materialize import build_persistent_view, materialize_persistent_state
from swm.world_model_v2.phase8_persistence import (PersistentStateKey, logit, persistent_features_for_policy,
                                                   sigmoid)


@dataclass
class PersistenceContext:
    """Bundles the durable event log + store + optional episodic memory for one scenario, so the pipeline
    and the experiments share one persistence surface."""
    store: object                            # phase8_service.PersistentStore
    memory_store: object = None              # swm.memory.EpisodicStore (optional)
    actor_map: dict = field(default_factory=dict)


# ------------------------------------------------------------------ shared-world actor-decision path
def materialize_and_decide(world, actor_id, posteriors, *, candidate_actions, policy_model=None,
                           observed_events=None, seed=0, decision=None):
    """Materialize persistent posteriors into ``world``, build the actor view with the REAL
    ``ActorViewBuilder``, and run the REAL ``ActorPolicyModel`` — returning (action_posterior, deltas,
    view). Persistent state reaches the policy exactly through the fields the view projects
    (policy_state / past_actions / relationships / beliefs), so removing history changes the action
    distribution. No LLM, no minted numbers."""
    from swm.world_model_v2.phase4_policy import (ActionSpaceBuilder, ActorPolicyModel, ActorViewBuilder,
                                                  FeasibilityEngine)
    deltas = materialize_persistent_state(world, posteriors, actor_map={})
    views = ActorViewBuilder()
    view = views.build(world, actor_id, observed_events=observed_events)
    dec = {"candidate_actions": candidate_actions, **(decision or {})}
    actions = ActionSpaceBuilder().build(None, world, view, decision=dec)
    feas = FeasibilityEngine()
    feasibility = [[feas.classify(a, view, world) for a in actions]]
    model = policy_model or ActorPolicyModel(parameter_pack=persistence_policy_pack())
    posterior = model.decide([view], actions, feasibility, seed=seed)
    return posterior, deltas, view


def persistence_policy_pack() -> dict:
    """A policy parameter pack that makes the reinforcement/habit families dominant, so the materialized
    engagement latent (``latent_state[phase4_policy_value:engage]``) and action history DRIVE the action
    distribution. All weights are labeled structural priors, not fitted claims — the empirical calibration
    lives in the readout temperature fitted on TRAIN (never minted, never on test)."""
    from swm.world_model_v2.phase4_policy import ActorPolicyModel
    pack = ActorPolicyModel._broad_pack()
    pack = dict(pack)
    pack["pack_id"] = "phase8:persistence-reinforcement:1.0"
    pack["support_grade"] = "transfer_supported"
    pack["policy_family_weights"] = {"reinforcement_learning": 0.6, "habit": 0.25, "random_utility": 0.15}
    pack["precision"] = 1.0
    pack["habit_strength"] = 0.3
    return pack


def engagement_readout(world, actor_id, *, temperature=1.0, floor=1e-4):
    """Terminal readout: P(this actor engages) = the materialized engagement-propensity field, optionally
    temperature-calibrated (a single scalar fitted on TRAIN). Reads the WorldState field the Phase-8 filter
    materialized — the number is produced by a sequential filter over ingested history, not by an LLM or a
    bypass regressor. This is the shared-world terminal projection for the persistence-engagement task."""
    from swm.world_model_v2.state import StateField
    ent = world.entities.get(actor_id)
    if ent is None:
        return 0.5
    latent = ent.get("latent_state") or {}
    sf = latent.get("phase4_policy_value:engage") if isinstance(latent, dict) else None
    p = float(sf.value) if isinstance(sf, StateField) and isinstance(sf.value, (int, float)) else 0.5
    p = min(1 - floor, max(floor, p))
    if temperature != 1.0:
        p = sigmoid(temperature * logit(p))
    return p


# ------------------------------------------------------------------ causal ablation harness (Part 17)
def run_history_ablation(build_and_execute, *, ablations=("full", "no_history", "last_event_only")):
    """Run ``build_and_execute(mode)`` for each ablation mode and report whether execution changed. The
    callable returns a dict with any of: {materialized_value, action_probabilities, terminal, view_hash}.
    Returns per-mode outputs plus a causal-relevance verdict (did removing history change execution?)."""
    out = {}
    for mode in ablations:
        out[mode] = build_and_execute(mode)
    base = out.get("full", {})

    def _delta(other):
        d = {}
        if "materialized_value" in base and "materialized_value" in other:
            d["materialized_value_delta"] = round(float(base["materialized_value"])
                                                  - float(other["materialized_value"]), 6)
        if "terminal" in base and "terminal" in other:
            d["terminal_delta"] = round(float(base["terminal"]) - float(other["terminal"]), 6)
        if "action_probabilities" in base and "action_probabilities" in other:
            keys = set(base["action_probabilities"]) | set(other["action_probabilities"])
            d["action_prob_l1"] = round(sum(abs(base["action_probabilities"].get(k, 0.0)
                                                - other["action_probabilities"].get(k, 0.0)) for k in keys), 6)
        if "view_hash" in base and "view_hash" in other:
            d["view_changed"] = base["view_hash"] != other["view_hash"]
        return d

    diffs = {mode: _delta(out[mode]) for mode in out if mode != "full"}
    changed = any(any(abs(v) > 1e-9 if isinstance(v, (int, float)) else bool(v) for v in d.values())
                  for d in diffs.values())
    return {"modes": out, "diffs": diffs, "history_changes_execution": changed,
            "verdict": ("history is causally consumed — removing it changes execution"
                        if changed else "ORNAMENTAL: removing history does not change execution")}


# ------------------------------------------------------------------ THE canonical persistence-aware run
def run_with_persistence(question, plan, *, llm=None, context=None, actor_history=None, family_ids=None,
                         intervention="", t0=None, n_particles=None, seed=0, calibrator=None, cal_key=""):
    """The single canonical persistence-aware run, shared by ``pipeline.simulate(persistence=…)`` and the
    compatibility wrapper ``simulate_with_persistence``. Given a compiled ``plan``, it:

      resolve identity → load prior checkpoint + watermark → ingest leakage-safe new history → replay
      sequential filters → SELECT causally-relevant families (block only quarantined/incompatible) →
      materialize into WorldState → expose actor-visible memory → rollout (persistence operators enabled) →
      persist feedback → commit a new versioned checkpoint → attach lineage/versions/support/limitations.

    No-abstention preserved. When no history/checkpoint exists it runs the ordinary path with broad priors."""
    from swm.world_model_v2.materialize import (build_world, check_readout_binding, operators_from_plan,
                                                queue_builder_from_plan)
    from swm.world_model_v2.init_state import InitialStateModel
    from swm.world_model_v2.rollout import WorldModelV2Run
    from swm.world_model_v2.pipeline import result_from_run
    from swm.world_model_v2.phase8_runtime import (family_runtime_manifest, select_families,
                                                   support_grade_effect)
    t0 = t0 if t0 is not None else _time.time()
    base = build_world(plan, evidence_hash=(plan.provenance or {}).get("evidence_bundle_hash", ""))
    family_ids = list(family_ids or ["engagement_propensity"])
    meta = {"materialized": 0, "history_events": 0, "checkpoint_loaded": False,
            "checkpoint_committed": False, "families_selected": [], "families_blocked": []}
    materialized, family_manifests = [], []
    prior_watermark = ""

    if context is not None and actor_history:
        from swm.world_model_v2.phase8_events import PersistentEvent
        # 1) resolve scenario/actor identity + load prior checkpoint watermark
        scenario = context.store.scenario_id
        prior_cp = None
        try:
            prior_cp = context.store.load_latest_checkpoint(base.clock.as_of)
        except Exception:
            prior_cp = None
        if prior_cp is not None:
            meta["checkpoint_loaded"] = True
            prior_watermark = prior_cp.event_watermark
        # 2) ingest new leakage-safe history (observed_time ≤ as_of enforced by the log's query)
        for eid, events in actor_history.items():
            for ev in events:
                context.store.log.append(PersistentEvent(
                    world_id=base.world_id, scenario_id=scenario,
                    event_type=ev.get("event_type", "passive_exposure"),
                    event_time=float(ev.get("event_time", 0.0)), actor_ids=(eid,),
                    observed_time=float(ev.get("observed_time", ev.get("event_time", 0.0))),
                    outcome=ev.get("outcome")))
                meta["history_events"] += 1
        # 3) SELECT causally-relevant families (block only quarantined/incompatible)
        selections = select_families(family_ids)
        selected_ids = [s.variable_id for s in selections if s.selected]
        meta["families_selected"] = selected_ids
        meta["families_blocked"] = [s.variable_id for s in selections if not s.selected]
        # 4) replay filters → posteriors, for the selected families over the present actors
        keys = [PersistentStateKey(base.world_id, scenario, "actor", eid, vid)
                for eid in actor_history for vid in selected_ids
                if get_persistent_variable_scope(vid) == "actor"]
        posteriors = context.store.replay(base.clock.as_of, variable_keys=keys) if keys else {}
        posteriors = {k: v for k, v in posteriors.items() if v.key.entity_id in base.entities}
        # 5) materialize into the base world
        materialized = materialize_persistent_state(base, list(posteriors.values()))
        meta["materialized"] = len(materialized)
        # 6) expose actor-visible memory (leakage-safe recall) into entity.memory
        _expose_actor_memory(base, context, actor_history, base.clock.as_of)
        # 7) family manifests + support-grade effect
        by_var = {}
        for p in posteriors.values():
            by_var.setdefault(p.variable_id, p)
        for s in selections:
            family_manifests.append(family_runtime_manifest(
                s.variable_id, posterior=by_var.get(s.variable_id)))
        load_bearing = [s.variable_id for s in selections if s.selected and by_var.get(s.variable_id)]
        grade_effect = support_grade_effect(plan.support_grade, selections, load_bearing_ids=load_bearing)
        meta["support_grade_effect"] = grade_effect

    check_readout_binding(plan, base)
    ops, _rej = operators_from_plan(plan, llm=llm)
    import swm.world_model_v2.phase8_transitions as _p8t
    ops = ops + [_p8t.PersistenceUpdateOperator(), _p8t.MemoryConsolidationOperator()]
    init = InitialStateModel(base_world=base, latents=list(plan.latents))
    npart = n_particles or plan.compute_plan.get("n_particles", 30)
    run = WorldModelV2Run(initial=init, queue_builder=queue_builder_from_plan(plan),
                          operators=ops, contract=plan.outcome_contract, n_particles=npart)
    result, branches = run.run(seed=seed)
    result["omissions"] = list(getattr(base, "omissions", []))
    from swm.world_model_v2.materialize import attach_actor_decision_distributions
    attach_actor_decision_distributions(ops, result)
    res = result_from_run(question, plan, result, branches, intervention=intervention, t0=t0,
                          calibrator=calibrator, cal_key=cal_key)
    if result.get("actor_decision_distributions"):
        res.provenance = {**(res.provenance or {}),
                          "actor_decision_distributions": result["actor_decision_distributions"],
                          "actor_policy_mode": result.get("actor_policy_mode", "")}

    # 8) persist feedback + commit a new versioned checkpoint (advance lineage) + attach support effect
    if context is not None and actor_history:
        try:
            new_cp = context.store.checkpoint(base.clock.as_of,
                                              variable_keys=[PersistentStateKey(base.world_id,
                                                             context.store.scenario_id, "actor", eid,
                                                             "engagement_propensity")
                                                             for eid in actor_history])
            if getattr(context.store.log, "backend", None) is not None:
                context.store.commit_checkpoint(new_cp)
                meta["checkpoint_committed"] = True
            meta["checkpoint_watermark"] = new_cp.event_watermark
            meta["prior_watermark"] = prior_watermark
            meta["lineage_advanced"] = new_cp.event_watermark != prior_watermark
        except Exception as e:  # noqa: BLE001 — checkpoint commit is best-effort; the log stays the truth
            meta["checkpoint_error"] = str(e)[:120]
        eff = meta.get("support_grade_effect", {})
        if eff.get("support_grade") and eff["support_grade"] != res.support_grade:
            res.support_grade = eff["support_grade"]           # experimental load-bearing family lowers grade
            res.limitations = list(res.limitations) + list(eff.get("limitations", []))
    res.provenance = {**(res.provenance or {}), "phase8": meta,
                      "phase8_family_manifests": family_manifests[:12],
                      "persistent_deltas": [d.as_dict() for d in materialized][:10]}
    return res, {"plan_hash": plan.plan_hash(), "materialized": materialized, "persistence_meta": meta,
                 "family_manifests": family_manifests}


def get_persistent_variable_scope(vid: str) -> str:
    from swm.world_model_v2.phase8_persistence import get_persistent_variable
    try:
        return get_persistent_variable(vid).scope
    except KeyError:
        return "actor"


def _expose_actor_memory(world, context, actor_history, as_of):
    """Materialize ONLY actor-recallable memories into each actor's ``memory`` field (leakage-safe). The
    omniscient event log keeps everything; the actor gets a probabilistic, recency/salience-weighted subset
    via the episodic store — never perfect recall (Part 8)."""
    from swm.world_model_v2.state import F
    store = getattr(context, "memory_store", None)
    for eid in actor_history:
        if eid not in world.entities:
            continue
        recalled = []
        if store is not None and hasattr(store, "retrieve"):
            try:
                hits = store.retrieve(eid, "", as_of=as_of, k=8)
                recalled = [{"at": getattr(h.get("episode"), "timestamp", as_of),
                             "text": getattr(h.get("episode"), "text", ""),
                             "salience": round(h.get("score", 0.5), 3), "recalled": True}
                            for h in hits]
            except Exception:
                recalled = []
        if recalled:
            world.entities[eid].set("memory", F(recalled, status="derived", method="phase8_actor_recall",
                                                updated_at=as_of))


# ------------------------------------------------------------------ compatibility wrapper (compiles first)
def simulate_with_persistence(question: str, *, llm=None, as_of: str, horizon: str, context=None,
                              actor_history=None, family_ids=None, intervention: str = "", seed: int = 0,
                              n_particles=None):
    """Compatibility wrapper preserved as an internal API: compiles the question then delegates to the
    canonical ``run_with_persistence``. Production callers should prefer ``pipeline.simulate(persistence=…)``
    — there is now ONE canonical execution path (this just compiles first)."""
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.result import (ClarificationRequired, CompilerExecutionError, SimulationResult)
    t0 = _time.time()
    try:
        plan = compile_world(question, llm=llm, evidence="", as_of=as_of, horizon=horizon,
                             intervention=intervention, seed=seed)
    except ClarificationRequired as e:
        return SimulationResult(question=question, simulation_status="clarification_required",
                                clarification_reason=str(e)[:200], latency_s=round(_time.time() - t0, 3)), {}
    except CompilerExecutionError as e:
        return SimulationResult(question=question, simulation_status="execution_failed",
                                failure_taxonomy=e.taxonomy, latency_s=round(_time.time() - t0, 3)), {}
    return run_with_persistence(question, plan, llm=llm, context=context, actor_history=actor_history,
                                family_ids=family_ids, intervention=intervention, t0=t0,
                                n_particles=n_particles, seed=seed)
