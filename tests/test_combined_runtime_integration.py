"""COMBINED-RUNTIME integration proofs for the merged production chain (PRs #119—#122):

    several independently generated structural models (#122)
    → a scenario-native temporal model compiled inside EACH structural model (#120)
    → scenario-generated actions evaluated across those models (#121)
    → every action through the attempt → mechanism → verified-outcome boundary (#119)
    → information becomes available, receives attention, and only then reaches actors
    → actors react through their own persistent simulations
    → results and actions compared across structural models (Phase 13).

Each test here proves a REQUIRED integration property of the combined runtime — not a
property of any single PR in isolation.
"""
import inspect
import json
import sys

import pytest

from swm.world_model_v2.compiler import compile_world
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.phase13 import api as p13
from swm.world_model_v2.phase13.contracts import DecisionProblem
from swm.world_model_v2.unified_runtime import simulate_world as _simulate_world_default
import functools
# These tests pin the FULL-FIDELITY pipeline (PR-#127 semantics). Since the §25 default
# switch, the bare entrypoint serves lean_adaptive, so the research-grade profile is
# selected EXPLICITLY here — same pin, same behavior, now by name.
simulate_world = functools.partial(_simulate_world_default, execution_profile="full_fidelity")

from tests.test_structural_ensemble import (EnsembleLLM, HERMETIC, decomp_payload, four_way_llm,
                                            recon_payload, run_default)
from tests.scenario_fixtures import build_context, council_schema
from tests.test_scenario_action_layer import MAKER, OFFICER, filing_candidate, officer_grants


# ------------------------------------------------------------------ shared mini fixtures
def _council_plan(mid: str, lean: str, seed: int = 4):
    """One structural-model plan for the council scenario: compiled through the real compiler
    (scripted backend), carrying the frozen council scenario schema + mechanism model."""
    llm = EnsembleLLM(recon_by_role={"r": recon_payload("t", [MAKER, OFFICER])},
                      decomp_by_model={mid: decomp_payload([MAKER, OFFICER], lean=lean,
                                                           hyp="h")})
    plan = compile_world("Will the variance be granted?", llm=llm, evidence="",
                         as_of="2023-01-01", horizon="2023-12-31", seed=seed, persist=False)
    plan.scenario_schema = council_schema()
    return plan


def _two_model_ensemble():
    return {"mA": _council_plan("mA", "weak_yes"), "mB": _council_plan("mB", "weak_no")}


def _problem(**kw):
    kw.setdefault("decision_id", "combined1")
    kw.setdefault("decision_maker", MAKER)
    kw.setdefault("authority", ["petitioner"])
    kw.setdefault("horizon", "2023-12-31T00:00:00Z")
    return DecisionProblem(**kw)


# ================================================================== 1. default simulate_world:
#    multiple structural models, EACH with its own compiled temporal model
def test_1_default_run_generates_multiple_models_each_with_distinct_temporal_model():
    res = run_default()
    # §NAP: materially disagreeing models serve per-model conditionals (partially_resolved)
    assert res.simulation_status in ("completed", "completed_with_degradation",
                                     "partially_resolved")
    assert res.provenance["structural_mode"] == "ensemble"
    handle = res._ensemble_handle
    surviving = [c for c in handle.surviving() if c.executable_plan is not None]
    assert len(surviving) >= 2                       # several independent structural models
    tmodels = []
    for c in surviving:
        tm = getattr(c.executable_plan, "temporal_model", None)
        assert tm is not None, f"model {c.model_id} has no compiled temporal model"
        # the temporal model was compiled FOR this structural model, not shared globally
        assert tm.structural_model_id == c.model_id
        assert (c.executable_plan.provenance or {}).get("temporal_model_hash")
        tmodels.append(tm)
    assert len({id(t) for t in tmodels}) == len(tmodels)      # distinct objects, not one global
    # and the per-model lineage recorded the compilation (not a degradation)
    for mid, prov in res.provenance["per_model_provenance"].items():
        lin = prov.get("plan_lineage") or {}
        assert "error" not in (lin.get("temporal_model") or {}), \
            f"temporal compile degraded in {mid}: {lin.get('temporal_model')}"


# ================================================================== 2. no production path
#    reaches deepen_trajectory / periodic reviews / fixed delays / one-second fallbacks
PRODUCTION_MODULES = [
    "swm/world_model_v2/unified_runtime.py", "swm/world_model_v2/structural_runtime.py",
    "swm/world_model_v2/generated_world.py", "swm/world_model_v2/causal_boundary.py",
    "swm/world_model_v2/materialize.py", "swm/world_model_v2/rollout.py",
    "swm/world_model_v2/phase8_pipeline.py", "swm/world_model_v2/phase13/api.py",
    "swm/world_model_v2/phase13/ensemble.py", "swm/world_model_v2/phase13/interventions.py",
    "swm/world_model_v2/phase13/scenario_actions/api.py",
    "swm/world_model_v2/phase13/scenario_actions/execution.py",
]


def test_2_no_production_path_reaches_legacy_scheduling():
    for path in PRODUCTION_MODULES:
        src = open(path, encoding="utf-8").read()
        assert "deepen_trajectory" not in src, path
        assert "legacy_periodic_review_ablation" not in src, path
        assert "background_every_days" not in src, path
        # fixed delivery/reconsideration constants of the pre-temporal runtime
        assert 'default_delay_s"' not in src, path
        assert 'public_delay_s"' not in src, path
        assert "delay_s=1800" not in src and "delay_s=3600" not in src, path
    # the quarantined ablation exists ONLY under its explicit legacy name
    from swm.world_model_v2 import legacy_ablations
    assert hasattr(legacy_ablations, "legacy_periodic_review_ablation")
    # the rollout engine is the event-driven temporal runtime, not a tick loop
    from swm.world_model_v2.rollout import RolloutEngine
    assert "run_branch_temporal" in inspect.getsource(RolloutEngine.run_branch)
    # no universal now+1s policy step: generated plan steps fire at their declared timing or
    # at the decision instant, and dependent steps fire when their dependency completes
    from swm.world_model_v2.phase13.scenario_actions import execution as sx
    src = inspect.getsource(sx)
    assert "now + 1" not in src and "now+1" not in src


# ================================================================== 3. a generated (#121)
#    action inside EVERY (#122) model passes the (#119) attempt/mechanism boundary
def test_3_generated_action_in_every_model_passes_causal_boundary():
    res = p13.evaluate_actions(_problem(), [filing_candidate()], _two_model_ensemble(),
                               goal_text="obtain the variance", seed=1, n_particles=2)
    se = res.provenance["structural_ensemble"]
    assert se.get("per_model_errors") in ({}, None)
    per = se["per_model_results"]
    assert set(per) == {"mA", "mB"}
    for mid, r in per.items():
        prov = r["provenance"]
        # the scenario-generated action layer ran INSIDE this structural model
        assert "scenario_report" in prov, f"{mid} did not route through the generated layer"
        # and its steps executed through the causal boundary: kernel steps fired, the
        # scenario's own mechanisms were invoked, and only mechanism output delivered
        cb = prov.get("causal_consequence_report") or {}
        assert cb.get("steps_fired", 0) >= 1, f"{mid}: no plan step executed"
        assert cb.get("mechanisms_invoked", 0) >= 1, f"{mid}: no mechanism invoked"
        assert cb.get("mechanism_successes", 0) + cb.get("mechanism_failures", 0) \
            + cb.get("mechanism_unresolved", 0) >= 1, f"{mid}: mechanism outcome missing"
        # no external success was ever written directly by the compiler/actor
        assert cb.get("external_successes_written_directly", 0) == 0
        assert cb.get("human_reactions_written_directly", 0) == 0


# ================================================================== 4. failed delivery →
#    no recipient observation, no actor reaction
def test_4_failed_delivery_produces_no_observation_and_no_reaction():
    import swm.world_model_v2.generated_world as gw
    from tests.test_causal_boundary import (T0, grant_world, run_ops, step_all_mechanisms,
                                            PIGEON_NOTE)
    # this branch's hidden state makes the courier channel FAIL the delivery mechanism
    w = grant_world(mailbox="over_quota")
    report = gw.generated_report()
    run_ops(w, [{"op": "emit_semantic_event", "semantic_type_id": "colleague_note_drafted",
                 "exact_content": PIGEON_NOTE, "direct_targets": ["prof_boyd"]}],
            report=report)
    invocations = step_all_mechanisms(w, report)
    assert report["mechanism_failures"] >= 1 and report["actual_deliveries"] == 0
    assert invocations == []                            # no reaction: nothing was observed
    assert not w.information.visible_to("prof_boyd", at=w.clock.now + 10 * 86400)


def test_4b_unverified_attempt_routes_to_nobody():
    """Attempted transmission ≠ delivery: the router refuses unverified events outright."""
    import swm.world_model_v2.generated_world as gw
    from tests.test_causal_boundary import grant_world
    w = grant_world()
    sev = {"event_id": "x1", "semantic_type_id": "colleague_note_drafted",
           "exact_content": "hello", "direct_targets": ["prof_boyd"],
           "intended_visibility": "participants", "source_actor_id": "prof_ada",
           "observability_verified": False}
    assert gw.route_semantic_event(w, sev, gw.generated_report()) == []


# ================================================================== 5. delivered-but-unnoticed
#    message produces NO decision until the attention trigger fires
def test_5_delivered_but_unnoticed_message_produces_no_decision_until_attention():
    import swm.world_model_v2.generated_world as gw
    from tests.test_causal_boundary import grant_world, run_ops, T0
    from swm.world_model_v2 import causal_boundary as cb
    w = grant_world()
    report = gw.generated_report()
    run_ops(w, [{"op": "emit_semantic_event", "semantic_type_id": "colleague_note_drafted",
                 "exact_content": "note", "direct_targets": ["prof_boyd"]}], report=report)
    attempt = w.semantic_log[0]
    op = cb.MechanismRuntimeOperator(report=report)
    router = gw.GeneratedSemanticEventOperator(report=report)
    deliver = gw.GeneratedObservationDeliveryOperator(report=report)
    # drive: mechanism delivers → router emits delivery → delivery records availability
    invocations, attentions = [], []
    for _ in range(4):
        pending = [i for i in w.mechanism_instances.values() if i.status == "pending"]
        if not pending:
            break
        for inst in pending:
            ts = max(w.clock.now, inst.pending_transition_at)
            w.clock.advance_to(ts)
            d, _ = op.run(w, Event(ts=ts, etype="ctrl_mechanism_step",
                                   payload={"instance_id": inst.instance_id}), None)
            for fu in d.follow_up_events:
                if fu["etype"] != "ctrl_semantic_event":
                    continue
                rd, _ = router.run(w, Event(ts=fu["ts"], etype=fu["etype"],
                                            payload=dict(fu["payload"])), None)
                for f2 in rd.follow_up_events:
                    if f2["etype"] == "ctrl_deliver_observation":
                        w.clock.advance_to(max(w.clock.now, f2["ts"]))
                        dd, _ = deliver.run(w, Event(ts=f2["ts"], etype=f2["etype"],
                                                     participants=list(f2["participants"]),
                                                     payload=dict(f2["payload"])), None)
                        for f3 in (dd.follow_up_events or []):
                            (attentions if f3["etype"] == "ctrl_attention"
                             else invocations).append(f3)
    # DELIVERED: the item is available — but NOT observed and NOT decided
    assert report["actual_deliveries"] >= 1
    assert not invocations                        # no direct post-delivery invocation exists
    assert attentions                             # the attention trigger is scheduled instead
    assert not w.information.visible_to("prof_boyd", at=w.clock.now)
    # only the ATTENTION event exposes the bundle and opens the actor's decision
    att = gw.GeneratedAttentionOperator(report=report)
    f = attentions[0]
    w.clock.advance_to(max(w.clock.now, f["ts"]))
    ad, _ = att.run(w, Event(ts=f["ts"], etype=f["etype"], participants=list(f["participants"]),
                             payload=dict(f["payload"])), None)
    assert w.information.visible_to("prof_boyd", at=w.clock.now)
    followups = list((ad.follow_up_events or []))
    assert any(x["etype"] == "ctrl_invoke_actor" for x in followups)
    inv = next(x for x in followups if x["etype"] == "ctrl_invoke_actor")
    trig = inv.get("trigger") or inv["payload"].get("trigger") or {}
    assert trig.get("trigger_type") == "newly_noticed_information"


# ================================================================== 6. Phase 13 compares the
#    SAME action set across ALL surviving structural models
def test_6_phase13_compares_same_actions_across_all_models():
    res = p13.evaluate_actions(_problem(), [filing_candidate()], _two_model_ensemble(),
                               goal_text="obtain the variance", seed=1, n_particles=2)
    se = res.provenance["structural_ensemble"]
    assert se["n_models_evaluated"] == 2
    per = se["per_model_results"]
    evaluated_sets = {mid: {e["action_id"] for e in (r.get("evaluated") or [])
                            if isinstance(e, dict)}
                      for mid, r in per.items()}
    common = set.intersection(*evaluated_sets.values())
    assert "file_petition" in common and "do_nothing" in common
    # the cross-model synthesis actually compared them
    assert "expected_utility_matrix" in se and "winner_by_model" in se
    assert set(se["winner_by_model"]) == {"mA", "mB"}
    assert "minimax_regret_across_models" in se


# ================================================================== 7. normal public APIs use
#    the complete combined runtime — no enable flags anywhere
def test_7_default_public_apis_run_combined_runtime_without_flags():
    # simulate_world: no flag → ensemble + per-model temporal compilation (test 1 proves the
    # temporal part); here: the DEFAULT policy dict carries no enabling switches
    res = run_default()
    assert res.provenance["structural_mode"] == "ensemble"
    # phase13 on the default runtime's own result object: routed across models
    # (extract_ensemble_models accepts the SimulationResult handle directly)
    from swm.world_model_v2.phase13.ensemble import extract_ensemble_models
    models = extract_ensemble_models(res)
    assert len(models) >= 2
    # the generated consequence mode is the resolved default, not opt-in
    from swm.world_model_v2 import semantic_consequences as sc
    assert sc.resolve_consequence_mode() == "generated_actor_mediated_world"
    # and the production operator stack materializes the FULL combined chain by default:
    # plan-step executor + router + delivery + attention + mechanism/scheduled-attempt runtime
    plan = _council_plan("mX", "weak_yes")
    from swm.world_model_v2.materialize import operators_from_plan
    ops, _rej = operators_from_plan(plan, llm=None)
    names = {getattr(o, "name", type(o).__name__) for o in ops}
    for needed in ("scenario_plan_step", "generated_semantic_event_router",
                   "generated_observation_delivery", "generated_attention",
                   "scenario_mechanism_runtime", "scheduled_attempt_runtime"):
        assert any(needed in n for n in names), f"missing {needed} in default ops: {names}"


# ================================================================== 8. legacy + single-model
#    ablations exist ONLY behind deliberately named flags
def test_8_ablations_only_behind_deliberate_flags():
    # (a) single structural model: only via the named execution_policy value
    res = simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=four_way_llm(), seed=3,
                         execution_policy={**HERMETIC,
                                           "structural_mode": "single_structural_model"})
    assert res.provenance["structural_mode"] == "single_structural_model"
    with pytest.raises(ValueError):
        simulate_world("q", as_of="2025-06-01", llm=four_way_llm(),
                       execution_policy={"structural_mode": "just_one_please"})
    # (b) a bare single plan in phase13 requires the explicit ablation flag — even a
    # generated-world plan
    plan = _council_plan("mY", "weak_yes")
    with pytest.raises(p13.SingleModelContextError):
        p13.evaluate_actions(_problem(), [filing_candidate()], plan,
                             goal_text="obtain the variance", seed=1, n_particles=2)
    r_ok = p13.evaluate_actions(_problem(), [filing_candidate()], plan,
                                goal_text="obtain the variance", seed=1, n_particles=2,
                                allow_single_structural_model=True)
    assert "scenario_report" in r_ok.provenance      # the ablation still runs generated
    # (c) fixed-v1 catalog: only via the named mode on a generated context
    wc, rep, _ = (lambda: build_context(council_schema(), [MAKER, OFFICER],
                                        script={OFFICER: officer_grants}, n_particles=2))()
    res_leg = p13.recommend_action(_problem(), wc, seed=1, mode="legacy_fixed_v1")
    assert "scenario_report" not in (res_leg.provenance or {})
    with pytest.raises(ValueError):
        p13.recommend_action(_problem(), wc, mode="fixed_v1_sneaky")
