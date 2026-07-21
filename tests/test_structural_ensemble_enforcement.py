"""Structural-ensemble ROUTE ENFORCEMENT (Sections 24.33–24.42 + Section 23 failure behavior).

AST- and call-spy-based guards (never brittle docstring matching): the production runtime cannot invoke
the single-plan compiler directly; specialized routes (personal reaction, event time, Phase 13, replay)
cannot bypass the ensemble; failure modes are loud."""
from __future__ import annotations

import ast
import inspect
import json

import pytest

import swm.world_model_v2.unified_runtime as U
import swm.world_model_v2.structural_runtime as SR
from swm.world_model_v2.unified_runtime import simulate_world as _simulate_world_default
import functools
# These tests pin the FULL-FIDELITY pipeline (PR-#127 semantics). Since the §25 default
# switch, the bare entrypoint serves lean_adaptive, so the research-grade profile is
# selected EXPLICITLY here — same pin, same behavior, now by name.
simulate_world = functools.partial(_simulate_world_default, execution_profile="full_fidelity")
from tests.test_structural_ensemble import (EnsembleLLM, FrozenBundle, HERMETIC, OMISSION_QUIET,
                                            critic_ok, decomp_payload, four_way_llm, recon_payload,
                                            run_default)


# ------------------------------------------------------------------ AST enforcement
def _calls_in(fn) -> set:
    """Names of every function called inside fn (AST walk — no string matching)."""
    tree = ast.parse(inspect.getsource(fn))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def test_ast_default_dispatcher_never_calls_single_plan_compiler():
    calls = _calls_in(U.simulate_world)
    assert "compile_world" not in calls, \
        "the canonical entry must not invoke the single-plan compiler directly"
    assert "simulate_structural_ensemble" in calls


def test_ast_ensemble_runtime_compiles_only_through_the_ensemble_compiler():
    for fn in (SR.simulate_structural_ensemble, SR._condition_and_pilot_model,
               SR._extend_to_full, SR._finalize_model):
        assert "compile_world" not in _calls_in(fn), \
            f"{fn.__name__} must not invoke the single-plan compiler directly"
    assert {"reconnoiter_structures", "compile_candidates"} <= _calls_in(SR.simulate_structural_ensemble)


def test_ast_single_model_body_reachable_only_behind_explicit_mode_gate():
    tree = ast.parse(inspect.getsource(U.simulate_world))
    gated = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            literals = [c.value for c in ast.walk(node) if isinstance(c, ast.Constant)]
            if "single_structural_model" in literals:
                gated = True
    assert gated, "single_structural_model must be an explicit compared mode, never a default branch"
    src_default = ast.get_source_segment(inspect.getsource(U.simulate_world),
                                         tree.body[0]) or ""
    assert 'policy.get("structural_mode", "ensemble")' in src_default


def test_call_spy_single_plan_compiler_only_invoked_with_a_structural_directive(monkeypatch):
    """Every compile_world call on the DEFAULT path must carry a candidate's structural directive —
    i.e. the single-plan compiler is only ever used as the ensemble's Stage-B backend."""
    import swm.world_model_v2.compiler as C
    real = C.compile_world
    seen = []

    def spy(question, **kw):
        seen.append(kw.get("structural_directive"))
        return real(question, **kw)

    monkeypatch.setattr(C, "compile_world", spy)
    res = run_default()
    assert res.structural_ensemble is not None
    assert seen, "Stage B must actually compile candidates"
    assert all(d for d in seen), \
        f"single-plan compiler invoked WITHOUT a structural directive on the default path: {seen}"


# ------------------------------------------------------------------ 24.33 personal-reaction route
def _individual_llm():
    from tests.test_qualitative_actor import qpayload
    q = json.dumps(qpayload())
    base = EnsembleLLM(
        recon_by_role={
            "actor_relationship": recon_payload("the relationship is warm and attentive",
                                                ["dana"], constraints=["closeness"],
                                                falsifiers=["dana says she is hurt"]),
            "institutional_procedural": recon_payload("work obligations dominate tonight",
                                                      ["dana"], constraints=["deadline"],
                                                      falsifiers=["dana left work early"]),
            "resource_constraint": recon_payload("attention is elsewhere",
                                                 ["dana"], constraints=["attention"],
                                                 falsifiers=["dana read the message twice"]),
            "information_distribution": recon_payload("attention is elsewhere",
                                                      ["dana"], constraints=["attention"],
                                                      falsifiers=["dana read the message twice"]),
        }, decomp_by_model={"unused": decomp_payload(["dana"])})

    class Ind:
        def __init__(self):
            self.inner = base
            self.prompts = base.prompts

        def __call__(self, prompt):
            if "ALTERNATIVE HYPOTHESES" in prompt or "CONSEQUENCE COMPILER" in prompt:
                return (json.dumps([{"op": "record_observation", "note": "s"}])
                        if "CONSEQUENCE" in prompt else "[]")
            if '"selected_action_id"' in prompt or "situation" in prompt.lower():
                return q
            return self.inner(prompt)

    return Ind()


INDIVIDUAL_CTX = {"individual": {"person_id": "dana", "relationship": "partner",
                                 "history": ["we cook together most nights"],
                                 "stimulus": "I have to skip dinner tonight",
                                 "n_hypotheses": 2, "samples_per_hypothesis": 1}}


def test_personal_reaction_route_cannot_bypass_the_ensemble():
    res = simulate_world("How will Dana react if I skip dinner tonight?", as_of="2025-06-01",
                         llm=_individual_llm(), seed=1, user_context=INDIVIDUAL_CTX)
    assert res.provenance["route"] == "individual_reaction_ensemble"
    se = res.structural_ensemble
    assert se is not None and se["n_independent_generation_calls"] >= 3
    frames = [m for m in se["models"] if m["promotion_status"] == "promoted"]
    assert len(frames) >= 2                       # several causal frames of the reaction
    assert se["n_merged"] >= 1                    # duplicate frames conservatively merged
    for mid, sim in se["simulation_manifest"].items():
        assert sim["final_particles"] >= sim["full_budget_required"]  # full per-frame budget


def test_personal_reaction_single_frame_requires_explicit_ablation():
    res = simulate_world("How will Dana react if I skip dinner tonight?", as_of="2025-06-01",
                         llm=_individual_llm(), seed=1, user_context=INDIVIDUAL_CTX,
                         execution_policy={"structural_mode": "single_structural_model"})
    assert res.provenance.get("route") == "individual_reaction"
    assert res.provenance["structural_mode"] == "single_structural_model"
    assert res.structural_ensemble is None


def test_personal_reaction_frames_condition_hypothesis_generation():
    from swm.world_model_v2.qualitative_actor import QualitativeConfig
    cfg = QualitativeConfig(llm=None, llm_hypotheses=False, n_hypotheses=2,
                            structural_frame="attention is elsewhere")
    hypothesizer = cfg.hypothesizer()
    assert hypothesizer.structural_frame == "attention is elsewhere"
    from swm.world_model_v2.qualitative_actor import _fallback_hypotheses
    from swm.world_model_v2.phase4_policy import ActorView
    view = ActorView(schema_version="v1", actor_id="dana", actor_role="person", observed_time=0.0)
    rows = _fallback_hypotheses(view, 2, structural_frame="attention is elsewhere")
    assert all("attention is elsewhere" in r["organizational_pressures"] for r in rows)
    assert all(any("conjecture" in a for a in r["assumptions"]) for r in rows)


# ------------------------------------------------------------------ 24.34 event-time route
def test_event_time_route_cannot_bypass_the_ensemble():
    llm = four_way_llm()
    res = simulate_world("When will the initiative be approved?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=llm, seed=3,
                         execution_policy={"drop_phases": ["phase2_evidence"]})
    se = res.structural_ensemble
    assert se is not None
    assert se["n_independent_generation_calls"] >= 3
    assert se["n_fully_simulated"] >= 1


# ------------------------------------------------------------------ 24.35 Phase 13 default
def test_phase13_cannot_evaluate_actions_inside_only_one_model_by_default():
    from swm.world_model_v2.phase13.api import SingleModelContextError, recommend_action
    from swm.world_model_v2.phase13.contracts import DecisionProblem, Stakeholder, UtilitySpec
    res = run_default()
    problem = DecisionProblem(
        decision_id="d1", decision_maker="avery", authority=["communicate", "gather_information"],
        as_of="2025-06-01",
        utility=UtilitySpec(stakeholders=[Stakeholder(
            "avery", utility_fn=lambda o: float(o.get("quantities", {}).get("outcome", 0.0) or 0.0))]))
    plan = res._ensemble_handle.surviving()[0].executable_plan
    with pytest.raises(SingleModelContextError):
        recommend_action(problem, plan, budget="diagnostic", seed=2, n_particles=4)
    r = recommend_action(problem, res, budget="diagnostic", seed=2, n_particles=6)
    se = r.provenance["structural_ensemble"]
    assert se["n_models_evaluated"] >= 2
    assert set(se["winner_by_model"]) == set(se["per_model_results"])
    assert "ranking_by_model" in se and "minimax_regret_across_models" in se
    assert se["recommendation_stability"] in ("structurally_stable", "mildly_structurally_sensitive",
                                              "materially_structurally_sensitive")
    r2 = recommend_action(problem, plan, budget="diagnostic", seed=2, n_particles=4,
                          allow_single_structural_model=True)
    assert r2.recommendation_kind in ("action", "pareto", "abstain", "gather_information")


# ------------------------------------------------------------------ 24.36 recompilation ancestry
def test_dynamic_recompilation_preserves_model_ancestry():
    bundle = FrozenBundle(claims=[{"claim_id": "c1", "text": "t", "claim_class": "actor_statement",
                                   "publication_time": 1748600000.0}])
    res = simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=four_way_llm(), seed=3,
                         execution_policy={"drop_phases": ["event_time"]}, prebuilt_bundle=bundle)
    se = res.structural_ensemble
    promoted = [m for m in se["models"] if m["promotion_status"] == "promoted"]
    lineages = [tuple(m["plan_lineage"]) for m in promoted]
    assert all(l for l in lineages), "every model keeps its own plan lineage"
    per_model = res.provenance["per_model_provenance"]
    for m in promoted:
        assert per_model[m["model_id"]]["plan_lineage"], "per-model lineage recorded"
    merged = [m for m in se["rejected_and_merged"] if m["status"] == "merged"]
    for m in merged:
        assert m["merge_record"]["merged_into"], "merge ancestry recorded"
    survivors_with_parents = [m for m in se["models"] if m["parent_ids"]]
    assert survivors_with_parents, "the merge survivor records the merged candidate as parent"


# ------------------------------------------------------------------ 24.37 model state isolation
def test_model_specific_world_state_is_isolated():
    res = run_default()
    ens = res._ensemble_handle
    plans = [c.executable_plan for c in ens.surviving() if c.executable_plan is not None]
    assert len({id(p) for p in plans}) == len(plans)
    for i, p in enumerate(plans):
        for q in plans[i + 1:]:
            assert p.scheduled_events is not q.scheduled_events
            assert p.entities is not q.entities
            assert p.provenance is not q.provenance
    ens.validate_integrity()                       # would raise on any shared plan object


# ------------------------------------------------------------------ 24.41 numeric fallback surfaced
def test_numeric_actor_fallback_is_prohibited_from_passing_silently():
    from swm.world_model_v2.phase8_pipeline import _surface_actor_policy_degradation
    from swm.world_model_v2.result import SimulationResult
    res = SimulationResult(question="q", simulation_status="completed",
                           support_grade="empirically_supported")
    _surface_actor_policy_degradation(res, {"degraded": True,
                                            "warning": "tier-1 actor served by numeric_fallback"})
    assert res.support_grade == "exploratory"      # the grade can never stay high through a fallback
    assert any("numeric_fallback" in l for l in res.limitations)


# ------------------------------------------------------------------ Section 23 loud failures
def test_identical_independent_generation_prompts_fail_loudly():
    from swm.world_model_v2.structural_contracts import (EnsembleIntegrityError,
                                                         StructuralModelEnsemble)
    ens = StructuralModelEnsemble(question="q")
    for i in range(3):
        ens.record_generation(role=f"r{i}", prompt_hash="SAME", response_hash=f"resp{i}", ok=True)
    with pytest.raises(EnsembleIntegrityError):
        ens.validate_integrity()


def test_all_generation_calls_reusing_one_response_is_recorded_as_degenerate():
    one = recon_payload("only story", ["dana"])
    llm = EnsembleLLM(recon_by_role={r: one for r in
                                     ("actor_relationship", "institutional_procedural",
                                      "resource_constraint", "information_distribution")},
                      decomp_by_model={"any": decomp_payload(["dana"])})
    res = run_default(llm)
    cert = res.structural_ensemble["convergence_certificate"]
    assert cert["degenerate_backend"] is True      # byte-identical responses to distinct prompts


def test_ceiling_with_open_omission_findings_marks_structurally_underidentified():
    omission = dict(OMISSION_QUIET)
    omission.update({"missing_decisive_actor": "an unmodeled veto holder",
                     "no_further_material_model": False,
                     "proposed_models": [{"causal_thesis": f"missing structure {i}",
                                          "decisive_actors": [f"actor_{i}"],
                                          "why_missing": "gap"} for i in range(9)]})
    llm = four_way_llm(omission=omission)
    res = run_default(llm, policy={**HERMETIC,
                                   "generation_policy": {"soft_ceiling": 5}})
    se = res.structural_ensemble
    assert se["structurally_underidentified"] is True
    assert se["unresolved_alternatives"]
    assert (se["structural_sensitivity"]["classification"] == "structurally_underidentified")
    assert any("structurally underidentified" in l for l in res.limitations)


def test_no_executable_candidate_fails_loudly_not_silently_single():
    class BrokenCompile(EnsembleLLM):
        def __call__(self, prompt):
            if "WORLD-SLICE COMPILER" in prompt:
                raise RuntimeError("compiler backend down")
            return super().__call__(prompt)

    llm = BrokenCompile(recon_by_role=four_way_llm().recon_by_role,
                        decomp_by_model={"x": decomp_payload(["a"])})
    res = run_default(llm)
    assert res.simulation_status == "execution_failed"
    assert res.failure_taxonomy in ("invalid_execution_plan", "unavailable_service")
    assert res.raw_distribution == {}


def test_aggregation_never_loses_model_identity():
    res = run_default()
    se = res.structural_ensemble
    assert set(se["model_distributions"]) == {m["model_id"] for m in se["models"]
                                              if m["promotion_status"] == "promoted"}
    assert res.structural_disagreement is not None
    for mid, d in se["model_distributions"].items():
        assert d, f"model {mid} lost its distribution in aggregation"


def test_post_pilot_collapse_to_one_model_is_marked_incomplete(monkeypatch):
    """Live-run finding (founder_launch forensic): models that die during conditioning/pilot must make
    the ensemble EXECUTION-INCOMPLETE — a single survivor left by execution failures (no generation-time
    convergence certificate) may never read as 'structurally stable'."""
    import swm.world_model_v2.phase8_pipeline as P8
    real_prepare = P8.prepare_persistence_run
    killed = []

    def sabotaged(question, plan, **kw):
        mid = (plan.provenance or {}).get("structural_model_id", "")
        if mid != "m0_actor_relationship":         # every other model dies at pilot preparation
            killed.append(mid)
            raise RuntimeError(f"induced pilot failure for {mid}")
        return real_prepare(question, plan, **kw)

    monkeypatch.setattr(P8, "prepare_persistence_run", sabotaged)
    res = run_default()
    assert killed, "sabotage must have hit at least one model"
    se = res.structural_ensemble
    assert se["n_fully_simulated"] == 1
    assert se["structural_sensitivity"]["classification"] == "ensemble_execution_incomplete"
    assert res.simulation_status == "completed_with_degradation"
    assert any("ensemble execution incomplete" in l for l in res.limitations)
    failed_rows = [m for m in se["rejected_and_merged"] if m["status"] == "failed"]
    assert failed_rows and all("pilot_failed" in m["reason"] for m in failed_rows)
    assert any(v.get("status", "").startswith("failed") for v in se["simulation_manifest"].values())
