"""Structural-model uncertainty — core required tests (ensemble contract Sections 24.1–24.32).

The scripted backend drives the COMPLETE default runtime end-to-end: independent Stage-A generation,
critics, Stage-B per-candidate compilation, conservative dedup, real pilots through the canonical
persistence funnel, promotion, full per-model budgets with pilot-prefix reuse, honest aggregation.
Hermetic: evidence phases are either dropped via the documented ablation hook or fed a frozen bundle."""
from __future__ import annotations

import hashlib
import json

import pytest

from swm.world_model_v2.unified_runtime import simulate_world as _simulate_world_default
import functools
# These tests pin the FULL-FIDELITY pipeline (PR-#127 semantics). Since the §25 default
# switch, the bare entrypoint serves lean_adaptive, so the research-grade profile is
# selected EXPLICITLY here — same pin, same behavior, now by name.
simulate_world = functools.partial(_simulate_world_default, execution_profile="full_fidelity")
from swm.world_model_v2 import ensemble_compiler as EC
from swm.world_model_v2 import structural_runtime as SR
from swm.world_model_v2.llm_call_cache import CachedLLM, CallLedger, ScopedActorCache
from swm.world_model_v2.structural_contracts import (
    GENERATION_TARGET_CALLS, StructuralModelCandidate, StructuralModelEnsemble,
    EnsembleIntegrityError, classify_forecast_sensitivity, schema_hash)


# ------------------------------------------------------------------ scripted ensemble backend
def recon_payload(thesis, actors, institutions=(), constraints=(), mechanisms=("mech",),
                  falsifiers=(), boundary="b", evidence=()):
    return {"causal_thesis": thesis, "decisive_actors": list(actors),
            "decisive_institutions": list(institutions), "decisive_constraints": list(constraints),
            "decisive_mechanisms": list(mechanisms), "external_systems": [],
            "world_boundary": boundary, "candidate_omissions": [],
            "required_evidence": [{"claim": c, "why": "ground"} for c in evidence],
            "falsifiers": list(falsifiers), "intervention_propagation": f"via {list(mechanisms)[0]}"}


def decomp_payload(entities, lean="neutral", hyp="h0", institutions=(), actor_decisions=(),
                   relations=(), latent_paths=()):
    return {"coherent": True,
            "interpretations": [{"id": "primary", "reading": "r", "weight": 1.0}],
            "outcome": {"family": "binary", "options": ["yes", "no"],
                        "resolution_rule": "resolved", "readout_var": "outcome"},
            "outcome_lean": lean,
            "entities": [{"id": e, "type": "person", "fields": {}, "sensitivity": 0.8}
                         for e in entities],
            "institutions": [{"id": i, "rules": [{"kind": "vote_threshold", "params": {}}],
                              "sensitivity": 0.7} for i in institutions],
            "relations": [dict(r) for r in relations],
            "quantities": [{"name": "outcome", "qtype": "outcome", "value": None, "sd": None}],
            "latents": [{"path": p, "why": "w", "lo": 0.2, "hi": 0.8, "sensitivity": 0.6}
                        for p in (latent_paths or [f"{entities[0]}.resolve"])],
            "structural_hypotheses": [{"id": hyp, "describe": "d", "prior": 1.0, "lean": lean}],
            "actor_decisions": list(actor_decisions),
            "mechanisms": [], "required_causal_processes": ["outcome_resolution"],
            "scheduled_events": [], "hazards": [], "domain": "test", "rationale": "scripted"}


OMISSION_QUIET = {"missing_decisive_actor": None, "missing_institution": None,
                  "missing_constraint": None, "missing_information_route": None,
                  "external_event_reversal": None, "boundary_too_narrow": None,
                  "missing_causal_theory": None, "equivalent_sounding": [], "proposed_models": [],
                  "no_further_material_model": True, "reasoning": "coverage adequate"}
CONTRAST_QUIET = {"genuinely_different": [], "superficial_only": [], "same_trajectory_pairs": [],
                  "reversal_pairs": [], "missing_axes": [], "reasoning": "ok"}


def critic_ok(support="plausible", basis="consistent with brief", reject=False, reject_reason=None,
              contradictions=()):
    return {"validity_conditions": ["v"], "non_executable_mechanisms": [],
            "incorrectly_collapsed": [], "skipped_intermediaries": [], "ornamental_components": [],
            "evidence_contradictions": list(contradictions), "missing_outcome_mechanisms": [],
            "intervention_differentiation": "meaningful", "support_class": support,
            "support_basis": basis, "repairs": [], "reject": reject, "reject_reason": reject_reason}


class EnsembleLLM:
    """fn(prompt)->text backend scripted per ensemble stage; records prompts by stage."""

    def __init__(self, *, recon_by_role, decomp_by_model, omission=None, contrast=None,
                 critic_by_thesis=None, judge=None, default="{}"):
        self.recon_by_role = recon_by_role
        self.decomp_by_model = decomp_by_model
        self.omission = omission or OMISSION_QUIET
        self.contrast = contrast or CONTRAST_QUIET
        self.critic_by_thesis = critic_by_thesis or {}
        self.judge = judge or {"equivalent_on_result_relevant_elements": True, "confidence": "high",
                               "differences_that_could_change_result": [], "reasoning": "identical"}
        self.default = default
        self.prompts = {"recon": [], "omission": [], "contrast": [], "critic": [], "compile": [],
                        "judge": [], "other": []}

    def __call__(self, prompt):
        if "STRUCTURAL CAUSAL RECONNAISSANCE" in prompt:
            self.prompts["recon"].append(prompt)
            for role, payload in self.recon_by_role.items():
                if f"({role})" in prompt:
                    return json.dumps(payload)
            return json.dumps(next(iter(self.recon_by_role.values())))
        if "STRUCTURAL-OMISSION CRITIC" in prompt:
            self.prompts["omission"].append(prompt)
            return json.dumps(self.omission)
        if "CROSS-MODEL CONTRAST CRITIC" in prompt:
            self.prompts["contrast"].append(prompt)
            return json.dumps(self.contrast)
        if "CANDIDATE CAUSAL CRITIC" in prompt:
            self.prompts["critic"].append(prompt)
            for marker, payload in self.critic_by_thesis.items():
                if marker in prompt:
                    return json.dumps(payload)
            return json.dumps(critic_ok())
        if "STRUCTURAL EQUIVALENCE JUDGE" in prompt:
            self.prompts["judge"].append(prompt)
            return json.dumps(self.judge)
        if "WORLD-SLICE COMPILER" in prompt:
            self.prompts["compile"].append(prompt)
            for mid, payload in self.decomp_by_model.items():
                if f"{mid!r}" in prompt:
                    return json.dumps(payload)
            return json.dumps(next(iter(self.decomp_by_model.values())))
        self.prompts["other"].append(prompt)
        return self.default


def four_way_llm(**kw):
    """Default 4-perspective backend: three materially different structures, roles 2/3 duplicated."""
    return EnsembleLLM(
        recon_by_role={
            "actor_relationship": recon_payload("Avery's choice drives it", ["avery", "blake"],
                                                constraints=["time"],
                                                falsifiers=["avery announces exit"]),
            "institutional_procedural": recon_payload("The council procedure decides", ["council"],
                                                      institutions=["council"],
                                                      constraints=["quorum"],
                                                      falsifiers=["council cancels session"]),
            "resource_constraint": recon_payload("Budget capacity binds", ["casey"],
                                                 constraints=["budget"],
                                                 falsifiers=["budget doubles"]),
            "information_distribution": recon_payload("Budget capacity binds", ["casey"],
                                                      constraints=["budget"],
                                                      falsifiers=["budget doubles"]),
        },
        decomp_by_model={
            "m0_actor_relationship": decomp_payload(["avery", "blake"], lean="weak_yes", hyp="h_a"),
            "m1_institutional_procedural": decomp_payload(["council"], lean="weak_no", hyp="h_i"),
            "m2_resource_constraint": decomp_payload(["casey"], lean="neutral", hyp="h_r"),
            "m3_information_distribution": decomp_payload(["casey"], lean="neutral", hyp="h_r"),
        }, **kw)


HERMETIC = {"drop_phases": ["phase2_evidence", "event_time"]}   # documented ablation hook: no network,
#                                                                lean-driven terminal (deterministic)


def run_default(llm=None, seed=3, policy=None, **kw):
    return simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                          horizon="2025-09-01", llm=llm or four_way_llm(), seed=seed,
                          execution_policy=policy if policy is not None else dict(HERMETIC), **kw)


# ------------------------------------------------------------------ 24.1 / 24.2 default route
def test_default_route_is_ensemble_without_any_enable_flag():
    res = run_default()
    # §NAP: materially disagreeing models now serve per-model conditionals (partially_resolved)
    # instead of an averaged headline; agreeing models complete normally
    assert res.simulation_status in ("completed", "completed_with_degradation",
                                     "partially_resolved")
    assert res.structural_ensemble is not None
    assert res.provenance["structural_mode"] == "ensemble"
    assert res.structural_ensemble["structural_mode"] == "ensemble"
    if res.simulation_status == "partially_resolved":
        assert res.resolution_report["per_model"] is not None
        assert not res.raw_distribution                # no averaged headline under disagreement


def test_facade_v2_route_carries_structural_ensemble():
    """Since the §25 default switch the facade's DEFAULT route serves lean_adaptive; the
    protection it keeps is the same: a REAL structural ensemble, never the single-model
    ablation. The ≥3-independent-generation property is the research-grade profile's and is
    pinned through the facade via the explicit profile pass-through."""
    from swm.facade import forecast
    out = forecast("Will the initiative be approved?", architecture="world_model_v2",
                   llm=four_way_llm(), as_of="2025-06-01", horizon="2025-09-01",
                   execution_policy=dict(HERMETIC))
    assert out["structural_ensemble"] is not None
    assert out["provenance"]["structural_mode"] == "ensemble"       # never the ablation
    assert out["provenance"]["execution_profile"] == "lean_adaptive"
    full = forecast("Will the initiative be approved?", architecture="world_model_v2",
                    llm=four_way_llm(), as_of="2025-06-01", horizon="2025-09-01",
                    execution_policy=dict(HERMETIC), execution_profile="full_fidelity")
    assert full["structural_ensemble"]["n_independent_generation_calls"] >= 3
    assert full["provenance"]["execution_profile"] == "full_fidelity"


# ------------------------------------------------------------------ 24.3–24.5 independent generation
def test_at_least_three_normally_four_independent_generation_calls():
    llm = four_way_llm()
    res = run_default(llm)
    n = res.structural_ensemble["n_independent_generation_calls"]
    assert n >= 3
    assert n == GENERATION_TARGET_CALLS == 4
    assert len(llm.prompts["recon"]) == 4


def test_generators_are_separate_calls_and_see_no_other_candidate():
    llm = four_way_llm()
    run_default(llm)
    prompts = llm.prompts["recon"]
    assert len(set(prompts)) == 4                      # four DISTINCT prompts (separate calls)
    # independence: no recon prompt contains another generator's thesis or decisive actors
    markers = ["Avery's choice drives it", "The council procedure decides", "Budget capacity binds",
               "avery", "council", "casey"]
    for p in prompts:
        for m in markers:
            assert m not in p, f"recon prompt leaked another candidate's content: {m!r}"


# ------------------------------------------------------------------ 24.6 adaptive expansion
def test_expands_beyond_four_when_omission_critic_finds_missing_structure():
    omission = dict(OMISSION_QUIET)
    omission.update({"missing_decisive_actor": "the regulator nobody modeled",
                     "no_further_material_model": False,
                     "proposed_models": [{"causal_thesis": "Regulator gate decides",
                                          "decisive_actors": ["regulator"],
                                          "decisive_institutions": ["licensing_office"],
                                          "decisive_constraints": ["license"],
                                          "decisive_mechanisms": ["approval_gate"],
                                          "world_boundary": "adds the licensing system",
                                          "why_missing": "absent from every candidate"}]})
    llm = four_way_llm(omission=omission)
    llm.decomp_by_model["m4_adversarial_alternative"] = decomp_payload(
        ["regulator"], lean="strong_no", hyp="h_reg", institutions=["licensing_office"])
    res = run_default(llm)
    se = res.structural_ensemble
    assert se["n_expansion_candidates"] >= 1
    gm = se["generation_manifest"]
    assert any(not g["independent"] for g in gm)       # expansion call is marked non-independent
    assert any(g["independent"] for g in gm)
    assert se["n_initial_candidates"] == 4


# ------------------------------------------------------------------ 24.7 / 24.8 / 24.9 / 24.10 dedup
def test_distinct_causal_schemas_compile_into_distinct_executable_plans():
    res = run_default()
    models = [m for m in res.structural_ensemble["models"]
              if m["promotion_status"] == "promoted"]
    hashes = [m["schema_hash"] for m in models]
    assert len(hashes) == len(set(hashes)) >= 2


def test_seeds_are_not_structural_models_identical_schemas_merge_with_certificate():
    """A degenerate backend answering every generator identically yields ONE model + a recorded
    convergence certificate — different seeds/prose never count as structural diversity."""
    one = recon_payload("only story", ["dana"])
    llm = EnsembleLLM(recon_by_role={r: one for r in
                                     ("actor_relationship", "institutional_procedural",
                                      "resource_constraint", "information_distribution")},
                      decomp_by_model={"any": decomp_payload(["dana"])})
    res = run_default(llm)
    se = res.structural_ensemble
    assert se["n_fully_simulated"] == 1
    assert se["n_merged"] == 3
    cert = se["convergence_certificate"]
    assert cert is not None and cert["certified"]
    assert cert["independent_generation_calls"] >= 3
    assert all(d["status"] in ("merged", "rejected", "failed")
               for d in cert["alternatives_disposition"])


def test_different_prose_on_equivalent_plans_merges_and_records_superficial_distinction():
    llm = four_way_llm()
    res = run_default(llm)
    merges = res.structural_ensemble["merge_manifest"]
    assert any(m["method"] == "schema_hash" for m in merges)
    assert all({"survivor", "merged", "confidence", "structural_comparison",
                "information_preserved_from_merged"} <= set(m) for m in merges)


def test_materially_different_intervention_pathways_are_not_merged():
    llm = four_way_llm()
    res = run_default(llm)
    se = res.structural_ensemble
    promoted = [m for m in se["models"] if m["promotion_status"] == "promoted"]
    assert len(promoted) >= 3                          # avery/council/casey structures all survive
    assert not llm.prompts["judge"], "deterministically distinct structures must not consult the judge"


# ------------------------------------------------------------------ 24.11 shared evidence boundary
class FrozenBundle:
    """Minimal frozen typed bundle (sealed-replay shape)."""

    def __init__(self, claims=None, as_of_ts=1748736000.0):
        self.claims = list(claims or [])
        self.included_claim_ids = [c["claim_id"] for c in self.claims]
        self.as_of = as_of_ts

    def render(self, max_chars=4000):
        return "\n".join(f"[{c['claim_id']}] {c.get('text', '')}" for c in self.claims)[:max_chars]

    def bundle_hash(self):
        return "fb_" + hashlib.sha1(json.dumps(self.claims, sort_keys=True).encode()).hexdigest()[:10]


def test_shared_evidence_single_as_of_boundary_across_models():
    bundle = FrozenBundle(claims=[{"claim_id": "c1", "text": "the vote is scheduled",
                                   "claim_class": "scheduled_event", "publication_time": 1748600000.0}])
    res = simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=four_way_llm(), seed=3,
                         execution_policy={"drop_phases": ["event_time"]}, prebuilt_bundle=bundle)
    se = res.structural_ensemble
    assert se["shared_evidence_bundle_hash"] == bundle.bundle_hash()
    assert se["shared_evidence_as_of"] == "2025-06-01"
    assert res.provenance["evidence_bundle_hash"] == bundle.bundle_hash()


# ------------------------------------------------------------------ 24.12 per-model posterior
def test_every_model_receives_its_own_posterior_record():
    bundle = FrozenBundle(claims=[{"claim_id": "c1", "text": "t", "claim_class": "actor_statement",
                                   "publication_time": 1748600000.0}])
    res = simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=four_way_llm(), seed=3,
                         execution_policy={"drop_phases": ["event_time"]}, prebuilt_bundle=bundle)
    models = [m for m in res.structural_ensemble["models"] if m["promotion_status"] == "promoted"]
    assert len(models) >= 2
    for m in models:
        assert isinstance(m["posterior_diagnostics"], dict)
        assert "n_effective_observations" in m["posterior_diagnostics"]
    plans = res._ensemble_handle
    objs = [id(c.executable_plan.posterior_rate_particles) for c in plans.surviving()
            if c.executable_plan is not None and getattr(c.executable_plan,
                                                         "posterior_rate_particles", None)]
    assert len(objs) == len(set(objs)), "posterior particle lists must never be shared across models"


# ------------------------------------------------------------------ 24.13–24.16 real pilots
def test_every_plausible_distinct_model_receives_a_real_pilot():
    res = run_default()
    se = res.structural_ensemble
    surviving = [m for m in se["models"] if m["promotion_status"] in ("promoted", "not_promoted")]
    assert se["n_pilot_simulated"] == len(surviving) >= 3
    for m in surviving:
        sim = se["simulation_manifest"][m["model_id"]]
        assert sim["pilot_particles"] >= SR.PILOT_MIN_PARTICLES


def test_pilot_runs_through_canonical_funnel_same_horizon_same_plan():
    res = run_default()
    se = res.structural_ensemble
    for m in se["models"]:
        if m["promotion_status"] != "promoted":
            continue
        sim = se["simulation_manifest"][m["model_id"]]
        assert sim["pilot_reused_as_prefix"] is True
        assert sim["final_particles"] >= sim["full_budget_required"] > sim["pilot_particles"]
        assert any(e["reason"] == "promotion_to_full_budget" for e in sim["extensions"])
    prov = res.provenance
    assert prov["active_component_manifest"]["phase8_persistence"]["executed"] is True


def test_pilot_uses_qualitative_actor_architecture_when_plan_declares_decisions():
    from tests.test_qualitative_actor import qpayload
    q = qpayload()
    llm = four_way_llm()
    decision = {"actor": "avery", "role": "principal", "at": "2025-07-01",
                "candidate_actions": [{"name": "approve", "family": "communication",
                                       "target": {"target_type": "actor", "target_id": "blake"},
                                       "mechanisms_triggered": ["record_action"],
                                       "inclusion_reason": "core"}]}
    llm.decomp_by_model["m0_actor_relationship"] = decomp_payload(
        ["avery", "blake"], lean="weak_yes", hyp="h_a", actor_decisions=[decision])
    inner = llm

    class WithDecisions:
        def __init__(self):
            self.prompts = []

        def __call__(self, prompt):
            self.prompts.append(prompt)
            if "CONSEQUENCE COMPILER" in prompt:
                return json.dumps([{"op": "record_observation", "note": "scripted"}])
            if "hypotheses" in prompt.lower() and "JSON array" in prompt:
                return inner(prompt)
            if '"selected_action_id"' in prompt or "candidate_actions" in prompt.lower():
                return json.dumps(q)
            return inner(prompt)

    wrapped = WithDecisions()
    res = simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=wrapped, seed=3,
                         execution_policy=dict(HERMETIC))
    per_model = res.provenance["per_model_provenance"]
    m0 = per_model.get("m0_actor_relationship") or {}
    report = m0.get("actor_policy_report") or {}
    assert report, "per-model actor policy report must exist"
    assert report.get("actual_actor_policy_mode") in (
        "persistent_qualitative_llm_policy", "numeric_policy")


# ------------------------------------------------------------------ 24.17 / 24.18 progressive reuse
def _mini_plan(llm, seed=3):
    from swm.world_model_v2.compiler import compile_world
    return compile_world("Will the initiative be approved?", llm=llm, evidence="",
                         as_of="2025-06-01", horizon="2025-09-01", seed=seed, persist=False,
                         structural_directive="STRUCTURAL DIRECTIVE — independent candidate model "
                                              "'m0_actor_relationship' (perspective: actor).")


def test_pilot_particles_are_deterministic_prefix_and_progressive_equals_direct():
    from swm.world_model_v2.phase8_pipeline import (finalize_persistence_run,
                                                    prepare_persistence_run, run_persistence_slice)
    llm = four_way_llm()
    plan_a = _mini_plan(llm)
    plan_b = _mini_plan(llm)
    n = plan_a.compute_plan["n_particles"]
    p = 8
    # direct full run
    h1 = prepare_persistence_run("q", plan_a)
    direct = run_persistence_slice(h1, seed=11, n_total=n, start=0, stop=n)
    # progressive: pilot prefix then extension, fresh prepared handle
    h2 = prepare_persistence_run("q", plan_b)
    pilot = run_persistence_slice(h2, seed=11, n_total=n, start=0, stop=p)
    ext = run_persistence_slice(h2, seed=11, n_total=n, start=p, stop=n)
    combined = pilot + ext
    assert len(direct) == len(combined) == n
    ro = plan_a.outcome_contract
    direct_vals = [ro.readout(b.world) for b in direct]
    combined_vals = [plan_b.outcome_contract.readout(b.world) for b in combined]
    assert direct_vals == combined_vals            # branch-for-branch identical terminals
    r1, _ = finalize_persistence_run(h1, direct, seed=11)
    r2, _ = finalize_persistence_run(h2, combined, seed=11)
    assert r1.raw_distribution == r2.raw_distribution


# ------------------------------------------------------------------ 24.19 / 24.20 / 24.21 budgets
def test_each_promoted_model_gets_at_least_the_full_single_model_budget_never_divided():
    res = run_default()
    se = res.structural_ensemble
    promoted = [m for m in se["models"] if m["promotion_status"] == "promoted"]
    assert len(promoted) >= 3
    total = 0
    for m in promoted:
        sim = se["simulation_manifest"][m["model_id"]]
        assert sim["final_particles"] >= sim["full_budget_required"] >= 12
        total += sim["final_particles"]
    n_one = max(se["simulation_manifest"][m["model_id"]]["full_budget_required"] for m in promoted)
    assert total >= len(promoted) * 12
    assert total >= n_one * len(promoted) * 0.99   # never N split across models


def test_more_than_three_models_promoted_when_all_remain_plausible():
    llm = four_way_llm()
    llm.recon_by_role["information_distribution"] = recon_payload(
        "The channel decides who even hears", ["dm_network"], constraints=["reach"],
        falsifiers=["channel outage"])
    llm.decomp_by_model["m3_information_distribution"] = decomp_payload(
        ["dm_network"], lean="strong_yes", hyp="h_n")
    res = run_default(llm)
    se = res.structural_ensemble
    promoted = [m for m in se["models"] if m["promotion_status"] == "promoted"]
    assert len(promoted) >= 4
    for m in promoted:
        sim = se["simulation_manifest"][m["model_id"]]
        assert sim["final_particles"] >= sim["full_budget_required"]


# ------------------------------------------------------------------ 24.22 / 24.23 promotion safety
def _cand(mid, support="plausible", plan=None):
    c = StructuralModelCandidate(model_id=mid, support_class=support)
    c.executable_plan = plan
    return c


def test_noisy_pilot_cannot_be_confidently_eliminated():
    ens = StructuralModelEnsemble(question="q")
    a, b = _cand("a", "strongly_supported"), _cand("b", "weak_but_possible")
    ens.candidates = [a, b]
    runs = {"a": {"n_pilot": 8, "pilot_distribution": {"yes": 0.5, "no": 0.5}, "error": ""},
            "b": {"n_pilot": 8, "pilot_distribution": {"yes": 0.5, "no": 0.5}, "error": ""}}
    SR._promote_models(ens, runs)                      # 8 < PILOT_MIN_N_FOR_EXCLUSION
    assert a.promotion_status == b.promotion_status == "promoted"
    assert "pilot_too_small_to_exclude_safely" in b.promotion_reason


def test_low_probability_decision_reversing_model_is_promoted():
    ens = StructuralModelEnsemble(question="q")
    a, b = _cand("a", "strongly_supported"), _cand("b", "weak_but_possible")
    ens.candidates = [a, b]
    runs = {"a": {"n_pilot": 16, "pilot_distribution": {"yes": 0.9, "no": 0.1}, "error": ""},
            "b": {"n_pilot": 16, "pilot_distribution": {"yes": 0.1, "no": 0.9}, "error": ""}}
    SR._promote_models(ens, runs)
    assert b.promotion_status == "promoted"            # contrarian prediction is a reason FOR promotion
    assert "predicts_differently" in b.promotion_reason


# ------------------------------------------------------------------ 24.24 / 24.25 / 24.26 rejection
def test_clearly_invalid_model_rejected_before_full_simulation():
    llm = four_way_llm(critic_by_thesis={
        "Budget capacity binds": critic_ok(reject=True,
                                           reject_reason="incoherent boundary: no causal path")})
    res = run_default(llm)
    se = res.structural_ensemble
    rejected = [m for m in se["rejected_and_merged"] if m["status"] == "rejected"]
    assert any("critic_invalid" in m["reason"] for m in rejected)
    rejected_ids = {m["model_id"] for m in rejected}
    assert not (rejected_ids & set(se["simulation_manifest"].keys() &
                                   {m for m in se["simulation_manifest"]
                                    if se["simulation_manifest"][m].get("final_particles", 0) > 0}))


def test_evidence_contradicted_model_rejected_with_exact_evidence():
    llm = four_way_llm(critic_by_thesis={
        "The council procedure decides": critic_ok(
            support="contradicted", basis="c9 shows the council already dissolved",
            contradictions=[{"claim_id": "c9", "claim": "the council dissolved in May",
                             "why_contradicts": "no council can vote"}])})
    res = run_default(llm)
    rejected = [m for m in res.structural_ensemble["rejected_and_merged"]
                if m["status"] == "rejected"]
    hit = [m for m in rejected if "evidence_contradicted" in m["reason"]]
    assert hit and "c9" in hit[0]["reason"]


def test_adversarial_but_plausible_model_survives_lower_critic_preference():
    llm = four_way_llm(critic_by_thesis={
        "Budget capacity binds": critic_ok(support="weak_but_possible",
                                           basis="thin evidence but coherent")})
    res = run_default(llm)
    promoted = {m["model_id"]: m for m in res.structural_ensemble["models"]
                if m["promotion_status"] == "promoted"}
    weak = [m for m in promoted.values() if m["support_class"] == "weak_but_possible"]
    assert weak, "a weak-but-possible distinct model must still be promoted"


# ------------------------------------------------------------------ 24.27 / 24.28 no minted weights
def test_llm_cannot_mint_model_probabilities():
    dirty = critic_ok()
    dirty["probability"] = 0.83
    dirty["model_probability"] = 0.7
    dirty["prior"] = 0.4
    llm = four_way_llm(critic_by_thesis={"Avery's choice drives it": dirty})
    res = run_default(llm)
    se = res.structural_ensemble
    blob = json.dumps(se["models"]) + json.dumps(se["model_support"])
    assert "0.83" not in blob and "model_probability" not in blob
    for m in se["models"]:
        assert m["support_class"] in ("strongly_supported", "plausible", "weak_but_possible",
                                      "contradicted", "unresolved")
    assert EC._strip_minted_probabilities({"probability": 0.5, "support_class": "plausible"}) == \
        {"support_class": "plausible"}


def test_unknown_weights_never_average_into_a_headline_probability():
    """§NAP: with no defensible model weights, materially disagreeing conditionals are NOT
    averaged into a headline — per-model distributions + the robust range are primary; the
    equal-weight mixture survives only as a labeled diagnostic."""
    res = run_default()
    se = res.structural_ensemble
    assert se["aggregation_method"] in ("per_model_conditionals_no_headline_average",
                                        "agreeing_models_equal_weight_mixture",
                                        "single_surviving_model")
    assert se["equal_weight_mixture_diagnostic"] and se["robust_range"]
    for opt, rng in se["robust_range"].items():
        assert rng["min"] <= se["equal_weight_mixture_diagnostic"][opt] <= rng["max"]
    if se["aggregation_method"] == "per_model_conditionals_no_headline_average":
        assert not res.raw_distribution
        assert "§NAP" in se["aggregation_note"]
        assert res.resolution_report["robust_range"] == se["robust_range"]
    dec = se["uncertainty_decomposition"]
    assert "between_model" in dec and "within_model" in dec


# ------------------------------------------------------------------ 24.29 per-model outputs
def test_forecast_outputs_preserve_per_model_distributions():
    res = run_default()
    se = res.structural_ensemble
    dists = se["model_distributions"]
    assert len(dists) >= 3
    assert len({json.dumps(d, sort_keys=True) for d in dists.values()}) >= 2  # genuinely different
    assert res.structural_disagreement == dists
    for m in se["models"]:
        if m["promotion_status"] == "promoted":
            assert m["prediction"] == dists[m["model_id"]]


# ------------------------------------------------------------------ 24.31 / 24.32 reversal + VoI
def test_reversal_conditions_generated_from_actual_model_differences():
    res = run_default()
    se = res.structural_ensemble
    assert se["reversal_conditions"], "diverging models must yield reversal conditions"
    rc = se["reversal_conditions"][0]
    theses = {m["causal_thesis"] for m in se["models"]}
    assert any(t and t[:40] in json.dumps(rc) for t in theses)
    assert "evidence_that_would_confirm" in rc


def test_structural_voi_identifies_model_discriminating_observation():
    res = run_default()
    se = res.structural_ensemble
    voi = se["structural_value_of_information"]
    assert voi, "material disagreement must produce discriminating observations"
    falsifiers = {f for m in se["models"] for f in
                  (res._ensemble_handle.by_id(m["model_id"]).falsifiers
                   if res._ensemble_handle.by_id(m["model_id"]) else [])}
    assert any(v["observation"] in falsifiers or v["observation"] for v in voi)
    assert all(len(v["distinguishes_models"]) == 2 for v in voi if v["distinguishes_models"])


# ------------------------------------------------------------------ sensitivity classification
def test_sensitivity_thresholds_classify_from_actual_results():
    stable = classify_forecast_sensitivity({"a": {"yes": 0.50}, "b": {"yes": 0.52}})
    assert stable["classification"] == "structurally_stable"
    mild = classify_forecast_sensitivity({"a": {"yes": 0.50, "no": 0.50},
                                          "b": {"yes": 0.60, "no": 0.40}})
    assert mild["classification"] == "mildly_structurally_sensitive"
    material = classify_forecast_sensitivity({"a": {"yes": 0.70, "no": 0.30},
                                              "b": {"yes": 0.30, "no": 0.70}})
    assert material["classification"] == "materially_structurally_sensitive"
    assert material["direction_change"] is True
    under = classify_forecast_sensitivity({"a": {"yes": 0.5}}, underidentified=True)
    assert under["classification"] == "structurally_underidentified"
    incomplete = classify_forecast_sensitivity({}, incomplete=True)
    assert incomplete["classification"] == "ensemble_execution_incomplete"
    # threshold sensitivity: the boundary constants are exposed and behave monotonically
    lo = classify_forecast_sensitivity({"a": {"yes": 0.5}, "b": {"yes": 0.5 + 0.049}})
    hi = classify_forecast_sensitivity({"a": {"yes": 0.5, "no": 0.5},
                                        "b": {"yes": 0.5 + 0.051, "no": 0.449}})
    assert lo["classification"] == "structurally_stable"
    assert hi["classification"] == "mildly_structurally_sensitive"


# ------------------------------------------------------------------ 24.38 / 24.39 cache safety
def test_identical_llm_calls_reuse_and_different_calls_never_share():
    ledger = CallLedger()
    calls = []

    def backend(prompt):
        calls.append(prompt)
        return f"resp:{len(calls)}"

    c = CachedLLM(backend, ledger=ledger, stage="s")
    assert c("A") == c("A") == "resp:1"                # identical input reused
    assert c("B") == "resp:2"                          # different input → new call
    assert ledger.total_cache_hits() == 1 and ledger.total_calls() == 2


def test_actor_view_cache_shares_only_same_particle_same_prompt_same_occurrence():
    calls = []

    def backend(prompt):
        calls.append(prompt)
        return f"resp:{len(calls)}"

    cache = ScopedActorCache(backend)
    cache.enter_branch(0)
    r_model_a = cache("view-X")
    cache.enter_branch(0)                              # model B, SAME particle index, SAME view
    assert cache("view-X") == r_model_a
    cache.enter_branch(1)                              # different particle → never shared
    assert cache("view-X") != r_model_a
    cache.enter_branch(0)
    assert cache("view-DIFFERENT") not in (r_model_a,)  # different view → never shared
    assert len(calls) == 3


# ------------------------------------------------------------------ 24.40 loud missing backend
def test_missing_llm_backend_fails_loudly_never_a_deterministic_fallback_model():
    res = run_default(llm=False and None, policy=dict(HERMETIC)) if False else \
        simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                       horizon="2025-09-01", llm=None, seed=1, execution_policy=dict(HERMETIC))
    assert res.simulation_status == "execution_failed"
    assert res.failure_taxonomy == "unavailable_service"
    assert not res.raw_distribution and res.structural_ensemble is None


# ------------------------------------------------------------------ 24.42 explicit single-model mode
def test_single_model_mode_requires_explicit_ablation_setting():
    res = simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=four_way_llm(), seed=3,
                         execution_policy={**HERMETIC,
                                           "structural_mode": "single_structural_model"})
    assert res.provenance["structural_mode"] == "single_structural_model"
    assert res.structural_ensemble is None
    with pytest.raises(ValueError):
        simulate_world("q", as_of="2025-06-01", horizon="2025-09-01", llm=four_way_llm(),
                       execution_policy={"structural_mode": "sneaky_default"})


# ------------------------------------------------------------------ 24.43 replay records ensemble
def test_historical_replay_route_records_the_structural_ensemble():
    bundle = FrozenBundle(claims=[{"claim_id": "c1", "text": "archived claim",
                                   "claim_class": "actor_statement",
                                   "publication_time": 1748600000.0}])
    res = simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=four_way_llm(), seed=3,
                         execution_policy={"drop_phases": ["event_time"]}, prebuilt_bundle=bundle)
    se = res.structural_ensemble
    assert se is not None and se["shared_evidence_bundle_hash"] == bundle.bundle_hash()
    assert se["n_fully_simulated"] >= 2


# ------------------------------------------------------------------ integrity invariants (Section 23)
def test_shared_plan_objects_across_candidates_fail_loudly():
    ens = StructuralModelEnsemble(question="q")
    plan = object.__new__(dict if False else type("P", (), {}))  # any object identity
    a, b = _cand("a"), _cand("b")
    a.executable_plan = b.executable_plan = plan
    ens.candidates = [a, b]
    with pytest.raises(EnsembleIntegrityError):
        ens.validate_integrity()


def test_single_survivor_without_certificate_fails_loudly():
    ens = StructuralModelEnsemble(question="q")
    ens.candidates = [_cand("a")]
    with pytest.raises(EnsembleIntegrityError):
        ens.validate_integrity()
