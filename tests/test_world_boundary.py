"""Contract tests for swm/world_model_v2/world_boundary.py.

Fake LLM callables return canned JSON keyed on prompt markers; everything is deterministic.
"""
import json
import types

import pytest

from swm.world_model_v2.world_boundary import (
    BOUNDARY_SUPPORT,
    QUALITATIVE_SENSITIVITY,
    REPRESENTATIONS,
    BoundaryComponent,
    WorldBoundary,
    _hash,
    boundary_sensitivity_analysis,
    generate_world_boundary,
    run_boundary_critics,
)

# ------------------------------------------------------------------ canned LLM responses
BOUNDARY_JSON = json.dumps({
    "components": [
        {"kind": "individual_actor", "name": "Alice", "representation": "individual",
         "reason": "holds the signing decision", "evidence": ["claim:c1"],
         "sensitivity": "decisive", "promotion_trigger": ""},
        {"kind": "population", "name": "retail_customers", "representation": "aggregate",
         "reason": "acts as a demand segment", "evidence": ["survey"],
         "sensitivity": "material", "promotion_trigger": "a named customer sues"},
        {"kind": "nonhuman_system", "name": "payment_gateway", "representation": "individual",
         "reason": "processes every transaction", "sensitivity": "material"},
        {"kind": "external_event_family", "name": "regional_weather",
         "representation": "external_process", "reason": "affects logistics",
         "sensitivity": "minor"},
        {"kind": "geography", "name": "overseas_market", "representation": "excluded",
         "reason": "no causal path within the horizon", "sensitivity": "negligible"},
        {"kind": "institution", "name": "tax_authority", "representation": "unresolved",
         "reason": "cannot model the audit process", "sensitivity": "material"},
        # normalization targets: empty kind, invalid representation/sensitivity, non-list evidence
        {"kind": "", "name": "mystery_process", "representation": "hybrid",
         "reason": "?", "sensitivity": "very-high", "evidence": "not-a-list"},
        # nameless component must be dropped
        {"kind": "individual_actor", "name": "", "representation": "individual"},
    ],
    "sensitivity_assumptions": ["weather stays in its normal band"],
    "boundary_evidence": ["filings name Alice as sole signer"],
})


def boundary_llm(log=None):
    def _llm(prompt):
        if log is not None:
            log.append(prompt)
        if "EXPLICIT WORLD BOUNDARY" in prompt:
            return BOUNDARY_JSON
        raise AssertionError("unexpected prompt: " + prompt[:60])
    return _llm


def fake_plan():
    return types.SimpleNamespace(
        entities=[{"id": "alice"}, "bob", types.SimpleNamespace(id="carol")],
        populations=["retail_customers", "warehouse_staff"],
        institutions=[{"id": "fda"}],
        accepted_mechanisms=[{"mech_id": "approval_flow"}, {"operator": "queue_step"},
                             "not-a-dict"])


def generated_boundary():
    return generate_world_boundary(
        question="will the deal close", structural_model_id="sm1", thesis="alice decides",
        plan=fake_plan(), as_of="2026-01-01", horizon="90d",
        llm=boundary_llm())


# ------------------------------------------------------------------ generation + normalization
def test_generation_normalizes_bad_kinds_and_representations():
    b = generated_boundary()
    m = b.component("mystery_process")
    assert m is not None
    assert m.kind == "nonhuman_system"                 # empty kind coerced
    assert m.representation == "unresolved"            # invalid representation coerced
    assert m.sensitivity == "unknown"                  # invalid sensitivity coerced
    assert m.evidence == []                            # non-list evidence dropped
    assert m.representation in REPRESENTATIONS and m.sensitivity in QUALITATIVE_SENSITIVITY
    assert all(c.name for c in b.components)           # nameless component dropped
    assert b.component("Alice").representation == "individual"
    assert b.sensitivity_assumptions == ["weather stays in its normal band"]


def test_rederive_views_buckets_aggregates_excluded_unresolved():
    b = generated_boundary()
    assert "Alice" in b.included_individual_actors
    assert "retail_customers" in b.included_populations
    assert "retail_customers" in b.represented_as_aggregates
    assert "payment_gateway" in b.included_nonhuman_systems
    assert b.represented_as_external_processes == ["regional_weather"]
    excl = [e for e in b.explicitly_excluded if e["name"] == "overseas_market"]
    assert excl and excl[0]["reason"] == "no causal path within the horizon"
    unresolved_names = {u["name"] for u in b.unresolved_components}
    assert {"tax_authority", "mystery_process"} <= unresolved_names
    tax = next(u for u in b.unresolved_components if u["name"] == "tax_authority")
    assert tax["sensitivity"] == "material"
    assert tax["why_unresolved"] == "cannot model the audit process"
    assert b.inclusion_reasons["Alice"] == "holds the signing decision"
    assert b.exclusion_reasons["overseas_market"]
    # excluded/unresolved never appear in inclusion views
    assert "overseas_market" not in b.inclusion_reasons
    assert "tax_authority" not in b.included_institutions
    # external_process components stay out of the kind buckets
    for view in (b.included_individual_actors, b.included_nonhuman_systems):
        assert "regional_weather" not in view


def test_plan_crosscheck_adds_instantiated_but_unmentioned_components():
    b = generated_boundary()
    for name in ("bob", "carol", "warehouse_staff", "fda", "approval_flow", "queue_step"):
        comp = b.component(name)
        assert comp is not None, name
        assert comp.source == "plan_crosscheck"
        assert "executable plan" in comp.reason
    assert b.component("warehouse_staff").representation == "aggregate"
    assert b.component("bob").representation == "individual"
    assert "fda" in b.included_institutions
    assert {"approval_flow", "queue_step"} <= set(b.included_mechanisms)
    # 'alice' was already named by the generation (case-insensitively) — never duplicated
    assert sum(1 for c in b.components if c.name.lower() == "alice") == 1


def test_generation_trace_records_prompt_and_response_hashes():
    log = []
    b = generate_world_boundary(question="q", structural_model_id="sm1",
                                plan=fake_plan(), llm=boundary_llm(log))
    tr = b.generation_trace[0]
    assert tr["stage"] == "boundary_generation" and tr["ok"] is True and tr["error"] == ""
    assert tr["prompt_hash"] == _hash(log[0]) and len(tr["prompt_hash"]) == 16
    assert tr["response_hash"] == _hash(BOUNDARY_JSON) and len(tr["response_hash"]) == 16


def test_generation_without_llm_is_plan_crosscheck_only_with_trace_error():
    b = generate_world_boundary(question="q", structural_model_id="sm1",
                                plan=fake_plan(), llm=None)
    assert b.components and all(c.source == "plan_crosscheck" for c in b.components)
    names = {c.name for c in b.components}
    assert {"alice", "bob", "carol", "retail_customers", "warehouse_staff", "fda"} <= names
    tr = b.generation_trace[0]
    assert tr["ok"] is False and tr["error"] == "no_llm_backend"
    assert tr["prompt_hash"] and tr["response_hash"] == ""


def test_generation_llm_exception_recorded_loudly():
    def broken(prompt):
        raise RuntimeError("backend down")
    b = generate_world_boundary(question="q", structural_model_id="sm1",
                                plan=fake_plan(), llm=broken)
    tr = b.generation_trace[0]
    assert tr["ok"] is False and "RuntimeError" in tr["error"]
    assert all(c.source == "plan_crosscheck" for c in b.components)


def test_generation_total_failure_is_marked_unresolved_not_supported():
    # no llm AND no plan: the boundary must carry an explicit unresolved marker that
    # survives rederive_views, and can never classify as boundary_supported
    b = generate_world_boundary(question="q", structural_model_id="sm1", plan=None, llm=None)
    assert b.unresolved_components, "failed generation must surface as unresolved"
    assert any("generation failed" in u["name"] for u in b.unresolved_components)
    assert b.classify_support() != "boundary_supported"


# ------------------------------------------------------------------ identity
def _manual_boundary():
    b = WorldBoundary(boundary_id="wbh", structural_model_id="smh", question="q?")
    b.components = [
        BoundaryComponent(component_id="c1", kind="individual_actor", name="alice",
                          representation="individual", reason="signs", evidence=["e1"]),
        BoundaryComponent(component_id="c2", kind="population", name="customers",
                          representation="aggregate", reason="demand"),
    ]
    b.rederive_views()
    return b


def test_boundary_hash_stable_under_prose_changes_but_not_structure():
    b1, b2 = _manual_boundary(), _manual_boundary()
    b2.components[0].reason = "totally different prose"
    b2.components[0].evidence = ["other evidence"]
    b2.components[1].sensitivity = "decisive"
    assert b1.boundary_hash() == b2.boundary_hash()
    b2.components[1].representation = "individual"     # structural change
    assert b1.boundary_hash() != b2.boundary_hash()
    assert b1.as_dict()["boundary_hash"] == b1.boundary_hash()


def test_add_component_dedups_and_lookup_by_id():
    b = _manual_boundary()
    n = len(b.components)
    same = b.add_component(BoundaryComponent(component_id="cX", kind="individual_actor",
                                             name="alice", representation="excluded"))
    assert same is b.components[0] and len(b.components) == n
    assert b.component("c2").name == "customers"
    assert b.component("nobody") is None


# ------------------------------------------------------------------ the three critics
def critic_llm():
    def _llm(prompt):
        if "OMITTED-ACTOR CRITIC" in prompt:
            return json.dumps({"findings": [
                {"component": "regulator_x", "kind": "institution",
                 "finding": "holds veto over the deal", "sensitivity": "decisive",
                 "activating_event": "merger filing"},
                {"component": "night_shift_union", "kind": "actor_group",
                 "finding": "can strike", "sensitivity": "catastrophic"},
                {"finding": "nameless finding must be skipped"}]})
        if "OMITTED-SYSTEM CRITIC" in prompt:
            return json.dumps({"findings": [
                {"component": "payment_rails", "kind": "nonhuman_system",
                 "finding": "an outage stalls closing", "sensitivity": "material",
                 "entry_path": "capacity change"}]})
        if "BOUNDARY ADVERSARY" in prompt:
            return json.dumps({"alternative_boundary": {
                "moves_inside": ["press corps"], "moves_outside": ["alice"],
                "why_result_differs": "coverage pressure flips the vote",
                "sensitivity": "material"}})
        raise AssertionError("unexpected prompt: " + prompt[:60])
    return _llm


def _critic_boundary():
    b = WorldBoundary(boundary_id="wbc", structural_model_id="smc",
                      question="will the deal close")
    b.add_component(BoundaryComponent(component_id="c1", kind="individual_actor",
                                      name="alice", representation="individual",
                                      reason="signs the deal"))
    return b


def test_critics_append_findings_and_register_unresolved_components():
    b = _critic_boundary()
    appended = run_boundary_critics(b, llm=critic_llm(), thesis="alice decides")
    assert appended == b.critic_findings and len(appended) == 4
    by_comp = {f["component"]: f for f in appended}
    assert by_comp["regulator_x"]["critic"] == "omitted_actor"
    assert by_comp["regulator_x"]["sensitivity"] == "decisive"
    assert by_comp["regulator_x"]["activating_event"] == "merger filing"
    assert by_comp["night_shift_union"]["sensitivity"] == "unknown"    # coerced
    assert by_comp["payment_rails"]["critic"] == "omitted_system"
    assert all(f["resolution"] == "open" for f in appended)
    # omitted components registered as unresolved boundary components
    for name, critic in (("regulator_x", "critic:omitted_actor"),
                         ("payment_rails", "critic:omitted_system")):
        comp = b.component(name)
        assert comp.representation == "unresolved" and comp.source == critic
    assert {"regulator_x", "payment_rails"} <= {u["name"] for u in b.unresolved_components}
    stages = [t["stage"] for t in b.generation_trace]
    assert stages == ["critic:omitted_actor", "critic:omitted_system",
                      "critic:boundary_adversary"]
    assert all(t["ok"] for t in b.generation_trace)
    assert b.classify_support() == "under_modeled_boundary"     # decisive unresolved


def test_adversary_critic_parses_alternative_boundary():
    b = _critic_boundary()
    run_boundary_critics(b, llm=critic_llm())
    adv = [f for f in b.critic_findings if f["critic"] == "boundary_adversary"]
    assert len(adv) == 1
    assert adv[0]["component"] == "(alternative boundary)"
    assert adv[0]["alternative"] == {"moves_inside": ["press corps"],
                                     "moves_outside": ["alice"]}
    assert adv[0]["finding"] == "coverage pressure flips the vote"
    assert adv[0]["sensitivity"] == "material"


def test_critics_without_llm_record_error_and_append_nothing():
    b = _critic_boundary()
    assert run_boundary_critics(b, llm=None) == []
    assert b.critic_findings == []
    assert b.generation_trace[-1]["error"] == "no_llm_backend"


def test_critic_unparseable_reply_is_traced_not_fatal():
    def half_llm(prompt):
        if "OMITTED-ACTOR CRITIC" in prompt:
            return "no json at all"
        if "OMITTED-SYSTEM CRITIC" in prompt:
            raise ValueError("timeout")
        return json.dumps({"alternative_boundary": {}})     # empty alternative → no finding
    b = _critic_boundary()
    assert run_boundary_critics(b, llm=half_llm) == []
    trace = {t["stage"]: t for t in b.generation_trace}
    assert trace["critic:omitted_actor"]["ok"] is True          # parsed to nothing, no findings
    assert trace["critic:omitted_system"]["ok"] is False
    assert "ValueError" in trace["critic:omitted_system"]["error"]


# ------------------------------------------------------------------ sensitivity analysis (§6)
def _sens_boundary():
    b = WorldBoundary(boundary_id="wbs", structural_model_id="sms", question="q")
    b.components = [
        BoundaryComponent(component_id="c1", kind="individual_actor", name="alice",
                          representation="individual", reason="r"),
        BoundaryComponent(component_id="c2", kind="institution", name="tax_authority",
                          representation="unresolved", reason="cannot model audits",
                          sensitivity="material"),
        BoundaryComponent(component_id="c3", kind="geography", name="overseas_market",
                          representation="excluded", reason="assumed irrelevant"),
        BoundaryComponent(component_id="c4", kind="nonhuman_system", name="office_cat",
                          representation="unresolved", reason="irrelevant",
                          sensitivity="negligible"),
    ]
    b.rederive_views()
    return b


def sens_llm(prompt):
    assert "BOUNDARY-SENSITIVITY ANALYSIS" in prompt
    if "tax_authority" in prompt:
        return json.dumps({"entry_path": "an audit freezes the accounts",
                           "can_reverse_forecast_direction": True,
                           "can_change_best_action": False, "changes_tail_risk": True,
                           "evidence_that_would_determine_importance": "audit-rate history",
                           "must_be_promoted_into_detailed_simulation": True,
                           "reasoning": "structural"})
    return json.dumps({"entry_path": "import demand shifts",
                       "can_reverse_forecast_direction": False,
                       "can_change_best_action": True, "changes_tail_risk": False,
                       "evidence_that_would_determine_importance": "export share",
                       "must_be_promoted_into_detailed_simulation": False,
                       "reasoning": "structural"})


def test_sensitivity_rows_set_flags_and_unresolved_decisive_picks_them_up():
    b = _sens_boundary()
    rows = boundary_sensitivity_analysis(b, llm=sens_llm, options=["close", "walk"])
    assert rows == b.omitted_component_sensitivity
    by_name = {r["omitted_component"]: r for r in rows}
    assert "office_cat" not in by_name                      # negligible components are skipped
    assert by_name["tax_authority"]["can_reverse_forecast_direction"] is True
    assert by_name["tax_authority"]["can_change_best_action"] is False
    assert by_name["overseas_market"]["can_change_best_action"] is True
    assert by_name["tax_authority"]["prior_sensitivity"] == "material"
    decisive = b.unresolved_decisive()
    decisive_names = {d["name"] for d in decisive}
    # both flagged rows are picked up — including the EXCLUDED component that only the
    # sensitivity analysis flags (it never sits in unresolved_components)
    assert {"tax_authority", "overseas_market"} <= decisive_names
    row_derived = [d for d in decisive if d["name"] == "overseas_market"]
    assert row_derived and row_derived[0]["sensitivity"] == "decisive"
    assert row_derived[0]["why_unresolved"].startswith("sensitivity analysis:")
    assert b.support_classification == "under_modeled_boundary"
    assert all(t["ok"] for t in b.generation_trace if t["stage"].startswith("sensitivity:"))


def test_sensitivity_without_llm_records_error_rows():
    b = _sens_boundary()
    rows = boundary_sensitivity_analysis(b, llm=None)
    assert rows and all(r["error"] == "no_llm_backend" for r in rows)
    assert all(r["can_reverse_forecast_direction"] is None for r in rows)
    # material unresolved component still drives the honest classification
    assert b.support_classification == "under_modeled_boundary"


# ------------------------------------------------------------------ classification
def test_classify_support_three_way():
    assert set(BOUNDARY_SUPPORT) == {"boundary_supported", "boundary_provisional",
                                     "under_modeled_boundary"}
    b = _manual_boundary()
    b.critic_findings = [{"critic": "omitted_actor", "component": "x",
                          "finding": "f", "sensitivity": "minor", "resolution": "resolved"}]
    assert b.classify_support() == "boundary_supported"
    b.critic_findings[0]["resolution"] = "open"
    assert b.classify_support() == "boundary_provisional"
    b.critic_findings[0]["resolution"] = "resolved"
    b.components.append(BoundaryComponent(component_id="cu", kind="institution",
                                          name="unknown_inst", representation="unresolved",
                                          reason="?", sensitivity="unknown"))
    b.rederive_views()
    assert b.classify_support() == "boundary_provisional"    # unknown-sensitivity unresolved
    b.component("unknown_inst").sensitivity = "material"
    b.rederive_views()
    assert b.classify_support() == "under_modeled_boundary"  # material/decisive unresolved


# ------------------------------------------------------------------ dynamic promotion
def test_record_promotion_moves_component_and_logs_exact_timestamp():
    b = _manual_boundary()
    ts = 1712000000.25
    rec = b.record_promotion(name="customers", kind="population", at=ts,
                             trigger="a named customer sued", promoted_to="individual_actor",
                             reconstructed_history=2)
    assert rec["at"] == ts and rec["reconstructed_history_events"] == 2
    assert b.dynamic_promotions == [rec]
    comp = b.component("customers")
    assert comp.representation == "individual" and comp.promoted_at == ts
    assert "promoted at t=1712000000.25: a named customer sued" in comp.reason
    assert "customers" not in b.represented_as_aggregates     # views rederived
    assert "customers" in b.included_populations
    # promoting a component the boundary had never named creates it with source=runtime
    rec2 = b.record_promotion(name="brand_new_actor", kind="individual_actor",
                              at=ts + 1, trigger="entered the room", promoted_to="individual")
    comp2 = b.component("brand_new_actor")
    assert comp2.source == "runtime" and comp2.promoted_at == ts + 1
    assert "brand_new_actor" in b.included_individual_actors
    assert rec2["at"] == ts + 1 and len(b.dynamic_promotions) == 2


# ------------------------------------------------------------------ the eight §3 answers
def test_boundary_answers_covers_all_eight_questions():
    b = generated_boundary()
    ans = b.boundary_answers()
    assert set(ans) == {"simulated_individually", "simulated_as_population",
                        "represented_as_nonhuman_system", "outside_detailed_world",
                        "inclusion_reasons", "exclusion_reasons", "boundary_evidence",
                        "omitted_components_that_could_change_answer", "expansion_triggers"}
    assert "Alice" in ans["simulated_individually"]
    assert "fda" in ans["simulated_individually"]             # institutions count
    assert "retail_customers" in ans["simulated_as_population"]
    assert "payment_gateway" in ans["represented_as_nonhuman_system"]
    assert {"regional_weather", "overseas_market"} <= set(ans["outside_detailed_world"])
    assert ans["inclusion_reasons"]["Alice"] and ans["exclusion_reasons"]["overseas_market"]
    assert ans["boundary_evidence"] == ["filings name Alice as sole signer"]
    assert {d["name"] for d in ans["omitted_components_that_could_change_answer"]} == \
        {"tax_authority"}                                     # the material unresolved one
    assert any(r["component"] == "retail_customers" and r["trigger"] == "a named customer sues"
               for r in ans["expansion_triggers"])
