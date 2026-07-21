"""First-class branch truncation + honest epistemic statuses (§8, §20, §21, §35) — contract tests.

Truncated branch mass is unresolved simulation, not Monte Carlo error: it is never renormalized away,
it bounds every reported option, and it can force a recommendation to be withheld. under_modeled and
truncated are epistemic/representational states of the RESULT — not engineering exceptions and not
clarification; a partial distribution may ride with them but stays explicitly conditional on the
modeled portion of the world. All scripted (no network, no LLM)."""
import random

import pytest

from swm.world_model_v2.result import (CONDITIONAL_FORECAST_SENTENCE, SIMULATION_STATUSES,
                                       UNDER_MODELED_SUBTYPES, SimulationResult,
                                       migrate_legacy_result)
from swm.world_model_v2.truncation import (BRANCH_STATUSES, TRUNCATED_BRANCH_STATUSES,
                                           BranchTruncationRecord, aggregate_branch_statuses,
                                           honest_note, map_truncation_kind,
                                           recommendation_eligibility, truncation_bounds)


# ---------------------------------------------------------------- §20 status vocabulary + kind mapping
def test_branch_status_vocabulary_is_first_class_and_complete():
    assert BRANCH_STATUSES == (
        "active", "completed", "absorbed", "quiescent",
        "truncated_actor_budget", "truncated_event_budget", "truncated_context_budget",
        "truncated_boundary_budget", "truncated_missing_mechanism", "truncated_provider_failure",
        "truncated_timeout", "invalid")
    assert len(set(BRANCH_STATUSES)) == len(BRANCH_STATUSES)
    assert set(TRUNCATED_BRANCH_STATUSES) == {s for s in BRANCH_STATUSES if s.startswith("truncated_")}


def test_map_truncation_kind_covers_every_runtime_kind():
    expected = {
        "actor_llm_budget_exhausted": "truncated_actor_budget",
        "invocation_safety_budget_reached": "truncated_actor_budget",
        "safety_max_events_reached": "truncated_event_budget",
        "provider_failure_all_families": "truncated_provider_failure",
        "cognition_stage_failure": "truncated_provider_failure",
        "missing_mechanism": "truncated_missing_mechanism",
        "context_budget": "truncated_context_budget",
        "boundary_budget": "truncated_boundary_budget",
        "timeout": "truncated_timeout",
    }
    for kind, status in expected.items():
        assert map_truncation_kind(kind) == status
        assert status in BRANCH_STATUSES


def test_unknown_kind_maps_conservatively_with_recorded_note():
    notes = []
    assert map_truncation_kind("cascade_depth_safety_budget_reached",
                               notes=notes) == "truncated_event_budget"
    assert notes and "cascade_depth_safety_budget_reached" in notes[0]
    assert map_truncation_kind("", notes=notes) == "truncated_event_budget"   # never dropped


def test_branch_truncation_record_validates_status():
    r = BranchTruncationRecord(branch_id="b0", status="truncated_actor_budget",
                               reason="actor_llm_budget_exhausted", at_ts=120.0,
                               pending_events=[{"etype": "reconsider"}], affected_actors=["ceo"])
    assert r.weight == 0.0                                     # equal-weight placeholder
    with pytest.raises(ValueError):
        BranchTruncationRecord(branch_id="b1", status="not_a_status")


# ---------------------------------------------------------------- §21 aggregation never renormalizes
def test_aggregation_weight_identity_property_random_cases():
    rng = random.Random(20260719)
    for _ in range(60):
        n = rng.randint(1, 40)
        stats = []
        for i in range(n):
            fate = rng.choice(["completed", "truncated", "invalid"])
            stats.append({"branch_id": f"b{i}", "weight": rng.random(),
                          "truncated": fate == "truncated",
                          "truncation": ({"reason": "safety_max_events_reached",
                                          "at_ts": rng.random() * 1000}
                                         if fate == "truncated" else {}),
                          "invalid": fate == "invalid"})
        rep = aggregate_branch_statuses(stats)
        # the identity holds EXACTLY — truncated/invalid weight is never renormalized away
        assert rep["completed_weight"] + rep["truncated_weight"] + rep["invalid_weight"] \
            == rep["total_weight"]
        if any(b["truncated"] for b in stats):
            assert rep["truncated_weight"] > 0.0
        if any(b["invalid"] for b in stats):
            assert rep["invalid_weight"] > 0.0


def test_aggregation_report_contents_hand_computed():
    stats = [
        {"branch_id": "b0", "truncated": False, "weight": 0.5},
        {"branch_id": "b1", "truncated": True, "weight": 0.25,
         "truncation": {"reason": "safety_max_events_reached", "at_ts": 200.0,
                        "pending_events": [{"etype": "reconsider", "ts": 210.0}],
                        "actors_not_processed": ["ceo"]}},
        {"branch_id": "b2", "truncated": True, "weight": 0.25,
         "truncation": {"reason": "actor_llm_budget_exhausted", "at_ts": 100.0,
                        "pending_events": [{"etype": "reconsider", "ts": 210.0},
                                           {"etype": "file_motion", "ts": 300.0}],
                        "affected_actors": ["board", "ceo"]}},
    ]
    rep = aggregate_branch_statuses(stats)
    assert rep["completed_weight"] == 0.5 and rep["truncated_weight"] == 0.5
    assert rep["invalid_weight"] == 0.0 and rep["total_weight"] == 1.0
    assert rep["truncation_reasons"] == {"safety_max_events_reached": 1,
                                         "actor_llm_budget_exhausted": 1}
    assert rep["earliest_truncation_ts"] == 100.0
    assert rep["actors_affected"] == ["board", "ceo"]
    assert rep["truncated_branch_ids"] == ["b1", "b2"]
    # pending events form a UNION: the duplicate reconsider event appears once
    assert len(rep["pending_high_sensitivity_events"]) == 2
    assert rep["honest_note"] == honest_note()


def test_aggregation_equal_weight_default_and_pending_cap():
    stats = [{"branch_id": f"b{i}", "truncated": True,
              "truncation": {"reason": "timeout",
                             "pending_events": [{"etype": f"ev{i}_{j}"} for j in range(10)]}}
             for i in range(8)]
    rep = aggregate_branch_statuses(stats)                      # no weights anywhere → 1/n each
    assert rep["total_weight"] == pytest.approx(1.0)
    assert rep["truncated_weight"] == pytest.approx(1.0)
    assert len(rep["pending_high_sensitivity_events"]) == 30    # capped union, never unbounded


# ---------------------------------------------------------------- §21 bounds — pure arithmetic
def test_truncation_bounds_binary_hand_computed():
    dist = {"b0": "yes", "b1": "yes", "b2": "no"}               # 3 completed branches share 0.75
    b = truncation_bounds(dist, 0.25, ["yes", "no"])
    assert b["yes"] == {"lower": 0.5, "upper": 0.75}            # 2×0.25 completed; +0.25 if all resolve yes
    assert b["no"] == {"lower": 0.25, "upper": 0.5}


def test_truncation_bounds_three_option_hand_computed():
    dist = {"b0": "a", "b1": "b", "b2": "c", "b3": "a"}         # 4 completed branches share 0.8
    b = truncation_bounds(dist, 0.2, ["a", "b", "c"])
    assert b["a"] == {"lower": 0.4, "upper": 0.6}
    assert b["b"] == {"lower": 0.2, "upper": 0.4}
    assert b["c"] == {"lower": 0.2, "upper": 0.4}


def test_truncation_bounds_accepts_per_branch_distributions():
    dist = {"b0": {"yes": 0.5, "no": 0.5}, "b1": "yes"}         # completed mass 0.5 → 0.25/branch
    b = truncation_bounds(dist, 0.5, ["yes", "no"])
    assert b["yes"] == {"lower": 0.375, "upper": 0.875}
    assert b["no"] == {"lower": 0.125, "upper": 0.625}


def test_truncation_bounds_degenerate_without_truncated_mass():
    b = truncation_bounds({"b0": "yes", "b1": "no", "b2": "no", "b3": "yes"}, 0.0, ["yes", "no"])
    for o in ("yes", "no"):
        assert b[o]["lower"] == b[o]["upper"] == 0.5            # bounds collapse to the point estimate


# ---------------------------------------------------------------- §21 recommendation gate
def test_recommendation_eligibility_flips_exactly_when_margin_below_truncated_span():
    scores = {"act_a": 1.0, "act_b": 0.5}
    ok = recommendation_eligibility(scores, 0.25, (0.0, 1.0))
    # completed margin 0.75·0.5 = 0.375 ≥ span 0.25 → the leader survives every completion
    assert ok["eligible"] is True and ok["leader"] == "act_a"
    assert ok["margin_worst_case"] == pytest.approx(0.375 - 0.25)
    bad = recommendation_eligibility(scores, 0.5, (0.0, 1.0))
    # completed margin 0.5·0.5 = 0.25 < span 0.5 → an admissible completion flips it → withheld
    assert bad["eligible"] is False
    assert honest_note() in bad["why"]
    edge = recommendation_eligibility({"a": 1.0, "b": 0.0}, 0.5, (0.0, 1.0))
    assert edge["eligible"] is True and edge["margin_worst_case"] == 0.0   # margin == span survives
    solo = recommendation_eligibility({"only": 0.1}, 0.9, (0.0, 1.0))
    assert solo["eligible"] is True                              # no rival can overtake
    assert recommendation_eligibility({}, 0.1, (0.0, 1.0))["eligible"] is False


def test_honest_note_exact_string():
    assert honest_note() == ("truncated branch mass is unresolved simulation, "
                             "not Monte Carlo error")


# ---------------------------------------------------------------- §8/§35 result statuses — under_modeled
def test_under_modeled_accepts_subtype_and_truncated_accepts_report():
    um = SimulationResult(question="q", simulation_status="under_modeled",
                          support_grade="exploratory",
                          under_modeled_subtypes=["under_modeled_boundary"])
    assert um.simulation_status in SIMULATION_STATUSES
    assert set(um.under_modeled_subtypes) <= set(UNDER_MODELED_SUBTYPES)
    tr = SimulationResult(question="q", simulation_status="truncated", support_grade="exploratory",
                          truncation_report={"truncated_weight": 0.4, "honest_note": honest_note()})
    assert tr.truncation_report["truncated_weight"] == 0.4
    # a structured component alone is also sufficient (no subtype required)
    comp = SimulationResult(question="q", simulation_status="under_modeled",
                            support_grade="exploratory",
                            under_modeled_components=[{"component": "grid_physics",
                                                       "kind": "nonhuman_mechanism",
                                                       "why": "no validated load-flow model",
                                                       "sensitivity": "high"}])
    assert comp.under_modeled_components[0]["sensitivity"] == "high"


def test_under_modeled_without_subtype_or_component_raises():
    with pytest.raises(ValueError):
        SimulationResult(question="q", simulation_status="under_modeled",
                         support_grade="exploratory")


def test_under_modeled_and_truncated_require_a_support_grade():
    with pytest.raises(ValueError):
        SimulationResult(question="q", simulation_status="under_modeled",
                         support_grade="not_a_grade",
                         under_modeled_subtypes=["under_modeled_actor"])
    with pytest.raises(ValueError):
        SimulationResult(question="q", simulation_status="truncated", support_grade="not_a_grade",
                         truncation_report={"truncated_weight": 0.3})


def test_conditional_forecast_note_auto_set_on_partial_under_modeled():
    r = SimulationResult(question="q", simulation_status="under_modeled",
                         support_grade="exploratory",
                         under_modeled_subtypes=["under_modeled_external_process"],
                         raw_distribution={"yes": 0.5, "no": 0.5})
    d = r.as_dict()
    assert d["conditional_forecast_note"] == CONDITIONAL_FORECAST_SENTENCE
    assert d["conditional_forecast_note"] == ("This distribution is conditional on the represented "
                                              "world boundary and excludes unresolved "
                                              "high-sensitivity processes.")
    custom = SimulationResult(question="q", simulation_status="under_modeled",
                              support_grade="exploratory",
                              under_modeled_subtypes=["under_modeled_external_process"],
                              raw_distribution={"yes": 0.5, "no": 0.5},
                              conditional_forecast_note="already conditional on X")
    assert custom.as_dict()["conditional_forecast_note"] == "already conditional on X"


def test_has_forecast_semantics_for_every_status():
    # forecast-availability contract: availability is decided by probability/distribution
    # PRESENCE for every status uniformly — the completed family included. A live completed
    # run always carries its numbers; a bare completed shell carries no forecast, and no
    # status can bless one into existence (nor erase one that exists).
    for s in ("completed", "completed_with_degradation", "temporally_truncated"):
        assert not SimulationResult(question="q", simulation_status=s,
                                    support_grade="exploratory").has_forecast()
        assert SimulationResult(question="q", simulation_status=s, support_grade="exploratory",
                                raw_probability=0.62).has_forecast()
    assert not SimulationResult(question="q", simulation_status="clarification_required",
                                clarification_reason="incoherent").has_forecast()
    assert not SimulationResult(question="q", simulation_status="execution_failed",
                                failure_taxonomy="runtime_exception").has_forecast()
    # under_modeled/truncated: forecast present IFF a partial conditional distribution is
    um_kw = dict(question="q", simulation_status="under_modeled", support_grade="exploratory",
                 under_modeled_subtypes=["under_modeled_parameterization"])
    assert not SimulationResult(**um_kw).has_forecast()
    assert SimulationResult(**um_kw, raw_distribution={"yes": 0.6, "no": 0.4}).has_forecast()
    tr_kw = dict(question="q", simulation_status="truncated", support_grade="exploratory",
                 truncation_report={"truncated_weight": 0.5})
    assert not SimulationResult(**tr_kw).has_forecast()
    assert SimulationResult(**tr_kw, raw_distribution={"yes": 0.7, "no": 0.3}).has_forecast()


def test_abstain_mirror_never_set_for_under_modeled_or_truncated():
    um = SimulationResult(question="q", simulation_status="under_modeled",
                          support_grade="exploratory",
                          under_modeled_subtypes=["under_modeled_population"])
    tr = SimulationResult(question="q", simulation_status="truncated", support_grade="exploratory",
                          truncation_report={"truncated_weight": 0.2})
    assert um.as_dict()["abstain"] is False                     # NOT clarification
    assert tr.as_dict()["abstain"] is False
    assert um.as_dict()["_semantics"] == "no_abstention_v2"


def test_legacy_statuses_still_construct_unchanged():
    for s in ("completed", "completed_with_degradation", "temporally_truncated"):
        r = SimulationResult(question="q", simulation_status=s, support_grade="exploratory")
        # construction unchanged; abstain mirrors only genuine incoherence. Forecast
        # availability follows the probability, not the status (forecast-availability
        # contract) — a bare shell has none, a served number is one.
        assert not r.has_forecast() and r.as_dict()["abstain"] is False
        r.raw_probability = 0.4
        assert r.has_forecast()
    clar = SimulationResult(question="q", simulation_status="clarification_required",
                            clarification_reason="no coherent outcome")
    assert clar.as_dict()["abstain"] is True                    # only genuine incoherence mirrors
    fail = SimulationResult(question="q", simulation_status="execution_failed",
                            failure_taxonomy="runtime_exception")
    assert fail.as_dict()["abstain"] is False


def test_migrate_legacy_result_behavior_unchanged():
    # mirrors tests/test_no_abstention_contract.py — migration semantics must not move
    old = {"question": "q", "abstain": True, "abstain_reason": "question is ambiguous; cannot interpret"}
    m = migrate_legacy_result(old)
    assert m["simulation_status"] == "clarification_required" and m["_migrated"] is True
    assert m["abstain"] is True                                 # original field preserved, not edited
    old2 = {"question": "q", "abstain": True, "abstain_reason": "no executable mechanism for this slice"}
    m2 = migrate_legacy_result(old2)
    assert m2["simulation_status"] == "execution_failed"
    assert m2["failure_taxonomy"] == "missing_required_operator"
    old3 = {"question": "q", "abstain": False, "p": 0.6}
    m3 = migrate_legacy_result(old3)
    assert m3["simulation_status"] == "completed" and m3["_migrated"] is False
