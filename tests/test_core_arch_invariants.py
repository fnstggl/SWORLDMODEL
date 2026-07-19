"""§36 core-architecture invariant ENFORCEMENT tests (offline, deterministic, no network).

Each test is numbered against the §36 invariant list. The strict-mode block (25-31) proves the
production route CANNOT reach a numeric actor policy, a template personality, or the generic
outcome prior: those tests explicitly UNSET the offline test markers SWM_ALLOW_NUMERIC_BASELINE
and SWM_ALLOW_GENERIC_PRIOR that tests/conftest.py sets for the legacy comparison arena.

Bug found while writing these tests (fixed minimally in swm/world_model_v2/boundary_monitor.py):
`BoundaryMonitorOperator.run` called `world.entity(actor_id)` expecting it to CREATE the promoted
entity, but `WorldState.entity` only looks up and raises KeyError on unknown ids — §7.1 dynamic
actor promotion crashed on every real promotion. The operator now creates the Entity itself.
"""
import ast
import json
import pathlib
import random
import re
from types import SimpleNamespace

import pytest

import swm.world_model_v2.generated_world  # noqa: F401 — registers ctrl_* event types
from swm.world_model_v2 import bounded_cognition as bc
from swm.world_model_v2 import transitions
from swm.world_model_v2.boundary_monitor import (
    OUTSIDE_EVENT_TYPE, BoundaryMonitorOperator, OutsideWorldEntryOperator,
    schedule_outside_arrivals,
)
from swm.world_model_v2.bounded_cognition import (
    ActorMemoryState, BeliefRecord, EpisodicMemory, WorkingMemoryItem, WorkingMemoryState,
    action_search_stage, interpretation_stage, load_working_memory, memory_retrieval_stage,
    memory_update_stage, run_cognition_pipeline, store_memory, working_memory_stage,
)
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.fallback import GenericOutcomeOperator, generic_prior_allowed
from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.model_families import FamilyIdentityError, FamilyPool, ModelFamily
from swm.world_model_v2.outside_world import (
    ArrivalModel, ExternalEventFamily, FORBIDDEN_WRITES, OutsideWorldProcess,
    generate_outside_world, validate_entry,
)
from swm.world_model_v2.phase4_execution import ProductionActorPolicyOperator
from swm.world_model_v2.phase4_policy import ActorPolicyModel, ActorViewBuilder, UtilityInference
from swm.world_model_v2.phase_consumers import (
    AggregateOutcomeOperator, CollectiveThresholdDecisionOperator,
)
from swm.world_model_v2.qualitative_actor import (
    ActorDecisionUnavailable, QualitativeActorPolicyRuntime, QualitativeActorState,
    QualitativeConfig, QualitativeDecisionEngine, load_actor_state, store_actor_state,
)
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.temporal_runtime import (
    TemporalRunStats, aggregate_temporal_stats, get_stats,
)
from swm.world_model_v2.truncation import (
    aggregate_branch_statuses, honest_note, recommendation_eligibility,
)
from swm.world_model_v2.world_boundary import (
    BoundaryComponent, WorldBoundary, generate_world_boundary,
)
from tests.test_llm_actor import DECISION, Plan, T0, world
from tests.test_qualitative_actor import QLLM, qpayload

DAY = 86400.0
WM_V2_ROOT = pathlib.Path(transitions.__file__).parent


def _strict(monkeypatch):
    """Run the SUT exactly as production runs it: no offline baseline markers."""
    monkeypatch.delenv("SWM_ALLOW_NUMERIC_BASELINE", raising=False)
    monkeypatch.delenv("SWM_ALLOW_GENERIC_PRIOR", raising=False)


def _numeric_spies(monkeypatch):
    """Count every call into the numeric actor machinery (must stay 0 on the strict route)."""
    counters = {"decide": 0, "utility": 0}
    orig = ActorPolicyModel.decide
    monkeypatch.setattr(
        ActorPolicyModel, "decide",
        lambda self, *a, **k: counters.__setitem__("decide", counters["decide"] + 1)
        or orig(self, *a, **k))
    orig_u = UtilityInference.infer
    monkeypatch.setattr(
        UtilityInference, "infer",
        lambda self, *a, **k: counters.__setitem__("utility", counters["utility"] + 1)
        or orig_u(self, *a, **k))
    return counters


def _seed_state(w, actor="alice"):
    st = QualitativeActorState(actor_id=actor, hypothesis_id="h0:seeded",
                               identity_and_role=f"{actor}, manager",
                               core_worldview="steady and deliberate")
    store_actor_state(w, st, method="test_seed")
    return st


def make_min_world(*actors, now=T0, info=True):
    w = WorldState("corew", "b0", SimulationClock(now=now, as_of=now),
                   information=InformationLedger() if info else None)
    for a in actors:
        e = Entity(a)
        e.set("roles", F(["person"], status="observed"))
        w.entities[a] = e
    return w


BOUNDARY_JSON = {
    "components": [
        {"kind": "individual_actor", "name": "ceo_dana", "representation": "individual",
         "reason": "holds the final call", "evidence": ["board minutes"],
         "sensitivity": "decisive", "promotion_trigger": ""},
        {"kind": "population", "name": "retail_customers", "representation": "aggregate",
         "reason": "aggregate demand only", "evidence": [], "sensitivity": "material",
         "promotion_trigger": ""},
        {"kind": "nonhuman_system", "name": "payment_platform", "representation": "individual",
         "reason": "processes every transaction", "evidence": [], "sensitivity": "material",
         "promotion_trigger": ""},
        {"kind": "external_event_family", "name": "press_coverage",
         "representation": "external_process", "reason": "left to the residual environment",
         "evidence": [], "sensitivity": "minor", "promotion_trigger": ""},
        {"kind": "institution", "name": "antitrust_agency", "representation": "excluded",
         "reason": "no jurisdiction over this deal", "evidence": [],
         "sensitivity": "negligible", "promotion_trigger": ""},
    ],
    "sensitivity_assumptions": ["press coverage stays mild"],
    "boundary_evidence": ["deal docket"],
}


# =====================================================================================
# (1)(2) every generated boundary lists inside/outside/excluded with reasons
# =====================================================================================
def test_inv_1_2_generated_boundary_answers_all_boundary_questions():
    b = generate_world_boundary(question="Will the launch land?", structural_model_id="mA",
                                llm=lambda p: json.dumps(BOUNDARY_JSON))
    ans = b.boundary_answers()
    assert set(ans) == {"simulated_individually", "simulated_as_population",
                        "represented_as_nonhuman_system", "outside_detailed_world",
                        "inclusion_reasons", "exclusion_reasons", "boundary_evidence",
                        "omitted_components_that_could_change_answer", "expansion_triggers"}
    assert ans["simulated_individually"] == ["ceo_dana"]
    assert ans["simulated_as_population"] == ["retail_customers"]
    assert ans["represented_as_nonhuman_system"] == ["payment_platform"]
    # outside = residual external processes + explicit exclusions, both with reasons
    assert set(ans["outside_detailed_world"]) == {"press_coverage", "antitrust_agency"}
    assert ans["inclusion_reasons"]["ceo_dana"] == "holds the final call"
    assert ans["exclusion_reasons"]["antitrust_agency"] == "no jurisdiction over this deal"
    assert b.generation_trace[0]["stage"] == "boundary_generation"
    assert b.generation_trace[0]["ok"] is True
    # the boundary is generated, not silently inferred: with no LLM it records the failure and
    # registers a REAL unresolved component (never a clean boundary_supported)
    b_none = generate_world_boundary(question="q", structural_model_id="mB", llm=None)
    assert b_none.generation_trace[0]["error"] == "no_llm_backend"
    assert b_none.unresolved_components
    assert b_none.classify_support() != "boundary_supported"


# =====================================================================================
# (3) residual outside world: families, or a JUSTIFIED empty residual
# =====================================================================================
def test_inv_3_empty_residual_requires_explicit_justification():
    boundary = WorldBoundary(boundary_id="wb3", structural_model_id="m3", question="q")
    justified = generate_outside_world(
        boundary, llm=lambda p: json.dumps({
            "families": [],
            "empty_residual_justification": "single-room negotiation; no external process "
                                            "can plausibly reach the outcome"}))
    assert justified.families == []
    assert "single-room negotiation" in justified.empty_residual_justification
    assert justified.unresolved() == [] and justified.unresolved_external_risks == []
    # an empty residual WITHOUT justification stays visibly unjustified (downstream
    # classification keys on the empty string — §36.3)
    unjustified = generate_outside_world(boundary, llm=lambda p: json.dumps({"families": []}))
    assert unjustified.families == [] and unjustified.empty_residual_justification == ""
    # LLM failure is recorded loudly and never manufactures a justification
    failed = generate_outside_world(boundary, llm=None)
    assert failed.empty_residual_justification == ""
    assert failed.generation_trace[0]["error"] == "no_llm_backend"


# =====================================================================================
# (4) outside events cannot claim terminal/actor-reaction writes (validate_entry)
# =====================================================================================
def test_inv_4_validate_entry_rejects_terminal_and_reaction_writes():
    def fam(**kw):
        base = dict(family_id="fx", description="d", impact_mechanism="observation_delivery",
                    impact_description="delivers a news item")
        base.update(kw)
        return ExternalEventFamily(**base)

    f1 = validate_entry(fam(impact_mechanism="write_answer_directly"))
    assert f1.validation_error and "not a typed" in f1.validation_error
    f2 = validate_entry(fam(impact_description="writes the forecast answer for the run"))
    assert "'forecast_answer'" in f2.validation_error
    f3 = validate_entry(fam(affected_boundary_components=["terminal utility of the plan"]))
    assert "'terminal_utility'" in f3.validation_error
    f4 = validate_entry(fam(affected_boundary_components=["actor reaction of the ceo"]))
    assert "'actor_reaction'" in f4.validation_error
    f5 = validate_entry(fam(impact_description="bumps the recommendation rank"))
    assert "'recommendation_rank'" in f5.validation_error
    # every forbidden target is actually screened
    for bad in FORBIDDEN_WRITES:
        f = validate_entry(fam(impact_description=f"writes {bad.replace('_', ' ')} directly"))
        assert f.validation_error, bad
    # rejected families are never samplable
    proc = OutsideWorldProcess(boundary_id="b", families=[f1])
    assert proc.samplable() == [] and proc.unresolved() == [f1]


# =====================================================================================
# (5) the entry operator never writes terminal/readout paths into the world
# =====================================================================================
def test_inv_5_outside_entry_operator_never_writes_terminal_paths():
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    w = make_min_world("alice")
    register_quantity_type("warehouse_capacity", units="units")
    w.quantities["warehouse_capacity"] = Quantity(name="warehouse_capacity",
                                                  qtype="warehouse_capacity", value=100.0,
                                                  timestamp=T0)
    register_quantity_type("readout", units="outcome")
    op = OutsideWorldEntryOperator(report={})
    cases = [
        {"outside_world_family": "press", "entry_mechanism": "observation_delivery",
         "mark": "leak article published", "affected_boundary_components": []},
        {"outside_world_family": "supply", "entry_mechanism": "capacity_change",
         "mark": "port closure", "affected_boundary_components": ["warehouse capacity"]},
        {"outside_world_family": "rules", "entry_mechanism": "institutional_rule_change",
         "mark": "new filing requirement", "affected_boundary_components": []},
        {"outside_world_family": "market", "entry_mechanism": "price_change",
         "mark": "input price spike", "affected_boundary_components": ["nonexistent thing"]},
    ]
    for payload in cases:
        ev = Event(ts=T0, etype=OUTSIDE_EVENT_TYPE, participants=[], payload=payload)
        delta, vr = op.run(w, ev, random.Random(0))
        assert vr.ok
        for ch in delta.changes:
            path = str(ch["path"])
            assert not path.startswith("quantities[readout]")
            for bad in FORBIDDEN_WRITES:
                assert bad not in path.lower(), (payload["entry_mechanism"], path)
    # the matching capacity quantity got a disturbance FLAG, never an invented magnitude
    assert w.quantities["warehouse_capacity"].value == 100.0
    assert "readout" not in w.quantities
    assert op.report["outside_events_entered"] == 4
    # a resource-like impact with no matching quantity is recorded unresolved, not invented
    assert any(r["mechanism"] == "price_change" for r in op.report["outside_unresolved_impacts"])
    # mechanisms whose full effect is not representable enter as observations, recorded
    assert any(d["declared_mechanism"] == "institutional_rule_change"
               for d in op.report["outside_entry_downgrades"])
    assert [d["quantity"] for d in op.report["outside_quantity_disturbances"]] == \
        ["warehouse_capacity"]


# =====================================================================================
# (6) decisive omitted component ⇒ under_modeled_boundary
# =====================================================================================
def test_inv_6_decisive_omitted_component_classifies_under_modeled_boundary():
    b = WorldBoundary(boundary_id="wb6", structural_model_id="m6", question="q")
    b.components.append(BoundaryComponent(component_id="c1", kind="nonhuman_system",
                                          name="grid_load", representation="unresolved",
                                          reason="no validated model", sensitivity="decisive"))
    b.rederive_views()
    assert b.classify_support() == "under_modeled_boundary"
    assert [u["name"] for u in b.unresolved_decisive()] == ["grid_load"]
    # a sensitivity-analysis row that can reverse the forecast is decisive too
    b2 = WorldBoundary(boundary_id="wb6b", structural_model_id="m6", question="q")
    b2.omitted_component_sensitivity.append(
        {"omitted_component": "labor_union", "kind": "actor_group",
         "can_reverse_forecast_direction": True, "can_change_best_action": False})
    assert b2.classify_support() == "under_modeled_boundary"
    # without decisive omissions and with closed findings the boundary is supported
    b3 = WorldBoundary(boundary_id="wb6c", structural_model_id="m6", question="q")
    b3.components.append(BoundaryComponent(component_id="c2", kind="individual_actor",
                                           name="dana", representation="individual",
                                           reason="in", sensitivity="decisive"))
    b3.rederive_views()
    assert b3.classify_support() == "boundary_supported"


# =====================================================================================
# (7)(8) runtime promotion at event time; promoted actor sees PUBLIC items only
# =====================================================================================
def test_inv_7_8_boundary_monitor_promotes_at_event_time_public_history_only():
    w = make_min_world("alice", now=T0 + 100)
    w.clock.as_of = T0
    w.information.publish(InformationItem("pub1", "public court filing", kind="public",
                                          created_at=T0 - 10))
    w.information.publish(InformationItem("priv1", "private strategy memo", kind="private",
                                          created_at=T0 - 10))
    b = WorldBoundary(boundary_id="wb7", structural_model_id="m7", question="q")
    b.components.append(BoundaryComponent(
        component_id="c1", kind="individual_actor", name="regulator_kim",
        representation="external_process", reason="outside for now",
        promotion_trigger="regulator_kim opens an inquiry"))
    b.promotion_rules = [{"component": "regulator_kim",
                          "trigger": "regulator_kim opens an inquiry",
                          "action": "promote_to_individual"}]
    b.rederive_views()
    op = BoundaryMonitorOperator(b, report={})
    ev = Event(ts=T0 + 100, etype="ctrl_semantic_event", participants=[],
               payload={"semantic_event": {
                   "event_id": "s1",
                   "exact_content": "breaking: regulator_kim opens an inquiry into the merger"}})
    delta, vr = op.run(w, ev, random.Random(0))
    assert vr.ok and "boundary_promotion:regulator_kim" in delta.reason_codes
    # (7) the entity exists now, promoted at the EXACT event time
    assert "regulator_kim" in w.entities
    rec = w.boundary_promotions[0]
    assert rec["component"] == "regulator_kim" and rec["at"] == T0 + 100
    assert b.dynamic_promotions[0]["at"] == T0 + 100
    assert b.component("regulator_kim").promoted_at == T0 + 100
    assert b.component("regulator_kim").representation == "individual"
    # (8) reconstructed history/exposure = PUBLIC items only — never the private memo
    exposed = [(x.actor_id, x.item_id) for x in w.information.exposures
               if x.actor_id == "regulator_kim"]
    assert exposed == [("regulator_kim", "pub1")]
    assert rec["reconstructed_history_events"] == 1
    # promotion record marks the lazy-cognition seed, not an omniscient snapshot
    promo = w.entities["regulator_kim"].value("latent_state", key="boundary_promotion_record")
    assert promo["promoted_by_boundary_monitor"] is True and promo["promoted_at"] == T0 + 100
    # a second matching event does not double-create
    delta2, vr2 = op.run(w, ev, random.Random(0))
    assert delta2 is None and "already_inside_boundary" in vr2.reasons


# =====================================================================================
# (9) promoting a population member records the segment decrement
# =====================================================================================
def test_inv_9_population_promotion_records_segment_decrement():
    w = make_min_world("alice", now=T0 + 50)
    w.populations = {"veteran_customers":
                     SimpleNamespace(description="veteran customers of the retail arm")}
    b = WorldBoundary(boundary_id="wb9", structural_model_id="m9", question="q")
    b.promotion_rules = [{"component": "veteran_customer", "trigger": "escalates a complaint",
                          "action": "promote_to_individual"}]
    b.rederive_views()
    op = BoundaryMonitorOperator(b, report={})
    ev = Event(ts=T0 + 50, etype="ctrl_semantic_event", participants=[],
               payload={"semantic_event": {
                   "event_id": "s9",
                   "exact_content": "a veteran_customer escalates a complaint publicly"}})
    delta, vr = op.run(w, ev, random.Random(0))
    assert vr.ok and delta is not None
    assert w.population_promotions == [{"segment": "veteran_customers",
                                        "person": "veteran_customer", "at": T0 + 50}]
    assert w.boundary_promotions[0]["population_decrement"]["segment"] == "veteran_customers"


# =====================================================================================
# (11)(12) cognition stages are separate; the decision prompt cannot see unnoticed items
# =====================================================================================
NOTICED_MARK = "OUTAGE_IN_EAST_CLUSTER_7Q"
MISSED_MARK = "ZULU_DISCOUNT_CODE_55X"


def _cog_llm(prompt):
    if "ATTENTION process" in prompt:
        return json.dumps({"noticed": [], "missed": []})
    if "private sense" in prompt:
        return json.dumps({"what_happened": "the east cluster went down on my watch",
                           "why_it_matters": "customers notice within minutes",
                           "unresolved_ambiguity": "how long recovery takes"})
    if "options even OCCUR" in prompt:
        return json.dumps({"shortlist": ["page the on-call vendor"],
                           "options_screened_out": []})
    raise AssertionError("unexpected prompt " + prompt[:50])


def test_inv_11_12_stages_separate_and_prompt_excludes_unnoticed_observations():
    w = make_min_world("maya")
    cog = run_cognition_pipeline(
        world=w, actor_id="maya", branch_id="bX", at=T0 + 60,
        available=[
            {"obs_id": "o_int", "channel": "pager", "source": "monitoring",
             "summary": f"{NOTICED_MARK} alarm firing", "interrupting": True},
            {"obs_id": "o_mut", "channel": "newsletter", "source": "vendor",
             "summary": f"{MISSED_MARK} promo offer"},
        ],
        identity="maya, on-call engineer",
        attention_context={"muted_channels": ["newsletter"]},
        rng=random.Random(0), llm=_cog_llm, family_id="famT")
    # (11) the pipeline is separate named stages, each with its own trace record
    assert [t["stage"] for t in cog.stage_traces] == \
        ["attention", "working_memory", "memory_retrieval", "interpretation", "action_search"]
    assert len({t["stage"] for t in cog.stage_traces}) == 5
    # (12) the decision prompt contains the noticed observation, never the missed one
    assert {m["obs_id"] for m in cog.attention["missed"]} == {"o_mut"}
    view = ActorViewBuilder().build(w, "maya")
    engine = QualitativeDecisionEngine(QualitativeConfig(llm=lambda p: "{}",
                                                         llm_hypotheses=False,
                                                         bounded_cognition=False))
    prompt = engine.build_prompt(view, None, "a decision point",
                                 [{"line": "- wait"}], cognition=cog)
    assert NOTICED_MARK in prompt
    assert MISSED_MARK not in prompt


# =====================================================================================
# (13) working memory is finite AND stateful across sequential invocations
# =====================================================================================
def test_inv_13_working_memory_finite_and_stateful_across_two_invocations():
    w = make_min_world("maya")
    def avail(tag, n):
        return [{"obs_id": f"{tag}{i}", "channel": "pager", "source": "mon",
                 "summary": f"{tag} incident number {i}", "interrupting": True}
                for i in range(n)]
    ctx = {"workload": "crisis overload", "urgency": "urgent deadline"}   # capacity 2
    cog1 = run_cognition_pipeline(world=w, actor_id="maya", branch_id="bS", at=T0 + 10,
                                  available=avail("alpha", 4), attention_context=ctx,
                                  rng=random.Random(0), llm=_cog_llm)
    cap1 = cog1.working_memory["capacity"]
    assert cap1 == 2
    # finite at DECISION time: the prompt-facing active view respects the situational capacity
    assert len(cog1.working_memory["active_items"]) <= cap1
    wm_after_1 = load_working_memory(w, "maya")
    # the stage-managed slots respect capacity; the interpretation item rides one extra slot
    # until the next invocation's stage clamps it (documented pipeline order)
    assert len([i for i in wm_after_1.active() if i.kind != "interpretation"]) <= cap1
    assert wm_after_1.displaced_log                             # overflow displaced, logged
    ids_after_1 = {i.item_id for i in wm_after_1.items}
    assert len(ids_after_1) >= 4                                # 4 noticed + interpretation
    cog2 = run_cognition_pipeline(world=w, actor_id="maya", branch_id="bS", at=T0 + 900,
                                  available=avail("beta", 3), attention_context=ctx,
                                  rng=random.Random(1), llm=_cog_llm)
    assert len(cog2.working_memory["active_items"]) <= cog2.working_memory["capacity"]
    wm_after_2 = load_working_memory(w, "maya")
    assert len([i for i in wm_after_2.active() if i.kind != "interpretation"]) \
        <= wm_after_2.capacity_last
    # STATEFUL: invocation 2 evolved the SAME store — every item from invocation 1 is still
    # accounted for (active or displaced with a logged reason), not recomputed from scratch
    ids_after_2 = {i.item_id for i in wm_after_2.items}
    assert ids_after_1 <= ids_after_2
    assert cog2.working_memory["displaced"] or cog2.working_memory["entered"]


# =====================================================================================
# (14) working memory is NOT last-N-of-global-events: refresh beats recency of entry
# =====================================================================================
def test_inv_14_old_refreshed_item_survives_newer_unrefreshed_item():
    wm = WorkingMemoryState(actor_id="a")
    wm.items.append(WorkingMemoryItem(item_id="w_old", kind="observation",
                                      content="the audit question from monday",
                                      entered_at=T0, refreshed_at=T0, source="o_old"))
    wm.items.append(WorkingMemoryItem(item_id="w_new", kind="observation",
                                      content="tuesday's minor status ping",
                                      entered_at=T0 + 500, refreshed_at=T0 + 500,
                                      source="o_new"))
    # the OLD item is re-noticed (refreshed) — same source, no duplicate
    out1 = working_memory_stage(wm=wm, actor_id="a", branch_id="b", at=T0 + 900,
                                noticed=[{"obs_id": "o_old"}],
                                available_by_id={"o_old": {"summary": "audit again"}})
    assert out1["refreshed"] == ["w_old"]
    # a third item arrives under tight capacity (crisis + urgent → 2 slots)
    working_memory_stage(wm=wm, actor_id="a", branch_id="b", at=T0 + 1000,
                         noticed=[{"obs_id": "o_c"}],
                         available_by_id={"o_c": {"summary": "brand new interrupt"}},
                         attention_context={"workload": "crisis", "urgency": "urgent"})
    active = {i.item_id for i in wm.active()}
    # last-N-of-global-events would keep the two most recent ENTRIES (w_new + the new one).
    # Working memory instead displaced the stalest-by-refresh item: w_new is gone, the OLDER
    # but refreshed w_old survives.
    assert "w_old" in active
    assert "w_new" not in active
    assert {d["item_id"] for d in wm.displaced_log} == {"w_new"}


# =====================================================================================
# (15) long-term retrieval can FAIL (mechanism documented, draw branch-seeded)
# =====================================================================================
def test_inv_15_retrieval_failure_is_real_and_branch_seeded():
    def setup():
        wm = WorkingMemoryState(actor_id="a")
        wm.capacity_last = 5
        wm.items.append(WorkingMemoryItem(item_id="w1", kind="observation",
                                          content="the vendor invoice arrived again",
                                          entered_at=T0, refreshed_at=T0, source="o1"))
        mem = ActorMemoryState(actor_id="a")
        mem.episodic.append(EpisodicMemory(memory_id="m_old", at=T0 - 40 * DAY,
                                           content="a forgettable note",
                                           retrieval_cues=["vendor invoice"],
                                           salience="low"))     # stale + low → fail band 0.6
        return wm, mem

    wm1, mem1 = setup()
    out_fail = memory_retrieval_stage(mem=mem1, wm=wm1, actor_id="a", branch_id="b",
                                      at=T0, rng=random.Random(1))
    assert out_fail["retrieved"] == []
    assert out_fail["retrieval_failures"][0]["memory_id"] == "m_old"
    assert "retrieval failure" in out_fail["retrieval_failures"][0]["why"]
    assert mem1.episodic[0].times_recalled == 0                 # the miss left no rehearsal
    wm2, mem2 = setup()
    out_ok = memory_retrieval_stage(mem=mem2, wm=wm2, actor_id="a", branch_id="b",
                                    at=T0, rng=random.Random(2))
    assert out_ok["retrieved"] == ["m_old"]                     # another branch retrieves it


# =====================================================================================
# (16) contradictory beliefs persist un-averaged (reinforcement touches one record)
# =====================================================================================
def test_inv_16_contradictory_beliefs_persist_unaveraged():
    mem = ActorMemoryState(actor_id="a")
    mem.beliefs.append(BeliefRecord(belief_id="b1", content="the vendor is reliable",
                                    conflicts_with="b2"))
    mem.beliefs.append(BeliefRecord(belief_id="b2", content="the vendor cuts corners",
                                    conflicts_with="b1"))
    assert len(mem.active_contradictions()) == 1
    out = memory_update_stage(
        mem=mem, wm=WorkingMemoryState(actor_id="a"), actor_id="a", branch_id="b", at=T0 + 5,
        interpretation={"what_happened": "the vendor shipped late again",
                        "active_belief": "the vendor cuts corners",
                        "perceived_threats": ["schedule slip"]},
        decision={"chosen_action": "audit_the_vendor"}, noticed=[{"obs_id": "o"}],
        rng=random.Random(0))
    assert out["belief_reinforced"] == "b2"
    # BOTH records still exist verbatim; the conflicting one was neither deleted nor blended
    assert [b.content for b in mem.beliefs] == ["the vendor is reliable",
                                                "the vendor cuts corners"]
    assert mem.beliefs[0].last_reinforced == 0.0
    assert mem.beliefs[1].last_reinforced == T0 + 5
    assert len(mem.active_contradictions()) == 1                # contradiction NOT resolved


# =====================================================================================
# (17) same observation, different interpretations across particles
# =====================================================================================
def test_inv_17_same_observation_different_interpretation_per_hypothesis():
    def particle(hypothesis_belief, llm):
        wm = WorkingMemoryState(actor_id="a")
        wm.items.append(WorkingMemoryItem(item_id="w1", kind="observation",
                                          content="the rival cut prices by a third",
                                          entered_at=T0, refreshed_at=T0, source="o1"))
        mem = ActorMemoryState(actor_id="a")
        mem.beliefs.append(BeliefRecord(belief_id="b", content=hypothesis_belief))
        return interpretation_stage(actor_id="a", branch_id="b", at=T0, identity="a founder",
                                    wm=wm, retrieved={"retrieved": []}, mem=mem, llm=llm)

    def keyed_llm(prompt):
        if "price wars end in ruin" in prompt:
            return json.dumps({"what_happened": "they are trying to bleed me out",
                               "active_belief": "price wars end in ruin",
                               "unresolved_ambiguity": ""})
        return json.dumps({"what_happened": "they are clearing old inventory, not attacking",
                           "active_belief": "rivals discount to clear stock",
                           "unresolved_ambiguity": ""})

    i1 = particle("price wars end in ruin", keyed_llm)
    i2 = particle("rivals discount to clear stock", keyed_llm)
    assert i1["what_happened"] != i2["what_happened"]
    assert i1["active_belief"] != i2["active_belief"]


# =====================================================================================
# (18) the searched shortlist may exclude a known feasible action — recorded
# =====================================================================================
def test_inv_18_shortlist_may_exclude_known_feasible_action():
    def llm(prompt):
        return json.dumps({"shortlist": ["call the client directly"],
                           "options_screened_out": [{"option": "wait it out",
                                                     "why_dismissed": "feels passive"}]})
    known = ["call the client directly", "escalate to the board", "file a complaint"]
    out = action_search_stage(actor_id="a", branch_id="b", at=T0, identity="lead",
                              interpretation={"what_happened": "churn risk"},
                              wm=WorkingMemoryState(actor_id="a"),
                              mem=ActorMemoryState(actor_id="a"),
                              known_options=known, llm=llm)
    assert out["shortlist"] == ["call the client directly"]
    assert out["actually_feasible_not_considered"] == ["escalate to the board",
                                                       "file a complaint"]


# =====================================================================================
# (19) QualitativeActorState persists across two decisions on one branch world
# =====================================================================================
def test_inv_19_actor_state_persists_across_two_decisions_on_one_branch():
    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=QLLM(), llm_hypotheses=False,
                                                    bounded_cognition=False)),
        mode="persistent_qualitative_llm_policy")
    w = world()
    sel1, post1, tr1 = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=1)
    rt.execute(w, sel1, post1, tr1, seed=1)
    s1 = load_actor_state(w, "alice")
    n_rev1, n_mem1 = len(s1.revision_log), len(s1.important_memories)
    assert n_mem1 >= 1
    sel2, post2, tr2 = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=2)
    rt.execute(w, sel2, post2, tr2, seed=2)
    s2 = load_actor_state(w, "alice")
    assert s2.hypothesis_id == s1.hypothesis_id                 # same particle, same hypothesis
    assert len(s2.revision_log) > n_rev1                        # the state EVOLVED in place
    assert len(s2.important_memories) >= n_mem1
    assert s2.identity_and_role == s1.identity_and_role         # identity never rewritten
    # the second prompt was conditioned on the persisted state, not a fresh template
    assert post2.provenance["qualitative"]["state_hash"]


# =====================================================================================
# (20)(21)(22) model families: recorded assignment, identity rule, monoculture
# =====================================================================================
def _fam(fid, provider, model, *, lineage="", tier="strong"):
    return ModelFamily(family_id=fid, provider=provider, model=model, lineage=lineage,
                       strength_tier=tier, availability="configured", client=lambda p: "ok")


def test_inv_20_family_assignments_recorded_and_deterministic():
    pool = FamilyPool()
    pool.register(_fam("famA", "provA", "model-a", lineage="la"))
    pool.register(_fam("famB", "provB", "model-b", lineage="lb"))
    picks = [pool.assign(particle_index=i, actor_id="dana") for i in range(24)]
    assert set(picks) == {"famA", "famB"}
    assert len(pool.assignment_log) == 24
    row = pool.assignment_log[0]
    assert set(row) >= {"particle", "actor", "family", "rule"}
    for i in (0, 7, 23):                                        # deterministic in (particle, actor)
        assert pool.assign(particle_index=i, actor_id="dana", record=False) == picks[i]


def test_inv_21_same_provider_model_reregistration_is_family_identity_error():
    pool = FamilyPool()
    pool.register(_fam("cold", "prov", "model-x"))
    with pytest.raises(FamilyIdentityError, match="NOT different families"):
        pool.register(_fam("hot_temperature_variant", "prov", "model-x"))


def test_inv_22_monoculture_true_with_one_lineage_and_reported():
    pool = FamilyPool()
    pool.register(_fam("chat", "prov", "model-chat", lineage="base-v1"))
    pool.register(_fam("reasoner", "prov2", "model-reasoner", lineage="base-v1"))
    assert pool.monoculture() is True                           # two models, ONE lineage
    rep = pool.report()
    assert rep["model_family_monoculture"] is True
    assert "correlated failure risk" in rep["monoculture_note"]
    assert rep["distinct_lineages"] == ["base-v1"]


# =====================================================================================
# (25)(26) STRICT: budget exhaustion raises; the numeric policy is NEVER invoked
# =====================================================================================
def test_inv_25_26_budget_exhaustion_raises_and_never_calls_numeric_policy(monkeypatch):
    _strict(monkeypatch)
    counters = _numeric_spies(monkeypatch)
    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=QLLM(), llm_hypotheses=False,
                                                    max_llm_calls=0, bounded_cognition=False)),
        mode="persistent_qualitative_llm_policy")
    w = world()
    _seed_state(w)
    with pytest.raises(ActorDecisionUnavailable) as ei:
        rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=1)
    assert ei.value.reason == "actor_llm_budget_exhausted"
    assert ei.value.actor_id == "alice"
    assert counters == {"decide": 0, "utility": 0}              # the equations never ran
    assert rt.decision_records == []                            # and no decision was minted


# =====================================================================================
# (27) STRICT: unparseable output after the retry ladder raises; zero numeric calls
# =====================================================================================
def test_inv_27_unparseable_llm_raises_after_retries_without_numeric_calls(monkeypatch):
    _strict(monkeypatch)
    counters = _numeric_spies(monkeypatch)
    garbage = QLLM(lambda p: "utterly not json")
    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=garbage, llm_hypotheses=False,
                                                    retries=1, bounded_cognition=False)),
        mode="persistent_qualitative_llm_policy")
    w = world()
    _seed_state(w)
    with pytest.raises(ActorDecisionUnavailable) as ei:
        rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=2)
    assert ei.value.reason == "unparseable_after_retries"
    assert len(garbage.prompts) == 2                            # initial + one retry, then stop
    assert counters == {"decide": 0, "utility": 0}


# =====================================================================================
# (28) STRICT: provider failure across every family raises; zero numeric calls
# =====================================================================================
def test_inv_28_provider_failure_all_families_raises(monkeypatch):
    _strict(monkeypatch)
    counters = _numeric_spies(monkeypatch)

    def down(prompt):
        raise ConnectionError("provider down")

    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=down, llm_hypotheses=False,
                                                    retries=0, bounded_cognition=False)),
        mode="persistent_qualitative_llm_policy")
    w = world()
    _seed_state(w)
    with pytest.raises(ActorDecisionUnavailable) as ei:
        rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=3)
    assert ei.value.reason == "provider_failure_all_families"
    assert ei.value.family_transitions                          # the ladder was walked, recorded
    assert counters == {"decide": 0, "utility": 0}


# =====================================================================================
# (29) STRICT: tier-3 routing promotes to qualitative instead of numeric
# =====================================================================================
def test_inv_29_tier3_actor_promoted_to_qualitative_in_strict_mode(monkeypatch):
    _strict(monkeypatch)
    counters = _numeric_spies(monkeypatch)
    llm = QLLM()
    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=llm, llm_hypotheses=False,
                                                    bounded_cognition=False)),
        mode="hybrid_relevant_actor_policy", tiers={}, selector=None)
    w = world()
    _seed_state(w)
    sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=4)
    q = post.provenance["qualitative"]
    assert q["routed"] is True
    assert q["decision_source"] == "persistent_qualitative_llm"
    assert "routine_actor_promoted_to_qualitative" in q["tier"]["integrity_promotion"]
    assert rt.tiers["alice"]["tier"] == 1                       # promotion persisted for the run
    assert llm.decision_prompts                                 # the MIND decided
    assert counters == {"decide": 0, "utility": 0}              # never the numeric policy


# =====================================================================================
# (30) the numeric route exists ONLY behind the marker / explicit-baseline integrity
# =====================================================================================
def test_inv_30_numeric_route_requires_marker_or_explicit_baseline(monkeypatch):
    _strict(monkeypatch)
    counters = _numeric_spies(monkeypatch)
    # strict: the multi-particle numeric bridge REFUSES
    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=QLLM(), llm_hypotheses=False,
                                                    bounded_cognition=False)),
        mode="persistent_qualitative_llm_policy")
    with pytest.raises(ActorDecisionUnavailable) as ei:
        rt.decide(Plan(), [world(), world()], "alice", decision=dict(DECISION), seed=5)
    assert ei.value.reason == "multi_particle_bridge_requires_explicit_baseline"
    assert counters == {"decide": 0, "utility": 0}
    # the SAME call on the explicitly named baseline arm reaches the numeric machinery
    rt_base = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=QLLM(), llm_hypotheses=False,
                                                    bounded_cognition=False,
                                                    integrity="numeric_baseline_explicit")),
        mode="persistent_qualitative_llm_policy")
    _, post, _ = rt_base.decide(Plan(), [world(), world()], "alice",
                                decision=dict(DECISION), seed=5)
    assert post.provenance["qualitative"]["decision_source"] == "numeric_policy"
    assert post.provenance["qualitative"]["explicit_baseline_arm"] is True
    assert counters["decide"] >= 1                              # numeric ran, loudly marked
    # the offline env marker re-opens the same arena without changing integrity
    monkeypatch.setenv("SWM_ALLOW_NUMERIC_BASELINE", "1")
    before = counters["decide"]
    _, post2, _ = rt.decide(Plan(), [world(), world()], "alice",
                            decision=dict(DECISION), seed=6)
    assert post2.provenance["qualitative"]["decision_source"] == "numeric_policy"
    assert counters["decide"] > before


# =====================================================================================
# (31) STRICT: after ActorDecisionUnavailable the branch HALTS and the loop STOPS
# =====================================================================================
def test_inv_31_production_operator_halts_branch_and_temporal_loop_stops(monkeypatch):
    _strict(monkeypatch)
    counters = _numeric_spies(monkeypatch)

    def down(prompt):
        raise ConnectionError("provider down")

    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=down, llm_hypotheses=False,
                                                    retries=0, bounded_cognition=False)),
        mode="persistent_qualitative_llm_policy")
    op = ProductionActorPolicyOperator(runtime=rt)
    w = world()
    _seed_state(w)
    horizon = T0 + 10 * DAY
    q = EventQueue(horizon_ts=horizon)
    q.schedule(Event(ts=T0 + 60, etype="decision_opportunity", participants=["alice"],
                     payload=dict(DECISION)))
    q.schedule(Event(ts=T0 + 2 * DAY, etype="decision_opportunity", participants=["alice"],
                     payload=dict(DECISION)))                   # must never run
    branch = RolloutEngine(operators=[op]).run_branch(w, q, seed=0)
    stats = branch.temporal_stats
    assert stats.branch_halted is True
    assert stats.branch_status == "truncated_provider_failure"  # mapped §20 status
    assert stats.temporally_truncated is True
    assert stats.truncation["reason"] == "provider_failure_all_families"
    assert stats.truncation["at_ts"] == T0 + 60                 # stopped at the exact trigger
    assert stats.truncation["unresolved_decision_trigger"]["actor_id"] == "alice"
    # the loop STOPPED: the queue still holds the later pending event, clock < horizon
    assert [e.ts for e in q.events] == [T0 + 2 * DAY]
    assert stats.truncation["pending_events"] and stats.pending_at_horizon
    assert w.clock.now == T0 + 60 < horizon
    assert stats.event_counts.get("decision_opportunity") == 1  # the second never processed
    # NO substitute action anywhere
    assert w.entity("alice").value("past_actions", default=[]) == []
    assert counters == {"decide": 0, "utility": 0}


def test_inv_31_generated_invocation_operator_halts_branch(monkeypatch):
    _strict(monkeypatch)
    from swm.world_model_v2.generated_world import (GeneratedActorInvocationOperator,
                                                    generated_report)
    from swm.world_model_v2.scenario_schema import ScenarioSemanticModel

    class StubRuntime:                                          # the §19 refusal, surfaced raw
        def decide(self, plan, worlds, actor_id, **kw):
            raise ActorDecisionUnavailable("budget gone", reason="actor_llm_budget_exhausted",
                                           actor_id=actor_id)

    w = make_min_world("maya", info=False)
    w.scenario_schema = ScenarioSemanticModel(schema_id="s1", question="q")
    report = generated_report()
    op = GeneratedActorInvocationOperator(StubRuntime(), report=report)
    horizon = T0 + 10 * DAY
    q = EventQueue(horizon_ts=horizon)
    q.schedule(Event(ts=T0 + 60, etype="ctrl_invoke_actor", participants=["maya"],
                     payload={"actor_id": "maya",
                              "triggering_semantic_event": {"event_id": "s1"},
                              "reason_actor_may_be_causally_relevant": "noticed the filing"}))
    q.schedule(Event(ts=T0 + 3 * DAY, etype="ctrl_invoke_actor", participants=["maya"],
                     payload={"actor_id": "maya"}))
    branch = RolloutEngine(operators=[op]).run_branch(w, q, seed=0)
    stats = branch.temporal_stats
    assert stats.branch_halted is True
    assert stats.branch_status == "truncated_actor_budget"      # mapped from the exception reason
    assert stats.truncation["halted"] is True
    assert stats.truncation["at_ts"] == T0 + 60
    assert w.clock.now == T0 + 60 < horizon
    assert [e.ts for e in q.events] == [T0 + 3 * DAY]           # pending event preserved
    rec = report["temporal_truncations"][0]
    assert rec["kind"] == "actor_llm_budget_exhausted"
    assert rec["branch_status"] == "truncated_actor_budget"
    assert rec["unresolved_decision_trigger"]["actor_id"] == "maya"
    assert report["actor_actions_executed"] == 0                # no substitute action


# =====================================================================================
# (32) aggregation preserves truncated weight — never renormalized away
# =====================================================================================
def test_inv_32_aggregate_branch_statuses_preserves_truncated_weight():
    stats = [
        {"branch_id": "b0", "truncated": False, "weight": 0.5},
        {"branch_id": "b1", "truncated": True, "weight": 0.3,
         "truncation": {"reason": "actor_llm_budget_exhausted", "at_ts": 120.0,
                        "affected_actors": ["dana"]}},
        {"branch_id": "b2", "invalid": True, "weight": 0.2},
    ]
    rep = aggregate_branch_statuses(stats)
    assert rep["completed_weight"] == 0.5
    assert rep["truncated_weight"] == 0.3                        # visible, not renormalized
    assert rep["invalid_weight"] == 0.2
    assert rep["completed_weight"] + rep["truncated_weight"] + rep["invalid_weight"] \
        == rep["total_weight"]
    assert rep["truncation_reasons"] == {"actor_llm_budget_exhausted": 1}
    assert rep["honest_note"] == honest_note()


# =====================================================================================
# (33) truncated branches stay individually visible in the temporal aggregate
# =====================================================================================
def test_inv_33_truncated_branch_list_visible_in_temporal_aggregate():
    s1 = TemporalRunStats()
    s1.temporally_truncated = True
    s1.branch_halted = True
    s1.branch_status = "truncated_actor_budget"
    s1.truncation = {"reason": "actor_llm_budget_exhausted",
                     "branch_status": "truncated_actor_budget", "at_ts": T0 + 60,
                     "actors_not_processed": ["maya"],
                     "unresolved_decision_trigger": {"trigger_type": "newly_noticed_information"}}
    s2 = TemporalRunStats()
    agg = aggregate_temporal_stats([SimpleNamespace(temporal_stats=s1, branch_id="b001",
                                                    world=None),
                                    SimpleNamespace(temporal_stats=s2, branch_id="b002",
                                                    world=None)])
    assert agg["n_branches"] == 2 and agg["n_branches_truncated"] == 1
    rows = agg["truncation"]["branches"]
    assert len(rows) == 1
    assert rows[0]["branch_id"] == "b001"
    assert rows[0]["branch_status"] == "truncated_actor_budget"
    assert rows[0]["reason"] == "actor_llm_budget_exhausted"
    assert rows[0]["actors"] == ["maya"]
    assert rows[0]["unresolved_decision_trigger"]["trigger_type"] == "newly_noticed_information"


# =====================================================================================
# (34) recommendation flips ineligible when the margin cannot cover the truncated span
# =====================================================================================
def test_inv_34_recommendation_ineligible_when_margin_below_truncated_span():
    scores = {"act_a": 1.0, "act_b": 0.5}
    ok = recommendation_eligibility(scores, 0.25, (0.0, 1.0))
    assert ok["eligible"] is True and ok["leader"] == "act_a"   # 0.75·0.5 ≥ 0.25
    bad = recommendation_eligibility(scores, 0.5, (0.0, 1.0))
    assert bad["eligible"] is False                             # 0.5·0.5 < 0.5 → withheld
    assert honest_note() in bad["why"]
    assert bad["margin_worst_case"] < 0.0


# =====================================================================================
# (35) STRICT: the generic prior writes NOTHING; the suppression is first-class
# =====================================================================================
def test_inv_35_generic_prior_suppressed_in_strict_default(monkeypatch):
    _strict(monkeypatch)
    assert generic_prior_allowed() is False
    w = make_min_world(info=False)
    op = GenericOutcomeOperator()
    ev = SimpleNamespace(etype="resolve_outcome",
                         payload={"outcome_var": "readout", "family": "binary",
                                  "lean": "weak_yes", "options": ["yes", "no"]})
    delta = op.apply(w, op.propose(w, ev, random.Random(0)))
    assert "generic_prior_suppressed_default" in delta.reason_codes
    assert "under_modeled_nonhuman_mechanism" in delta.reason_codes
    assert delta.changes == []                                  # suppression delta, no write
    assert "readout" not in w.quantities                        # the readout stays UNRESOLVED
    sup = get_stats(w).mechanism_suppressions
    assert sup and sup[0]["mechanism"] == "generic_outcome_prior"
    assert sup[0]["classification"] == "under_modeled_nonhuman_mechanism"
    assert w.omissions[0]["kind"] == "generic_prior_suppressed"
    # categorical and continuous families are quarantined identically
    for payload in ({"outcome_var": "cat_out", "family": "categorical",
                     "options": ["a", "b", "c"], "lean": "neutral"},
                    {"outcome_var": "cont_out", "family": "continuous", "lean": "neutral",
                     "options": [], "lo": 0.0, "hi": 1.0}):
        d = op.apply(w, op.propose(w, SimpleNamespace(etype="resolve_outcome",
                                                      payload=payload), random.Random(0)))
        assert "generic_prior_suppressed_default" in d.reason_codes
        assert payload["outcome_var"] not in w.quantities
    # a POSTERIOR-parameterized draw remains legal and resolves (§28)
    w2 = make_min_world(info=False)
    ev2 = SimpleNamespace(etype="resolve_outcome",
                          payload={"outcome_var": "readout", "family": "binary",
                                   "lean": "neutral", "options": ["yes", "no"],
                                   "posterior_rate_particles": [(0.9, 1.0)]})
    d2 = op.apply(w2, op.propose(w2, ev2, random.Random(0)))
    assert w2.quantities["readout"].value in ("yes", "no")
    assert d2.uncertainty["rate_source"] == "posterior"
    assert get_stats(w2).mechanism_suppressions == []


# =====================================================================================
# (36) STRICT: the institutional and aggregate prior_beta rungs are quarantined too
# =====================================================================================
def test_inv_36_institutional_and_aggregate_prior_beta_rungs_suppressed(monkeypatch):
    _strict(monkeypatch)
    # institutional members from a bare lean prior → refused, recorded
    w = make_min_world(info=False)
    iop = CollectiveThresholdDecisionOperator()
    iev = SimpleNamespace(etype="institutional_decision",
                          payload={"institution_id": "board", "n_members": 9,
                                   "outcome_var": "vote_out", "options": ["yes", "no"]})
    assert iop.apply(w, iop.propose(w, iev, random.Random(0))) is None
    sup = get_stats(w).mechanism_suppressions
    assert sup[0]["mechanism"] == "institutional_prior_beta_members"
    assert sup[0]["classification"] == "under_modeled_nonhuman_mechanism"
    assert "vote_out" not in w.quantities
    # with an evidence posterior the SAME institution resolves
    iev2 = SimpleNamespace(etype="institutional_decision",
                           payload={"institution_id": "board", "n_members": 9,
                                    "outcome_var": "vote_out", "options": ["yes", "no"],
                                    "posterior_rate_particles": [(0.95, 1.0)]})
    assert iop.apply(w, iop.propose(w, iev2, random.Random(0))) is not None
    assert w.quantities["vote_out"].value in ("yes", "no")
    # aggregate realization from a bare lean prior → refused, recorded
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    w2 = make_min_world(info=False)
    register_quantity_type("population_aggregate:seg", units="share")
    w2.quantities["population_aggregate:seg"] = Quantity(
        name="population_aggregate:seg", qtype="population_aggregate:seg", value=0.8,
        timestamp=T0)
    aop = AggregateOutcomeOperator()
    aev = SimpleNamespace(etype="aggregate_outcome_resolution",
                          payload={"outcome_var": "agg_out", "options": ["yes", "no"],
                                   "lean": "neutral",
                                   "consume": [{"var": "population_aggregate:seg",
                                                "weight": 0.4}]})
    assert aop.apply(w2, aop.propose(w2, aev, random.Random(0))) is None
    sup2 = get_stats(w2).mechanism_suppressions
    assert sup2[0]["mechanism"] == "aggregate_outcome_prior_beta"
    assert "agg_out" not in w2.quantities
    aev2 = SimpleNamespace(etype="aggregate_outcome_resolution",
                           payload={"outcome_var": "agg_out", "options": ["yes", "no"],
                                    "lean": "neutral",
                                    "consume": [{"var": "population_aggregate:seg",
                                                 "weight": 0.4}],
                                    "posterior_rate_particles": [(0.9, 1.0)]})
    assert aop.apply(w2, aop.propose(w2, aev2, random.Random(0))) is not None
    assert w2.quantities["agg_out"].value in ("yes", "no")


# =====================================================================================
# (37) LLMs cannot invent executable equations: no eval/exec of text anywhere
# =====================================================================================
def test_inv_37_no_builtin_eval_or_exec_in_mechanism_execution_sources():
    bare_calls, attr_eval, attr_exec = [], [], []
    for path in sorted(WM_V2_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(WM_V2_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Name) and f.id in ("eval", "exec", "compile"):
                bare_calls.append(f"{rel}:{node.lineno}:{f.id}")
            elif isinstance(f, ast.Attribute) and f.attr == "exec":
                attr_exec.append(f"{rel}:{node.lineno}")
            elif isinstance(f, ast.Attribute) and f.attr == "eval":
                attr_eval.append(rel)
    # builtin eval/exec/compile over ANY text (LLM or otherwise): zero occurrences
    assert bare_calls == []
    assert attr_exec == []
    # `.eval` METHOD calls exist only in the nonlinear form library — inspected: these are
    # `FunctionalForm.eval(params, inputs)`, closed-form registered math over typed parameter
    # packs (swm/world_model_v2/nonlinear/forms.py), never execution of model-generated text.
    assert set(attr_eval) <= {"nonlinear/forms.py", "nonlinear/operators.py",
                              "nonlinear/posterior.py", "nonlinear/fit.py",
                              "nonlinear/compare.py"}, sorted(set(attr_eval))


# =====================================================================================
# (38) every registered operator gets a MechanismSpec with declared read/write sets
# =====================================================================================
def test_inv_38_every_registered_operator_has_spec_with_read_write_sets():
    from swm.world_model_v2.mechanism_spec import MechanismSpec, build_spec_index
    idx = build_spec_index()
    assert idx and all(isinstance(s, MechanismSpec) for s in idx.values())
    covered = {s.operator for s in idx.values() if s.operator}
    missing = [op for op in transitions._OPERATORS if op not in covered]
    assert missing == []                                        # no operator escapes the contract
    for spec in idx.values():
        assert isinstance(spec.read_set, tuple) and isinstance(spec.write_set, tuple)
        assert spec.calibration_status in ("fitted_validated", "domain_validated",
                                           "transfer_validated", "documented_prior",
                                           "grounded_scenario", "experimental_visible",
                                           "unresolved")
    g = next(s for s in idx.values() if s.operator == "generic_outcome_prior")
    assert "quantities" in g.read_set and "quantities" in g.write_set
    p = next(s for s in idx.values() if s.operator == "production_actor_policy")
    assert p.mechanism_kind == "qualitative_actor"
    assert p.write_set                                          # human mechanism declares writes


# =====================================================================================
# (39) human and nonhuman operators run on ONE WorldState and ONE clock
# =====================================================================================
def test_inv_39_human_and_nonhuman_operators_share_one_world_and_clock():
    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=QLLM(), llm_hypotheses=False,
                                                    bounded_cognition=False)),
        mode="persistent_qualitative_llm_policy")
    human_op = ProductionActorPolicyOperator(runtime=rt)
    nonhuman_op = GenericOutcomeOperator()
    w = world()
    _seed_state(w)
    clock_obj = w.clock
    q = EventQueue(horizon_ts=T0 + 10 * DAY)
    q.schedule(Event(ts=T0 + 60, etype="decision_opportunity", participants=["alice"],
                     payload=dict(DECISION)))
    q.schedule(Event(ts=T0 + DAY, etype="resolve_outcome", participants=[],
                     payload={"outcome_var": "readout", "family": "binary", "lean": "neutral",
                              "options": ["yes", "no"],
                              "posterior_rate_particles": [(0.9, 1.0)]}))
    branch = RolloutEngine(operators=[human_op, nonhuman_op]).run_branch(w, q, seed=0)
    # one shared clock advanced through both mechanisms' event times
    assert branch.world is w and w.clock is clock_obj
    assert w.clock.now == T0 + DAY
    stats = branch.temporal_stats
    assert stats.event_counts == {"decision_opportunity": 1, "resolve_outcome": 1}
    # the HUMAN mechanism wrote into the same world the NONHUMAN one resolved
    assert w.entity("alice").value("past_actions")[-1]["action"] == "approve"
    assert w.quantities["readout"].value in ("yes", "no")
    ats = [d.at for d in branch.log if d.operator in ("production_actor_policy",
                                                      "generic_outcome_prior")]
    assert ats == sorted(ats) and ats[0] == T0 + 60 and ats[-1] == T0 + DAY


# =====================================================================================
# (48-adjacent) production sources only READ the offline markers, never set them
# =====================================================================================
def test_inv_48_markers_never_assigned_in_production_sources():
    setter = re.compile(
        r"environ(?:\.setdefault\(\s*|\[\s*)['\"](SWM_ALLOW_NUMERIC_BASELINE|"
        r"SWM_ALLOW_GENERIC_PRIOR)")
    indirect = re.compile(r"environ(?:\.setdefault\(\s*|\[\s*)_NUMERIC_TEST_MARKER")
    offenders = []
    for path in sorted(WM_V2_ROOT.rglob("*.py")):
        text = path.read_text()
        rel = path.relative_to(WM_V2_ROOT).as_posix()
        for m in setter.finditer(text):
            frag = text[m.start():m.end() + 40]
            # environ[...] read access is fine; assignment (`] =`) and setdefault are not
            if ".setdefault(" in frag or re.search(r"\]\s*=[^=]", frag):
                offenders.append(f"{rel}: {frag[:60]}")
        if indirect.search(text):
            offenders.append(f"{rel}: indirect marker setdefault/assignment")
    assert offenders == []
    # the scan itself detects real setters: tests/conftest.py DOES setdefault the markers
    conftest = pathlib.Path(__file__).parent / "conftest.py"
    assert setter.search(conftest.read_text())
    # and production only ever READS them
    marker_files = [p for p in WM_V2_ROOT.rglob("*.py")
                    if "SWM_ALLOW_NUMERIC_BASELINE" in p.read_text()
                    or "SWM_ALLOW_GENERIC_PRIOR" in p.read_text()]
    assert marker_files                                         # the reads exist (strict gates)
    for p in marker_files:
        for line in p.read_text().splitlines():
            if ("SWM_ALLOW_NUMERIC_BASELINE" in line or "SWM_ALLOW_GENERIC_PRIOR" in line) \
                    and "environ" in line:
                assert ".get(" in line, f"{p.name}: {line.strip()}"
