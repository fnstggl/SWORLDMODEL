"""Persistent qualitative LLM actor acceptance tests — offline, deterministic, scripted.

Covers the merged required-test lists: no numerical cognition in qualitative mode; the LLM (not
the numeric policy) chooses Tier-1 actions; several distinct persistent hypothesis particles;
independent per-branch decision calls that can diverge and each execute their own action; raw
distributions equal observed branch-selection frequencies (no softmax of self-reports);
qualitative persistence without scalar reduction and without personality rewrites; information
boundaries (no future events, no other minds, different actors different views); single-
individual auto-Tier-1 routing end to end; novel actions compiled-with-mechanisms or explicitly
unmodeled, and blocked when impossible; anticipated reactions stay subjective; causal tier
selection with dynamic promotion and cheap routine actors; external calibration strictly after
aggregation with `unvalidated` labeling; numeric and persona-blended baselines still runnable."""
import copy
import json
from types import SimpleNamespace

import pytest

from swm.world_model_v2.actor_selection import RelevantActorSelector, reaction_target
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.individual_reaction import simulate_individual_reaction
from swm.world_model_v2.llm_actor import PersonaActorPolicyRuntime
from swm.world_model_v2.materialize import (
    attach_actor_decision_distributions, operators_from_plan, resolve_actor_policy_mode,
)
from swm.world_model_v2.phase4_execution import ActorPolicyRuntime, ProductionActorPolicyOperator
from swm.world_model_v2.phase4_policy import ActorPolicyModel, ActorViewBuilder, UtilityInference
from swm.world_model_v2.qualitative_actor import (
    QUAL_STATE_KEY, ActionClusterer, ActorPolicyCalibrator, QualitativeActorPolicyRuntime,
    QualitativeConfig, QualitativeDecisionEngine, aggregate_actor_decisions, load_actor_state,
    parse_qualitative_decision,
)
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import F
from tests.test_llm_actor import DECISION, Plan, T0, world


def qpayload(chosen="approve", target="board", **over):
    base = {
        "schema_version": "qualitative.actor.v1",
        "situation_interpretation": {"what_changed": "the vote is upon me",
                                     "why_it_matters": "my project hangs on it",
                                     "perceived_opportunities": "lock in support now",
                                     "perceived_threats": "bob's objection gains traction"},
        "actor_state_update": {"current_private_beliefs": ["The board leans my way"],
                               "beliefs_about_others": {"bob": "will grumble but comply"},
                               "personal_condition": "focused, slightly tense",
                               "important_memories": ["The audit fight is behind us"]},
        "anticipated_reactions": [{"actor_or_group": "bob",
                                   "expected_reaction": "objects publicly",
                                   "reasoning_summary": "he opposed the proposal before",
                                   "uncertainty_description": "moderate"}],
        "decision": {"act_or_wait": "act", "chosen_action": chosen, "target": target,
                     "timing": "immediate", "observability": "public",
                     "intended_effect": "secure the approval"},
        "novel_action_proposal": {"present": False},
        "alternatives_considered": [{"action": "delay", "why_not_selected": "momentum loss"}],
        "decision_summary": "I act now while support holds",
    }
    base.update(over)
    return base


class QLLM:
    """Scripted qualitative backend. `decide` is a dict, a JSON string, or fn(prompt)->payload."""

    def __init__(self, decide=None):
        self.decide = decide
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        out = self.decide(prompt) if callable(self.decide) else (self.decide or qpayload())
        return out if isinstance(out, str) else json.dumps(out)


def qruntime(llm, mode="persistent_qualitative_llm_policy", **cfg):
    defaults = dict(llm=llm, llm_hypotheses=False, n_hypotheses=3)
    defaults.update(cfg)
    return QualitativeActorPolicyRuntime(QualitativeDecisionEngine(QualitativeConfig(**defaults)),
                                         mode=mode)


def particle_worlds(n):
    out = []
    for i in range(n):
        w = world()
        w.branch_id = f"b{i:03d}"
        out.append(w)
    return out


# ---------------------------------------------------------------- 1. no numerical cognition
def test_qualitative_schema_and_prompt_contain_no_numeric_cognition():
    rt = qruntime(QLLM())
    w = world()
    _, posterior, _ = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=1)
    prompt = rt.engine.config.llm.prompts[-1]
    # the schema must request NO numeric self-reports: no inclination ratings, no confidence
    # scalars, no numeric belief deltas, no utilities (qualitative PROSE may use such words)
    for banned in ("inclination", "<0..1>", '"confidence":', "utility", "belief_delta",
                   "risk score", "+= 0."):
        assert banned not in prompt
    assert "NO numbers" in prompt
    qual = posterior.provenance["qualitative"]
    assert "inclination" not in json.dumps(qual)
    # numeric values smuggled into cognition fields are DROPPED and counted, never consumed
    smuggled = qpayload()
    smuggled["actor_state_update"]["confidence"] = 0.8
    smuggled["actor_state_update"]["current_private_beliefs"] = ["fine", 0.63]
    qd = parse_qualitative_decision(json.dumps(smuggled), "alice")
    assert qd.numeric_fields_dropped >= 2
    assert "confidence" not in qd.actor_state_update
    assert qd.actor_state_update["current_private_beliefs"] == ["fine"]


def test_llm_chooses_without_the_numeric_policy(monkeypatch):
    counters = {"decide": 0, "utility": 0}
    orig = ActorPolicyModel.decide
    monkeypatch.setattr(ActorPolicyModel, "decide",
                        lambda self, *a, **k: counters.__setitem__("decide", counters["decide"] + 1)
                        or orig(self, *a, **k))
    orig_u = UtilityInference.infer
    monkeypatch.setattr(UtilityInference, "infer",
                        lambda self, *a, **k: counters.__setitem__("utility", counters["utility"] + 1)
                        or orig_u(self, *a, **k))
    rt = qruntime(QLLM())
    selected, posterior, _ = rt.decide(Plan(), [world()], "alice", decision=dict(DECISION), seed=2)
    assert counters == {"decide": 0, "utility": 0}          # the mind decided, not the equations
    assert posterior.provenance["qualitative"]["decision_source"] == "persistent_qualitative_llm"
    assert posterior.provenance["llm_probability_minting"] is False
    assert posterior.provenance["llm_action_choice"] is True
    assert posterior.action_probabilities == {selected.action_id: 1.0}


# ---------------------------------------------------------------- particles
def test_several_distinct_hypothesis_particles_are_created_and_isolated():
    rt = qruntime(QLLM())
    worlds = particle_worlds(3)
    for i, w in enumerate(worlds):
        sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=i)
        rt.execute(w, sel, post, tr, seed=i)
    states = [load_actor_state(w, "alice") for w in worlds]
    ids = {s.hypothesis_id for s in states}
    assert len(ids) == 3                                    # three genuinely different realities
    worldviews = {s.core_worldview for s in states}
    assert len(worldviews) == 3
    # distinctness at initialization (before any decision touches personal_condition)
    fresh = [rt.engine.hypothesizer.state_for_branch(w, ActorViewBuilder().build(w, "alice"))
             for w in particle_worlds(3)]
    assert len({s.personal_condition for s in fresh}) == 3
    assert len({s.organizational_pressures for s in fresh}) == 3
    # isolation: mutating one particle's state leaves the others untouched
    states[0].current_private_beliefs.append("only particle zero believes this")
    assert "only particle zero believes this" not in json.dumps(states[1].as_dict())
    assert "only particle zero believes this" not in json.dumps(
        load_actor_state(worlds[1], "alice").as_dict())


def test_particles_decide_independently_and_can_choose_differently():
    def decide(prompt):
        # the particle inhabiting private doubt behaves differently from the confident one
        if "privately doubts" in prompt:
            return qpayload(chosen="delay", target="")
        return qpayload(chosen="approve")
    llm = QLLM(decide)
    rt = qruntime(llm)
    worlds = particle_worlds(4)
    chosen = []
    for i, w in enumerate(worlds):
        sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=i)
        rt.execute(w, sel, post, tr, seed=i)
        chosen.append(sel.action_name)
    assert len(llm.prompts) == 4                            # one independent call per particle
    assert set(chosen) == {"approve", "delay"}              # hidden state changed the decision
    # each branch executed ITS OWN action — no globally shared selection
    for w, name in zip(worlds, chosen):
        history = w.entity("alice").value("past_actions")
        assert history[-1]["action"] == name


def test_raw_distribution_equals_observed_branch_frequencies():
    def decide(prompt):
        return qpayload(chosen="delay", target="") if "privately doubts" in prompt \
            else qpayload(chosen="approve")
    rt = qruntime(QLLM(decide))
    for i, w in enumerate(particle_worlds(6)):
        sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=i)
        rt.execute(w, sel, post, tr, seed=i)
    agg = aggregate_actor_decisions(rt.decision_records)["alice"]
    raw = agg["raw_qualitative_simulation_distribution"]
    # 6 branches over 3 hypotheses: exactly 2 chose delay (private_doubt), 4 chose approve
    assert raw == {"approve@board": pytest.approx(4 / 6, abs=1e-4),
                   "delay": pytest.approx(2 / 6, abs=1e-4)}
    assert agg["n_qualitative_branches"] == 6
    # counting, not scoring: every mass is a whole number of branches (4-decimal storage)
    assert all(abs(v * 6 - round(v * 6)) < 2e-3 for v in raw.values())
    assert agg["cluster_version"] == ActionClusterer.version
    for row in agg["rows"]:
        assert row["decision_source"] == "persistent_qualitative_llm"
        assert row["hypothesis_id"] and row["branch_id"]


# ---------------------------------------------------------------- persistence
def test_state_persists_updates_only_named_sections_and_keeps_identity():
    rt = qruntime(QLLM())
    w = world()
    sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=1)
    rt.execute(w, sel, post, tr, seed=1)
    s1 = load_actor_state(w, "alice")
    identity, worldview = s1.identity_and_role, s1.core_worldview
    assert s1.current_private_beliefs == ["The board leans my way"]
    assert s1.beliefs_about_others["bob"] == "will grumble but comply"
    assert any("I chose to approve" in m["memory"] for m in s1.important_memories)
    assert s1.revision_log[-1]["source"] == "qualitative_llm_decision"
    assert s1.revision_log[-1]["sections_changed"]
    # a second event revises beliefs without rewriting the personality
    rt2 = qruntime(QLLM(qpayload(actor_state_update={
        "current_private_beliefs": ["bob is angrier than I thought"]})))
    sel2, post2, tr2 = rt2.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=2)
    rt2.execute(w, sel2, post2, tr2, seed=2)
    s2 = load_actor_state(w, "alice")
    assert s2.current_private_beliefs == ["bob is angrier than I thought"]
    assert s2.identity_and_role == identity and s2.core_worldview == worldview
    assert len(s2.revision_log) > len(s1.revision_log) - 1
    # no scalar reduction anywhere in the persisted state
    def no_bare_numbers(x):
        if isinstance(x, dict):
            return all(no_bare_numbers(v) for k, v in x.items() if k != "at")
        if isinstance(x, list):
            return all(no_bare_numbers(v) for v in x)
        return not isinstance(x, (int, float)) or isinstance(x, bool)
    assert no_bare_numbers({k: v for k, v in s2.as_dict().items() if k != "revision_log"})
    # the persisted state conditions the NEXT prompt directly
    rt3 = qruntime(QLLM())
    rt3.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=3)
    assert "bob is angrier than I thought" in rt3.engine.config.llm.prompts[-1]


def test_stateless_mode_runs_without_persistent_state():
    rt = qruntime(QLLM(), mode="stateless_llm_policy")
    w = world()
    sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=1)
    delta, _ = rt.execute(w, sel, post, tr, seed=1)
    assert load_actor_state(w, "alice") is None
    assert "qualitative_stateless_no_persistence" in delta.reason_codes
    assert post.provenance["qualitative"]["decision_source"] == "stateless_llm"


# ---------------------------------------------------------------- information boundary
def test_actor_local_boundary_no_future_no_other_minds_different_views():
    rt = qruntime(QLLM())
    w = world()
    w.entity("bob").set("beliefs", F(0.1, status="inferred"), key="bobs_hidden_opinion_marker")
    rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=1)
    prompt = rt.engine.config.llm.prompts[-1]
    assert "proposal resolved" not in prompt                # future information item
    assert "bobs_hidden_opinion_marker" not in prompt       # bob's mind
    assert "reservation=3" not in prompt                    # simulator-only actor field
    assert "hidden_truth" not in prompt and "posterior_truth" not in prompt
    assert "proposal filed" in prompt                       # alice's own observation
    # different actors, different information, different views
    va = ActorViewBuilder().build(w, "alice")
    vb = ActorViewBuilder().build(w, "bob")
    assert va.observed_evidence_ids != vb.observed_evidence_ids
    eng = rt.engine
    pb = eng.build_prompt(vb, None, "the vote",
                          [{"key": "wait", "action_id": "x", "line": "- wait"}])
    assert "proposal filed" not in pb                       # bob never saw it


# ---------------------------------------------------------------- single-individual mode
def test_reaction_question_target_is_automatically_tier_1():
    assert reaction_target("How will Dana react if I skip dinner?", ["Dana", "alice"]) == "Dana"
    plan = SimpleNamespace(entities=[{"id": "Dana"}], institutions=[], scheduled_events=[],
                           actor_decisions=[], relations=[], quantities=[],
                           _intention_stances=[], question="How will Dana react if I skip dinner?")
    tiers = RelevantActorSelector().select(plan, plan.question)
    assert tiers["Dana"]["tier"] == 1
    assert "reaction_is_the_question" in tiers["Dana"]["reasons"]


def test_individual_reaction_end_to_end():
    def decide(prompt):
        chosen = "reply_now" if "privately doubts" not in prompt else "reply_later"
        return qpayload(chosen=chosen, target="you", actor_state_update={
            "personal_condition": "disappointed but understanding",
            "important_memories": ["They cancelled dinner tonight"]})
    result = simulate_individual_reaction(
        person_id="Dana", stimulus="So sorry — I can't make dinner tonight, work blew up.",
        context={"relationship": "close friend", "role": "friend",
                 "history": ["We rescheduled twice last month", "Dana cooked last time"]},
        llm=QLLM(decide), n_hypotheses=3, samples_per_hypothesis=2, seed=0, as_of=T0,
        config=QualitativeConfig(llm=QLLM(decide), llm_hypotheses=False, n_hypotheses=3))
    assert result["calibration_status"] == "unvalidated"
    raw = result["raw_qualitative_simulation_distribution"]
    assert set(raw) == {"reply_now@you", "reply_later@you"}
    assert raw["reply_later@you"] == pytest.approx(2 / 6, abs=1e-4)
    assert len(result["samples"]) == 6
    sample = result["samples"][0]
    assert sample["interpretation"] and sample["internal_reaction"]
    assert sample["observable_response"] in ("reply_now", "reply_later")
    assert sample["decision_source"] == "persistent_qualitative_llm"
    assert result["n_excluded_numeric_fallbacks"] == 0


# ---------------------------------------------------------------- novel actions
def test_novel_action_with_causal_reading_is_compiled_and_moves_the_world():
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    payload = qpayload(chosen="escalate_the_campaign", target="bob",
                       novel_action_proposal={"present": True,
                                              "description": "escalate pressure through bob",
                                              "required_authority": "none",
                                              "required_resources": "attention",
                                              "proposed_mechanisms": "public escalation"})
    rt = qruntime(QLLM(payload))
    w = world()
    register_quantity_type("pathway_progress", units="process_state")
    w.quantities["pathway_progress:cooperative_agreement"] = Quantity(
        name="pathway_progress:cooperative_agreement", qtype="pathway_progress",
        value=0.5, timestamp=T0)
    sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=1)
    assert sel.action_name == "escalate_the_campaign"
    anchor = sel.parameters["ontology_anchor"]
    assert anchor["name"] == "escalate"                     # validated causal reading
    assert post.provenance["qualitative"]["novel_action_unmodeled"] is False
    assert sel.mechanisms_triggered                         # executable, not decorative
    delta, _ = rt.execute(w, sel, post, tr, seed=1)
    moved = [c for c in delta.changes if "pathway_progress" in c.get("path", "")]
    assert moved                                            # the novel action moved the process


def test_novel_action_without_causal_reading_is_marked_unmodeled():
    payload = qpayload(chosen="quiet_backchannel_probe", target="bob",
                       novel_action_proposal={"present": True,
                                              "description": "privately probe willingness",
                                              "proposed_mechanisms": "an intermediary"})
    rt = qruntime(QLLM(payload))
    w = world()
    sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=1)
    assert post.provenance["qualitative"]["novel_action_unmodeled"] is True
    assert sel.support_status == "llm_chosen_unmodeled"
    assert any("novel_action_unmodeled" in wn for wn in tr.warnings)
    assert sel.mechanisms_triggered == ["message_delivery", "reaction_scheduling"]
    assert sel.observability["default"] == "public"          # actor said public? no — private
    # unreachable targets are stripped: the actor cannot aim at what it cannot see
    payload2 = qpayload(chosen="call_the_minister", target="carol")
    rt2 = qruntime(QLLM(payload2))
    sel2, _, _ = rt2.decide(Plan(), [world()], "alice", decision=dict(DECISION), seed=2)
    assert sel2.target.target_id == ""


def test_impossible_action_is_blocked_at_execution():
    # bob chooses 'approve' — the board's decision_right holder is alice, so reality refuses
    rt = qruntime(QLLM(qpayload(chosen="approve", target="board")))
    w = world()
    w.entity("bob").set("past_actions", F([], status="observed"))
    sel, post, tr = rt.decide(Plan(), [w], "bob",
                              decision={"candidate_actions": ["approve", "wait"],
                                        "situation": "the vote"}, seed=3)
    delta, events = rt.execute(w, sel, post, tr, seed=3)
    assert delta.event_type == "action_blocked"
    assert any(e.etype == "action_blocked" for e in events)
    assert w.entity("bob").value("past_actions")[-1]["status"] == "blocked"


# ---------------------------------------------------------------- subjectivity of anticipation
def test_anticipated_reactions_are_subjective_never_world_truth():
    rt = qruntime(QLLM())
    w = world()
    sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=1)
    rt.execute(w, sel, post, tr, seed=1)
    expectation = w.entity("alice").value("expected_reactions")["bob"]
    assert expectation["subjective"] is True
    assert expectation["expects"] == "objects publicly"
    # bob himself did nothing: no action, no state, no belief was written for him
    assert not w.entity("bob").value("past_actions", default=[])
    assert load_actor_state(w, "bob") is None
    assert w.entity("bob").value("beliefs", key="private_belief") == 0.1


# ---------------------------------------------------------------- routing, tiers, promotion
def test_selector_assigns_causal_tiers_with_reasons():
    plan = SimpleNamespace(
        question="Will the board approve the merger?",
        entities=[{"id": a} for a in ("alice", "carol", "putin", "aide", "rando")],
        institutions=[{"id": "board", "rules": [
            {"kind": "decision_right", "params": {"actions": ["approve"], "holders": ["alice"]}},
            {"kind": "veto", "params": {"holders": ["carol"]}}]}],
        scheduled_events=[{"etype": "decision_opportunity", "participants": ["carol"]}],
        actor_decisions=[], relations=[{"src": "aide", "rel": "influences", "dst": "alice"}],
        quantities=[{"name": "pathway_principals:cooperative_agreement", "value": "putin"}],
        _intention_stances=[{"actor": "putin", "pathway": "cooperative_agreement",
                             "commitment_level": "committed_to_prevent", "capability": "high"}])
    tiers = RelevantActorSelector().select(plan, plan.question)
    assert tiers["alice"]["tier"] == 1
    assert any(r.startswith("direct_decision_authority") for r in tiers["alice"]["reasons"])
    assert tiers["carol"]["tier"] == 1                      # scheduled decision event
    assert tiers["putin"]["tier"] == 1                      # pathway principal + capability
    assert tiers["aide"]["tier"] == 2
    assert any(r.startswith("persuasive_access_to") for r in tiers["aide"]["reasons"])
    assert tiers["rando"]["tier"] == 3
    assert tiers["rando"]["reasons"]                        # why-not is recorded too


def test_hybrid_routes_routine_actors_to_numeric_and_promotes_dynamically():
    llm = QLLM()
    engine = QualitativeDecisionEngine(QualitativeConfig(llm=llm, llm_hypotheses=False))
    rt = QualitativeActorPolicyRuntime(engine, mode="hybrid_relevant_actor_policy",
                                       tiers={"alice": {"tier": 1, "reasons": ["authority"]}},
                                       selector=RelevantActorSelector())
    w = world()
    w.entity("bob").set("past_actions", F([], status="observed"))
    # bob has no causal signals → numeric policy, recorded
    _, post_bob, _ = rt.decide(Plan(), [w], "bob",
                               decision={"candidate_actions": ["wait", "acknowledge"],
                                         "situation": "routine"}, seed=1)
    assert post_bob.provenance["qualitative"]["routed"] is False
    assert post_bob.provenance["qualitative"]["decision_source"] == "numeric_policy"
    assert post_bob.provenance["qualitative"]["reason"].startswith("tier")
    assert not llm.prompts                                  # the cheap path spent no cognition
    # bob becomes causally consequential mid-run → dynamic promotion to the expensive policy
    w.entity("bob").set("stances", F([{"actor": "bob", "commitment_level": "actively_pursuing",
                                       "pathway": "institutional_procedure"}], status="derived"))
    w.entity("bob").set("resources", F(0.6, status="derived"), key="capacity")
    _, post_bob2, _ = rt.decide(Plan(), [w], "bob",
                                decision={"candidate_actions": ["wait", "acknowledge"],
                                          "situation": "suddenly pivotal"}, seed=2)
    assert post_bob2.provenance["qualitative"]["routed"] is True
    assert llm.prompts
    promotions = w.uncertainty_meta["actor_tier_promotions"]
    assert promotions and promotions[-1]["actor"] == "bob"
    assert "dynamically_promoted_at_event_time" in promotions[-1]["reasons"]
    # alice (tier 1) uses the expensive policy
    _, post_alice, _ = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=3)
    assert post_alice.provenance["qualitative"]["routed"] is True


def test_multi_particle_bridge_and_fallback_stay_numeric_and_marked():
    rt = qruntime(QLLM())
    worlds = [world(), world()]
    _, post, _ = rt.decide(Plan(), worlds, "alice", decision=dict(DECISION), seed=1)
    assert post.provenance["qualitative"]["routed"] is False
    assert post.provenance["qualitative"]["reason"] == "multi_particle_bridge_is_numeric"
    # a garbage backend → numeric fallback, marked and excluded from pure aggregation
    bad = qruntime(QLLM(lambda p: "not json"), retries=0)
    sel, post2, tr2 = bad.decide(Plan(), [world()], "alice", decision=dict(DECISION), seed=2)
    q = post2.provenance["qualitative"]
    assert q["decision_source"] == "numeric_fallback" and q["excluded_from_qualitative_aggregation"]
    agg = aggregate_actor_decisions(bad.decision_records)["alice"]
    assert agg["raw_qualitative_simulation_distribution"] == {}
    assert agg["n_excluded_numeric_fallbacks"] == 1


# ---------------------------------------------------------------- calibration
def test_calibration_is_external_after_aggregation_and_labels_unvalidated():
    rt = qruntime(QLLM())
    for i, w in enumerate(particle_worlds(3)):
        sel, post, tr = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=i)
        rt.execute(w, sel, post, tr, seed=i)
    no_fit = aggregate_actor_decisions(rt.decision_records,
                                       calibrator=ActorPolicyCalibrator({}))["alice"]
    assert no_fit["calibration_status"] == "unvalidated"
    assert no_fit["calibrated_distribution"] == no_fit["raw_qualitative_simulation_distribution"]
    fitted = aggregate_actor_decisions(
        rt.decision_records,
        calibrator=ActorPolicyCalibrator({"actor": {"alice": {"temperature": 2.0,
                                                              "fit": "test-history"}}}))["alice"]
    assert fitted["calibration_status"] == "calibrated" and fitted["calibration_level"] == "actor"
    # the raw counted distribution is preserved unchanged next to the calibrated one
    assert fitted["raw_qualitative_simulation_distribution"] == \
        no_fit["raw_qualitative_simulation_distribution"]


# ---------------------------------------------------------------- modes + baselines + wiring
def test_mode_resolution_and_default_on_wiring(monkeypatch):
    monkeypatch.delenv("SWM_ACTOR_POLICY", raising=False)
    monkeypatch.delenv("SWM_LLM_ACTORS", raising=False)
    assert resolve_actor_policy_mode(None) == "numeric_policy"
    assert resolve_actor_policy_mode(QLLM()) == "hybrid_relevant_actor_policy"
    monkeypatch.setenv("SWM_ACTOR_POLICY", "persona_blended_numeric_policy")
    assert resolve_actor_policy_mode(QLLM()) == "persona_blended_numeric_policy"
    monkeypatch.setenv("SWM_LLM_ACTORS", "off")
    assert resolve_actor_policy_mode(QLLM()) == "numeric_policy"
    monkeypatch.delenv("SWM_LLM_ACTORS", raising=False)
    plan = SimpleNamespace(
        accepted_mechanisms=[{"mech_id": "production_actor_policy",
                              "operator": "production_actor_policy"}],
        entities=[{"id": "alice"}], institutions=[], scheduled_events=[], actor_decisions=[],
        relations=[], quantities=[], _intention_stances=[], question="q")
    monkeypatch.setenv("SWM_ACTOR_POLICY", "hybrid_relevant_actor_policy")
    ops, _ = operators_from_plan(plan, llm=QLLM())
    assert isinstance(ops[0].runtime, QualitativeActorPolicyRuntime)
    assert ops[0].runtime.mode == "hybrid_relevant_actor_policy"
    monkeypatch.setenv("SWM_ACTOR_POLICY", "persona_blended_numeric_policy")
    ops_b, _ = operators_from_plan(plan, llm=QLLM())
    assert isinstance(ops_b[0].runtime, PersonaActorPolicyRuntime)
    monkeypatch.setenv("SWM_ACTOR_POLICY", "numeric_policy")
    ops_a, _ = operators_from_plan(plan, llm=QLLM())
    assert type(ops_a[0].runtime) is ActorPolicyRuntime     # arm A untouched and runnable


def test_truncated_hypothesis_array_salvages_complete_objects():
    from swm.world_model_v2.qualitative_actor import QualitativeParticleHypothesizer
    full = json.dumps({"hypothesis_label": "steady", "core_worldview": "w1",
                       "current_private_beliefs": ["b1"], "personal_condition": "calm"})
    second = json.dumps({"hypothesis_label": "doubting", "core_worldview": "w2",
                         "current_private_beliefs": ["b2"], "personal_condition": "tired"})
    truncated = f'[{full}, {second}, {{"hypothesis_label": "clipped", "core_worldview": "w3'
    rows = QualitativeParticleHypothesizer._parse(truncated)
    assert [r["hypothesis_label"] for r in rows] == ["steady", "doubting"]
    assert QualitativeParticleHypothesizer._parse("no json at all") is None


def _load_benchmark_module():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "actor_policy_benchmark", Path("experiments/actor_policy_benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_prevents_post_outcome_leakage():
    bench = _load_benchmark_module()
    cases = bench.load_cases()                              # every shipped case passes the guard
    assert len(cases) >= 8
    tampered = copy.deepcopy(cases[0])
    tampered["evidence"].append({"date": "2100-01-01", "text": "the future leaked"})
    with pytest.raises(ValueError):
        bench.leakage_check(tampered)
    early = copy.deepcopy(cases[0])
    early["actual_action_date"] = "1900-01-01"
    with pytest.raises(ValueError):
        bench.leakage_check(early)
    # the label never reaches any actor-facing structure
    w, decision = bench.build_case_world(cases[0])
    view = ActorViewBuilder().build(w, cases[0]["actor_id"])
    rendered = json.dumps(view.as_dict()) + json.dumps(decision)
    assert cases[0]["source_note"][:30] not in rendered
    for ev in view.observed_events:
        assert float(ev["at"]) <= view.observed_time
    # candidate order is de-biased: not simply the file's label-first order
    assert any(bench.build_case_world(c)[1]["candidate_actions"][0] != c["candidate_actions"][0]
               for c in cases)


def test_qualitative_operator_in_the_event_loop_aggregates_per_branch():
    def decide(prompt):
        return qpayload(chosen="delay", target="") if "privately doubts" in prompt \
            else qpayload(chosen="approve")
    llm = QLLM(decide)
    engine = QualitativeDecisionEngine(QualitativeConfig(llm=llm, llm_hypotheses=False))
    rt = QualitativeActorPolicyRuntime(engine, mode="persistent_qualitative_llm_policy")
    op = ProductionActorPolicyOperator(runtime=rt)
    for i in range(3):
        w = world()
        w.branch_id = f"b{i:03d}"
        q = EventQueue(horizon_ts=T0 + 5 * 86400)
        q.schedule(Event(ts=T0 + 60, etype="decision_opportunity", participants=["alice"],
                         payload=dict(DECISION)))
        RolloutEngine(operators=[op]).run_branch(w, q, seed=i)
    assert len(llm.prompts) == 3
    container = {}
    attach_actor_decision_distributions([op], container)
    dist = container["actor_decision_distributions"]["alice"]
    assert dist["n_qualitative_branches"] == 3
    raw = dist["raw_qualitative_simulation_distribution"]
    assert raw["approve@board"] == pytest.approx(2 / 3, abs=1e-4)
    assert raw["delay"] == pytest.approx(1 / 3, abs=1e-4)
    assert container["actor_policy_mode"] == "persistent_qualitative_llm_policy"
