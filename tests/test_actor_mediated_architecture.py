"""Deterministic architecture tests for universal actor-mediated causal execution.

These tests validate ARCHITECTURE (information boundaries, routing, recursion, budgets,
demotions, joint-world coherence, counterfactual parity), never predictive accuracy. Actors are
scripted mocks; every run is deterministic and offline."""
import copy
import json

import pytest

from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.institutions import Rule, RuleSystem
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase4_execution import ProductionActorPolicyOperator
from swm.world_model_v2.qualitative_actor import (
    QualitativeActorPolicyRuntime, QualitativeConfig, QualitativeDecisionEngine,
    load_actor_state,
)
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import InstitutionalVoteOperator

T0 = 1_700_000_000.0


# ------------------------------------------------------------------ scripted mock actors
def scripted_llm(script):
    """script: {actor_id: callable(prompt) -> decision dict} — deterministic mock actors."""
    def llm(prompt):
        for aid, decide in script.items():
            if f"You ARE {aid}" in prompt:
                d = decide(prompt)
                break
        else:
            d = {"act_or_wait": "wait", "chosen_action": "wait", "target": "",
                 "observability": "private", "intended_effect": ""}
        return json.dumps({
            "schema_version": "qualitative.actor.v1",
            "decision": {"timing": "immediate", **d},
            "decision_summary": d.get("intended_effect", ""),
            "novel_action_proposal": d.get("novel_action_proposal", {"present": False}),
            "situation_interpretation": {"what_changed": "x", "why_it_matters": "y"},
            "actor_state_update": {"important_memories": ["noted the event"]},
        })
    return llm


def build_world(bid="b000", *, actors=("leader_a", "member_b", "member_c", "member_d"),
                coalition=True):
    w = WorldState("w", bid, SimulationClock(T0, T0), network=RelationGraph(),
                   information=InformationLedger())
    for aid in actors:
        e = Entity(aid)
        e.set("roles", F(["coalition member"], status="observed"))
        e.set("past_actions", F([], status="observed"))
        w.entities[aid] = e
    for other in actors[1:]:
        w.network.add(actors[0], "influences", other)
    if len(actors) > 2:
        w.network.add(actors[1], "communicates_with", actors[2])
    if coalition:
        w.institutions["coalition"] = RuleSystem("coalition", [
            Rule("coalition:0", "decision_right",
                 {"holders": list(actors), "actions": ["support", "oppose"]})])
    return w


def runtime_for(script, **cfg_kw):
    cfg = QualitativeConfig(llm=scripted_llm(script), llm_hypotheses=False, n_hypotheses=3,
                            max_llm_calls=cfg_kw.pop("max_llm_calls", 64), **cfg_kw)
    return QualitativeActorPolicyRuntime(QualitativeDecisionEngine(cfg),
                                         mode="persistent_qualitative_llm_policy")


def statement_event(actor="leader_a", *, situation="State your position publicly.", ts=T0 + 60):
    return Event(ts=ts, etype="decision_opportunity", participants=[actor],
                 payload={"situation": situation,
                          "candidate_actions": [
                              {"name": "hold_position", "observability": {"default": "public"},
                               "parameters": {"content": "We stay the course. No concessions."}},
                              {"name": "concede"}, {"name": "wait"}]},
                 source="scheduled")


def run_branch(world, runtime, events, *, seed=7, max_events=80, extra_ops=()):
    op = ProductionActorPolicyOperator(runtime=runtime)
    q = EventQueue(horizon_ts=T0 + 30 * 86400)
    for ev in events:
        q.schedule(ev)
    branch = RolloutEngine(operators=[op, *extra_ops]).run_branch(world, q, seed=seed,
                                                                  max_events=max_events)
    return branch, op


# ------------------------------------------------------------------ 1. public-statement cascade
def test_public_statement_cascade_with_institutional_threshold():
    """A's public statement → B/C/D observe → coherent-hypothesis-dependent reactions → new
    events → a collective vote computed from EXECUTED actions — with no direct
    coalition-support scalar write anywhere."""
    script = {
        "leader_a": lambda p: {"act_or_wait": "act", "chosen_action": "hold_position",
                               "target": "", "observability": "public",
                               "intended_effect": "signal resolve"},
        # B and C react differently BY HYPOTHESIS (their private state text is in the prompt)
        "member_b": lambda p: ({"act_or_wait": "act", "chosen_action": "support", "target": "coalition",
                                "observability": "public", "intended_effect": "back the leader"}
                               if "steady_confident" in p else
                               {"act_or_wait": "act", "chosen_action": "oppose", "target": "coalition",
                                "observability": "public", "intended_effect": "break ranks"}),
        "member_c": lambda p: {"act_or_wait": "act", "chosen_action": "support",
                               "target": "coalition", "observability": "public",
                               "intended_effect": "stay loyal"},
        "member_d": lambda p: {"act_or_wait": "wait", "chosen_action": "wait", "target": "",
                               "observability": "private", "intended_effect": "wait and see"},
    }
    w = build_world()
    rt = runtime_for(script)
    vote = Event(ts=T0 + 5 * 86400, etype="collective_vote",
                 participants=["member_b", "member_c", "member_d"],
                 payload={"threshold": 0.5, "outcome_var": "coalition_holds"}, source="scheduled")
    branch, _ = run_branch(w, rt, [statement_event(), vote], seed=3,
                           extra_ops=(InstitutionalVoteOperator(),))

    sevs = w.uncertainty_meta.get("semantic_events", [])
    assert any(s["event_type"] == "public_statement" and s["actor_id"] == "leader_a"
               for s in sevs)
    # invariant 1: the public event was observed by multiple actors
    exposed = {e.actor_id for e in w.information.exposures}
    assert {"member_b", "member_c", "member_d"} <= exposed
    # reactions executed as the members' OWN actions
    acted = {aid: [a["action"] for a in (w.entities[aid].value("past_actions") or [])]
             for aid in w.entities}
    assert "hold_position" in acted["leader_a"]
    assert acted["member_c"] and acted["member_c"][0] in ("support", "oppose")
    # the institutional outcome came from run_vote over EXECUTED current_action values
    assert "coalition_holds" in w.quantities
    # no direct scalar coalition-support write anywhere in the log
    for d in branch.log:
        for ch in d.changes:
            assert "coalition_support" not in str(ch.get("path", ""))
    # cascade bookkeeping exists and terminated
    cascade = w.uncertainty_meta["event_cascade"]
    assert cascade["scheduled"] >= 2
    assert cascade["scheduled"] <= 24


def test_different_particles_produce_different_actions():
    """Invariant 8: branch b000 and b001 inhabit different hypotheses → different choices."""
    script = {
        "leader_a": lambda p: {"act_or_wait": "act", "chosen_action": "hold_position",
                               "target": "", "observability": "public",
                               "intended_effect": "resolve"},
        "member_b": lambda p: ({"act_or_wait": "act", "chosen_action": "support",
                                "target": "coalition", "observability": "public",
                                "intended_effect": "back"}
                               if "steady_confident" in p else
                               {"act_or_wait": "act", "chosen_action": "defect", "target": "",
                                "observability": "public", "intended_effect": "leave"}),
    }
    chosen = {}
    for bid in ("b000", "b001"):
        w = build_world(bid, actors=("leader_a", "member_b"))
        rt = runtime_for(script)
        run_branch(w, rt, [statement_event()], seed=11)
        chosen[bid] = [a["action"] for a in (w.entities["member_b"].value("past_actions") or [])]
    assert chosen["b000"] != chosen["b001"], chosen


# ------------------------------------------------------------------ 2. private concession
def test_private_concession_boundary_and_conditional_relay():
    """A privately concedes to B: C does not observe it. In the relay branch B tells C and only
    then does C react (to B's account, not to A's original)."""
    def a(p):
        return {"act_or_wait": "act", "chosen_action": "concede", "target": "member_b",
                "observability": "private",
                "intended_effect": "privately offer terms to B"}

    # branch 1: B keeps it to itself
    w1 = build_world(actors=("leader_a", "member_b", "member_c"))
    rt1 = runtime_for({"leader_a": a,
                       "member_b": lambda p: {"act_or_wait": "wait", "chosen_action": "wait",
                                              "target": "", "observability": "private",
                                              "intended_effect": "sit on it"}})
    ev = Event(ts=T0 + 60, etype="decision_opportunity", participants=["leader_a"],
               payload={"situation": "Decide whether to concede privately.",
                        "candidate_actions": [
                            {"name": "concede", "target": "member_b",
                             "observability": {"default": "participants"},
                             "parameters": {"content": "I can offer you better terms quietly."}},
                            {"name": "wait"}]}, source="scheduled")
    run_branch(w1, rt1, [copy.deepcopy(ev)])
    seen_c = [w1.information.items[e.item_id].content for e in w1.information.exposures
              if e.actor_id == "member_c"]
    seen_b = [w1.information.items[e.item_id].content for e in w1.information.exposures
              if e.actor_id == "member_b"]
    assert any("better terms" in s for s in seen_b)
    assert not seen_c, f"private event leaked to member_c: {seen_c}"     # invariant 2
    assert not [a_ for a_ in (w1.entities["member_c"].value("past_actions") or [])]

    # branch 2: B relays to C → C reacts only because the communication actually occurred
    w2 = build_world(actors=("leader_a", "member_b", "member_c"))
    rt2 = runtime_for({
        "leader_a": a,
        "member_b": lambda p: ({"act_or_wait": "act", "chosen_action": "coordinate",
                                "target": "member_c", "observability": "private",
                                "intended_effect": "tell C about the private offer"}
                               if "better terms" in p else
                               {"act_or_wait": "wait", "chosen_action": "wait", "target": "",
                                "observability": "private", "intended_effect": ""}),
        "member_c": lambda p: {"act_or_wait": "act", "chosen_action": "clarify",
                               "target": "member_b", "observability": "private",
                               "intended_effect": "ask B what this means"},
    })
    run_branch(w2, rt2, [copy.deepcopy(ev)])
    c_actions = [x["action"] for x in (w2.entities["member_c"].value("past_actions") or [])]
    assert "clarify" in c_actions, "C reacted only after B's relay actually occurred"
    seen_c2 = [w2.information.items[e.item_id].source for e in w2.information.exposures
               if e.actor_id == "member_c"]
    assert "member_b" in seen_c2 and "leader_a" not in seen_c2, \
        "C's information came from B's account, never A's original private message"


# ------------------------------------------------------------------ 3. joint-world coherence
def test_joint_world_conditions_actor_states_and_rejects_incoherence():
    from swm.world_model_v2.init_state import InitialStateModel
    from swm.world_model_v2.joint_world import (JointWorldHypothesizer, attach_joint_hypotheses,
                                                coherent)
    base = build_world()
    init = InitialStateModel(base_world=base)
    rows = JointWorldHypothesizer(None, k=3).generate(question="will the coalition hold?")
    attach_joint_hypotheses(init, rows)
    worlds = init.sample_particles(3, seed=0)
    ids = [w.uncertainty_meta["joint_world_hypothesis"]["hypothesis_id"] for w in worlds]
    assert len(set(ids)) == 3, "three particles inhabit three different joint hypotheses"
    # actor states are conditioned on the branch hypothesis and share it within a particle
    rt = runtime_for({})
    for w in worlds:
        states = []
        for aid in ("leader_a", "member_b"):
            view = rt.views.build(w, aid)
            st = rt.engine.hypothesizer.state_for_branch(w, view)
            states.append(st)
            assert st.hypothesis_id.startswith(
                w.uncertainty_meta["joint_world_hypothesis"]["hypothesis_id"] + "/")
        # invariant 9: same shared world hypothesis for all actors in one particle
        assert states[0].hypothesis_id.split("/")[0] == states[1].hypothesis_id.split("/")[0]
    # the adverse world leads actor fallbacks with doubt, the stable world with confidence
    by_label = {w.uncertainty_meta["joint_world_hypothesis"]["label"]: w for w in worlds}
    v_stable = rt.views.build(by_label["stable_aligned"], "leader_a")
    st_stable = rt.engine.hypothesizer.state_for_branch(by_label["stable_aligned"], v_stable)
    v_adv = rt.views.build(by_label["private_collapse"], "leader_a")
    st_adv = rt.engine.hypothesizer.state_for_branch(by_label["private_collapse"], v_adv)
    assert st_stable.core_worldview != st_adv.core_worldview
    # contradictory latent combinations are rejected
    assert not coherent({"evidence_reliability": "heavily filtered",
                         "coalition_cohesion": "fully transparent alignment"})


def test_independent_particles_do_not_share_mutable_private_state():
    """Invariant 10: mutating one particle's persistent actor state leaves siblings intact."""
    from swm.world_model_v2.init_state import InitialStateModel
    init = InitialStateModel(base_world=build_world())
    w1, w2 = init.sample_particles(2, seed=0)
    rt = runtime_for({})
    view = rt.views.build(w1, "member_b")
    st = rt.engine.hypothesizer.state_for_branch(w1, view)
    from swm.world_model_v2.qualitative_actor import store_actor_state
    store_actor_state(w1, st, method="test")
    st.apply_update({"current_private_beliefs": ["only in w1"]}, at=T0, event="t", source="t")
    store_actor_state(w1, st, method="test")
    assert load_actor_state(w2, "member_b") is None
    assert "only in w1" in (load_actor_state(w1, "member_b").current_private_beliefs)


# ------------------------------------------------------------------ 4. novel coordination
def test_novel_multi_target_coordination_needs_no_coefficient():
    """A novel action 'privately ask two wavering members' compiles into TWO private
    communication events with explicit targets; both recipients observe; no ontology
    coefficient exists for it anywhere."""
    script = {
        "leader_a": lambda p: {
            "act_or_wait": "act", "chosen_action": "sound_out_waverers",
            "target": "member_b", "observability": "private",
            "intended_effect": "ask B and C privately whether they would defect together",
            "linked_actions": ["sound_out_waverers@member_c"],
            "novel_action_proposal": {
                "present": True,
                "description": "privately ask the two wavering members whether they would "
                               "defect together",
                "required_authority": "none", "required_resources": "none",
                "proposed_mechanisms": "private conversations"}},
        "member_b": lambda p: {"act_or_wait": "wait", "chosen_action": "wait", "target": "",
                               "observability": "private", "intended_effect": ""},
        "member_c": lambda p: {"act_or_wait": "wait", "chosen_action": "wait", "target": "",
                               "observability": "private", "intended_effect": ""},
    }
    w = build_world(actors=("leader_a", "member_b", "member_c"))
    rt = runtime_for(script)
    ev = Event(ts=T0 + 60, etype="decision_opportunity", participants=["leader_a"],
               payload={"situation": "Coalition discipline is slipping; decide your move.",
                        "candidate_actions": [{"name": "wait"}]}, source="scheduled")
    run_branch(w, rt, [ev])
    sevs = w.uncertainty_meta.get("semantic_events", [])
    private = [s for s in sevs if s["actor_id"] == "leader_a"
               and s["observability"] == "participants"]
    targets = sorted(t for s in private for t in s["targets"])
    assert targets == ["member_b", "member_c"], f"two per-target private events: {targets}"
    from swm.world_model_v2.phase4_policy import action_pathway_effects
    assert action_pathway_effects("generic", "sound_out_waverers") == {}
    exposed = {e.actor_id for e in w.information.exposures}
    assert {"member_b", "member_c"} <= exposed


# ------------------------------------------------------------------ 5. tier promotion
def test_event_time_tier_promotion_grants_cognition():
    """A previously routine actor holding a live decision right is promoted by the frontier and
    receives persistent qualitative cognition (invariant 7)."""
    from swm.world_model_v2.actor_selection import RelevantActorSelector
    from swm.world_model_v2.causal_frontier import CausalFrontierDiscovery
    from swm.world_model_v2.semantic_events import SemanticEvent
    w = build_world(actors=("leader_a", "routine_r"))
    w.institutions["board"] = RuleSystem("board", [
        Rule("board:0", "decision_right", {"holders": ["routine_r"], "actions": ["approve"]})])
    sev = SemanticEvent(event_id="s1", event_type="public_statement", actor_id="leader_a",
                        targets=[], intended_audience=["*"], observability="public",
                        timestamp=T0)
    frontier = CausalFrontierDiscovery(selector=RelevantActorSelector()).discover(
        w, sev, [], tiers={})
    r = next(a for a in frontier if a.actor_id == "routine_r")
    assert r.tier == 1 and any("decision_right" in x for x in r.reasons)
    # promoted actor gets persistent qualitative state through the standard path
    rt = runtime_for({"routine_r": lambda p: {"act_or_wait": "act", "chosen_action": "approve",
                                              "target": "board", "observability": "public",
                                              "intended_effect": "approve"}})
    ev = Event(ts=T0 + 60, etype="actor_reconsideration", participants=["routine_r", "leader_a"],
               payload={"situation": "the statement changes your calculus",
                        "candidate_actions": [{"name": "approve", "family": "institutional",
                                               "target": "board",
                                               "institutional_permissions": ["approve"],
                                               "mechanisms_triggered": ["institution_processing"]},
                                              {"name": "wait"}],
                        "depth": 1}, source="endogenous:test")
    run_branch(w, rt, [ev])
    st = load_actor_state(w, "routine_r")
    assert st is not None and st.revision_log, "promoted actor carries persistent cognition"


# ------------------------------------------------------------------ 6. structural boundary
def test_unauthorized_action_blocked_without_semantic_success():
    """The LLM chooses an unauthorized institutional action; deterministic feasibility blocks
    it; NO downstream semantic success event or reconsideration is emitted."""
    script = {"member_b": lambda p: {"act_or_wait": "act", "chosen_action": "approve",
                                     "target": "treasury", "observability": "public",
                                     "intended_effect": "approve the disbursement"}}
    w = build_world(actors=("member_b", "member_c"), coalition=False)
    w.institutions["treasury"] = RuleSystem("treasury", [
        Rule("treasury:0", "decision_right", {"holders": ["member_c"], "actions": ["approve"]})])
    rt = runtime_for(script)
    ev = Event(ts=T0 + 60, etype="decision_opportunity", participants=["member_b"],
               payload={"situation": "the disbursement decision is open",
                        "candidate_actions": [
                            {"name": "approve", "family": "institutional", "target": "treasury",
                             "institutional_permissions": ["approve"],
                             "mechanisms_triggered": ["institution_processing"]},
                            {"name": "wait"}]}, source="scheduled")
    branch, _ = run_branch(w, rt, [ev])
    blocked = [d for d in branch.log if d.event_type == "action_blocked"]
    assert blocked, "the world blocked the unauthorized attempt"
    assert not w.uncertainty_meta.get("semantic_events"), "no semantic success event emitted"
    assert w.uncertainty_meta.get("event_cascade", {}).get("scheduled", 0) == 0


# ------------------------------------------------------------------ 7. scalar-bypass regression
def test_no_direct_belief_or_pathway_write_for_consequential_targets():
    """An action carrying a legacy belief_delta consequence and an actor-mediated pathway
    coefficient executes; the recipient's belief must NOT move directly and the pathway must
    NOT move from the coefficient — the reaction routes through the recipient."""
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    script = {
        "leader_a": lambda p: {"act_or_wait": "act", "chosen_action": "persuade",
                               "target": "member_b", "observability": "private",
                               "intended_effect": "bring B around"},
        "member_b": lambda p: {"act_or_wait": "act", "chosen_action": "hold_position",
                               "target": "", "observability": "public",
                               "intended_effect": "unmoved"},
    }
    w = build_world(actors=("leader_a", "member_b"))
    register_quantity_type("pathway_progress", units="process_state")
    w.quantities["pathway_progress:cooperative_agreement"] = Quantity(
        name="pathway_progress:cooperative_agreement", qtype="pathway_progress", value=0.5,
        timestamp=T0)
    w.entities["member_b"].set("beliefs", F(0.5, status="inferred"), key="support_pact")
    rt = runtime_for(script)
    ev = Event(ts=T0 + 60, etype="decision_opportunity", participants=["leader_a"],
               payload={"situation": "B is wavering — what do you do?",
                        "candidate_actions": [
                            {"name": "persuade", "target": "member_b",
                             "possible_consequences": [{"kind": "belief_delta",
                                                        "belief": "support_pact",
                                                        "delta": 0.2}]},
                            {"name": "wait"}]}, source="scheduled")
    branch, _ = run_branch(w, rt, [ev])
    b = w.entities["member_b"].get("beliefs", key="support_pact")
    assert float(b.value) == 0.5, "recipient belief must not move by direct scalar write"
    # the persuade coefficient (+0.2 on cooperative_agreement) must NOT fire; the only pathway
    # movement allowed is the RECIPIENT's own executed reaction (hold_position, negative)
    assert float(w.quantities["pathway_progress:cooperative_agreement"].value) <= 0.5, \
        "actor-mediated persuade coefficient must not advance the process"
    demoted = w.uncertainty_meta.get("demoted_scalar_writes", [])
    assert {d["kind"] for d in demoted} >= {"belief_delta", "pathway_effect"}
    # ...and the recipient actually got a reconsideration through the canonical queue
    assert any(d["kind"] == "belief_delta" for d in demoted)
    b_actions = [x["action"] for x in (w.entities["member_b"].value("past_actions") or [])]
    assert "hold_position" in b_actions, "the recipient's own decision carried the reaction"


# ------------------------------------------------------------------ 8. quiescence / loop control
def test_mutual_acknowledgement_terminates_by_dedup():
    """Two actors repeatedly acknowledging each other must not cascade forever: semantic
    duplicates suppress, the branch terminates through quiescence within budget."""
    script = {
        "member_b": lambda p: {"act_or_wait": "act", "chosen_action": "acknowledge",
                               "target": "member_c", "observability": "private",
                               "intended_effect": "ack"},
        "member_c": lambda p: {"act_or_wait": "act", "chosen_action": "acknowledge",
                               "target": "member_b", "observability": "private",
                               "intended_effect": "ack"},
    }
    w = build_world(actors=("member_b", "member_c"), coalition=False)
    rt = runtime_for(script, max_llm_calls=64)
    ev = Event(ts=T0 + 60, etype="decision_opportunity", participants=["member_b"],
               payload={"situation": "C sent you a note",
                        "candidate_actions": [{"name": "acknowledge", "target": "member_c"},
                                              {"name": "wait"}]}, source="scheduled")
    branch, _ = run_branch(w, rt, [ev], max_events=200)
    cascade = w.uncertainty_meta["event_cascade"]
    assert cascade["scheduled"] <= 24
    assert cascade["suppressed_duplicate"] >= 1
    assert cascade["quiescence"] in ("duplicate_semantic_event", "event_budget",
                                     "frontier_empty", "depth_budget")
    assert rt.engine.calls_used() < 64, "cascade terminated well before the LLM budget"


# ------------------------------------------------------------------ 9. counterfactual parity
def test_phase13_matched_arms_share_shocks_but_reactions_diverge():
    """Two Phase 13 arms clone the SAME joint particles under CRN: hazard shocks identical
    across arms; actor-mediated downstream reactions may diverge because the initiating
    action differs (invariants 18/19)."""
    from swm.world_model_v2.phase13.counterfactual import MatchedEvaluator
    from swm.world_model_v2.phase13.ontology import ActionSchema
    from swm.world_model_v2.init_state import InitialStateModel
    from swm.world_model_v2.joint_world import JointWorldHypothesizer, attach_joint_hypotheses
    from swm.world_model_v2.contracts import OutcomeContract

    script = {
        "member_b": lambda p: ({"act_or_wait": "act", "chosen_action": "support",
                                "target": "", "observability": "public",
                                "intended_effect": "approve of the concession"}
                               if "concession" in p.lower() else
                               {"act_or_wait": "act", "chosen_action": "oppose", "target": "",
                                "observability": "public", "intended_effect": "resist"}),
    }
    base = build_world(actors=("leader_a", "member_b"))
    # phase13's DecisionActionOperator records the decision-maker's act as KEYED past_actions
    # state; leave the field unset so both writers use their own layout without conflict
    base.entities["leader_a"].fields.pop("past_actions", None)
    init = InitialStateModel(base_world=base)
    attach_joint_hypotheses(init, JointWorldHypothesizer(None, k=2).generate(question="q"))

    def outcome(world):
        acts = [a["action"] for a in (world.entities["member_b"].value("past_actions") or [])
                if isinstance(a, dict)]
        return {"b_supported": "support" in acts}

    def qb(world):
        q = EventQueue(horizon_ts=T0 + 10 * 86400)
        q.schedule(Event(ts=T0 + 3600, etype="decision_opportunity",
                         participants=["member_b"],
                         payload={"situation": "react to the leader's latest move",
                                  "candidate_actions": [{"name": "support"}, {"name": "oppose"},
                                                        {"name": "wait"}]},
                         source="scheduled"))
        return q

    contract = OutcomeContract(family="categorical", options=["True", "False"],
                               readout=lambda w: str(outcome(w)["b_supported"]))
    ev = MatchedEvaluator(
        initial=init, queue_builder=qb,
        operators=[ProductionActorPolicyOperator(runtime=runtime_for(script))],
        contract=contract, n_particles=2, seed=5, outcome_fn=outcome)
    concede = ActionSchema(action_id="concede_pub", actor="leader_a", operation="publish",
                           params={"content": "the leader announces a public concession"},
                           recipients=["member_b"], observability="public")
    nothing = ActionSchema(action_id="do_nothing", actor="leader_a", operation="do_nothing")
    bundle = ev.evaluate([concede, nothing], reference_id="do_nothing")
    match = bundle.crn_manifest["exogenous_trace_match_vs_reference"]
    for arm_id, frac in match.items():
        assert frac == 1.0, f"matched arms saw identical exogenous shocks ({arm_id}: {frac})"
    outcomes = {a: [o.get("readout") for o in arm.outcomes]
                for a, arm in bundle.arms.items()}
    assert outcomes["concede_pub"] != outcomes["do_nothing"], \
        f"actor-mediated reactions diverge across arms: {outcomes}"


# ------------------------------------------------------------------ leakage + representation
def test_private_state_and_simulator_truth_never_enter_prompts():
    """Invariants 4/5: another actor's qualitative state and simulator-only truth never appear
    in a decision prompt; the actor's own state does."""
    captured = []

    def spy_llm(prompt):
        captured.append(prompt)
        return json.dumps({"schema_version": "qualitative.actor.v1",
                           "decision": {"act_or_wait": "wait", "chosen_action": "wait",
                                        "target": "", "timing": "immediate",
                                        "observability": "private", "intended_effect": ""},
                           "decision_summary": "", "novel_action_proposal": {"present": False},
                           "situation_interpretation": {}, "actor_state_update": {}})

    w = build_world(actors=("leader_a", "member_b"))
    from swm.world_model_v2.init_state import InitialStateModel
    from swm.world_model_v2.joint_world import JointWorldHypothesizer, attach_joint_hypotheses
    init = InitialStateModel(base_world=w)
    attach_joint_hypotheses(init, JointWorldHypothesizer(None, k=2).generate(question="q"))
    world = init.sample_particles(1, seed=0)[0]
    cfg = QualitativeConfig(llm=spy_llm, llm_hypotheses=False, n_hypotheses=2)
    rt = QualitativeActorPolicyRuntime(QualitativeDecisionEngine(cfg),
                                       mode="persistent_qualitative_llm_policy")
    # give member_b a private marker that must never surface in leader_a's prompt
    view_b = rt.views.build(world, "member_b")
    st_b = rt.engine.hypothesizer.state_for_branch(world, view_b)
    st_b.current_private_beliefs = ["SECRET_B_MARKER: I will defect on Tuesday"]
    from swm.world_model_v2.qualitative_actor import store_actor_state
    store_actor_state(world, st_b, method="test")
    ev = Event(ts=T0 + 60, etype="decision_opportunity", participants=["leader_a"],
               payload={"situation": "decide", "candidate_actions": [{"name": "wait"}]},
               source="scheduled")
    run_branch(world, rt, [ev])
    a_prompts = [p for p in captured if "You ARE leader_a" in p]
    assert a_prompts
    for p in a_prompts:
        assert "SECRET_B_MARKER" not in p                      # invariant 4
        assert "particle_weight" not in p and "branch_id" not in p \
            and "hypothesis_id" not in p.replace("HYPOTHESIS OF YOUR HIDDEN REALITY", "")
        assert "joint_world_hypothesis" not in p               # invariant 5 (simulator key)


def test_broadcast_recipients_get_summarized_representation():
    """Invariant 3: a non-target public observer may receive a SUMMARY of a long statement
    while the direct target receives the original — different representations, recorded."""
    from swm.world_model_v2.observation_delivery import ObservationRouter
    from swm.world_model_v2.semantic_events import SemanticEvent
    w = build_world(actors=("leader_a", "member_b", "member_c"), coalition=False)
    long_text = "We will hold our position. " * 40
    sev = SemanticEvent(event_id="s_long", event_type="public_statement", actor_id="leader_a",
                        targets=["member_b"], intended_audience=["*"], observability="public",
                        exact_content=long_text, timestamp=T0, channel="public_broadcast")
    deliveries = ObservationRouter().deliver(w, sev)
    by_recipient = {d.recipient_id: d for d in deliveries}
    assert by_recipient["member_b"].representation == "original"
    assert by_recipient["member_c"].representation == "summary"
    assert by_recipient["member_c"].distortion.get("summarized") is True


def test_terminal_probability_cannot_be_written_by_actor_code():
    """Invariant 17: an action consequence naming a probability quantity is refused."""
    from swm.world_model_v2.phase4_policy import TypedAction, ActionTarget
    from swm.world_model_v2.phase4_execution import ActorPolicyRuntime
    w = build_world(actors=("leader_a",), coalition=False)
    action = TypedAction(action_id="x", actor_id="leader_a", actor_role="r",
                         action_family="generic", action_name="act",
                         target=ActionTarget(),
                         possible_consequences=[{"kind": "quantity_delta",
                                                 "name": "terminal_probability", "delta": 0.9}],
                         mechanisms_triggered=["record_action"])
    from swm.world_model_v2.transitions import StateDelta
    delta = StateDelta(at=T0, event_type="actor_action", operator="test")
    ActorPolicyRuntime._apply_immediate_consequences(w, action, delta)
    assert "terminal_probability" not in w.quantities


def test_replayable_same_seed_same_cascade():
    """Invariant 20: identical inputs + seed → identical cascade manifest and decisions."""
    script = {
        "leader_a": lambda p: {"act_or_wait": "act", "chosen_action": "hold_position",
                               "target": "", "observability": "public",
                               "intended_effect": "resolve"},
        "member_b": lambda p: {"act_or_wait": "act", "chosen_action": "support",
                               "target": "coalition", "observability": "public",
                               "intended_effect": "back"},
    }
    manifests = []
    for _ in range(2):
        w = build_world(actors=("leader_a", "member_b"))
        rt = runtime_for(script)
        run_branch(w, rt, [statement_event()], seed=13)
        c = w.uncertainty_meta["event_cascade"]
        manifests.append((c["scheduled"], c["max_depth_reached"],
                          tuple(tuple(x) for x in c["reconsidered"]),
                          tuple(a["action"] for a in
                                (w.entities["member_b"].value("past_actions") or []))))
    assert manifests[0] == manifests[1]
