"""Phase 4L LLM persona actor acceptance tests — offline, deterministic, scripted backends.

Covers: the first-person prompt's information boundary, strict abstaining parse, the anchored
log-pool blend (including the fail-closed identity with the numeric path), novel-action
validation through the TypedAction contract, persistent cognition write-back and its round-trip
into the next decision's view/prompt, relevance gating and dynamic promotion, cache/budget
behavior, operator integration, and provenance honesty (llm_probability_minting)."""
import copy
import json
import random

import pytest

from swm.world_model_v2.events import Event
from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.institutions import Rule, RuleSystem
from swm.world_model_v2.llm_actor import (
    LLMActorPolicyModel, PERSONA_MEMORY_KEY, PersonaActorPolicyRuntime, PersonaCalibration,
    PersonaCognition, PersonaConfig, PersonaEngine, PersonaPromptBuilder, action_menu,
    build_persona_runtime, novel_actions_to_typed, parse_persona_response, persona_relevance,
)
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase4_execution import ActorPolicyRuntime, ProductionActorPolicyOperator
from swm.world_model_v2.phase4_policy import (
    ActionTarget, ActorPolicyModel, ActorViewBuilder, FeasibilityDecision, TypedAction,
)
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

T0 = 1_700_000_000.0


class Plan:
    question = "Will the organization approve the proposal?"

    @staticmethod
    def plan_hash():
        return "planphase4l"


def world():
    w = WorldState("p4l", "b0", SimulationClock(T0, T0), network=RelationGraph(),
                   information=InformationLedger())
    alice = Entity("alice")
    alice.set("roles", F(["manager"], status="observed", sources=["org"]))
    alice.set("goals", F(["complete_project"], status="observed"))
    alice.set("beliefs", F(0.7, status="inferred"), key="proposal_succeeds")
    alice.set("resources", F(10.0, status="observed"), key="budget")
    alice.set("resources", F(0.8, status="observed"), key="capacity")
    alice.set("authority", F(["approve"], status="observed"))
    alice.set("commitments", F([{"id": "c1", "statement": "no approval before audit",
                                 "binding": True, "prohibits": ["launch"]}], status="observed"))
    alice.set("stances", F([{"actor": "alice", "commitment_level": "committed_to_prevent",
                             "pathway": "cooperative_agreement", "quote": "no deal until audit",
                             "reliability": "high", "capability": "high"}], status="derived"))
    alice.set("private_information", F("reservation=3", status="observed"))
    alice.set("latent_state", F("omniscient_truth", status="sampled"), key="hidden_truth")
    alice.set("past_actions", F([], status="observed"))
    bob = Entity("bob")
    bob.set("roles", F(["analyst"], status="observed"))
    bob.set("beliefs", F(0.1, status="inferred"), key="private_belief")
    w.entities = {"alice": alice, "bob": bob}
    w.network.add("alice", "communicates_with", "bob")
    w.information.publish(InformationItem("public1", "proposal filed", source="registry",
                                          created_at=T0 - 5))
    w.information.publish(InformationItem("future1", "proposal resolved", source="future",
                                          created_at=T0 + 50))
    w.information.expose("alice", "public1", T0 - 4)
    w.information.expose("alice", "future1", T0 + 50)
    w.institutions["board"] = RuleSystem("board", [
        Rule("approve_right", "decision_right", {"actions": ["approve"], "holders": ["alice"]}),
    ])
    return w


DECISION = {"candidate_actions": ["approve", "reject", "delay"],
            "situation": "the board vote is imminent"}


class ScriptedLLM:
    """Deterministic fn(prompt) -> text backend; records every prompt it is shown."""

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        out = self.payload(prompt) if callable(self.payload) else self.payload
        return out if isinstance(out, str) else json.dumps(out)


def good_payload(**over):
    base = {
        "schema_version": "persona.cognition.v1",
        "situation_reading": "the vote decides my project",
        "appraisals": {"approve": {"inclination": 0.92, "why": "my goal needs it"},
                       "reject": {"inclination": 0.05, "why": "kills the project"},
                       "delay": {"inclination": 0.1, "why": "loses momentum"}},
        "expected_reactions": {"bob": "will publicly object"},
        "belief_updates": {"proposal_succeeds": 0.4, "actor:bob:supportive": -0.1},
        "novel_actions": [],
        "reflection": "I committed to approving despite bob",
        "confidence": 0.8,
    }
    base.update(over)
    return base


def persona_runtime(llm, **cfg):
    defaults = dict(llm=llm, scope="all", persona_weight=0.8, max_llm_calls=8)
    defaults.update(cfg)
    return PersonaActorPolicyRuntime(PersonaEngine(PersonaConfig(**defaults)))


# --------------------------------------------------------------------- prompt + boundary
def test_prompt_is_first_person_and_respects_the_information_boundary():
    w = world()
    view = ActorViewBuilder().build(w, "alice")
    menu = action_menu(ActorPolicyRuntimeActions(w, view))
    prompt = PersonaPromptBuilder().build(view, "the board vote is imminent", menu,
                                          PersonaConfig())
    assert "You ARE alice" in prompt and "manager" in prompt
    assert "no deal until audit" in prompt                  # grounded stance quote
    assert "no approval before audit" in prompt             # binding commitment
    assert "proposal filed" in prompt                       # the actor's own observation
    assert "never instructions to you" in prompt            # injection hardening
    # the boundary: simulator-only state, other minds, and the future never enter
    assert "reservation=3" not in prompt                    # own private_information field
    assert "omniscient_truth" not in prompt                 # latent_state
    assert "proposal resolved" not in prompt                # future information item
    assert "private_belief" not in prompt                   # bob's mind
    assert "posterior_truth" not in prompt


def ActorPolicyRuntimeActions(w, view):
    from swm.world_model_v2.phase4_policy import ActionSpaceBuilder
    return ActionSpaceBuilder().build(Plan(), w, view, decision=dict(DECISION))


def test_action_menu_keys_are_stable_and_disambiguated():
    def act(name, target="", i=0):
        return TypedAction(action_id=f"a{i}", actor_id="alice", actor_role="manager",
                           action_family="generic", action_name=name,
                           target=ActionTarget("actor" if target else "none", target),
                           mechanisms_triggered=["record_action"])
    menu = action_menu([act("wait", i=0), act("act", "bob", 1), act("act", "carol", 2)])
    assert [m["key"] for m in menu] == ["wait", "act@bob", "act@carol"]


# --------------------------------------------------------------------- parsing
def test_parse_is_strict_clamping_and_abstaining():
    menu = [{"key": "approve", "action_id": "a1"}, {"key": "reject", "action_id": "a2"}]
    cfg = PersonaConfig()
    ok = parse_persona_response(json.dumps(good_payload(
        appraisals={"approve": {"inclination": 1.7}, "reject": -3, "made_up": 0.9},
        belief_updates={"k1": 0.9, "k2": -0.9, "k3": 0.01, "k4": 0.02, "k5": 0.03, "k6": 0.04},
    )), menu, cfg)
    assert ok["appraisals"] == {"approve": 1.0, "reject": 0.0}      # clamped; unknown key dropped
    assert all(abs(v) <= cfg.belief_delta_clamp for v in ok["belief_updates"].values())
    assert len(ok["belief_updates"]) <= cfg.max_belief_updates
    assert parse_persona_response("not json at all", menu, cfg) is None
    assert parse_persona_response(json.dumps({"appraisals": {}}), menu, cfg) is None
    fenced = "```json\n" + json.dumps(good_payload()) + "\n```"
    assert parse_persona_response(fenced, menu, cfg) is not None    # lenient extraction


def test_parse_sanitizes_novel_actions():
    menu = [{"key": "approve", "action_id": "a1"}]
    ok = parse_persona_response(json.dumps(good_payload(novel_actions=[
        {"name": "Secret Back-Channel!", "family": "bogus_family", "target": "bob",
         "why": "test terms quietly", "inclination": 0.7},
        {"name": "", "family": "negotiation"},
        {"name": "x" * 99, "family": "negotiation", "probability": 0.9},
    ])), menu, PersonaConfig(max_novel_actions=3))
    names = [p["name"] for p in ok["novel_actions"]]
    assert names[0] == "secret_back_channel"
    assert ok["novel_actions"][0]["family"] == "generic"            # bogus family demoted
    assert ok["novel_actions"][0]["inclination"] == 0.7
    assert all(len(n) <= 40 for n in names)
    assert all("probability" not in p for p in ok["novel_actions"])  # numerics never survive


# --------------------------------------------------------------------- relevance
def test_relevance_scores_consequence_from_the_live_view():
    w = world()
    rich = ActorViewBuilder().build(w, "alice")
    poor = ActorViewBuilder().build(w, "bob")
    r_score, r_why = persona_relevance(rich, DECISION)
    p_score, _ = persona_relevance(poor, {})
    assert r_score >= 0.5 and "grounded_stances" in r_why
    assert p_score < 0.5
    # dynamic promotion: bob acquires a stance mid-run and crosses the threshold
    w.entity("bob").set("stances", F([{"actor": "bob", "commitment_level": "actively_pursuing",
                                       "pathway": "institutional_procedure"}], status="derived"))
    w.entity("bob").set("resources", F(0.5, status="derived"), key="capacity")
    promoted = ActorViewBuilder().build(w, "bob")
    assert persona_relevance(promoted, DECISION)[0] >= 0.5


def test_scope_relevant_skips_inconsequential_actors():
    llm = ScriptedLLM(good_payload())
    w = world()
    views = [ActorViewBuilder().build(w, "bob")]
    actions = ActorPolicyRuntimeActions(w, views[0])
    engine = PersonaEngine(PersonaConfig(llm=llm, scope="relevant"))
    assert engine.cognize(views, None, actions, {}) is None
    assert llm.prompts == []                                        # no call was spent
    assert PersonaEngine(PersonaConfig(llm=llm, scope="all")).cognize(
        views, None, actions, {}) is not None


# --------------------------------------------------------------------- cache + budget
def test_cognition_is_cached_by_prompt_and_budgeted():
    llm = ScriptedLLM(good_payload())
    w = world()
    view = ActorViewBuilder().build(w, "alice")
    actions = ActorPolicyRuntimeActions(w, view)
    engine = PersonaEngine(PersonaConfig(llm=llm, scope="all", max_llm_calls=8))
    c1 = engine.cognize([view], None, actions, DECISION)
    c2 = engine.cognize([view, copy.deepcopy(view)], [0.5, 0.5], actions, DECISION)
    assert c1.response_source == "llm" and c2.response_source == "cache"
    assert len(llm.prompts) == 1
    exhausted = PersonaEngine(PersonaConfig(llm=llm, scope="all", max_llm_calls=0))
    assert exhausted.cognize([view], None, actions, DECISION) is None
    assert len(llm.prompts) == 1                                    # budget spent nothing


def test_parse_failures_are_cached_and_do_not_burn_budget_per_particle():
    llm = ScriptedLLM("garbage, not json")
    w = world()
    view = ActorViewBuilder().build(w, "alice")
    actions = ActorPolicyRuntimeActions(w, view)
    engine = PersonaEngine(PersonaConfig(llm=llm, scope="all", retries=0, max_llm_calls=8))
    assert engine.cognize([view], None, actions, DECISION) is None
    assert engine.cognize([view], None, actions, DECISION) is None
    assert len(llm.prompts) == 1                                    # failure cached


# --------------------------------------------------------------------- calibration + blend
def cognition(appraisals, menu_map):
    return PersonaCognition(actor_id="alice", appraisals=appraisals, menu_map=menu_map)


def test_persona_distribution_redistributes_only_within_rated_mass():
    cfg = PersonaConfig()
    anchor = {"a1": 0.4, "a2": 0.4, "a3": 0.2}
    cog = cognition({"approve": 0.9, "reject": 0.1}, {"approve": "a1", "reject": "a2"})
    p, diag = PersonaCalibration.persona_distribution(cog, anchor, cfg)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["a1"] > p["a2"]                                        # persona preference expressed
    assert abs(p["a3"] - anchor["a3"]) < 1e-9                       # silence changes nothing
    assert abs((p["a1"] + p["a2"]) - 0.8) < 1e-9                    # rated mass preserved
    assert diag["appraised_fraction"] == round(2 / 3, 3)


def test_blend_endpoints_and_infeasible_mass():
    anchor = {"a1": 0.5, "a2": 0.5}
    cog = cognition({"approve": 0.9, "reject": 0.1, "ghost": 0.9},
                    {"approve": "a1", "reject": "a2", "ghost": "aX"})   # aX infeasible everywhere
    p, diag = PersonaCalibration.persona_distribution(cog, anchor, PersonaConfig())
    assert "aX" not in p and diag["llm_mass_on_infeasible"] == 0.9  # zero known-impossible mass
    assert PersonaCalibration.blend(anchor, p, 0.0) == pytest.approx(anchor)
    assert PersonaCalibration.blend(anchor, p, 1.0) == pytest.approx(p)
    mid = PersonaCalibration.blend(anchor, p, 0.5)
    assert anchor["a1"] < mid["a1"] < p["a1"]


def feasibility_rows(actions, infeasible=()):
    return [[FeasibilityDecision(a.action_id,
                                 "institutionally_prohibited" if a.action_id in infeasible else "feasible",
                                 "feasible") for a in actions]]


def test_model_without_cognition_is_bit_identical_to_the_anchor():
    w = world()
    view = ActorViewBuilder().build(w, "alice")
    actions = ActorPolicyRuntimeActions(w, view)
    rows = feasibility_rows(actions)
    anchor = ActorPolicyModel().decide([view], actions, rows, seed=7)
    model = LLMActorPolicyModel(PersonaEngine(PersonaConfig()), ActorPolicyModel())
    blended = model.decide([view], actions, rows, seed=7, cognition=None)
    assert blended.action_probabilities == anchor.action_probabilities
    assert blended.provenance["persona"] == {"active": False, "reason": "no_cognition"}
    assert blended.provenance.get("llm_probability_minting") is not True


# --------------------------------------------------------------------- runtime end-to-end
def test_persona_runtime_decides_executes_and_persists_the_mind():
    llm = ScriptedLLM(good_payload())
    rt = persona_runtime(llm)
    w = world()
    selected, posterior, trace = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=3)
    # provenance honesty
    assert posterior.provenance["llm_probability_minting"] is True
    persona = posterior.provenance["persona"]
    assert persona["active"] is True and persona["response_source"] == "llm"
    assert any(f.get("reason") == "persona_blend_weight_unfitted_prior"
               for f in posterior.fallbacks_used)
    assert trace.cost["llm_calls"] == 1 and trace.verify()
    # the persona's preference moved the posterior relative to the numeric anchor
    plain = ActorPolicyRuntime()
    _, anchor_post, _ = plain.decide(Plan(), [world()], "alice", decision=dict(DECISION), seed=3)
    aid = next(a["action_id"] for a in trace.candidate_actions if a["action_name"] == "approve")
    assert posterior.action_probabilities[aid] > anchor_post.action_probabilities[aid]
    # execution writes the cognition back onto the actor, on the SAME delta
    delta, events = rt.execute(w, selected, posterior, trace, seed=3)
    assert "llm_persona_state_update" in delta.reason_codes
    alice = w.entity("alice")
    memory = alice.value("latent_state", key=PERSONA_MEMORY_KEY)
    assert memory and memory[-1]["note"].startswith("I committed")
    assert alice.value("beliefs", key="proposal_succeeds") == pytest.approx(0.85)  # 0.7 + clamped 0.15
    assert alice.value("beliefs", key="actor:bob:supportive") == pytest.approx(0.4)
    reactions = alice.value("expected_reactions")
    assert reactions["bob"]["expects"] == "will publicly object"
    # the next decision's view (and prompt) carries the mind forward
    view2 = ActorViewBuilder().build(w, "alice")
    assert view2.policy_state[PERSONA_MEMORY_KEY][-1]["note"].startswith("I committed")
    assert "actor:bob:supportive" in view2.beliefs_about_actors
    prompt2 = PersonaPromptBuilder().build(view2, "again", action_menu([selected]),
                                           PersonaConfig())
    assert "I committed to approving despite bob" in prompt2
    assert "bob: will publicly object" in prompt2


def test_persona_runtime_is_deterministic_for_a_fixed_seed_and_backend():
    for _ in range(2):
        picks = set()
        for run in range(2):
            rt = persona_runtime(ScriptedLLM(good_payload()))
            selected, _, _ = rt.decide(Plan(), [world()], "alice",
                                       decision=dict(DECISION), seed=11)
            picks.add(selected.action_id)
        assert len(picks) == 1


def test_novel_actions_enter_only_through_the_typed_contract():
    payload = good_payload(novel_actions=[
        {"name": "quiet_side_channel", "family": "negotiation", "target": "bob",
         "why": "test terms privately", "inclination": 0.85},
        {"name": "call_the_president", "family": "messaging", "target": "carol",   # unreachable
         "why": "escalate", "inclination": 0.9},
    ])
    rt = persona_runtime(ScriptedLLM(payload))
    w = world()
    selected, posterior, trace = rt.decide(Plan(), [w], "alice", decision=dict(DECISION), seed=5)
    rows = {a["action_name"]: a for a in trace.candidate_actions}
    novel = rows["quiet_side_channel"]
    assert novel["provenance"]["source"] == "llm_persona_proposal"
    assert novel["support_status"] == "llm_proposed"
    assert novel["mechanisms_triggered"]                            # executable or it never builds
    assert novel["target"]["target_id"] == "bob"                    # reachable target kept
    assert rows["call_the_president"]["target"]["target_id"] == ""  # unseen target stripped
    assert novel["action_id"] in posterior.action_probabilities     # priced by the blend
    # execution path accepts it like any typed action
    if selected.action_name == "quiet_side_channel":
        delta, _ = rt.execute(w, selected, posterior, trace, seed=5)
        assert delta.event_type == "actor_action"


def test_novel_targets_must_be_visible_to_the_actor():
    w = world()
    view = ActorViewBuilder().build(w, "alice")
    cog = PersonaCognition(actor_id="alice", novel_actions=[
        {"name": "lobby_board", "family": "institutional", "target": "board", "why": "w"},
        {"name": "wait", "family": "generic", "target": "", "why": "duplicate of menu"},
    ])
    existing = [TypedAction(action_id="w1", actor_id="alice", actor_role="manager",
                            action_family="generic", action_name="wait",
                            mechanisms_triggered=["record_action"])]
    out = novel_actions_to_typed(cog, view, {}, existing, PersonaConfig())
    assert [a.action_name for a in out] == ["lobby_board"]          # duplicate dropped
    assert out[0].target.target_type == "institution"               # board visible via rules


# --------------------------------------------------------------------- operator + wiring
def test_production_operator_accepts_a_persona_runtime():
    rt = persona_runtime(ScriptedLLM(good_payload()))
    op = ProductionActorPolicyOperator(runtime=rt)
    w = world()
    event = Event(ts=T0, etype="decision_opportunity", participants=["alice"],
                  payload=dict(DECISION))
    delta, vr = op.run(w, event, random.Random(0))
    assert vr.ok and op.traces and op.traces[-1].verify()
    assert op.traces[-1].cost["llm_calls"] == 1
    history = w.entity("alice").value("past_actions")
    assert history and history[-1]["status"] in ("executed", "blocked")
    # legacy constructions still work
    assert isinstance(ProductionActorPolicyOperator().runtime, ActorPolicyRuntime)
    assert isinstance(ProductionActorPolicyOperator(ActorPolicyModel()).runtime,
                      ActorPolicyRuntime)


def test_build_persona_runtime_gating(monkeypatch):
    assert build_persona_runtime(llm=None) is None
    monkeypatch.setenv("SWM_LLM_ACTORS", "off")
    assert build_persona_runtime(llm=ScriptedLLM(good_payload())) is None
    monkeypatch.setenv("SWM_LLM_ACTORS", "relevant")
    monkeypatch.setenv("SWM_LLM_ACTOR_BUDGET", "5")
    monkeypatch.setenv("SWM_LLM_ACTOR_WEIGHT", "0.25")
    rt = build_persona_runtime(llm=ScriptedLLM(good_payload()))
    assert isinstance(rt, PersonaActorPolicyRuntime)
    assert rt.engine.config.max_llm_calls == 5
    assert rt.engine.config.persona_weight == 0.25
    assert rt.engine.config.source == "env_override"
    explicit = build_persona_runtime(config=PersonaConfig(llm=ScriptedLLM(good_payload()),
                                                          scope="all"))
    assert isinstance(explicit, PersonaActorPolicyRuntime)


def test_materialize_binds_persona_baseline_only_when_its_mode_is_selected(monkeypatch):
    from swm.world_model_v2.materialize import _actor_policy_runtime
    monkeypatch.delenv("SWM_ACTOR_POLICY", raising=False)
    rt, report = _actor_policy_runtime(None, None)
    assert rt is None and report["actual_actor_policy_mode"] == "numeric_policy"
    assert report["requested_actor_policy_mode"] == "hybrid_relevant_actor_policy"
    assert report["reason"] == "no_llm_backend" and report["warning"]   # loud, never silent
    monkeypatch.setenv("SWM_ACTOR_POLICY", "persona_blended_numeric_policy")
    rt, report = _actor_policy_runtime(None, ScriptedLLM(good_payload()))
    assert isinstance(rt, PersonaActorPolicyRuntime) and not report["degraded"]
    monkeypatch.setenv("SWM_ACTOR_POLICY", "numeric_policy")
    rt, report = _actor_policy_runtime(None, ScriptedLLM(good_payload()))
    assert rt is None and "explicitly requested" in report["reason"]


def test_operators_from_plan_mode_router(monkeypatch):
    from types import SimpleNamespace
    from swm.world_model_v2.materialize import operators_from_plan
    from swm.world_model_v2.qualitative_actor import QualitativeActorPolicyRuntime
    plan = SimpleNamespace(
        accepted_mechanisms=[{"mech_id": "production_actor_policy",
                              "operator": "production_actor_policy"}],
        entities=[], institutions=[], scheduled_events=[], actor_decisions=[], relations=[],
        quantities=[], _intention_stances=[], question="q")
    monkeypatch.delenv("SWM_ACTOR_POLICY", raising=False)
    monkeypatch.delenv("SWM_LLM_ACTORS", raising=False)
    # DEFAULT-ON: an LLM backend routes the core funnel to hybrid qualitative cognition
    with_llm, _ = operators_from_plan(plan, llm=ScriptedLLM(good_payload()))
    assert isinstance(with_llm[0], ProductionActorPolicyOperator)
    assert isinstance(with_llm[0].runtime, QualitativeActorPolicyRuntime)
    assert with_llm[0].runtime.mode == "hybrid_relevant_actor_policy"
    monkeypatch.setenv("SWM_ACTOR_POLICY", "persona_blended_numeric_policy")
    persona, _ = operators_from_plan(plan, llm=ScriptedLLM(good_payload()))
    assert isinstance(persona[0].runtime, PersonaActorPolicyRuntime)
    monkeypatch.delenv("SWM_ACTOR_POLICY", raising=False)
    without, _ = operators_from_plan(plan, llm=None)
    assert type(without[0].runtime) is ActorPolicyRuntime
    monkeypatch.setenv("SWM_LLM_ACTORS", "off")
    disabled, _ = operators_from_plan(plan, llm=ScriptedLLM(good_payload()))
    assert type(disabled[0].runtime) is ActorPolicyRuntime


def test_numeric_fallback_records_why_the_persona_was_skipped():
    rt = persona_runtime(ScriptedLLM(good_payload()), scope="relevant")
    w = world()
    # bob is below the relevance threshold → pure numeric decision, reason on its face
    _, posterior, trace = rt.decide(Plan(), [w], "bob",
                                    decision={"candidate_actions": ["wait", "act"]}, seed=2)
    persona = posterior.provenance["persona"]
    assert persona["active"] is False
    assert persona["reason"].startswith("below_relevance_threshold")
    assert trace.cost["llm_calls"] == 0
    assert posterior.provenance.get("llm_probability_minting") is not True


def test_persona_operator_inside_the_event_loop():
    from swm.world_model_v2.events import EventQueue
    from swm.world_model_v2.rollout import RolloutEngine
    llm = ScriptedLLM(good_payload())
    op = ProductionActorPolicyOperator(runtime=persona_runtime(llm))
    w = world()
    q = EventQueue(horizon_ts=T0 + 10 * 86400)
    q.schedule(Event(ts=T0 + 60, etype="decision_opportunity", participants=["alice"],
                     payload=dict(DECISION)))
    branch = RolloutEngine(operators=[op]).run_branch(w, q, seed=4)
    persona_deltas = [d for d in branch.log if "llm_persona_state_update" in d.reason_codes]
    assert persona_deltas                                           # the mind acted in the loop
    assert len(llm.prompts) == 1
    assert w.entity("alice").value("latent_state", key=PERSONA_MEMORY_KEY)


# --------------------------------------------------------------------- multi-particle
def test_multi_particle_prompts_pool_and_particles_share_the_cache():
    llm = ScriptedLLM(good_payload())
    rt = persona_runtime(llm, particle_prompts=1)
    worlds = [world(), world(), world()]
    for i, w in enumerate(worlds):
        w.branch_id = f"b{i}"
    selected, posterior, trace = rt.decide(Plan(), worlds, "alice",
                                           decision=dict(DECISION), seed=9)
    assert len(llm.prompts) == 1                                    # identical particles → one call
    assert posterior.provenance["persona"]["active"] is True
    two = persona_runtime(ScriptedLLM(good_payload()), particle_prompts=2)
    worlds = [world(), world()]
    worlds[1].entity("alice").set("beliefs", F(0.2, status="derived"), key="proposal_succeeds")
    _, posterior2, _ = two.decide(Plan(), worlds, "alice", decision=dict(DECISION), seed=9)
    assert posterior2.provenance["persona"]["n_particle_prompts"] == 2


def test_blended_posterior_keeps_zero_mass_on_perceived_infeasible_actions():
    llm = ScriptedLLM(good_payload(appraisals={"approve": {"inclination": 0.9},
                                               "launch": {"inclination": 0.95},
                                               "reject": {"inclination": 0.1},
                                               "delay": {"inclination": 0.1}}))
    rt = persona_runtime(llm)
    decision = {"candidate_actions": ["approve", "reject", "delay", "launch"],
                "situation": "the board vote is imminent"}
    # launch is prohibited by alice's binding commitment → perceived infeasible in every particle
    _, posterior, trace = rt.decide(Plan(), [world()], "alice", decision=decision, seed=13)
    name_of = {a["action_id"]: a["action_name"] for a in trace.candidate_actions}
    assert "launch" in name_of.values()                             # it was on the table
    assert all(name_of[aid] != "launch" for aid in posterior.action_probabilities)
    assert posterior.provenance["persona"]["llm_mass_on_infeasible"] >= 0.9
