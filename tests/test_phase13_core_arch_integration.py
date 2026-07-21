"""§31/§32/§33 core-architecture integration — Phase 13 consumes the new uncertainty axes.

§31: cross-model action evaluation must consume truncation_report / under_modeled_subtypes /
model_family_report from the ensemble SimulationResult it was asked against; the recommendation
is WITHHELD when an under-modeled subtype is present, when the winner does not survive every
admissible completion of the truncated branch mass (truncation.recommendation_eligibility), or
when >1 model family was configured but the recommendation was exercised under only one; and
structural_ensemble["recommendation_stability"] (None until Phase 13 runs) is filled.

§33: the personal-reaction route records failed samples as FIRST-CLASS TRUNCATED samples (never
numeric substitution, never silent resampling), counts the distribution over completed samples
only (explicit note + §21 bounds), routes the recipient through bounded cognition, and maps
outcomes onto the distinguishable nonresponse states.

§32 (PR#115): the exact realized message text reaches the recipient's cognition whole — the
availability set, working memory, and the decision prompt carry the full content, and the three
separated reply-first judges never write recipient reaction/notice/memory (they only
select/realize the message).
"""
import inspect
import json
from pathlib import Path

import pytest

from swm.world_model_v2.phase13.contracts import (DecisionProblem, DecisionResult, Stakeholder,
                                                  UtilitySpec)
from swm.world_model_v2.phase13.ensemble import recommend_action_across_models
from swm.world_model_v2.result import RECOMMENDATION_STATUSES, SimulationResult

T0 = 1_772_000_000.0


# ================================================================ §31 fixtures
def _problem():
    return DecisionProblem(
        decision_id="d1", decision_maker="avery", authority=["communicate"],
        utility=UtilitySpec(stakeholders=[Stakeholder(
            "avery", utility_fn=lambda o: float(o.get("outcome", 0.0) or 0.0))]))


def _model_result(recommended, eu: dict) -> DecisionResult:
    r = DecisionResult(decision_id="d1", seed=0)
    r.recommended = recommended
    r.recommendation_kind = "action"
    r.reference_action = "do_nothing"
    r.evaluated = [{"action_id": a, "expected_utility": u} for a, u in eu.items()]
    r.provenance["ranking"] = {"order": [{"action_id": a, "score": u}
                                         for a, u in sorted(eu.items(), key=lambda kv: -kv[1])]}
    r.feasibility = []
    return r


def _patch_per_model(monkeypatch, results_by_plan: dict):
    import swm.world_model_v2.phase13.api as api

    def fake_recommend(problem, plan, **kw):
        return results_by_plan[plan]

    def fake_evaluate(problem, actions, plan, **kw):
        return results_by_plan[plan]
    monkeypatch.setattr(api, "recommend_action", fake_recommend)
    monkeypatch.setattr(api, "evaluate_actions", fake_evaluate)


def _models():
    return {"m1": {"plan": "P1", "meta": {"causal_thesis": "thesis one"}},
            "m2": {"plan": "P2", "meta": {"causal_thesis": "thesis two"}}}


def _source(subtypes=(), trunc_share=0.0, fam=None) -> SimulationResult:
    return SimulationResult(
        question="q", simulation_status="completed", support_grade="exploratory",
        raw_distribution={"yes": 0.6, "no": 0.4},
        structural_ensemble={"recommendation_stability": None},
        under_modeled_subtypes=list(subtypes),
        under_modeled_components=([{"component": "grid_ops", "kind": "external_event_family",
                                    "why": "no arrival model", "sensitivity": "decisive"}]
                                  if subtypes else []),
        truncation_report={"total_weight": 1.0, "truncated_weight": trunc_share,
                           "truncated_branch_share": trunc_share,
                           "truncation_reasons": ({"safety_max_events_reached": 2}
                                                  if trunc_share else {}),
                           "answer_settled_under_truncation": trunc_share == 0.0},
        model_family_report=dict(fam or {"model_family_monoculture": True,
                                         "configured_families": ["primary_configured"],
                                         "families": []}))


def _run(monkeypatch, src, eu_a=1.0, eu_b=0.9):
    per = {"P1": _model_result("act_a", {"act_a": eu_a, "act_b": eu_b}),
           "P2": _model_result("act_a", {"act_a": eu_a, "act_b": eu_b})}
    _patch_per_model(monkeypatch, per)
    return recommend_action_across_models(_problem(), _models(), seed=0, source_result=src)


# ================================================================ §31 withholding rules
def test_withheld_on_under_modeled_subtype(monkeypatch):
    src = _source(subtypes=["under_modeled_external_process"])
    r = _run(monkeypatch, src)
    assert r.recommendation_status == "withheld"
    axes = r.provenance["structural_ensemble"]["core_uncertainty_axes"]
    codes = {w["code"] for w in axes["withheld_reasons"]}
    assert "under_modeled_subtype_present" in codes
    assert axes["under_modeled_subtypes"] == ["under_modeled_external_process"]
    assert axes["under_modeled_components"]              # the named components ride along
    # the ensemble result's own recommendation axis mirrors the withholding
    assert src.recommendation_status == "withheld"


def test_withheld_on_truncated_ineligible_winner(monkeypatch):
    # eu range (0.9, 1.0); t=0.6 ⇒ truncated span 0.06 > completed margin 0.4*0.1=0.04 ⇒ an
    # admissible completion of the truncated mass lets act_b overtake act_a — withheld (§21)
    src = _source(trunc_share=0.6)
    r = _run(monkeypatch, src)
    assert r.recommendation_status == "withheld"
    axes = r.provenance["structural_ensemble"]["core_uncertainty_axes"]
    codes = {w["code"] for w in axes["withheld_reasons"]}
    assert "winner_not_best_under_truncated_completions" in codes
    assert axes["eligibility_under_truncation"]["eligible"] is False
    assert axes["eligibility_under_truncation"]["leader"] == "act_a"
    assert axes["truncated_branch_weight"] == pytest.approx(0.6)


def test_eligible_when_margin_survives_truncation(monkeypatch):
    # t=0.05 ⇒ span 0.005 < completed margin 0.95*0.1=0.095 ⇒ the winner survives every
    # admissible truncated completion; existing gates untouched (still an action)
    src = _source(trunc_share=0.05)
    r = _run(monkeypatch, src)
    assert r.recommendation_status == "eligible"
    assert r.recommended == "act_a" and r.recommendation_kind == "action"
    axes = r.provenance["structural_ensemble"]["core_uncertainty_axes"]
    assert axes["eligibility_under_truncation"]["eligible"] is True
    assert axes["withheld_reasons"] == []


def test_withheld_when_recommendation_exists_under_single_family(monkeypatch):
    fam = {"model_family_monoculture": False,
           "configured_families": ["fam_a", "fam_b"],
           "families": [{"family_id": "fam_a"}, {"family_id": "fam_b"}],
           "assignments": [{"particle": i, "actor": "x", "family": "fam_a"} for i in range(4)]}
    r = _run(monkeypatch, _source(fam=fam))
    assert r.recommendation_status == "withheld"
    axes = r.provenance["structural_ensemble"]["core_uncertainty_axes"]
    codes = {w["code"] for w in axes["withheld_reasons"]}
    assert "recommendation_exists_under_single_family" in codes
    # control: both configured families actually exercised the run — no monoculture withholding
    fam_ok = dict(fam, assignments=[{"particle": 0, "actor": "x", "family": "fam_a"},
                                    {"particle": 1, "actor": "x", "family": "fam_b"}])
    r2 = _run(monkeypatch, _source(fam=fam_ok))
    assert r2.recommendation_status == "eligible"


def test_recommendation_stability_filled_on_source_result(monkeypatch):
    src = _source(trunc_share=0.05)
    assert src.structural_ensemble["recommendation_stability"] is None
    r = _run(monkeypatch, src)
    block = src.structural_ensemble["recommendation_stability"]
    assert set(block) == {"winner_by_model", "winner_stable_across_models", "truncated_weight",
                          "eligible_under_truncation", "withheld_reasons"}
    assert block["winner_by_model"] == {"m1": "act_a", "m2": "act_a"}
    assert block["winner_stable_across_models"] is True
    assert block["truncated_weight"] == pytest.approx(0.05)
    assert block["eligible_under_truncation"] is True
    assert block["withheld_reasons"] == []
    assert r.provenance["structural_ensemble"]["recommendation_stability_detail"] == block


def test_core_uncertainty_axes_reported_and_status_vocabulary(monkeypatch):
    src = _source(trunc_share=0.2)
    r = _run(monkeypatch, src)
    axes = r.provenance["structural_ensemble"]["core_uncertainty_axes"]
    # §31: per-model action performance exists and is pointed at, not duplicated
    assert "expected_utility_matrix" in axes["action_performance_by_model"][
        "expected_utility_matrix"]
    assert r.provenance["structural_ensemble"]["expected_utility_matrix"]
    assert "truncated_branch_weight" in axes and "model_family_monoculture" in axes
    assert axes["note"] == "truncated branch mass is unresolved simulation, not Monte Carlo error"
    assert r.recommendation_status in RECOMMENDATION_STATUSES
    assert "recommendation_status" in r.as_dict()
    assert DecisionResult(decision_id="x").recommendation_status == "not_requested"


# ================================================================ §33 personal-reaction route
def _decision_json(chosen="reply_now", act="act"):
    return json.dumps({
        "schema_version": "qualitative.actor.v1",
        "decision": {"act_or_wait": act, "chosen_action": chosen, "target": "you",
                     "timing": "immediate", "observability": "private",
                     "intended_effect": "respond"},
        "decision_summary": "my reaction"})


def _cognition_aware(decision_response):
    """A backend that answers every bounded-cognition stage properly and delegates the
    decision prompt to `decision_response(prompt) -> text`."""
    def call(prompt):
        if "ATTENTION process" in prompt:
            return json.dumps({"noticed": [{"obs_id": "stimulus", "why": "direct"}],
                               "missed": []})
        if "making private sense" in prompt:
            return json.dumps({"what_happened": "they sent me a message",
                               "why_it_matters": "it concerns our plans",
                               "perceived_sender_or_cause_intent": "genuine",
                               "activated_memories": [], "active_belief": "",
                               "perceived_opportunities": [], "perceived_threats": [],
                               "unresolved_ambiguity": ""})
        if "options even OCCUR" in prompt:
            return json.dumps({"options_recalled": [], "options_generated": [],
                               "options_screened_out": [], "shortlist": ["reply_later"]})
        return decision_response(prompt)
    return call


def test_personal_reaction_truncated_sample_honesty():
    from swm.world_model_v2.individual_reaction import simulate_individual_reaction
    from swm.world_model_v2.qualitative_actor import QualitativeConfig

    class _AlwaysFails:
        def __init__(self):
            self.calls = 0

        def __call__(self, prompt):
            self.calls += 1
            raise RuntimeError("provider down")

    llm = _AlwaysFails()
    art = simulate_individual_reaction(
        person_id="Dana", stimulus="are we still on for tonight?",
        context={"relationship": "close friend"}, llm=llm,
        n_hypotheses=2, samples_per_hypothesis=1, seed=0, as_of=T0,
        config=QualitativeConfig(llm=llm, llm_hypotheses=False, n_hypotheses=2,
                                 max_llm_calls=500))
    total = art["n_samples_total"]
    assert total == 2 and len(art["samples"]) == total          # never silently resampled
    assert art["n_samples_truncated"] + art["n_unread_by_horizon"] == total
    assert art["n_samples_truncated"] >= 1
    for t in art["truncated_samples"]:
        assert t["status"] == "truncated_provider_failure"      # first-class §20 status
        assert t["decision_source"] == "none_truncated"
    # NO numeric substitution anywhere: no sample carries a numeric decision source and the
    # aggregation saw zero excluded fallbacks (nothing numeric was ever produced)
    assert all("numeric" not in str(s.get("decision_source", "")) for s in art["samples"])
    assert art["n_excluded_numeric_fallbacks"] == 0
    # the distribution counts ONLY completed samples, with the explicit note + §21 bounds
    raw = art["raw_qualitative_simulation_distribution"]
    assert all(k == "unread_no_response_yet" for k in raw)      # no minted action mass
    assert art["truncation"]["truncated_share"] > 0
    assert "unresolved" in art["truncation"]["distribution_note"]
    assert art["truncation"]["honest_note"] == \
        "truncated branch mass is unresolved simulation, not Monte Carlo error"


def test_personal_reaction_mixed_truncation_bounds():
    from swm.world_model_v2.individual_reaction import simulate_individual_reaction
    from swm.world_model_v2.qualitative_actor import QualitativeConfig

    state = {"decision_calls": 0}

    def decide(prompt):
        state["decision_calls"] += 1
        if state["decision_calls"] <= 2:            # primary attempt + retry of sample 0
            raise RuntimeError("transient provider failure")
        return _decision_json("reply_now")
    llm = _cognition_aware(decide)
    art = simulate_individual_reaction(
        person_id="Dana", stimulus="ping", context={"relationship": "friend"}, llm=llm,
        n_hypotheses=3, samples_per_hypothesis=1, seed=0, as_of=T0,
        config=QualitativeConfig(llm=llm, llm_hypotheses=False, n_hypotheses=3, retries=1,
                                 max_llm_calls=500))
    assert art["n_samples_truncated"] == 1
    assert art["n_samples_completed"] == 2
    share = art["truncation"]["truncated_share"]
    assert share == pytest.approx(1 / 3, abs=1e-6)
    raw = art["raw_qualitative_simulation_distribution"]
    assert raw and abs(sum(raw.values()) - 1.0) < 1e-6          # completed mass only
    bounds = art["truncation"]["bounds_under_truncation"]
    for opt, b in bounds.items():
        # §21 semantics: the truncated share can swing ANY option fully for or against
        assert b["upper"] - b["lower"] == pytest.approx(min(share, 1.0 - b["lower"]), abs=1e-6)
        assert b["upper"] <= 1.0


def test_personal_reaction_nonresponse_breakdown():
    from swm.world_model_v2.bounded_cognition import NONRESPONSE_STATES
    from swm.world_model_v2.individual_reaction import simulate_individual_reaction
    from swm.world_model_v2.qualitative_actor import QualitativeConfig

    llm = _cognition_aware(lambda p: _decision_json("reply_later", act="wait"))
    art = simulate_individual_reaction(
        person_id="Dana", stimulus="can we move dinner?", context={"relationship": "friend"},
        llm=llm, n_hypotheses=2, samples_per_hypothesis=1, seed=0, as_of=T0,
        config=QualitativeConfig(llm=llm, llm_hypotheses=False, n_hypotheses=2,
                                 max_llm_calls=500))
    assert "nonresponse_breakdown" in art
    assert set(art["nonresponse_breakdown"]) <= set(NONRESPONSE_STATES)
    deferred = [s for s in art["samples"] if s["temporal_state"] == "read_but_deferred"]
    assert deferred and all(s["nonresponse_state"] == "considered_but_deferred"
                            for s in deferred)
    assert art["nonresponse_breakdown"].get("considered_but_deferred", 0) == len(deferred)

    llm2 = _cognition_aware(lambda p: _decision_json("ignore", act="act"))
    art2 = simulate_individual_reaction(
        person_id="Dana", stimulus="can we move dinner?", context={"relationship": "friend"},
        llm=llm2, n_hypotheses=1, samples_per_hypothesis=1, seed=0, as_of=T0,
        config=QualitativeConfig(llm=llm2, llm_hypotheses=False, n_hypotheses=1,
                                 max_llm_calls=500))
    chosen = [s for s in art2["samples"] if s.get("observable_response") == "ignore"]
    assert chosen and all(s["nonresponse_state"] == "no_response_chosen" for s in chosen)


def test_nonresponse_state_mapping_helper():
    from swm.world_model_v2.bounded_cognition import NONRESPONSE_STATES
    from swm.world_model_v2.individual_reaction import _nonresponse_state
    assert _nonresponse_state(temporal_state="unread_by_horizon") == "unread"
    assert _nonresponse_state(cognition={"observations_missed": [{"obs_id": "stimulus"}]}) \
        == "unread"                                           # attention missed the stimulus
    assert _nonresponse_state(cognition={
        "observations_noticed": ["stimulus"],
        "working_memory_active_sources": ["other_item"]}) == "noticed_but_deprioritized"
    assert _nonresponse_state(act_or_wait="wait") == "considered_but_deferred"
    assert _nonresponse_state(observable_response="ignore") == "no_response_chosen"
    assert _nonresponse_state(act_or_wait="do_nothing") == "no_response_chosen"
    assert _nonresponse_state(blocked=True) == "response_blocked_by_outside_circumstances"
    assert _nonresponse_state(act_or_wait="act", observable_response="reply_now") == ""
    for v in ("unread", "noticed_but_deprioritized", "considered_but_deferred",
              "no_response_chosen", "response_blocked_by_outside_circumstances"):
        assert v in NONRESPONSE_STATES


# ================================================================ §32 full realized text
def _mini_world_for(actor_id):
    from swm.world_model_v2.state import Entity, SimulationClock, WorldState
    w = WorldState("t", "b0", SimulationClock(0.0, 0.0))
    w.entities = {actor_id: Entity(actor_id)}
    return w


def test_full_message_content_reaches_cognition_availability(monkeypatch):
    import swm.world_model_v2.bounded_cognition as BC
    from swm.world_model_v2.phase4_policy import ActorView
    from swm.world_model_v2.qualitative_actor import (QualitativeActorPolicyRuntime,
                                                      QualitativeConfig,
                                                      QualitativeDecisionEngine)
    msg = ("I wanted to lay out the whole proposal so you can judge it on its actual terms. "
           * 30)[:1500]
    captured = {}
    orig = BC.attention_stage

    def spy(**kw):
        captured["available"] = kw.get("available")
        return orig(**kw)
    monkeypatch.setattr(BC, "attention_stage", spy)
    llm = _cognition_aware(lambda p: _decision_json())
    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=llm, llm_hypotheses=False)),
        mode="persistent_qualitative_llm_policy")
    view = ActorView(schema_version="v1", actor_id="dana", actor_role="person",
                     observed_time=0.0)
    world = _mini_world_for("dana")
    decision = {"situation": "a message arrived",
                "observation_bundle": [
                    {"iid": "m1", "channel": "direct_message", "source": "you",
                     "content": msg},
                    {"iid": "m2", "channel": "email", "source": "list", "content": msg}]}
    rt._run_bounded_cognition(world, view, None, decision, "dana", 1, [])
    avail = {a["obs_id"]: a for a in captured["available"]}
    # the attention stage INPUT carries the exact realized message whole (>1000 chars), while
    # ordinary items keep the ordinary summary width
    assert len(avail["m1"]["summary"]) == 1500 and avail["m1"]["summary"] == msg
    assert avail["m1"]["exact_realized_message"] is True
    assert len(avail["m2"]["summary"]) == 300


def test_full_message_survives_working_memory_and_decision_prompt():
    from swm.world_model_v2 import bounded_cognition as BC
    from swm.world_model_v2.phase4_policy import ActorView
    from swm.world_model_v2.qualitative_actor import (QualitativeConfig,
                                                      QualitativeDecisionEngine)
    msg = ("Here is the full text of the message with every argument spelled out in order. "
           * 25)[:1500]
    llm = _cognition_aware(lambda p: _decision_json())
    world = _mini_world_for("dana")
    cog = BC.run_cognition_pipeline(
        world=world, actor_id="dana", branch_id="b0", at=0.0,
        available=[{"obs_id": "stimulus", "channel": "direct_message", "source": "you",
                    "summary": msg, "interrupting": True, "exact_realized_message": True}],
        identity="dana, person", llm=llm, family_id="primary")
    wm_items = cog.decision_context()["working_memory"]
    stim = next(i for i in wm_items if i.get("source") == "stimulus")
    assert stim["exact"] is True and len(stim["content"]) == 1500   # not a 400-char cut
    engine = QualitativeDecisionEngine(QualitativeConfig(llm=llm, llm_hypotheses=False))
    view = ActorView(schema_version="v1", actor_id="dana", actor_role="person",
                     observed_time=0.0)
    prompt = engine.build_prompt(view, None, "the message", [], cognition=cog)
    assert msg in prompt                        # the decision reads the actual words (§32)


def test_exact_message_flag_rides_delivery_and_attention_bundle():
    import swm.world_model_v2.generated_world as GW
    from swm.world_model_v2.temporal_runtime import (collect_attention_bundle,
                                                     record_available_observation)
    # behavioral: availability → attention bundle preserves the full content and the flag
    msg = ("the realized message, byte for byte, as composed for this recipient. " * 25)[:1500]
    world = _mini_world_for("dana")
    record_available_observation(
        world, recipient="dana",
        item={"iid": "m1", "content": msg, "exact_realized_message": True,
              "semantic_event": {}},
        available_ts=0.0, channel="direct_message")
    bundle = collect_attention_bundle(world, actor_id="dana", now_ts=1.0)
    assert bundle and bundle[0]["content"] == msg and bundle[0]["exact_realized_message"]
    # seam enforcement: the kernel preserves realized text to the realizer's cap (2000, not a
    # 1200 clip), delivery carries the flag (and only representation=='summary' summarizes),
    # and the attention operator's invocation-bundle projection keeps the flag
    src = inspect.getsource(GW)
    assert 'op.get("exact_content", op.get("content", "")))[:2000]' in src
    assert "exact_realized_message" in inspect.getsource(
        GW.GeneratedObservationDeliveryOperator.run)
    att_src = inspect.getsource(GW.GeneratedAttentionOperator.run)
    assert "exact_realized_message" in att_src
    deliver_src = inspect.getsource(GW.GeneratedObservationDeliveryOperator.run)
    assert '== "summary"' in deliver_src        # summary-representation semantics intact


def test_compiler_flags_realized_message_ops():
    from swm.world_model_v2.phase13.scenario_actions.compiler import ScenarioActionCompiler
    src = inspect.getsource(ScenarioActionCompiler)
    assert "message_realizer" in src and "exact_realized_message" in src
    # the bounded-cognition availability seam honors the flag with the full realizer cap
    from swm.world_model_v2 import bounded_cognition as BC
    assert BC.EXACT_MESSAGE_CHARS == 2000
    assert BC._is_exact_message({"exact_realized_message": True})
    assert BC._is_exact_message({"channel": "direct_message"})
    assert not BC._is_exact_message({"channel": "email"})


# ================================================================ §32 judge separation
def test_judges_never_write_recipient_state():
    """The three separated reply-first judges (truth / language / blind outcome) may only
    select/realize the message. Nothing from the judge path writes recipient reaction, notice,
    or memory: reply_first is reachable ONLY through the message bridge, whose output is
    {text, label, gates} consumed solely as step wording + provenance."""
    root = Path(__file__).resolve().parents[1] / "swm"
    importers = []
    for p in root.rglob("*.py"):
        if "decision" in p.parts:                      # reply_first's own home package
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "import" in text and ("from swm.decision.reply_first" in text
                                 or "import reply_first" in text):
            importers.append(p.name)
    assert set(importers) <= {"message_bridge.py"}, \
        f"reply_first reachable outside the composition seam: {importers}"
    bridge_src = (root / "world_model_v2" / "phase13" / "scenario_actions"
                  / "message_bridge.py").read_text()
    for marker in (".set(", ".publish(", ".expose(", "latent_state", "entity("):
        assert marker not in bridge_src, f"message bridge writes state via {marker!r}"
    from swm.world_model_v2.phase13.scenario_actions.planner import GoalBackwardPlanner
    realize_src = inspect.getsource(GoalBackwardPlanner._realize_messages)
    # realizer output feeds ONLY the step's wording and provenance — never world/actor state
    assert 'step.exact_content = str(realized["text"])' in realize_src
    for marker in (".set(", ".publish(", ".expose(", "latent_state", "entity(",
                   "information."):
        assert marker not in realize_src, f"realize seam writes state via {marker!r}"
