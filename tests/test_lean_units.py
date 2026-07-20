"""Focused unit tests for the lean-adaptive components (§21 of the lean contract):
decision-context projection hits/misses, cohort collapse/expansion, cache safety and
single-flight, invalidation and duplicate suppression, prompt losslessness, prechecks,
consequence-cache behavior, structural reversal triggering, and progressive particles."""
from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace

import pytest

from swm.world_model_v2.lean_cohorts import (ActorCohortManifest, LeanCohortHypothesizer,
                                             behavioral_signature)
from swm.world_model_v2.lean_consequences import ConsequenceProgramCache
from swm.world_model_v2.lean_context import (DecisionContextDifference,
                                             DecisionDependencySpec, MANDATORY_COMPONENTS,
                                             DecisionRelevantContextBuilder, canonical_fact_id,
                                             context_rng_seed)
from swm.world_model_v2.lean_decision_cache import (ActorDecisionTemplate,
                                                    DecisionEquivalenceCache)
from swm.world_model_v2.lean_invalidation import (DecisionInvalidationPolicy,
                                                  ExecutionClassification,
                                                  PriorDecisionValidity, precheck,
                                                  should_invoke)
from swm.world_model_v2.lean_particles import LeanParticleTolerances, run_progressive_particles
from swm.world_model_v2.lean_prompts import (ActorContextSnapshot, ActorDecisionDelta,
                                             effective_actor_view)
from swm.world_model_v2.lean_structural import apply_reversal_verdict, reversal_verdict
from swm.world_model_v2.structural_contracts import StructuralModelEnsemble


# ------------------------------------------------------------------ fixtures
def _view(**over):
    base = dict(actor_id="tim", actor_role="chief executive", authority=["announce", "approve"],
                commitments=[{"statement": "ship it right", "binding": True,
                              "prohibits": ["premature launch"]}],
                stances=[{"commitment_level": "strong", "pathway": "launch"}],
                relationships=[{"other_actor": "jeff", "relation": "trusted deputy"}],
                institution_rules=[{"institution_id": "board", "kind": "decision_right",
                                    "params": {"who": "ceo"}}],
                resources={"budget": 1, "stage": 1},
                action_history=[{"action": "review_readiness"}],
                observed_events=[{"channel": "internal", "source": "eng",
                                  "content": "the demo build is stable", "iid": "ev-123"}],
                goals=["successful launch"], observed_time=1_760_000_000.0)
    base.update(over)
    return SimpleNamespace(**base)


def _state(**over):
    from swm.world_model_v2.qualitative_actor import QualitativeActorState
    base = dict(actor_id="tim", hypothesis_id="c0:ready",
                identity_and_role="tim, chief executive",
                current_private_beliefs=["the product is ready"],
                current_goals=["successful launch"])
    base.update(over)
    return QualitativeActorState(**base)


MENU = [{"key": "announce", "action_id": "a1", "line": "- announce: announce the product"},
        {"key": "wait", "action_id": "a2", "line": "- wait: hold and gather information"}]


def _builder(**over):
    kw = dict(prompt_version="lean.onecall.v1", backend_fingerprint="fp1",
              structural_frame="", public_facts=[])
    kw.update(over)
    return DecisionRelevantContextBuilder(**kw)


def _ctx(builder=None, view=None, state=None, menu=None, decision=None, day="2026-05-14",
         **kw):
    b = builder or _builder()
    return b.build(view=view or _view(), state=state if state is not None else _state(),
                   situation="decide on the announcement", menu=menu or MENU,
                   decision=decision or {"etype": "decision_opportunity"}, day=day, **kw)


# ------------------------------------------------------------------ §21 decision-context tests
def test_irrelevant_differences_do_not_create_misses():
    a = _ctx()
    # branch/particle/trace identity is never read; dict ordering canonicalized; duplicate fact
    # wording with the same canonical id deduplicates; event UUIDs dropped
    v = _view(observed_events=[{"channel": "internal", "source": "eng",
                                "content": "the   demo build is stable", "iid": "ev-999",
                                "event_uuid": "deadbeef"},
                               {"channel": "internal", "source": "eng",
                                "content": "the demo build is stable", "iid": "ev-777"}],
              institution_rules=[{"kind": "decision_right", "institution_id": "board",
                                  "params": {"who": "ceo"}}])
    v.branch_id, v.particle_index, v.trace_id = "b017", 17, "trace-xyz"
    b = _ctx(view=v)
    diff = DecisionContextDifference.between(a, b)
    assert diff.equal, f"irrelevant differences created a miss: {diff.differing_components}"
    assert a.signature() == b.signature()


def test_subday_timestamp_differences_do_not_create_misses():
    a = _ctx(view=_view(observed_time=1_760_000_000.0))
    b = _ctx(view=_view(observed_time=1_760_000_000.0 + 3600 * 5))   # same day, 5h later
    assert a.signature() == b.signature()


def test_material_differences_create_misses():
    base = _ctx()
    cases = {
        "different_private_belief": _ctx(state=_state(
            current_private_beliefs=["a serious blocker remains"])),
        "different_noticed_fact": _ctx(view=_view(observed_events=[
            {"channel": "internal", "source": "eng", "content": "the demo build crashes"}])),
        "different_deadline_day": _ctx(day="2026-06-10"),
        "different_authority": _ctx(view=_view(authority=["observe"])),
        "different_feasible_actions": _ctx(menu=[MENU[1]]),
        "different_target_or_content": _ctx(menu=[
            {"key": "announce", "action_id": "a1",
             "line": "- announce: announce the DELAY"}, MENU[1]]),
        "different_commitment": _ctx(view=_view(commitments=[])),
        "different_relationship": _ctx(view=_view(relationships=[
            {"other_actor": "jeff", "relation": "distrusted rival"}])),
        "different_resource": _ctx(view=_view(resources={"budget": 1})),
        "different_institution_rule": _ctx(view=_view(institution_rules=[
            {"institution_id": "board", "kind": "quorum", "params": {"n": 5}}])),
        "different_invalidation_condition": _ctx(prior_decision={
            "action": "announce", "revisit": {"condition": {"etype": "board_meeting"}}}),
        "different_structural_assumption": _ctx(builder=_builder(structural_frame="alt frame")),
        "different_replicate_index": _ctx(replicate_index=1),
        "different_fact_credibility": _ctx(view=_view(observed_events=[
            {"channel": "internal", "source": "eng",
             "content": "the demo build is stable [credibility: contested]"}])),
    }
    for name, other in cases.items():
        assert base.signature() != other.signature(), f"{name} MUST create a miss"


def test_dependency_spec_cannot_narrow_below_the_floor():
    spec = DecisionDependencySpec(components=("actor_identity",), proposed_by="test").validate()
    assert set(MANDATORY_COMPONENTS) <= set(spec.components)
    assert spec.narrowed, "the attempted narrowing must be recorded"


def test_canonical_fact_id_stable_under_wording_noise_only():
    a = canonical_fact_id("The board  meets\n tomorrow", "internal")
    b = canonical_fact_id("The board meets tomorrow", "internal")
    c = canonical_fact_id("The board meets next week", "internal")
    assert a == b and a != c


# ------------------------------------------------------------------ §21 cohort tests
ROWS = [
    {"hypothesis_label": "ready", "current_private_beliefs": ["the product is ready"],
     "current_goals": ["launch now"]},
    {"hypothesis_label": "ready_reworded",
     "current_private_beliefs": ["ready the product is"],      # paraphrase (token multiset)
     "current_goals": ["now launch"]},
    {"hypothesis_label": "blocker", "current_private_beliefs": ["a serious blocker remains"],
     "current_goals": ["delay"]},
]


def test_paraphrases_collapse_and_material_beliefs_stay_separate():
    sigs = [behavioral_signature(r) for r in ROWS]
    assert sigs[0] == sigs[1] and sigs[0] != sigs[2]
    cohorts, collapsed = LeanCohortHypothesizer._collapse(ROWS)
    assert len(cohorts) == 2 and collapsed == 1
    assert cohorts[0].merged_labels == ["ready_reworded"]


class _CohortLLM:
    def __init__(self, hyps, critic):
        self.hyps, self.critic, self.calls = hyps, critic, []

    def __call__(self, prompt):
        self.calls.append(prompt[:60])
        if "COHORT CRITIC" in prompt:
            return json.dumps(self.critic)
        return json.dumps(self.hyps)


def _hyp_view():
    return _view()


def test_reversal_relevant_omission_triggers_expansion():
    llm = _CohortLLM(ROWS, {"paraphrase_pairs": [],
                            "missing_states": [{"label": "awaiting_signal",
                                                "belief_core": "waiting on one internal test",
                                                "could_reverse_decision": True,
                                                "evidence_support": "unresolved test reports",
                                                "distinguishing_observation": "test outcome"}]})
    h = LeanCohortHypothesizer(llm, k=3, ceiling=6, manifest=ActorCohortManifest())
    rows = h.hypothesis_set(_hyp_view())
    cs = h.manifest.sets["tim"]
    assert cs.expanded == 1 and len(cs.cohorts) == 3            # 2 after collapse + 1 expansion
    assert any(c.reversal_relevant for c in cs.cohorts)
    assert len(rows) == 3


def test_ceiling_hit_marks_under_modeled_never_silently_collapses():
    llm = _CohortLLM(ROWS, {"paraphrase_pairs": [],
                            "missing_states": [{"label": "x", "belief_core": "materially new",
                                                "could_reverse_forecast": True}]})
    h = LeanCohortHypothesizer(llm, k=3, ceiling=2, manifest=ActorCohortManifest())
    h.hypothesis_set(_hyp_view())
    cs = h.manifest.sets["tim"]
    assert cs.under_modeled and "ceiling" in cs.under_modeled_reason
    assert len(cs.cohorts) == 2


def test_cohort_layer_invents_no_probabilities():
    llm = _CohortLLM(ROWS, {"paraphrase_pairs": [], "missing_states": []})
    h = LeanCohortHypothesizer(llm, k=3, ceiling=6, manifest=ActorCohortManifest())
    h.hypothesis_set(_hyp_view())
    d = h.manifest.as_dict()
    assert "weight" not in json.dumps(d).lower().replace("weights", "")  # no cohort weights


# ------------------------------------------------------------------ §21 cache safety tests
_ONE_CALL_RESPONSE = json.dumps({
    "attention": {"noticed": [{"obs_id": "a0", "why": "core"}], "ignored": []},
    "interpretation": {"what_happened": "the build is stable", "why_it_matters": "launch"},
    "considered_actions": ["announce"], "screened_out": [],
    "decision": {"chosen_action": "announce", "act_or_wait": "act", "target": "",
                 "timing": "immediate", "observability": "public",
                 "intended_effect": "announce the product", "linked_actions": [],
                 "revisit": {}},
    "decision_summary": "announce now",
    "actor_state_update": {"current_private_beliefs": ["we are committed"]},
    "reconsideration_conditions": ["a serious regression appears"]})


def _template(ctx, response=_ONE_CALL_RESPONSE):
    from swm.world_model_v2.qualitative_actor import QualitativeDecision
    return ActorDecisionTemplate(
        context_hash=ctx.signature(), actor_id="tim", cohort_id="c0", prompt_hash="ph",
        response_hash="rh", response=response,
        qd_snapshot={f: getattr(QualitativeDecision(actor_id="tim", chosen_action="announce"),
                                f) for f in QualitativeDecision.__dataclass_fields__},
        model_fingerprint="fp1", prompt_version="lean.onecall.v1", replicate_index=0,
        source_branch="b000", context=ctx.as_dict())


def test_one_template_serves_equivalent_branches_and_states_stay_independent():
    cache = DecisionEquivalenceCache()
    ctx = _ctx()
    key = cache.key_for(ctx)
    assert cache.get(key) is None                       # miss
    cache.store(key, _template(ctx))
    t = cache.get(key)
    assert t is not None                                # hit
    qd_a, cert_a = cache.reuse(t, receiving_branch="b001", revalidation={"ok": True})
    qd_b, _cert_b = cache.reuse(t, receiving_branch="b002", revalidation={"ok": True})
    qd_a["actor_state_update"] = {"mutated": "by branch a"}
    assert qd_b.get("actor_state_update") != {"mutated": "by branch a"}, \
        "reused decisions must be independent deep copies"
    assert t.qd_snapshot.get("actor_state_update") != {"mutated": "by branch a"}
    m = cache.manifest()
    assert m["reuses"] == 2 and m["unique_decision_contexts"] == 1
    assert cert_a.receiving_branch == "b001" and cert_a.source_branch == "b000"


def test_failures_are_never_cached():
    cache = DecisionEquivalenceCache()
    cache.record_failure()
    assert cache.manifest()["failures_not_cached"] == 1
    assert cache.manifest()["unique_decision_contexts"] == 0


def test_single_flight_one_leader_rest_wait():
    """Concurrent identical contexts make exactly ONE provider call. Mirrors the controller's
    protocol: check the cache, then begin single-flight; a late arrival that becomes 'leader'
    after the first leader finished finds the template in the cache and never calls."""
    cache = DecisionEquivalenceCache()
    ctx = _ctx()
    key = cache.key_for(ctx)
    provider_calls = []
    barrier = threading.Barrier(6)

    def worker():
        barrier.wait()
        if cache.peek(key):
            return
        role, ev = cache.single_flight.begin(key)
        if role == "leader":
            if not cache.peek(key):                      # still uncached → the one real call
                provider_calls.append(1)
                cache.store(key, _template(ctx))
            cache.single_flight.finish(key)
        else:
            ev.wait(timeout=10)
            assert cache.peek(key), "waiter released before the leader stored the template"

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(provider_calls) == 1, f"expected 1 provider call, got {len(provider_calls)}"


def test_single_flight_failure_releases_waiters_for_retry():
    cache = DecisionEquivalenceCache()
    key = "k1"
    role, _ = cache.single_flight.begin(key)
    assert role == "leader"
    got = {}

    def waiter():
        r, ev = cache.single_flight.begin(key)
        got["role"] = r
        if r == "waiter":
            ev.wait(timeout=10)
            got["after"] = cache.single_flight.begin(key)[0]   # may retry as new leader

    t = threading.Thread(target=waiter)
    t.start()
    cache.single_flight.finish(key)                    # leader FAILED (nothing stored)
    t.join()
    assert got["role"] == "waiter" and got["after"] == "leader"


def test_cross_run_cache_is_disabled_by_default():
    ctx = _ctx()
    c1 = DecisionEquivalenceCache()
    c1.store(c1.key_for(ctx), _template(ctx))
    c2 = DecisionEquivalenceCache()                     # a new run's cache
    assert c2.get(c2.key_for(ctx)) is None
    assert "run-scoped" in c1.manifest()["cross_run_persistence"]


def test_explain_equivalence_is_auditable():
    cache = DecisionEquivalenceCache()
    ctx = _ctx()
    cache.store(cache.key_for(ctx), _template(ctx))
    t = cache.get(cache.key_for(ctx))
    cache.reuse(t, receiving_branch="b007", revalidation={"ok": True, "feasibility": "re-run"})
    text = cache.explain_equivalence()
    for needle in ("DECISION REUSE", "matched components", "ignored differences",
                   "branch_id/particle_index", "revalidation", "b007"):
        assert needle in text, f"explain_equivalence missing {needle!r}"


# ------------------------------------------------------------------ §21 invalidation tests
def _prior(ctx, action="announce"):
    return PriorDecisionValidity(context_signature=ctx.signature(), chosen_action=action,
                                 act_or_wait="act", decided_day=ctx.day,
                                 processed_fact_ids=[o["fact_id"] for o in ctx.observations])


class _PriorState:
    def __init__(self, prior):
        self._lean_prior = prior.as_dict()


def test_duplicate_notifications_do_not_trigger_new_calls():
    ctx = _ctx(decision={"etype": "decision_opportunity", "observation_bundle": [
        {"channel": "internal", "source": "eng", "content": "the demo build is stable"}]})
    prior = _prior(ctx)
    prior.processed_fact_ids = [f["fact_id"] for f in ctx.trigger["payload_facts"]]
    v = precheck(ctx=ctx, state=_PriorState(prior), view=_view(), menu=MENU,
                 decision={}, policy=DecisionInvalidationPolicy(), prior_ctx=ctx)
    assert v.skip and v.reason == "duplicate_notification"
    assert v.classification.classification == "fully_mechanical"


def test_meaningful_new_evidence_triggers_a_call():
    ctx0 = _ctx()
    ctx1 = _ctx(view=_view(observed_events=[{"channel": "internal", "source": "eng",
                                             "content": "a NEW regression was found"}]))
    v = precheck(ctx=ctx1, state=_PriorState(_prior(ctx0)), view=_view(), menu=MENU,
                 decision={}, policy=DecisionInvalidationPolicy(), prior_ctx=ctx0)
    assert not v.skip
    assert v.classification.classification == "human_discretion_required"


def test_entering_a_deadline_window_triggers_a_call():
    ctx0 = _ctx(day="2026-05-14")
    ctx1 = _ctx(day="2026-06-09")                       # WWDC week begins — day changed
    v = precheck(ctx=ctx1, state=_PriorState(_prior(ctx0)), view=_view(), menu=MENU,
                 decision={}, policy=DecisionInvalidationPolicy(), prior_ctx=ctx0)
    assert not v.skip


def test_unchanged_active_decision_remains_valid_without_periodic_reconsideration():
    ctx = _ctx()
    v = precheck(ctx=ctx, state=_PriorState(_prior(ctx)), view=_view(), menu=MENU,
                 decision={}, policy=DecisionInvalidationPolicy(), prior_ctx=ctx)
    assert v.skip and v.reason == "unchanged_prior_decision"
    # no time-based field exists anywhere in the policy: only material conditions reopen it
    import inspect
    src = inspect.getsource(DecisionInvalidationPolicy)
    assert "interval" not in src and "periodic" not in src.lower().replace(
        "no periodic reconsideration", "")


def test_actor_named_revisit_condition_reopens_the_decision():
    ctx0 = _ctx()
    prior = _prior(ctx0)
    prior.revisit = {"condition": {"etype": "board_meeting"}}
    ctx1 = _ctx(decision={"etype": "board_meeting"})
    mc = DecisionInvalidationPolicy().material_change(prior, ctx0.signature(), ctx1, ctx0)
    assert mc.changed and "revisit condition" in mc.named_condition_met


# ------------------------------------------------------------------ §21 mechanical prechecks
def test_actor_without_authority_or_feasible_action_is_skipped():
    ctx = _ctx(view=_view(authority=[], resources={}),
               menu=[{"key": "wait", "action_id": "a2", "line": "- wait: do nothing"}])
    v = precheck(ctx=ctx, state=None, view=_view(authority=[]), menu=MENU, decision={},
                 policy=DecisionInvalidationPolicy())
    assert v.skip and v.reason == "actor_lacked_authority"


def test_unobserved_event_does_not_invoke_actor():
    ev = SimpleNamespace(participants=["someone_else"], payload={},
                         visibility="participants")
    invoke, reason = should_invoke(None, ev, "tim")
    assert not invoke and reason == "no_observed_event"


def test_participant_and_public_events_still_invoke_human_discretion():
    ev = SimpleNamespace(participants=["tim"], payload={}, visibility="participants")
    assert should_invoke(None, ev, "tim")[0]
    ev2 = SimpleNamespace(participants=["someone"], payload={}, visibility="public")
    assert should_invoke(None, ev2, "tim")[0]


def test_empty_availability_skips_but_never_invents_a_vote():
    ctx = _ctx(view=_view(observed_events=[]),
               decision={"etype": "decision_opportunity"})
    ctx.working_memory = []
    v = precheck(ctx=ctx, state=None, view=_view(observed_events=[]), menu=MENU, decision={},
                 policy=DecisionInvalidationPolicy())
    assert v.skip and v.reason == "no_observed_event"
    from swm.world_model_v2.lean_controller import LeanActorController
    qd = LeanActorController()._noop_decision("tim", v, ctx)
    assert qd.act_or_wait == "wait" and qd.chosen_action == "wait" and qd.llm_calls == 0


def test_execution_classification_names_are_closed():
    with pytest.raises(ValueError):
        ExecutionClassification(classification="shortcut_forecaster")


# ------------------------------------------------------------------ §21 prompt tests
def test_stable_context_is_not_resent_and_delta_reconstructs_the_boundary():
    snap = ActorContextSnapshot.build(view=_view(), state=_state(),
                                      public_facts_lines=["- WWDC begins 2026-06-09"])
    delta = ActorDecisionDelta.build(
        day="2026-05-14", situation="decide on the announcement",
        observations=[{"obs_id": "a0", "channel": "internal", "source": "eng",
                       "content": "the demo build is stable"}],
        working_memory=[{"kind": "observation", "content": "the demo build is stable"}],
        retrieved=[{"content": "last year's launch slipped a week"}],
        changed_state_rows=["current_private_beliefs: a blocker was fixed yesterday"],
        resources=["budget", "stage"], action_history=["review_readiness"],
        menu_lines=[m["line"] for m in MENU])
    full = effective_actor_view(snap, delta)
    for needle in ("tim", "chief executive", "announce", "the product is ready",
                   "the demo build is stable", "WWDC begins", "decision_right",
                   "successful launch", "budget", "review_readiness",
                   "a blocker was fixed yesterday", "last year's launch",
                   "INFORMATION BOUNDARY"):
        assert needle in full, f"losslessness: {needle!r} missing from the effective view"
    snap2 = ActorContextSnapshot.build(view=_view(), state=_state(),
                                       public_facts_lines=["- WWDC begins 2026-06-09"])
    assert snap.content_hash == snap2.content_hash, "stable prefix must be byte-stable"
    assert snap.rendered not in delta.rendered, "the delta must not resend the stable context"


# ------------------------------------------------------------------ §21 consequence cache
def test_identical_actions_reuse_compilation_and_different_content_misses():
    backend_calls = []

    def backend(prompt):
        backend_calls.append(prompt)
        return '{"actor_controlled_operations": []}'

    cache = ConsequenceProgramCache(backend, fingerprint="fp1")
    p1 = "COMPILE actor=tim action=announce content='we ship in june' target=press"
    assert cache(p1) and cache(p1)
    assert len(backend_calls) == 1 and cache.reuses == 1
    cache("COMPILE actor=tim action=announce content='we ship in JULY' target=press")
    cache("COMPILE actor=tim action=announce content='we ship in june' target=STAFF")
    assert len(backend_calls) == 3, "different content/target must force fresh compiles"
    assert cache.manifest()["distinct_programs"] == 3
    assert "rerun per receiving branch" in cache.manifest()["reuse_design"]


def test_consequence_cache_never_stores_failures():
    def bad(prompt):
        raise RuntimeError("provider down")
    cache = ConsequenceProgramCache(bad, fingerprint="fp1")
    with pytest.raises(RuntimeError):
        cache("p")
    assert cache.failures_not_cached == 1 and cache.manifest()["distinct_programs"] == 0

    def empty(prompt):
        return "   "
    cache2 = ConsequenceProgramCache(empty, fingerprint="fp1")
    cache2("p")
    assert cache2.failures_not_cached == 1 and cache2.manifest()["distinct_programs"] == 0


# ------------------------------------------------------------------ §21 structural tests
def _ens():
    return StructuralModelEnsemble(question="q", as_of="2026-05-14", horizon="2026-06-16")


NO_REVERSAL = {"materially_different_model_plausible": False,
               "supported_or_left_open_by_evidence": False,
               "could_reverse_binary_forecast": False,
               "could_reverse_recommended_action": False, "causally_executable": False,
               "prose_variation_only": False, "additional_credible_alternatives": []}
REVERSAL = {"materially_different_model_plausible": True,
            "supported_or_left_open_by_evidence": True,
            "could_reverse_binary_forecast": True, "could_reverse_recommended_action": False,
            "causally_executable": True, "prose_variation_only": False,
            "differing_assumption": "the board can veto", "reversal_causal_chain": "veto → no",
            "challenger_thesis": "the board vetoes the launch",
            "additional_credible_alternatives": []}


def test_no_reversal_capable_challenger_means_one_certified_model():
    ens = _ens()
    rec = apply_reversal_verdict(ens, dict(NO_REVERSAL), llm=lambda p: "{}",
                                 as_of=ens.as_of, horizon=ens.horizon)
    assert not rec["challenger_generated"]
    assert ens.convergence_certificate is not None
    assert ens.convergence_certificate["kind"] == "lean_reversal_critic_no_reversal"
    assert not ens.structurally_underidentified


def test_credible_reversal_capable_challenger_means_two_models():
    assert reversal_verdict(REVERSAL)
    ens = _ens()
    gen_calls = []

    def llm(prompt):
        gen_calls.append(prompt)
        return json.dumps({"causal_thesis": "the board vetoes the launch",
                           "decisive_actors": ["board"], "decisive_institutions": ["board"],
                           "decisive_mechanisms": ["veto"], "world_boundary": "b",
                           "falsifiers": [], "candidate_omissions": [],
                           "required_evidence": [], "intervention_propagation": "via veto"})
    rec = apply_reversal_verdict(ens, dict(REVERSAL), llm=llm, as_of=ens.as_of,
                                 horizon=ens.horizon)
    assert rec["challenger_generated"] and len(ens.candidates) == 1
    assert ens.candidates[0].generation_role == "lean_reversal_challenger"
    assert "veto" in ens.candidates[0].provenance["lean_reversal"]["reversal_causal_chain"]


def test_unresolved_extra_alternatives_trigger_underidentification():
    ens = _ens()
    v = dict(NO_REVERSAL, additional_credible_alternatives=["a supply-chain-driven model"])
    rec = apply_reversal_verdict(ens, v, llm=lambda p: "{}", as_of=ens.as_of,
                                 horizon=ens.horizon)
    assert rec["underidentified"] and ens.structurally_underidentified
    assert ens.convergence_certificate is None
    assert ens.unresolved_alternatives


def test_dead_critic_never_certifies_a_single_model():
    ens = _ens()
    rec = apply_reversal_verdict(ens, {"error": "provider down"}, llm=lambda p: "{}",
                                 as_of=ens.as_of, horizon=ens.horizon)
    assert rec["underidentified"] and ens.convergence_certificate is None


# ------------------------------------------------------------------ §21 adaptive particles
class _FakeRun:
    """Index-keyed fake: branches are integers; the projection script drives stability."""

    def __init__(self, dist_script):
        self.dist_script = dist_script                  # n -> (dist, unresolved, truncated)

    def run_particle_range(self, *, seed=0, n_total=None, start=0, stop=None,
                           particle_scope=None):
        return list(range(start, stop))

    def project(self, branches):
        dist, unresolved, truncated = self.dist_script(len(branches))
        return {"distribution": dist, "unresolved_share": unresolved,
                "truncated_share": truncated}


def _handle(n_full, script):
    return {"n_particles": n_full, "run": _FakeRun(script)}


TOL = LeanParticleTolerances(batch_size=8, drift_tolerance=0.04, ci_halfwidth_max=0.35,
                             require_batches_stable=2)


def test_progressive_prefixes_match_full_run_prefixes():
    h = _handle(32, lambda n: ({"yes": 0.8, "no": 0.2}, 0.0, 0.0))
    branches, rec = run_progressive_particles(h, seed=0, tolerances=TOL)
    assert branches == list(range(len(branches))), "progressive branches must be the exact " \
                                                   "index prefix of a full run"


def test_stable_questions_stop_early_and_unstable_expand_to_full():
    stable = _handle(64, lambda n: ({"yes": 0.85, "no": 0.15}, 0.0, 0.0))
    b1, r1 = run_progressive_particles(stable, seed=0, tolerances=TOL)
    assert r1.stopped_early and r1.n_executed < 64 and r1.as_dict()["particles_avoided"] > 0

    drifting = _handle(64, lambda n: ({"yes": 0.5 + (0.2 if (n // 8) % 2 else -0.2)}, 0.0, 0.0))
    b2, r2 = run_progressive_particles(drifting, seed=0, tolerances=TOL)
    assert not r2.stopped_early and r2.n_executed == 64


def test_near_half_questions_do_not_falsely_stop():
    # p=0.52 with small n: the interval crosses 0.5 — stopping must be blocked until the
    # full budget resolves it
    h = _handle(48, lambda n: ({"yes": 0.52, "no": 0.48}, 0.0, 0.0))
    b, rec = run_progressive_particles(h, seed=0, tolerances=TOL)
    assert not rec.stopped_early and rec.n_executed == 48
    assert any(not c["conditions"]["side_of_half_stable"] for c in rec.checkpoints)


def test_structural_disagreement_and_reversal_prevent_early_stopping():
    h = _handle(40, lambda n: ({"yes": 0.9, "no": 0.1}, 0.0, 0.0))
    b, rec = run_progressive_particles(h, seed=0, tolerances=TOL,
                                       structural_disagreement=True)
    assert not rec.stopped_early and rec.n_executed == 40
    assert "material_structural_disagreement" in rec.forced_full_reasons
    h2 = _handle(40, lambda n: ({"yes": 0.9, "no": 0.1}, 0.0, 0.0))
    b2, rec2 = run_progressive_particles(h2, seed=0, tolerances=TOL,
                                         reversal_outstanding=True)
    assert not rec2.stopped_early
    assert "reversal_capable_hypothesis_outstanding" in rec2.forced_full_reasons


def test_high_unresolved_mass_prevents_early_stopping():
    h = _handle(40, lambda n: ({"yes": 0.9, "no": 0.1}, 0.6, 0.0))
    b, rec = run_progressive_particles(h, seed=0, tolerances=TOL)
    assert not rec.stopped_early and rec.n_executed == 40


def test_tolerances_are_explicit_and_recorded_compute_controls():
    h = _handle(16, lambda n: ({"yes": 0.9}, 0.0, 0.0))
    _, rec = run_progressive_particles(h, seed=0, tolerances=TOL)
    d = rec.as_dict()
    assert d["tolerances"] == TOL.as_dict()
    assert d["version"].startswith("lean.particles")


# ------------------------------------------------------------------ replicate policy
def test_replicate_index_enters_the_signature_and_default_is_one():
    from swm.world_model_v2.lean_controller import LeanActorConfig, LeanActorController
    assert LeanActorConfig().behavioral_replicates_per_decision_context == 1
    a, b = _ctx(replicate_index=0), _ctx(replicate_index=1)
    assert a.signature() != b.signature()
    c = LeanActorController()
    c.config.behavioral_replicates_per_decision_context = 2
    w0 = SimpleNamespace(particle_index=4)
    w1 = SimpleNamespace(particle_index=5)
    assert c._replicate_for(w0) == 0 and c._replicate_for(w1) == 1  # deterministic assignment
    assert context_rng_seed("sig", replicate_index=0) != context_rng_seed(
        "sig", replicate_index=1)
