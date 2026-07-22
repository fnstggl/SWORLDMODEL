"""§37 cross-domain validation fixtures as OFFLINE deterministic scenario tests.

Scenario names are RANDOMIZED per test run (seeded from os.urandom), so production code cannot
pass by memorizing example strings — every assertion is structural (which stage recorded what,
at which timestamp, through which mechanism), never name-based. The scripted fake LLMs key on
the randomly generated tokens through closures, so behavior stays fully deterministic within a
run while the surface vocabulary changes on every run.

Implemented: 37.1 omitted-actor promotion, 37.2 outside event entering through a typed
mechanism and changing availability, 37.5 nonhuman bottleneck, 37.6 attention difference,
37.7 memory difference, 37.8 contradictory-belief-driven behavior, 37.9 limited search missing
the global best, 37.10 model-family disagreement, 37.11 actor-budget exhaustion stopping a
branch, 37.12 missing-mechanism suppression + under_modeled_nonhuman_mechanism classification.

SKIPPED honestly: 37.3 and 37.4. The §37 fixture spec is not committed to this repository, and
the surviving numbered list jumps from the outside-world entry fixture (37.2) to the nonhuman
bottleneck (37.5); the missing two are the live-generation fixtures (boundary generation +
critics, and outside-world family identification, against a real scenario corpus), which
require an actual LLM backend and are exercised by experiments/core_arch_forensics.py — they
are not implementable as offline deterministic tests without faking the very generation step
they exist to validate.
"""
import json
import os
import random
from types import SimpleNamespace

import pytest

import swm.world_model_v2.generated_world  # noqa: F401 — registers ctrl_* event types
from swm.world_model_v2.boundary_monitor import (
    OUTSIDE_EVENT_TYPE, BoundaryMonitorOperator, OutsideWorldEntryOperator,
    schedule_outside_arrivals,
)
from swm.world_model_v2.bounded_cognition import (
    ActorMemoryState, BeliefRecord, EpisodicMemory, WorkingMemoryItem, WorkingMemoryState,
    memory_retrieval_stage, run_cognition_pipeline, store_memory,
)
from swm.world_model_v2.causal_boundary import MechanismInstance, refresh_attempt_status
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.fallback import GenericOutcomeOperator
from swm.world_model_v2.generated_world import GeneratedAttentionOperator
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.model_families import FamilyPool, ModelFamily
from swm.world_model_v2.outside_world import (
    ArrivalModel, ExternalEventFamily, OutsideWorldProcess, validate_entry,
)
from swm.world_model_v2.phase4_execution import ProductionActorPolicyOperator
from swm.world_model_v2.phase4_policy import ActorViewBuilder
from swm.world_model_v2.qualitative_actor import (
    QualitativeActorPolicyRuntime, QualitativeActorState, QualitativeConfig,
    QualitativeDecisionEngine, store_actor_state,
)
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.temporal_runtime import aggregate_temporal_stats, get_stats
from swm.world_model_v2.world_boundary import BoundaryComponent, WorldBoundary

T0 = 1_700_000_000.0
DAY = 86400.0

# ---------------------------------------------------------------------- randomized naming
_RNG = random.Random(int.from_bytes(os.urandom(8), "big"))
_SYLLABLES = ("ka", "lo", "mi", "ver", "dan", "tor", "sel", "na", "rim", "bo", "tu", "zen",
              "gar", "pel", "os", "vin", "dra", "mel")
_used_names = set()


def rand_name(prefix=""):
    """A per-run random snake_case identifier — never a memorizable fixture constant."""
    while True:
        core = "".join(_RNG.choice(_SYLLABLES) for _ in range(3)) + str(_RNG.randrange(10, 99))
        name = f"{prefix}_{core}" if prefix else core
        if name not in _used_names:
            _used_names.add(name)
            return name


def make_world(*actors, now=T0, info=True, branch="b000"):
    w = WorldState(rand_name("world"), branch, SimulationClock(now=now, as_of=now),
                   information=InformationLedger() if info else None)
    for a in actors:
        e = Entity(a)
        e.set("roles", F(["person"], status="observed"))
        e.set("past_actions", F([], status="observed"))
        w.entities[a] = e
    return w


def _strict(monkeypatch):
    monkeypatch.delenv("SWM_ALLOW_NUMERIC_BASELINE", raising=False)
    monkeypatch.delenv("SWM_ALLOW_GENERIC_PRIOR", raising=False)


def _decision_payload(chosen):
    return json.dumps({
        "schema_version": "qualitative.actor.v1",
        "decision": {"act_or_wait": "act", "chosen_action": chosen, "target": "",
                     "timing": "immediate", "observability": "public",
                     "intended_effect": "respond to the development"},
        "decision_summary": f"I go with {chosen}"})


# =====================================================================================
# 37.1 — omitted-actor promotion: a boundary rule matches a LATER event; the actor is
#        promoted into the world at exactly that event's timestamp
# =====================================================================================
def test_fixture_37_1_omitted_actor_promoted_at_exact_event_time():
    insider = rand_name("actor")
    outsider = rand_name("actor")
    trigger_phrase = f"{outsider} files a formal objection"
    w = make_world(insider)
    boundary = WorldBoundary(boundary_id=rand_name("wb"), structural_model_id=rand_name("m"),
                             question="will the deal close")
    boundary.components.append(BoundaryComponent(
        component_id=rand_name("c"), kind="individual_actor", name=outsider,
        representation="external_process", reason="not yet causally active",
        promotion_trigger=trigger_phrase))
    boundary.promotion_rules = [{"component": outsider, "trigger": trigger_phrase,
                                 "action": "promote_to_individual"}]
    boundary.rederive_views()
    assert outsider not in w.entities
    assert boundary.dynamic_promotions == []
    event_ts = T0 + 3 * DAY
    q = EventQueue(horizon_ts=T0 + 10 * DAY)
    q.schedule(Event(ts=event_ts, etype="ctrl_semantic_event", participants=[],
                     payload={"semantic_event": {
                         "event_id": rand_name("ev"),
                         "exact_content": f"registry note: {trigger_phrase} against the deal"}}))
    report = {}
    branch = RolloutEngine(operators=[BoundaryMonitorOperator(boundary, report=report)]) \
        .run_branch(w, q, seed=0)
    # promoted at the EXACT timestamp of the matching event — not at compile, not at horizon
    assert outsider in w.entities
    rec = w.boundary_promotions[0]
    assert rec["at"] == event_ts and rec["component"] == outsider
    assert boundary.dynamic_promotions[0]["at"] == event_ts
    assert boundary.component(outsider).promoted_at == event_ts
    assert boundary.component(outsider).representation == "individual"
    assert report["boundary_promotions"][0]["trigger"].startswith("event matched promotion rule")
    assert w.clock.now == event_ts
    # the branch itself completed normally — promotion is expansion, not failure
    assert branch.temporal_stats.branch_halted is False
    assert branch.temporal_stats.temporally_truncated is False
    # the promotion delta is machine-readable in the branch log
    promo_deltas = [d for d in branch.log if any(
        str(c.get("path", "")).endswith("latent_state[boundary_promotion_record]")
        for c in d.changes)]
    assert promo_deltas and promo_deltas[0].at == event_ts


# =====================================================================================
# 37.2 — an outside-world event enters through its typed mechanism and CHANGES an
#        actor's availability (delivery → availability → attention), never the answer
# =====================================================================================
def test_fixture_37_2_outside_arrival_enters_and_changes_availability():
    actor = rand_name("actor")
    family_id = rand_name("family")
    mark = f"external development {rand_name('mark')}"
    w = make_world(actor)
    fam = validate_entry(ExternalEventFamily(
        family_id=family_id, description="an external development relevant to the scenario",
        marks=[mark], impact_mechanism="observation_delivery",
        impact_description="delivers a public news item",
        arrival=ArrivalModel(kind="scheduled_exact", scheduled_times=[T0 + DAY],
                             provenance="published docket")))
    assert fam.validation_error == ""
    proc = OutsideWorldProcess(boundary_id=rand_name("wb"), structural_model_id=rand_name("m"),
                               families=[fam])
    plan = SimpleNamespace(_outside_world=proc, horizon_ts=T0 + 5 * DAY)
    q = EventQueue(horizon_ts=T0 + 5 * DAY)
    n = schedule_outside_arrivals(plan, w, q)
    assert n == 1 and [e.ts for e in q.events] == [T0 + DAY]
    report = {}
    ops = [OutsideWorldEntryOperator(report=report), GeneratedAttentionOperator(report=report)]
    branch = RolloutEngine(operators=ops).run_branch(w, q, seed=0)
    stats = branch.temporal_stats
    # the arrival ENTERED at its exact time, through its declared mechanism
    entry_deltas = [d for d in branch.log if d.operator == "outside_world_entry"]
    assert entry_deltas and entry_deltas[0].at == T0 + DAY
    assert f"outside:{family_id}" in entry_deltas[0].reason_codes
    assert "entry:observation_delivery" in entry_deltas[0].reason_codes
    assert report["outside_events_entered"] == 1
    # availability CHANGED: the item became available at arrival time and the actor's real
    # attention opportunity later collected it into their information set
    assert stats.event_counts.get(OUTSIDE_EVENT_TYPE) == 1
    assert stats.event_counts.get("ctrl_attention", 0) >= 1
    assert stats.attention_batches and stats.attention_batches[0] == 1
    assert stats.delivery_to_attention_s and stats.delivery_to_attention_s[0] >= 0.0
    exposures = [x for x in w.information.exposures if x.actor_id == actor]
    assert len(exposures) == 1 and exposures[0].at >= T0 + DAY
    assert mark in w.information.items[exposures[0].item_id].content
    # and it NEVER wrote a terminal answer: no quantities appeared at all
    assert w.quantities == {}


# =====================================================================================
# 37.5 — nonhuman bottleneck: a pending mechanism instance BLOCKS action completion
#        until the nonhuman process itself resolves
# =====================================================================================
def test_fixture_37_5_pending_mechanism_blocks_completion_until_resolution():
    actor = rand_name("actor")
    action_id = rand_name("act")
    mech_id = rand_name("mech")
    w = make_world(actor, info=False)
    w.entities[actor].set("past_actions", F([{
        "action_id": action_id, "action": rand_name("submit"), "status": "attempted"}],
        status="observed"))
    inst = MechanismInstance(instance_id=rand_name("mi"), mechanism_id=mech_id,
                             branch_id=w.branch_id, originating_action_id=action_id,
                             initiating_actor_id=actor, state="queued", status="pending")
    w.mechanism_instances[inst.instance_id] = inst
    # while the nonhuman process is pending, the human action CANNOT complete
    status = refresh_attempt_status(w, inst)
    assert status == "mechanism_pending"
    row = w.entities[actor].value("past_actions")[0]
    assert row["completion_status"] == "mechanism_pending"
    assert row["status"] == "mechanism_pending"
    # only the MECHANISM's own resolution unblocks it — no human write can
    inst.status = "succeeded"
    inst.state = "processed"
    assert refresh_attempt_status(w, inst) == "mechanism_succeeded"
    assert w.entities[actor].value("past_actions")[0]["completion_status"] == \
        "mechanism_succeeded"
    # the honest unresolved path stays distinguishable from success and failure
    inst.status = "unresolved"
    assert refresh_attempt_status(w, inst) == "mechanism_unresolved"


# =====================================================================================
# 37.6 — attention difference: two particles receive the SAME available event; only one
#        notices it, driven by their different workload contexts
# =====================================================================================
def test_fixture_37_6_same_available_event_noticed_by_one_particle_only():
    actor = rand_name("actor")
    obs_id = rand_name("obs")
    content = f"the {rand_name('client')} account flagged a renewal risk"
    calm_tag = rand_name("calm")
    swamped_tag = rand_name("swamped")

    def scripted(prompt):
        if "ATTENTION process" in prompt:
            if calm_tag in prompt:
                return json.dumps({"noticed": [{"obs_id": obs_id, "why": "room to scan"}],
                                   "missed": []})
            return json.dumps({"noticed": [],
                               "missed": [{"obs_id": obs_id, "why": "buried under load"}]})
        if "private sense" in prompt:
            return json.dumps({"what_happened": "another item crossed my desk",
                               "unresolved_ambiguity": ""})
        if "options even OCCUR" in prompt:
            return json.dumps({"shortlist": ["keep working"]})
        raise AssertionError("unexpected prompt")

    def particle(branch, workload):
        w = make_world(actor, branch=branch)
        return run_cognition_pipeline(
            world=w, actor_id=actor, branch_id=branch, at=T0 + 60,
            available=[{"obs_id": obs_id, "channel": "chat", "source": "teammate",
                        "summary": content}],
            attention_context={"workload": workload},
            rng=random.Random(7), llm=scripted)

    cog_calm = particle("b001", f"steady week {calm_tag}")
    cog_swamped = particle("b002", f"triple deadline crunch {swamped_tag}")
    # identical availability on both particles
    assert cog_calm.observations_available == cog_swamped.observations_available == [obs_id]
    # divergent ATTENTION, with recorded reasons on the miss
    assert [n["obs_id"] for n in cog_calm.attention["noticed"]] == [obs_id]
    assert cog_calm.attention["missed"] == []
    assert cog_swamped.attention["noticed"] == []
    assert [m["obs_id"] for m in cog_swamped.attention["missed"]] == [obs_id]
    assert cog_swamped.attention["missed"][0]["why"]
    # the unnoticed content never reaches the swamped particle's decision context
    assert content in json.dumps(cog_calm.decision_context())
    assert content not in json.dumps(cog_swamped.decision_context())


# =====================================================================================
# 37.7 — memory difference: the same noticed event retrieves a memory in one particle
#        and fails retrieval in another (salience/staleness band, branch-seeded draw)
# =====================================================================================
def test_fixture_37_7_same_cue_different_retrieval_across_particles():
    actor = rand_name("actor")
    topic = rand_name("topic")
    memory_id = rand_name("mem")

    def setup():
        wm = WorkingMemoryState(actor_id=actor)
        wm.capacity_last = 5
        wm.items.append(WorkingMemoryItem(item_id=rand_name("wmi"), kind="observation",
                                          content=f"the {topic} matter surfaced again today",
                                          entered_at=T0, refreshed_at=T0, source="o1"))
        mem = ActorMemoryState(actor_id=actor)
        mem.episodic.append(EpisodicMemory(
            memory_id=memory_id, at=T0 - 40 * DAY,
            content=f"an old unrehearsed note about the {topic} matter",
            retrieval_cues=[f"{topic} matter"], salience="low"))   # stale + low → band 0.6
        return wm, mem

    wm1, mem1 = setup()
    out_fail = memory_retrieval_stage(mem=mem1, wm=wm1, actor_id=actor, branch_id="b001",
                                      at=T0, rng=random.Random(1))
    wm2, mem2 = setup()
    out_ok = memory_retrieval_stage(mem=mem2, wm=wm2, actor_id=actor, branch_id="b002",
                                    at=T0, rng=random.Random(2))
    assert out_fail["retrieved"] == []
    assert out_fail["retrieval_failures"][0]["memory_id"] == memory_id
    assert out_ok["retrieved"] == [memory_id] and out_ok["retrieval_failures"] == []
    # the divergence is REAL state: one particle's mind now holds the memory, the other's not
    assert not any(i.source == memory_id for i in wm1.active())
    assert any(i.source == memory_id for i in wm2.active())
    assert mem1.episodic[0].times_recalled == 0
    assert mem2.episodic[0].times_recalled == 1


# =====================================================================================
# 37.8 — contradictory beliefs: the same observation activates DIFFERENT beliefs under
#        different active contexts, driving different behavior
# =====================================================================================
def test_fixture_37_8_contradictory_beliefs_drive_context_dependent_behavior():
    actor = rand_name("actor")
    belief_bold = f"moving first wins the {rand_name('market')}"
    belief_wary = f"moving first invites the {rand_name('regulator')}"
    act_bold = rand_name("seize")
    act_wary = rand_name("hold")
    rested_tag = rand_name("rested")
    depleted_tag = rand_name("depleted")
    observation = f"a rival opening appeared in the {rand_name('segment')} segment"

    def scripted(prompt):
        if "ATTENTION process" in prompt:
            raise AssertionError("interrupting item should bypass the LLM band")
        if "private sense" in prompt:
            if rested_tag in prompt:
                return json.dumps({"what_happened": "an opening worth taking now",
                                   "active_belief": belief_bold, "unresolved_ambiguity": ""})
            return json.dumps({"what_happened": "bait that would expose me",
                               "active_belief": belief_wary, "unresolved_ambiguity": ""})
        if "options even OCCUR" in prompt:
            return json.dumps({"shortlist": [act_bold, act_wary]})
        raise AssertionError("unexpected prompt")

    def decision_llm(prompt):
        return _decision_payload(act_bold if "an opening worth taking" in prompt else act_wary)

    def particle(branch, condition_tag):
        w = make_world(actor, branch=branch)
        mem = ActorMemoryState(actor_id=actor)
        mem.beliefs.append(BeliefRecord(belief_id="b1", content=belief_bold,
                                        conflicts_with="b2"))
        mem.beliefs.append(BeliefRecord(belief_id="b2", content=belief_wary,
                                        conflicts_with="b1"))
        store_memory(w, mem)
        cog = run_cognition_pipeline(
            world=w, actor_id=actor, branch_id=branch, at=T0 + 60,
            available=[{"obs_id": rand_name("obs"), "channel": "phone", "source": "scout",
                        "summary": observation, "interrupting": True}],
            condition=condition_tag, rng=random.Random(3), llm=scripted)
        engine = QualitativeDecisionEngine(QualitativeConfig(
            llm=decision_llm, llm_hypotheses=False, bounded_cognition=False))
        view = ActorViewBuilder().build(w, actor)
        qd = engine.decide(view, None, observation,
                           [{"line": f"- {act_bold}"}, {"line": f"- {act_wary}"}],
                           cognition=cog)
        return cog, qd

    cog_a, qd_a = particle("b001", f"fresh after a quiet weekend {rested_tag}")
    cog_b, qd_b = particle("b002", f"running on four hours of sleep {depleted_tag}")
    # both particles hold the SAME contradictory belief pair, surfaced un-averaged
    for cog in (cog_a, cog_b):
        assert len(cog.retrieval["active_contradictions"]) == 1
        contents = cog.retrieval["active_contradictions"][0]["contents"]
        assert {belief_bold, belief_wary} == set(contents)
    # the same observation activated DIFFERENT beliefs per context
    assert cog_a.interpretation["active_belief"] == belief_bold
    assert cog_b.interpretation["active_belief"] == belief_wary
    # and the active belief drove genuinely different behavior
    assert qd_a.chosen_action == act_bold
    assert qd_b.chosen_action == act_wary
    assert qd_a.chosen_action != qd_b.chosen_action


# =====================================================================================
# 37.9 — limited search: the shortlist misses the globally best known action; the
#        decision is made from the shortlist and the gap is recorded
# =====================================================================================
def test_fixture_37_9_limited_search_misses_global_best_action():
    actor = rand_name("actor")
    best_action = rand_name("winning_motion")
    habit_action = rand_name("quiet_note")
    situation = f"the {rand_name('dispute')} escalated overnight"

    def scripted(prompt):
        if "ATTENTION process" in prompt:
            raise AssertionError("interrupting item should bypass the LLM band")
        if "private sense" in prompt:
            return json.dumps({"what_happened": "I need to respond today",
                               "unresolved_ambiguity": ""})
        if "options even OCCUR" in prompt:
            # this person simply does not think of the winning move
            return json.dumps({"options_recalled": [habit_action],
                               "shortlist": [habit_action],
                               "options_screened_out": []})
        raise AssertionError("unexpected prompt")

    w = make_world(actor)
    cog = run_cognition_pipeline(
        world=w, actor_id=actor, branch_id="b001", at=T0 + 60,
        available=[{"obs_id": rand_name("obs"), "channel": "phone", "source": "counsel",
                    "summary": situation, "interrupting": True}],
        known_options=[best_action, habit_action],
        rng=random.Random(5), llm=scripted)
    assert cog.search["shortlist"] == [habit_action]
    # the §15 record: the feasible best action existed and was never considered
    assert best_action in cog.search["actually_feasible_not_considered"]

    def decision_llm(prompt):
        # the actively-considered set the prompt presents comes from the SEARCH stage; if the
        # shortlist machinery ever vanished, this scripted mind would take the best action and
        # the fixture would fail
        considering = [ln for ln in prompt.splitlines()
                       if "(you are actively considering)" in ln]
        assert considering and all(best_action not in ln for ln in considering)
        chosen = habit_action if any(habit_action in ln for ln in considering) else best_action
        return _decision_payload(chosen)

    engine = QualitativeDecisionEngine(QualitativeConfig(
        llm=decision_llm, llm_hypotheses=False, bounded_cognition=False))
    view = ActorViewBuilder().build(w, actor)
    # the KNOWN menu still offers the globally best action — the search stage is what missed it
    menu = [{"line": f"- {best_action}"}, {"line": f"- {habit_action}"}]
    qd = engine.decide(view, None, situation, menu, cognition=cog)
    assert qd.chosen_action == habit_action
    assert qd.chosen_action != best_action
    assert qd.chosen_action in cog.search["shortlist"]


# =====================================================================================
# 37.10 — model-family disagreement: two fake families produce different choices from
#         EQUIVALENT structured states; per-family decisions are traceable
# =====================================================================================
def test_fixture_37_10_model_family_disagreement_recorded_per_family():
    actor = rand_name("actor")
    fam_a = rand_name("family_a")
    fam_b = rand_name("family_b")
    act_a = rand_name("push_back")
    act_b = rand_name("acknowledge")
    hostile_mark = f"a hostile move against my {rand_name('position')}"
    routine_mark = f"a routine update about the {rand_name('docket')}"
    bundle_content = f"the rival filed the same {rand_name('claim')} claim"
    obs_id = rand_name("obs")

    def fam_client(tag):
        def call(prompt):
            if "ATTENTION process" in prompt:
                return json.dumps({"noticed": [{"obs_id": obs_id, "why": "relevant"}],
                                   "missed": []})
            if "private sense" in prompt:
                return json.dumps({"what_happened": hostile_mark if tag == fam_a
                                   else routine_mark, "unresolved_ambiguity": ""})
            if "options even OCCUR" in prompt:
                return json.dumps({"shortlist": [act_a if tag == fam_a else act_b]})
            raise AssertionError("unexpected prompt")
        return call

    def decision_llm(prompt):
        return _decision_payload(act_a if hostile_mark in prompt else act_b)

    pool = FamilyPool()
    pool.register(ModelFamily(family_id=fam_a, provider=rand_name("prov"),
                              model=rand_name("model"), lineage=rand_name("lineage"),
                              availability="configured", client=fam_client(fam_a)))
    pool.register(ModelFamily(family_id=fam_b, provider=rand_name("prov"),
                              model=rand_name("model"), lineage=rand_name("lineage"),
                              availability="configured", client=fam_client(fam_b)))
    # find one particle index served by each family (deterministic stable-hash assignment)
    idx_of = {}
    for i in range(64):
        idx_of.setdefault(pool.assign(particle_index=i, actor_id=actor, record=False), i)
        if len(idx_of) == 2:
            break
    assert set(idx_of) == {fam_a, fam_b}
    engine = QualitativeDecisionEngine(QualitativeConfig(
        llm=decision_llm, llm_hypotheses=False, bounded_cognition=True, family_pool=pool))
    rt = QualitativeActorPolicyRuntime(engine, mode="persistent_qualitative_llm_policy")
    outcomes = {}
    for fam_id, idx in idx_of.items():
        w = make_world(actor, branch=f"b{idx:03d}")
        w.particle_index = idx
        store_actor_state(w, QualitativeActorState(actor_id=actor, hypothesis_id="h0:same",
                                                   identity_and_role=f"{actor}, founder"),
                          method="test_seed")
        decision = {"candidate_actions": [act_a, act_b], "situation": "a rival filing",
                    # EQUIVALENT structured availability for both particles
                    "observation_bundle": [{"iid": obs_id, "channel": "chat",
                                            "source": "colleague", "content": bundle_content}]}
        sel, post, tr = rt.decide(None, [w], actor, decision=decision, seed=idx)
        cogp = post.provenance["cognition"]
        outcomes[fam_id] = (cogp["model_family"], sel.action_name,
                            cogp["observations_available"])
    # equivalent structured states in, per-family divergent choices out
    assert outcomes[fam_a][2] == outcomes[fam_b][2] == [obs_id]
    assert outcomes[fam_a][0] == fam_a and outcomes[fam_b][0] == fam_b
    assert outcomes[fam_a][1] == act_a and outcomes[fam_b][1] == act_b
    assert outcomes[fam_a][1] != outcomes[fam_b][1]
    # the assignment log makes each decision traceable to its serving family
    logged = {(r["particle"], r["actor"], r["family"]) for r in pool.assignment_log}
    assert (idx_of[fam_a], actor, fam_a) in logged
    assert (idx_of[fam_b], actor, fam_b) in logged


# =====================================================================================
# 37.11 — actor-budget exhaustion stops the branch at exactly the unresolved decision;
#         no substitute action ever enters the delta log
# =====================================================================================
def test_fixture_37_11_budget_exhaustion_stops_branch_at_unresolved_decision(monkeypatch):
    _strict(monkeypatch)
    actor = rand_name("actor")
    trigger = {"trigger_type": "newly_noticed_information", "trigger_id": rand_name("trg")}
    engine = QualitativeDecisionEngine(QualitativeConfig(
        llm=lambda p: _decision_payload("anything"), llm_hypotheses=False,
        max_llm_calls=0, bounded_cognition=False))               # budget exhausted up front
    rt = QualitativeActorPolicyRuntime(engine, mode="persistent_qualitative_llm_policy")
    op = ProductionActorPolicyOperator(runtime=rt)
    w = make_world(actor, info=False)
    first_ts, second_ts = T0 + 60, T0 + 2 * DAY
    horizon = T0 + 10 * DAY
    q = EventQueue(horizon_ts=horizon)
    for ts in (first_ts, second_ts):
        q.schedule(Event(ts=ts, etype="decision_opportunity", participants=[actor],
                         payload={"situation": "a development needs a response",
                                  "candidate_actions": [rand_name("a"), rand_name("b")],
                                  "trigger": dict(trigger)}))
    branch = RolloutEngine(operators=[op]).run_branch(w, q, seed=0)
    stats = branch.temporal_stats
    # stopped at EXACTLY the first unresolved decision
    assert stats.branch_halted is True
    assert stats.branch_status == "truncated_actor_budget"
    assert stats.truncation["reason"] == "actor_llm_budget_exhausted"
    assert stats.truncation["at_ts"] == first_ts
    assert w.clock.now == first_ts < horizon
    assert stats.event_counts.get("decision_opportunity") == 1   # the second never ran
    assert [e.ts for e in q.events] == [second_ts]               # pending, preserved
    assert stats.truncation["pending_events"]
    # the unresolved decision trigger rides the truncation record
    assert stats.truncation["unresolved_decision_trigger"]["trigger_id"] == \
        trigger["trigger_id"]
    assert actor in stats.truncation["actors_not_processed"]
    # NO substitute action after the truncation: no executed action anywhere in the log,
    # no action in the actor's history, and no delta beyond the halt timestamp
    assert all("executed_action" not in d.uncertainty for d in branch.log)
    assert all(d.at <= first_ts for d in branch.log)
    assert w.entities[actor].value("past_actions") == []
    halt_delta = branch.log[-1]
    assert "temporally_truncated:actor_llm_budget_exhausted" in halt_delta.reason_codes


# =====================================================================================
# 37.12 — missing mechanism: the broad-prior resolution is SUPPRESSED with a first-class
#         record classifying under_modeled_nonhuman_mechanism (unit-level: the structural
#         assembly consumes exactly these records via temporal_runtime.mechanism_suppressions
#         → structural_runtime._assemble_ensemble_result; the full ensemble path needs a
#         compiled multi-model run and is exercised by the structural suites)
# =====================================================================================
def test_fixture_37_12_missing_mechanism_yields_suppression_and_classification(monkeypatch):
    _strict(monkeypatch)
    outcome_var = rand_name("readout")
    w = make_world(info=False)
    q = EventQueue(horizon_ts=T0 + 5 * DAY)
    q.schedule(Event(ts=T0 + DAY, etype="resolve_outcome", participants=[],
                     payload={"outcome_var": outcome_var, "family": "binary",
                              "lean": "weak_yes", "options": ["yes", "no"]}))
    branch = RolloutEngine(operators=[GenericOutcomeOperator()]).run_branch(w, q, seed=0)
    # the outcome stays honestly UNRESOLVED — no broad-prior draw happened
    assert outcome_var not in w.quantities
    supp_deltas = [d for d in branch.log
                   if "generic_prior_suppressed_default" in d.reason_codes]
    assert supp_deltas and supp_deltas[0].changes == []
    assert "under_modeled_nonhuman_mechanism" in supp_deltas[0].reason_codes
    # the first-class suppression record names the mechanism, the variable, and the class
    sup = get_stats(w).mechanism_suppressions
    assert len(sup) == 1
    assert sup[0]["mechanism"] == "generic_outcome_prior"
    assert sup[0]["outcome_var"] == outcome_var
    assert sup[0]["classification"] == "under_modeled_nonhuman_mechanism"
    assert sup[0]["at_ts"] == T0 + DAY
    assert w.omissions[0]["kind"] == "generic_prior_suppressed"
    # and it survives temporal aggregation — the exact input the §35 result classification
    # reads to emit under_modeled_nonhuman_mechanism
    agg = aggregate_temporal_stats([branch])
    assert agg["mechanism_suppressions"][0]["outcome_var"] == outcome_var
    assert agg["mechanism_suppressions"][0]["classification"] == \
        "under_modeled_nonhuman_mechanism"


# ---------------------------------------------------------------------- 37.3 / 37.4
@pytest.mark.skip(reason="37.3/37.4 are the live-generation fixtures (boundary generation + "
                         "critics; outside-world family identification) — they require an "
                         "actual LLM backend and are covered by "
                         "experiments/core_arch_forensics.py; faking the generation step "
                         "offline would test the fake, not the architecture")
def test_fixture_37_3_and_37_4_live_generation_fixtures():
    raise AssertionError("intentionally skipped — see reason")
