"""Contract tests for swm/world_model_v2/bounded_cognition.py and model_families.py.

All LLM calls use fake callables returning canned JSON keyed on prompt markers; every RNG is
seeded, so the suite is fully deterministic.
"""
import json
import random

import pytest

from swm.world_model_v2 import bounded_cognition as bc
from swm.world_model_v2.bounded_cognition import (
    ACTOR_MEMORY_KEY,
    COGNITION_SCHEMA,
    WM_BASE,
    WM_FLOOR,
    WM_SAFETY_MAX,
    WORKING_MEMORY_KEY,
    ActorMemoryState,
    BeliefRecord,
    CognitionStageFailure,
    EpisodicMemory,
    WorkingMemoryItem,
    WorkingMemoryState,
    action_search_stage,
    attention_stage,
    commit_decision,
    interpretation_stage,
    load_memory,
    load_working_memory,
    memory_retrieval_stage,
    memory_update_stage,
    run_cognition_pipeline,
    situational_capacity,
    store_memory,
    working_memory_stage,
)
from swm.world_model_v2.model_families import (
    HONEST_ACTOR_LANGUAGE,
    FamilyIdentityError,
    FamilyPool,
    ModelFamily,
    default_family_pool,
)
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

T0 = 1_700_000_000.0
DAY = 86400.0


def make_world(*actors, now=T0):
    w = WorldState("cogw", "b7", SimulationClock(now=now, as_of=now))
    for a in actors:
        e = Entity(a)
        e.set("roles", F(["person"], status="observed"))
        w.entities[a] = e
    return w


def _fam(fid, provider, model, *, lineage="", tier="strong", configured=True):
    return ModelFamily(family_id=fid, provider=provider, model=model, lineage=lineage,
                       strength_tier=tier,
                       availability="configured" if configured else "unknown",
                       client=(lambda p: "ok") if configured else None)


# =====================================================================================
# model families (§17)
# =====================================================================================
def test_same_provider_model_twice_raises_family_identity_error():
    pool = FamilyPool()
    pool.register(_fam("deepseek_cold", "deepseek", "deepseek-chat"))
    with pytest.raises(FamilyIdentityError, match="NOT different families"):
        pool.register(_fam("deepseek_hot", "deepseek", "deepseek-chat"))   # the temperature move


def test_duplicate_family_id_raises():
    pool = FamilyPool()
    pool.register(_fam("famA", "deepseek", "deepseek-chat"))
    with pytest.raises(FamilyIdentityError, match="duplicate family_id"):
        pool.register(_fam("famA", "openrouter", "qwen-2.5-72b"))


def test_model_family_metadata_rules():
    with pytest.raises(ValueError):
        ModelFamily(family_id="x", provider="p", model="m", strength_tier="medium")
    f = ModelFamily(family_id="x", provider="p", model="m")
    assert f.lineage == "p:m"                              # lineage defaults to provider:model
    assert "client" not in f.as_dict()


def test_monoculture_true_for_one_lineage_even_with_two_models():
    pool = FamilyPool()
    pool.register(_fam("ds_chat", "deepseek", "deepseek-chat", lineage="deepseek-v3"))
    pool.register(_fam("ds_reasoner", "deepseek", "deepseek-reasoner", lineage="deepseek-v3"))
    assert pool.distinct_lineages() == ["deepseek-v3"]
    assert pool.monoculture() is True
    pool.register(_fam("qwen", "openrouter", "qwen-2.5-72b", lineage="qwen-2.5"))
    assert pool.monoculture() is False
    assert pool.distinct_lineages() == ["deepseek-v3", "qwen-2.5"]


def test_assignment_deterministic_and_distributed_across_strong_families():
    def build():
        p = FamilyPool()
        p.register(_fam("alpha_family", "prov_a", "model-a", lineage="la"))
        p.register(_fam("beta_family", "prov_b", "model-b", lineage="lb"))
        return p
    pool, pool2 = build(), build()
    picks = [pool.assign(particle_index=i, actor_id="actor_1") for i in range(60)]
    assert set(picks) == {"alpha_family", "beta_family"}    # both strong families serve particles
    for i in (0, 3, 17, 59):
        assert pool.assign(particle_index=i, actor_id="actor_1", record=False) == picks[i]
        assert pool2.assign(particle_index=i, actor_id="actor_1", record=False) == picks[i]
    log = pool.assignment_log[0]
    assert log["rule"] == "stable_hash_across_comparable_strong_families"
    assert log["particle"] == 0 and log["actor"] == "actor_1"
    assert len(pool.assignment_log) == 60


def test_weaker_family_never_serves_primary_particles():
    pool = FamilyPool()
    pool.register(_fam("strong_one", "prov_a", "model-a"))
    pool.register(_fam("cheap_helper", "prov_c", "model-c", tier="weaker"))
    for i in range(50):
        assert pool.assign(particle_index=i, actor_id=f"actor_{i % 3}") == "strong_one"
    assert all(rec["rule"] == "single_strong_family" for rec in pool.assignment_log)
    weak_only = FamilyPool()
    weak_only.register(_fam("cheap_helper", "prov_c", "model-c", tier="weaker"))
    with pytest.raises(RuntimeError, match="no configured strong model family"):
        weak_only.assign(particle_index=0, actor_id="a")
    # weaker families are reachable only as ADDITIONAL adversarial coverage
    assert pool.adversarial_extra_family() == "cheap_helper"
    assert pool.adversarial_extra_family(exclude="cheap_helper") == "strong_one"


def test_comparable_alternative_is_same_tier_only():
    pool = FamilyPool()
    pool.register(_fam("strong_a", "prov_a", "model-a"))
    pool.register(_fam("strong_b", "prov_b", "model-b"))
    pool.register(_fam("weak_w", "prov_w", "model-w", tier="weaker"))
    assert pool.comparable_alternative("strong_a") == "strong_b"
    assert pool.comparable_alternative("strong_b") == "strong_a"
    assert pool.comparable_alternative("weak_w") is None    # no weaker peer → stop, not degrade
    assert pool.comparable_alternative("missing") is None


def test_failure_transitions_recorded_and_reported():
    pool = FamilyPool()
    pool.register(_fam("strong_a", "prov_a", "model-a"))
    pool.register(_fam("strong_b", "prov_b", "model-b"))
    rec = pool.record_failure_transition(particle_index=4, actor_id="maya",
                                         from_family="strong_a", to_family="strong_b",
                                         error="HTTP 500 after retries", at=T0)
    assert rec == {"particle": 4, "actor": "maya", "from": "strong_a", "to": "strong_b",
                   "error": "HTTP 500 after retries", "at": T0}
    assert pool.failure_transitions == [rec]
    assert pool.report()["failure_transitions"] == [rec]


def test_report_carries_honest_language_and_monoculture_flag():
    pool = FamilyPool()
    pool.register(_fam("only_family", "prov_a", "model-a", lineage="la"))
    rep = pool.report()
    assert rep["actor_independence_language"] == HONEST_ACTOR_LANGUAGE
    assert "independently situated LLM actor instances" in rep["actor_independence_language"]
    assert rep["model_family_monoculture"] is True
    assert "correlated failure risk" in rep["monoculture_note"]
    pool.register(_fam("second_family", "prov_b", "model-b", lineage="lb"))
    rep2 = pool.report()
    assert rep2["model_family_monoculture"] is False
    assert "distributed across distinct lineages" in rep2["monoculture_note"]
    assert rep2["configured_families"] == ["only_family", "second_family"]


def test_default_family_pool_registers_deepseek_single_strong_family(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "{}"

    pool = default_family_pool(llm=fake_llm)
    assert [f.family_id for f in pool.families] == ["deepseek_v3"]
    fam = pool.by_id("deepseek_v3")
    assert fam.provider == "deepseek" and fam.model == "deepseek-chat"
    assert fam.strength_tier == "strong" and fam.availability == "configured"
    assert pool.strong() == [fam] and pool.monoculture() is True
    assert pool.assign(particle_index=0, actor_id="a") == "deepseek_v3"
    assert pool.call("deepseek_v3", "hello") == "{}" and calls == ["hello"]
    assert default_family_pool(llm=None).families == []


# =====================================================================================
# situational working-memory capacity (§11)
# =====================================================================================
def test_situational_capacity_matches_hand_computed_cases():
    assert situational_capacity()[0] == WM_BASE
    assert situational_capacity(workload="light afternoon")[0] == 6
    assert situational_capacity(workload="very high pressure")[0] == 3
    assert situational_capacity(workload="busy")[0] == 4
    assert situational_capacity(urgency="hard deadline")[0] == 4
    assert situational_capacity(interruptions=2)[0] == 4
    assert situational_capacity(interruptions=1)[0] == 5           # below threshold
    assert situational_capacity(condition="sleep-deprived")[0] == 4
    assert situational_capacity(n_active_tasks=3)[0] == 4
    cap, basis = situational_capacity(workload="high", urgency="urgent", interruptions=2)
    assert cap == 2 and "workload:high(-1)" in basis and "urgency:tunnel(-1)" in basis


def test_situational_capacity_bounds_under_extreme_inputs():
    cap, basis = situational_capacity(workload="crisis overload swamped", urgency="urgent now",
                                      interruptions=99, condition="exhausted burned out sick",
                                      n_active_tasks=40)
    assert cap == WM_FLOOR                                          # 5-2-1-1-1-1 clamps at floor
    assert f"safety_max={WM_SAFETY_MAX}" in basis
    for wl in ("", "light", "high", "crisis"):
        for ints in (0, 5):
            for tasks in (0, 9):
                c, _ = situational_capacity(workload=wl, urgency="deadline",
                                            interruptions=ints, condition="fatigued",
                                            n_active_tasks=tasks)
                assert WM_FLOOR <= c <= WM_SAFETY_MAX


# =====================================================================================
# attention stage (§10)
# =====================================================================================
def test_attention_deterministic_rules_interrupt_and_muted():
    available = [
        {"obs_id": "o_int", "channel": "pager", "summary": "outage", "interrupting": True},
        {"obs_id": "o_crit", "channel": "phone", "summary": "call", "urgency": "critical"},
        {"obs_id": "o_mut", "channel": "Newsletter", "summary": "promo"},
        {"obs_id": "o_addr", "channel": "chat", "summary": "boss ping",
         "directly_addressed": True, "urgency": "high"},
    ]
    out = attention_stage(actor_id="a", branch_id="b", at=T0, available=available,
                          attention_context={"muted_channels": ["newsletter"]}, llm=None)
    noticed = {n["obs_id"]: n["why"] for n in out["noticed"]}
    missed = {m["obs_id"]: m["why"] for m in out["missed"]}
    assert set(noticed) == {"o_int", "o_crit", "o_addr"}
    assert set(missed) == {"o_mut"}
    assert "interrupting" in noticed["o_int"]
    assert "muted/deprioritized" in missed["o_mut"]
    assert set(noticed) | set(missed) == {o["obs_id"] for o in available}
    asleep = attention_stage(actor_id="a", branch_id="b", at=T0,
                             available=[{"obs_id": "o1", "channel": "chat", "summary": "s"}],
                             attention_context={"asleep": True}, llm=None)
    assert asleep["missed"][0]["why"] == "asleep (temporal availability)"


def test_attention_ambiguous_band_judged_by_llm():
    prompts = []

    def att_llm(prompt):
        prompts.append(prompt)
        assert "ATTENTION process" in prompt
        return json.dumps({"noticed": [{"obs_id": "amb1", "why": "mentions her own project"}],
                           "missed": [{"obs_id": "amb2", "why": "routine digest"},
                                      {"obs_id": "ghost", "why": "hallucinated id"}]})

    available = [{"obs_id": "amb1", "channel": "chat", "summary": "your project slipped"},
                 {"obs_id": "amb2", "channel": "email", "summary": "weekly digest"}]
    out = attention_stage(actor_id="a", branch_id="b", at=T0, available=available,
                          attention_context={"focus": "project"}, llm=att_llm,
                          family_id="famX")
    assert [n["obs_id"] for n in out["noticed"]] == ["amb1"]
    assert out["noticed"][0]["why"] == "mentions her own project"
    assert [m["obs_id"] for m in out["missed"]] == ["amb2"]
    assert "ghost" not in json.dumps(out)                       # hallucinated ids filtered
    mc = out["trace"]["model_call"]
    assert mc["family"] == "famX" and mc["attempts"] == 1
    assert mc["prompt_hash"] == bc._hash(prompts[0])


def test_attention_unjudged_ambiguous_item_is_honestly_missed():
    def partial_llm(prompt):
        return json.dumps({"noticed": [{"obs_id": "amb1", "why": "seen"}], "missed": []})
    available = [{"obs_id": "amb1", "channel": "chat", "summary": "a"},
                 {"obs_id": "amb2", "channel": "chat", "summary": "b"}]
    out = attention_stage(actor_id="a", branch_id="b", at=T0, available=available,
                          llm=partial_llm)
    missed = {m["obs_id"]: m["why"] for m in out["missed"]}
    assert missed == {"amb2": "not registered (attention judgment)"}


def test_attention_never_invents_attention_without_llm():
    available = [{"obs_id": "amb1", "channel": "chat", "summary": "a"},
                 {"obs_id": "amb2", "channel": "chat", "summary": "b"}]
    out = attention_stage(actor_id="a", branch_id="b", at=T0, available=available, llm=None)
    assert out["noticed"] == []
    assert all(m["why"] == "attention_judgment_unavailable (no llm)" for m in out["missed"])

    def broken(prompt):
        raise TimeoutError("llm gone")
    out2 = attention_stage(actor_id="a", branch_id="b", at=T0, available=available,
                           llm=broken, llm_retries=1)
    assert out2["noticed"] == []
    assert all(m["why"] == "attention_judgment_unavailable" for m in out2["missed"])
    assert "TimeoutError" in out2["trace"]["failure"]           # loud, never raising


# =====================================================================================
# working-memory stage (§11)
# =====================================================================================
def test_working_memory_refreshes_instead_of_duplicating():
    wm = WorkingMemoryState(actor_id="a")
    wm.items.append(WorkingMemoryItem(item_id="w_prev", kind="observation", content="old",
                                      entered_at=50.0, refreshed_at=50.0, source="o1"))
    out = working_memory_stage(wm=wm, actor_id="a", branch_id="b", at=100.0,
                               noticed=[{"obs_id": "o1"}],
                               available_by_id={"o1": {"summary": "same thing again"}})
    assert out["refreshed"] == ["w_prev"] and out["entered"] == []
    assert len(wm.items) == 1 and wm.items[0].refreshed_at == 100.0


def test_working_memory_displacement_logged_and_plan_survives_longest():
    wm = WorkingMemoryState(actor_id="a")
    wm.items.append(WorkingMemoryItem(item_id="plan1", kind="plan", content="ship the fix",
                                      entered_at=0.0, refreshed_at=0.0, source="plan"))
    wm.items.append(WorkingMemoryItem(item_id="uq1", kind="unresolved_question",
                                      content="who approves", entered_at=0.0,
                                      refreshed_at=0.0, source="uq"))
    noticed = [{"obs_id": f"o{i}"} for i in range(4)]
    by_id = {f"o{i}": {"summary": f"observation number {i}"} for i in range(4)}
    out = working_memory_stage(wm=wm, actor_id="a", branch_id="b", at=100.0, noticed=noticed,
                               available_by_id=by_id,
                               attention_context={"workload": "crisis", "urgency": "urgent"})
    assert out["capacity"] == 2                                    # 5 - 2 (crisis) - 1 (urgent)
    active = wm.active()
    assert {i.item_id for i in active} == {"plan1", "uq1"}         # plan outlives observations
    assert len(out["displaced"]) == 4
    assert all("capacity 2" in d["why"] for d in wm.displaced_log)
    assert {d["item_id"] for d in wm.displaced_log} == {i.item_id for i in wm.items
                                                        if i.activation == "displaced"}


def test_working_memory_active_items_view_is_exact():
    wm = WorkingMemoryState(actor_id="a")
    out = working_memory_stage(wm=wm, actor_id="a", branch_id="b", at=100.0,
                               noticed=[{"obs_id": "o1"}, {"obs_id": "o2"}],
                               available_by_id={"o1": {"summary": "first fact"},
                                                "o2": {"summary": "second fact"}})
    assert out["active_items"] == [i.as_dict() for i in wm.active()]
    assert [i["content"] for i in out["active_items"]] == ["first fact", "second fact"]
    assert out["trace"]["stage"] == "working_memory" and out["trace"]["output_hash"]


# =====================================================================================
# long-term memory retrieval (§12)
# =====================================================================================
def test_retrieval_cue_match_enters_working_memory_with_distortion_note():
    wm = WorkingMemoryState(actor_id="a")
    wm.capacity_last = 5
    wm.items.append(WorkingMemoryItem(item_id="w1", kind="observation",
                                      content="An urgent FDA letter about approval arrived",
                                      entered_at=T0, refreshed_at=T0, source="o1"))
    mem = ActorMemoryState(actor_id="a")
    mem.episodic.append(EpisodicMemory(
        memory_id="m_fda", at=T0 - 5 * DAY,
        content="Last quarter the FDA letter stalled our submission",
        retrieval_cues=["fda letter"], salience="high",
        distortions=[{"at": T0 - DAY, "reinterpretation": "maybe it was our own filing error",
                      "why": "hindsight"}]))
    out = memory_retrieval_stage(mem=mem, wm=wm, actor_id="a", branch_id="b", at=T0 + 60,
                                 rng=random.Random(0))            # 0.844 > 0.02 → retrieved
    assert out["retrieved"] == ["m_fda"] and out["retrieval_failures"] == []
    m = mem.episodic[0]
    assert m.times_recalled == 1 and m.last_recalled == T0 + 60
    items = {i.source: i for i in wm.active()}
    assert items["m_fda"].kind == "retrieved_memory"
    assert "[later reinterpretation: maybe it was our own filing error" in items["m_fda"].content


def test_retrieval_failure_for_low_salience_stale_memory_is_deterministic():
    def setup():
        wm = WorkingMemoryState(actor_id="a")
        wm.capacity_last = 5
        wm.items.append(WorkingMemoryItem(item_id="w1", kind="observation",
                                          content="thinking about the vendor invoice today",
                                          entered_at=T0, refreshed_at=T0, source="o1"))
        mem = ActorMemoryState(actor_id="a")
        mem.episodic.append(EpisodicMemory(
            memory_id="m_old", at=T0 - 40 * DAY,
            content="an unremarkable note", retrieval_cues=["vendor invoice"],
            salience="low"))                                       # stale + low → band 0.6
        return wm, mem

    wm, mem = setup()
    out = memory_retrieval_stage(mem=mem, wm=wm, actor_id="a", branch_id="b", at=T0,
                                 rng=random.Random(1))             # 0.134 < 0.6 → fails
    assert out["retrieved"] == []
    assert out["retrieval_failures"] == [{"memory_id": "m_old",
                                          "why": "retrieval failure (salience=low, stale=True)"}]
    assert mem.episodic[0].times_recalled == 0
    wm2, mem2 = setup()
    out2 = memory_retrieval_stage(mem=mem2, wm=wm2, actor_id="a", branch_id="b", at=T0,
                                  rng=random.Random(2))            # 0.956 > 0.6 → succeeds
    assert out2["retrieved"] == ["m_old"]


def test_retrieved_memory_subject_to_finite_capacity():
    wm = WorkingMemoryState(actor_id="a")
    wm.capacity_last = 2
    wm.items.append(WorkingMemoryItem(item_id="w1", kind="observation",
                                      content="oldest note about the audit", entered_at=T0,
                                      refreshed_at=T0, source="o1"))
    wm.items.append(WorkingMemoryItem(item_id="w2", kind="observation",
                                      content="newer note about the audit", entered_at=T0 + 1,
                                      refreshed_at=T0 + 1, source="o2"))
    mem = ActorMemoryState(actor_id="a")
    mem.episodic.append(EpisodicMemory(memory_id="m1", at=T0 - DAY,
                                       content="the audit memory", retrieval_cues=["audit"],
                                       salience="high"))
    memory_retrieval_stage(mem=mem, wm=wm, actor_id="a", branch_id="b", at=T0 + 10,
                           rng=random.Random(0))
    active_ids = {i.item_id for i in wm.active()}
    assert len(active_ids) == 2 and "w1" not in active_ids         # stalest displaced
    assert any(i.source == "m1" for i in wm.active())
    assert wm.displaced_log[-1]["why"] == "displaced by retrieved memory"


def test_retrieval_surfaces_contradictions_and_accessible_beliefs():
    wm = WorkingMemoryState(actor_id="a")
    mem = ActorMemoryState(actor_id="a")
    mem.beliefs.append(BeliefRecord(belief_id="b1", content="the vendor is reliable",
                                    conflicts_with="b2", contradiction_awareness="noticed"))
    mem.beliefs.append(BeliefRecord(belief_id="b2", content="the vendor cuts corners",
                                    conflicts_with="b1"))
    mem.beliefs.append(BeliefRecord(belief_id="b3", content="forgotten belief",
                                    currently_accessible=False))
    out = memory_retrieval_stage(mem=mem, wm=wm, actor_id="a", branch_id="b", at=T0,
                                 rng=random.Random(0))
    assert out["beliefs_accessible"] == ["b1", "b2"]
    assert len(out["active_contradictions"]) == 1                  # pair listed exactly once
    pair = out["active_contradictions"][0]
    assert sorted(pair["beliefs"]) == ["b1", "b2"]
    assert pair["contents"] == ["the vendor is reliable", "the vendor cuts corners"]
    # contradictory records coexist un-averaged
    assert [b.content for b in mem.beliefs][:2] == ["the vendor is reliable",
                                                    "the vendor cuts corners"]


def test_actor_memory_state_round_trips_with_distortions():
    mem = ActorMemoryState(actor_id="a")
    mem.episodic.append(EpisodicMemory(
        memory_id="m1", at=T0, content="the launch day", retrieval_cues=["launch"],
        salience="high", times_recalled=2, accessible=False,
        distortions=[{"at": T0 + 1, "reinterpretation": "it was chaos", "why": "retelling"}]))
    mem.beliefs.append(BeliefRecord(belief_id="b1", content="we ship late",
                                    conflicts_with="b2",
                                    contradiction_awareness="compartmentalized"))
    mem.commitments.append({"content": "call mom", "made_at": T0, "to_whom": "mom"})
    mem.habits.append({"action": "check email first", "context_cue": "morning"})
    mem.relationship_memories["bob"] = ["helped me once"]
    mem.unresolved_tasks.append({"task": "renew the license", "since": T0, "source": "note"})
    back = ActorMemoryState.from_dict(mem.as_dict())
    assert back.as_dict() == mem.as_dict()
    assert back.episodic[0].distortions == [{"at": T0 + 1, "reinterpretation": "it was chaos",
                                             "why": "retelling"}]
    assert back.episodic[0].accessible is False and back.episodic[0].times_recalled == 2
    assert back.beliefs[0].contradiction_awareness == "compartmentalized"
    wm = WorkingMemoryState(actor_id="a", capacity_last=3, capacity_basis="test")
    wm.items.append(WorkingMemoryItem(item_id="i1", kind="plan", content="c",
                                      entered_at=1.0, refreshed_at=2.0, source="s",
                                      activation="displaced"))
    wm.displaced_log.append({"at": 2.0, "item_id": "i1", "why": "capacity"})
    back_wm = WorkingMemoryState.from_dict(wm.as_dict())
    assert back_wm.as_dict() == wm.as_dict()


# =====================================================================================
# interpretation stage (§14)
# =====================================================================================
INTERP_JSON = json.dumps({
    "what_happened": "The rival cut prices and my top client noticed",
    "why_it_matters": "renewals are at stake",
    "perceived_sender_or_cause_intent": "the rival wants my mid-market accounts",
    "activated_memories": ["m_outage"],
    "active_belief": "price wars punish the smaller player",
    "perceived_opportunities": ["lock in annual contracts"],
    "perceived_threats": ["client churn"],
    "unresolved_ambiguity": "whether the cut is temporary",
})


def _interp_args(**over):
    args = dict(actor_id="a", branch_id="b", at=T0, identity="a mid-market sales lead",
                wm=WorkingMemoryState(actor_id="a"), retrieved={"retrieved": []},
                mem=ActorMemoryState(actor_id="a"), condition="tired")
    args.update(over)
    return args


def test_interpretation_raises_without_llm_and_on_unparseable_output():
    with pytest.raises(CognitionStageFailure) as ei:
        interpretation_stage(**_interp_args(), llm=None)
    assert ei.value.stage == "interpretation"
    assert ei.value.actor_id == "a" and ei.value.branch_id == "b"
    calls = []

    def word_salad(prompt):
        calls.append(prompt)
        return "definitely not json"

    with pytest.raises(CognitionStageFailure, match="no valid output after retries"):
        interpretation_stage(**_interp_args(), llm=word_salad, llm_retries=1)
    assert len(calls) == 2                                          # initial call + one retry


def test_interpretation_parses_and_enters_working_memory():
    wm = WorkingMemoryState(actor_id="a")

    def llm(prompt):
        assert "private sense" in prompt
        return INTERP_JSON

    out = interpretation_stage(**_interp_args(wm=wm), llm=llm, family_id="famZ")
    assert out["what_happened"] == "The rival cut prices and my top client noticed"
    assert out["active_belief"] == "price wars punish the smaller player"
    assert out["leakage_screen"] == {"markers_found": [], "clean": True}
    assert out["trace"]["model_call"]["family"] == "famZ"
    interp_items = [i for i in wm.active() if i.kind == "interpretation"]
    assert len(interp_items) == 1
    assert interp_items[0].content == out["what_happened"]


def test_interpretation_leakage_screen_flags_simulator_concepts():
    def leaky(prompt):
        return json.dumps({"what_happened": "As part of the simulation I saw the update",
                           "perceived_threats": ["the particle might get reweighted"],
                           "unresolved_ambiguity": ""})
    out = interpretation_stage(**_interp_args(), llm=leaky)
    assert out["leakage_screen"]["clean"] is False
    assert {"simulation", "particle"} <= set(out["leakage_screen"]["markers_found"])


# =====================================================================================
# limited action search (§15)
# =====================================================================================
SEARCH_JSON = json.dumps({
    "options_recalled": ["call the client directly"],
    "options_generated": ["offer a loyalty discount"],
    "options_screened_out": [{"option": "match the price cut",
                              "why_dismissed": "margins cannot absorb it"}],
    "shortlist": ["call the client directly", "offer a loyalty discount"],
})


def _search_args(**over):
    args = dict(actor_id="a", branch_id="b", at=T0, identity="sales lead",
                interpretation=json.loads(INTERP_JSON), wm=WorkingMemoryState(actor_id="a"),
                mem=ActorMemoryState(actor_id="a"))
    args.update(over)
    return args


def test_action_search_raises_on_missing_shortlist():
    with pytest.raises(CognitionStageFailure) as ei:
        action_search_stage(**_search_args(), llm=None)
    assert ei.value.stage == "action_search"
    with pytest.raises(CognitionStageFailure, match="no valid shortlist"):
        action_search_stage(**_search_args(),
                            llm=lambda p: json.dumps({"options_recalled": ["x"]}),
                            llm_retries=0)


def test_action_search_parses_and_lists_feasible_but_unconsidered_options():
    def llm(prompt):
        assert "options even OCCUR" in prompt
        return SEARCH_JSON

    known = ["call the client directly", "escalate to the board", "do nothing"]
    out = action_search_stage(**_search_args(), known_options=known, llm=llm)
    assert out["shortlist"] == ["call the client directly", "offer a loyalty discount"]
    assert out["options_recalled"] == ["call the client directly"]
    assert out["options_generated"] == ["offer a loyalty discount"]
    assert out["options_screened_out"] == [{"option": "match the price cut",
                                            "why_dismissed": "margins cannot absorb it"}]
    # the §15 distinction: feasible-but-never-considered stays visible to the auditor
    assert out["actually_feasible_not_considered"] == ["escalate to the board", "do nothing"]
    assert out["trace"]["stage"] == "action_search" and out["trace"]["output_hash"]


# =====================================================================================
# memory update after the choice (§12/§13)
# =====================================================================================
def test_memory_update_appends_episode_reinforces_belief_and_files_task():
    mem = ActorMemoryState(actor_id="a")
    mem.beliefs.append(BeliefRecord(belief_id="b1",
                                    content="price wars punish the smaller player"))
    mem.beliefs.append(BeliefRecord(belief_id="b2", content="discounts erode trust"))
    mem.habits.append({"action": "call the client", "context_cue": "churn risk"})
    wm = WorkingMemoryState(actor_id="a")
    out = memory_update_stage(mem=mem, wm=wm, actor_id="a", branch_id="b", at=T0,
                              interpretation=json.loads(INTERP_JSON),
                              decision={"chosen_action": "call the client directly",
                                        "decision_id": "d9"},
                              noticed=[{"obs_id": "o1"}], rng=random.Random(0))
    epi = mem.episodic[-1]
    assert out["episodic_added"] == epi.memory_id and out["episodic_encoded"] is True
    assert epi.salience == "high"                                  # stakes present → high
    assert epi.source_trace == "d9"
    assert "I chose: call the client directly" in epi.content
    assert "call the client directly" in epi.retrieval_cues
    assert out["belief_reinforced"] == "b1"
    assert mem.beliefs[0].last_reinforced == T0
    assert mem.beliefs[1].last_reinforced == 0.0                   # the other record untouched
    assert [b.content for b in mem.beliefs] == ["price wars punish the smaller player",
                                                "discounts erode trust"]   # never averaged
    assert out["habit_reinforced"] == "call the client"
    assert mem.habits[0]["reinforced_at"] == T0
    assert mem.unresolved_tasks[-1]["task"] == "whether the cut is temporary"


def test_memory_update_low_salience_episode_may_not_encode_under_load():
    def run(seed):
        mem = ActorMemoryState(actor_id="a")
        wm = WorkingMemoryState(actor_id="a")
        wm.displaced_log = [{"at": T0, "item_id": f"i{k}", "why": "capacity"} for k in range(3)]
        out = memory_update_stage(mem=mem, wm=wm, actor_id="a", branch_id="b", at=T0,
                                  interpretation={"what_happened": "a quiet afternoon"},
                                  decision={"chosen_action": "keep working"},
                                  noticed=[], rng=random.Random(seed))
        return out, mem.episodic[-1]

    out1, epi1 = run(1)                                            # 0.134 < 0.3 → not encoded
    assert epi1.salience == "low" and out1["episodic_encoded"] is False
    assert epi1.accessible is False
    assert epi1.distortions[0]["why"] == "not encoded (low salience under load)"
    out2, epi2 = run(2)                                            # 0.956 → encoded
    assert out2["episodic_encoded"] is True and epi2.accessible is True


# =====================================================================================
# assembled pipeline (§9)
# =====================================================================================
def pipeline_llm(prompt):
    if "ATTENTION process" in prompt:
        return json.dumps({"noticed": [{"obs_id": "o_amb", "why": "asks about her report"}],
                           "missed": []})
    if "private sense" in prompt:
        return INTERP_JSON
    if "options even OCCUR" in prompt:
        return SEARCH_JSON
    raise AssertionError("unexpected prompt: " + prompt[:60])


PIPE_AVAILABLE = [
    {"obs_id": "o_int", "channel": "pager", "source": "monitoring",
     "summary": "SERVER OUTAGE alarm in the east cluster", "interrupting": True},
    {"obs_id": "o_amb", "channel": "chat", "source": "teammate",
     "summary": "teammate asks how the quarterly report is going"},
    {"obs_id": "o_secret", "channel": "newsletter", "source": "vendor",
     "summary": "ZEBRA_SECRET_TOKEN_99 discount offer"},
]


def _run_pipeline(world):
    return run_cognition_pipeline(
        world=world, actor_id="maya", branch_id="b7x", at=T0 + 3600,
        available=PIPE_AVAILABLE, identity="maya, on-call engineer",
        attention_context={"focus": "quarterly report", "workload": "busy",
                           "muted_channels": ["newsletter"]},
        known_options=["call the client directly", "escalate to the on-call vendor"],
        rng=random.Random(0), llm=pipeline_llm, family_id="famP")


def _seed_memory(world):
    mem0 = ActorMemoryState(actor_id="maya")
    mem0.episodic.append(EpisodicMemory(
        memory_id="m_outage", at=T0 - 3 * DAY,
        content="The last outage alarm in the east cluster took a full day to clear",
        retrieval_cues=["outage alarm"], salience="high"))
    mem0.beliefs.append(BeliefRecord(belief_id="b1",
                                     content="price wars punish the smaller player",
                                     conflicts_with="b2"))
    mem0.beliefs.append(BeliefRecord(belief_id="b2",
                                     content="matching prices is the only defense",
                                     conflicts_with="b1"))
    store_memory(world, mem0)


def test_pipeline_decision_context_contains_only_surviving_material():
    world = make_world("maya")
    _seed_memory(world)
    cog = _run_pipeline(world)
    assert cog.failure == ""
    assert cog.observations_available == ["o_int", "o_amb", "o_secret"]
    missed_ids = {m["obs_id"] for m in cog.attention["missed"]}
    assert "o_secret" in missed_ids                                # muted channel missed
    ctx = cog.decision_context()
    js = json.dumps(ctx)
    assert "ZEBRA_SECRET_TOKEN_99" not in js                       # missed content never leaks in
    assert "SERVER OUTAGE alarm in the east cluster" in js         # noticed content survives
    assert ctx["retrieved_memories"] == ["m_outage"]
    assert len(ctx["active_contradictions"]) == 1
    assert ctx["shortlist"] == ["call the client directly", "offer a loyalty discount"]
    assert ctx["interpretation"]["active_belief"] == "price wars punish the smaller player"
    assert "trace" not in ctx["interpretation"]
    kinds = {i["kind"] for i in ctx["working_memory"]}
    assert kinds == {"observation", "retrieved_memory"}
    assert ctx["note"].startswith("context is the surviving bounded-cognition material")
    assert cog.search["actually_feasible_not_considered"] == ["escalate to the on-call vendor"]


def test_pipeline_stage_traces_carry_full_provenance():
    world = make_world("maya")
    _seed_memory(world)
    cog = _run_pipeline(world)
    assert [t["stage"] for t in cog.stage_traces] == \
        ["attention", "working_memory", "memory_retrieval", "interpretation", "action_search"]
    for t in cog.stage_traces:
        assert t["actor_id"] == "maya" and t["branch_id"] == "b7x"
        assert t["at"] == T0 + 3600 and t["schema"] == COGNITION_SCHEMA
        assert len(t["input_hash"]) == 16 and len(t["output_hash"]) == 16
        assert t["failure"] == ""
    by_stage = {t["stage"]: t for t in cog.stage_traces}
    assert by_stage["interpretation"]["model_call"]["family"] == "famP"
    assert by_stage["action_search"]["model_call"]["prompt_hash"]
    assert by_stage["working_memory"]["deterministic_rule"] == \
        "situational_capacity+displacement"
    # persisted state landed in the world under the typed latent keys
    ent = world.entities["maya"]
    assert ent.value("latent_state", key=WORKING_MEMORY_KEY)["actor_id"] == "maya"
    assert ent.value("latent_state", key=ACTOR_MEMORY_KEY)["actor_id"] == "maya"
    assert load_memory(world, "maya").episodic[0].times_recalled == 1   # retrieval persisted


def test_pipeline_commit_decision_appends_episode():
    world = make_world("maya")
    _seed_memory(world)
    cog = _run_pipeline(world)
    upd = commit_decision(world=world, cog=cog,
                          decision={"decision_id": "d1",
                                    "chosen_action": "call the client directly"})
    assert upd["episodic_added"].startswith("em_") and upd["episodic_encoded"] is True
    assert upd["belief_reinforced"] == "b1"
    assert cog.stage_traces[-1]["stage"] == "memory_update"
    mem = load_memory(world, "maya")
    assert len(mem.episodic) == 2
    assert "I chose: call the client directly" in mem.episodic[-1].content
    assert len({m.memory_id for m in mem.episodic}) == 2


def test_pipeline_interpretation_failure_persists_memory_before_raising():
    world = make_world("rex")

    def flaky(prompt):
        if "ATTENTION process" in prompt:
            return json.dumps({"noticed": [], "missed": []})
        return "utter nonsense"

    with pytest.raises(CognitionStageFailure) as ei:
        run_cognition_pipeline(
            world=world, actor_id="rex", branch_id="bF", at=T0 + 10,
            available=[{"obs_id": "o1", "channel": "pager", "summary": "pipeline broke",
                        "interrupting": True}],
            llm=flaky, rng=random.Random(0))
    assert ei.value.stage == "interpretation"
    assert ei.value.actor_id == "rex" and ei.value.branch_id == "bF"
    wm = load_working_memory(world, "rex")
    assert [i.source for i in wm.active()] == ["o1"]               # persisted before the raise
    assert wm.capacity_last == WM_BASE
    ent = world.entities["rex"]
    assert ent.value("latent_state", key=ACTOR_MEMORY_KEY)["actor_id"] == "rex"
