"""Lean V2 focused tests — exactly the 15 changed-behavior proofs (no broad campaign).

A scripted LLM drives `simulate_world(execution_profile="lean_v2")` through the REAL canonical
funnel: one blueprint call, deterministic validators, three-valued preflight, backward slice,
weighted causal waves with exact coalescing, template consequences, selective deliberation,
conditional challenger, forecast recovery."""
from __future__ import annotations

import json
import threading

import pytest

from swm.world_model_v2.lean_v2.blueprint import blueprint_from_dict
from swm.world_model_v2.lean_v2.challenger import decide_challenger, should_replicate
from swm.world_model_v2.lean_v2.worlds import WeightedBranchCoalescer, WeightedWorldNode
from swm.world_model_v2.unified_runtime import simulate_world

AS_OF, HORIZON = "2026-06-01", "2026-07-01"
QUESTION = "Will the board's June 25 decision be unanimous?"
EVIDENCE = ("Resolution criteria: unanimous means all three members cast the same vote. "
            "The board of three votes on June 25. Chair Ana Diaz has signaled a hold. "
            "Member Bo Ruiz is expected to follow the chair. Member Cy Vega dissented at the "
            "last meeting and may dissent again. Historically about seventy percent of "
            "decisions were unanimous.")


def _world(omit_vote_for_m3=False, alt_reading=False) -> dict:
    m3_actions = [] if omit_vote_for_m3 else ["m3"]
    return {
        "resolution": {"interpretation": "unanimous = all three cast the same vote",
                       "yes_means": "all same", "no_means": "any split",
                       "options": ["Unanimous", "Split"], "resolution_day": "2026-06-25"},
        "causal_thesis": "Three members vote; the chair anchors expectations; dissent by Vega "
                         "is the live risk.",
        "world_boundary": {"included": ["board vote"], "excluded_low_sensitivity": ["media"],
                           "reversal_capable_omissions": []},
        "actors": [
            {"id": "m1", "name": "Ana Diaz", "role": "chair", "aliases": ["Chair Diaz"],
             "authority": ["casts a vote"], "discretion": "decisive",
             "information_channels": ["board"],
             "private_state_variants": [
                 {"variant_id": "anchor", "state": {"beliefs": ["LEAN_HOLD: hold is right"],
                                                    "goals": ["stability"]},
                  "evidence_basis": "Chair Ana Diaz has signaled a hold.",
                  "support": "well_supported"}]},
            {"id": "m1_dupe", "name": "A. Diaz", "role": "chair (duplicate mention)",
             "aliases": ["Ana Diaz"], "authority": ["casts a vote"],
             "discretion": "decisive", "information_channels": ["board"],
             "private_state_variants": []},
            {"id": "m2", "name": "Bo Ruiz", "role": "member", "aliases": [],
             "authority": ["casts a vote"], "discretion": "decisive",
             "information_channels": ["board"],
             "private_state_variants": [
                 {"variant_id": "follower", "state": {"beliefs": ["LEAN_HOLD: follow chair"],
                                                      "goals": ["consensus"]},
                  "evidence_basis": "Member Bo Ruiz is expected to follow the chair.",
                  "support": "well_supported"}]},
            {"id": "m3", "name": "Cy Vega", "role": "member", "aliases": [],
             "authority": ["casts a vote"], "discretion": "decisive",
             "information_channels": ["board"],
             "private_state_variants": [
                 {"variant_id": "dove", "state": {"beliefs": ["LEAN_HOLD: fall in line"],
                                                  "goals": ["credibility"]},
                  "evidence_basis": "unstated", "support": "well_supported"},
                 {"variant_id": "dissenter", "state": {"beliefs": ["LEAN_HIKE: dissent again"],
                                                       "goals": ["signal concern"]},
                  "evidence_basis": "Member Cy Vega dissented at the last meeting and may "
                                    "dissent again.",
                  "support": "plausible"}]},
            {"id": "sec", "name": "Board Secretary", "role": "minutes", "aliases": [],
             "authority": [], "discretion": "ceremonial", "information_channels": ["board"],
             "private_state_variants": []}],
        "institutions": [{"id": "board", "name": "The Board", "aliases": [],
                          "members": ["m1", "m2", "m3"], "decision_rule": "unanimity",
                          "rule_params": {}, "procedure": [
                              {"stage": "meeting", "day": "2026-06-25", "rule": "vote"}]}],
        "mechanisms": [{"id": "tally", "description": "count votes", "kind": "institutional",
                       "deterministic_rule": "unanimity iff one distinct vote",
                        "writes_terminal": True}],
        "temporal_anchors": [{"day": "2026-06-25", "what": "board meeting",
                              "certainty": "scheduled"}],
        "event_types": [{"etype": "meeting_convened", "description": "meeting starts",
                         "observers": ["m1", "m2", "m3"]}],
        "decision_triggers": [
            {"actor_id": "m1", "etype": "meeting_convened", "when_day": "2026-06-25",
             "situation": "cast your vote on the rate decision"},
            {"actor_id": "m2", "etype": "meeting_convened", "when_day": "2026-06-25",
             "situation": "cast your vote on the rate decision"},
            {"actor_id": "m3", "etype": "meeting_convened", "when_day": "2026-06-25",
             "situation": "cast your vote on the rate decision"}],
        "action_templates": [
            {"action_id": "cast_vote", "description": "cast your vote",
             "actor_ids": ["m1", "m2"] + m3_actions, "authority_required": ["casts a vote"],
             "targets": ["board"],
             "effects": [{"kind": "record_vote",
                          "params": {"institution_id": "board",
                                     "options": ["hold", "hike"]}}],
             "emits_events": [], "writes_terminal": True, "validation": ""}],
        "terminal": {"kind": "institution_vote", "institution_id": "board",
                     "decision_rule": "unanimity", "rule_params": {},
                     "yes_when": "all votes equal", "no_when": "votes differ",
                     "written_by_action_ids": ["cast_vote"],
                     "evaluation_day": "2026-06-25"},
        "grounded_rates": [{"quantity": "share of unanimous decisions",
                            "value_range": [0.6, 0.8],
                            "basis_quote": "Historically about seventy percent of decisions "
                                           "were unanimous.",
                            "source_class": "reference_class"}],
        "outside_risks": [],
        "unresolved_assumptions": (
            [{"assumption": "Vega falls in line with the chair",
              "reversal_capable": True,
              "alternative": "Vega dissents and unanimity fails",
              "evidence_conflict": "Member Cy Vega dissented at the last meeting and may "
                                   "dissent again."}] if alt_reading else []),
        "alternative_causal_reading": (
            {"exists": True, "reading": "Vega drives the outcome; the chair follows Vega",
             "evidence_quote": "Member Cy Vega dissented at the last meeting and may dissent "
                               "again.",
             "diverges_at": "m3"} if alt_reading else
            {"exists": False, "reading": "", "evidence_quote": "", "diverges_at": ""})}


class ScriptedLLM:
    """Content-addressed scripted backend. Thread-safe; counts calls by kind."""

    def __init__(self, world: dict):
        self.world = world
        self.calls = {"blueprint": 0, "decision": 0, "repair": 0, "deliberation": 0,
                      "novel": 0, "challenger_delta": 0, "grounding": 0, "state_gen": 0,
                      "other": 0}
        self._lock = threading.Lock()
        self.decision_log = []

    def __call__(self, prompt: str) -> str:
        with self._lock:
            return self._route(prompt)

    def _grounding(self) -> str:
        # counted historical cases (all pre-as_of) — the LLM proposes CASES, never rates
        unanimous = [{"description": f"meeting {i}", "date": f"2025-0{i}-01",
                      "outcome": i <= 6, "source": "record",
                      "basis_quote": "vote", "hierarchy_level": "same_institution"}
                     for i in range(1, 9)]           # 6 of 8 unanimous → ~0.72
        dissent = [{"description": f"m3 meeting {i}", "date": f"2025-0{i}-15",
                    "outcome": i <= 2, "source": "record", "basis_quote": "dissent",
                    "hierarchy_level": "same_role_same_institution"}
                   for i in range(1, 9)]             # m3 dissents 2 of 8 → ~0.28
        return json.dumps({
            "shared_world_conditions": [],
            "actor_state_reference_classes": [
                {"actor_id": "m3", "quantity": "m3 dissents on a hold",
                 "definition": "member m3 casts a dissenting vote", "reference_cases": dissent}],
            "outcome_reference_class": {"quantity": "board vote is unanimous",
                                        "definition": "5-0 vote", "reference_cases": unanimous},
            "institutional_obligations": [
                {"institution_id": "board", "deadline_day": "2026-06-25",
                 "required_participants": ["m1", "m2", "m3"],
                 "allowed_terminal_actions": ["hold", "hike"],
                 "abstention_allowed": False, "waiting_allowed_before_deadline": True,
                 "consequence_of_nonparticipation": "vote fails"}]})

    def _state_gen(self, prompt: str) -> str:
        # propose states per actor — NO numbers (weights come from the counted classes)
        actors = []
        for aid, lean in (("m1", "hold"), ("m2", "hold")):
            actors.append({"actor_id": aid, "states": [
                {"state_id": "aligned", "claim": f"{aid} holds", "beliefs": ["LEAN_HOLD"],
                 "action_if_state": "hold", "reversal_capable": False}]})
        actors.append({"actor_id": "m3", "states": [
            {"state_id": "dove", "claim": "m3 falls in line", "beliefs": ["LEAN_HOLD"],
             "action_if_state": "hold", "reversal_capable": False},
            {"state_id": "dissenter", "claim": "m3 dissents", "beliefs": ["LEAN_HIKE"],
             "action_if_state": "hike", "reversal_capable": True}]})
        return json.dumps({"actors": actors})

    def _route(self, prompt: str) -> str:
        if "assembling the GROUNDING inputs" in prompt:
            self.calls["grounding"] += 1
            return self._grounding()
        if "Propose the genuinely DIFFERENT private realities" in prompt:
            self.calls["state_gen"] += 1
            return self._state_gen(prompt)
        if "Compile the MINIMAL terminal-relevant world" in prompt:
            self.calls["blueprint"] += 1
            return json.dumps(self.world)
        if "deterministic validator rejected" in prompt:
            self.calls["repair"] += 1
            return json.dumps(self.world)          # unhelpful repair (returns same world)
        if "minimal delta" in prompt:
            self.calls["challenger_delta"] += 1
            return json.dumps({"challenger_thesis": "Vega leads",
                               "changed_actor_variants": {"m3": [
                                   {"variant_id": "leader",
                                    "state": {"beliefs": ["LEAN_HIKE: lead dissent"]},
                                    "evidence_basis": "unstated",
                                    "support": "plausible"}]},
                               "changed_assumption": "who anchors"})
        if "SAME person continuing" in prompt:
            self.calls["deliberation"] += 1
            return json.dumps({"reflection_summary": "reconsidered; keeping course",
                               "decision": {"chosen_action": "cast_vote",
                                            "act_or_wait": "act"},
                               "changed": False, "residual_uncertainty": "",
                               "actor_state_update": {}})
        if "no precompiled template" in prompt:
            self.calls["novel"] += 1
            return json.dumps({"action_id": "novel_statement", "description": "statement",
                               "actor_ids": ["m1"], "authority_required": [],
                               "targets": ["m3"],
                               "effects": [{"kind": "send_message",
                                            "params": {"value": "please align"}}],
                               "emits_events": [], "writes_terminal": False,
                               "validation": ""})
        if "YOU ARE:" in prompt:
            self.calls["decision"] += 1
            vote = "hike" if "LEAN_HIKE" in prompt else "hold"
            self.decision_log.append(vote)
            return json.dumps({
                "attention": {"noticed": [], "ignored": []},
                "interpretation": {"what_happened": "the vote is called",
                                   "why_it_matters": "policy", "unresolved_ambiguity": "",
                                   "missing_decisive_fact": ""},
                "considered_actions": ["cast_vote"], "screened_out": [],
                "decision": {"chosen_action": "cast_vote", "act_or_wait": "act",
                             "vote_option": vote, "target": "board",
                             "timing": "immediate", "intended_effect": f"vote {vote}",
                             "revisit_when": ""},
                "decision_summary": f"votes {vote}",
                "actor_state_update": {"beliefs": [], "goals": [], "stances": [],
                                       "pressures": ""}})
        self.calls["other"] += 1
        return "{}"


def _run(world=None, policy_extra=None, llm=None):
    llm = llm or ScriptedLLM(world or _world())
    res = simulate_world(QUESTION, as_of=AS_OF, horizon=HORIZON, llm=llm,
                         evidence=EVIDENCE, seed=0,
                         execution_policy={"lean_v2": {"persistent_cache": False,
                                                       "max_workers": 1,
                                                       **(policy_extra or {})}},
                         execution_profile="lean_v2")
    return res, llm


# ---------------------------------------------------------------------- 1 + 2: preflight
def test_1_preflight_stops_impossible_rollout_before_actors_run():
    res, llm = _run(_world(omit_vote_for_m3=True))
    assert res.simulation_status == "under_modeled"
    assert llm.calls["decision"] == 0                      # NO actor simulation was spent
    assert llm.calls["repair"] == 1                        # exactly one targeted repair
    pf = res.provenance["lean_v2"]["preflight"]
    assert pf["verdict"] == "unanswerable"
    assert any("no actor simulation" in l.lower() for l in res.limitations)
    # best defensible labeled forecast still served — the COUNTED outcome reference class
    # (6 of 8 historical meetings unanimous → beta mean 0.7222), never a qualitative label
    assert abs(res.raw_probability - 0.7222) < 1e-3
    assert res.probability_source == "grounded_reference_prior"


def test_2_valid_rollout_passes_preflight_unchanged():
    res, llm = _run()
    assert res.provenance["lean_v2"]["preflight"]["verdict"] == "answerable"
    # m1/m2 carry coverage-driven unknown-state mass (single ungrounded state) → the run is
    # honestly partially_resolved, never silently completed
    assert res.simulation_status == "partially_resolved"
    assert res.has_forecast()
    # the m3 dove/dissent weights are COUNTED (2 of 8 dissent → 0.28 / 0.72), NOT label
    # midpoints; the grounded prior and simulation-conditional agree at 0.7222
    fd = res.provenance["lean_v2"]["forecast_decomposition"]
    assert abs(fd["grounded_prior"]["p"] - 0.7222) < 1e-3
    assert abs(res.raw_probability - 0.7222) < 1e-2
    law = res.provenance["lean_v2"]["engine_primary"]["grounded_weight_law"]["m3"]
    assert abs(law["dissenter"] - 0.2778) < 1e-3   # counted, not a 0.15-0.45 label range
    assert res.provenance["execution_profile"] == "lean_v2"


# ---------------------------------------------------------------------- 3 + 4 + 5: slice
def test_3_backward_slice_preserves_decisive_actors():
    res, _ = _run()
    kept = res.provenance["lean_v2"]["slice"]["kept_actors"]
    assert {"m1", "m2", "m3"} <= set(kept)


def test_4_backward_slice_removes_duplicate_and_ceremonial_actors():
    res, _ = _run()
    sl = res.provenance["lean_v2"]["slice"]
    assert any(m["removed"] == "m1_dupe" for m in sl["merged"])
    assert any(p["actor_id"] == "sec" and "ceremonial" in p["reason"] for p in sl["pruned"])


def test_5_aliases_do_not_create_duplicate_decision_calls():
    res, llm = _run()
    # distinct contexts: m1(anchor), m2(follower), m3(dove), m3(dissenter) = 4 — the merged
    # duplicate chair adds ZERO calls
    assert llm.calls["decision"] == 4
    man = res.provenance["lean_v2"]["engine_primary"]["decisions"]
    assert man["unique_decision_contexts"] == 4


# ---------------------------------------------------------------------- 6: shared compile
def test_6_shared_compilation_prevents_real_recompilation(tmp_path):
    world = _world()
    llm1 = ScriptedLLM(world)
    policy = {"persistent_cache": True, "persistent_cache_dir": str(tmp_path),
              "max_workers": 1}
    _run(world, policy_extra=policy, llm=llm1)
    assert llm1.calls["blueprint"] == 1
    # a second run over the same dependency vector must NOT recompile the blueprint
    llm2 = ScriptedLLM(world)
    res2, _ = _run(world, policy_extra=policy, llm=llm2)
    assert llm2.calls["blueprint"] == 0
    assert res2.provenance["lean_v2"]["blueprint"]["from_cache"] is True
    cm = res2.provenance["lean_v2"]["compile_cache"]
    assert cm["hits_persistent"] >= 1


# ---------------------------------------------------------------------- 7: info limitation
def test_7_missing_information_decisions_do_not_escalate_without_new_information():
    world = _world()

    class WaitingLLM(ScriptedLLM):
        def _route(self, prompt):
            if "YOU ARE: Cy Vega" in prompt:
                self.calls["decision"] += 1
                return json.dumps({
                    "attention": {"noticed": [], "ignored": []},
                    "interpretation": {"what_happened": "vote called",
                                       "why_it_matters": "policy",
                                       "unresolved_ambiguity": "",
                                       "missing_decisive_fact": "the staff inflation memo"},
                    "considered_actions": [], "screened_out": [],
                    "decision": {"chosen_action": "", "act_or_wait": "wait",
                                 "vote_option": "", "target": "", "timing": "",
                                 "intended_effect": "", "revisit_when": "when memo arrives"},
                    "decision_summary": "waits for the memo",
                    "actor_state_update": {}})
            return super()._route(prompt)

    llm = WaitingLLM(world)
    res, _ = _run(world, llm=llm)
    eng = res.provenance["lean_v2"]["engine_primary"]
    # no deliberation, no staged escalation for the stated missing fact...
    assert llm.calls["deliberation"] == 0
    assert not any("missing" in str(e.get("reason", "")) for e in eng["escalations"])
    # ...and the SAME known absence is never re-asked (m3 decides once per variant, even
    # though the wait scheduled a reconsideration with no new information)
    assert llm.calls["decision"] == 4
    # m3 never voted -> votes missing -> honest partial/unresolved accounting
    assert res.simulation_status in ("partially_resolved", "unresolved")
    assert res.resolution_report["missing_mechanisms"]


# ---------------------------------------------------------------------- 8 + 9: consequences
def test_8_known_consequence_templates_require_no_runtime_compile_call():
    res, llm = _run()
    assert llm.calls["novel"] == 0
    cons = res.provenance["lean_v2"]["consequences"]
    # terminal votes are recorded directly on the tally (mechanical, zero compile); any
    # non-terminal action runs from a precompiled template — never a runtime compile call
    assert cons["novel_compiled"] == []
    stages = res.provenance["lean_v2"]["budget"]["by_stage"]
    assert "consequence_compile" not in stages
    # the vote resolved mechanically with no novel-action compiles
    assert res.provenance["lean_v2"]["budget"]["novel_consequence_compiles"] == 0


def test_9_novel_actions_still_compile_safely():
    world = _world()

    class NovelLLM(ScriptedLLM):
        def _route(self, prompt):
            if "YOU ARE: Ana Diaz" in prompt:
                self.calls["decision"] += 1
                return json.dumps({
                    "attention": {"noticed": [], "ignored": []},
                    "interpretation": {"what_happened": "vote called",
                                       "why_it_matters": "policy",
                                       "unresolved_ambiguity": "",
                                       "missing_decisive_fact": ""},
                    "considered_actions": ["issue a public statement"], "screened_out": [],
                    "decision": {"chosen_action": "issue an unprecedented public statement",
                                 "act_or_wait": "act", "vote_option": "",
                                 "target": "m3", "timing": "immediate",
                                 "intended_effect": "urge alignment", "revisit_when": ""},
                    "decision_summary": "makes a statement",
                    "actor_state_update": {}})
            return super()._route(prompt)

    llm = NovelLLM(world)
    res, _ = _run(world, llm=llm)
    assert llm.calls["novel"] == 1
    cons = res.provenance["lean_v2"]["consequences"]
    assert len(cons["novel_compiled"]) == 1
    assert res.provenance["lean_v2"]["budget"]["novel_consequence_compiles"] == 1


# ---------------------------------------------------------------------- 10-12: coalescing
def _node(nid, w, votes=None, belief="x"):
    n = WeightedWorldNode(node_id=nid, weight=w, day="2026-06-25")
    n.actor_states = {"m1": {"beliefs": [belief]}}
    n.institution_state = {"board": {"votes": dict(votes or {})}}
    return n


def test_10_equivalent_weighted_worlds_merge_and_preserve_total_weight():
    c = WeightedBranchCoalescer()
    a, b = _node("a", 0.4, {"m1": "hold"}), _node("b", 0.6, {"m1": "hold"})
    out = c.coalesce([a, b])
    assert len(out) == 1
    assert abs(out[0].weight - 1.0) < 1e-12
    assert c.merge_log and "a" in c.merge_log[0]["source_particle_ids"] \
        and "b" in c.merge_log[0]["source_particle_ids"]


def test_11_meaningfully_different_worlds_do_not_merge():
    c = WeightedBranchCoalescer()
    out = c.coalesce([_node("a", 0.4, {"m1": "hold"}), _node("b", 0.6, {"m1": "hike"})])
    assert len(out) == 2
    # independent stochastic streams also refuse the merge even with equal visible state
    x, y = _node("x", 0.5, {"m1": "hold"}), _node("y", 0.5, {"m1": "hold"})
    y.independent_stream_tag = "stream_B"
    assert len(c.coalesce([x, y])) == 2


def test_12_merged_worlds_split_correctly_after_divergence():
    c = WeightedBranchCoalescer()
    merged = c.coalesce([_node("a", 0.5, {"m1": "hold"}), _node("b", 0.5, {"m1": "hold"})])[0]

    def _mut(v):
        def m(child):
            child.actor_states["m1"]["beliefs"] = [v]
        return m
    kids = c.split(merged, [("h", 0.5, _mut("dove")), ("k", 0.5, _mut("hawk"))])
    assert len(kids) == 2
    assert abs(sum(k.weight for k in kids) - merged.weight) < 1e-12
    assert kids[0].key() != kids[1].key()
    assert all(merged.node_id in k.ancestry for k in kids)
    with pytest.raises(AssertionError):
        c.split(kids[0], [("bad", 0.7, None)])             # fractions must sum to 1


# ---------------------------------------------------------------------- 13: concurrency
def test_13_sequential_and_concurrent_deterministic_execution_match():
    r1, l1 = _run(policy_extra={"max_workers": 1})
    r2, l2 = _run(policy_extra={"max_workers": 6})
    assert r1.raw_distribution == r2.raw_distribution
    assert r1.raw_probability == r2.raw_probability
    assert l1.calls["decision"] == l2.calls["decision"] == 4
    m1 = r1.provenance["lean_v2"]["engine_primary"]["coalescer"]
    m2 = r2.provenance["lean_v2"]["engine_primary"]["coalescer"]
    assert m1["merges"] == m2["merges"] and m1["truncated_mass"] == m2["truncated_mass"]


# ---------------------------------------------------------------------- 14 + 15: escalation
def test_14_unresolved_result_does_not_trigger_pointless_replicate():
    ok, why = should_replicate(status="unresolved", p_mid=None, unresolved_share=1.0,
                               requested_behavioral_replicates=3,
                               terminal_mechanism_failed=False)
    assert not ok and "missing mechanism" in why
    ok2, why2 = should_replicate(status="completed", p_mid=0.6, unresolved_share=0.0,
                                 requested_behavioral_replicates=3,
                                 terminal_mechanism_failed=True)
    assert not ok2 and "repeats the same failure" in why2
    # the live run records the policy verdict
    res, _ = _run(_world(omit_vote_for_m3=True))
    assert res.provenance["lean_v2"].get("replicate_policy") is None \
        or res.provenance["lean_v2"]["replicate_policy"]["ran"] is False


def test_15_genuinely_unstable_scoreable_result_may_still_escalate():
    ok, why = should_replicate(status="completed", p_mid=0.52, unresolved_share=0.0,
                               requested_behavioral_replicates=2,
                               terminal_mechanism_failed=False)
    assert ok and "genuine" in why
    # and the CHALLENGER escalation fires on a near-threshold result with a verified
    # evidence-supported alternative reading — then actually runs
    res, llm = _run(_world(alt_reading=True))
    ch = res.provenance["lean_v2"]["challenger"]
    assert ch["triggered"] is True
    assert any("disputed_reversal_assumption" in t for t in ch["triggers"])
    assert llm.calls["challenger_delta"] == 1
    assert res.structural_disagreement is not None
    # localized fork: the challenger reuses every unchanged decision context — only m3's
    # replacement variant costs a call (4 primary + 1 challenger)
    assert llm.calls["decision"] == 5
    bud = res.provenance["lean_v2"]["budget"]
    assert bud["structural_models"] == 2
