"""Simulation-completion focused tests — the 25 proofs of the completion fix.

The through-line: private-state uncertainty is an INPUT to simulation (branch it, weight it,
simulate it), never a reason to stop a world; a broken terminal pathway triggers diagnosis
and targeted repair, never `missing_mechanism = 1.0`; rollout starts only after the world is
PROVEN able to reach its measured outcome; completed simulations map exactly to the terminal
answer; and every recovery attempt is recorded with its outcome."""
from __future__ import annotations

import json

import pytest

from swm.world_model_v2.lean_v2.blueprint import ConsumerWorldBlueprint
from swm.world_model_v2.lean_v2.budget import BudgetLedger, ConsumerComputeBudget
from swm.world_model_v2.lean_v2.compile_cache import CompilationCache
from swm.world_model_v2.lean_v2.mechanisms import (diagnose_missing_mechanism,
                                                   evaluate_bounded_process,
                                                   extract_observations, recover_mechanism,
                                                   validate_mechanism)
from swm.world_model_v2.lean_v2.readiness import (CANONICAL_TERMINAL_KEY,
                                                  build_terminal_mapping,
                                                  canonicalize_terminal_writers,
                                                  pure_terminal_outcome,
                                                  simulation_readiness, terminal_round_trip)
from swm.world_model_v2.lean_v2.state_completeness import (
    MAX_ACTOR_RESIDUAL, actor_evidence_slice, ensure_actor_state_completeness,
    feasible_options_for, reversal_focused_search)
from swm.world_model_v2.lean_v2.states import (ActorStateHypothesis,
                                               ActorStatePosteriorEngine,
                                               generate_actor_states)

AS_OF = "2026-06-01"


# ---------------------------------------------------------------------------- fixtures
def _vote_bp(members=("m1", "m2", "m3"), options=("hold", "hike")) -> ConsumerWorldBlueprint:
    return ConsumerWorldBlueprint(
        resolution={"interpretation": "will the board vote unanimously to hold?",
                    "options": ["Yes", "No"]},
        causal_thesis="the board decides",
        actors=[{"id": m, "name": m.upper(), "role": "board member",
                 "authority": ["vote"], "private_state_variants": []} for m in members],
        institutions=[{"id": "board", "name": "The Board", "members": list(members),
                       "decision_rule": "unanimity"}],
        decision_triggers=[{"actor_id": m, "when_day": "2026-06-10",
                            "situation": "the meeting is called", "etype": "meeting"}
                           for m in members],
        action_templates=[{"action_id": "cast_vote", "description": "cast your vote",
                           "actor_ids": list(members), "writes_terminal": True,
                           "effects": [{"kind": "record_vote",
                                        "params": {"institution_id": "board",
                                                   "options": list(options)}}]}],
        terminal={"kind": "institution_vote", "institution_id": "board",
                  "decision_rule": "unanimity", "evaluation_day": "2026-06-10"})


def _predicate_bp(threshold_text="the daily transit count reaches 80 or more on any day"
                  ) -> ConsumerWorldBlueprint:
    return ConsumerWorldBlueprint(
        resolution={"interpretation": f"resolves YES if {threshold_text}",
                    "options": ["Yes", "No"]},
        causal_thesis="operators decide; transits result",
        actors=[{"id": "op1", "name": "Operator", "role": "fleet operator",
                 "authority": ["route"], "private_state_variants": []}],
        institutions=[],
        decision_triggers=[{"actor_id": "op1", "when_day": "2026-06-05",
                            "situation": "routing decision", "etype": "routing"}],
        action_templates=[{"action_id": "declare_outcome", "description": "mark occurrence",
                           "actor_ids": ["op1"], "writes_terminal": True,
                           "effects": [{"kind": "set_state",
                                        "params": {"key": "some_other_key",
                                                   "value": "true"}}]}],
        terminal={"kind": "state_predicate", "yes_when": threshold_text,
                  "no_when": "it does not", "evaluation_day": "2026-06-20"})


class OneShotGateway:
    """Scripted gateway: returns queued replies per stage; records every call."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []
        self.backend_fingerprint = "scripted"

    def call(self, stage, prompt):
        self.calls.append({"stage": stage, "prompt": prompt})
        if self.replies:
            return self.replies.pop(0)
        return "{}"


class FailingGateway(OneShotGateway):
    def call(self, stage, prompt):
        self.calls.append({"stage": stage, "prompt": prompt})
        raise RuntimeError("provider down")


def _ledger():
    return BudgetLedger(ConsumerComputeBudget())


def _complete(bp, states_by_actor, gateway=None, grounding=None, evidence=""):
    return ensure_actor_state_completeness(
        bp=bp, consequential_actors=[a["id"] for a in bp.actors],
        states_by_actor=states_by_actor, grounding=grounding or {},
        evidence_text=evidence, hard_evidence_ids=set(),
        gateway=gateway or OneShotGateway(), budget_ledger=_ledger())


def _h(aid, sid, action="", reversal=False, **kw):
    return ActorStateHypothesis(actor_id=aid, state_id=sid, claim=f"{aid} {sid}",
                                action_if_state=action, reversal_capable=reversal, **kw)


# ============================================================ 1-8: the completeness ladder
# 1 — an empty state set NEVER survives: the ladder ends in a decision-spanning basis
def test_1_empty_state_set_recovers_through_ladder():
    bp = _vote_bp()
    completed, rep = _complete(bp, {})
    for m in ("m1", "m2", "m3"):
        assert completed[m], "every consequential actor must exit with states"
        assert rep.actors[m].final_state_count >= 2
        assert rep.actors[m].final_source in ("regenerated", "fallback", "repaired")
    assert rep.ok
    assert rep.empty_sets_detected == 3


# 2 — attempt 1: states filed under a name/alias variant are recovered with ZERO calls
def test_2_alias_misfiled_states_repaired_deterministically():
    bp = _vote_bp()
    bp.actors[0]["aliases"] = ["Chair Vega"]
    hyps = [_h("Chair Vega", "hawk", action="hike", reversal=True),
            _h("Chair Vega", "dove", action="hold")]
    for x in hyps:
        x.actor_id = "Chair Vega"
    completed, rep = _complete(bp, {"Chair Vega": hyps})
    assert {s.state_id for s in completed["m1"]} == {"hawk", "dove"}
    a1 = rep.actors["m1"].attempts[0]
    assert a1["action"] == "deterministic_alias_parse_repair" and a1["calls"] == 0
    assert rep.actors["m1"].final_source == "repaired"


# 3 — attempt 2: targeted regeneration asks for EXACTLY the missing actor, never healthy ones
def test_3_targeted_regen_only_for_missing_actor():
    bp = _vote_bp()
    good = [_h("m2", "hawk", action="hike", reversal=True), _h("m2", "dove", action="hold")]
    regen = json.dumps({"actor_id": "m1", "states": [
        {"state_id": "regen_hawk", "claim": "m1 leans hike", "action_if_state": "hike",
         "reversal_capable": True},
        {"state_id": "regen_dove", "claim": "m1 leans hold", "action_if_state": "hold"}]})
    gw = OneShotGateway([regen, regen])   # m1 + m3 each get one targeted call
    completed, rep = _complete(bp, {"m2": good}, gateway=gw)
    assert {s.state_id for s in completed["m1"]} == {"regen_hawk", "regen_dove"}
    assert rep.actors["m2"].attempts == []          # the healthy actor was never touched
    for c in gw.calls:
        assert "m2" not in c["prompt"].split("Actor:")[1].split("\n")[0]


# 4 — attempt 3: the actor-local evidence slice is deterministic and actor-scoped
def test_4_actor_evidence_slice_is_actor_local():
    ev = ("Maria Vega signalled a pause last week. The weather was mild. "
          "Vega's staff drafted a dissent. Someone else did something.")
    sl = actor_evidence_slice(ev, {"id": "m1", "name": "Maria Vega", "role": "governor",
                                   "aliases": ["Vega"]})
    assert "Vega" in sl and "weather" not in sl


# 5 — attempt 4: the fallback basis SPANS the feasible decision space
def test_5_fallback_basis_spans_options():
    bp = _vote_bp()
    completed, rep = _complete(bp, {}, gateway=FailingGateway())
    acts = {s.action_if_state for s in completed["m1"] if s.action_if_state}
    assert acts == {"hold", "hike"}                 # every option has a represented state
    assert any(s.state_id == "fallback_internally_conflicted" for s in completed["m1"])
    # provider failures were RECORDED and the ladder CONTINUED — never unknown mass
    outcomes = [a["outcome"] for a in rep.actors["m1"].attempts]
    assert any("provider_failure" in o for o in outcomes)
    assert rep.actors["m1"].final_source == "fallback"


# 6 — residual law: a decision-spanning basis has residual 0; nothing exceeds the cap
def test_6_residuals_bounded_and_zero_for_spanning_basis():
    bp = _vote_bp()
    completed, rep = _complete(bp, {})
    for m in ("m1", "m2", "m3"):
        assert rep.actors[m].residual_r == 0.0      # spanning basis by construction
    assert 0.0 <= rep.joint_residual_bound() <= 1.0
    r = rep.actors["m1"]
    r.residual_r = 0.9                              # even a corrupt record is capped
    assert rep.joint_residual_bound() <= 1 - (1 - MAX_ACTOR_RESIDUAL) ** 3 + 1e-9


# 7 — the joint bound is 1 - prod(1-r_a): a BOUND for interval widening, not branch mass
def test_7_joint_residual_bound_formula():
    bp = _vote_bp()
    completed, rep = _complete(bp, {})
    rep.actors["m1"].residual_r = 0.1
    rep.actors["m2"].residual_r = 0.2
    rep.actors["m3"].residual_r = 0.0
    assert abs(rep.joint_residual_bound() - (1 - 0.9 * 0.8)) < 1e-9


# 8 — the engine refuses LOUDLY when the invariant is bypassed for a consequential actor
def test_8_engine_hard_fails_on_bypassed_invariant():
    from swm.world_model_v2.lean_v2.consequences import TemplateExecutor
    from swm.world_model_v2.lean_v2.engine import WaveEngine
    bp = _vote_bp()
    bp.actors[0]["private_state_variants"] = [{"variant_id": "v1", "state": {}}]
    with pytest.raises(RuntimeError, match="completeness invariant bypassed"):
        WaveEngine(bp=bp, kept_actors=["m1"], promotable=[],
                   executor=TemplateExecutor({}, bp), gateway=OneShotGateway(),
                   budget_ledger=_ledger(), compile_cache=CompilationCache(persist=False),
                   grounded_weights={"m1": {"mid": {}}},          # zero represented states
                   consequential_actors=["m1"])


# ==================================================== 9-11: reversal search + posteriors
# 9 — the reversal-focused search ADDS proposed omitted states to the completed set
def test_9_reversal_search_adds_states():
    bp = _vote_bp()
    completed = {"m1": [_h("m1", "steady", action="hold")]}
    reply = json.dumps({"proposals": [{"actor_id": "m1", "state_id": "quiet_dissenter",
                                       "claim": "m1 privately favors a hike",
                                       "action_if_state": "hike",
                                       "basis": "dissented in 2019", "none_found": False}]})
    rec = reversal_focused_search(bp=bp, completed=completed, evidence_text="",
                                  gateway=OneShotGateway([reply]), budget_ledger=_ledger())
    assert rec["added"] == 1
    assert any(s.state_id == "quiet_dissenter" and s.reversal_capable
               for s in completed["m1"])


# 10 — represented states carry the FULL branch mass (weights sum to 1, no unknown worlds)
def test_10_represented_states_carry_full_mass():
    eng = ActorStatePosteriorEngine({})
    rows, residual, prov = eng.weight_actor_states(
        "a", [_h("a", "s1", action="hold"), _h("a", "s2", action="hike", reversal=True)])
    assert abs(sum(r.mid for r in rows) - 1.0) < 1e-9
    assert residual <= MAX_ACTOR_RESIDUAL


# 11 — an empty survivor set returns empty rows (a readiness failure, never silent mass)
def test_11_empty_survivors_flagged_not_massed():
    eng = ActorStatePosteriorEngine({})
    rows, residual, prov = eng.weight_actor_states("a", [])
    assert rows == [] and prov.get("empty_state_set") is True


# ============================================================ 12-13: cache correctness
# 12 — an empty/unparseable generation result is NEVER cached
def test_12_empty_generation_never_cached():
    cache = CompilationCache(persist=False)
    gw = OneShotGateway(["not json at all"])
    out, rej, meta = generate_actor_states(
        question="q", as_of=AS_OF, evidence_text="e",
        actors=[{"id": "m1", "role": "r"}], shared_condition_ids=[],
        gateway=gw, cache=cache)
    assert out == {} and meta["from_cache"] is False
    assert cache.stores == 0                        # nothing empty was stored
    gw2 = OneShotGateway([json.dumps({"actors": []})])
    out2, _rej2, _m2 = generate_actor_states(
        question="q", as_of=AS_OF, evidence_text="e",
        actors=[{"id": "m1", "role": "r"}], shared_condition_ids=[],
        gateway=gw2, cache=cache)
    assert out2 == {} and cache.stores == 0


# 13 — a cached artifact proven inadequate is INVALIDATED (both layers, recorded)
def test_13_inadequate_cached_artifact_invalidated():
    cache = CompilationCache(persist=False)
    deps = {"k": "v"}
    cache.put("actor_state_generation", deps, "stale")
    assert cache.get("actor_state_generation", deps) == "stale"
    assert cache.invalidate("actor_state_generation", deps) is True
    assert cache.get("actor_state_generation", deps) is None
    assert any(e["outcome"] == "invalidated" for e in cache.events)


# ======================================================= 14-18: readiness + round-trip
# 14 — a complete world is READY (round-trip proven)
def test_14_ready_world_passes_gate():
    from swm.world_model_v2.lean_v2.consequences import TemplateExecutor, precompile_templates
    from swm.world_model_v2.lean_v2.obligations import build_obligations
    bp = _vote_bp()
    completed, _rep = _complete(bp, {})
    gw = {m: {"mid": {s.state_id: 1.0 / len(completed[m]) for s in completed[m]}}
          for m in completed}
    for m in gw:
        z = sum(gw[m]["mid"].values())
        gw[m]["mid"] = {k: v / z for k, v in gw[m]["mid"].items()}
    ex = TemplateExecutor(precompile_templates(bp, CompilationCache(persist=False)), bp)
    rep = simulation_readiness(bp=bp, consequential_actors=list(completed),
                               completed_states=completed, grounded_weights=gw,
                               obligations=build_obligations(bp, {}), executor=ex,
                               shared_combos=[({}, 1.0)])
    assert rep.verdict == "ready"
    assert rep.round_trip["ok"] is True


# 15 — the synthetic round-trip proves YES→1 and NO→0 through the LIVE path
def test_15_terminal_round_trip_yes1_no0():
    bp = _vote_bp()
    rt = terminal_round_trip(bp, obligations={})
    assert rt["ok"] is True
    got = {c["case"]: c for c in rt["checks"]}
    assert got["known_yes"]["recovery_p"] == 1.0
    assert got["known_no"]["recovery_p"] == 0.0
    assert all(c["evaluator_ok"] and c["mapping_ok"] for c in rt["checks"])


# 16 — the visionOS class at the source: mismatched terminal writer keys are canonicalized
def test_16_terminal_writer_canonicalization():
    bp = _predicate_bp()
    rec = canonicalize_terminal_writers(bp)
    assert rec["needed"] is True
    assert rec["rewritten"][0]["old_key"] == "some_other_key"
    eff = bp.action_templates[0]["effects"][0]["params"]
    assert eff["key"] == CANONICAL_TERMINAL_KEY
    # and the pure law reads exactly that key
    out = pure_terminal_outcome(bp, world_state={CANONICAL_TERMINAL_KEY: True})
    assert out["resolved"] and out["outcome"] == "YES"


# 17 — a NON-substantive unanimity break: abstention is executed, resolves NO (not unknown)
def test_17_pure_law_abstention_resolves():
    bp = _vote_bp()
    votes = {"m1": "hold", "m2": "hold", "m3": "__abstain__"}
    out = pure_terminal_outcome(bp, votes=votes)
    assert out["resolved"] is True and out["outcome"] == "NO"
    assert out["detail"]["non_substantive"] == ["m3"]


# 18 — terminal mapping tolerates label spelling/casing (aliases), never overlaps
def test_18_terminal_mapping_aliases():
    bp = _vote_bp()
    m = build_terminal_mapping(bp)
    assert m.canonical("YES") == "YES" and m.canonical("yes") == "YES"
    assert m.canonical("No") == "NO"
    assert not (m.aliases_yes & m.aliases_no)


# ===================================================== 19-23: missing-mechanism recovery
# 19 — diagnosis parses threshold / comparator / aggregation from the resolution text
def test_19_mechanism_diagnosis_parses_threshold():
    bp = _predicate_bp("the daily transit count reaches 80 or more on any day")
    d = diagnose_missing_mechanism(bp)
    assert d.threshold == 80.0 and d.comparator == ">=" and d.aggregation == "any_day"


# 20 — observation extraction: values must be VERBATIM and dates pre-as_of (leakage killed)
def test_20_observation_extraction_leakage_checked():
    bp = _predicate_bp()
    d = diagnose_missing_mechanism(bp)
    reply = json.dumps({"observations": [
        {"obs_id": "o1", "value": 82, "unit": "transits", "date": "2026-05-20",
         "condition_tag": "normal_operations", "basis_quote": "82 transits recorded"},
        {"obs_id": "o2", "value": 999, "unit": "transits", "date": "2026-05-21",
         "condition_tag": "normal_operations", "basis_quote": "not in evidence"},
        {"obs_id": "o3", "value": 60, "unit": "transits", "date": "2026-07-01",
         "condition_tag": "crisis", "basis_quote": "60 transits"}]})
    obs, rec = extract_observations(
        d, evidence_text="On 2026-05-20 there were 82 transits. Later 60 transits.",
        as_of=AS_OF, gateway=OneShotGateway([reply]),
        cache=CompilationCache(persist=False), budget_ledger=_ledger())
    assert [o["obs_id"] for o in obs] == ["o1"]
    whys = {r["why"] for r in rec["rejected"]}
    assert any("not verbatim" in w for w in whys)
    assert any("leakage" in w for w in whys)


# 21 — all rates computed by CODE (min/central/max); regimes select by world condition
def test_21_bounded_process_code_computed_and_regime_selected():
    mech = {"kind": "bounded_numeric_process", "variable": "transits", "threshold": 80.0,
            "comparator": ">=", "aggregation": "any_day",
            "min_rate": 60.0, "central_rate": 75.0, "max_rate": 85.0,
            "observations": [{"obs_id": "o1", "value": 85.0}],
            "regimes": [{"condition_key": "blockade", "condition_value": "active",
                         "min_rate": 10.0, "central_rate": 20.0, "max_rate": 30.0,
                         "n_observations": 2, "observation_ids": ["o1"]}]}
    hit = evaluate_bounded_process(mech, world_conditions={})
    assert hit["outcome"] == "YES"                  # any_day reads the MAX (85 ≥ 80)
    blocked = evaluate_bounded_process(mech, world_conditions={"blockade": "active"})
    assert blocked["outcome"] == "NO"               # regime max 30 < 80
    straddle = evaluate_bounded_process(
        {**mech, "regimes": []}, world_conditions={})
    assert straddle["detail"]["straddle"] is True   # min 60 < 80 ≤ max 85 — disclosed


# 22 — the full ladder builds a VALIDATED mechanism from sealed evidence (Hormuz class)
def test_22_recover_mechanism_full_ladder():
    bp = _predicate_bp()
    reply = json.dumps({"observations": [
        {"obs_id": "o1", "value": 82, "unit": "t", "date": "2026-05-20",
         "condition_tag": "normal_operations", "basis_quote": "82 transits"},
        {"obs_id": "o2", "value": 75, "unit": "t", "date": "2026-05-10",
         "condition_tag": "normal_operations", "basis_quote": "75 transits"},
        {"obs_id": "o3", "value": 12, "unit": "t", "date": "2026-04-01",
         "condition_tag": "blockade_active", "basis_quote": "12 transits"}]})
    mech, man = recover_mechanism(
        bp, cause="state_predicate_not_mechanically_bound",
        evidence_text="counts: 82 transits, 75 transits, then 12 transits under blockade",
        as_of=AS_OF, gateway=OneShotGateway([reply]),
        cache=CompilationCache(persist=False), budget_ledger=_ledger(),
        world_condition_keys=["blockade_active"])
    assert mech is not None and man["validated"] is True
    assert mech["min_rate"] == 12.0 and mech["max_rate"] == 82.0
    ok, vrep = validate_mechanism(mech, bp)
    assert ok and all(c["ok"] for c in vrep["checks"])
    # the pure terminal law consumes it
    out = pure_terminal_outcome(bp, world_state={}, mechanism=mech,
                                world_conditions={"blockade_active": "blockade_active"})
    assert out["resolved"] is True


# 23 — when NO defensible mechanism exists, the failure carries its PROOF (never silent)
def test_23_mechanism_failure_carries_proof():
    bp = _predicate_bp("something qualitative happens with no number")
    mech, man = recover_mechanism(
        bp, cause="x", evidence_text="no numbers here", as_of=AS_OF,
        gateway=OneShotGateway(), cache=CompilationCache(persist=False),
        budget_ledger=_ledger(), world_condition_keys=[])
    assert mech is None
    assert "no parseable numeric threshold" in man["failure_proof"] \
        or "threshold" in man["failure_proof"]
    bp2 = _predicate_bp()
    mech2, man2 = recover_mechanism(
        bp2, cause="x", evidence_text="qualitative only",
        as_of=AS_OF, gateway=OneShotGateway([json.dumps({"observations": [],
                                                         "none_found": True})]),
        cache=CompilationCache(persist=False), budget_ledger=_ledger(),
        world_condition_keys=[])
    assert mech2 is None and "cannot be built without inventing numbers" \
        in man2["failure_proof"]


# ================================================ 24-25: deadline completion + audit law
# 24 — deadline-forced completion drives missing votes to the terminal (bounded, audited)
def test_24_completion_pass_resolves_missing_votes():
    from swm.world_model_v2.lean_v2.consequences import TemplateExecutor, precompile_templates
    from swm.world_model_v2.lean_v2.engine import WaveEngine
    from swm.world_model_v2.lean_v2.obligations import build_obligations

    class VoteOnlyAtDeadline:
        """Waits at the meeting; votes 'hold' only when the mandatory menu arrives."""
        backend_fingerprint = "scripted"

        def __init__(self):
            self.calls = []

        def call(self, stage, prompt):
            self.calls.append(stage)
            if "cast_vote: choose exactly one" in prompt:
                return json.dumps({
                    "attention": {"noticed": [], "ignored": []},
                    "interpretation": {"what_happened": "deadline", "why_it_matters": "",
                                       "unresolved_ambiguity": "",
                                       "missing_decisive_fact": ""},
                    "considered_actions": ["vote"], "screened_out": [],
                    "decision": {"chosen_action": "cast_vote", "act_or_wait": "act",
                                 "vote_option": "hold", "target": "", "timing": "immediate",
                                 "intended_effect": "vote hold", "revisit_when": ""},
                    "decision_summary": "votes hold", "actor_state_update": {}})
            return json.dumps({
                "attention": {"noticed": [], "ignored": []},
                "interpretation": {"what_happened": "meeting", "why_it_matters": "",
                                   "unresolved_ambiguity": "",
                                   "missing_decisive_fact": ""},
                "considered_actions": [], "screened_out": [],
                "decision": {"chosen_action": "", "act_or_wait": "wait",
                             "vote_option": "", "target": "", "timing": "",
                             "intended_effect": "", "revisit_when": "later"},
                "decision_summary": "waits", "actor_state_update": {}})

    bp = _vote_bp()
    completed, _rep = _complete(bp, {
        m: [_h(m, "steady", action="hold"), _h(m, "torn", action="hike", reversal=True)]
        for m in ("m1", "m2", "m3")})
    for a in bp.actors:
        a["private_state_variants"] = [s.to_variant() for s in completed[a["id"]]]
    gwts = {m: {"mid": {"steady": 0.7, "torn": 0.3}} for m in completed}
    cache = CompilationCache(persist=False)
    ex = TemplateExecutor(precompile_templates(bp, cache), bp)
    eng = WaveEngine(bp=bp, kept_actors=["m1", "m2", "m3"], promotable=[],
                     executor=ex, gateway=VoteOnlyAtDeadline(), budget_ledger=_ledger(),
                     compile_cache=cache, grounded_weights=gwts,
                     obligations=build_obligations(bp, {}),
                     consequential_actors=["m1", "m2", "m3"])
    res = eng.run(as_of=AS_OF, horizon="2026-06-15")
    # every world reached the exact terminal outcome — zero unresolved mass
    assert res.unresolved_mass == 0.0
    assert abs(res.yes_mass + res.no_mass - 1.0) < 1e-6
    assert res.completion_audit["rounds"], "the completion pass must have run"
    assert res.completion_audit["rounds"][0]["reopened_decisions"] > 0


# 25 — resume_with_mechanism re-evaluates ONLY unresolved worlds; finalize is idempotent
def test_25_resume_with_mechanism_completes_unresolved_only():
    from swm.world_model_v2.lean_v2.consequences import TemplateExecutor, precompile_templates
    from swm.world_model_v2.lean_v2.engine import WaveEngine
    bp = _predicate_bp()
    bp.action_templates = []                        # no writer, no mechanism → unresolved
    cache = CompilationCache(persist=False)
    ex = TemplateExecutor(precompile_templates(bp, cache), bp)
    eng = WaveEngine(bp=bp, kept_actors=[], promotable=[], executor=ex,
                     gateway=OneShotGateway(), budget_ledger=_ledger(),
                     compile_cache=cache, grounded_weights={}, obligations={},
                     consequential_actors=[])
    res = eng.run(as_of=AS_OF, horizon="2026-06-20")
    assert res.unresolved_mass == 1.0
    assert "state_predicate_not_mechanically_bound" in res.unresolved_reasons
    mech = {"kind": "bounded_numeric_process", "variable": "transits", "threshold": 80.0,
            "comparator": ">=", "aggregation": "any_day", "min_rate": 81.0,
            "central_rate": 82.0, "max_rate": 85.0,
            "observations": [{"obs_id": "o1", "value": 85.0}], "regimes": []}
    res2 = eng.resume_with_mechanism(mech)
    # the recovered mechanism resolves the formerly-unresolved mass — and the totals still
    # conserve exactly (idempotent re-finalize, no double counting)
    assert res2.unresolved_mass == 0.0
    assert abs(res2.yes_mass + res2.no_mass - 1.0) < 1e-6
    assert res2.yes_mass == 1.0                     # max 85 ≥ 80 on an any-day question
    assert res2.completion_audit["post_run_mechanism_resume"][0]["re_evaluated"] >= 1
