"""Tier A foundational fixes (gap audit): provenance honesty, loud failures, closed rule kinds,
readout binding, option-space coverage, policy-path repair, endogenous events, correlation ranges,
rejuvenation ranges, executable-mechanism gating."""
import json
import random

import pytest

import swm.world_model_v2.actor_cognition  # registers the responsiveness/workload entity extension
from swm.world_model_v2.compiler import CompileAbstention, compile_world
from swm.world_model_v2.contracts import OutcomeContract
from swm.world_model_v2.events import Event, EventQueue, register_event_type
from swm.world_model_v2.init_state import CorrelationRule, InitialStateModel, LatentVariableRecord
from swm.world_model_v2.institutions import EXECUTABLE_RULE_KINDS, Rule, RuleSystem
from swm.world_model_v2.materialize import (MaterializeAbstention, build_world, check_readout_binding,
                                            operators_from_plan, run_from_plan)
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldBranch, WorldState
from swm.world_model_v2.transitions import (AgentDecisionOperator, FittedDecisionOperator,
                                            StateDelta, TransitionOperator, TransitionProposal)

T0 = 1.0e9


def _world(entities=("alice",)):
    w = WorldState(world_id="t", branch_id="root", clock=SimulationClock(now=T0, as_of=T0))
    for e in entities:
        ent = Entity(identity=e)
        ent.set("current_action", F(None, status="assumed"))
        w.entities[e] = ent
    return w


def _plan_dict(mechs, readout="q_out", entities=None, institutions=None):
    return {"outcome": {"family": "binary", "options": ["yes", "no"], "resolution_rule": "r",
                        "readout_var": readout},
            "entities": entities if entities is not None else [{"id": "alice", "type": "person",
                                                                "fields": {"goals": "win"}}],
            "institutions": institutions or [],
            "quantities": [{"name": "q_out", "qtype": "q_out", "value": None}],
            "scheduled_events": [{"etype": "decision_opportunity", "at": "2001-09-10",
                                  "participants": ["alice"], "payload": {}}],
            "mechanisms": mechs, "sensitivity": {}, "latents": [],
            "domain": "organizational_decision", "population_kind": "organizational",
            "time_scale": "days", "available_data": []}


def _compile(plan_dict):
    return compile_world("test?", llm=lambda p: json.dumps(plan_dict), evidence="",
                         as_of="2001-09-09", horizon="2001-09-20")


# ---------------------------------------------------------------- provenance honesty
def test_compiler_proposals_are_not_stamped_observed():
    plan = _compile(_plan_dict(["agent_decision"]))
    w = build_world(plan)
    sf = w.entities["alice"].get("goals")
    assert sf.prov.status == "inferred"                     # was fabricated as "observed" pre-audit
    assert sf.prov.method.startswith("compiler:proposal:")


def test_dropped_fields_are_recorded_not_silent():
    d = _plan_dict(["agent_decision"])
    d["entities"][0]["fields"]["not_a_real_field"] = "x"
    d["relations"] = [{"src": "alice", "rel": "made_up_relation", "dst": "bob"}]
    plan = _compile(d)
    w = build_world(plan)
    kinds = {o["kind"] for o in w.omissions}
    # non-schema field is KEPT in the typed latent_state namespace (not silently dropped) and recorded;
    # an unregistered relation IS dropped and recorded loudly
    assert "entity_field_routed_to_latent_state" in kinds and "relation" in kinds
    assert w.entities["alice"].get("latent_state", key="not_a_real_field") is not None


# ---------------------------------------------------------------- closed rule kinds
def test_unknown_rule_kind_fails_closed_and_is_omitted():
    r = Rule(rule_id="x", kind="vibes_based_governance", params={})
    ok, why = r.check(_world(), {"actor": "alice", "type": "anything"})
    assert not ok and "no executable semantics" in why
    d = _plan_dict(["agent_decision"],
                   institutions=[{"id": "board", "rules": [{"kind": "vibes_based_governance"}]}])
    plan = _compile(d)
    w = build_world(plan)
    assert any(o["kind"] == "institutional_rule" for o in w.omissions)
    assert len(w.institutions["board"].rules) == 0


def test_deadline_rule_with_bad_ts_fails_closed():
    r = Rule(rule_id="d", kind="deadline", params={"actions": ["sign"], "by_ts": "not-a-date"})
    ok, why = r.check(_world(), {"actor": "alice", "type": "sign"})
    assert not ok and "failing closed" in why


def test_capacity_and_quorum_kinds_execute():
    w = _world(("a", "b"))
    rs = RuleSystem(institution_id="i", rules=[
        Rule(rule_id="q", kind="quorum", params={"actions": ["vote"], "members": ["a", "b"],
                                                 "min_present": 3})])
    ok, reasons = rs.validate_action(w, {"actor": "a", "type": "vote"})
    assert not ok and "quorum" in reasons[0]


# ---------------------------------------------------------------- executable-mechanism gating
def test_operatorless_mechanism_rejected_but_fallback_still_forecasts():
    """MIGRATION (no-abstention): an unported/experimental mechanism is STILL rejected (never fabricated),
    but 'no production-eligible mechanism' no longer means 'no forecast'. The generic broad-prior resolver
    is attached as the terminal safety net so compilation completes and the plan runs."""
    plan_only_bad = _compile(_plan_dict(["whipcount_binomial"]))   # experimental/unported → not executable
    assert any(r["id"] == "whipcount_binomial" for r in plan_only_bad.rejected_mechanisms)
    assert any(m["mech_id"] == "generic_outcome_prior" for m in plan_only_bad.accepted_mechanisms)
    assert plan_only_bad.support_grade in ("exploratory", "highly_speculative")
    result, _ = run_from_plan(plan_only_bad, n_particles=6, seed=1)
    assert result["distribution"] and result["n_deltas"] > 0       # forecasts, never abstains

    plan = _compile(_plan_dict(["whipcount_binomial", "agent_decision"]))
    assert any(r["id"] == "whipcount_binomial" for r in plan.rejected_mechanisms)
    # the domain mechanism is accepted AND the generic resolver is appended as the terminal safety net
    accepted_ids = [m["mech_id"] for m in plan.accepted_mechanisms]
    # Phase 4 migration: the legacy semantic ID is compiled into the actor-view,
    # feasibility, calibrated-policy, event, and StateDelta production operator.
    assert accepted_ids == ["production_actor_policy", "generic_outcome_prior"]


def test_poisson_arrival_is_now_executable():
    d = _plan_dict(["poisson_arrival"])
    d["scheduled_events"] = [{"etype": "external_shock", "at": "2001-09-10", "participants": [],
                              "payload": {"outcome_var": "q_out"}}]
    plan = _compile(d)
    result, branches = run_from_plan(plan, n_particles=4, seed=1)
    assert result["n_deltas"] >= 4                          # the shock fired and wrote the quantity
    assert branches[0].world.quantities["q_out"].value is True


# ---------------------------------------------------------------- readout binding + coverage
def test_dangling_readout_is_repaired_not_aborted():
    """MIGRATION (no-abstention): a readout pointing at an entity.field no mechanism writes used to abort
    (MaterializeAbstention). It is now REPAIRED at compile time to the canonical `outcome` quantity that the
    terminal resolver writes, so the run completes and the readout binds — a technical unbindable readout is
    an engineering failure (CompilerExecutionError), never a silent forecast refusal."""
    plan = _compile(_plan_dict(["agent_decision"], readout="ghost_entity.mood"))
    assert plan.outcome_contract.readout_var == "outcome"          # repaired to canonical quantity
    assert plan.provenance["readout_repaired"] is True
    result, branches = run_from_plan(plan, n_particles=4, seed=1)
    assert result["distribution"] and result["readout"] == "terminal_states"


# ---------------------------------------------------------------- binary option polarity (lean convention)
def test_binary_option_polarity_is_order_invariant():
    """The generic resolver applies outcome_lean toward options[0] and the projection reports P(options[0]);
    both require options[0] to be the AFFIRMATIVE answer. LLMs order options inconsistently, so the compiler
    normalizes by lexical negativity. A 'yes'-leaning question must give HIGH P(affirmative) regardless of the
    order the LLM listed the options in — otherwise the forecast polarity silently inverts."""
    from swm.world_model_v2.compiler import _affirmative_first, compile_world
    assert _affirmative_first(["no_reply", "reply"]) == ["reply", "no_reply"]
    assert _affirmative_first(["reply", "no_reply"]) == ["reply", "no_reply"]
    assert _affirmative_first(["fail", "pass"]) == ["pass", "fail"]
    assert _affirmative_first(["yes", "no"]) == ["yes", "no"]
    assert _affirmative_first(["not_ratified", "ratified"]) == ["ratified", "not_ratified"]

    def p_affirmative(options, lean, n=200):
        decomp = {"outcome": {"family": "binary", "options": options,
                              "resolution_rule": "affirmative iff the manager replies", "readout_var": "reply"},
                  "outcome_lean": lean, "entities": [{"id": "m", "type": "person", "fields": {}}],
                  "required_causal_processes": ["response_decision"], "rationale": "polarity"}
        plan = compile_world("Will the manager reply?", llm=lambda pr: json.dumps(decomp),
                             evidence="", as_of="2023-05-01", horizon="2023-05-08")
        res, _ = run_from_plan(plan, n_particles=n, seed=3)
        return res["distribution"].get("reply", 0.0)

    # order-invariant AND directionally correct: strong_yes >> strong_no for P(reply), both orders
    assert p_affirmative(["no_reply", "reply"], "strong_yes") > 0.6
    assert p_affirmative(["reply", "no_reply"], "strong_yes") > 0.6
    assert p_affirmative(["no_reply", "reply"], "strong_no") < 0.4
    assert p_affirmative(["reply", "no_reply"], "strong_no") < 0.4


def test_unresolved_terminal_mass_is_reported_not_counted():
    c = OutcomeContract(family="binary", options=["yes", "no"], resolution_rule="r",
                        readout=lambda w: w.quantities["q_out"].value if "q_out" in w.quantities else None,
                        horizon_ts=T0 + 100).validate()
    branches = [WorldBranch(branch_id=str(i), world=_world()) for i in range(4)]
    out = c.project(branches)
    assert out["unresolved_share"] == 1.0 and "warning" in out
    assert out["distribution"] == {}


# ---------------------------------------------------------------- policy-path repair (A3)
def test_llm_cannot_mint_probabilities_by_default():
    minting_llm = lambda p: json.dumps({"p": {"act": 0.99, "wait": 0.01}})
    op = AgentDecisionOperator(llm=minting_llm)             # no explicit experimental opt-in
    w = _world()
    ev = Event(ts=T0 + 1, etype="decision_opportunity", participants=["alice"],
               payload={"actions": [{"type": "act"}, {"type": "wait"}]})
    prop = op.propose(w, ev, random.Random(0))
    assert prop.p_dist == {"act": 0.5, "wait": 0.5}
    assert "policy_unsupported_uniform" in prop.reason_codes
    op2 = AgentDecisionOperator(llm=minting_llm, allow_llm_probabilities=True)   # experimental opt-in
    prop2 = op2.propose(w, ev, random.Random(0))
    assert prop2.p_dist["act"] == pytest.approx(0.99)


def test_fitted_decision_operator_uses_bound_policy():
    pol = lambda world, actor, actions, event: ({"act": 0.8, "wait": 0.2}, ["from_pack"])
    op = FittedDecisionOperator(pol, pack_id="test.pack", source="fitted")
    w = _world()
    ev = Event(ts=T0 + 1, etype="decision_opportunity", participants=["alice"],
               payload={"actions": [{"type": "act"}, {"type": "wait"}]})
    prop = op.propose(w, ev, random.Random(0))
    assert prop.p_dist["act"] == pytest.approx(0.8)
    assert "pack=test.pack" in prop.reason_codes


# ---------------------------------------------------------------- endogenous events (A4)
def test_action_emits_follow_up_event_that_drives_second_actor():
    """A's decision → endogenous exposure event → B's decision becomes possible: the chain the audit
    found missing (interaction through the shared world, not inside one actor's head)."""
    register_event_type("nudge", scheduling="endogenous", validated=True)

    class ActAndNudge(TransitionOperator):
        name = "act_and_nudge"

        def applicable(self, world, event):
            return event.etype == "decision_opportunity" and event.participants == ["a"]

        def propose(self, world, event, rng):
            return TransitionProposal(operator=self.name, action={"actor": "a", "type": "act"},
                                      follow_up_events=[{"etype": "nudge", "ts": world.clock.now + 60,
                                                         "participants": ["b"]}])

        def apply(self, world, proposal):
            world.entity("a").set("current_action", F("act", status="derived"))
            return StateDelta(at=world.clock.now, event_type="decision", operator=self.name)

    class RespondToNudge(TransitionOperator):
        name = "respond_to_nudge"

        def applicable(self, world, event):
            return event.etype == "nudge" and world.entity("a").value("current_action") == "act"

        def propose(self, world, event, rng):
            return TransitionProposal(operator=self.name, action={"actor": "b", "type": "respond"})

        def apply(self, world, proposal):
            world.entity("b").set("current_action", F("responded", status="derived"))
            return StateDelta(at=world.clock.now, event_type="response", operator=self.name)

    w = _world(("a", "b"))
    q = EventQueue(horizon_ts=T0 + 3600)
    q.schedule(Event(ts=T0 + 1, etype="decision_opportunity", participants=["a"]))
    branch = RolloutEngine(operators=[ActAndNudge(), RespondToNudge()]).run_branch(
        w, q, seed=0)
    assert w.entity("b").value("current_action") == "responded"
    assert any(d.event_type == "response" for d in branch.log)


# ---------------------------------------------------------------- correlation + rejuvenation ranges
def test_correlation_rule_respects_non_unit_ranges():
    base = _world()
    base.entities["alice"].set("attention", F(0.7, status="assumed"))
    lat = [LatentVariableRecord(path="alice.responsiveness",
                                candidates={"mean": 1.0, "sd": 0.0, "lo": 0.5, "hi": 1.8}),
           LatentVariableRecord(path="alice.attention",
                                candidates={"mean": 0.5, "sd": 0.0, "lo": 0.0, "hi": 1.0})]
    # responsiveness sampled at exactly 1.0 (its own mid is 1.15) → small NEGATIVE shift on attention;
    # under the old [0,1] default the "mid" was 0.5 and the shift was +0.15 (a silent bias)
    init = InitialStateModel(base_world=base, latents=lat,
                             correlations=[CorrelationRule(src="alice.responsiveness",
                                                           dst="alice.attention", strength=0.3)])
    w = init.sample_particle("b0", random.Random(3))
    att = w.entities["alice"].value("attention")
    expected = 0.5 + 0.3 * ((1.0 - 1.15) / 1.3) * 1.0
    assert att == pytest.approx(expected, abs=1e-9)


def test_rejuvenation_does_not_clamp_into_unit_box():
    from swm.world_model_v2.posterior import ParticlePosterior
    w = _world()
    w.entities["alice"].set("resources", F(50.0, status="sampled"))   # no dist → scale-relative jitter
    ws = [w.clone(branch_id=f"c{i}") for i in range(6)]
    post = ParticlePosterior.from_worlds(ws)
    post.resample(random.Random(0))
    vals = [p.world.entities["alice"].value("resources") for p in post.particles]
    assert all(v > 10.0 for v in vals)                       # old code clamped everything into [0,1]
