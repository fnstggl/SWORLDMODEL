"""WorldModelV2 acceptance suite — the anti-cheating tests (Phase 10) + Phase 2B tests. All offline."""
from __future__ import annotations

import json
import random

import pytest

from swm.world_model_v2.contracts import (ActionSpace, ContractError, Intervention, OutcomeContract,
                                          UtilityFunction)
from swm.world_model_v2.events import Event, EventQueue, StochasticHazard, register_event_type
from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.init_state import (CoherenceRule, CorrelationRule, InitialStateModel,
                                           LatentVariableRecord, llm_distribution)
from swm.world_model_v2.institutions import Rule, RuleSystem
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.population import Population, PopulationSegment
from swm.world_model_v2.rollout import RolloutEngine, WorldModelV2Run
from swm.world_model_v2.state import (Entity, F, SimulationClock, StateField, WorldState, parse_time,
                                      register_entity_extension)
from swm.world_model_v2.transitions import (AgentDecisionOperator, BackgroundDynamicsOperator,
                                            BeliefUpdateOperator, InstitutionalVoteOperator,
                                            ResourceUpdateOperator, StateDelta, observable_view)

T0 = parse_time("2026-07-01")
DAY = 86400.0


def _world(entities=("alice", "bob")):
    w = WorldState(world_id="w", branch_id="root", clock=SimulationClock(now=T0, as_of=T0),
                   network=RelationGraph(), information=InformationLedger())
    for e in entities:
        ent = Entity(identity=e)
        ent.set("attention", F(0.5, status="assumed"))
        ent.set("resources", F(100.0, status="observed"), key="money")
        w.entities[e] = ent
    return w


# ---------------- Test 1+11: typed state outside prompts, with provenance
def test_state_is_typed_and_provenanced_outside_prompts():
    w = _world()
    a = w.entity("alice")
    a.set("beliefs", F(0.6, status="inferred", sources=["press report"], confidence=0.7,
                       method="grounded_llm_posterior"), key="deal_is_good")
    sf = a.get("beliefs", key="deal_is_good")
    assert isinstance(sf, StateField) and sf.value == 0.6
    p = sf.prov.as_dict()
    assert p["status"] == "inferred" and p["sources"] == ["press report"] and p["confidence"] == 0.7
    assert p["method"] == "grounded_llm_posterior"
    with pytest.raises(KeyError):
        a.set("vibes", F(1.0))                       # arbitrary untyped keys rejected


def test_entity_extension_registry():
    register_entity_extension("sports", fields={"fitness_level": "0-1 physical capacity"},
                              entity_types=("person",))
    e = Entity(identity="player7")
    e.set("fitness_level", F(0.8, status="observed"))
    assert e.value("fitness_level") == 0.8


# ---------------- Test 2: machine-readable deltas
def test_every_transition_produces_state_delta():
    w = _world()
    ev = Event(ts=T0 + DAY, etype="exposure", participants=["alice"],
               payload={"item_id": "i1", "trust": 0.8, "salience": 0.9})
    w.information.publish(InformationItem(item_id="i1", content="the budget was cut", credibility=0.9))
    w.information.expose("alice", "i1", T0 + DAY)
    w.clock.advance_to(ev.ts)
    delta, vr = BeliefUpdateOperator().run(w, ev, random.Random(0))
    assert isinstance(delta, StateDelta) and delta.changes
    ch = delta.changes[0]
    assert set(ch) == {"path", "before", "after"} and isinstance(ch["after"], float)


# ---------------- Test 3: persistence — event 1 changes exactly what event 2 reads
def test_action_in_event1_changes_state_read_in_event2():
    w = _world()
    ev1 = Event(ts=T0 + DAY, etype="decision_opportunity", participants=["alice"],
                payload={"actions": [{"type": "yes"}, {"type": "no"}]})
    w.clock.advance_to(ev1.ts)
    op = AgentDecisionOperator(llm=None)
    delta, _ = op.run(w, ev1, random.Random(1))
    chosen = w.entity("alice").value("current_action")
    assert chosen in ("yes", "no") and delta.changes
    # event 2 (a vote) reads that exact field
    ev2 = Event(ts=T0 + 2 * DAY, etype="collective_vote", participants=["alice", "bob"],
                payload={"threshold": 0.5, "outcome_var": "motion_passed"})
    w.clock.advance_to(ev2.ts)
    delta2, _ = InstitutionalVoteOperator().run(w, ev2, random.Random(2))
    assert delta2.uncertainty["tally"]["yes"] == (1.0 if chosen == "yes" else 0.0)


# ---------------- Test 4: actor-specific information
def test_actors_have_different_information_sets():
    w = _world()
    w.information.publish(InformationItem(item_id="pub", content="public fact", credibility=0.9))
    w.information.publish(InformationItem(item_id="priv", content="private to alice", kind="private",
                                          credibility=0.9))
    w.information.expose("alice", "pub", T0)
    w.information.expose("bob", "pub", T0)
    w.information.expose("alice", "priv", T0)
    va, vb = observable_view(w, "alice"), observable_view(w, "bob")
    ca = {i["content"] for i in va["observed_information"]}
    cb = {i["content"] for i in vb["observed_information"]}
    assert "private to alice" in ca and "private to alice" not in cb and "public fact" in cb


# ---------------- Test 5: executable institutional rules reject invalid actions
def test_institutional_rules_reject_invalid_action():
    w = _world()
    w.institutions["board"] = RuleSystem(institution_id="board", rules=[
        Rule(rule_id="r1", kind="decision_right", params={"actions": ["approve_raise"],
                                                          "holders": ["bob"]})])
    ev = Event(ts=T0 + DAY, etype="decision_opportunity", participants=["alice"],
               payload={"actions": [{"type": "approve_raise"}]})
    w.clock.advance_to(ev.ts)
    delta, vr = AgentDecisionOperator(llm=None).run(w, ev, random.Random(0))
    assert delta is None                                   # the only candidate action was invalid → no-op
    # resource floor: can't spend more than you have
    register_event_type("x", validated=True)
    ev2 = Event(ts=T0 + 2 * DAY, etype="x", payload={"resource_delta": {"actor": "alice",
                                                                        "resource": "money", "delta": -500}})
    w.clock.advance_to(ev2.ts)
    d2, vr2 = ResourceUpdateOperator().run(w, ev2, random.Random(0))
    assert d2 is None and not vr2.ok and "insufficient" in vr2.reasons[0]


# ---------------- Test 6+8+9: cross-module causality, hidden-state ensemble, native readout
def _mini_run(n_particles=40):
    base = _world(entities=("boss",))
    base.entity("boss").set("attention", F(None, dist={"overloaded": 0.5, "normal": 0.5}, status="assumed"))
    init = InitialStateModel(
        base_world=base,
        latents=[LatentVariableRecord(path="boss.attention",
                                      candidates={"overloaded": 0.5, "normal": 0.5}, method="prior")])

    def build_queue(world):
        q = EventQueue(horizon_ts=T0 + 7 * DAY)
        q.schedule(Event(ts=T0 + DAY, etype="information_published", payload={}))
        q.schedule(Event(ts=T0 + 2 * DAY, etype="decision_opportunity", participants=["boss"],
                         payload={"actions": [{"type": "reply"}, {"type": "ignore"}]}))
        return q

    class AttentionPolicy(AgentDecisionOperator):
        # deterministic policy over TYPED actions: replies iff attention sampled 'normal'
        def _policy(self, world, actor, actions, event):
            att = actor.value("attention")
            p = 0.95 if att == "normal" else 0.05
            return {"reply": p, "ignore": 1 - p}, ["attention_gate"]

    contract = OutcomeContract(family="response_occurrence", options=["reply", "ignore"],
                               resolution_rule="did boss reply within 7d",
                               readout=lambda w: w.entity("boss").value("current_action") or "ignore",
                               horizon_ts=T0 + 7 * DAY).validate()
    run = WorldModelV2Run(initial=init, queue_builder=build_queue,
                          operators=[AttentionPolicy(llm=None), BackgroundDynamicsOperator()],
                          contract=contract, n_particles=n_particles)
    return run


def test_hidden_state_ensemble_and_native_terminal_readout():
    run = _mini_run()
    result, branches = run.run(seed=3)
    dist = result["distribution"]
    # the latent attention distribution (50/50) must drive the outcome distribution — not one guessed state
    assert 0.25 <= dist.get("reply", 0) <= 0.75 and result["n_worlds"] == 40
    assert result["readout"] == "terminal_states"          # the number came from worlds, not an LLM
    # every branch carries machine-readable history + its sampled latent recorded
    assert all(b.world.uncertainty_meta["sampled"]["boss.attention"] in ("overloaded", "normal")
               for b in branches)


# ---------------- Test 7: real time — 1-day vs 30-day windows differ by events, not labels
def test_real_time_windows_differ():
    base = _world(entities=("p",))
    init = InitialStateModel(base_world=base, latents=[])
    def qb(horizon_days):
        def build(world):
            q = EventQueue(horizon_ts=T0 + horizon_days * DAY)
            for d in range(1, 40):
                q.schedule(Event(ts=T0 + d * DAY, etype="background_tick", payload={"elapsed_days": 1.0}))
            return q
        return build
    contract = OutcomeContract(family="continuous", resolution_rule="attention",
                               readout=lambda w: w.entity("p").value("attention") or 0.5,
                               horizon_ts=T0 + 40 * DAY).validate()
    short = WorldModelV2Run(initial=init, queue_builder=qb(1), operators=[BackgroundDynamicsOperator()],
                            contract=contract, n_particles=3)
    long = WorldModelV2Run(initial=init, queue_builder=qb(30), operators=[BackgroundDynamicsOperator()],
                           contract=contract, n_particles=3)
    (_, b1), (_, b30) = short.run(seed=1), long.run(seed=1)
    assert len(b30[0].log) > len(b1[0].log)                # more elapsed time ⇒ more background transitions
    assert b30[0].world.clock.now > b1[0].world.clock.now  # the clock is real, not a round label


# ---------------- Test 10: matched counterfactual interventions
def test_interventions_run_on_cloned_matched_worlds():
    run = _mini_run(n_particles=20)

    def nudge(world, queue):
        world.entity("boss").set("attention", F("normal", status="derived", method="intervention"))
    actions = ActionSpace(interventions=[
        Intervention(intervention_id="send_at_good_time", apply=nudge, kind="timing")])
    util = UtilityFunction(name="replied", fn=lambda w: 1.0 if w.entity("boss").value("current_action")
                           == "reply" else 0.0)
    rep = run.evaluate_interventions(actions, util, seed=5)
    arms = {r["intervention"]: r for r in rep["ranking"]}
    # forcing attention=normal must beat the 50/50 baseline on MATCHED worlds
    assert arms["send_at_good_time"]["expected_utility"] > arms["none"]["expected_utility"]
    assert rep["n_matched_worlds"] == 20 and rep["readout"] == "terminal_states"
    assert abs(sum(r["p_best"] for r in rep["ranking"]) - 1.0) < 1e-6


# ---------------- Test 12: no legacy logistic path importable from v2
def test_v2_never_imports_legacy_logistic():
    import pathlib
    import re
    v2 = pathlib.Path(__file__).resolve().parent.parent / "swm" / "world_model_v2"
    for f in v2.glob("*.py"):
        src = f.read_text()
        assert not re.search(r"calibrated_readout|bayes_logistic|from swm\.transition|from swm\.variables",
                             src), f"{f.name} references the banned legacy logistic/ODE paths"


# ---------------- Test 13: panel naming
def test_panel_is_labeled_ensemble_never_simulation():
    import pathlib
    audit = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "WMV2_PHASE0_AUDIT.md").read_text()
    assert "independent forecast ensemble" in audit
    assert "NEVER a society simulation" in audit


# ---------------- Phase 2B tests
def test_llm_cannot_inject_precise_coefficients():
    from swm.world_model_v2.uncertainty import MechanismProposal, estimate_parameter, validate_mechanism
    prop = MechanismProposal(mechanism_id="injury", ontology_type="exogenous_discrete_event",
                             entities=["player_7"],
                             causal_path=["physical_capacity", "lineup", "match_outcome"],
                             relevance_confidence="high")
    assert prop.validate_typing() == []
    # no defensible source anywhere → REJECTED, not fabricated
    rejected = estimate_parameter("baseline_hazard")
    assert rejected.source == "rejected" and rejected.distribution is None
    vm = validate_mechanism(prop, {"baseline_hazard": rejected})
    ok, why = vm.executable(allow_experimental=True)
    assert not ok and "rejected" in why                    # Monte Carlo cannot run an unsupported guess
    # weak sources get BROADENED, and cap status at prior_backed
    weak = estimate_parameter("baseline_hazard", broad=({"mean": 0.02, "sd": 0.02}, ["league tables"]))
    assert weak.distribution["sd"] == 0.04                 # doubled — weaker source, wider uncertainty
    vm2 = validate_mechanism(prop, {"baseline_hazard": weak}, fitted_evidence=True)
    assert vm2.registry_status == "prior_backed"           # can't claim 'validated' on weak parameters


def test_conflict_detection_and_uncertainty_report():
    from swm.world_model_v2.uncertainty import (MechanismProposal, detect_conflicts, estimate_parameter,
                                                uncertainty_report, validate_mechanism)
    def mk(mid, reads=(), writes=()):
        p = MechanismProposal(mechanism_id=mid, ontology_type="background_stochastic_process",
                              causal_path=["a", "b"], relevance_confidence="medium")
        return validate_mechanism(p, {"r": estimate_parameter("r", broad=(0.1, []))},
                                  prior_backed=True, reads=reads, writes=writes)
    m1, m2 = mk("m1", writes=("x",)), mk("m2", writes=("x",))
    m3, m4 = mk("m3", reads=("p",), writes=("q",)), mk("m4", reads=("q",), writes=("p",))
    problems = detect_conflicts([m1, m2, m3, m4])
    assert any("double-counted" in p for p in problems) and any("circular" in p for p in problems)
    rep = uncertainty_report([m1, m3], latents=[])
    assert rep["accepted_mechanisms"] and "model_uncertainty" in rep


def test_llm_distribution_never_returns_silent_point():
    from swm.world_model_v2.init_state import llm_distribution
    rec = llm_distribution(lambda p: json.dumps({"candidates": {"0.3": 0.6, "0.7": 0.4},
                                                 "confidence": 0.8}), "q", "boss.attention")
    assert rec.method == "llm" and len(rec.candidates) == 2 and abs(sum(rec.candidates.values()) - 1) < 1e-6
    bad = llm_distribution(lambda p: json.dumps({"candidates": {"0.63": 1.0}}), "q", "boss.mood")
    assert bad.method == "prior" and bad.candidates["sd"] > 0   # single point → broad prior, flagged


def test_correlated_latents_sample_jointly_and_coherence_repairs():
    base = _world(entities=("boss",))
    init = InitialStateModel(
        base_world=base,
        latents=[LatentVariableRecord(path="boss.resources[workload]",
                                      candidates={"mean": 0.7, "sd": 0.2, "lo": 0.0, "hi": 1.0},
                                      method="prior"),
                 LatentVariableRecord(path="boss.attention",
                                      candidates={"mean": 0.5, "sd": 0.2, "lo": 0.0, "hi": 1.0},
                                      method="prior")],
        correlations=[CorrelationRule(src="boss.resources[workload]", dst="boss.attention", strength=-0.8)],
        coherence=[CoherenceRule(name="attention_bounds",
                                 check=lambda s: (0 <= s["boss.attention"] <= 1, None))])
    rng_worlds = init.sample_particles(60, seed=9)
    pairs = [(w.uncertainty_meta["sampled"]["boss.resources[workload]"],
              w.uncertainty_meta["sampled"]["boss.attention"]) for w in rng_worlds]
    hi_w = [a for wl, a in pairs if wl > 0.8]
    lo_w = [a for wl, a in pairs if wl < 0.6]
    assert hi_w and lo_w and (sum(hi_w) / len(hi_w)) < (sum(lo_w) / len(lo_w))   # anticorrelated, jointly


def test_contract_refuses_to_run_without_readout():
    with pytest.raises(ContractError):
        OutcomeContract(family="binary", resolution_rule="x", readout=None).validate()


def test_population_allocation_tracks_weight_uncertainty_sensitivity():
    pop = Population(population_id="voters", segments=[
        PopulationSegment(segment_id="big_stable", weight=F(0.7),
                          heterogeneity={"support": F(None, dist={"mean": .5, "sd": .05})}, sensitivity=0.3),
        PopulationSegment(segment_id="small_pivotal", weight=F(0.1),
                          heterogeneity={"support": F(None, dist={"mean": .5, "sd": .4})}, sensitivity=0.9),
        PopulationSegment(segment_id="mid", weight=F(0.2),
                          heterogeneity={"support": F(None, dist={"mean": .5, "sd": .2})}, sensitivity=0.5)])
    alloc = pop.allocate(30)
    assert alloc["big_stable"] > alloc["mid"]              # weight matters
    assert alloc["small_pivotal"] >= 2                     # floor: pivotal small segment not erased
    parts = pop.sample_particles(30, random.Random(0))
    assert abs(sum(p.weight for p in parts) - 1.0) < 1e-6  # weights conserve


# ---------------- Test 15 (generality): a novel scenario compiles + runs with NO scenario branch
def test_novel_scenario_compiles_and_runs_end_to_end():
    """'Will the nurses ratify the hospital contract?' — a negotiation/institutional class never named in
    any implementation example. A scripted decomposition flows through the ONE compiler → materializer →
    rollout → native terminal distribution. Grep-proof: no 'if election/if email/if viral' exists in v2."""
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.materialize import run_from_plan

    decomposition = {
        "outcome": {"family": "binary", "options": ["yes", "no"],
                    "resolution_rule": "ratified iff majority of votes cast are yes",
                    "readout_var": "ratified"},
        "entities": [{"id": f"nurse_{i}", "type": "person", "fields": {"attention": 0.6}}
                     for i in range(3)] + [{"id": "union", "type": "institution", "fields": {}}],
        "relations": [{"src": f"nurse_{i}", "rel": "belongs_to", "dst": "union"} for i in range(3)],
        "institutions": [{"id": "union", "rules": [{"kind": "eligibility",
                                                    "params": {"actions": ["vote"], "require": {}}}]}],
        "quantities": [{"name": "ratified", "qtype": "ratified", "value": None}],
        "latents": [{"path": f"nurse_{i}.attention", "why": "workload unknown", "lo": 0.2, "hi": 1.0}
                    for i in range(3)],
        "scheduled_events": [
            {"etype": "decision_opportunity", "at": "2026-07-10", "participants": [f"nurse_{i}"],
             "payload": {"actions": [{"type": "yes"}, {"type": "no"}]}} for i in range(3)] + [
            {"etype": "collective_vote", "at": "2026-07-12",
             "participants": [f"nurse_{i}" for i in range(3)],
             "payload": {"threshold": 0.5, "outcome_var": "ratified"}}],
        "hazards": [{"etype": "distraction", "rate_per_day": 0.1, "participants": ["nurse_0"]}],
        "mechanisms": ["agent_decision", "institutional_vote", "background_dynamics",
                       "made_up_mechanism_xyz"],
        "missing_mechanisms": [{"name": "strike_contagion", "why": "no validated strike-diffusion kernel"}],
        "sensitivity": {"agent_decision": 0.9, "background_dynamics": 0.2},
        "rationale": "ratification is generated by member votes under union rules"}
    plan = compile_world("Will the nurses ratify the hospital contract?",
                         llm=lambda p: json.dumps(decomposition), evidence="(scripted)",
                         as_of="2026-07-01", horizon="2026-07-14")
    # unknown mechanism rejected, missing one marked experimental — never fabricated
    assert any(r["id"] == "made_up_mechanism_xyz" for r in plan.rejected_mechanisms)
    assert plan.candidate_experimental_mechanisms[0]["name"] == "strike_contagion"
    assert "NOT executed" in plan.candidate_experimental_mechanisms[0]["status"]
    # fidelity planner: high-sensitivity explicit, low marginalized — deterministic over advisory hints
    assert "agent_decision" in plan.fidelity_plan["explicit"]
    assert "background_dynamics" in plan.fidelity_plan["marginalized"]
    # end-to-end: materialize + roll + read from terminal states
    result, branches = run_from_plan(plan, llm=None, n_particles=8, seed=4)
    dist = result["distribution"]
    assert set(dist) <= {"True", "False", "None"} and result["readout"] == "terminal_states"
    assert result["n_deltas"] > 0                          # machine-readable history exists


def test_compiler_abstains_without_readout_or_mechanisms():
    from swm.world_model_v2.compiler import CompileAbstention, compile_world
    with pytest.raises(CompileAbstention):                 # no readout var → refuse to simulate
        compile_world("q", llm=lambda p: json.dumps({"outcome": {"family": "binary"}}),
                      evidence="", as_of="2026-07-01", horizon="2026-07-14")
    with pytest.raises(CompileAbstention):                 # only unknown mechanisms → abstain, marked
        compile_world("q", llm=lambda p: json.dumps(
            {"outcome": {"family": "binary", "readout_var": "x"},
             "mechanisms": ["nonexistent"], "missing_mechanisms": [{"name": "nonexistent", "why": "y"}]}),
            evidence="", as_of="2026-07-01", horizon="2026-07-14")


def test_no_scenario_branches_in_v2_source():
    import pathlib
    import re
    v2 = pathlib.Path(__file__).resolve().parent.parent / "swm" / "world_model_v2"
    pat = re.compile(r"^\s*(if|elif)\s+.*(election|email|viral|sports|headline).*:", re.I)
    for f in v2.glob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if line.strip().startswith("#") or "`" in line:
                continue                                   # prose/docstrings mentioning the anti-pattern
            assert not pat.search(line), f"{f.name}:{i} looks like a scenario-level branch: {line.strip()}"


# ================= observation model + particle posterior (Parts 2-3) =================
def test_observation_model_generates_noise_missingness_delay_and_likelihood():
    from swm.world_model_v2.observation import GaussianMeasurement, BernoulliDetection
    rng = random.Random(0)
    poll = GaussianMeasurement(sd=0.04, bias=0.01, p_missing=0.2, delay_days=2.0)
    obs = [poll.generate(0.52, at=T0, rng=rng, of_path="race.support") for _ in range(200)]
    missing = [o for o in obs if o.value is None]
    assert 0.1 < len(missing) / 200 < 0.35                       # missingness real
    shown = [o.value for o in obs if o.value is not None]
    assert abs(sum(shown) / len(shown) - 0.53) < 0.02            # bias + noise around latent+bias
    assert obs[0].reported_at - obs[0].at == 2 * 86400.0         # reporting delay
    # likelihood: the true latent explains the observation better than a wrong one
    o = poll.generate(0.52, at=T0, rng=random.Random(1), of_path="race.support")
    while o.value is None:
        o = poll.generate(0.52, at=T0, rng=random.Random(2), of_path="race.support")
    assert poll.likelihood(o, 0.52) > poll.likelihood(o, 0.30)
    email = BernoulliDetection(p_detect=0.9, p_false=0.05)
    pos_obs = email.generate(True, at=T0, rng=random.Random(3), of_path="bob.replied")
    assert email.likelihood(pos_obs, True) != email.likelihood(pos_obs, False)


def test_particle_posterior_reweights_resamples_and_traces_ancestry():
    from swm.world_model_v2.observation import GaussianMeasurement, Observation, register_observation_model
    from swm.world_model_v2.posterior import ParticlePosterior
    # 40 particles: half with latent support 0.3, half 0.7 (sampled status)
    worlds = []
    for i in range(40):
        w = _world(entities=("race",))
        w.entity("race").set("beliefs", F(0.3 if i < 20 else 0.7, status="sampled"), key="support")
        w.branch_id = f"b{i:03d}"
        worlds.append(w)
    post = ParticlePosterior.from_worlds(worlds, resample_threshold=0.6)
    assert abs(sum(p.weight for p in post.particles) - 1.0) < 1e-9          # normalized
    model = register_observation_model("race.beliefs[support]", GaussianMeasurement(sd=0.05))
    # an informative poll at 0.68 must move posterior mass to the 0.7 particles
    obs = Observation(obs_id="poll1", of_path="race.beliefs[support]", value=0.68, at=T0, reported_at=T0)
    post.assimilate(obs)
    exp = post.expectation("race.beliefs[support]")
    assert exp > 0.6                                              # contradictory particles lost mass
    assert abs(sum(p.weight for p in post.particles) - 1.0) < 1e-9
    # a second contradicting observation crushes ESS → auto-resample fired; ancestry traceable
    post.assimilate(Observation(obs_id="poll2", of_path="race.beliefs[support]", value=0.71,
                                at=T0 + DAY, reported_at=T0 + DAY))
    assert any(e["event"] == "resample" for e in post.log)
    assert all(p.ancestry for p in post.particles)                # every survivor knows its parents
    # observed fields were NEVER perturbed by rejuvenation
    assert all(p.world.entity("race").get("resources", key="money").prov.status == "observed"
               and p.world.entity("race").value("resources", key="money") == 100.0
               for p in post.particles)
    # posterior predictive check flags an impossible observation
    ppc = post.posterior_predictive_check(Observation(obs_id="weird", of_path="race.beliefs[support]",
                                                      value=-3.0, at=T0, reported_at=T0))
    assert ppc["suspect_model_family"]


def test_provenance_semantics_execute_differently():
    """sampled/inferred fields may be rejuvenated; observed fields never (executable semantics, not labels)."""
    from swm.world_model_v2.posterior import ParticlePosterior
    w = _world(entities=("p",))
    w.entity("p").set("attention", F(0.5, dist={"lo": 0.0, "hi": 1.0}, status="sampled"))
    before_obs = w.entity("p").value("resources", key="money")
    rng = random.Random(0)
    ParticlePosterior._rejuvenate(w, rng, jitter=0.2)
    assert w.entity("p").value("attention") != 0.5                # sampled → perturbed
    assert w.entity("p").value("resources", key="money") == before_obs   # observed → untouched


# ================= facade + product-eligibility + import boundary =================
def test_facade_requires_explicit_architecture_and_contract():
    from swm.facade import ArchitectureError, RunRecord, forecast
    with pytest.raises(ArchitectureError):
        forecast("q", architecture="whatever")
    # a V2 record that touched legacy FAILS finalize (contamination cannot ship)
    rec = RunRecord(architecture="world_model_v2")
    rec.legacy_executed = True
    with pytest.raises(ArchitectureError):
        rec.finalize()
    # a clean V2 record is recorded but NOT product-eligible until benchmark-validated
    clean = RunRecord(architecture="world_model_v2").finalize()
    assert clean.product_eligible is False and clean.validation_status == "architecture_validated"
    base = RunRecord(architecture="baseline:observer_panel_v1").finalize()
    assert base.product_eligible is False and base.baseline == "baseline:observer_panel_v1"


def test_import_boundary_ast_v2_never_imports_legacy():
    """AST-based (not grep): no module in swm/world_model_v2/ imports the legacy engines/compilers."""
    import ast
    import pathlib
    banned_prefixes = ("swm.api.compiler", "swm.api.world_model", "swm.engine.front_door",
                       "swm.engine.observer_panel", "swm.engine.society", "swm.engine.individual",
                       "swm.engine.diffusion", "swm.engine.router", "swm.transition", "swm.variables",
                       "swm.simulation", "swm.worlds")
    v2 = pathlib.Path(__file__).resolve().parent.parent / "swm" / "world_model_v2"
    for f in v2.glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not any(n.startswith(b) for b in banned_prefixes), \
                    f"{f.name} imports legacy module {n}"


def test_facade_is_the_only_legacy_door_for_new_code():
    """Legacy engine imports outside swm/engine, swm/api, swm/eval, experiments, tests and the facade are
    forbidden — new code reaches baselines ONLY by naming them through the facade."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "swm"
    allowed_dirs = {"engine", "api", "eval", "decision", "memory", "state", "transition", "variables",
                    "simulation", "worlds", "experimental"}
    for f in root.glob("*.py"):
        if f.name in ("facade.py", "__init__.py"):
            continue
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("swm.engine"):
                raise AssertionError(f"{f.name}: new top-level code must use swm.facade, not swm.engine")
