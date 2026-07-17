"""The ONE causal chain — grounded stances condition actor POLICIES, chosen actions move the
PATHWAY-PROCESS state, and hazard rounds consume that state (universal, ontology-level; no scenario
keywords). Plus the mode-graph layers: canonical decomposition, decision structures, process
grounding, and the fitting/scoring surfaces."""
import math
import types

import pytest

from swm.world_model_v2.mode_graph import (PATHWAYS, canonical_modes, declare_pathway_processes,
                                           ground_process_states, mode_pathway,
                                           pathway_orientation, progress_var)
from swm.world_model_v2.phase4_execution import ActorPolicyRuntime
from swm.world_model_v2.phase4_policy import (ACTION_ONTOLOGY, ActionTarget, ActorPolicyModel,
                                              ActorViewBuilder, FeasibilityEngine, TypedAction,
                                              action_pathway_effects, actions_advancing_pathway,
                                              stance_action_alignment)
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

T0 = 1_700_000_000.0

REFUSER_STANCES = [
    # committed against the shared cooperative process; pursuing their OWN unilateral campaign;
    # committed to preventing the RIVAL'S victory (targeted — must not damp their own campaign)
    {"actor": "leader_a", "commitment_level": "committed_to_prevent", "reliability": "high",
     "capability": "high", "pathway": "cooperative_agreement", "control": "veto"},
    {"actor": "leader_a", "commitment_level": "actively_pursuing", "reliability": "high",
     "capability": "high", "pathway": "unilateral_action", "control": "sole_authority",
     "target_mode": "a_victory"},
    {"actor": "leader_a", "commitment_level": "committed_to_prevent", "reliability": "high",
     "capability": "high", "pathway": "unilateral_action", "control": "operational_capability",
     "target_mode": "b_victory"},
]


def _world(**quants):
    w = WorldState("w", "b1:x", SimulationClock(now=T0, as_of=T0))
    for k, v in quants.items():
        register_quantity_type(k, units="unit")
        w.quantities[k] = Quantity(name=k, qtype=k, value=v, timestamp=T0)
    return w


def _actor(world, aid="leader_a", stances=None, commitments=None):
    e = Entity(aid)
    e.set("roles", F(["principal"], status="observed"))
    e.set("goals", F(["prevail"], status="inferred"))
    e.set("past_actions", F([], status="observed"))
    if stances is not None:
        e.set("stances", F(stances, status="observed", method="grounded_stances"))
    e.set("commitments", F(commitments or [], status="observed"))
    world.entities[aid] = e
    return e


def _action(name, family=None, aid="leader_a"):
    fam = family or next((f for f, names in ACTION_ONTOLOGY.items() if name in names), "generic")
    return TypedAction(action_id=f"a:{name}", actor_id=aid, actor_role="principal",
                       action_family=fam, action_name=name, target=ActionTarget(),
                       mechanisms_triggered=["record_action"])


# ---------------------------------------------------------------- stance → policy (behavior channel)
def test_pathway_orientation_shared_vs_per_actor_semantics():
    # against the SHARED cooperative process → negative
    assert pathway_orientation(REFUSER_STANCES, "cooperative_agreement") < -0.5
    # pursuing own campaign; the TARGETED prevent (rival's victory) contributes 0 on a per-actor
    # pathway — Russia preventing Ukraine's victory is not Russia opposing military resolution
    assert pathway_orientation(REFUSER_STANCES, "unilateral_action") > 0.4
    # unrelated pathway → 0
    assert pathway_orientation(REFUSER_STANCES, "institutional_procedure") == 0.0


def test_stance_action_alignment_rewards_consistent_actions():
    reject = stance_action_alignment(REFUSER_STANCES, _action("reject"))
    accept = stance_action_alignment(REFUSER_STANCES, _action("accept"))
    mobilize = stance_action_alignment(REFUSER_STANCES, _action("mobilize"))
    wait = stance_action_alignment(REFUSER_STANCES, _action("wait"))
    assert reject > 0.3 and accept < -0.5 and mobilize > 0.2 and wait == 0.0
    # escalate: advances own campaign AND stalls the refused talks — doubly aligned for the refuser
    assert stance_action_alignment(REFUSER_STANCES, _action("escalate")) > reject


def test_policy_posterior_shifts_with_grounded_stances():
    """The SAME candidate set, with vs without grounded stances: a committed refuser's calibrated
    action posterior must move mass from accept toward reject/hold — intentions condition action
    selection through the utility contract, not a hard-coded override."""
    model = ActorPolicyModel()
    actions = [_action(n) for n in ("accept", "reject", "hold_position", "wait")]

    def _posterior(stances):
        w = _world()
        _actor(w, stances=stances)
        view = ActorViewBuilder().build(w, "leader_a")
        fes = [FeasibilityEngine().classify(a, view, w) for a in actions]
        return model.decide([view], actions, [fes], seed=7).action_probabilities
    p_neutral = _posterior(None)
    p_refuser = _posterior(REFUSER_STANCES)
    assert p_refuser["a:accept"] < p_neutral["a:accept"] * 0.6
    assert p_refuser["a:reject"] > p_neutral["a:reject"]
    # never deterministic: a calibrated posterior, not a scripted actor
    assert 0.0 < p_refuser["a:accept"] < p_refuser["a:reject"] < 1.0


def test_binding_commitment_blocks_strongly_contrary_action():
    """A high-reliability categorical stance is a commitment device: the feasibility contract (not
    the utility) blocks the most contrary actions until evidence revises the commitment."""
    prohibits = actions_advancing_pathway("cooperative_agreement", min_effect=0.5)
    assert "accept" in prohibits and "concede" in prohibits and "reject" not in prohibits
    w = _world()
    _actor(w, commitments=[{"id": "grounded_stance:committed_to_prevent:cooperative_agreement",
                            "binding": True, "prohibits": prohibits}])
    view = ActorViewBuilder().build(w, "leader_a")
    fe = FeasibilityEngine()
    assert fe.classify(_action("accept"), view, w).perceived_status == "binding_commitment_conflict"
    assert fe.classify(_action("reject"), view, w).perceived_status == "feasible"


def test_actor_view_wraps_legacy_string_commitments_without_char_splitting():
    w = _world()
    e = _actor(w)
    e.set("commitments", F("no frozen conflict", status="inferred"))
    view = ActorViewBuilder().build(w, "leader_a")
    assert view.commitments == [{"statement": "no frozen conflict"}]


# ---------------------------------------------------------------- action → world (pathway processes)
def test_executed_action_moves_declared_pathway_progress_only():
    runtime = ActorPolicyRuntime()
    w = _world(**{progress_var("cooperative_agreement"): 0.5})
    _actor(w, stances=REFUSER_STANCES)
    view = ActorViewBuilder().build(w, "leader_a")
    act = _action("reject")
    fes = [FeasibilityEngine().classify(act, view, w)]
    post = runtime.model.decide([view], [act], [fes], seed=3)
    trace = types.SimpleNamespace()
    from swm.world_model_v2.phase4_policy import build_trace
    trace = build_trace(question_id="q", plan=None, worlds=[w], views=[view], actions=[act],
                        feasibility=[fes], posterior=post, selected_action_id=act.action_id, seed=3)
    delta, _ = runtime.execute(w, act, post, trace, seed=3)
    v = w.quantities[progress_var("cooperative_agreement")].value
    assert v == pytest.approx(0.5 + 0.04 * -0.7)              # reject: effect −0.7 × step 0.04
    assert any(progress_var("cooperative_agreement") in c.get("path", "") for c in delta.changes)
    # undeclared pathway (unilateral) untouched even though reject has no unilateral effect anyway
    assert progress_var("unilateral_action") not in w.quantities


def test_pathway_step_is_bounded_and_clamped():
    from swm.world_model_v2.phase4_execution import ActorPolicyRuntime as RT
    w = _world(**{progress_var("cooperative_agreement"): 0.06})
    _actor(w)
    d = types.SimpleNamespace(changes=[], change=lambda *a, **k: None)

    class _D:
        def __init__(self):
            self.changes = []

        def change(self, path, before, after):
            self.changes.append({"var": path, "before": before, "after": after})
    for _ in range(10):                                        # repeated exits can't go below floor
        dd = _D()
        RT._apply_pathway_effects(w, _action("exit"), dd)
    assert w.quantities[progress_var("cooperative_agreement")].value >= 0.05


# ---------------------------------------------------------------- state → hazard (endogenous clock)
def test_hazard_round_consumes_pathway_progress_relative():
    from swm.world_model_v2.event_time import HazardRoundOperator
    op = HazardRoundOperator()
    payload = {"mode": "ceasefire", "base_hazard": 0.01, "as_of": T0, "span_s": 100 * 86400.0,
               "consume": [{"var": progress_var("cooperative_agreement"), "weight": 1.0}]}
    ev = types.SimpleNamespace(etype="hazard_round", payload=payload)
    # dormant process (0.15) suppresses; advanced process (0.85) amplifies — multiplicatively
    w_lo = _world(**{progress_var("cooperative_agreement"): 0.15})
    d_lo = op.apply(w_lo, op.propose(w_lo, ev, None))
    w_hi = _world(**{progress_var("cooperative_agreement"): 0.85})
    d_hi = op.apply(w_hi, op.propose(w_hi, ev, None))
    assert d_lo.uncertainty["state_hazard_factor"] < 0.7
    assert d_hi.uncertainty["state_hazard_factor"] > 1.4
    assert d_lo.uncertainty["consumed"] == [progress_var("cooperative_agreement")]
    assert d_hi.uncertainty["hazard"] > d_lo.uncertainty["hazard"]


def test_full_chain_actions_change_absorption_timing():
    """END-TO-END: two identical worlds, SAME hazard chain; in one the actor's simulated actions
    advance the cooperative process, in the other they stall it. The absorbed mass by horizon must
    differ — the timing distribution is (partly) endogenous to simulated behavior."""
    from swm.world_model_v2.event_time import HazardRoundOperator
    op = HazardRoundOperator()
    var = progress_var("cooperative_agreement")
    span = 200 * 86400.0

    def run(action_name, n_particles=160):
        absorbed = 0
        for i in range(n_particles):
            w = _world(**{var: 0.5})
            w.branch_id = f"b{i}:chain"
            _actor(w)
            from swm.world_model_v2.phase4_execution import ActorPolicyRuntime as RT

            class _D:
                def __init__(self):
                    self.changes = []

                def change(self, path, before, after):
                    self.changes.append(path)
            for k in range(1, 11):                             # 10 rounds: act, then a hazard round
                w.clock.advance_to(T0 + k / 11.0 * span)
                RT._apply_pathway_effects(w, _action(action_name), _D())
                payload = {"mode": "deal", "base_hazard": 0.03, "as_of": T0, "span_s": span,
                           "consume": [{"var": var, "weight": 1.0}]}
                ev = types.SimpleNamespace(etype="hazard_round", payload=payload)
                if op.applicable(w, ev):
                    op.apply(w, op.propose(w, ev, None))
            if getattr(w.quantities.get("absorbing_state_reached"), "value", None):
                absorbed += 1
        return absorbed
    n_advance = run("accept")                                  # +1.0 per round → progress climbs
    n_stall = run("exit")                                      # −1.0 per round → progress sinks
    assert n_advance > n_stall * 1.5                           # behavior moves the clock, robustly


# ---------------------------------------------------------------- canonical decomposition + processes
def test_canonical_modes_majority_vote_and_time_index_merge():
    calls = {"n": 0}

    def llm(prompt):
        calls["n"] += 1
        if "PASS 1" in prompt:
            return ('{"modes": [{"id": "ceasefire_2026", "prior": 0.4, "pathway": "cooperative_agreement"},'
                    '{"id": "russian_victory", "prior": 0.3, "pathway": "unilateral_action"},'
                    '{"id": "alien_intervention", "prior": 0.1, "pathway": "stochastic_external"}]}')
        if "PASS 2" in prompt:
            return ('{"modes": [{"id": "ceasefire", "prior": 0.5, "pathway": "cooperative_agreement",'
                    '"decision_structure": {"rule": "unanimity", "approvers": ["Russia", "Ukraine"]}},'
                    '{"id": "russian_victory", "prior": 0.3, "pathway": "unilateral_action"}]}')
        return ('{"modes": [{"id": "ceasefire_agreement", "prior": 0.45, "pathway": "cooperative_agreement"},'
                '{"id": "russian_victory", "prior": 0.35, "pathway": "unilateral_action"}]}')
    modes, rep = canonical_modes(question="When will the conflict end?",
                                 criterion={"resolves_yes_iff": "hostilities end"},
                                 hypotheses=[], options=[], llm=llm, k_passes=3)
    ids = {m["id"] for m in modes}
    assert "ceasefire" in ids and "russian_victory" in ids
    assert "alien_intervention" not in ids                     # 1/3 sources — minority dropped
    assert rep["n_sources"] == 3 and rep["agreement"] > 0.6
    cf = next(m for m in modes if m["id"] == "ceasefire")
    assert cf["pathway"] == "cooperative_agreement"
    assert cf["decision_structure"]["rule"] == "unanimity"     # structure survives reconciliation
    assert any(d["id"] == "alien_intervention" for d in rep["dropped_minority_candidates"])


def test_canonical_modes_no_llm_passes_compiler_structure_through():
    modes, rep = canonical_modes(question="q", criterion={},
                                 hypotheses=[{"id": "deal_2026", "prior": 0.5},
                                             {"id": "deal_2027", "prior": 0.2},
                                             {"id": "collapse", "prior": 0.3}],
                                 options=[], llm=None)
    by = {m["id"]: m for m in modes}
    assert set(by) == {"deal", "collapse"}                     # time-indexed merged, priors summed
    assert by["deal"]["prior"] == pytest.approx(0.7 / 1.0, rel=0.35)   # averaged within source


def test_declare_pathway_processes_and_grounding_map():
    plan = types.SimpleNamespace(quantities=[])
    modes = [{"id": "ceasefire", "pathway": "cooperative_agreement"},
             {"id": "collapse", "pathway": "unilateral_action"}]
    grounding = {"cooperative_agreement": {"state": "exploratory", "value": 0.3, "basis": "feelers"}}
    rep = declare_pathway_processes(plan, modes, grounding=grounding)
    names = {q["name"]: q for q in plan.quantities}
    assert names[progress_var("cooperative_agreement")]["value"] == pytest.approx(0.3)
    assert names[progress_var("unilateral_action")]["value"] == pytest.approx(0.5)   # unknown→neutral
    assert plan._declared_pathways == ["cooperative_agreement", "unilateral_action"]
    assert rep["declared"]["cooperative_agreement"]["state"] == "exploratory"
    # idempotent
    declare_pathway_processes(plan, modes, grounding=grounding)
    assert len(plan.quantities) == 2


def test_ground_process_states_classification_only():
    def llm(prompt):
        assert "dormant" in prompt
        return ('{"process_states": [{"pathway": "cooperative_agreement", "state": "exploratory",'
                '"basis": "Trump calls with both"}, {"pathway": "unilateral_action",'
                '"state": "active", "basis": "summer offensive"}]}')
    out = ground_process_states("q", {}, ["cooperative_agreement", "unilateral_action"], llm=llm)
    assert out["cooperative_agreement"]["value"] == pytest.approx(0.3)
    assert out["unilateral_action"]["value"] == pytest.approx(0.5)


def test_world_driven_mode_couples_to_nonlinear_and_population_state():
    from swm.world_model_v2.event_time import _endogenous_consume
    plan = types.SimpleNamespace(quantities=[{"name": "nonlinear_state", "value": 0.5},
                                             {"name": "population_aggregate:adopters", "value": 0.4}],
                                 _declared_pathways=["diffusion_adoption"])
    declare_pathway_processes(plan, [{"id": "adoption_20pct", "pathway": "diffusion_adoption"}])
    consume, live = _endogenous_consume(plan, {"id": "adoption_20pct",
                                               "pathway": "diffusion_adoption"}, [])
    vars_ = {c["var"] for c in consume}
    assert progress_var("diffusion_adoption") in vars_ and live
    assert "nonlinear_state" in vars_ and "population_aggregate:adopters" in vars_


def test_absorbing_institutional_decision_works_on_non_binary_event_time():
    from swm.world_model_v2.phase_consumers import CollectiveThresholdDecisionOperator
    op = CollectiveThresholdDecisionOperator()
    w = _world()
    ev = types.SimpleNamespace(etype="institutional_decision",
                               payload={"institution_id": "senate", "n_members": 100, "needed": 1,
                                        "outcome_var": "outcome",
                                        "options": ["a", "b", "c"],   # non-binary: absorbing still runs
                                        "absorbing": True, "absorbing_mode": "bill_passes",
                                        "lean": "affirmative"})
    d = op.apply(w, op.propose(w, ev, None))
    assert d is not None
    if getattr(w.quantities.get("absorbing_state_reached"), "value", None):
        assert w.quantities["absorbing_mode"].value == "bill_passes"


# ---------------------------------------------------------------- scoring (frozen-benchmark surface)
def test_crps_first_passage_perfect_and_censored():
    from swm.world_model_v2.event_time import crps_first_passage
    span = 100 * 86400.0
    grid = [T0 + k / 10.0 * span for k in range(1, 11)]
    # forecast: everything absorbed immediately; event at T0+1 → near-zero CRPS
    good = crps_first_passage(grid, [1.0] * 10, event_ts=T0 + 1, as_of=T0, horizon_ts=T0 + span)
    assert good < 0.11
    # same forecast against a CENSORED outcome → maximal penalty (mass inside, nothing happened)
    bad = crps_first_passage(grid, [1.0] * 10, event_ts=None, as_of=T0, horizon_ts=T0 + span)
    assert bad > 0.85
    # honest flat-zero forecast on a censored outcome → 0
    zero = crps_first_passage(grid, [0.0] * 10, event_ts=None, as_of=T0, horizon_ts=T0 + span)
    assert zero == pytest.approx(0.0)
    # a mid-window event scores better for a CDF that rises near it than one that rises late
    ev = T0 + 0.35 * span
    early = crps_first_passage(grid, [0.1, 0.3, 0.6, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 1.0],
                               event_ts=ev, as_of=T0, horizon_ts=T0 + span)
    late = crps_first_passage(grid, [0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
                              event_ts=ev, as_of=T0, horizon_ts=T0 + span)
    assert early < late


def test_interval_coverage_censoring_aware():
    from swm.world_model_v2.event_time import interval_coverage
    assert interval_coverage({"0.1": T0 + 10.0, "0.9": T0 + 90.0}, T0 + 50.0) is True
    assert interval_coverage({"0.1": T0 + 10.0, "0.9": T0 + 90.0}, T0 + 95.0) is False
    assert interval_coverage({"0.1": T0 + 10.0, "0.9": None}, None) is True     # both say beyond
    assert interval_coverage({"0.1": T0 + 10.0, "0.9": T0 + 90.0}, None) is False
    assert interval_coverage({"0.1": None, "0.9": None}, T0 + 5.0) is False


def test_pathway_registry_covers_non_actor_worlds():
    assert not PATHWAYS["physical_process"].actor_driven
    assert not PATHWAYS["threshold_crossing"].actor_driven
    assert PATHWAYS["cooperative_agreement"].actor_driven
    assert mode_pathway({"id": "inflation_crosses_3pct"}) == "threshold_crossing"
    assert mode_pathway({"id": "hurricane_landfall"}) == "physical_process"
    assert mode_pathway({"id": "adoption_reaches_20pct"}) == "diffusion_adoption"
    assert mode_pathway({"id": "ceasefire_agreement"}) == "cooperative_agreement"
    assert mode_pathway({"id": "board_confirms_via_vote"}) == "institutional_procedure"


def test_action_effects_are_ontology_level_and_symmetric_surface():
    assert action_pathway_effects("negotiation", "accept")["cooperative_agreement"] == 1.0
    assert action_pathway_effects("institutional", "veto")["institutional_procedure"] == -1.0
    assert action_pathway_effects("organizational_market", "launch")["operational_execution"] == 0.9
    assert action_pathway_effects("generic", "wait") == {}
    # name-level fallback resolves family-ambiguous names
    assert "cooperative_agreement" in action_pathway_effects("", "accept")
