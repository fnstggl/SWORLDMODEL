"""Phase 6 — evidence-backed mechanism registry: families, execution, selection, transport, composition,
and honest lifecycle gates. Referenced as `test_ref` by the Phase-6 family records."""
import random

import pytest

from swm.world_model_v2.registry.families import behavioral as B
from swm.world_model_v2.registry.families import hazard as H
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState, parse_time
from swm.world_model_v2.events import Event
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.information import InformationLedger

T0 = parse_time("2020-01-01T00:00:00Z")


def _world():
    w = WorldState(world_id="w", branch_id="root", clock=SimulationClock(now=T0, as_of=T0),
                   network=RelationGraph(), information=InformationLedger())
    w.entities["a"] = Entity(identity="a")
    return w


# ---------------- family math (verified published parameters) ----------------
def test_bass_diffusion_math():
    assert B.bass_new_adopters(0.03, 0.38, 1000, 0) == pytest.approx(30.0)   # p·M at N=0
    traj = B.bass_trajectory(0.03, 0.38, 1000, steps=60)
    assert traj[-1] > 900 and all(traj[i] <= traj[i + 1] + 1e-9 for i in range(len(traj) - 1))  # monotone
    assert B.bass_peak_time(0.03, 0.38) == pytest.approx(5.96, abs=0.5)      # ln(q/p)/(p+q)


def test_ultimatum_behavioral_not_spe():
    # behavioral: generous offers accepted, low offers rejected (NOT the ~0 SPE)
    assert B.ultimatum_response(0.40, 0.25) > 0.9
    assert B.ultimatum_response(0.05, 0.25) < 0.1
    # terminal sensitivity: proposer payoff is single-peaked, maximized well above the SPE offer
    payoffs = {o: B.ultimatum_expected_proposer_payoff(o, 0.25) for o in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)}
    assert max(payoffs, key=payoffs.get) >= 0.2


def test_trust_game_returns_match_meta():
    o = B.trust_game_outcome(0.50, 0.37)         # Johnson-Mislin means
    assert o["sent"] == pytest.approx(5.0) and o["returned"] == pytest.approx(5.55)
    assert o["investor_net_from_trust"] > 0       # trust pays off at the meta-analytic means


def test_social_pressure_levels_verified():
    # exact GGL 2008 turnout levels (independently verified)
    assert B.social_pressure_turnout_p("control") == pytest.approx(0.297)
    assert B.social_pressure_turnout_p("neighbors") == pytest.approx(0.378)
    # transported to a different base turnout: the ADDITIVE effect (8.1pp) is applied
    assert B.social_pressure_turnout_p("neighbors", base_turnout=0.5) == pytest.approx(0.5 + 0.081)


def test_donation_match_relative_and_ratio_null():
    assert B.matching_donation_p(0.20, True) == pytest.approx(0.244)   # ×1.22
    assert B.matching_donation_p(0.20, False) == pytest.approx(0.20)
    assert B.matching_donation_ratio_is_flat("3:1")                    # ratio is flat (preserved null)


def test_reputation_beta_update_and_premium():
    a, b = 1.0, 1.0
    for _ in range(9):
        a, b = B.reputation_update(a, b, True)
    assert B.reputation_score(a, b) > 0.8
    assert 0 < B.reputation_price_premium(a, b) <= 0.081   # approaches the Resnick causal estimate


# ---------------- EXECUTION plane: transition → StateDelta → terminal sensitivity ----------------
def test_behavioral_operator_executes_and_writes_statedelta():
    op = B.BehavioralMechanismOperator()
    w = _world()
    ev = Event(ts=T0, etype="behavioral_mechanism", participants=["a"],
               payload={"hazard_spec": {"kind": "behavioral", "mechanism": "social_pressure_turnout",
                                        "params": {"treatment": "neighbors"}, "outcome_var": "turnout",
                                        "family": "social_pressure_turnout", "pack_id": "ggl_2008_michigan"}})
    w.clock.advance_to(ev.ts)
    assert op.applicable(w, ev)
    delta, vr = op.run(w, ev, random.Random(0))
    assert delta is not None and delta.changes                 # a StateDelta with a real change
    assert "turnout" in w.quantities                            # terminal quantity written
    assert delta.changes[0]["path"] == "quantities[turnout]"


def test_behavioral_terminal_sensitivity_to_parameter():
    """The mechanism must MATERIALLY affect the terminal outcome: a generous vs stingy ultimatum offer
    changes P(accept) → changes the expected terminal acceptance across branches."""
    op = B.BehavioralMechanismOperator()
    def accept_rate(offer):
        hits = 0
        for seed in range(200):
            w = _world()
            w.branch_id = f"b{seed}"
            ev = Event(ts=T0, etype="behavioral_mechanism",
                       payload={"hazard_spec": {"kind": "behavioral", "mechanism": "ultimatum_offer_response",
                                                "params": {"offer_frac": offer, "accept_threshold": 0.25},
                                                "outcome_var": "accepted", "options": ["accept", "reject"]}})
            w.clock.advance_to(ev.ts)
            op.run(w, ev, random.Random(seed))
            hits += int(w.quantities["accepted"].value == "accept")
        return hits / 200
    assert accept_rate(0.40) - accept_rate(0.05) > 0.5          # terminal outcome tracks the parameter


def test_feature_hazard_from_pack_executes():
    hz = H.FeatureHazard({"x": 1.0}, -0.5, {"x": [0.0, 1.0]})
    assert 0 < hz.p({"x": 2.0}) < 1
    op = H.FeatureHazardOperator()
    w = _world()
    ev = Event(ts=T0, etype="outcome_hazard", participants=["a"],
               payload={"hazard_spec": {"weights": {"x": 1.0}, "intercept": -0.5,
                                        "standardizer": {"x": [0.0, 1.0]}, "features": {"x": 3.0},
                                        "outcome_var": "churned", "actor": "a", "family": "attrition"}})
    w.clock.advance_to(ev.ts)
    delta, vr = op.run(w, ev, random.Random(0))
    assert delta is not None and "churned" in w.quantities


# ---------------- selection: BY CAUSAL PROCESS, not name similarity ----------------
def test_per_process_selection_routes_by_causal_need():
    from swm.world_model_v2.registry import load_registry, select_for_process
    st = load_registry(reload=True)
    r = select_for_process(st, "content_response", {"domain": "content_ab_test",
                                                    "available_state": ["populations"]})
    assert r["selected"] and r["selected"]["family_id"] == "content_response_click"


def test_adversarial_incompatible_family_not_selected():
    """A similarly-scoped but process-incompatible family must NOT be selected: diffusion families do not
    answer 'offer_response', trust families do not answer 'adoption_after_repeated_exposure'."""
    from swm.world_model_v2.registry import load_registry, select_for_process
    from swm.world_model_v2.registry.applicability import _process_match
    st = load_registry(reload=True)
    # no diffusion family answers offer_response
    r = select_for_process(st, "offer_response", {"domain": "economic_game"})
    picked = {r["selected"]["family_id"]} | {c["family_id"] for c in r["competing"]}
    assert "exposure_response_hazard" not in picked and "bass_diffusion" not in picked
    # process-match is 0 for the incompatible family
    assert _process_match(st.records["exposure_response_hazard"], "offer_response") == 0.0


# ---------------- transport engine ----------------
def test_transport_decisions():
    from swm.world_model_v2.registry.transport import assess_transport
    same = assess_transport({"domain": "x", "population": "y"}, {"domain": "x", "population": "y"})
    assert same.decision == "transport_direct" and same.widening == pytest.approx(1.0)
    far = assess_transport({"domain": "lab_economic_game", "population": "students",
                            "outcome_definition": "offer_share"},
                           {"domain": "political_donation", "population": "online_adults",
                            "outcome_definition": "vote_choice"})
    assert far.widening > 1.0 and far.decision in ("transport_widened", "experimental", "reject")
    # a decisive-axis mismatch + missing required var forces reject
    rej = assess_transport({"outcome_definition": "clicks"}, {"outcome_definition": "revenue"},
                           missing_required_vars=["price"])
    assert rej.decision == "reject"


# ---------------- composition engine ----------------
def test_composition_double_counting_and_competing():
    from swm.world_model_v2.registry.composition import compose
    sels = [
        {"process": "adoption_after_repeated_exposure",
         "selected": {"family_id": "exposure_response_hazard", "status": "locally_validated"},
         "competing": [{"family_id": "simple_contagion_hazard"}]},
        {"process": "cascade_saturation",
         "selected": {"family_id": "bass_diffusion", "status": "domain_restricted"}, "competing": []},
    ]
    plan = compose(sels)
    # both write the 'adoption' channel via different processes → double-counting flagged, single-writer kept
    assert any(dc["channel"] == "adoption" for dc in plan.double_counting)
    assert plan.competing and plan.competing[0]["competing"] == ["simple_contagion_hazard"]


# ---------------- registry integrity + HONEST lifecycle gates ----------------
def test_registry_builds_and_statuses_are_honest():
    from swm.world_model_v2.registry.build_registry import build
    s = build()
    # Hawkes failure preserved
    assert s.records["hawkes_self_excitation"].status == "quarantined"
    # verified published estimates are domain_restricted, NOT production
    assert s.records["bass_diffusion"].status == "domain_restricted"
    assert s.records["ultimatum_offer_response"].status == "domain_restricted"
    # local held-out win → locally_validated; its FAILED transfer blocks production
    assert s.records["attrition_dropout_hazard"].status == "locally_validated"
    assert s.records["attrition_dropout_hazard"].promotion_blockers("production_eligible")
    # honest NULLs are NOT locally validated
    assert s.records["response_occurrence_hazard"].status == "implemented"
    assert s.records["argument_persuasion_success"].status == "implemented"
    # the genuine new production-eligible content-response mechanism
    assert s.records["content_response_click"].status == "production_eligible"


def test_published_estimate_cannot_reach_production_without_local_validation():
    from swm.world_model_v2.registry.build_registry import build
    s = build()
    blockers = s.records["social_pressure_turnout"].promotion_blockers("production_eligible")
    assert blockers   # a verified published estimate is NOT production-eligible for arbitrary scenarios
