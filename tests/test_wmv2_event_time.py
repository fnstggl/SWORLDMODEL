"""Event-time contract architecture — unit tests (first-passage timing, universal)."""
import types

import pytest

from swm.world_model_v2.event_time import (AbsorptionMonitorOperator, EventTimeContract,
                                           HazardRoundOperator, convert_to_event_time,
                                           fit_survival_pack, is_when_question)
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.state import SimulationClock, WorldBranch, WorldState

T0 = 1_700_000_000.0
T1 = T0 + 100 * 86400.0


def _world(now=T0, **quants):
    w = WorldState(world_id="w", branch_id="b1:x", clock=SimulationClock(now=now, as_of=T0))
    for k, v in quants.items():
        register_quantity_type(k, units="unit")
        w.quantities[k] = Quantity(name=k, qtype=k, value=v, timestamp=now)
    return w


def _branch(absorbed_at=None, absorbed_by=None):
    q = {}
    if absorbed_at is not None:
        q["absorbed_at"] = absorbed_at
        q["absorbed_by"] = absorbed_by or "modeA"
    w = _world(**q)
    return WorldBranch(branch_id=w.branch_id, world=w)


def test_projection_distribution_cdf_monotone_and_binary_view():
    c = EventTimeContract(as_of=T0, horizon_ts=T1, modes=["modeA"]).validate()
    branches = [_branch(T0 + f * (T1 - T0)) for f in (0.1, 0.3, 0.5)] + [_branch(), _branch()]
    out = c.project(branches)
    d = out["distribution"]
    assert d["absorbed_by_horizon"] == pytest.approx(0.6)
    assert d["censored_beyond_horizon"] == pytest.approx(0.4)
    et = out["event_time"]
    assert et["cdf"] == sorted(et["cdf"])                     # monotone by construction
    assert et["cdf"][-1] == pytest.approx(0.6)
    assert et["survival"][0] == pytest.approx(1 - et["cdf"][0])
    assert et["mode_distribution"] == {"modeA": pytest.approx(0.6)}
    # binary unification: P(yes by D) = F(D)
    assert c.cdf_at(T0 + 0.4 * (T1 - T0), branches) == pytest.approx(0.4)
    assert c.cdf_at(T1, branches) == pytest.approx(0.6)


def test_absorption_monitor_is_first_passage_and_immutable():
    op = AbsorptionMonitorOperator()
    w = _world(now=T0 + 5 * 86400, absorbing_state_reached=True, absorbing_mode="ceasefire")
    ev = types.SimpleNamespace(etype="anything", payload={})
    assert op.applicable(w, ev)
    d = op.apply(w, op.propose(w, ev, None))
    assert w.quantities["absorbed_at"].value == pytest.approx(w.clock.now)
    assert w.quantities["absorbed_by"].value == "ceasefire"
    assert d.event_type == "absorption"
    # first passage only: once stamped, never again applicable (immutability)
    w.clock.now += 86400
    assert not op.applicable(w, ev)
    assert w.quantities["absorbed_at"].value == pytest.approx(T0 + 5 * 86400)


def test_hazard_round_intention_factor_crushes_hazard():
    op = HazardRoundOperator()

    def run(intention_factor, seed_salt):
        hits = 0
        for i in range(400):
            w = _world(now=T0 + 30 * 86400)
            w.branch_id = f"b{i}:{seed_salt}"
            payload = {"mode": "deal", "base_hazard": 0.3, "intention_factor": intention_factor,
                       "as_of": T0, "span_s": T1 - T0, "consume": []}
            prop = op.propose(w, types.SimpleNamespace(etype="hazard_round", payload=payload), None)
            d = op.apply(w, prop)
            assert d.uncertainty["hazard"] <= 0.95
            if getattr(w.quantities.get("absorbing_state_reached"), "value", None):
                hits += 1
        return hits
    assert run(0.1, "a") < run(1.6, "a")                      # refusal crushes; commitment boosts


def test_hazard_round_not_applicable_after_absorption():
    op = HazardRoundOperator()
    w = _world(absorbed_at=T0 + 1)
    assert not op.applicable(w, types.SimpleNamespace(etype="hazard_round", payload={}))


def test_is_when_question_routing():
    assert is_when_question("When will the Russia-Ukraine conflict end?")
    assert is_when_question("How long until the Fed cuts rates?")
    assert not is_when_question("Will Powell be out as Fed Chair by 2025-12-31?")


def _plan():
    p = types.SimpleNamespace(
        question="When will the conflict end?", as_of=T0, horizon_ts=T1,
        structural_hypotheses=[{"id": "negotiated_settlement", "prior": 0.5},
                               {"id": "military_collapse", "prior": 0.25},
                               {"id": "frozen_conflict", "prior": 0.25}],
        outcome_contract=types.SimpleNamespace(options=["a", "b", "c"]),
        scheduled_events=[{"etype": "resolve_outcome", "ts": T1 - 1, "participants": [], "payload": {}}],
        accepted_mechanisms=[], quantities=[{"name": "actor_intentions", "qtype": "actor_intentions",
                                             "value": 0.2, "sd": None}],
        _intention_stances=[{"actor": "leader_a", "commitment_level": "categorical_refusal",
                             "reliability": "high", "entails": "no"},
                            {"actor": "leader_b", "commitment_level": "openness_to_agreement",
                             "reliability": "medium", "entails": "yes"}],
        _consumed_state=[{"var": "actor_intentions", "weight": 0.2}],
        compute_plan={"n_particles": 30})
    return p


def test_convert_to_event_time_full_rewire():
    p = _plan()
    lin = {}
    rep = convert_to_event_time(p, {"resolves_yes_iff": "an agreed end of hostilities holds"}, lineage=lin)
    assert isinstance(p.outcome_contract, EventTimeContract)
    assert p.outcome_contract.resolution_rule.startswith("an agreed end")
    assert not any(e["etype"] == "resolve_outcome" for e in p.scheduled_events)
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert rounds and rep["n_hazard_rounds"] == len(rounds)
    assert {e["payload"]["mode"] for e in rounds} == {"negotiated_settlement", "military_collapse",
                                                      "frozen_conflict"}
    assert all(e["payload"]["consume"] == [{"var": "actor_intentions", "weight": 0.2}] for e in rounds)
    assert all(T0 < e["ts"] < T1 for e in rounds)
    # total scheduled hazard mass equals the base envelope (0.5) — no len(modes) inflation
    assert sum(e["payload"]["base_hazard"] for e in rounds) == pytest.approx(0.5)
    # grounded stances modulate ONLY agreement modes, as a DISTRIBUTION (binding = most-opposed
    # veto actor: high-reliability categorical refusal → median 0.55), never a unilateral end-state
    by_mode = {e["payload"]["mode"]: e["payload"]["hr"] for e in rounds}
    assert by_mode["negotiated_settlement"]["median"] == pytest.approx(0.55)
    assert by_mode["negotiated_settlement"]["lo80"] < by_mode["negotiated_settlement"]["hi80"]
    assert by_mode["military_collapse"]["median"] == 1.0
    assert by_mode["frozen_conflict"]["median"] == 1.0
    assert rep["agreement_hazard_ratio"]["binding_actor"] == "leader_a"
    assert rep["hazard_ratio_source"] in ("documented_priors_unfitted", "fitted_pack")
    ops = {m["operator"] for m in p.accepted_mechanisms}
    assert {"absorption_monitor", "hazard_round"} <= ops
    declared = {q["name"] for q in p.quantities}
    assert {"absorbed_at", "absorbing_state_reached"} <= declared   # readout binds at materialization
    assert p.compute_plan["n_particles"] == 200               # CDF particle floor
    assert lin["event_time"] is rep


def test_mode_entailment_filter_drops_non_absorbing_modes():
    p = _plan()

    def fake_llm(prompt):
        return ('{"absorbing_mode_ids": ["negotiated_settlement", "military_collapse"], '
                '"rejected": [{"id": "frozen_conflict", "why": "not an end state"}]}')
    rep = convert_to_event_time(p, {"resolves_yes_iff": "hostilities formally end"}, llm=fake_llm)
    assert set(rep["modes"]) == {"negotiated_settlement", "military_collapse"}
    assert rep["rejected_non_absorbing_modes"] == ["frozen_conflict"]
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert not any(e["payload"]["mode"] == "frozen_conflict" for e in rounds)
    assert sum(e["payload"]["base_hazard"] for e in rounds) == pytest.approx(0.5)  # renormalized


def test_mode_entailment_filter_fails_open():
    p = _plan()

    def broken_llm(prompt):
        raise RuntimeError("llm down")
    rep = convert_to_event_time(p, {"resolves_yes_iff": "hostilities formally end"}, llm=broken_llm)
    assert len(rep["modes"]) == 3                              # filter must never block the forecast


def test_sampled_hazard_ratio_is_per_branch_persistent_and_bounded():
    op = HazardRoundOperator()
    w = _world(now=T0 + 30 * 86400)
    hr = {"median": 0.55, "lo80": 0.30, "hi80": 0.90}
    v1 = op._sampled_hr(w, "deal", hr)
    v2 = op._sampled_hr(w, "deal", hr)                        # same branch → same draw
    assert v1 == v2 and 0.05 <= v1 <= 3.0
    w2 = _world(now=T0 + 30 * 86400)
    w2.branch_id = "b7:y"
    draws = {op._sampled_hr(_w, "deal", hr) for _w in
             [w2] + [_world(now=T0) for _ in range(0)]}
    vals = []
    for i in range(50):
        wi = _world(now=T0 + 30 * 86400)
        wi.branch_id = f"b{i}:z"
        vals.append(op._sampled_hr(wi, "deal", hr))
    assert len(set(vals)) > 10                                # cross-particle spread survives
    assert all(0.05 <= v <= 3.0 for v in vals)


def test_agreement_hazard_ratio_binding_actor_and_reliability_shrink():
    from swm.world_model_v2.event_time import agreement_hazard_ratio
    # most-opposed veto actor binds
    hr = agreement_hazard_ratio([
        {"actor": "a", "commitment_level": "openness_to_agreement", "reliability": "high"},
        {"actor": "b", "commitment_level": "categorical_refusal", "reliability": "high"}])
    assert hr["binding_actor"] == "b" and hr["median"] == pytest.approx(0.55)
    # low reliability shrinks the effect toward 1.0
    hr_low = agreement_hazard_ratio([
        {"actor": "b", "commitment_level": "categorical_refusal", "reliability": "low"}])
    assert 0.55 < hr_low["median"] < 1.0
    # no grounded stances → no effect, honest default
    assert agreement_hazard_ratio([])["binding_level"] == "no_grounded_stance"


def test_sensitivity_overrides_force_point_ratios():
    import swm.world_model_v2.event_time as ET
    p = _plan()
    ET.AGREEMENT_HR_OVERRIDE, ET.VICTORY_HR_OVERRIDE = 0.5, 0.8
    try:
        rep = convert_to_event_time(p, {"resolves_yes_iff": "x"})
        assert rep["agreement_hazard_ratio"]["median"] == 0.5
        assert rep["hazard_ratio_by_mode"]["military_collapse"]["median"] == 0.8
        assert rep["agreement_hazard_ratio"]["binding_level"] == "sensitivity_override"
    finally:
        ET.AGREEMENT_HR_OVERRIDE = ET.VICTORY_HR_OVERRIDE = None


def test_mode_elicitation_when_compiler_declares_none():
    p = _plan()
    p.structural_hypotheses = []                              # compiler variance: no hypotheses
    p.outcome_contract = types.SimpleNamespace(options=["yes", "no"])   # and no categorical options

    def fake_llm(prompt):
        if "END-STATES" in prompt:
            return ('{"modes": [{"id": "peace_deal", "prior": 0.4, "requires_agreement": true},'
                    '{"id": "unilateral_collapse", "prior": 0.6, "requires_agreement": false}]}')
        return '{"absorbing_mode_ids": ["peace_deal", "unilateral_collapse"]}'
    rep = convert_to_event_time(p, {"resolves_yes_iff": "the dispute is settled"}, llm=fake_llm)
    assert set(rep["modes"]) == {"peace_deal", "unilateral_collapse"}
    # the semantic flag (not keywords) decides which modes the stance ratio touches
    assert rep["hazard_ratio_by_mode"]["peace_deal"]["requires_agreement"] is True
    assert rep["hazard_ratio_by_mode"]["peace_deal"]["median"] == pytest.approx(0.55)
    assert rep["hazard_ratio_by_mode"]["unilateral_collapse"]["requires_agreement"] is False
    assert rep["hazard_ratio_by_mode"]["unilateral_collapse"]["median"] == 1.0


def test_hazard_state_consumption_is_relative_not_absolute():
    op = HazardRoundOperator()
    # a mid-level consumed state (0.5) must be NO-EFFECT on the hazard — under the old absolute
    # blend it would swamp a 0.005 hazard to ~0.23 every round and absorb everything
    w = _world(now=T0 + 30 * 86400, nonlinear_state=0.5)
    h, used, f = op._consume_state_hazard(w, 0.005, [{"var": "nonlinear_state", "weight": 0.45}])
    assert used == ["nonlinear_state"] and f == pytest.approx(1.0) and h == pytest.approx(0.005)
    # extreme state at full weight: bounded multiplicative push (×2^0.45), never an absolute jump
    w2 = _world(now=T0, nonlinear_state=1.0)
    h2, _, f2 = op._consume_state_hazard(w2, 0.005, [{"var": "nonlinear_state", "weight": 0.45}])
    assert f2 == pytest.approx(2 ** 0.45, rel=1e-3) and h2 < 0.01
    w3 = _world(now=T0, nonlinear_state=0.0)
    h3, _, f3 = op._consume_state_hazard(w3, 0.005, [{"var": "nonlinear_state", "weight": 0.45}])
    assert f3 == pytest.approx(2 ** -0.45, rel=1e-3)


def test_time_indexed_duplicate_modes_merge():
    p = _plan()
    p.structural_hypotheses = [{"id": "ceasefire_2026", "prior": 0.2},
                               {"id": "ceasefire_2027", "prior": 0.3},
                               {"id": "military_collapse", "prior": 0.5}]
    rep = convert_to_event_time(p, {"resolves_yes_iff": "x"})
    assert set(rep["modes"]) == {"ceasefire", "military_collapse"}
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"
              and e["payload"]["mode"] == "ceasefire"]
    assert sum(e["payload"]["base_hazard"] for e in rounds) == pytest.approx(0.5 * 0.5)  # priors summed


def test_fit_intention_hazard_ratios_pools_toward_no_effect():
    from swm.world_model_v2.event_time import fit_intention_hazard_ratios
    rows = [{"commitment_level": "categorical_refusal", "hazard_ratio": r}
            for r in (0.4, 0.5, 0.6, 0.5, 0.45)]
    pack = fit_intention_hazard_ratios(rows)
    med = pack["hazard_ratios"]["categorical_refusal"][0]
    assert 0.5 < med < 1.0                                    # pooled toward 1.0 with n=5
    lo, hi = pack["hazard_ratios"]["categorical_refusal"][1:3]
    assert lo < med < hi


def test_fit_survival_pack_calibration_only_shape():
    rows = ([{"question": "Will X and Y sign a deal?", "lifetime_fraction_resolved": f}
             for f in (0.1, 0.3, None, None)] +
            [{"question": "Will team A win the match?", "lifetime_fraction_resolved": 0.9}])
    pack = fit_survival_pack(rows)
    assert pack["fit_on"] == "calibration split only"
    assert len(pack["global_hazards"]) == 5
    assert all(0.0 <= h <= 1.0 for h in pack["global_hazards"])
    assert set(pack["families"]) == {"meeting_or_deal_by_date", "sports_match"}
