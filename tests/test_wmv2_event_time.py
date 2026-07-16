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
        _consumed_state=[{"var": "actor_intentions", "weight": 0.2}])
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
    # grounded intentions modulate ONLY agreement modes: refusal crushes deal hazards, never a
    # unilateral end-state's hazard
    by_mode = {e["payload"]["mode"]: e["payload"]["intention_factor"] for e in rounds}
    assert by_mode["negotiated_settlement"] == pytest.approx(0.2 + 1.4 * 0.2)
    assert by_mode["military_collapse"] == 1.0
    assert by_mode["frozen_conflict"] == 1.0
    ops = {m["operator"] for m in p.accepted_mechanisms}
    assert {"absorption_monitor", "hazard_round"} <= ops
    declared = {q["name"] for q in p.quantities}
    assert {"absorbed_at", "absorbing_state_reached"} <= declared   # readout binds at materialization
    assert 0.1 <= rep["intention_factor"] <= 1.6
    assert rep["intention_factor"] == pytest.approx(0.2 + 1.4 * 0.2)  # grounded intentions shape hazards
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


def test_fit_survival_pack_calibration_only_shape():
    rows = ([{"question": "Will X and Y sign a deal?", "lifetime_fraction_resolved": f}
             for f in (0.1, 0.3, None, None)] +
            [{"question": "Will team A win the match?", "lifetime_fraction_resolved": 0.9}])
    pack = fit_survival_pack(rows)
    assert pack["fit_on"] == "calibration split only"
    assert len(pack["global_hazards"]) == 5
    assert all(0.0 <= h <= 1.0 for h in pack["global_hazards"])
    assert set(pack["families"]) == {"meeting_or_deal_by_date", "sports_match"}
