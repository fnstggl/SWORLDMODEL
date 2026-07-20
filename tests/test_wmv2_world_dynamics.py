"""World-dynamics layer after §NAP — observational persistence semantics, live qualitative stance
access, and the quarantine of the coupling/capacity/stance-rule numerics (offline)."""
import types

import pytest

from swm.world_model_v2.event_time import EventTimeContract, criterion_persistence_s
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.world_dynamics import (PersistenceCheckOperator, break_provisional_state,
                                               coupling_pack_info, live_stances,
                                               stance_state_hash)

T0 = 1_700_000_000.0
DAY = 86400.0


def _world(now=T0, **quants):
    w = WorldState(world_id="w", branch_id="b1:x", clock=SimulationClock(now=now, as_of=T0))
    for k, v in quants.items():
        register_quantity_type(k, units="unit")
        w.quantities[k] = Quantity(name=k, qtype=k, value=v, timestamp=now)
    return w


def _actor(world, aid, stances=None):
    e = Entity(aid)
    e.set("roles", F(["principal"], status="observed"))
    if stances is not None:
        e.set("stances", F(stances, status="observed", method="grounded"))
    world.entities[aid] = e
    return e


# ---------------------------------------------------------------- quarantine verdicts
def test_coupling_channel_is_quarantined():
    info = coupling_pack_info()
    assert info["source"] == "quarantined_no_production_coupling_channel"


def test_world_dynamics_exposes_no_numeric_tables():
    import swm.world_model_v2.world_dynamics as wd
    for name in ("COUPLING_PRIORS", "CAPACITY_INIT", "EFFORTFUL_ACTION_COST",
                 "EXHAUSTION_THRESHOLD", "RIPENESS_THRESHOLD", "BANDWAGON_THRESHOLD",
                 "sampled_coupling", "StanceReviewOperator", "declare_actor_capacity",
                 "contested_attrition_interval", "live_capacity"):
        assert not hasattr(wd, name), f"{name} must not exist in production world_dynamics"


def test_legacy_tables_require_the_ablation_token():
    from swm.world_model_v2.legacy_numeric_ablations import (ABLATION_TOKEN,
                                                             legacy_numeric_table)
    with pytest.raises(PermissionError):
        legacy_numeric_table("COUPLING_PRIORS")
    with pytest.raises(PermissionError):
        legacy_numeric_table("STANCE_ORIENTATION", acknowledge="please")
    t = legacy_numeric_table("COUPLING_PRIORS", acknowledge=ABLATION_TOKEN)
    assert t["pathway_step"][0] == 0.04                        # the buried historical value


# ---------------------------------------------------------------- live stance access (qualitative)
def test_live_stances_reads_current_entity_records():
    w = _world()
    st = [{"actor": "a", "commitment_level": "committed_to_prevent",
           "pathway": "cooperative_agreement"}]
    _actor(w, "a", stances=st)
    _actor(w, "b")
    assert live_stances(w) == st
    h1 = stance_state_hash(live_stances(w))
    w.entities["a"].set("stances", F([{**st[0], "commitment_level": "inclined_toward"}],
                                     status="derived"))
    assert stance_state_hash(live_stances(w)) != h1


# ---------------------------------------------------------------- persistence semantics
def test_persistence_criterion_parsing():
    assert criterion_persistence_s({"resolves_yes_iff":
                                    "no active hostilities for >=30 consecutive days"}) \
        == 30 * DAY
    assert criterion_persistence_s({"persistence_days": 7}) == 7 * DAY
    assert criterion_persistence_s({"resolves_yes_iff": "a deal is signed"}) == 0.0


def test_persistence_check_confirms_by_observation_no_coin():
    """A provisional end-state that HELD its whole window in the simulated world confirms —
    deterministically, by observation. There is no survival probability draw."""
    w = _world(provisional_absorbing_mode="ceasefire")
    op = PersistenceCheckOperator()
    ev = types.SimpleNamespace(etype="persistence_check",
                               payload={"mode": "ceasefire", "pathway": "cooperative_agreement"})
    assert op.applicable(w, ev)
    d = op.apply(w, op.propose(w, ev, None))
    assert "persisted_criterion_satisfied" in d.reason_codes
    assert w.quantities["absorbing_state_reached"].value is True
    assert d.uncertainty["semantics"] == "observational_no_survival_coin"


def test_persistence_check_collapses_only_via_modeled_break():
    w = _world(provisional_absorbing_mode="ceasefire")
    assert break_provisional_state(w, reason="shelling resumed (actor action consequence)")
    op = PersistenceCheckOperator()
    ev = types.SimpleNamespace(etype="persistence_check",
                               payload={"mode": "ceasefire", "pathway": "cooperative_agreement"})
    d = op.apply(w, op.propose(w, ev, None))
    assert "near_miss_realized_collapse" in d.reason_codes
    assert not w.quantities.get("absorbing_state_reached")
    assert "shelling resumed" in d.uncertainty["collapse_cause"]["reason"]


def test_break_provisional_state_is_noop_without_provisional():
    w = _world()
    assert break_provisional_state(w, reason="nothing to break") is False


def test_persistence_confirmation_is_deterministic_across_branches():
    """No coin: every branch with an unbroken provisional state confirms — the old 0.75/0.85
    sampled survival split cannot reappear as cross-branch randomness."""
    outcomes = set()
    for i in range(20):
        w = _world(provisional_absorbing_mode="deal")
        w.branch_id = f"b{i}:x"
        op = PersistenceCheckOperator()
        ev = types.SimpleNamespace(etype="persistence_check", payload={"mode": "deal"})
        op.apply(w, op.propose(w, ev, None))
        outcomes.add(bool(getattr(w.quantities.get("absorbing_state_reached"), "value", False)))
    assert outcomes == {True}


# ---------------------------------------------------------------- categorical readout is honest
def test_categorical_projection_maps_modes_with_honest_residual():
    from swm.world_model_v2.state import WorldBranch
    c = EventTimeContract(as_of=T0, horizon_ts=T0 + 100 * DAY,
                          categorical_options=["deal", "collapse"],
                          mode_option_map={"deal": "deal", "collapse": "collapse"}).validate()

    def b(mode=None):
        w = _world()
        if mode:
            register_quantity_type("absorbed_at", units="unix_ts")
            register_quantity_type("absorbed_by", units="mode")
            w.quantities["absorbed_at"] = Quantity(name="absorbed_at", qtype="absorbed_at",
                                                   value=T0 + 5 * DAY, timestamp=T0)
            w.quantities["absorbed_by"] = Quantity(name="absorbed_by", qtype="absorbed_by",
                                                   value=mode, timestamp=T0)
        return WorldBranch(branch_id=w.branch_id, world=w)
    out = c.project([b("deal"), b("deal"), b("collapse"), b()])
    d = out["distribution"]
    assert d["deal"] == pytest.approx(0.5)
    assert d["collapse"] == pytest.approx(0.25)
    assert d["none_of_the_options_by_horizon"] == pytest.approx(0.25)
