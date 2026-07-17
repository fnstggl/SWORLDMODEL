"""World-dynamics layer — stance dynamics, persistence semantics, capacity, sampled couplings,
contested per-mode channels, categorical unification, actor perception (universal; offline)."""
import json
import types

import pytest

from swm.world_model_v2.event_time import (EventTimeContract, HazardRoundOperator,
                                           convert_to_event_time, criterion_persistence_s,
                                           declare_contested_mode_channels)
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldBranch, WorldState
from swm.world_model_v2.world_dynamics import (COUPLING_PRIORS, PersistenceCheckOperator,
                                               StanceReviewOperator, declare_actor_capacity,
                                               fit_coupling_pack, live_capacity, live_stances,
                                               sampled_coupling, stance_state_hash)

T0 = 1_700_000_000.0
DAY = 86400.0


def _world(now=T0, **quants):
    w = WorldState(world_id="w", branch_id="b1:x", clock=SimulationClock(now=now, as_of=T0))
    for k, v in quants.items():
        register_quantity_type(k, units="unit")
        w.quantities[k] = Quantity(name=k, qtype=k, value=v, timestamp=now)
    return w


def _actor(world, aid, stances=None, capacity=None):
    e = Entity(aid)
    e.set("roles", F(["principal"], status="observed"))
    if stances is not None:
        e.set("stances", F(stances, status="observed", method="grounded"))
    if capacity is not None:
        e.set("resources", F(capacity, status="observed"), key="capacity")
    world.entities[aid] = e
    return e


# ---------------------------------------------------------------- sampled couplings
def test_sampled_coupling_per_branch_persistent_clamped_and_spread():
    w = _world()
    v1 = sampled_coupling(w, "pathway_step")
    assert v1 == sampled_coupling(w, "pathway_step")           # persisted per branch
    lo, hi = 0.005, 0.15
    vals = set()
    for i in range(60):
        wi = _world()
        wi.branch_id = f"b{i}:c"
        vals.add(sampled_coupling(wi, "pathway_step"))
    assert len(vals) > 20 and all(lo <= v <= hi for v in vals)  # a distribution, inside clamps


def test_fit_coupling_pack_reweights_toward_low_crps_draws():
    rows = ([{"coupling_draws": {"pathway_step": 0.02}, "crps": 0.05} for _ in range(10)]
            + [{"coupling_draws": {"pathway_step": 0.10}, "crps": 0.60} for _ in range(10)])
    pack = fit_coupling_pack(rows)
    med = pack["couplings"]["pathway_step"][0]
    assert 0.02 <= med < 0.05                                  # pulled toward the better draws
    assert pack["n_trajectories"] == 20


# ---------------------------------------------------------------- stance dynamics
RIPE_STANCES = [
    {"actor": "loser", "commitment_level": "committed_to_prevent", "reliability": "high",
     "capability": "high", "pathway": "cooperative_agreement", "control": "veto"},
    {"actor": "loser", "commitment_level": "committed_to_prevent", "reliability": "high",
     "capability": "high", "pathway": "unilateral_action", "control": "operational_capability",
     "target_mode": "rival_victory"},
]


def _review(world, round_=5):
    op = StanceReviewOperator()
    ev = types.SimpleNamespace(etype="stance_review", payload={"round": round_})
    assert op.applicable(world, ev)
    return op.apply(world, op.propose(world, ev, None))


def test_ripeness_softens_shared_refusal_when_rival_mode_nears_completion():
    w = _world(**{"mode_progress:unilateral_action:rival_victory": 0.75})
    _actor(w, "loser", stances=[dict(s) for s in RIPE_STANCES])
    d = _review(w)
    assert d is not None and any("ripeness" in r for r in d.reason_codes)
    new = live_stances(w)
    coop = next(s for s in new if s["pathway"] == "cooperative_agreement")
    assert coop["commitment_level"] == "conditionally_opposed"   # softened exactly one level
    assert coop["updates"][0]["rule"].startswith("ripeness")
    # ... and NOT again next review (cooldown/hysteresis)
    w.clock.now += 30 * DAY
    op = StanceReviewOperator()
    ev = types.SimpleNamespace(etype="stance_review", payload={"round": 6})
    assert op.apply(w, op.propose(w, ev, None)) is None


def test_winning_hardens_and_exhaustion_softens():
    # WINNING: own pursued mode near completion → shared openness hardens one level
    w = _world(**{"mode_progress:unilateral_action:own_victory": 0.8})
    _actor(w, "winner", stances=[
        {"actor": "winner", "commitment_level": "inclined_toward", "reliability": "high",
         "capability": "high", "pathway": "cooperative_agreement", "control": "veto"},
        {"actor": "winner", "commitment_level": "actively_pursuing", "reliability": "high",
         "capability": "high", "pathway": "unilateral_action", "control": "sole_authority",
         "target_mode": "own_victory"}])
    d = _review(w)
    assert d is not None and any("winning" in r for r in d.reason_codes)
    coop = next(s for s in live_stances(w) if s["pathway"] == "cooperative_agreement")
    assert coop["commitment_level"] == "neutral"               # inclined → neutral (hardened)
    # EXHAUSTION: drained capacity → pursue-stance on a per-actor pathway softens
    w2 = _world()
    _actor(w2, "tired", capacity=0.2, stances=[
        {"actor": "tired", "commitment_level": "actively_pursuing", "reliability": "high",
         "capability": "high", "pathway": "unilateral_action", "control": "sole_authority",
         "target_mode": "own_victory"}])
    d2 = _review(w2)
    assert d2 is not None and any("exhaustion" in r for r in d2.reason_codes)
    assert live_stances(w2)[0]["commitment_level"] == "inclined_toward"


def test_bandwagon_and_honest_noop():
    w = _world(**{"pathway_progress:cooperative_agreement": 0.75})
    _actor(w, "holdout", stances=[
        {"actor": "holdout", "commitment_level": "weakly_opposed", "reliability": "medium",
         "capability": "high", "pathway": "cooperative_agreement", "control": "coalition_member"}])
    d = _review(w)
    assert d is not None and any("bandwagon" in r for r in d.reason_codes)
    assert live_stances(w)[0]["commitment_level"] == "neutral"
    # nothing triggered → honest no-op (None), no fabricated drift
    w2 = _world(**{"pathway_progress:cooperative_agreement": 0.4})
    _actor(w2, "steady", stances=[
        {"actor": "steady", "commitment_level": "committed_to_prevent", "reliability": "high",
         "capability": "high", "pathway": "cooperative_agreement", "control": "veto"}])
    assert _review(w2) is None


def test_hazard_round_recomputes_hr_from_live_stances():
    """The stance-review operator rewrites records mid-trajectory; the NEXT hazard round must see
    the new stance state (live recompute + re-sampled effect size), not the baked one."""
    op = HazardRoundOperator()
    hard = [{"actor": "p", "commitment_level": "committed_to_prevent", "reliability": "high",
             "capability": "high", "pathway": "cooperative_agreement", "control": "veto"}]
    soft = [dict(hard[0], commitment_level="weakly_opposed")]
    payload = {"mode": "deal", "base_hazard": 0.02, "as_of": T0, "span_s": 100 * DAY,
               "hr": {"median": 0.55, "lo80": 0.55, "hi80": 0.55},
               "mode_def": {"id": "deal", "pathway": "cooperative_agreement",
                            "decision_structure": {"rule": "unanimity"}},
               "stances_hash": stance_state_hash(hard), "consume": []}
    ev = types.SimpleNamespace(etype="hazard_round", payload=payload)

    def mean_hr(stances, n=60):
        tot = 0.0
        for i in range(n):
            w = _world()
            w.branch_id = f"b{i}:live"
            _actor(w, "p", stances=[dict(s) for s in stances])
            d = op.apply(w, op.propose(w, ev, None))
            tot += d.uncertainty["sampled_hazard_ratio"]
        return tot / n
    # unchanged stances → baked distribution serves (median 0.55, degenerate interval)
    assert mean_hr(hard) == pytest.approx(0.55, abs=0.03)
    # softened stances → recomputed ratio is materially HIGHER (weakly_opposed ≈ 0.90 median)
    assert mean_hr(soft) > mean_hr(hard) + 0.15


# ---------------------------------------------------------------- persistence semantics
def test_persistence_criterion_parsing():
    assert criterion_persistence_s({"resolves_yes_iff": "no active hostilities for >=30 consecutive days"}) \
        == pytest.approx(30 * DAY)
    assert criterion_persistence_s({"resolves_yes_iff": "the price stays there for 2 weeks"}) \
        == pytest.approx(14 * DAY)
    assert criterion_persistence_s({"persistence_days": 45}) == pytest.approx(45 * DAY)
    assert criterion_persistence_s({"resolves_yes_iff": "a treaty is signed"}) == 0.0


def test_hazard_success_is_provisional_under_persistence_and_pauses_hazards():
    op = HazardRoundOperator()
    payload = {"mode": "ceasefire", "base_hazard": 0.999, "as_of": T0, "span_s": 100 * DAY,
               "persistence_s": 30 * DAY, "pathway": "cooperative_agreement", "consume": []}
    ev = types.SimpleNamespace(etype="hazard_round", payload=payload)
    w = _world(now=T0 + 10 * DAY)
    d = op.apply(w, op.propose(w, ev, None))
    assert w.quantities["provisional_absorbing_mode"].value == "ceasefire"
    assert "absorbing_state_reached" not in w.quantities        # NOT absorbed yet
    fu = d.follow_up_events
    assert fu and fu[0]["etype"] == "persistence_check"
    assert fu[0]["ts"] == pytest.approx(w.clock.now + 30 * DAY)
    # while provisional pending, hazard rounds pause (the world IS in the candidate end-state)
    assert not op.applicable(w, ev)


def test_persistence_check_confirms_or_collapses():
    pc = PersistenceCheckOperator()
    ev = types.SimpleNamespace(etype="persistence_check",
                               payload={"mode": "ceasefire", "pathway": "cooperative_agreement"})
    held, collapsed = 0, 0
    for i in range(120):
        w = _world(now=T0 + 40 * DAY,
                   provisional_absorbing_mode="ceasefire",
                   **{"pathway_progress:cooperative_agreement": 0.6})
        w.branch_id = f"b{i}:pc"
        assert pc.applicable(w, ev)
        d = pc.apply(w, pc.propose(w, ev, None))
        if getattr(w.quantities.get("absorbing_state_reached"), "value", None):
            held += 1
            assert w.quantities["absorbing_mode"].value == "ceasefire"
        else:
            collapsed += 1
            assert not w.quantities["provisional_absorbing_mode"].value
            # the near-miss knocked the process back
            assert w.quantities["pathway_progress:cooperative_agreement"].value < 0.6
            assert any("near_miss_realized_collapse" in r for r in d.reason_codes)
    # shared-pathway survival prior centers ~0.75: both outcomes must actually occur
    assert held > 60 and collapsed > 10


# ---------------------------------------------------------------- contested channels + capacity
def test_contested_mode_channels_and_principals_declared():
    plan = types.SimpleNamespace(quantities=[{"name": "pathway_progress:unilateral_action",
                                              "qtype": "pathway_progress", "value": 0.5}],
                                 _declared_pathways=["unilateral_action"])
    modes = [{"id": "a_victory", "pathway": "unilateral_action"},
             {"id": "b_victory", "pathway": "unilateral_action"},
             {"id": "deal", "pathway": "cooperative_agreement",
              "decision_structure": {"rule": "unanimity", "approvers": ["A", "B"]}}]
    rep = declare_contested_mode_channels(plan, modes)
    names = {q["name"]: q for q in plan.quantities}
    assert "mode_progress:unilateral_action:a_victory" in names
    assert "mode_progress:unilateral_action:b_victory" in names
    assert names["mode_progress:unilateral_action:a_victory"]["value"] == pytest.approx(0.5)
    assert "mode_progress:cooperative_agreement:deal" not in names   # shared stays one process
    assert names["pathway_principals:cooperative_agreement"]["value"] == "A|B"
    # gated off for bare plans
    bare = types.SimpleNamespace(quantities=[])
    assert declare_contested_mode_channels(bare, modes)["skipped"]


def test_contested_writes_advance_own_mode_and_suppress_rival():
    from swm.world_model_v2.phase4_execution import ActorPolicyRuntime as RT
    from swm.world_model_v2.phase4_policy import ACTION_ONTOLOGY, ActionTarget, TypedAction
    w = _world(**{"pathway_progress:unilateral_action": 0.5,
                  "mode_progress:unilateral_action:a_victory": 0.5,
                  "mode_progress:unilateral_action:b_victory": 0.5})
    _actor(w, "A", stances=[{"actor": "A", "commitment_level": "actively_pursuing",
                             "reliability": "high", "capability": "high",
                             "pathway": "unilateral_action", "control": "sole_authority",
                             "target_mode": "a_victory"}])
    act = TypedAction(action_id="a:mobilize", actor_id="A", actor_role="principal",
                      action_family="participation", action_name="mobilize", target=ActionTarget(),
                      mechanisms_triggered=["record_action"])

    class _D:
        def __init__(self):
            self.changes = []

        def change(self, path, before, after):
            self.changes.append(path)
    RT._apply_pathway_effects(w, act, _D())
    a = w.quantities["mode_progress:unilateral_action:a_victory"].value
    b = w.quantities["mode_progress:unilateral_action:b_victory"].value
    assert a > 0.5 > b                                          # own campaign up, rival suppressed
    assert w.quantities["pathway_progress:unilateral_action"].value > 0.5   # aggregate spillover


def test_principal_weighting_and_capacity_scaling():
    from swm.world_model_v2.phase4_execution import ActorPolicyRuntime as RT
    from swm.world_model_v2.phase4_policy import ActionTarget, TypedAction

    def move(actor, principals, capacity=None):
        w = _world(**{"pathway_progress:cooperative_agreement": 0.5})
        if principals:
            register_quantity_type("pathway_principals", units="names")
            w.quantities["pathway_principals:cooperative_agreement"] = Quantity(
                name="pathway_principals:cooperative_agreement", qtype="pathway_principals",
                value=principals, timestamp=T0)
        _actor(w, actor, capacity=capacity)
        act = TypedAction(action_id="a:accept", actor_id=actor, actor_role="principal",
                          action_family="negotiation", action_name="accept", target=ActionTarget(),
                          mechanisms_triggered=["record_action"])

        class _D:
            changes = []

            def change(self, *a):
                pass
        RT._apply_pathway_effects(w, act, _D())
        return w.quantities["pathway_progress:cooperative_agreement"].value - 0.5
    assert move("A", "A|B") > move("C", "A|B") > 0              # principal moves talks more
    assert move("A", "A|B", capacity=0.9) > move("A", "A|B", capacity=0.15)   # exhausted moves less


def test_declare_actor_capacity_from_grounded_capability():
    plan = types.SimpleNamespace(
        entities=[{"id": "A", "type": "person", "fields": {}},
                  {"id": "B", "type": "person", "fields": {}}],
        _intention_stances=[{"actor": "A", "capability": "high"},
                            {"actor": "B", "capability": "low"}])
    rep = declare_actor_capacity(plan)
    assert rep["initialized"] == {"A": 0.85, "B": 0.35}
    assert plan.entities[0]["fields"]["resources"]["capacity"] == 0.85


# ---------------------------------------------------------------- categorical + perception
def test_categorical_projection_maps_modes_to_option_labels_with_honest_residual():
    c = EventTimeContract(as_of=T0, horizon_ts=T0 + 100 * DAY,
                          modes=["deal", "collapse"],
                          categorical_options=["Deal", "Collapse", "Other thing"],
                          mode_option_map={"deal": "Deal", "collapse": "Collapse"}).validate()
    assert c.options == ["Deal", "Collapse", "Other thing", "none_of_the_options_by_horizon"]

    def _branch(absorbed_frac=None, mode=None):
        q = {}
        if absorbed_frac is not None:
            q = {"absorbed_at": T0 + absorbed_frac * 100 * DAY, "absorbed_by": mode}
        w = _world(**q)
        return WorldBranch(branch_id=w.branch_id, world=w)
    branches = ([_branch(0.2, "deal")] * 2 + [_branch(0.5, "collapse")] * 1
                + [_branch(0.6, "unmapped_mode")] * 1 + [_branch()] * 4)
    out = c.project(branches)
    d = out["distribution"]
    assert d["Deal"] == pytest.approx(0.25) and d["Collapse"] == pytest.approx(0.125)
    assert d["Other thing"] == 0.0
    # censored worlds + absorbed-by-unmapped-mode = honest residual, never force-picked
    assert d["none_of_the_options_by_horizon"] == pytest.approx(0.625)
    assert out["event_time"]["unmapped_absorbed_share"] == pytest.approx(0.125)


def test_actor_view_projects_public_process_state_never_outcome_state():
    from swm.world_model_v2.phase4_policy import ActorViewBuilder
    w = _world(**{"pathway_progress:cooperative_agreement": 0.3,
                  "mode_progress:unilateral_action:a_victory": 0.7,
                  "nonlinear_state": 0.62})
    register_quantity_type("absorbed_at", units="unix_ts")
    w.quantities["absorbed_at"] = Quantity(name="absorbed_at", qtype="absorbed_at",
                                           value=T0 + 5 * DAY, timestamp=T0)
    register_quantity_type("sampled_coupling", units="coefficient")
    w.quantities["sampled_coupling:pathway_step"] = Quantity(
        name="sampled_coupling:pathway_step", qtype="sampled_coupling", value=0.04, timestamp=T0)
    _actor(w, "A")
    view = ActorViewBuilder().build(w, "A")
    assert view.beliefs["process:pathway_progress:cooperative_agreement"] == pytest.approx(0.3)
    assert view.beliefs["process:mode_progress:unilateral_action:a_victory"] == pytest.approx(0.7)
    assert view.beliefs["process:nonlinear_state"] == pytest.approx(0.62)
    assert not any("absorbed" in k or "sampled_coupling" in k for k in view.beliefs)


# ---------------------------------------------------------------- conversion integration
def test_convert_schedules_stance_reviews_persistence_and_mode_defs():
    p = types.SimpleNamespace(
        question="When will the conflict end?", as_of=T0, horizon_ts=T0 + 899 * DAY,
        structural_hypotheses=[{"id": "deal", "prior": 0.5, "pathway": "cooperative_agreement",
                                "decision_structure": {"rule": "unanimity", "approvers": ["A", "B"]}},
                               {"id": "a_victory", "prior": 0.5, "pathway": "unilateral_action"}],
        outcome_contract=types.SimpleNamespace(options=[]),
        scheduled_events=[], accepted_mechanisms=[],
        quantities=[{"name": "pathway_progress:cooperative_agreement",
                     "qtype": "pathway_progress", "value": 0.3},
                    {"name": "pathway_progress:unilateral_action",
                     "qtype": "pathway_progress", "value": 0.5}],
        _declared_pathways=["cooperative_agreement", "unilateral_action"],
        _intention_stances=[{"actor": "A", "commitment_level": "committed_to_prevent",
                             "reliability": "high", "pathway": "cooperative_agreement",
                             "control": "veto"}],
        _consumed_state=[], compute_plan={"n_particles": 30})
    rep = convert_to_event_time(
        p, {"resolves_yes_iff": "hostilities end with no active hostilities for >=30 consecutive days"})
    assert rep["persistence_window_days"] == 30
    assert rep["n_stance_reviews"] == rep["rounds_per_mode"]
    assert rep["rounds_per_mode"] == 40                        # 899d horizon → fine timing grid
    reviews = [e for e in p.scheduled_events if e["etype"] == "stance_review"]
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert len(reviews) == 40 and all(e["payload"]["round"] for e in reviews)
    for e in rounds:
        pl = e["payload"]
        assert pl["persistence_s"] == pytest.approx(30 * DAY)
        assert pl["mode_def"]["id"] == pl["mode"] and pl["stances_hash"]
        assert pl["endogenous_live"] is True
        assert any(c.get("coupling") == "own_pathway_weight" for c in pl["consume"])
    # contested channel declared for the unilateral mode; unsplit hr in payload, split in report
    names = {q["name"] for q in p.quantities}
    assert "mode_progress:unilateral_action:a_victory" in names
    ops = {m["operator"] for m in p.accepted_mechanisms}
    assert {"stance_review", "persistence_check", "hazard_round", "absorption_monitor"} <= ops
    assert rep["hazard_ratio_by_mode"]["deal"]["split_applied_at"] == "runtime_sampled"
    assert rep["coupling_source"]["source"] in ("documented_priors_unfitted", "fitted_pack")
