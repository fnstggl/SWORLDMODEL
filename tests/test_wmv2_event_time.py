"""Event-time contract architecture — unit tests (first-passage timing, universal).

Includes the binary unification: the answer to a binary deadline question is a READOUT of the same
first-passage trajectories (P(yes) = F(deadline), polarity-mapped) — never a resolver's draw."""
import types

import pytest

from swm.world_model_v2.event_time import (AbsorptionMonitorOperator, EventTimeContract,
                                           HazardRoundOperator, _lexical_event_polarity,
                                           _mass_weights_from_curve, convert_binary_to_event_time,
                                           convert_to_event_time, fit_survival_pack, is_when_question)
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
        _intention_stances=[{"actor": "leader_a", "commitment_level": "committed_to_prevent",
                             "reliability": "high", "pathway": "cooperative_agreement",
                             "controls_pathway": True, "entails": "no"},
                            {"actor": "leader_b", "commitment_level": "inclined_toward",
                             "reliability": "medium", "pathway": "cooperative_agreement",
                             "controls_pathway": True, "entails": "yes"}],
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
    # most-opposed veto actor binds; legacy agreement-specific labels map onto the universal set
    hr = agreement_hazard_ratio([
        {"actor": "a", "commitment_level": "openness_to_agreement", "reliability": "high"},
        {"actor": "b", "commitment_level": "categorical_refusal", "reliability": "high"}])
    assert hr["binding_actor"] == "b" and hr["median"] == pytest.approx(0.55)
    # low reliability shrinks the effect toward 1.0
    hr_low = agreement_hazard_ratio([
        {"actor": "b", "commitment_level": "committed_to_prevent", "reliability": "low"}])
    assert 0.55 < hr_low["median"] < 1.0
    # no grounded stances → no effect, honest default
    assert agreement_hazard_ratio([])["binding_level"] == "no_grounded_stance"


def test_mode_hazard_ratio_is_pathway_aware_universal():
    from swm.world_model_v2.event_time import mode_hazard_ratio
    ctrl = {"actor": "ceo", "commitment_level": "committed_to_prevent", "reliability": "high",
            "pathway": "unilateral_action", "control": "sole_authority"}
    # sole-authority controller of a unilateral pathway binds it at full effect (a CEO refusing to
    # launch) — the unilateral rule derives from the pathway's default decision structure
    hr = mode_hazard_ratio([ctrl], "unilateral_action")
    assert hr["binding_actor"] == "ceo" and hr["median"] == pytest.approx(0.55)
    assert hr["combination_rule"] == "unilateral"
    # ... and their stance does NOT touch cooperative modes (different pathway)
    assert mode_hazard_ratio([ctrl], "cooperative_agreement")["binding_level"] == "no_grounded_stance"
    # informal influence over someone else's unilateral act moves its hazard far less than authority
    res = dict(ctrl, control="informal_influence")
    med_res = mode_hazard_ratio([res], "unilateral_action")["median"]
    assert 0.55 < med_res < 1.0 and med_res == pytest.approx(0.55 ** 0.3, rel=1e-3)
    # legacy controls_pathway booleans still resolve (True→veto, False→informal_influence)
    legacy = dict(ctrl, control=None, controls_pathway=False)
    assert mode_hazard_ratio([legacy], "unilateral_action")["median"] == pytest.approx(0.55 ** 0.3,
                                                                                       rel=1e-3)
    # an actively-pursuing controller RAISES the pathway's hazard (works for resignations, launches)
    go = {"actor": "ceo", "commitment_level": "actively_pursuing", "reliability": "high",
          "pathway": "unilateral_action", "control": "sole_authority"}
    assert mode_hazard_ratio([go], "unilateral_action")["median"] == pytest.approx(1.70)
    # pathway 'any' stances apply everywhere
    anyst = {"actor": "x", "commitment_level": "conditionally_opposed", "reliability": "high",
             "pathway": "any", "control": "veto"}
    assert mode_hazard_ratio([anyst], "institutional_procedure")["median"] < 1.0


def test_decision_structure_derives_combination_rule_not_most_opposed():
    """'Most-opposed binds' is the unanimity case, NOT a universal law: under a majority structure
    the most-opposed legislator does not determine passage — the weighted center does."""
    from swm.world_model_v2.event_time import mode_hazard_ratio
    stances = [
        {"actor": "sen_blocker", "commitment_level": "committed_to_prevent", "reliability": "high",
         "pathway": "institutional_procedure", "control": "coalition_member"},
        {"actor": "sen_a", "commitment_level": "actively_pursuing", "reliability": "high",
         "pathway": "institutional_procedure", "control": "coalition_member"},
        {"actor": "sen_b", "commitment_level": "actively_pursuing", "reliability": "high",
         "pathway": "institutional_procedure", "control": "coalition_member"}]
    # unanimity (a treaty): the single blocker binds → median crushed below 1
    treaty = {"id": "treaty", "decision_structure": {"rule": "unanimity"}}
    hr_u = mode_hazard_ratio(stances, "institutional_procedure", mode=treaty)
    assert hr_u["combination_rule"] == "unanimity" and hr_u["median"] == pytest.approx(0.55)
    assert hr_u["binding_actor"] == "sen_blocker"
    # majority (a bill): 2-of-3 pursue → combined ratio ABOVE 1 despite the blocker
    bill = {"id": "bill", "decision_structure": {"rule": "majority"}}
    hr_m = mode_hazard_ratio(stances, "institutional_procedure", mode=bill)
    assert hr_m["combination_rule"] == "majority" and hr_m["median"] > 1.0
    # aggregation (a market): stances shrink to near-irrelevance
    mkt = {"id": "adoption", "decision_structure": {"rule": "aggregation"}}
    hr_a = mode_hazard_ratio(stances, "market_aggregation", mode=mkt)
    assert 0.9 < hr_a["median"] < 1.15
    # none (a physical process): stances have NO effect — a hurricane has no stance
    phys = {"id": "landfall", "pathway": "physical_process"}
    hr_p = mode_hazard_ratio(stances, "physical_process", mode=phys)
    assert hr_p["median"] == 1.0 and hr_p["binding_level"] == "pathway_not_stance_driven"


def test_mode_scoped_stances_beat_pathway_scoped_and_ignore_other_modes():
    """stance(actor, mode): Russia can pursue ITS victory while committed to preventing UKRAINE'S —
    a stance targeting another mode must not bind this mode's hazard."""
    from swm.world_model_v2.event_time import mode_hazard_ratio
    stances = [
        {"actor": "russia", "commitment_level": "actively_pursuing", "reliability": "high",
         "pathway": "unilateral_action", "control": "sole_authority",
         "target_mode": "russian_victory"},
        {"actor": "russia", "commitment_level": "committed_to_prevent", "reliability": "high",
         "pathway": "unilateral_action", "control": "operational_capability",
         "target_mode": "ukrainian_victory"},
        {"actor": "ukraine", "commitment_level": "actively_pursuing", "reliability": "high",
         "pathway": "unilateral_action", "control": "sole_authority",
         "target_mode": "ukrainian_victory"}]
    rus = mode_hazard_ratio(stances, "unilateral_action", mode={"id": "russian_victory"})
    # only Russia's own pursue-stance targets this mode → hazard RAISED, binding actor = russia
    assert rus["median"] == pytest.approx(1.70) and rus["binding_actor"] == "russia"
    ukr = mode_hazard_ratio(stances, "unilateral_action", mode={"id": "ukrainian_victory"})
    # Ukraine pursues its own victory; Russia (operational capability) resists THAT mode — combined
    # unilateral rule: controller binds, resistance shrinks the product below the pure 1.70
    assert ukr["median"] < 1.70 and ukr["median"] > 0.55
    # a mode neither stance targets falls back to pathway/'any'-scoped stances only → neutral here
    other = mode_hazard_ratio(stances, "cooperative_agreement", mode={"id": "ceasefire"})
    assert other["binding_level"] == "no_grounded_stance"


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
    # the semantic pathway label (not keywords) decides which modes the stance ratio touches
    assert rep["hazard_ratio_by_mode"]["peace_deal"]["pathway"] == "cooperative_agreement"
    assert rep["hazard_ratio_by_mode"]["peace_deal"]["median"] == pytest.approx(0.55)
    assert rep["hazard_ratio_by_mode"]["unilateral_collapse"]["pathway"] == "unilateral_action"
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
    # legacy labels canonicalize onto the universal taxonomy at fit time
    med = pack["hazard_ratios"]["committed_to_prevent"][0]
    assert 0.5 < med < 1.0                                    # pooled toward 1.0 with n=5
    lo, hi = pack["hazard_ratios"]["committed_to_prevent"][1:3]
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


# ================================================================ binary unification (readout, no resolver)
def _binary_plan(question="Will X be out as CEO by the deadline?", options=("yes", "no"),
                 posterior=None, extra_events=None, consumed=None):
    contract = types.SimpleNamespace(family="binary", options=list(options), resolution_rule="")
    return types.SimpleNamespace(
        question=question, as_of=T0, horizon_ts=T1, outcome_contract=contract,
        scheduled_events=[{"etype": "resolve_outcome", "ts": T1 - 1.0, "participants": [],
                           "payload": {"outcome_var": "outcome", "family": "binary",
                                       "options": list(options), "lean": "neutral"}}]
        + list(extra_events or []),
        accepted_mechanisms=[], quantities=[], provenance={"outcome_lean": "neutral"},
        posterior_rate_particles=list(posterior or []) or None,
        _consumed_state=list(consumed or []))


def _rollout(p, n_particles=300, seed=3, extra_ops=None, meta=None):
    from swm.world_model_v2.events import Event, EventQueue
    from swm.world_model_v2.rollout import RolloutEngine
    ops = [HazardRoundOperator(), AbsorptionMonitorOperator()] + list(extra_ops or [])
    branches = []
    for i in range(n_particles):
        w = _world()
        w.branch_id = f"b{i:03d}"
        if meta:
            w.uncertainty_meta = dict(meta)
        q = EventQueue(horizon_ts=p.horizon_ts)
        for ev in p.scheduled_events:
            q.schedule(Event(ts=ev["ts"], etype=ev["etype"],
                             participants=list(ev.get("participants") or []),
                             payload=dict(ev["payload"]), source="scheduled"))
        branches.append(RolloutEngine(operators=ops).run_branch(w, q, seed=seed * 7919 + i))
    return p.outcome_contract.project(branches), branches


def test_binary_conversion_removes_every_resolver_and_rewires():
    p = _binary_plan(posterior=[(0.3, 1.0)], consumed=[{"var": "actor_intentions", "weight": 0.2}],
                     extra_events=[{"etype": "aggregate_outcome_resolution", "ts": T1 - 0.5,
                                    "participants": [], "payload": {"outcome_var": "outcome",
                                                                    "options": ["yes", "no"],
                                                                    "consume": []}}])
    lin = {}
    rep = convert_binary_to_event_time(p, {"resolves_yes_iff": "X departs the CEO role"}, lineage=lin)
    assert rep["contract"] == "binary_first_passage" and rep["n_resolver_events_removed"] == 2
    assert not any(e["etype"] in ("resolve_outcome", "aggregate_outcome_resolution")
                   for e in p.scheduled_events)                # NO component declares the answer
    assert isinstance(p.outcome_contract, EventTimeContract)
    assert p.outcome_contract.options == ["yes", "no"]         # the question's own options project
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert rounds and all("calibration" in e["payload"] for e in rounds)
    assert sum(e["payload"]["calibration"]["exponent"] for e in rounds) == pytest.approx(1.0)
    assert all(e["payload"]["consume"] == [{"var": "actor_intentions", "weight": 0.2}] for e in rounds)
    assert all(T0 < e["ts"] <= p.outcome_contract.deadline_ts for e in rounds)
    order = [m["operator"] for m in p.accepted_mechanisms]
    # absorbing writers run BEFORE the monitor so first passage stamps at the write's own clock time
    assert order.index("hazard_round") < order.index("absorption_monitor")
    assert {q["name"] for q in p.quantities} >= {"absorbed_at", "absorbing_state_reached"}
    assert lin["event_time"] is rep and rep["posterior_calibrated"] is True


def test_binary_residual_chain_consumes_declared_pathway_processes():
    """The endogenous channel is universal: a binary question whose plan declared pathway processes
    (from the stances' own named pathways) gets them on the residual chain — actions move the binary
    answer through the same state the timing questions consume; survival polarity inverts them."""
    p = _binary_plan(question="Will X remain CEO through the deadline?",
                     consumed=[{"var": "actor_intentions", "weight": 0.2}])
    p.quantities = [{"name": "pathway_progress:unilateral_action", "qtype": "pathway_progress",
                     "value": 0.5, "sd": 0.15}]
    p._declared_pathways = ["unilateral_action"]
    rep = convert_binary_to_event_time(p, {"resolves_yes_iff": "X remains CEO",
                                           "event_polarity": "occurrence_resolves_no"})
    assert rep["endogenous_channels"] == ["pathway_progress:unilateral_action"]
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    chan = [c for e in rounds for c in e["payload"]["consume"]
            if c["var"] == "pathway_progress:unilateral_action"]
    assert chan and all(c["weight"] == 0.6 for c in chan)
    # survival polarity: the process advancing the state-BREAKING event must be inverted
    assert all(c.get("invert") for c in chan)


def test_binary_answer_is_first_passage_readout_matching_posterior_target():
    p = _binary_plan(posterior=[(0.3, 1.0)])
    convert_binary_to_event_time(p, {})
    out, _ = _rollout(p, n_particles=400)
    d = out["distribution"]
    assert set(d) == {"yes", "no"}
    # the calibrated chain's total absorbed mass reproduces the posterior target — but as OBSERVED events
    assert d["yes"] == pytest.approx(0.3, abs=0.07)
    et = out["event_time"]
    assert et["cdf"] == sorted(et["cdf"])                      # multi-cutoff coherence by construction
    assert et["p_event_by_deadline"] == d["yes"]               # the binary answer IS the timing readout
    assert et["mode_distribution"].get("resolution", 0.0) == pytest.approx(d["yes"], abs=1e-9)
    assert out["readout"] == "terminal_states"


def test_binary_survival_polarity_maps_absorption_to_no():
    p = _binary_plan(question="Will X remain CEO through the deadline?", posterior=[(0.7, 1.0)])
    rep = convert_binary_to_event_time(p, {})
    assert rep["event_polarity"] == "occurrence_resolves_no" and rep["polarity_source"] == "lexical"
    assert p.outcome_contract.occurrence_resolves == "no"
    out, _ = _rollout(p, n_particles=400)
    # P(remain)=0.7 ⇒ the state-breaking event carries mass 0.3; absorption maps to the NO option
    assert out["distribution"]["yes"] == pytest.approx(0.7, abs=0.07)


def test_entailed_fact_absorbs_at_its_real_date_and_floors_the_answer():
    import swm.world_model_v2.scheduled_facts  # noqa: F401 — registers the scheduled_fact event type
    fact_ts = T0 + 10 * 86400.0
    fact_ev = {"etype": "scheduled_fact", "ts": fact_ts, "participants": [],
               "payload": {"fact": "the term ends", "kind": "term_expiry", "entity": "X",
                           "confidence": 0.9, "outcome_entailing": True, "entailed_direction": "yes",
                           "source": "model_knowledge"}}
    p = _binary_plan(posterior=[(0.3, 1.0)], extra_events=[fact_ev],
                     consumed=[{"var": "fact_entailment", "weight": 0.4}])
    rep = convert_binary_to_event_time(p, {})
    assert rep["n_absorbing_fact_events"] == 1 and rep["fact_floor"] == pytest.approx(0.9)
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"
              and "calibration" in e["payload"]]
    # the fact acts through ABSORPTION at its date — the rate-blend channel is removed (no double count)
    assert all(m["var"] != "fact_entailment" for e in rounds for m in e["payload"]["consume"])
    out, branches = _rollout(p, n_particles=400)
    # total mass = max(fact confidence, posterior target) = 0.9 — and the event happens AT ITS DATE
    assert out["distribution"]["yes"] == pytest.approx(0.9, abs=0.05)
    assert out["event_time"]["mode_distribution"].get("entailed_fact:term_expiry", 0.0) >= 0.8
    c = p.outcome_contract
    fact_times = [c.readout(b.world) for b in branches
                  if c.readout(b.world) and c._mode_of(b.world).startswith("entailed_fact")]
    assert fact_times and all(t == pytest.approx(fact_ts) for t in fact_times)


def test_absorbing_institution_is_the_resolution_path():
    inst_ev = {"etype": "institutional_decision", "ts": T1 - 2.0, "participants": [],
               "payload": {"institution_id": "senate", "outcome_var": "outcome",
                           "options": ["yes", "no"], "n_members": 9, "needed": 5,
                           "posterior_rate_particles": [[0.95, 1.0]]}}
    p = _binary_plan(extra_events=[inst_ev])
    rep = convert_binary_to_event_time(p, {})
    assert rep["absorbing_institutions"] == ["senate"]
    assert rep["n_residual_rounds"] == 0 and rep["residual_skipped_reason"]
    ie = next(e for e in p.scheduled_events if e["etype"] == "institutional_decision")
    assert ie["payload"]["absorbing"] is True
    from swm.world_model_v2.phase_consumers import CollectiveThresholdDecisionOperator
    out, _ = _rollout(p, n_particles=200, extra_ops=[CollectiveThresholdDecisionOperator()])
    # 0.95 member propensity under a declared 5-of-9 rule → passes; YES is READ from the absorption
    assert out["distribution"]["yes"] >= 0.9
    assert out["event_time"]["mode_distribution"].get("institutional:senate", 0.0) >= 0.9


def test_conversion_skips_non_binary_contracts_without_mutation():
    p = _binary_plan(options=("a", "b", "c"))
    p.outcome_contract.family = "categorical"
    before = [dict(e) for e in p.scheduled_events]
    rep = convert_binary_to_event_time(p, {})
    assert "skipped" in rep
    assert p.scheduled_events == before and p.accepted_mechanisms == []


def test_lexical_polarity_and_criterion_override():
    assert _lexical_event_polarity("Will Powell be out as Fed Chair by Aug 31?") == \
        "occurrence_resolves_yes"
    assert _lexical_event_polarity("Will Powell remain Fed Chair through Aug 31?") == \
        "occurrence_resolves_no"
    assert _lexical_event_polarity("Will X still be CEO on June 30?") == "occurrence_resolves_no"
    p = _binary_plan(question="Will the ceasefire hold?")
    rep = convert_binary_to_event_time(p, {"event_polarity": "occurrence_resolves_no"})
    assert rep["event_polarity"] == "occurrence_resolves_no"
    assert rep["polarity_source"] == "criterion_parser"


def test_hypothesis_lean_shifts_first_passage_mass():
    p = _binary_plan(posterior=[(0.4, 1.0)])
    convert_binary_to_event_time(p, {})

    def absorbed_share(lean):
        out, branches = _rollout(p, n_particles=300, seed=11, meta={"hypothesis_lean": lean})
        c = p.outcome_contract
        return sum(1 for b in branches if c.readout(b.world)) / 300.0
    # competing structures shift the per-particle target rate — genuinely different terminals
    assert absorbed_share("strong_no") < absorbed_share("neutral") < absorbed_share("strong_yes")


def test_consume_invert_channel():
    from swm.world_model_v2.phase_consumers import consume_state_rate
    w = _world(actor_intentions=0.9)
    p1, used1 = consume_state_rate(w, 0.5, [{"var": "actor_intentions", "weight": 0.4}])
    p2, used2 = consume_state_rate(w, 0.5, [{"var": "actor_intentions", "weight": 0.4, "invert": True}])
    assert used1 == used2 == ["actor_intentions"]
    assert p1 == pytest.approx(0.5 * 0.6 + 0.4 * 0.9)
    assert p2 == pytest.approx(0.5 * 0.6 + 0.4 * 0.1)


def test_readout_falls_back_to_absorbing_write_time():
    """A writer that fires on the branch's LAST event leaves the monitor no later event to stamp on —
    the readout recovers first passage from the predicate quantity's own write time."""
    c = EventTimeContract(as_of=T0, horizon_ts=T1, binary_options=["yes", "no"]).validate()
    w = _world()
    register_quantity_type("absorbing_state_reached", units="bool")
    w.quantities["absorbing_state_reached"] = Quantity(
        name="absorbing_state_reached", qtype="absorbing_state_reached", value=True,
        timestamp=T0 + 5 * 86400.0)
    assert c.readout(w) == pytest.approx(T0 + 5 * 86400.0)


def test_criterion_deadline_bounds_the_chain_and_the_binary_view():
    p = _binary_plan(posterior=[(0.5, 1.0)])
    rep = convert_binary_to_event_time(p, {"deadline": "2024-01-01"})   # inside [T0, T1]
    assert T0 < rep["deadline_ts"] < T1
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert rounds and all(e["ts"] <= rep["deadline_ts"] for e in rounds)
    assert p.outcome_contract.deadline_ts == pytest.approx(rep["deadline_ts"])


def test_mass_weights_from_curve_shape():
    assert _mass_weights_from_curve(None) == [0.2] * 5
    w = _mass_weights_from_curve([0.5, 0.0, 0.0, 0.0, 0.0])
    assert w[0] == pytest.approx(1.0) and sum(w) == pytest.approx(1.0)
    w2 = _mass_weights_from_curve([0.1, 0.1, 0.1, 0.1, 0.1])
    assert sum(w2) == pytest.approx(1.0) and w2[0] > w2[-1]    # early buckets carry more surviving mass


def test_unified_runtime_routes_binary_through_event_time():
    import inspect
    from swm.world_model_v2 import unified_runtime as U
    src = inspect.getsource(U.simulate_world)
    assert "convert_binary_to_event_time" in src and "convert_to_event_time" in src
