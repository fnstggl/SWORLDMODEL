"""Event-time contract architecture — unit tests for the §NAP provenance-gated design.

The answer to a binary deadline question is a READOUT of first-passage trajectories, resolved
ONLY through approved-provenance mechanisms: evidence-cited dated facts (deterministic at their
real dates), institutional decisions, or a posterior-parameterized residual process. A question
with none of those channels yields explicit unresolved_mechanism mass with honest bounds — never
a family-rate or lean-Beta draw, and never a silent `resolved_no`."""
import types

import pytest

from swm.world_model_v2.event_time import (AbsorptionMonitorOperator, EventTimeContract,
                                           FirstPassageOperator, HazardRoundOperator,
                                           _lexical_event_polarity, convert_binary_to_event_time,
                                           convert_to_event_time, ensure_first_passage_state,
                                           family_survival_pack_eligibility, fit_survival_pack,
                                           is_when_question)
from swm.world_model_v2.numeric_provenance import unresolved_mechanisms_of
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


def _branch(absorbed_at=None, absorbed_by=None, unresolved=None):
    q = {}
    if absorbed_at is not None:
        q["absorbed_at"] = absorbed_at
        q["absorbed_by"] = absorbed_by or "modeA"
    w = _world(**q)
    if unresolved:
        w._unresolved_mechanisms = [{"mechanism": m, "classification": "unresolved_mechanism"}
                                    for m in unresolved]
    return WorldBranch(branch_id=w.branch_id, world=w)


def _binary_plan(question="Will X be out as CEO by the deadline?", options=("yes", "no"),
                 posterior=None, extra_events=None):
    contract = types.SimpleNamespace(family="binary", options=list(options), resolution_rule="")
    return types.SimpleNamespace(
        question=question, as_of=T0, horizon_ts=T1, outcome_contract=contract,
        scheduled_events=[{"etype": "resolve_outcome", "ts": T1 - 1.0, "participants": [],
                           "payload": {"outcome_var": "outcome", "family": "binary",
                                       "options": list(options), "lean": "neutral"}}]
        + list(extra_events or []),
        accepted_mechanisms=[], quantities=[], provenance={"outcome_lean": "neutral"},
        posterior_rate_particles=list(posterior or []) or None)


def _rollout(p, n_particles=300, seed=3, extra_ops=None):
    from swm.world_model_v2.events import Event, EventQueue
    from swm.world_model_v2.rollout import RolloutEngine
    from swm.world_model_v2.temporal_hazards import schedule_crossing
    ops = [HazardRoundOperator(), FirstPassageOperator(), AbsorptionMonitorOperator()] \
        + list(extra_ops or [])
    branches = []
    for i in range(n_particles):
        w = _world()
        w.branch_id = f"b{i:03d}"
        for rec in (getattr(p, "_unresolved_mechanisms", None) or []):
            w._unresolved_mechanisms = list(getattr(w, "_unresolved_mechanisms", None) or [])
            w._unresolved_mechanisms.append(dict(rec))
        q = EventQueue(horizon_ts=p.horizon_ts)
        for ev in p.scheduled_events:
            q.schedule(Event(ts=ev["ts"], etype=ev["etype"],
                             participants=list(ev.get("participants") or []),
                             payload=dict(ev["payload"]), source="scheduled"))
        for spec in (getattr(p, "first_passage_processes", None) or []):
            st = ensure_first_passage_state(w, spec)
            if st is not None:
                schedule_crossing(q, w, st, etype="first_passage")
        branches.append(RolloutEngine(operators=ops).run_branch(w, q, seed=seed * 7919 + i))
    return p.outcome_contract.project(branches), branches


# ================================================================ projection semantics
def test_projection_distribution_cdf_monotone_and_binary_view():
    c = EventTimeContract(as_of=T0, horizon_ts=T1, modes=["modeA"],
                          binary_options=["yes", "no"]).validate()
    branches = [_branch(T0 + f * (T1 - T0)) for f in (0.1, 0.3, 0.5)] + [_branch(), _branch()]
    out = c.project(branches)
    et = out["event_time"]
    assert et["cdf"] == sorted(et["cdf"])
    assert out["distribution"]["yes"] == pytest.approx(0.6)
    assert out["distribution"]["no"] == pytest.approx(0.4)     # modeled non-occurrence
    assert et["unresolved_share"] == 0.0
    assert et["frequency_semantics"] == "simulated_scenario_frequency"


def test_projection_unresolved_mass_never_reads_as_no():
    """A branch whose outcome mechanism was refused (unresolved record) must NOT resolve `no` —
    its mass stays explicit unresolved_mechanism mass with honest bounds."""
    c = EventTimeContract(as_of=T0, horizon_ts=T1, binary_options=["yes", "no"]).validate()
    branches = ([_branch(T0 + 0.2 * (T1 - T0))]                          # resolved yes
                + [_branch(unresolved=["residual_outcome_process"])] * 2  # unresolved
                + [_branch()])                                           # modeled no
    out = c.project(branches)
    d, et = out["distribution"], out["event_time"]
    assert d["yes"] == pytest.approx(0.25)
    assert d["no"] == pytest.approx(0.25)
    assert d["unresolved_mechanism"] == pytest.approx(0.5)
    assert d["yes"] + d["no"] + d["unresolved_mechanism"] == pytest.approx(1.0)
    b = et["bounds"]["yes"]
    assert b["min_supported"] == pytest.approx(0.25)
    assert b["max_possible"] == pytest.approx(0.75)            # all unresolved mass could be yes
    assert et["resolved_conditional"]["yes"] == pytest.approx(0.5)
    assert "residual_outcome_process" in et["unresolved_mechanisms"]
    assert et["branch_terminals"]["unresolved_mechanism"] == pytest.approx(0.5)


def test_projection_categorical_separates_unresolved_from_none_of_the_options():
    c = EventTimeContract(as_of=T0, horizon_ts=T1,
                          categorical_options=["deal", "collapse"],
                          mode_option_map={"deal": "deal", "collapse": "collapse"}).validate()
    branches = [_branch(T0 + 10.0, "deal"), _branch(),
                _branch(unresolved=["mode_transition:collapse"])]
    out = c.project(branches)
    d = out["distribution"]
    assert d["deal"] == pytest.approx(1 / 3, abs=1e-3)
    assert d["none_of_the_options_by_horizon"] == pytest.approx(1 / 3, abs=1e-3)
    assert d["unresolved_mechanism"] == pytest.approx(1 / 3, abs=1e-3)


def test_absorption_monitor_is_first_passage_and_immutable():
    w = _world(absorbing_state_reached=True)
    op = AbsorptionMonitorOperator()
    ev = types.SimpleNamespace(etype="anything", payload={})
    assert op.applicable(w, ev)
    d = op.apply(w, op.propose(w, ev, None))
    assert w.quantities["absorbed_at"].value == T0
    assert not op.applicable(w, ev)                            # immutable after first passage
    assert d.event_type == "absorption"


# ================================================================ provenance-gated hazard rounds
def test_hazard_round_refuses_legacy_parameterizations_and_records_unresolved():
    w = _world()
    op = HazardRoundOperator()
    ev = types.SimpleNamespace(etype="hazard_round",
                               payload={"mode": "m", "calibration": {"exponent": 0.5},
                                        "hr": {"median": 2.0}})
    d = op.apply(w, op.propose(w, ev, None))
    assert "numeric_provenance_rejected_unresolved" in d.reason_codes
    assert not w.quantities.get("absorbing_state_reached")
    recs = unresolved_mechanisms_of(w)
    assert recs and recs[0]["classification"] == "unresolved_mechanism"


def test_hazard_round_success_prob_requires_approved_provenance():
    w = _world()
    op = HazardRoundOperator()
    ev = types.SimpleNamespace(etype="hazard_round",
                               payload={"mode": "fact", "success_prob": 1.0})   # no provenance
    d = op.apply(w, op.propose(w, ev, None))
    assert "numeric_provenance_rejected_unresolved" in d.reason_codes
    assert unresolved_mechanisms_of(w)
    w2 = _world()
    ev2 = types.SimpleNamespace(etype="hazard_round", payload={
        "mode": "fact", "success_prob": 1.0,
        "numeric_provenance": {"source_class": "observed_measurement", "evidence_id": "c1"}})
    op.apply(w2, op.propose(w2, ev2, None))
    assert w2.quantities["absorbing_state_reached"].value is True
    man = w2._numeric_ledger.manifest()
    assert man["approved_and_consumed"] and not man["rejected"]


def test_hazard_round_not_applicable_after_absorption():
    w = _world(absorbed_at=T0 - 1.0)
    op = HazardRoundOperator()
    assert not op.applicable(w, types.SimpleNamespace(etype="hazard_round", payload={}))


# ================================================================ first-passage gating
def test_first_passage_state_requires_posterior_calibration():
    w = _world()
    st = ensure_first_passage_state(w, {"process_id": "mode:x", "kind": "mode_transition",
                                        "mode": "x", "share": 0.5, "as_of": T0,
                                        "span_s": T1 - T0})
    assert st is None                                          # mode-share intensity refused
    assert any("first_passage:mode:x" == r["mechanism"] for r in unresolved_mechanisms_of(w))
    w2 = _world()
    st2 = ensure_first_passage_state(w2, {
        "process_id": "resolution", "kind": "calibrated_resolution", "mode": "resolution",
        "as_of": T0, "span_s": T1 - T0,
        "calibration": {"posterior_rate_particles": [[0.3, 1.0]], "absorb_from": "rate"}})
    assert st2 is not None and st2.modulation == 1.0           # no stance/progress modulation
    man = w2._numeric_ledger.manifest()
    assert any(e["name"] == "residual_target_mass" for e in man["approved_and_consumed"])


def test_first_passage_state_without_posterior_is_unresolved():
    w = _world()
    st = ensure_first_passage_state(w, {
        "process_id": "resolution", "kind": "calibrated_resolution", "mode": "resolution",
        "as_of": T0, "span_s": T1 - T0, "calibration": {"lean": "weak_yes"}})
    assert st is None                                          # NO lean-Beta rung exists
    assert any(r["mechanism"] == "first_passage:resolution"
               for r in unresolved_mechanisms_of(w))


# ================================================================ routing
def test_is_when_question_routing():
    assert is_when_question("When will the deal be signed?")
    assert not is_when_question("Will the deal be signed by June?")


def test_lexical_polarity():
    assert _lexical_event_polarity("Will X remain CEO through March?") == "occurrence_resolves_no"
    assert _lexical_event_polarity("Will X resign by March?") == "occurrence_resolves_yes"


# ================================================================ binary conversion (§NAP)
def test_binary_conversion_removes_resolvers_and_uses_posterior_residual():
    p = _binary_plan(posterior=[(0.3, 1.0)],
                     extra_events=[{"etype": "aggregate_outcome_resolution", "ts": T1 - 0.5,
                                    "participants": [], "payload": {"outcome_var": "outcome",
                                                                    "options": ["yes", "no"],
                                                                    "consume": []}}])
    lin = {}
    rep = convert_binary_to_event_time(p, {"resolves_yes_iff": "X departs the CEO role"},
                                       lineage=lin)
    assert rep["contract"] == "binary_first_passage" and rep["n_resolver_events_removed"] == 2
    assert not any(e["etype"] in ("resolve_outcome", "aggregate_outcome_resolution")
                   for e in p.scheduled_events)                # NO component declares the answer
    assert isinstance(p.outcome_contract, EventTimeContract)
    procs = [s for s in p.first_passage_processes if s["kind"] == "calibrated_resolution"]
    assert len(procs) == 1 and rep["n_residual_processes"] == 1
    assert rep["posterior_calibrated"] is True
    # §NAP: no consume channels, no curve, no lean on the residual spec
    assert "consume" not in procs[0] and procs[0].get("curve") is None
    assert "lean" not in procs[0]["calibration"]
    # the family rate is REGISTERED AS REJECTED, never consumed
    man = rep["numeric_provenance_manifest"]
    assert any(r["name"] == "family_fallback_rate" for r in man["rejected"])
    assert any(r["name"] == "lean_beta_target" for r in man["rejected"])
    order = [m["operator"] for m in p.accepted_mechanisms]
    assert order.index("first_passage") < order.index("absorption_monitor")


def test_binary_answer_matches_posterior_target_as_observed_events():
    p = _binary_plan(posterior=[(0.3, 1.0)])
    convert_binary_to_event_time(p, {})
    out, _ = _rollout(p, n_particles=400)
    d = out["distribution"]
    assert d["yes"] == pytest.approx(0.3, abs=0.07)
    et = out["event_time"]
    assert et["cdf"] == sorted(et["cdf"])
    assert et["p_event_by_deadline"] == d["yes"]
    assert et["unresolved_share"] == 0.0


def test_binary_without_posterior_facts_or_institution_is_unresolved():
    """The forced-forecast loophole is CLOSED: no posterior, no evidence-cited fact, no
    institution ⇒ no residual process, no family rate, no lean-Beta — the mechanism is recorded
    unresolved and every unabsorbed branch classifies unresolved_mechanism."""
    p = _binary_plan(posterior=None)
    rep = convert_binary_to_event_time(p, {})
    assert rep["n_residual_processes"] == 0
    assert not getattr(p, "first_passage_processes", None)
    recs = getattr(p, "_unresolved_mechanisms", None)
    assert recs and any(r["mechanism"] == "residual_outcome_process" for r in recs)
    out, _ = _rollout(p, n_particles=40)
    d = out["distribution"]
    assert d["unresolved_mechanism"] == pytest.approx(1.0)
    assert d["yes"] == 0.0 and d["no"] == 0.0
    et = out["event_time"]
    assert et["bounds"]["yes"] == {"min_supported": 0.0, "max_possible": 1.0}
    assert et["resolved_conditional"] is None


def test_binary_survival_polarity_maps_absorption_to_no():
    p = _binary_plan(question="Will X remain CEO through the deadline?", posterior=[(0.7, 1.0)])
    rep = convert_binary_to_event_time(p, {})
    assert rep["event_polarity"] == "occurrence_resolves_no" and rep["polarity_source"] == "lexical"
    assert p.outcome_contract.occurrence_resolves == "no"
    out, _ = _rollout(p, n_particles=400)
    # absorb_from=one_minus_rate: P(remain)=0.7 ⇒ breaking-event mass 0.3 ⇒ yes≈0.7
    assert out["distribution"]["yes"] == pytest.approx(0.7, abs=0.07)


def test_evidence_cited_fact_absorbs_deterministically_at_its_real_date():
    import swm.world_model_v2.scheduled_facts  # noqa: F401 — registers the event type
    fact_ts = T0 + 10 * 86400.0
    fact_ev = {"etype": "scheduled_fact", "ts": fact_ts, "participants": [],
               "payload": {"fact": "the term ends", "kind": "term_expiry", "entity": "X",
                           "confidence": 0.9, "outcome_entailing": True,
                           "entailed_direction": "yes", "source": "evidence",
                           "evidence_quote": "the charter terminates the term on that date",
                           "claim_id": "c42"}}
    p = _binary_plan(extra_events=[fact_ev])
    rep = convert_binary_to_event_time(p, {})
    assert rep["n_absorbing_fact_events"] == 1
    hz = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert len(hz) == 1
    assert hz[0]["payload"]["success_prob"] == 1.0             # deterministic, never Bernoulli(conf)
    prov = hz[0]["payload"]["numeric_provenance"]
    assert prov["source_class"] == "observed_measurement" and prov["evidence_id"] == "c42"
    assert prov["extraction_confidence_label"] == 0.9          # a LABEL, not a parameter
    out, _ = _rollout(p, n_particles=60)
    assert out["distribution"]["yes"] == pytest.approx(1.0)
    assert out["event_time"]["first_passage_quantiles_ts"]["0.5"] == pytest.approx(fact_ts, abs=2.0)


def test_model_knowledge_fact_cannot_absorb_and_is_recorded_unresolved():
    import swm.world_model_v2.scheduled_facts  # noqa: F401
    fact_ev = {"etype": "scheduled_fact", "ts": T0 + 10 * 86400.0, "participants": [],
               "payload": {"fact": "a rumor", "kind": "rumor", "entity": "X",
                           "confidence": 0.9, "outcome_entailing": True,
                           "entailed_direction": "yes", "source": "model_knowledge"}}
    p = _binary_plan(extra_events=[fact_ev])
    rep = convert_binary_to_event_time(p, {})
    assert rep["n_absorbing_fact_events"] == 0
    assert rep["n_ungrounded_facts_unresolved"] == 1
    assert not any(e["etype"] == "hazard_round" for e in p.scheduled_events)
    assert any(r["mechanism"].startswith("scheduled_fact:")
               for r in (getattr(p, "_unresolved_mechanisms", None) or []))


def test_absorbing_institution_is_the_resolution_path():
    inst_ev = {"etype": "institutional_decision", "ts": T0 + 30 * 86400.0, "participants": [],
               "payload": {"institution_id": "board", "outcome_var": "outcome",
                           "n_members": 9, "threshold_share": 0.5,
                           "options": ["yes", "no"]}}
    p = _binary_plan(extra_events=[inst_ev])
    rep = convert_binary_to_event_time(p, {})
    assert rep["absorbing_institutions"] == ["board"]
    assert rep["n_residual_processes"] == 0
    # the institution IS the modeled resolution path — no unresolved residual record
    assert not any(r["mechanism"] == "residual_outcome_process"
                   for r in (getattr(p, "_unresolved_mechanisms", None) or []))
    assert inst_ev["payload"]["absorbing"] is True


def test_conversion_skips_non_binary_contracts_without_mutation():
    contract = types.SimpleNamespace(family="continuous", options=[], resolution_rule="")
    p = types.SimpleNamespace(question="How high?", as_of=T0, horizon_ts=T1,
                              outcome_contract=contract, scheduled_events=[], quantities=[],
                              accepted_mechanisms=[], provenance={})
    rep = convert_binary_to_event_time(p, {})
    assert "skipped" in rep and p.outcome_contract is contract


# ================================================================ when/categorical conversion
def _when_plan():
    contract = types.SimpleNamespace(family="event_time", options=[], resolution_rule="")
    return types.SimpleNamespace(
        question="When will the ceasefire deal be signed?", as_of=T0, horizon_ts=T1,
        outcome_contract=contract, scheduled_events=[], accepted_mechanisms=[], quantities=[],
        provenance={}, structural_hypotheses=[{"id": "ceasefire", "pathway":
                                               "cooperative_agreement"}],
        compute_plan={"n_particles": 30})


def test_convert_to_event_time_mints_no_intensity_and_records_unresolved_modes():
    p = _when_plan()
    lin = {}
    rep = convert_to_event_time(p, {"resolves_yes_iff": "a ceasefire deal is signed"},
                                lineage=lin)
    assert rep["scheduling"] == "provenance_gated_event_time"
    # NO first-passage specs are minted from mode shares / stance ratios / family curves
    assert not getattr(p, "first_passage_processes", None)
    assert rep["unresolved_mode_transitions"] == ["ceasefire"]
    assert any(r["mechanism"] == "mode_transition:ceasefire"
               for r in (getattr(p, "_unresolved_mechanisms", None) or []))
    assert rep["stance_hazard_channel"] == "removed_quarantined (§NAP)"
    assert rep["hr_pack"]["source"] == "quarantined_no_production_stance_hazard_channel"
    man = rep["numeric_provenance_manifest"]
    assert any(r["name"] == "family_survival_curve" for r in man["rejected"])
    assert isinstance(p.outcome_contract, EventTimeContract)
    assert p.compute_plan["n_particles"] >= 200


def test_convert_to_event_time_institution_absorbs_and_mode_is_modeled():
    p = _when_plan()
    p.structural_hypotheses = [{"id": "board_approval", "pathway": "institutional_procedure"}]
    p.scheduled_events = [{"etype": "institutional_decision", "ts": T0 + 20 * 86400.0,
                           "participants": [], "payload": {"institution_id": "board",
                                                           "outcome_var": "outcome"}}]
    rep = convert_to_event_time(p, {})
    assert rep["n_absorbing_institutional_decisions"] == 1
    assert rep["unresolved_mode_transitions"] == []            # the institution models it
    assert p.scheduled_events[0]["payload"]["absorbing"] is True


def test_persistence_window_parsed_and_approved_as_institutional_rule():
    p = _when_plan()
    rep = convert_to_event_time(
        p, {"resolves_yes_iff": "no active hostilities for >=30 consecutive days"})
    assert rep["persistence_window_days"] == 30.0
    man = rep["numeric_provenance_manifest"]
    approved = [e for e in man["approved_and_consumed"]
                if e["name"] == "criterion_persistence_window"]
    assert approved and approved[0]["source_class"] == "institutional_rule"


# ================================================================ fitting utilities stay offline
def test_fit_survival_pack_shape_and_ineligibility():
    rows = ([{"question": "Will X and Y sign a deal?", "lifetime_fraction_resolved": f}
             for f in (0.1, 0.3, None, None)] +
            [{"question": "Will team A win the match?", "lifetime_fraction_resolved": 0.9}])
    pack = fit_survival_pack(rows)
    assert pack["fit_on"] == "calibration split only"
    assert len(pack["global_hazards"]) == 5
    # §NAP: the pack (like the on-disk one) FAILS production eligibility — diagnostic only
    from swm.world_model_v2.numeric_provenance import fitted_artifact_eligible
    ok, why = fitted_artifact_eligible(pack)
    assert not ok and "missing required provenance keys" in why


def test_family_survival_pack_eligibility_verdict_is_computed_not_assumed():
    v = family_survival_pack_eligibility()
    assert v["eligible"] is False                              # the current pack must never pass
