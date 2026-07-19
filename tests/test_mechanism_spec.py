"""§22–§25 tests: the typed MechanismSpec join, the unified calibration vocabulary, declared
write-set checking, external-simulator adapter gating, and the four ported legacy kernels
(poll_error_aggregation, whipcount_binomial, outside_world_hazard, population_segment_exposure)
— each exercised on a minimal WorldState through the canonical StateDelta contract and pinned
numerically against its legacy source where one exists."""
import json
import random
from pathlib import Path

import pytest

import swm.world_model_v2.kernel_ports  # noqa: F401 — registers the ported operators + specs
from swm.api.mechanisms import sim_aggregation, sim_whipcount
from swm.world_model_v2 import mechanism_spec as ms
from swm.world_model_v2 import transitions
from swm.world_model_v2.events import Event
from swm.world_model_v2.kernel_ports import (OutsideWorldHazardOperator,
                                             PollErrorAggregationOperator,
                                             PopulationSegmentExposureOperator,
                                             WhipcountBinomialOperator,
                                             assert_no_forbidden_paths)
from swm.world_model_v2.outside_world import (ArrivalModel, ExternalEventFamily,
                                              OutsideWorldProcess)
from swm.world_model_v2.registry.record import STATUSES as PHASE6_STATUSES
from swm.world_model_v2.state import SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta

T0 = 1.0e9


def _world():
    return WorldState(world_id="t", branch_id="root",
                      clock=SimulationClock(now=T0, as_of=T0))


# ================================================================ the spec index (§22)
def test_spec_index_builds_and_covers_all_registered_operators():
    idx = ms.build_spec_index()
    assert idx, "spec index is empty"
    covered_ops = {s.operator for s in idx.values() if s.operator}
    missing = set(transitions._OPERATORS) - covered_ops
    assert not missing, f"registered operators without a MechanismSpec: {sorted(missing)}"
    for spec in idx.values():                    # dataclass invariants hold for every spec
        assert spec.mechanism_kind in ms.known_kinds()
        assert spec.calibration_status in ms.CALIBRATION_STATUSES


def test_spec_index_distinguishes_fitted_from_broad_prior_at_execution_layer():
    """The audit's gap 3: every operator claims validated=True, so fitted and broad-prior
    mechanisms were indistinguishable. The joined index separates them."""
    idx = ms.build_spec_index()
    assert idx["production_actor_policy"].calibration_status == "fitted_validated"
    assert idx["generic_outcome_prior"].calibration_status == "experimental_visible"
    # operator-only registration with no lean entry and no Phase-6 evidence: the honest floor
    assert idx["network_diffusion"].calibration_status == "documented_prior"
    # the two formerly-dead entries are now backed by ported operators with real specs
    assert idx["poll_error_aggregation"].operator == "poll_error_aggregation"
    assert idx["whipcount_binomial"].operator == "whipcount_binomial"


def test_spec_index_joins_phase6_evidence_to_executing_operators():
    """Phase-6 records reach the executing operator by family_id or code reference: the
    institutional_vote family matches by id; belief_update joins belief_update_exposure via
    the operator class's code key — version and evidence fields ride onto the spec."""
    idx = ms.build_spec_index()
    assert idx["institutional_vote"].version == "1.0.0"       # joined from the Phase-6 record
    assert idx["belief_update"].version == "1.0.0"            # joined via code_ref match
    # Phase-6 'implemented' (code+tests only) must NOT override the lean declared status
    assert idx["belief_update"].calibration_status == "documented_prior"   # lean 'prior'
    # event-I/O backfill: declared inputs exist for the operators the audit found hardcoded
    assert idx["belief_update"].event_inputs == ("exposure",)
    assert "decision_opportunity" in idx["production_actor_policy"].event_inputs


def test_spec_kind_vocabulary_is_extensible_but_never_implicit():
    with pytest.raises(ValueError):
        ms.MechanismSpec(mechanism_id="x", mechanism_kind="vibes")
    with pytest.raises(ValueError):
        ms.register_mechanism_kind("biochemical", rationale="")     # rationale required
    ms.register_mechanism_kind("biochemical", rationale="wet-lab simulator adapters")
    spec = ms.MechanismSpec(mechanism_id="x", mechanism_kind="biochemical")
    assert spec.mechanism_kind == "biochemical"


# ================================================================ unified vocabulary
def test_unified_vocabulary_maps_every_legacy_status():
    # vocabulary 1: transitions validated/experimental flags
    for s in ("validated", "experimental", "unvalidated"):
        assert ms.unify_calibration_status(s, vocabulary="transitions") in ms.CALIBRATION_STATUSES
    assert ms.status_from_flags(validated=True, experimental=False) == "documented_prior"
    assert ms.status_from_flags(validated=True, experimental=True) == "experimental_visible"
    # vocabulary 2: the lean 4-value enum PLUS the observed 'deterministic' bypass
    for s in ("calibrated", "prior", "uncalibrated", "experimental", "deterministic"):
        assert ms.unify_calibration_status(s, vocabulary="lean") in ms.CALIBRATION_STATUSES
    assert ms.unify_calibration_status("deterministic", vocabulary="lean") == "grounded_scenario"
    # vocabulary 3: every Phase-6 lifecycle status
    for s in PHASE6_STATUSES:
        assert ms.unify_calibration_status(s, vocabulary="phase6") in ms.CALIBRATION_STATUSES
    assert ms.unify_calibration_status("quarantined", vocabulary="phase6") == "unresolved"
    # unknown claims are never rounded UP; unknown vocabularies refuse
    assert ms.unify_calibration_status("shiny_new_status", vocabulary="lean") == "unresolved"
    with pytest.raises(KeyError):
        ms.unify_calibration_status("calibrated", vocabulary="vibes")


# ================================================================ write-set checking
def test_declared_write_violation_detection():
    spec = ms.MechanismSpec(mechanism_id="m", mechanism_kind="measurement",
                            write_set=("quantities[",), operator="op",
                            calibration_status="documented_prior")
    d = StateDelta(at=T0, event_type="e", operator="op")
    d.change("quantities[x]", None, True)
    d.change("alice.current_action", None, "act")             # OUT of the declared write set
    violations = ms.declared_write_violations(spec, d)
    assert len(violations) == 1 and violations[0]["path"] == "alice.current_action"
    # the entity.<field> dialect matches concrete entity ids (delta paths spell the id)
    spec2 = ms.MechanismSpec(mechanism_id="m2", mechanism_kind="qualitative_actor",
                             write_set=("entity.current_action", "entity.past_actions"),
                             operator="op2", calibration_status="documented_prior")
    d2 = StateDelta(at=T0, event_type="e", operator="op2")
    d2.change("alice.current_action", None, "act")
    d2.change("quantities[x]", None, 1.0)                     # out of set
    v2 = ms.declared_write_violations(spec2, d2)
    assert [v["path"] for v in v2] == ["quantities[x]"]
    # an EMPTY write_set is legacy-advisory: nothing to violate (the gap shows in the spec)
    spec3 = ms.MechanismSpec(mechanism_id="m3", mechanism_kind="numerical",
                             calibration_status="documented_prior")
    assert ms.declared_write_violations(spec3, d) == []


# ================================================================ external adapters (§25)
def test_adapter_validation_rejects_missing_version_and_data_cutoff():
    bad = ms.ExternalSimulatorAdapter(adapter_id="weather_sim")
    reasons = ms.validate_adapter(bad)
    assert any("version" in r for r in reasons)
    assert any("data_cutoff" in r for r in reasons)
    with pytest.raises(ValueError):
        ms.register_external_adapter(bad)


def _honest_sim(inputs, *, seed=0):
    return {"out": 0.0}


def test_adapter_guard_refuses_dynamically_generated_callables():
    dyn = lambda inputs, seed=0: {}                     # noqa: E731
    dyn.__module__ = "<generated>"                      # the signature of run-time-minted code
    assert ms.is_dynamically_generated(dyn)
    bad = ms.ExternalSimulatorAdapter(adapter_id="gen", version="1.0", data_cutoff="2026-01",
                                      simulate=dyn)
    assert any("dynamically generated" in r for r in ms.validate_adapter(bad))
    with pytest.raises(ValueError):
        ms.register_external_adapter(bad)
    exec_ns = {}
    exec("def g(inputs, seed=0):\n    return {}", exec_ns)   # no __name__ in globals
    assert ms.is_dynamically_generated(exec_ns["g"])
    ok = ms.ExternalSimulatorAdapter(
        adapter_id="honest", version="1.2.0", data_cutoff="2026-01-01",
        accepted_inputs=("temperature_c",), output_schema={"out": "float"},
        units={"out": "share"}, time_semantics="adapter hours == simulation hours",
        deterministic_seed_behavior="bitwise reproducible per seed",
        calibration_status="documented_prior", failure_behavior="reject",
        counterfactual_safe=True, simulate=_honest_sim)
    assert ms.validate_adapter(ok) == []
    assert ms.register_external_adapter(ok) == "honest"
    assert ms.get_external_adapter("honest").version == "1.2.0"


# ================================================================ (a) poll_error_aggregation
def test_poll_error_aggregation_executes_and_matches_legacy_monte_carlo():
    """Per-branch latent draw; across seeded branches the legacy sim_aggregation Monte Carlo
    (52% lead + 4pt error ≈ 0.7, not 1.0) emerges — the closed form is never hardcoded."""
    op = PollErrorAggregationOperator()
    spec = {"share": 0.52, "share_sd": 0.04, "threshold": 0.5, "outcome_var": "ref_pass",
            "provenance": "poll avg 2026-07 (n=5 polls)"}
    yes = 0
    n = 400
    for i in range(n):
        w = _world()
        ev = Event(ts=T0 + 1, etype="poll_error_aggregation",
                   payload={"aggregation_spec": dict(spec)})
        delta, vr = op.run(w, ev, random.Random(i))
        assert vr.ok and isinstance(delta, StateDelta)
        assert delta.operator == "poll_error_aggregation"
        assert any(c["path"] == "quantities[ref_pass]" for c in delta.changes)
        assert w.quantities["ref_pass"].value in (True, False)
        yes += 1 if w.quantities["ref_pass"].value else 0
    legacy = sim_aggregation(0.52, share_sd=0.04)
    assert abs(yes / n - legacy) < 0.06                      # same measurement model
    assert 0.6 < yes / n < 0.8                               # the legacy honesty pin


def test_poll_error_aggregation_rejects_ungrounded_params():
    op = PollErrorAggregationOperator()
    w = _world()
    # no share at all: never substitutes a base rate
    ev = Event(ts=T0 + 1, etype="poll_error_aggregation",
               payload={"aggregation_spec": {"outcome_var": "q", "provenance": "x"}})
    delta, vr = op.run(w, ev, random.Random(0))
    assert delta is None and not vr.ok and "grounded share" in vr.reasons[0]
    # share without provenance: rejected, not defaulted
    ev2 = Event(ts=T0 + 1, etype="poll_error_aggregation",
                payload={"aggregation_spec": {"share": 0.52, "outcome_var": "q"}})
    delta2, vr2 = op.run(w, ev2, random.Random(0))
    assert delta2 is None and not vr2.ok and "provenance" in vr2.reasons[0]
    assert "q" not in w.quantities                           # nothing was written


# ================================================================ (b) whipcount_binomial
def test_whipcount_short_circuits_when_arithmetic_decides():
    op = WhipcountBinomialOperator()
    prov = {"counts": "whip count 2026-07-01"}
    for cy, und, needed, expect in ((60, 0, 50, True), (30, 0, 50, False), (45, 4, 50, False)):
        w = _world()
        ev = Event(ts=T0 + 1, etype="whipcount_binomial",
                   payload={"whipcount_spec": {"committed_yes": cy, "undecided": und,
                                               "needed": needed, "outcome_var": "passes",
                                               "provenance": prov}})
        delta, vr = op.run(w, ev, random.Random(0))
        assert vr.ok and w.quantities["passes"].value is expect
        assert delta.uncertainty["mode"].startswith("arithmetic_decides")


def test_whipcount_binomial_matches_legacy_kernel_monte_carlo():
    op = WhipcountBinomialOperator()
    spec = {"committed_yes": 45, "undecided": 20, "needed": 50, "lean": 0.5,
            "outcome_var": "passes",
            "provenance": {"counts": "whip count", "lean": "party lean (historical breaks)"}}
    yes = 0
    n = 300
    for i in range(n):
        w = _world()
        ev = Event(ts=T0 + 1, etype="whipcount_binomial",
                   payload={"whipcount_spec": dict(spec)})
        delta, vr = op.run(w, ev, random.Random(i))
        assert vr.ok and delta.uncertainty["mode"] == "binomial_lean_breaks"
        yes += 1 if w.quantities["passes"].value else 0
    legacy = sim_whipcount(committed_yes=45, undecided=20, needed=50, lean=0.5)
    assert abs(yes / n - legacy) < 0.05
    assert 0.5 < yes / n <= 1.0                              # the legacy test's pin


def test_whipcount_refuses_invented_probabilities():
    """The legacy kernel defaulted lean=0.5. The port must return a rejection ValidationResult
    whenever the outcome depends on how undecideds break and no grounded probabilities were
    supplied — never a default, never a silent 0.5."""
    op = WhipcountBinomialOperator()
    w = _world()
    base = {"committed_yes": 45, "undecided": 20, "needed": 50, "outcome_var": "passes",
            "provenance": {"counts": "whip count"}}
    ev = Event(ts=T0 + 1, etype="whipcount_binomial", payload={"whipcount_spec": dict(base)})
    delta, vr = op.run(w, ev, random.Random(0))
    assert delta is None and not vr.ok
    assert "no invented" in vr.reasons[0] or "REFUSED" in vr.reasons[0]
    assert "passes" not in w.quantities
    # lean present but WITHOUT provenance: still refused
    ev2 = Event(ts=T0 + 1, etype="whipcount_binomial",
                payload={"whipcount_spec": {**base, "lean": 0.5}})
    delta2, vr2 = op.run(w, ev2, random.Random(0))
    assert delta2 is None and not vr2.ok and "provenance" in vr2.reasons[0]
    # per-member probabilities without provenance: refused too
    ev3 = Event(ts=T0 + 1, etype="whipcount_binomial",
                payload={"whipcount_spec": {**base, "member_yes_probabilities": [0.5] * 20}})
    delta3, vr3 = op.run(w, ev3, random.Random(0))
    assert delta3 is None and not vr3.ok
    # conservation: declared total cannot be exceeded
    ev4 = Event(ts=T0 + 1, etype="whipcount_binomial",
                payload={"whipcount_spec": {**base, "committed_no": 50, "total": 100,
                                            "lean": 0.6,
                                            "provenance": {"counts": "c", "lean": "l"}}})
    delta4, vr4 = op.run(w, ev4, random.Random(0))
    assert delta4 is None and not vr4.ok and "conservation" in vr4.reasons[0]


# ================================================================ (c) outside_world_hazard
def _families():
    sched = ExternalEventFamily(
        family_id="cpi_print", description="scheduled macro data print",
        marks=["hot print", "cool print"],
        affected_boundary_components=["market attention"],
        impact_mechanism="observation_delivery",
        impact_description="delivers a macro observation to watching actors",
        arrival=ArrivalModel(kind="scheduled_exact",
                             scheduled_times=[T0 + 3600.0, T0 + 7200.0, T0 + 9.0e6]))
    surprise = ExternalEventFamily(
        family_id="platform_outage", description="unscheduled outage",
        marks=["regional outage"],
        affected_boundary_components=["channel capacity"],
        impact_mechanism="capacity_change",
        impact_description="reduces delivery capacity while active",
        arrival=ArrivalModel(kind="observed_base_rate", rate_per_day=2.0,
                             provenance="status-page incident logs 2024-2025"))
    unresolved = ExternalEventFamily(
        family_id="regulatory_wildcard", description="speculative legal change",
        impact_mechanism="institutional_rule_change",
        arrival=ArrivalModel(kind="unresolved"))
    return sched, surprise, unresolved


def test_outside_world_hazard_emits_entry_events_and_never_writes_forbidden_paths():
    sched, surprise, unresolved = _families()
    proc = OutsideWorldProcess(boundary_id="b0", families=[sched, surprise, unresolved])
    op = OutsideWorldHazardOperator()
    w = _world()
    ev = Event(ts=T0 + 1, etype="outside_world_window",
               payload={"outside_world": proc, "window_end_ts": T0 + 3 * 86400.0})
    delta, vr = op.run(w, ev, random.Random(7))
    assert vr.ok and isinstance(delta, StateDelta)
    # the scheduled calendar arrivals (v1 FutureEvent semantics): exactly the in-window dates
    cal = [f for f in delta.follow_up_events
           if f["payload"]["outside_world_family"] == "cpi_print"]
    assert [f["ts"] for f in cal] == [T0 + 3600.0, T0 + 7200.0]
    # every arrival is a TYPED entry event through the queue, carrying its entry mechanism
    assert delta.follow_up_events
    for fu in delta.follow_up_events:
        assert fu["etype"] == "outside_world_arrival"
        assert fu["payload"]["entry_mechanism"] in ("observation_delivery", "capacity_change")
        assert fu["payload"]["arrival_provenance"] is not None
    # base-rate (v1 SurpriseHazard) arrivals are counted consistently with the follow-ups
    n_surprise = len([f for f in delta.follow_up_events
                      if f["payload"]["outside_world_family"] == "platform_outage"])
    assert delta.uncertainty["families"]["platform_outage"]["n"] == n_surprise
    assert w.quantities["outside_world_arrivals:platform_outage"].value == n_surprise
    # the operator's own writes are bookkeeping counts only — never a terminal/readout path
    from swm.world_model_v2.outside_world import FORBIDDEN_WRITES
    for ch in delta.changes:
        assert ch["path"].startswith("quantities[outside_world_arrivals:")
        assert not any(bad in ch["path"].lower() for bad in FORBIDDEN_WRITES)
    assert_no_forbidden_paths(delta)                          # the final guard passes
    # unresolved families were surfaced, never sampled (§5.2)
    assert "regulatory_wildcard" in delta.uncertainty["unresolved_families"]
    assert any("unresolved_family_never_sampled:regulatory_wildcard" in r
               for r in delta.reason_codes)
    assert "outside_world_arrivals:regulatory_wildcard" not in w.quantities


def test_outside_world_hazard_rejects_families_targeting_forbidden_writes():
    bad = ExternalEventFamily(
        family_id="oracle", description="a family that claims the answer",
        affected_boundary_components=["forecast_answer"],
        impact_mechanism="observation_delivery",
        impact_description="writes the forecast answer directly",
        arrival=ArrivalModel(kind="scheduled_exact", scheduled_times=[T0 + 100.0]))
    proc = OutsideWorldProcess(boundary_id="b1", families=[bad])
    op = OutsideWorldHazardOperator()
    w = _world()
    ev = Event(ts=T0 + 1, etype="outside_world_window",
               payload={"outside_world": proc, "window_end_ts": T0 + 86400.0})
    delta, vr = op.run(w, ev, random.Random(0))
    assert delta is None and not vr.ok and "forbidden" in vr.reasons[0]
    # and the final guard itself refuses a delta that names a forbidden path
    poisoned = StateDelta(at=T0, event_type="x", operator="outside_world_hazard")
    poisoned.change("quantities[forecast_answer]", None, True)
    with pytest.raises(ValueError):
        assert_no_forbidden_paths(poisoned)


# ============================================================ (d) population_segment_exposure
_CELLS = [(0.2, 0.5, 1.0), (0.9, 0.3, 2.0), (0.5, 0.4, 1.0)]
_COUPLINGS = {"k_social": {"value": 0.4, "source": "documented prior (world_dynamics style)"},
              "k_event": {"value": 1.0, "source": "documented prior"},
              "k_proof": {"value": 0.2, "source": "documented prior"},
              "proof_center": {"value": 0.5, "source": "documented prior"}}


def test_population_segment_exposure_matches_legacy_mean_field_numerically():
    from swm.simulation.mean_field import MeanFieldRollout, agents_from_cells
    op = PopulationSegmentExposureOperator()
    w = _world()
    spec = {"segments": [{"belief": b, "responsiveness": r, "influence": inf}
                         for b, r, inf in _CELLS],
            "couplings": _COUPLINGS, "steps": 4, "event_impacts": [0.05, 0.0, 0.0, 0.0],
            "outcome_var": "support_share",
            "provenance": {"segments": "census cells 2025 (poststratified)"}}
    ev = Event(ts=T0 + 1, etype="population_segment_exposure",
               payload={"mean_field_spec": spec})
    delta, vr = op.run(w, ev, random.Random(0))
    assert vr.ok and isinstance(delta, StateDelta)
    # the LEGACY kernel, run directly, must agree to numerical identity
    agents = agents_from_cells(_CELLS)
    _, legacy_final = MeanFieldRollout(k_social=0.4, k_event=1.0, k_proof=0.2,
                                       proof_center=0.5).roll(agents, 4,
                                                              events=[0.05, 0.0, 0.0, 0.0])
    assert abs(w.quantities["support_share"].value - legacy_final) < 1e-12
    for i, ag in enumerate(agents):
        assert abs(w.quantities[f"segment_belief:support_share:{i}"].value - ag.belief) < 1e-12
    # the delta is the canonical machine-readable record: aggregate + per-segment changes
    assert any(c["path"] == "quantities[support_share]" for c in delta.changes)
    assert sum(1 for c in delta.changes if c["path"].startswith(
        "quantities[segment_belief:")) == len(_CELLS)
    assert "mean_field" in delta.uncertainty["kernel"]        # provenance names the source kernel


def test_population_segment_exposure_rejects_unsourced_parameters():
    op = PopulationSegmentExposureOperator()
    w = _world()
    good_segments = [{"belief": b, "responsiveness": r, "influence": inf}
                     for b, r, inf in _CELLS]
    # segments without provenance
    ev = Event(ts=T0 + 1, etype="population_segment_exposure",
               payload={"mean_field_spec": {"segments": good_segments,
                                            "couplings": _COUPLINGS, "steps": 3,
                                            "outcome_var": "s"}})
    delta, vr = op.run(w, ev, random.Random(0))
    assert delta is None and not vr.ok and "provenance" in vr.reasons[0]
    # a coupling without a source (the legacy in-code default is never silently used)
    bad_couplings = {**_COUPLINGS, "k_proof": {"value": 0.2, "source": ""}}
    ev2 = Event(ts=T0 + 1, etype="population_segment_exposure",
                payload={"mean_field_spec": {"segments": good_segments,
                                             "couplings": bad_couplings, "steps": 3,
                                             "outcome_var": "s",
                                             "provenance": {"segments": "census"}}})
    delta2, vr2 = op.run(w, ev2, random.Random(0))
    assert delta2 is None and not vr2.ok and "k_proof" in vr2.reasons[0]
    # a MISSING coupling is equally a rejection — no dataclass default sneaks in
    missing = {k: v for k, v in _COUPLINGS.items() if k != "k_social"}
    ev3 = Event(ts=T0 + 1, etype="population_segment_exposure",
                payload={"mean_field_spec": {"segments": good_segments,
                                             "couplings": missing, "steps": 3,
                                             "outcome_var": "s",
                                             "provenance": {"segments": "census"}}})
    delta3, vr3 = op.run(w, ev3, random.Random(0))
    assert delta3 is None and not vr3.ok and "k_social" in vr3.reasons[0]
    assert "s" not in w.quantities


# ================================================================ ports in the spec index
def test_ported_kernels_have_specs_with_correct_kinds_and_write_sets():
    idx = ms.build_spec_index()
    expect = {"poll_error_aggregation": "measurement", "whipcount_binomial": "institution",
              "outside_world_hazard": "exogenous", "population_segment_exposure": "population"}
    for mech_id, kind in expect.items():
        spec = idx[mech_id]
        assert spec.mechanism_kind == kind
        assert spec.version == "1.0.0" and spec.operator == mech_id
        assert spec.event_inputs and spec.parameter_sources
        assert spec.calibration_status in ms.CALIBRATION_STATUSES
    assert idx["outside_world_hazard"].event_outputs == ("outside_world_arrival",)
    # the outside-world spec's write set admits ONLY its bookkeeping namespace
    ow = idx["outside_world_hazard"]
    d = StateDelta(at=T0, event_type="x", operator="outside_world_hazard")
    d.change("quantities[outside_world_arrivals:cpi_print]", 0, 1)
    assert ms.declared_write_violations(ow, d) == []
    d.change("quantities[forecast_answer]", None, True)
    assert len(ms.declared_write_violations(ow, d)) == 1


def test_migration_manifest_covers_all_19_candidates():
    path = Path(__file__).resolve().parent.parent / "artifacts" / "core_arch" / \
        "kernel_migration_manifest.json"
    doc = json.loads(path.read_text())
    rows = doc["kernels"]
    assert len(rows) == 19
    for row in rows:
        for key in ("kernel", "old_location", "validated_semantics", "new_operator",
                    "parameter_source", "tests", "production_eligibility",
                    "reason_accepted_or_rejected"):
            assert key in row, f"manifest row missing {key}: {row.get('kernel')}"
    rejected = [r for r in rows if r["production_eligibility"] == "rejected"]
    ported = [r for r in rows if r["production_eligibility"] == "ported_contract_tested"]
    deferred = [r for r in rows if r["production_eligibility"] == "not_ported_this_round"]
    assert len(rejected) == 9 and len(ported) == 4 and len(deferred) == 6
    assert all(r["new_operator"] is None for r in rejected + deferred)
    assert {r["new_operator"].split(" ")[0] for r in ported} == {
        "poll_error_aggregation", "whipcount_binomial", "outside_world_hazard",
        "population_segment_exposure"}
    assert all(r["reason_accepted_or_rejected"].startswith("REJECTED") for r in rejected)
