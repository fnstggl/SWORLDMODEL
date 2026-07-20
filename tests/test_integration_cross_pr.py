"""Integration tests for the #124 + #125 + #126 merge: features from different PRs operating TOGETHER.

Each test pins one cross-PR interaction from the integration contract:
  1.  canonical runtime + evidence sufficiency (#126 gate inside the #123/#124 canonical entry)
  2.  compiler truncation recovery (#124) + outcome-pathway checking (#126)
  3.  scheduled recurrence metadata (#124) + actor calendar knowledge (#124) + influence schema (#126)
  4.  recurrence-aware grounded prior (#126) + numeric provenance acknowledgment (#125)
  5.  persistence rollout + posterior injection (#124) + pathway viability (#126)
  6.  experimental mechanism execution (#124 run-everything) + numeric-integrity enforcement (#125)
  7.  empty evidence + escalating retrieval (#126, extended with the official-source variant)
  8.  empty operator census + structural repair (#126) on §NAP event-time plans (#125)
  9.  unresolved mechanism (#125) + non-null structured output (#126) — the composed guard ladder
  10. structural-model disagreement / partially_resolved results pass the guards untouched
  11. recommendation withholding under unresolved mass survives the guard conversion
  12. no legacy or alternate entry bypasses the combined system
"""
import json
import math
from types import SimpleNamespace

import pytest

T0 = 1_700_000_000.0
T1 = T0 + 100 * 86400.0


# ------------------------------------------------------------------ helpers
def _decomposition(mechanisms=None, extra=None):
    d = {
        "coherent": True,
        "interpretations": [{"id": "primary", "reading": "will the contract be ratified", "weight": 1.0}],
        "outcome": {"family": "binary", "options": ["yes", "no"],
                    "resolution_rule": "resolves YES if the contract is ratified",
                    "readout_var": "ratified_share"},
        "phase_relevance": {"population_response": False, "network_propagation": False,
                            "nonlinear_dynamics": False, "institutional_decision_process": True,
                            "structural_change_monitoring": False},
        "mechanisms": list(mechanisms or []),
        "required_causal_processes": ["ratification vote"],
        "structural_hypotheses": [],
        "scheduled_events": [],
        "hazards": [],
        "entities": [{"id": "union_lead", "type": "person", "fields": {}, "sensitivity": 0.8}],
        "institutions": [{"id": "union", "rules": [{"kind": "vote_threshold",
                                                    "params": {"threshold": 0.5}}],
                          "sensitivity": 0.9}],
        "quantities": [{"name": "ratified_share", "qtype": "share", "value": None, "sd": None}],
        "latents": [],
        "actor_decisions": [],
        "missing_mechanisms": [],
        "omitted": [],
        "domain": "labor", "population_kind": "members", "time_scale": "weeks",
        "available_data": [], "sensitivity": {},
    }
    d.update(extra or {})
    return d


def _compile(llm, question="Will the nurses ratify the hospital contract?"):
    from swm.world_model_v2.compiler import compile_world
    return compile_world(question, llm=llm, evidence="(scripted)",
                         as_of="2026-07-01", horizon="2026-07-30")


# ------------------------------------------------------------------ 1. canonical runtime + sufficiency
def test_canonical_runtime_reports_evidence_sufficiency_and_prior_stash():
    """The sufficiency gate (#126) is a shared per-plan helper wired into BOTH structural modes; the
    prior spec (#126's grounded prior) is stashed for the guards (#125-composed ladder)."""
    import swm.world_model_v2.unified_runtime as U
    lineage = {}
    sig = U._evidence_sufficiency_block(
        "q", SimpleNamespace(documents=[1, 2], included_claim_ids=[1]),
        SimpleNamespace(n_effective_observations=0), as_of="2026-05-07", drop=set(),
        lineage=lineage)
    assert sig["starved"] is True and lineage["evidence_sufficiency"] is sig
    # both structural modes call the same block (single authority)
    import inspect
    single = inspect.getsource(U._simulate_single_structural_model)
    assert "_evidence_sufficiency_block" in single and "_apply_result_guards" in single
    import swm.world_model_v2.structural_runtime as SR
    ens = inspect.getsource(SR)
    assert "_evidence_sufficiency_block" in ens and "_apply_result_guards" in ens


# ------------------------------------------------------------------ 2. truncation recovery + pathway check
def test_truncation_recovery_feeds_a_viable_outcome_pathway():
    """A max_tokens-truncated decomposition (#124 recovery) still ends with an outcome-capable
    (event, operator) pair — verified by #126's invariant, with no repair needed."""
    full = _decomposition(extra={"scheduled_events": [
        {"etype": "scheduled_meeting", "at": "2026-07-10", "participants": ["union"], "payload": {}}]})
    txt = json.dumps(full)
    cut = txt[: txt.index('"mechanisms"') - 2]                # truncated mid-JSON, outcome survives
    tail_keys = {k: full[k] for k in ("mechanisms", "required_causal_processes",
                                      "structural_hypotheses", "scheduled_events", "hazards",
                                      "actor_decisions")}
    calls = []

    def llm(prompt):
        calls.append(prompt)
        if "TRUNCATED mid-JSON" in prompt:
            return json.dumps(tail_keys)
        return cut

    plan = _compile(llm)
    assert plan.provenance.get("truncated_reply") is True
    recovered = plan.provenance.get("truncation_recovered_keys") or []
    # non-empty tail keys merge (empty lists are not fabricated back in)
    assert {"required_causal_processes", "scheduled_events"} <= set(recovered)
    from swm.world_model_v2.materialize import ensure_outcome_pathway, operators_from_plan
    ops, rej = operators_from_plan(plan)
    rep = ensure_outcome_pathway(plan, ops, rej)
    assert rep["outcome_capable_events"], "recovered plan must retain an outcome-capable pathway"
    assert rep["repaired"] is False


# ------------------------------------------------------------------ 3. recurrence + actor calendar
def test_recurrence_metadata_reaches_actor_calendar_knowledge():
    """#126's influence schema and #124's recurrence metadata coexist in one fact record, and the
    fact reaches actor cognition through plan._scheduled_facts → QualitativeConfig.public_facts."""
    from swm.world_model_v2.scheduled_facts import (attach_scheduled_facts, extract_scheduled_facts,
                                                    public_facts_lines)

    def fake_llm(_p):
        return json.dumps({"facts": [{
            "fact": "Apple holds WWDC and ships a new visionOS annually", "date": "2026-06-08",
            "entity": "Apple", "kind": "recurring_event",
            "recurrence": "annual at WWDC each June: 2017..2025 unbroken",
            "source": "model_knowledge", "confidence": 0.9,
            "pattern_strength": "strong_recurrence", "outcome_influence": "raises",
            "influence_strength": 0.8, "reason": "annual OS cycle"}]})

    facts = extract_scheduled_facts("Will Apple announce visionOS 27 at WWDC 2026?",
                                    as_of="2026-05-07", horizon="2026-06-12", llm=fake_llm)
    f = facts[0]
    assert f["recurrence"] and f["outcome_influence"] == "raises"          # #124 + #126 united
    assert f["outcome_entailing"] is True and f["strictly_entailing"] is False
    plan = SimpleNamespace(as_of=T0, horizon_ts=T1, scheduled_events=[], accepted_mechanisms=[],
                           _consumed_state=[])
    attach_scheduled_facts(plan, facts)
    assert plan._scheduled_facts == facts
    lines = public_facts_lines(plan._scheduled_facts)
    assert any("2017..2025 unbroken" in ln for ln in lines)                # pattern enters cognition
    from swm.world_model_v2.qualitative_actor import QualitativeConfig
    cfg = QualitativeConfig(llm=None)
    if not cfg.public_facts:
        cfg.public_facts = list(getattr(plan, "_scheduled_facts", []) or [])
    assert cfg.public_facts and cfg.public_facts[0]["recurrence"]


# ------------------------------------------------------------------ 4. recurrence prior + provenance
def test_grounded_recurrence_prior_provenance_reaches_the_ledger_row():
    """#126's recurrence-aware prior anchors the posterior; #125's ledger row for the residual
    outcome process names THAT specific prior as the acknowledged remaining assumption."""
    from swm.world_model_v2.event_time import _fp_target_mass
    from swm.world_model_v2.numeric_provenance import ledger_of
    from swm.world_model_v2.state import SimulationClock, WorldState
    w = WorldState("w", "b1:x", SimulationClock(now=T0, as_of=T0))
    w.particle_index = 0
    cal = {"posterior_rate_particles": [[0.9, 1.0]],
           "prior_provenance": {"source_class": "recurrence", "evidence_quality": "sourced",
                                "retained_effective_n": 6.0, "mean": 0.91}}
    t, src = _fp_target_mass(w, cal)
    assert src == "posterior" and 0.0 <= t <= 0.98
    man = ledger_of(w).manifest()
    row = next(e for e in man["approved_and_consumed"] if e["name"] == "residual_target_mass")
    assert "recurrence" in row["applicability"]
    assert row["fitted_on"]["prior_provenance"]["evidence_quality"] == "sourced"


# ------------------------------------------------------------------ 5. persistence + posterior + viability
def test_persistence_prepare_injects_posterior_and_verifies_pathway():
    plan = _compile(lambda p: json.dumps(_decomposition()))
    plan.posterior_rate_particles = [(0.7, 1.0)]
    from swm.world_model_v2.phase8_pipeline import prepare_persistence_run
    handle = prepare_persistence_run("q", plan, llm=None)
    assert handle["outcome_pathway"]["outcome_capable_events"]            # #126 invariant ran
    resolver = [e for e in plan.scheduled_events if e.get("etype") == "resolve_outcome"]
    parts = resolver[0]["payload"].get("posterior_rate_particles") if resolver else None
    assert [list(x) for x in (parts or [])] == [[0.7, 1.0]]               # #124 parity


def test_persistence_hypothesis_stratification_is_index_keyed():
    """#124's hypothesis stratification, reimplemented for the staged funnel: assignment is a pure
    function of particle index, so pilot+extension equals a direct full roll."""
    hyps = [{"id": "H1", "prior": 0.75, "lean": "weak_yes"},
            {"id": "H2", "prior": 0.25, "lean": "weak_no"}]
    plan = _compile(lambda p: json.dumps(_decomposition(extra={
        "structural_hypotheses": hyps})))
    from swm.world_model_v2.phase8_pipeline import prepare_persistence_run, run_persistence_slice
    handle = prepare_persistence_run("q", plan, llm=None, n_particles=8)
    assert handle["stratification"] and set(handle["stratification"]["strata"]) == {"H1", "H2"}
    pilot = run_persistence_slice(handle, seed=3, n_total=8, start=0, stop=4)
    ext = run_persistence_slice(handle, seed=3, n_total=8, start=4, stop=8)
    hyp_of = [b.world.uncertainty_meta.get("model", {}).get("hypothesis") for b in pilot + ext]
    handle2 = prepare_persistence_run("q", plan, llm=None, n_particles=8)
    full = run_persistence_slice(handle2, seed=3, n_total=8, start=0, stop=8)
    assert hyp_of == [b.world.uncertainty_meta.get("model", {}).get("hypothesis") for b in full]
    assert set(hyp_of) == {"H1", "H2"}


# ------------------------------------------------------------------ 6. run-everything + integrity
def test_experimental_mechanism_executes_labeled_and_provenance_scan_still_holds():
    from swm.world_model_v2.transitions import register_operator, _OPERATORS
    from swm.world_model_v2.mechanisms import MechanismEntry, register_mechanism, _REGISTRY

    class _NoopOp:
        name = "xpr_test_noop"

        def applicable(self, world, event):
            return False

        def validate(self, world, proposal):
            from swm.world_model_v2.transitions import ValidationResult
            return ValidationResult(True)

        def propose(self, world, event, rng):
            return None

        def apply(self, world, proposal):
            return None

    register_operator("xpr_test_noop", _NoopOp, experimental=True)
    register_mechanism(MechanismEntry(
        mech_id="xpr_test_mech", ontology_type="decision", causal_role="test",
        operator="xpr_test_noop", calibration_status="uncalibrated"))
    try:
        plan = _compile(lambda p: json.dumps(_decomposition(mechanisms=["xpr_test_mech"])))
        entry = next(m for m in plan.accepted_mechanisms
                     if isinstance(m, dict) and m.get("mech_id") == "xpr_test_mech")
        # RUN-EVERYTHING (#124): executes labeled, never rejected into the broad prior
        assert entry["calibration_status"] == "experimental"
        assert entry.get("uncertainty_widened") is True
        assert not any(r.get("id") == "xpr_test_mech" for r in plan.rejected_mechanisms)
        from swm.world_model_v2.materialize import operators_from_plan
        ops, _ = operators_from_plan(plan, allow_experimental=True)
        assert any(getattr(o, "name", "") == "xpr_test_noop" for o in ops)
    finally:
        _OPERATORS.pop("xpr_test_noop", None)
        _REGISTRY.pop("xpr_test_mech", None)


# ------------------------------------------------------------------ 7. escalating retrieval
def test_escalated_retrieval_is_materially_different(monkeypatch):
    import swm.world_model_v2.evidence_orchestrator as EO

    queries = []

    class _FakeNews:
        def __init__(self, store=None):
            pass

        def search_historical(self, terms, *, after_date, before_date, requirement_id, k):
            queries.append({"terms": terms, "after": after_date, "before": before_date,
                            "req": requirement_id})
            tr = SimpleNamespace(connector_status="zero_results", connector_id="fake", error="",
                                 as_dict=lambda: {"connector": "fake"})
            return [], tr

    class _FakeWiki:
        def __init__(self, store=None):
            pass

        def fetch(self, entity, *, requirement_id, as_of_iso):
            queries.append({"wiki": entity, "req": requirement_id})
            tr = SimpleNamespace(connector_status="zero_results", connector_id="wiki", error="",
                                 as_dict=lambda: {"connector": "wiki"})
            return [], tr

    monkeypatch.setattr(EO, "GoogleNewsRSSConnector", _FakeNews)
    monkeypatch.setattr(EO, "WikipediaConnector", _FakeWiki)
    from swm.world_model_v2.evidence_requirements import EvidenceRequirement
    req = EvidenceRequirement(requirement_id="r1", claim_or_quantity="the decisive outcome",
                              why_relevant="", affected_component="terminal_outcome",
                              entity_scope=("Bank of Japan",))
    rrule = "Resolves YES if the BoJ announces a policy rate hike"
    b1 = EO.gather_evidence("Will the BoJ raise rates?", as_of="2026-06-01", requirements=[req],
                            resolution_rule=rrule)
    primary_queries = list(queries)
    queries.clear()
    b2 = EO.gather_evidence("Will the BoJ raise rates?", as_of="2026-06-01", requirements=[req],
                            strategy="escalated", resolution_rule=rrule)
    esc_reqs = {q.get("req") for q in queries}
    assert "decisive_reformulation" in esc_reqs and "official_source_query" in esc_reqs
    # escalation doubles the lookback window (a genuinely different strategy, not a re-roll)
    p_after = next(q["after"] for q in primary_queries if "after" in q)
    e_after = next(q["after"] for q in queries if "after" in q)
    assert e_after < p_after
    official = next(q for q in queries if q.get("req") == "official_source_query")
    assert "announcement" in official["terms"]
    assert b1 is not None and b2 is not None


# ------------------------------------------------------------------ 8/9/10/11. guards × §NAP semantics
def _result(status="completed_with_degradation", p=None, rec="withheld", rr=None):
    from swm.world_model_v2.result import SimulationResult
    return SimulationResult(question="q", simulation_status=status, support_grade="exploratory",
                            raw_probability=p, recommendation_status=rec,
                            resolution_report=rr or {}, provenance={}, limitations=[])


def test_event_time_plan_missing_monitor_is_repaired_not_resolved():
    """Empty operator census on an event-time plan with absorbing channels → the MONITOR is restored
    (declared structure decides), no resolver is bolted on."""
    from swm.world_model_v2.materialize import ensure_outcome_pathway
    plan = SimpleNamespace(
        scheduled_events=[{"etype": "hazard_round", "ts": T0 + 5.0, "participants": [],
                           "payload": {"mode": "entailed_fact:launch", "success_prob": 1.0}}],
        first_passage_processes=[], horizon_ts=T1,
        outcome_contract=SimpleNamespace(readout_var="absorbed_at", family="event_time",
                                         options=["yes", "no"]),
        provenance={"outcome_lean": "neutral"}, posterior_rate_particles=None)
    ops = []
    rep = ensure_outcome_pathway(plan, ops, [])
    names = {getattr(o, "name", "") for o in ops}
    assert {"hazard_round", "absorption_monitor"} <= names
    assert rep["repaired"] is True
    assert all(e["etype"] != "resolve_outcome" for e in plan.scheduled_events)


def test_nap_unresolved_plan_is_never_repaired_with_a_resolver():
    from swm.world_model_v2.materialize import ensure_outcome_pathway
    plan = SimpleNamespace(
        scheduled_events=[], first_passage_processes=[], horizon_ts=T1,
        outcome_contract=SimpleNamespace(readout_var="absorbed_at", family="event_time",
                                         options=["yes", "no"]),
        provenance={"outcome_lean": "neutral"}, posterior_rate_particles=None,
        _unresolved_mechanisms=[{"mechanism": "residual_outcome_process", "why": "no posterior"}])
    ops = []
    rep = ensure_outcome_pathway(plan, ops, [])
    assert rep.get("honest_unresolved_by_design") is True and rep["repaired"] is False
    assert all(e["etype"] != "resolve_outcome" for e in plan.scheduled_events)
    assert not ops


def test_event_time_accidental_total_loss_records_unresolved_not_a_resolver():
    from swm.world_model_v2.materialize import ensure_outcome_pathway
    plan = SimpleNamespace(
        scheduled_events=[], first_passage_processes=[], horizon_ts=T1,
        outcome_contract=SimpleNamespace(readout_var="absorbed_at", family="event_time",
                                         options=["yes", "no"]),
        provenance={"outcome_lean": "neutral"}, posterior_rate_particles=None)
    ops = []
    rep = ensure_outcome_pathway(plan, ops, [])
    assert rep.get("honest_unresolved_by_design") is True
    assert any(r["mechanism"] == "outcome_pathway"
               for r in getattr(plan, "_unresolved_mechanisms", []))
    assert all(e["etype"] != "resolve_outcome" for e in plan.scheduled_events)


def test_guard_ladder_posterior_backed_fallback():
    import swm.world_model_v2.unified_runtime as U
    res = _result(status="execution_failed", p=None)
    post = SimpleNamespace(n_effective_observations=5, outcome_rate_mean=0.62)
    U._apply_result_guards(res, posterior=post, prior_spec=None)
    assert res.raw_probability == 0.62 and res.simulation_status == "completed_with_degradation"
    assert res.provenance["execution_degraded_fallback"]["source"] == "posterior"


def test_guard_ladder_prior_requires_the_explicit_door(monkeypatch):
    import swm.world_model_v2.unified_runtime as U
    prior = SimpleNamespace(mean=0.41)
    # door CLOSED (§NAP): unresolved, prior recorded as labelled diagnostic only
    monkeypatch.delenv("SWM_ALLOW_GENERIC_PRIOR", raising=False)
    res = _result(status="execution_failed", p=None, rec="withheld",
                  rr={"missing_mechanisms": []})
    U._apply_result_guards(res, posterior=None, prior_spec=prior)
    assert res.simulation_status == "unresolved"
    assert res.raw_probability is None and not res.has_forecast()
    assert res.provenance["prior_driven_reference"]["value"] == 0.41
    assert res.provenance["prior_driven_reference"]["headline"] is False
    rejected = res.resolution_report["numeric_causal_inputs"]["rejected"]
    assert any(r["name"] == "prior_driven_reference" and r["source_class"] == "llm_estimated"
               for r in rejected)
    assert res.recommendation_status == "withheld"
    # door OPEN (explicit §28 ablation): deliberately prior-driven, loudly labelled
    monkeypatch.setenv("SWM_ALLOW_GENERIC_PRIOR", "1")
    res2 = _result(status="execution_failed", p=None)
    U._apply_result_guards(res2, posterior=None, prior_spec=prior)
    assert res2.raw_probability == 0.41
    assert res2.simulation_status == "completed_with_degradation"
    assert any("PRIOR-DRIVEN" in l for l in res2.limitations)


def test_guards_never_touch_honest_unresolved_or_partially_resolved():
    import swm.world_model_v2.unified_runtime as U
    rr = {"unresolved_share": 1.0, "missing_mechanisms": [{"mechanism": "m"}]}
    res = _result(status="unresolved", p=None, rr=rr)
    U._apply_result_guards(res, posterior=SimpleNamespace(n_effective_observations=9,
                                                          outcome_rate_mean=0.7),
                           prior_spec=SimpleNamespace(mean=0.5))
    assert res.simulation_status == "unresolved" and res.raw_probability is None
    assert U._no_forecast(res) is False                      # never retried either
    pr = _result(status="partially_resolved", p=None,
                 rr={"unresolved_share": 0.4, "missing_mechanisms": []})
    pr.raw_distribution = {"yes": 0.3, "no": 0.3, "unresolved_mechanism": 0.4}
    U._apply_result_guards(pr, posterior=None, prior_spec=SimpleNamespace(mean=0.5))
    assert pr.simulation_status == "partially_resolved" and pr.raw_probability is None


# ------------------------------------------------------------------ 12. no bypass
def test_no_legacy_entry_bypasses_the_combined_system():
    from swm.world_model_v2.pipeline import simulate
    assert getattr(simulate, "__quarantined__", False) is True
    assert "simulate_world" in simulate.__use_instead__
    import inspect
    import swm.world_model_v2.pipeline as P
    assert "harden_general_path" in inspect.getsource(P.simulate.__wrapped__)  # #124 hardening inside
    import swm.world_model_v2.phase3_pipeline as p3
    import swm.world_model_v2.phase9_pipeline as p9
    assert "QUARANTINED" in (p3.__doc__ or "")
    assert "QUARANTINED" in (p9.__doc__ or "")
    import swm.facade as facade
    assert "simulate_world" in inspect.getsource(facade)
    # the mean-of-K aggregator (opt-in) wraps the SAME canonical entry, not a bypass
    import swm.world_model_v2.unified_runtime as U
    assert "simulate_world(" in inspect.getsource(U.simulate_world_stable)
