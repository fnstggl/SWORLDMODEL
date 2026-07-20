"""§NAP enforcement — the NO-ARBITRARY-NUMERIC-REALITY contract, enforced four ways:

  1. STATIC: no production module references a quarantined numeric symbol; the only access path
     is legacy_numeric_ablations.legacy_numeric_table under the acknowledgement token (and the
     one explicitly legacy-gated writer inside phase4_execution).
  2. RUNTIME CALL-SPY: forbidden draws (lean-Beta sampling, legacy tables) raise if touched —
     ordinary default event-time execution never touches them.
  3. MUTATION INVARIANCE: multiplying every buried legacy constant by 1000 changes NOTHING about
     default execution, because production does not consume them.
  4. HONEST UNRESOLVED BEHAVIOR + PROVENANCE COMPLETENESS: a question with no validated
     mechanism returns `unresolved` (no forced yes/no, recommendations withheld, the missing
     mechanism named), and every result carries the numeric_causal_inputs manifest.
"""
import ast
import pathlib
import types

import pytest

import swm.world_model_v2 as _pkg

PKG_DIR = pathlib.Path(_pkg.__file__).parent
T0 = 1_700_000_000.0
T1 = T0 + 100 * 86400.0

#: every production module in the package EXCEPT the explicit quarantine/ablation modules and
#: the offline fitting utilities that only PRODUCE artifacts
_EXEMPT_FILES = {"legacy_numeric_ablations.py", "legacy_ablations.py"}

#: identifiers that may not appear in ANY production module (they exist only inside the
#: quarantine module). Checked as bare tokens over the AST's Name/Attribute nodes.
_FORBIDDEN_NAMES = (
    "ACTION_PATHWAY_EFFECTS", "action_pathway_effects", "actions_advancing_pathway",
    "stance_action_alignment", "STANCE_ORIENTATION", "RELIABILITY_SHRINK", "CAPABILITY_SHRINK",
    "CONTROL_WEIGHTS", "ENDOGENOUS_STANCE_SPLIT", "PROCESS_STATE_LEVELS", "INTENTION_HR_PRIORS",
    "COUPLING_PRIORS", "sampled_coupling", "declare_pathway_processes", "combine_stances",
    "pathway_orientation", "declare_actor_capacity", "contested_attrition_interval",
    "CAPACITY_INIT", "EFFORTFUL_ACTION_COST", "StanceReviewOperator", "_LEAN_SHIFT",
)
#: (filename, name) pairs allowed to keep the name: the definition site inside the quarantine
#: gate, or an explicitly legacy-gated caller
_ALLOWED = {
    ("phase4_execution.py", "legacy_numeric_table"),
    ("semantic_consequences.py", "derive_pathway_summaries"),
}


def _production_files():
    for path in sorted(PKG_DIR.rglob("*.py")):
        if path.name in _EXEMPT_FILES:
            continue
        yield path


def _names_in(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                yield a.name.rsplit(".", 1)[-1]


# ================================================================ 1. static enforcement
def test_no_production_module_references_quarantined_numeric_symbols():
    offenders = []
    for path in _production_files():
        for name in _names_in(path):
            if name in _FORBIDDEN_NAMES and (path.name, name) not in _ALLOWED:
                offenders.append(f"{path.name}:{name}")
    assert not offenders, ("quarantined numeric symbols referenced by production modules: "
                           f"{sorted(set(offenders))}")


def test_no_module_level_import_of_the_legacy_quarantine():
    """The quarantine module may be imported only INSIDE explicitly gated functions (the legacy
    writer, the token-gated summary projection), never at module level — production cannot drift
    into holding a live reference."""
    offenders = []
    for path in _production_files():
        tree = ast.parse(path.read_text())
        for node in tree.body:                                # module level only
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                names = " ".join(a.name for a in node.names)
                if "legacy_numeric_ablations" in mod or "legacy_numeric_ablations" in names:
                    offenders.append(path.name)
    assert not offenders, f"module-level legacy imports: {offenders}"


def test_event_time_has_no_lean_beta_or_family_rate_consumption():
    src = (PKG_DIR / "event_time.py").read_text()
    assert "LEAN_BETA" not in src
    assert 'get("fallback_rate")' not in src                   # the family-rate rung is gone
    assert '"fallback_rate"' not in src
    assert "_calibrated_target" not in src                     # the ladder itself is gone
    assert "intention_factor" not in src                       # no stance point factors
    assert "_apply_lean_shift" not in src                      # no hypothesis-lean shifts


def test_no_silent_env_door_for_the_new_quarantines():
    """The §NAP quarantines have NO environment-variable door: the only access is the literal
    acknowledgement token."""
    src = (PKG_DIR / "legacy_numeric_ablations.py").read_text()
    assert "os.environ" not in src and "getenv" not in src


# ================================================================ 2. runtime call spies
def _spy_raise(*a, **k):
    raise AssertionError("forbidden legacy numerical function was invoked on the default path")


def test_default_event_time_rollout_never_touches_beta_or_legacy_tables(monkeypatch):
    import swm.world_model_v2.fallback as fb
    import swm.world_model_v2.legacy_numeric_ablations as lna
    from tests.test_wmv2_event_time import _binary_plan, _rollout
    from swm.world_model_v2.event_time import convert_binary_to_event_time
    monkeypatch.setattr(fb, "_beta_sample", _spy_raise)
    monkeypatch.setattr(lna, "legacy_numeric_table", _spy_raise)
    # (a) posterior-parameterized: resolves through the residual process without any beta/table
    p = _binary_plan(posterior=[(0.4, 1.0)])
    convert_binary_to_event_time(p, {})
    out, _ = _rollout(p, n_particles=50)
    assert out["distribution"]["yes"] > 0.0
    # (b) nothing approved: fully unresolved without any beta/table
    p2 = _binary_plan(posterior=None)
    convert_binary_to_event_time(p2, {})
    out2, _ = _rollout(p2, n_particles=20)
    assert out2["distribution"]["unresolved_mechanism"] == pytest.approx(1.0)


def test_structural_process_prior_is_suppressed_by_default(monkeypatch):
    monkeypatch.delenv("SWM_ALLOW_GENERIC_PRIOR", raising=False)
    from swm.world_model_v2.phase_consumers import StructuralProcessPriorOperator
    from swm.world_model_v2.quantities import Quantity, register_quantity_type  # noqa: F401
    from swm.world_model_v2.state import SimulationClock, WorldState
    w = WorldState("w", "b1:x", SimulationClock(now=T0, as_of=T0))
    op = StructuralProcessPriorOperator()
    ev = types.SimpleNamespace(etype="structural_process_prior",
                               payload={"process": "adoption", "out_var": "proc_share",
                                        "lean": "weak_yes"})
    d = op.apply(w, op.propose(w, ev, None))
    assert d is None and "proc_share" not in w.quantities
    from swm.world_model_v2.numeric_provenance import unresolved_mechanisms_of
    assert any(r["mechanism"] == "structural_process:adoption"
               for r in unresolved_mechanisms_of(w))


def test_network_diffusion_transmissibility_is_suppressed_by_default(monkeypatch):
    monkeypatch.delenv("SWM_ALLOW_GENERIC_PRIOR", raising=False)
    from swm.world_model_v2.phase_consumers import NetworkDiffusionOperator
    from swm.world_model_v2.state import SimulationClock, WorldState
    w = WorldState("w", "b1:x", SimulationClock(now=T0, as_of=T0))
    op = NetworkDiffusionOperator()
    proposal = types.SimpleNamespace(action={"out_var": "reach", "seed_id": None},
                                     reason_codes=[])
    assert op.apply(w, proposal) is None
    assert "reach" not in w.quantities


# ================================================================ 3. mutation invariance
def test_mutating_every_legacy_constant_changes_nothing_in_default_execution():
    import swm.world_model_v2.legacy_numeric_ablations as lna
    from tests.test_wmv2_event_time import _binary_plan, _rollout
    from swm.world_model_v2.event_time import convert_binary_to_event_time

    def run_once():
        p = _binary_plan(posterior=[(0.35, 1.0)])
        convert_binary_to_event_time(p, {"resolves_yes_iff": "X departs"})
        out, _ = _rollout(p, n_particles=80, seed=11)
        return out["distribution"], out["event_time"]["cdf"]
    before = run_once()
    saved = {k: v for k, v in lna._TABLES.items()}
    try:
        for k, v in list(lna._TABLES.items()):
            if isinstance(v, dict):
                lna._TABLES[k] = {kk: (tuple(x * 1000 for x in vv) if isinstance(vv, tuple)
                                       else (vv * 1000 if isinstance(vv, (int, float))
                                             else {p2: e * 1000 for p2, e in vv.items()}))
                                  for kk, vv in v.items()}
            elif isinstance(v, (int, float)):
                lna._TABLES[k] = v * 1000
        after = run_once()
    finally:
        lna._TABLES.clear()
        lna._TABLES.update(saved)
    assert before == after, "default execution consumed a quarantined legacy constant"


# ================================================================ 4. honest unresolved behavior
def _fake_plan():
    return types.SimpleNamespace(
        question="q", as_of=T0, horizon_ts=T1, support_grade="exploratory", degraded=False,
        omissions=[], fallbacks_used=[], mechanism_choices=[], latents=[], interpretations=[],
        provenance={}, compute_plan={"n_particles": 20},
        outcome_contract=types.SimpleNamespace(options=["yes", "no"], readout_var="absorbed_at"),
        plan_hash=lambda: "h", _unresolved_mechanisms=[
            {"mechanism": "residual_outcome_process", "why": "no posterior (§NAP)",
             "missing": "evidence-updated posterior or eligible fitted rate"}])


def test_fully_unresolved_result_returns_unresolved_status_and_withholds_recommendations():
    from swm.world_model_v2.pipeline import result_from_run
    result = {"distribution": {"yes": 0.0, "no": 0.0, "unresolved_mechanism": 1.0},
              "event_time": {"unresolved_share": 1.0,
                             "branch_terminals": {"resolved_yes": 0.0, "resolved_no": 0.0,
                                                  "censored_by_real_horizon": 0.0,
                                                  "unresolved_mechanism": 1.0},
                             "bounds": {"yes": {"min_supported": 0.0, "max_possible": 1.0},
                                        "no": {"min_supported": 0.0, "max_possible": 1.0}},
                             "resolved_conditional": None,
                             "unresolved_mechanisms": ["residual_outcome_process"]}}
    res = result_from_run("q", _fake_plan(), result, [], intervention="what should we do?")
    assert res.simulation_status == "unresolved"
    assert not res.has_forecast()
    assert res.recommendation_status == "withheld"
    rr = res.resolution_report
    assert rr["unresolved_share"] == 1.0
    assert any(m["mechanism"] == "residual_outcome_process" for m in rr["missing_mechanisms"])
    assert "Outcome unresolved under the current model" in res.limitations[0]


def test_partially_resolved_result_keeps_unresolved_mass_and_bounds():
    from swm.world_model_v2.pipeline import result_from_run
    result = {"distribution": {"yes": 0.3, "no": 0.3, "unresolved_mechanism": 0.4},
              "event_time": {"unresolved_share": 0.4,
                             "bounds": {"yes": {"min_supported": 0.3, "max_possible": 0.7}},
                             "branch_terminals": {"resolved_yes": 0.3, "resolved_no": 0.3,
                                                  "censored_by_real_horizon": 0.3,
                                                  "unresolved_mechanism": 0.4},
                             "resolved_conditional": {"yes": 0.5, "no": 0.5},
                             "unresolved_mechanisms": ["mode_transition:deal"]}}
    res = result_from_run("q", _fake_plan(), result, [], intervention="advise me")
    assert res.simulation_status == "partially_resolved"
    assert res.recommendation_status == "withheld"
    assert res.raw_distribution["unresolved_mechanism"] == pytest.approx(0.4)
    # unresolved mass is NOT normalized away
    assert sum(res.raw_distribution.values()) == pytest.approx(1.0)


# ================================================================ 5. provenance completeness
def test_numeric_manifest_reaches_the_result_and_names_every_load_bearing_input():
    from swm.world_model_v2.pipeline import result_from_run
    from tests.test_wmv2_event_time import _binary_plan, _rollout
    from swm.world_model_v2.event_time import convert_binary_to_event_time
    p = _binary_plan(posterior=[(0.4, 1.0)])
    convert_binary_to_event_time(p, {})
    out, branches = _rollout(p, n_particles=30)
    plan = _fake_plan()
    plan._numeric_ledger = p._numeric_ledger
    plan._unresolved_mechanisms = []
    res = result_from_run("q", plan, out, branches)
    man = res.resolution_report["numeric_causal_inputs"]
    consumed = {e["name"] for e in man["approved_and_consumed"]}
    rejected = {e["name"] for e in man["rejected"]}
    assert "residual_target_mass" in consumed                  # the load-bearing input is registered
    assert {"family_fallback_rate", "lean_beta_target"} <= rejected
    assert man["n_rejected"] >= 2


def test_fitted_artifact_eligibility_gate():
    from swm.world_model_v2.numeric_provenance import fitted_artifact_eligible
    ok, why = fitted_artifact_eligible({"version": "x"})
    assert not ok
    full = {"version": "v1", "training_population": "resolved 2024 markets",
            "outcome_definition": "binary yes-by-deadline", "data_cutoff": "2025-01-01",
            "n": 400, "fitting_procedure": "partial pooling", "heldout_metrics": {"brier": 0.18},
            "calibration": {"ece": 0.03}, "domain_restrictions": "markets only",
            "transport_check": "passed", "architecture_version": "nap-1"}
    ok, why = fitted_artifact_eligible(full)
    assert ok, why
    ok, why = fitted_artifact_eligible({**full, "n": 10})
    assert not ok and "sample support" in why
    ok, why = fitted_artifact_eligible({**full, "transport_check": "failed"})
    assert not ok


def test_unapproved_source_class_raises_and_is_recorded():
    from swm.world_model_v2.numeric_provenance import (NumericProvenanceLedger,
                                                       NumericProvenanceRejected)
    led = NumericProvenanceLedger()
    with pytest.raises(NumericProvenanceRejected):
        led.approve(name="x", value=0.5, units="probability", causal_role="social effect",
                    source_class="llm_estimated", consumer="test")
    man = led.manifest()
    assert man["rejected"] and man["rejected"][0]["name"] == "x"
    # compute-safety values register but can never alter the terminal
    led.approve(name="max_events", value=500, units="events", causal_role="safety budget",
                source_class="compute_safety", consumer="runtime")
    assert led.manifest()["compute_safety_only"][0]["can_alter_terminal"] is False


# ================================================================ 6. no numerical psychology
def test_qualitative_actor_carries_no_capacity_costs():
    src = (PKG_DIR / "qualitative_actor.py").read_text()
    assert "EFFORTFUL_ACTION_COST" not in src
    src2 = (PKG_DIR / "fidelity.py").read_text()
    assert "EFFORTFUL_ACTION_COST" not in src2


def test_temporal_runtime_has_no_stance_watch_numerics():
    src = (PKG_DIR / "temporal_runtime.py").read_text()
    assert "STANCE_MATERIAL_CHANGE = 0.08" not in src
    assert "_STANCE_WATCH_THRESHOLDS" not in src
    assert "contested_attrition_interval" not in src
