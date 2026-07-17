"""Mandatory phase supervision — the runtime, not the compiler, owns phase completeness.

Every normal simulation invokes ONE supervisor pass over all eleven phases. Each phase gets a
`PhaseExecutionRecord`; the active-component manifest is DERIVED from these records (there is no separate,
weaker activation accounting). A phase can never disappear silently:

  causally_active            — relevant, inputs present, executed, produced StateDeltas (or, for
                               non-operator phases, produced its documented artifact: bundle/posterior/
                               checkpoint/recompilation traces)
  no_op_causally_irrelevant  — assessed NOT causally required; the ONLY normal no-op
  blocked_missing_state      — relevant but the state it must execute over was never declared/materialized
  blocked_no_mechanism       — relevant, state present, but no executable mechanism/operator fired
  blocked_invalid_contract   — relevant but its input contract is unsatisfiable (e.g. no as_of for evidence)
  execution_failed           — an exception inside the phase

Any `blocked_*` on a RELEVANT phase is an integration failure: it lowers support, enters the failure
artifact (`provenance.phase_integration_failures`), and fails the phase gate — the run may not be labeled
completely integrated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

PHASES = ("phase1_compiler", "phase2_evidence", "phase3_posterior", "phase4_actor_policy",
          "phase6_registry", "phase7_nonlinear", "phase8_persistence", "phase9_populations",
          "phase9_networks", "phase10_institutions", "phase11_recompilation")

STATUSES = ("causally_active", "no_op_causally_irrelevant", "blocked_missing_state",
            "blocked_no_mechanism", "blocked_invalid_contract", "execution_failed")

#: operator families per operator-phase — the census key the record derives StateDelta evidence from.
PHASE_OPERATORS = {
    "phase4_actor_policy": ("production_actor_policy", "agent_decision", "fitted_decision",
                            "actor_action_aggregation"),
    "phase6_registry": ("behavioral_mechanism", "feature_hazard", "structural_process_prior"),
    "phase7_nonlinear": ("nonlinear_state_step", "nonlinear_mechanism", "nonlinear_contagion"),
    "phase9_populations": ("population_aggregation",),
    "phase9_networks": ("network_diffusion",),
    "phase10_institutions": ("institutional_decision", "institutional_vote", "institution_action"),
}

#: what state each operator-phase needs declared on the plan to be executable at all.
PHASE_STATE_REQUIREMENTS = {
    "phase4_actor_policy": "entities (a strategic actor) or actor_decisions",
    "phase6_registry": "a required causal process a registry family answers",
    "phase7_nonlinear": "a nonlinear causal process",
    "phase9_populations": "populations",
    "phase9_networks": "relations",
    "phase10_institutions": "institutions",
}


@dataclass
class PhaseExecutionRecord:
    phase: str
    available: bool = True
    relevance_assessed: bool = False
    relevant: bool = False
    relevance_reasons: str = ""
    input_state_requirements: str = ""
    input_state_present: bool = False
    scenario_state_materialized: bool = False
    selected_mechanism: str = ""
    selected_parameter_pack: str = ""
    event_ids: list = field(default_factory=list)
    execution_status: str = "no_op_causally_irrelevant"
    no_op_reason: str = ""
    n_state_deltas: int = 0
    state_fields_read: list = field(default_factory=list)
    state_fields_written: list = field(default_factory=list)
    downstream_events_produced: list = field(default_factory=list)
    terminal_influence: str = ""            # direct_resolution | rate_modulation | constraint | none | unknown
    ablation_influence: float = None        # filled by the ablation harness, not here
    latency_s: float = 0.0
    cost: dict = field(default_factory=dict)
    implementation_version: str = ""
    validation_status: str = ""
    support_implication: str = ""
    errors: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def _plan_state_present(phase, plan) -> bool:
    if phase == "phase4_actor_policy":
        return bool(getattr(plan, "entities", []) or getattr(plan, "actor_decisions", []))
    if phase == "phase9_populations":
        return True                                          # segments constructible from broad priors
    if phase == "phase9_networks":
        return bool(getattr(plan, "relations", []) or
                    len([e for e in (getattr(plan, "entities", []) or []) if isinstance(e, dict)]) >= 2 or
                    getattr(plan, "populations", []))         # relations are inferable from the causal world
    if phase == "phase10_institutions":
        return bool(getattr(plan, "institutions", []))
    return True                                             # p6/p7 execute from mechanisms/events, not sections


def assess(plan, *, has_as_of=True, has_bundle=None, has_posterior=None, versions=None,
           req=None) -> dict:
    """The supervisor pass: one PhaseExecutionRecord per phase, relevance + input-contract verdicts filled.
    Execution evidence (deltas, terminal influence) is finalized after the rollout via `finalize`.

    `req` must be the SAME relevance verdict the synthesis step used (computed BEFORE synthesis mutated
    the plan): re-deriving it afterwards lets synthesis-added mechanism prose lexically trigger unrelated
    phases (e.g. an institutional 'threshold rule' phrase firing the nonlinear detector), producing
    phantom blocked-relevant verdicts."""
    from swm.world_model_v2.activation_synthesis import phase_requirements
    req = req if req is not None else phase_requirements(plan)
    versions = versions or {}
    records = {}
    for ph in PHASES:
        r = PhaseExecutionRecord(phase=ph, relevance_assessed=True,
                                 implementation_version=str(versions.get(ph, "")),
                                 input_state_requirements=PHASE_STATE_REQUIREMENTS.get(ph, "core pipeline"))
        if ph == "phase1_compiler":
            r.relevant, r.relevance_reasons = True, "always required"
            r.input_state_present = True
        elif ph == "phase2_evidence":
            r.relevant = bool(has_as_of)
            r.relevance_reasons = "as_of supplied" if has_as_of else "no as_of — evidence contract unsatisfiable"
            r.input_state_present = bool(has_as_of)
        elif ph == "phase3_posterior":
            r.relevant = bool(has_bundle) if has_bundle is not None else bool(has_as_of)
            r.relevance_reasons = "evidence bundle present" if r.relevant else "no evidence bundle"
            r.input_state_present = r.relevant
        elif ph == "phase8_persistence":
            r.relevant, r.relevance_reasons = True, "default-on persistence-aware rollout"
            r.input_state_present = True
        elif ph == "phase11_recompilation":
            r.relevant = bool(has_bundle) if has_bundle is not None else bool(has_as_of)
            r.relevance_reasons = "observations available" if r.relevant else "no observations"
            r.input_state_present = r.relevant
        else:
            g = req.get(ph, {"required": False, "why": "not assessed"})
            r.relevant, r.relevance_reasons = bool(g["required"]), str(g["why"])
            r.input_state_present = _plan_state_present(ph, plan)
            # a causal SIGNAL with no declared structure to execute over is a missing-state block on a
            # RELEVANT phase — never a normal no-op (the compiler failed to declare what the causal
            # analysis says the outcome depends on).
            if not r.relevant and g.get("signal") and not r.input_state_present:
                r.relevant = True
        if not r.relevant:
            r.execution_status = "no_op_causally_irrelevant"
            r.no_op_reason = r.relevance_reasons
        elif not r.input_state_present:
            r.execution_status = "blocked_missing_state"
        records[ph] = r
    return records


def finalize(records: dict, plan, res, *, phase_meta: dict = None) -> dict:
    """Fill execution evidence from the rollout's operator-delta census + core-phase metadata, escalate
    blocked relevant phases into the failure artifact, and derive the manifest. Returns
    {records, manifest, integration_failures, fully_integrated}."""
    census = ((getattr(res, "provenance", {}) or {}).get("operator_delta_census")) or {}
    meta = phase_meta or {}
    mechs = {str(m.get("operator", "")): m for m in (getattr(plan, "accepted_mechanisms", []) or [])
             if isinstance(m, dict)}
    for ph, ops in PHASE_OPERATORS.items():
        r = records[ph]
        selected = [o for o in ops if o in mechs]
        if selected:
            r.selected_mechanism = ",".join(selected)
            r.selected_parameter_pack = ";".join(
                str(mechs[o].get("parameter_source", ""))[:60] for o in selected)
        n, written, evtypes = 0, [], []
        for o in ops:
            c = census.get(o)
            if c:
                n += c["n_deltas"]
                written += [f for f in c["fields_written"] if f not in written]
                evtypes += [e for e in c["event_types"] if e not in evtypes]
        r.n_state_deltas = n
        r.state_fields_written = written[:8]
        r.downstream_events_produced = evtypes[:6]
        if r.relevant and r.execution_status != "blocked_missing_state":
            if n > 0:
                r.execution_status = "causally_active"
                r.terminal_influence = _terminal_influence(ph, plan, written)
            elif selected:
                r.execution_status = "blocked_no_mechanism"
                r.no_op_reason = "mechanism selected but produced no StateDelta (event never consumed)"
            else:
                r.execution_status = "blocked_no_mechanism"
                r.no_op_reason = "no executable mechanism selected for a relevant phase"
    # ---- core phases from runtime metadata: {phase: {executed, reason, error}} ----
    for ph in ("phase1_compiler", "phase2_evidence", "phase3_posterior", "phase8_persistence",
               "phase11_recompilation"):
        r = records[ph]
        m = meta.get(ph, {})
        if m.get("error"):
            r.execution_status = "execution_failed"
            r.errors.append(str(m["error"])[:160])
        elif r.relevant:
            if m.get("executed", True):
                r.execution_status = "causally_active"
                r.no_op_reason = ""
            else:
                reason = str(m.get("reason", ""))
                # a posterior/evidence pass that ran but found nothing to update on is a NORMAL no-op
                # (the phase executed its contract; the world simply carried no admissible signal) —
                # only a genuine contract failure is blocked.
                benign = any(t in reason.lower() for t in
                             ("no admissible", "did not update", "no as_of", "no observations",
                              "0 eff obs", "no evidence bundle"))
                r.execution_status = "no_op_causally_irrelevant" if benign else "blocked_invalid_contract"
                r.no_op_reason = reason[:120]
        if m.get("reason") and not r.relevance_reasons.endswith(str(m["reason"])[:40]):
            r.validation_status = str(m.get("reason", ""))[:120]
    failures = [{"phase": ph, "status": rec.execution_status, "reason": rec.no_op_reason or
                 rec.relevance_reasons, "requirements": rec.input_state_requirements}
                for ph, rec in records.items()
                if rec.relevant and rec.execution_status.startswith(("blocked", "execution_failed"))]
    for f in failures:
        records[f["phase"]].support_implication = "lowers_support:integration_failure"
    manifest = {ph: {"available": rec.available, "selected": rec.relevant,
                     "executed": rec.execution_status == "causally_active",
                     "omitted": rec.execution_status != "causally_active",
                     "reason": rec.no_op_reason or rec.relevance_reasons,
                     "status": rec.execution_status, "n_state_deltas": rec.n_state_deltas,
                     "causally_irrelevant": rec.execution_status == "no_op_causally_irrelevant",
                     "version": rec.implementation_version,
                     "removal_changes_terminal": rec.ablation_influence}
                for ph, rec in records.items()}
    return {"records": records, "manifest": manifest, "integration_failures": failures,
            "fully_integrated": not failures}


def _terminal_influence(phase, plan, written) -> str:
    rev = next((e for e in plan.scheduled_events if e.get("etype") == "resolve_outcome"), None)
    outcome_var = (rev or {}).get("payload", {}).get("outcome_var", "outcome")
    consumed = set()
    for e in plan.scheduled_events:
        if e.get("etype") in ("institutional_decision", "aggregate_outcome_resolution", "hazard_round"):
            consumed |= {m.get("var") for m in (e.get("payload", {}).get("consume") or [])}
    for w in written:
        # event-time plans have no resolver: writing the absorbing state IS resolving the terminal
        if outcome_var in w or "absorbing_state_reached" in w or "absorbed_at" in w:
            return "direct_resolution"
        if any(cv and cv in w for cv in consumed):
            return "consumed_by_mechanism"
    if phase == "phase10_institutions":
        return "constraint"
    return "state_only"
