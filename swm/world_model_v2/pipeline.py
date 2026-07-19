"""The production simulation pipeline — question → SimulationResult (no-abstention).

`simulate()` is the one V2 production entry. It ALWAYS attempts a simulation for a coherent question and
returns a `SimulationResult` with a forecast whenever the simulation ran. Epistemic weakness lowers the
support grade and widens uncertainty; it never refuses. Only technical failures → execution_failed; only
genuinely incoherent questions → clarification_required (rare).
"""
from __future__ import annotations

import time as _time

from swm.world_model_v2.compiler import compile_world
from swm.world_model_v2.result import (ClarificationRequired, CompilerExecutionError, SimulationResult)


def _binary_projection(distribution: dict, options) -> float:
    """Project a terminal distribution onto P(first option / 'True'/'yes')."""
    if not distribution:
        return None
    yes_keys = [str(o) for o in (options or [])][:1] + ["True", "true", "yes", "Yes", "1", "reply"]
    for k in yes_keys:
        if k in distribution:
            return round(float(distribution[k]), 4)
    # fall back to the max-mass non-null key
    nonnull = {k: v for k, v in distribution.items() if k not in ("None", "no_choice")}
    return round(float(max(nonnull.values())), 4) if nonnull else None


def _recommendation_status(intervention: str, support_grade: str) -> str:
    if not intervention:
        return "not_requested"
    if support_grade == "empirically_supported":
        return "eligible"
    if support_grade == "transfer_supported":
        return "limited"
    return "withheld"                                          # exploratory/highly_speculative → withhold


def _operator_delta_census(branches) -> dict:
    """{operator: {n_deltas, fields_written[:8], event_types[:6]}} across all branches."""
    census = {}
    for b in (branches or []):
        for d in getattr(b, "log", []):
            c = census.setdefault(d.operator, {"n_deltas": 0, "fields_written": [], "event_types": []})
            c["n_deltas"] += 1
            for ch in (d.changes or [])[:2]:
                p = str(ch.get("path", ""))
                if p and p not in c["fields_written"] and len(c["fields_written"]) < 8:
                    c["fields_written"].append(p)
            if d.event_type not in c["event_types"] and len(c["event_types"]) < 6:
                c["event_types"].append(d.event_type)
    return census


def result_from_run(question, plan, result, branches, *, intervention="", t0=None, calibrator=None,
                    cal_key="") -> SimulationResult:
    """Build the shipped SimulationResult from a completed (plan, terminal result, branches). Extracted so
    both simulate() and the validation harness construct the SAME contract from ONE compile + ONE rollout
    (no double LLM calls). Epistemic weakness lives in support_grade + limitations — never in a refusal."""
    from swm.world_model_v2.calibration import decompose_uncertainty
    dist = result.get("distribution") or {}
    quant = result.get("quantiles") or {}
    unresolved = result.get("unresolved_share", 0.0)
    # §28: when the DEFAULT runtime refused a broad-prior terminal resolution (no validated
    # mechanism, no posterior), an unresolved readout is the HONEST epistemic state — the run
    # classifies under_modeled_nonhuman_mechanism, naming the missing mechanism. It is not an
    # engineering failure and not a completed forecast.
    _temporal_pre = result.get("temporal_runtime") or {}
    _suppressions = list(_temporal_pre.get("mechanism_suppressions") or [])
    if not dist and not quant and _suppressions:
        return SimulationResult(
            question=question, simulation_status="under_modeled",
            support_grade=plan.support_grade,
            under_modeled_subtypes=["under_modeled_nonhuman_mechanism"],
            under_modeled_components=[
                {"component": s.get("outcome_var", ""), "kind": "nonhuman_mechanism",
                 "why": f"missing validated mechanism: {s.get('mechanism', '')} — "
                        f"{str(s.get('why', ''))[:160]}",
                 "sensitivity": "decisive"} for s in _suppressions[:8]],
            limitations=["no validated mechanism resolved the terminal outcome; the default "
                         "runtime refuses the broad-prior terminal draw (§28) — the missing "
                         "mechanism is named in under_modeled_components"],
            plan_hash=plan.plan_hash(),
            provenance={"temporal_runtime": _temporal_pre or None,
                        "n_particles": plan.compute_plan.get("n_particles")},
            latency_s=round(_time.time() - t0, 3) if t0 is not None else 0.0)
    # INVARIANT: a completed simulation must carry a forecast. If the rollout produced NO binnable terminal
    # distribution AND no quantiles (every terminal world fell outside the declared option space), the
    # terminal readout is technically unbindable — an ENGINEERING failure (execution_failed), never a
    # completed-but-empty "forecast abstention". The fallback resolver normally prevents this.
    if not dist and not quant:
        return SimulationResult(
            question=question, simulation_status="execution_failed",
            failure_taxonomy="terminal_readout_unbindable", plan_hash=plan.plan_hash(),
            support_grade=plan.support_grade,
            limitations=[f"{unresolved:.0%} of terminal worlds fell outside the declared option space "
                         f"{plan.outcome_contract.options} — readout did not bin"],
            latency_s=round(_time.time() - t0, 3) if t0 is not None else 0.0)
    raw_p = _binary_projection(dist, plan.outcome_contract.options)
    struct_post = result.get("structural_posterior")
    unc = decompose_uncertainty(branches, structural_posterior=struct_post,
                                evidence_grade=plan.provenance.get("evidence_basis", ""))
    cal_p = None
    if calibrator is not None and raw_p is not None and hasattr(calibrator, "apply"):
        try:
            cal_p = round(calibrator.apply(raw_p, cal_key), 4) if cal_key else round(calibrator.apply(raw_p), 4)
        except Exception:
            cal_p = None
    limitations = list(plan.omissions and [f"omitted (negligible-sensitivity): {o.get('component')}"
                                           for o in plan.omissions[:5]] or [])
    if plan.degraded:
        limitations.append(f"degraded: support grade {plan.support_grade}; "
                           f"{len(plan.fallbacks_used)} fallback mechanism(s) used")
    if unresolved > 0.0:
        limitations.append(f"{unresolved:.0%} of terminal worlds outside the declared option space")
    high_sens_unknowns = [l.path for l in plan.latents if getattr(l, "sensitivity", 0.5) >= 0.6]
    status = "completed_with_degradation" if plan.degraded else "completed"
    # ---- §27 temporal-runtime block + §12 truncation surfacing ----
    temporal = result.get("temporal_runtime") or {}
    support_grade = plan.support_grade
    truncation_report = {}
    if temporal.get("temporally_truncated"):
        share = temporal.get("truncated_branch_share", 0.0)
        trunc = temporal.get("truncation") or {}
        branch_status = str(trunc.get("branch_status", ""))
        # §20/§8: actor/provider/cognition-driven halts carry the first-class `truncated`
        # status + a truncation report; the legacy pure event-safety budget keeps its
        # historical `temporally_truncated` label (readable alias)
        if branch_status and branch_status != "truncated_event_budget":
            status = "truncated"
        else:
            status = "temporally_truncated"
        from swm.world_model_v2.truncation import honest_note
        truncation_report = {
            "branch_status": branch_status or "truncated_event_budget",
            "reason": trunc.get("reason", ""),
            "truncated_branch_share": share,
            "n_branches_truncated": temporal.get("n_branches_truncated"),
            "earliest_truncation_ts": trunc.get("at_ts"),
            "actors_affected": trunc.get("actors_not_processed", []),
            "pending_events": (trunc.get("pending_events") or [])[:12],
            "unresolved_decision_trigger": trunc.get("unresolved_decision_trigger", {}),
            "branches": (trunc.get("branches") or [])[:20],
            "note": honest_note()}
        limitations.append(
            f"truncated: {share:.0%} of branches stopped before causal quiescence "
            f"({trunc.get('reason', 'safety budget')}) — pending events/actors recorded; "
            f"consequential recommendations are withheld at this support level")
        if support_grade in ("empirically_supported", "transfer_supported"):
            support_grade = "exploratory"
    tmodel = getattr(plan, "temporal_model", None)
    if tmodel is not None:
        temporal = {**temporal,
                    "temporal_model_id": tmodel.scenario_id,
                    "temporal_model_hash": tmodel.temporal_model_hash(),
                    "as_of": tmodel.as_of, "horizon_ts": tmodel.horizon_ts,
                    "timezone_assumptions": dict(tmodel.timezones),
                    "calendar_assumptions": dict(tmodel.calendars),
                    "known_scheduled_facts": list(tmodel.scheduled_facts)[:12],
                    "generated_channels": sorted(tmodel.channels),
                    "generated_actor_profiles": sorted(tmodel.actor_profiles),
                    "institutional_stage_processes": [p.process_id for p in
                                                      tmodel.institutional_processes],
                    "continuous_processes": [p.process_id for p in
                                             tmodel.continuous_processes],
                    "temporal_latent_hypotheses": list(tmodel.correlated_latents)[:8],
                    "temporal_uncertainties": list(tmodel.temporal_uncertainties)[:12],
                    "unresolved_timing_mechanisms": list(tmodel.unresolved_mechanisms)[:12],
                    "timing_support_classification": tmodel.support_classification,
                    "temporal_compilation_llm_calls": len(tmodel.compilation_trace),
                    "critic_findings": list(tmodel.critic_findings)[:12],
                    "degraded": tmodel.degraded or None}
    return SimulationResult(
        question=question, simulation_status=status, support_grade=support_grade,
        recommendation_status=_recommendation_status(intervention, plan.support_grade),
        raw_distribution={k: round(v, 4) for k, v in dist.items()} if dist else result.get("quantiles", {}),
        calibrated_distribution=({str(k): cal_p} if cal_p is not None else None),
        raw_probability=raw_p, calibrated_probability=cal_p,
        uncertainty_decomposition=unc, structural_disagreement=struct_post,
        mechanism_disagreement={"choices": plan.mechanism_choices,
                                "n_fallback": len(plan.fallbacks_used)},
        evidence_quality=plan.provenance.get("evidence_basis", ""),
        limitations=limitations, fallbacks_used=plan.fallbacks_used,
        mechanism_tiers={c["process"]: c["tier"] for c in plan.mechanism_choices},
        omitted_high_sensitivity_variables=high_sens_unknowns,
        sensitivity_contributors=[c["process"] for c in plan.mechanism_choices if c["tier"] >= 5][:5],
        interpretation_hypotheses=plan.interpretations,
        truncation_report=truncation_report,
        plan_hash=plan.plan_hash(),
        provenance={"compiler_version": plan.provenance.get("compiler_version"),
                    "prompt_hash": plan.provenance.get("prompt_hash"),
                    "readout_var": plan.outcome_contract.readout_var,
                    "readout_repaired": plan.provenance.get("readout_repaired"),
                    "n_deltas": result.get("n_deltas"), "n_particles": plan.compute_plan.get("n_particles"),
                    # event-time contracts: the full first-passage readout (CDF/survival/quantiles/mode×time)
                    "event_time": result.get("event_time"),
                    # §27: the temporal-runtime block — model hash/assumptions, event counts by
                    # type, invocations by actor and trigger, stage delays, batches, conflicts,
                    # cancellations, pending-at-horizon, truncations, unresolved mechanisms
                    "temporal_runtime": temporal or None,
                    # phase-supervision inputs: exactly which operators produced StateDeltas, and which
                    # state paths they wrote — the PhaseExecutionRecord is derived from THIS census, so
                    # activation accounting cannot drift from what actually executed.
                    "operator_delta_census": _operator_delta_census(branches)},
        latency_s=round(_time.time() - t0, 3) if t0 is not None else 0.0)


def simulate(question: str, *, llm, evidence="", as_of: str, horizon: str, intervention: str = "",
             n_particles=None, seed: int = 0, calibrator=None, cal_key: str = "",
             persistence=None, actor_history=None, persistence_families=None,
             persistence_provider=None) -> SimulationResult:
    """LEGACY SINGLE-STRUCTURAL-MODEL compatibility helper — NOT the public production entry. The
    canonical default is `unified_runtime.simulate_world`, which runs the structural-model ENSEMBLE
    (several independently generated causal models, each fully simulated); this helper compiles exactly
    one plan and is retained for phase-scoped science and frozen-artifact compatibility only, equivalent
    in status to the explicit `single_structural_model` ablation.

    When `actor_history` names a durable actor identity, persistence is
    part of THIS path AUTOMATICALLY — no caller needs to hand-build a PersistenceContext: an explicit
    `persistence=` wins; otherwise a `PersistenceContextProvider` (injected or built from environment config)
    resolves identity, constructs/reuses the transactional store, loads prior history/checkpoints, and returns
    a ready context. Prior state is loaded, sequential posteriors updated, persistent state materialized into
    the standard WorldState, causally-relevant families execute by default (only quarantined/incompatible are
    blocked), and lineage/support effects are returned in the result. Anonymous/stateless requests (no
    actor_history) and any storage/identity failure DEGRADE honestly to the ordinary non-persistent path —
    never an abstention, never a crash. Without persistence work, behaviour is byte-identical (no regression)."""
    t0 = _time.time()
    # ---- automatic persistence-context construction (final usability gap): if the caller did not pass an
    #      explicit context but a durable actor identity is available, request one from the provider. ----
    auto_persistence_meta = None
    if persistence is None and actor_history:
        try:
            from swm.world_model_v2.phase8_provider import build_provider
            provider = persistence_provider or build_provider()
            persistence, auto_persistence_meta = provider.for_request(
                question, as_of, actor_tokens=list(actor_history))
        except Exception as e:  # noqa: BLE001 — provider failure must degrade, not crash
            auto_persistence_meta = {"degraded": f"provider_unavailable: {type(e).__name__}: {e}"[:160]}
            persistence = None
    # ---- compile (never epistemically abstains) ----
    try:
        plan = compile_world(question, llm=llm, evidence=evidence, as_of=as_of, horizon=horizon,
                             intervention=intervention, seed=seed)
    except ClarificationRequired as e:
        return SimulationResult(question=question, simulation_status="clarification_required",
                                clarification_reason=str(e),
                                interpretation_hypotheses=e.interpretations_tried,
                                recommendation_status=_recommendation_status(intervention, "highly_speculative"),
                                latency_s=round(_time.time() - t0, 3))
    except CompilerExecutionError as e:
        return SimulationResult(question=question, simulation_status="execution_failed",
                                failure_taxonomy=e.taxonomy, limitations=[f"compiler: {e}"],
                                latency_s=round(_time.time() - t0, 3))

    # ---- CANONICAL PERSISTENCE PATH: when history/checkpoint is available, persistence is part of the
    #      ordinary simulate() flow (no separate entry point). Delegates to the shared run_with_persistence.
    if persistence is not None and (actor_history or getattr(persistence, "store", None) is not None):
        try:
            from swm.world_model_v2.phase8_pipeline import run_with_persistence
            res, _artifacts = run_with_persistence(
                question, plan, llm=llm, context=persistence, actor_history=actor_history,
                family_ids=persistence_families, intervention=intervention, t0=t0,
                n_particles=n_particles, seed=seed, calibrator=calibrator, cal_key=cal_key)
            return res
        except CompilerExecutionError as e:
            return SimulationResult(question=question, simulation_status="execution_failed",
                                    failure_taxonomy=e.taxonomy, plan_hash=plan.plan_hash(),
                                    support_grade=plan.support_grade, limitations=[f"persistence: {e}"],
                                    latency_s=round(_time.time() - t0, 3))
        except Exception as e:  # noqa: BLE001 — persistence must never crash the canonical path
            return SimulationResult(question=question, simulation_status="execution_failed",
                                    failure_taxonomy="runtime_exception", plan_hash=plan.plan_hash(),
                                    limitations=[f"persistence runtime: {type(e).__name__}: {e}"],
                                    latency_s=round(_time.time() - t0, 3))

    # ---- run (technical failure → execution_failed, NOT abstention) ----
    from swm.world_model_v2.materialize import run_from_plan
    try:
        result, branches = run_from_plan(plan, llm=llm, n_particles=n_particles, seed=seed)
    except CompilerExecutionError as e:
        return SimulationResult(question=question, simulation_status="execution_failed",
                                failure_taxonomy=e.taxonomy, plan_hash=plan.plan_hash(),
                                support_grade=plan.support_grade, limitations=[f"execution: {e}"],
                                latency_s=round(_time.time() - t0, 3))
    except Exception as e:
        return SimulationResult(question=question, simulation_status="execution_failed",
                                failure_taxonomy="runtime_exception", plan_hash=plan.plan_hash(),
                                limitations=[f"runtime: {type(e).__name__}: {e}"],
                                latency_s=round(_time.time() - t0, 3))

    res = result_from_run(question, plan, result, branches, intervention=intervention, t0=t0,
                          calibrator=calibrator, cal_key=cal_key)
    # honest degradation note: persistence was requested but unavailable (anonymous / storage / identity).
    if auto_persistence_meta and auto_persistence_meta.get("degraded"):
        deg = auto_persistence_meta["degraded"]
        if deg != "anonymous_no_durable_identity":          # anonymous is expected, not a degradation
            res.limitations = list(res.limitations) + [f"persistence unavailable ({deg}); ran without "
                                                        "longitudinal state"]
        res.provenance = {**(res.provenance or {}), "phase8_provider": auto_persistence_meta}
    return res
