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


def result_from_run(question, plan, result, branches, *, intervention="", t0=None, calibrator=None,
                    cal_key="") -> SimulationResult:
    """Build the shipped SimulationResult from a completed (plan, terminal result, branches). Extracted so
    both simulate() and the validation harness construct the SAME contract from ONE compile + ONE rollout
    (no double LLM calls). Epistemic weakness lives in support_grade + limitations — never in a refusal."""
    from swm.world_model_v2.calibration import decompose_uncertainty
    dist = result.get("distribution") or {}
    quant = result.get("quantiles") or {}
    unresolved = result.get("unresolved_share", 0.0)
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
    return SimulationResult(
        question=question, simulation_status=status, support_grade=plan.support_grade,
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
        plan_hash=plan.plan_hash(),
        provenance={"compiler_version": plan.provenance.get("compiler_version"),
                    "prompt_hash": plan.provenance.get("prompt_hash"),
                    "readout_var": plan.outcome_contract.readout_var,
                    "readout_repaired": plan.provenance.get("readout_repaired"),
                    "n_deltas": result.get("n_deltas"), "n_particles": plan.compute_plan.get("n_particles")},
        latency_s=round(_time.time() - t0, 3) if t0 is not None else 0.0)


def simulate(question: str, *, llm, evidence="", as_of: str, horizon: str, intervention: str = "",
             n_particles=None, seed: int = 0, calibrator=None, cal_key: str = "") -> SimulationResult:
    t0 = _time.time()
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

    return result_from_run(question, plan, result, branches, intervention=intervention, t0=t0,
                           calibrator=calibrator, cal_key=cal_key)
