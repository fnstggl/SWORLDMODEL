"""Evidence-conditioned simulation entry — Phase 2.

The full production path:

  question → preliminary compile → typed evidence requirements → live multisource retrieval →
  temporal verification → claims → entities → dependence → contradictions → visibility → leakage →
  immutable bundle → evidence-conditioned recompile (plan diff) → evidence materialization
  (observation StateDeltas + actor views) → rollout → terminal readout → SimulationResult.

Returns the Phase-1 SimulationResult (no-abstention contract preserved) plus the evidence artifacts
(bundle, plan diff, pre/post plan hashes) so a reviewer can trace evidence → world change → outcome.
"""
from __future__ import annotations

import time as _time

from swm.world_model_v2.compiler import compile_world
from swm.world_model_v2.evidence_materialize import attach_evidence_observations
from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig, gather_evidence
from swm.world_model_v2.evidence_recompile import recompile_with_evidence
from swm.world_model_v2.evidence_requirements import requirements_from_plan
from swm.world_model_v2.pipeline import result_from_run
from swm.world_model_v2.result import ClarificationRequired, CompilerExecutionError, SimulationResult


def simulate_with_evidence(question: str, *, llm, as_of: str, horizon: str, intervention: str = "",
                           seed: int = 0, config: OrchestratorConfig | None = None,
                           user_documents: list | None = None, dataset_path: str = "",
                           prior_bundle_path: str = "", store=None) -> tuple:
    """Run the evidence-conditioned simulation. Returns (SimulationResult, artifacts) where artifacts carries
    the frozen evidence bundle, the plan diff, and pre/post plan hashes. Evidence weakness degrades the
    support grade; it never blocks a forecast (no-abstention contract preserved)."""
    t0 = _time.time()
    # --- 1. preliminary compile (no-abstention) ---
    try:
        plan = compile_world(question, llm=llm, evidence="", as_of=as_of, horizon=horizon,
                             intervention=intervention, seed=seed)
    except ClarificationRequired as e:
        return (SimulationResult(question=question, simulation_status="clarification_required",
                                 clarification_reason=str(e), latency_s=round(_time.time() - t0, 3)),
                {"stage": "compile"})
    except CompilerExecutionError as e:
        return (SimulationResult(question=question, simulation_status="execution_failed",
                                 failure_taxonomy=e.taxonomy, latency_s=round(_time.time() - t0, 3)),
                {"stage": "compile"})

    # --- 2. emit typed evidence requirements ---
    reqs = requirements_from_plan(plan, as_of_iso=_iso(as_of), question=question)

    # --- 3. gather evidence (live retrieval → immutable bundle) ---
    bundle = gather_evidence(question, as_of=as_of, requirements=reqs, llm=llm,
                             user_documents=user_documents, dataset_path=dataset_path,
                             prior_bundle_path=prior_bundle_path, config=config,
                             plan_hash=plan.plan_hash(), seed=seed, store=store)

    # --- 4. evidence-conditioned recompile (plan diff) ---
    revised, diff = recompile_with_evidence(plan, bundle, llm=llm, horizon=horizon)
    revised = attach_evidence_observations(revised, bundle)

    # --- 5. rollout the evidence-conditioned world ---
    from swm.world_model_v2.materialize import run_from_plan
    try:
        result, branches = run_from_plan(revised, llm=llm, seed=seed)
    except CompilerExecutionError as e:
        return (SimulationResult(question=question, simulation_status="execution_failed",
                                 failure_taxonomy=e.taxonomy, plan_hash=revised.plan_hash(),
                                 latency_s=round(_time.time() - t0, 3)),
                {"bundle": bundle, "plan_diff": diff.as_dict(), "stage": "rollout"})
    except Exception as e:  # noqa: BLE001
        return (SimulationResult(question=question, simulation_status="execution_failed",
                                 failure_taxonomy="runtime_exception", limitations=[str(e)[:120]],
                                 latency_s=round(_time.time() - t0, 3)),
                {"bundle": bundle, "plan_diff": diff.as_dict(), "stage": "rollout"})

    res = result_from_run(question, revised, result, branches, intervention=intervention, t0=t0)
    res.provenance["evidence_bundle_hash"] = bundle.bundle_hash()
    res.provenance["evidence_plan_diff_structural_changes"] = diff.n_structural_changes
    res.provenance["evidence_plan_diff_lean_only"] = diff.lean_only
    res.provenance["n_evidence_documents"] = len(bundle.documents)
    res.provenance["n_included_claims"] = len(bundle.included_claim_ids)
    res.evidence_quality = f"{len(bundle.included_claim_ids)} included claims from " \
                           f"{bundle.evidence_uncertainty.get('n_independent_sources', 0)} independent sources"
    if bundle.included_claim_ids:
        res.limitations.append(f"evidence-conditioned: {diff.n_structural_changes} structural plan changes")
    else:
        res.limitations.append("no admissible as-of evidence found — broad-prior forecast")
    return res, {"bundle": bundle, "plan_diff": diff, "pre_plan_hash": plan.plan_hash(),
                 "post_plan_hash": revised.plan_hash(), "n_requirements": len(reqs)}


def _iso(as_of: str) -> str:
    from swm.world_model_v2.state import parse_time
    return _time.strftime("%Y-%m-%d", _time.gmtime(parse_time(as_of)))
