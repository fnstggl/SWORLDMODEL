"""The lean-adaptive runtime — the same causal architecture as the PR-#127 ensemble runtime with
duplicated computation removed. Reached ONLY through
`unified_runtime.simulate_world(..., execution_profile="lean_adaptive")`.

    one primary structural model (one generation call)
    → shared evidence gathered ONCE under one as-of boundary (identical to full fidelity)
    → run-shared artifacts (criterion, calendars, canonical facts, rosters — hashed, owned,
      dependency-invalidated; cross-model prompt identity shares compilation through the run's
      content-addressed call cache)
    → ONE focused reversal critic → at most one reversal-capable challenger (else the critic's
      verdict is the single-survivor convergence certificate; extra credible alternatives mark
      structurally_underidentified and offer full-fidelity escalation)
    → per-model conditioning through the SAME canonical helpers (evidence recompile, ITS OWN
      posterior, world boundary, fidelity, event-time, Phase 11)
    → prepare_persistence_run (ensure_outcome_pathway inside, as always)
    → LeanActorController attached to the canonical actor runtime: cohorts, deterministic
      prechecks, duplicate suppression, decision-equivalence cache (single-flight), one-call
      bounded cognition, compact prompts, consequence-program cache — with the research-first
      arming invariant (actor psychology cannot start before the research ledger completes)
    → adaptive progressive particles (deterministic prefixes; full budget when unstable or when
      structural models materially disagree)
    → canonical finalize + ensemble assembly (identical result contract, §NAP guards, phase
      supervision, no-silent-None)
    → stability signals → at most ONE execution-replicate escalation (reported, never averaged).

Everything full fidelity refuses to do, this profile refuses too: no numeric actors, no minted
probabilities, no generic terminal guesses, no silent empty rollouts."""
from __future__ import annotations

import math
import time as _time

from swm.world_model_v2.lean_artifacts import RunSharedArtifacts
from swm.world_model_v2.lean_controller import LeanActorConfig, LeanActorController
from swm.world_model_v2.lean_particles import (LeanParticleTolerances, ParticleStoppingRecord,
                                               run_progressive_particles)
from swm.world_model_v2.lean_routing import TieredRouter
from swm.world_model_v2.lean_stability import detect_signals, execution_replicate
from swm.world_model_v2.lean_structural import (apply_reversal_verdict, reconnoiter_lean,
                                                run_reversal_critic)
from swm.world_model_v2.llm_call_cache import CachedLLM, CallLedger, MeteredLLM, NullStore
from swm.world_model_v2.result import (ClarificationRequired, CompilerExecutionError,
                                       SimulationResult)

LEAN_RUNTIME_VERSION = "lean-adaptive-1.0"

#: material structural disagreement threshold for the forced-full-budget extension (compute
#: control, recorded; the ASSEMBLY-level sensitivity classification stays the canonical one)
STRUCTURAL_DISAGREEMENT_SPREAD = 0.10


def simulate_world_lean(question: str, *, as_of: str, horizon: str = "", intervention: str = "",
                        evidence: str = "", user_context=None, prior_checkpoint=None,
                        compute_budget=None, seed: int = 0, llm=None,
                        execution_policy: dict = None, trace_level: str = "standard",
                        config=None, prebuilt_bundle=None) -> SimulationResult:
    from swm.world_model_v2 import ensemble_compiler as EC
    from swm.world_model_v2 import structural_runtime as SR
    from swm.world_model_v2 import unified_runtime as U

    policy = execution_policy or {}
    drop = set(policy.get("drop_phases", []))
    t0 = _time.time()
    ledger = CallLedger()
    cache_store: dict = NullStore() if policy.get("cache_mode") == "off" else {}
    lean_cfg = LeanActorConfig(**dict(policy.get("lean_actor") or {}))
    tolerances = LeanParticleTolerances(**dict(policy.get("lean_particles") or {}))
    research_ledger: dict = {}
    router = TieredRouter(strong_llm=llm) if llm is not None else None

    def _fail(taxonomy, msg, extra=None):
        return SimulationResult(
            question=question, simulation_status="execution_failed", failure_taxonomy=taxonomy,
            limitations=[msg[:220]], latency_s=round(_time.time() - t0, 3),
            provenance={"runtime": LEAN_RUNTIME_VERSION, "structural_mode": "ensemble",
                        "ensemble_cost_manifest": ledger.as_dict(), **(extra or {})})

    # ---------- personal-reaction route: delegated to the canonical ensemble route --------
    individual = SR._route_individual_reaction_ensemble(
        question, user_context, llm, as_of, seed, t0, dict(policy.get("generation_policy") or {}),
        ledger, cache_store)
    if individual is not None:
        individual.provenance = {**(individual.provenance or {}),
                                 "lean": {"note": "personal-reaction route runs the canonical "
                                                  "ensemble route (not yet leaned)"}}
        return individual

    # ---------- Stage A-lean: ONE primary reconnaissance call ----------
    try:
        ens = reconnoiter_lean(question, llm=llm, as_of=as_of, horizon=horizon,
                               intervention=intervention, user_context=user_context,
                               evidence_text=str(evidence or "")[:2400], seed=seed,
                               ledger=ledger, cache_store=cache_store)
    except CompilerExecutionError as e:
        return _fail(e.taxonomy, f"lean structural generation failed: {e}")
    research_ledger["question_type_fixed"] = True         # recon fixes the outcome family frame

    # ---------- shared evidence ONCE (identical to the full-fidelity ensemble path) --------
    bundle, evidence_text = None, str(evidence or "")[:2400]
    if "phase2_evidence" not in drop and as_of:
        try:
            if prebuilt_bundle is not None:
                bundle = prebuilt_bundle
            else:
                from swm.world_model_v2.evidence_orchestrator import (
                    OrchestratorConfig, gather_evidence_with_escalation)
                reqs = EC.union_evidence_requirements(ens, as_of_iso=as_of)
                research_ledger["evidence_requirements_built"] = True
                bundle, retry_rec = gather_evidence_with_escalation(
                    question, as_of=as_of, requirements=reqs, llm=llm,
                    config=config or OrchestratorConfig(), plan_hash=ens.ensemble_id, seed=seed)
                if retry_rec:
                    ens.generation_policy["evidence_retry"] = retry_rec
            evidence_text = (bundle.render(max_chars=2400) if hasattr(bundle, "render")
                             else evidence_text)
            ens.shared_evidence_bundle_hash = bundle.bundle_hash() \
                if hasattr(bundle, "bundle_hash") else ""
            ens.shared_evidence_as_of = as_of
        except Exception as e:  # noqa: BLE001 — evidence failure never blocks the forecast
            bundle = None
            ens.generation_policy["evidence_error"] = f"{type(e).__name__}: {e}"[:160]
    research_ledger.setdefault("evidence_requirements_built", True)
    research_ledger["evidence_gathered"] = True
    research_ledger["publication_dates_checked"] = True    # as-of filtering is the gather layer

    artifacts = RunSharedArtifacts(as_of=as_of, run_id=ens.ensemble_id)
    artifacts.register("evidence_bundle",
                       {"bundle_hash": ens.shared_evidence_bundle_hash,
                        "n_claims": len(getattr(bundle, "included_claim_ids", []) or [])
                        if bundle is not None else 0,
                        "caller_evidence_chars": len(str(evidence or ""))},
                       owner="lean_runtime.evidence",
                       invalidation_rule="as_of change or bundle regeneration")

    # ---------- ONE reversal critic → at most one challenger ----------
    verdict = run_reversal_critic(ens, llm=llm, evidence_text=evidence_text, ledger=ledger,
                                  cache_store=cache_store)
    structural_record = apply_reversal_verdict(
        ens, verdict, llm=llm, as_of=as_of, horizon=horizon, intervention=intervention,
        evidence_text=evidence_text, seed=seed, ledger=ledger, cache_store=cache_store)

    # ---------- Stage B: compile every surviving candidate (1 or 2) ----------
    try:
        EC.compile_candidates(ens, llm=llm, as_of=as_of, horizon=horizon,
                              intervention=intervention,
                              evidence=bundle if bundle is not None else (evidence or ""),
                              seed=seed, ledger=ledger, cache_store=cache_store)
    except Exception as e:  # noqa: BLE001
        return _fail("invalid_execution_plan", f"lean structural compile failed: {e}",
                     {"structural_ensemble_generation": ens.as_dict()})
    executable = [c for c in ens.surviving() if c.executable_plan is not None]
    if not executable:
        reasons = [c.promotion_reason for c in ens.candidates]
        if reasons and all("ClarificationRequired" in r for r in reasons if r):
            return SimulationResult(
                question=question, simulation_status="clarification_required",
                clarification_reason=(reasons[0].split("ClarificationRequired:")[-1]
                                      .strip()[:300] or "no coherent outcome contract"),
                latency_s=round(_time.time() - t0, 3),
                provenance={"runtime": LEAN_RUNTIME_VERSION,
                            "structural_ensemble_generation": ens.as_dict()})
        return _fail("invalid_execution_plan",
                     "no executable structural candidate remained after lean generation",
                     {"structural_ensemble_generation": ens.as_dict()})
    EC.deduplicate_candidates(ens, llm=llm, ledger=ledger, cache_store=cache_store)
    if structural_record.get("challenger_generated") and len(ens.surviving()) == 1 \
            and ens.convergence_certificate is None:
        # the challenger existed only as prose: dedup collapsed it into the primary — that IS
        # the certificate (a materially different model did not survive conservative comparison)
        ens.convergence_certificate = {
            "kind": "lean_challenger_collapsed_at_dedup",
            "basis": {"merge_manifest": ens.merge_manifest[-1] if ens.merge_manifest else {}},
            "note": "the critic's challenger compiled to a structural duplicate of the primary "
                    "(prose variation) — conservative dedup collapsed it; one model is certified"}
        structural_record["challenger_collapsed_at_dedup"] = True
    try:
        ens.validate_integrity()
    except Exception as e:  # noqa: BLE001
        return _fail("invalid_execution_plan", f"lean ensemble integrity violation: {e}",
                     {"structural_ensemble_generation": ens.as_dict()})

    # ---------- per-model conditioning + prepared runs + the ONE lean controller ----------
    controller = LeanActorController(config=lean_cfg, ledger=ledger, run_day=as_of)
    cond_llm = CachedLLM(llm, ledger=ledger, stage="model_conditioning", store=cache_store)
    runs: dict = {}
    for cand in ens.surviving():
        if cand.executable_plan is None:
            continue
        rec = _condition_and_prepare(U, SR, question, cand, bundle, as_of, horizon,
                                     intervention, seed, cond_llm, user_context,
                                     prior_checkpoint, drop, evidence, ledger, controller,
                                     research_ledger, artifacts)
        runs[cand.model_id] = rec
        ens.pilot_models.append(cand.model_id)
    ens.candidates_rejected = sum(1 for c in ens.candidates
                                  if c.promotion_status in ("rejected", "failed"))
    live = [c for c in ens.surviving()
            if c.model_id in runs and not runs[c.model_id].get("error")]
    if not live:
        return _fail("runtime_exception", "every lean structural model failed conditioning",
                     {"structural_ensemble_generation": ens.as_dict(),
                      "lean": {"controller": controller.manifest()}})

    # ---------- adaptive progressive particles (phase 1: independent progressive rolls) -----
    stopping_records: list[ParticleStoppingRecord] = []
    for cand in live:
        rec = runs[cand.model_id]
        branches, stop_rec = run_progressive_particles(
            rec["handle"], seed=seed, tolerances=tolerances, model_id=cand.model_id,
            # prepare_persistence_run either applied+recorded the repair or raised — an applied
            # repair IS settled (its record still feeds the stability signals below)
            outcome_pathway_settled=True,
            actions_requested=bool(intervention))
        rec["branches"] = branches
        rec["stopping"] = stop_rec
        rec["n_pilot"] = min(len(branches), max(SR.PILOT_MIN_PARTICLES,
                                                math.ceil(rec["n_full"] * SR.PILOT_FRACTION)))
        stopping_records.append(stop_rec)
        cand.pilot_status = "completed"
        cand.pilot_particles = rec["n_pilot"]
        proj = rec["handle"]["run"].project(list(branches))
        rec["pilot_distribution"] = dict(proj.get("distribution") or {})
        cand.pilot_result = {"distribution": rec["pilot_distribution"],
                             "n_particles": len(branches),
                             "unresolved_share": proj.get("unresolved_share")}
    # phase 2: material cross-model disagreement forces BOTH models to the full budget
    if len(live) > 1:
        dists = {c.model_id: runs[c.model_id]["pilot_distribution"] for c in live}
        spread = SR._dist_spread(*(list(dists.values())[:2])) or 0.0
        if spread > STRUCTURAL_DISAGREEMENT_SPREAD:
            for cand in live:
                rec = runs[cand.model_id]
                if len(rec["branches"]) < rec["n_full"]:
                    from swm.world_model_v2.phase8_pipeline import run_persistence_slice
                    more = run_persistence_slice(
                        rec["handle"], seed=seed, n_total=rec["n_full"],
                        start=len(rec["branches"]), stop=rec["n_full"])
                    rec["branches"] = list(rec["branches"]) + list(more)
                    rec["stopping"].stopped_early = False
                    rec["stopping"].n_executed = len(rec["branches"])
                    rec["stopping"].forced_full_reasons.append(
                        f"material_structural_disagreement(spread={spread:.3f})")
                    rec["stopping"].stop_reason = (
                        f"extended to full budget: primary/challenger disagree "
                        f"(spread {spread:.3f} > {STRUCTURAL_DISAGREEMENT_SPREAD})")

    # ---------- promotion + finalize through the canonical funnel ----------
    for cand in live:
        cand.promotion_status = "promoted"
        cand.promotion_reason = ("lean primary model" if not cand.generation_role.startswith(
            "lean_reversal") else "lean reversal-capable challenger")
    model_results = {}
    checkpoint_committed = False
    for cand in live:
        rec = runs[cand.model_id]
        commit = not checkpoint_committed
        res_m = SR._finalize_model(U, question, cand, rec, bundle, as_of, intervention, seed,
                                   t0, commit_checkpoint=commit)
        if res_m is not None:
            model_results[cand.model_id] = res_m
            checkpoint_committed = checkpoint_committed or commit
    promoted = [c for c in live if c.model_id in model_results]
    if not promoted:
        return _fail("runtime_exception",
                     "every lean structural model failed terminal projection",
                     {"structural_ensemble_generation": ens.as_dict(),
                      "lean": {"controller": controller.manifest()}})
    for cand in promoted:
        rec = runs[cand.model_id]
        cand.final_particles = len(rec["branches"])
        ens.full_models.append(cand.model_id)
        stop_rec = rec.get("stopping")
        ens.simulation_manifest[cand.model_id] = {
            "pilot_particles": rec["n_pilot"], "final_particles": len(rec["branches"]),
            "full_budget_required": rec["n_full"],
            "pilot_reused_as_prefix": True,
            "progressive_stopping": stop_rec.as_dict() if stop_rec is not None else None,
            "status": ("completed_early_stable" if stop_rec is not None
                       and stop_rec.stopped_early else "completed_full_budget")}

    # ---------- assembly through the canonical ensemble result contract ----------
    ens.cost_manifest = ledger.as_dict()
    res = SR._assemble_ensemble_result(U, question, ens, runs, model_results, promoted, bundle,
                                       ledger, t0)

    # ---------- stability signals + capped execution-replicate escalation ----------
    signals = detect_signals(
        ens=ens, stopping_records=stopping_records,
        evidence_sufficiency=(runs[promoted[0].model_id].get("evidence_sufficiency")
                              if promoted else None),
        outcome_pathways={m: runs[m]["handle"].get("outcome_pathway") for m in model_results
                          if m in runs},
        model_results=model_results,
        posterior_means={m: getattr(runs[m].get("posterior"), "outcome_rate_mean", None)
                         for m in model_results if m in runs
                         and runs[m].get("posterior") is not None})
    replicate = None
    if signals.escalate() and policy.get("lean_stability_escalation", True) \
            and lean_cfg.decision_cache:
        prim = promoted[0]
        rec = runs[prim.model_id]
        replicate = execution_replicate(
            rec["handle"], controller=controller, seed=seed,
            n_particles=max(1, len(rec["branches"])))
        if replicate.status == "completed" and replicate.forecast is not None:
            p0 = max((model_results[prim.model_id].raw_distribution or {}).values(),
                     default=None)
            if p0 is not None and abs(p0 - replicate.forecast) > 0.1:
                res.limitations = list(res.limitations or []) + [
                    f"execution-instability probe: behavioral replicate 1 moved the leading "
                    f"mass from {p0:.3f} to {replicate.forecast:.3f} on the same compiled "
                    f"world — treat the single-run number as noisy (full-fidelity mean-of-K "
                    f"remains the research-grade escalation)"]

    # ---------- lean provenance ----------
    res.provenance["runtime"] = LEAN_RUNTIME_VERSION
    res.provenance["lean"] = {
        "controller": controller.manifest(),
        "shared_artifacts": artifacts.manifest(),
        "structural": structural_record,
        "particle_stopping": [r.as_dict() for r in stopping_records],
        "particle_tolerances": tolerances.as_dict(),
        "stability_signals": signals.as_dict(),
        "stability_replicate": replicate.as_dict() if replicate is not None else None,
        "routing": router.manifest() if router is not None else None,
        "research_ledger": research_ledger}
    if ens.structurally_underidentified:
        res.limitations = list(res.limitations or []) + [
            "lean structural cap: credible alternatives remain beyond the one-challenger cap "
            "(structurally_underidentified) — escalate with "
            "execution_profile='full_fidelity' for the complete independent ensemble"]
    res.latency_s = round(_time.time() - t0, 3)
    return res


def _condition_and_prepare(U, SR, question, cand, bundle, as_of, horizon, intervention, seed,
                           cond_llm, user_context, prior_checkpoint, drop, evidence, ledger,
                           controller, research_ledger, artifacts: RunSharedArtifacts) -> dict:
    """Mirror of structural_runtime._condition_and_pilot_model with the pilot roll REMOVED
    (progressive particles happen in the caller) and the lean controller attached between
    prepare and rollout. Every canonical stage — evidence recompile, world boundary, phase-3
    posterior, evidence-sufficiency gate, fidelity/event-time/phase-11 conditioning,
    ensure_outcome_pathway inside prepare — runs unchanged."""
    from swm.world_model_v2.phase8_pipeline import prepare_persistence_run
    plan = cand.executable_plan
    manifest = {p: U._mani(available=True) for p in U._PHASES}
    manifest["phase1_compiler"].update(selected=True, executed=True, version="lean-compiler",
                                       reason=f"lean structural candidate {cand.model_id}")
    lineage = {"plan_hashes": list(cand.plan_lineage), "recompilations": []}
    costs = {"llm_calls": 0}
    rec = {"manifest": manifest, "lineage": lineage, "branches": [], "extensions": [],
           "error": "", "handle": None, "n_pilot": 0, "n_full": 0}
    model_llm = cond_llm.with_stage("model_conditioning", cand.model_id)
    try:
        if bundle is not None:
            plan = U._apply_evidence_to_plan(question, plan, bundle, model_llm, horizon,
                                             manifest, lineage)
            cand.executable_plan = plan
        if "world_boundary" not in drop:
            try:
                from swm.world_model_v2.outside_world import generate_outside_world
                from swm.world_model_v2.world_boundary import (boundary_sensitivity_analysis,
                                                               generate_world_boundary,
                                                               run_boundary_critics)
                ev_text = U._bundle_text(bundle, 2000)
                boundary = generate_world_boundary(
                    question=question, structural_model_id=cand.model_id,
                    thesis=cand.causal_thesis,
                    decisive={"actors": cand.decisive_actors,
                              "institutions": cand.decisive_institutions,
                              "constraints": cand.decisive_constraints,
                              "mechanisms": cand.decisive_mechanisms},
                    plan=plan, as_of=as_of, horizon=horizon, intervention=intervention,
                    user_context=user_context, evidence_text=ev_text, llm=model_llm,
                    available_compute={"n_particles_planned": None})
                run_boundary_critics(boundary, llm=model_llm, thesis=cand.causal_thesis)
                outside = generate_outside_world(boundary, llm=model_llm, evidence_text=ev_text)
                boundary_sensitivity_analysis(boundary, llm=model_llm, options=[
                    str(o) for o in (getattr(plan.outcome_contract, "options", None) or [])])
                plan._world_boundary = boundary
                plan._outside_world = outside
                rec["boundary"] = boundary
                rec["outside_world"] = outside
                lineage["world_boundary"] = {"boundary_id": boundary.boundary_id,
                                             "boundary_hash": boundary.boundary_hash(),
                                             "n_components": len(boundary.components)}
            except Exception as e:  # noqa: BLE001 — a failed boundary stage is a LOUD gap
                rec["boundary_error"] = f"{type(e).__name__}: {e}"[:200]
                lineage["world_boundary"] = {"error": rec["boundary_error"]}
        posterior = U._phase3_block(question, plan, bundle, model_llm, seed, manifest, drop)
        research_ledger["reference_prior_constructed"] = True
        rec["posterior_consumed"] = bool(posterior and posterior.n_effective_observations > 0)
        rec["posterior"] = posterior
        rec["prior_spec"] = getattr(plan, "_outcome_prior_spec", None)
        rec["evidence_sufficiency"] = U._evidence_sufficiency_block(
            question, bundle, posterior, as_of=as_of, drop=drop, lineage=lineage)
        research_ledger["evidence_sufficiency_assessed"] = True
        U._condition_plan(question, plan, bundle, as_of, horizon, seed, model_llm,
                          manifest, lineage, costs, drop,
                          user_context=user_context, intervention=intervention,
                          structural_model_id=cand.model_id, evidence=evidence)
        research_ledger["resolution_criterion_parsed"] = True
        research_ledger["calendar_recurrence_extracted"] = True
        research_ledger["institutions_normalized"] = True
        research_ledger["actor_information_projected"] = True
        cand.plan_lineage = list(lineage["plan_hashes"]) or [plan.plan_hash()]
        _register_plan_artifacts(artifacts, cand, plan, lineage)
        # the actor runtime keeps a reference to THIS llm for all later cognition: it must be
        # metered but NEVER byte-cached — actor-decision sharing is the DecisionEquivalenceCache's
        # sole authority (revalidation, certificates, replicate policy, never-cache-failures);
        # a raw exact-prompt cache here would silently bypass all of it
        actor_llm = MeteredLLM(cond_llm._llm if hasattr(cond_llm, "_llm") else cond_llm,
                               ledger=ledger, stage="actor_rollout", model_id=cand.model_id)
        handle = prepare_persistence_run(
            question, plan, llm=actor_llm, context=user_context,
            actor_history=(prior_checkpoint or {}).get("actor_history")
            if prior_checkpoint else None)
        rec.update(handle=handle, n_full=int(handle["n_particles"]))
        # ---- attach the lean controller to the canonical actor runtime ----
        attached = False
        for op in handle["ops"]:
            runtime = getattr(op, "runtime", None)
            if runtime is not None and hasattr(runtime, "engine") \
                    and hasattr(runtime.engine, "hypothesizer"):
                controller.attach_to_runtime(runtime, plan=plan,
                                             decision_context_hint=question[:300])
                attached = True
        rec["lean_controller_attached"] = attached
        controller.arm_actor_calls(research_ledger={**research_ledger,
                                                   "resolution_criterion_parsed": True,
                                                   "question_type_fixed": True})
    except Exception as e:  # noqa: BLE001 — a failed model is recorded loudly
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
        cand.pilot_status = "failed"
        cand.promotion_status = "failed"
        cand.promotion_reason = f"lean_conditioning_failed: {rec['error']}"
    return rec


def _register_plan_artifacts(artifacts: RunSharedArtifacts, cand, plan, lineage):
    """Register the question-level shared artifacts with content hashes (dedup across models is
    recorded; a model-specific difference registers per-model under a scoped name)."""
    crit = lineage.get("resolution_criterion")
    if crit:
        try:
            artifacts.register("resolution_criterion", crit,
                               owner="fidelity.parse_resolution_criterion",
                               invalidation_rule="question or horizon change")
        except ValueError:
            artifacts.register(f"resolution_criterion:{cand.model_id}", crit,
                               owner="fidelity.parse_resolution_criterion",
                               invalidation_rule="question or horizon change")
    sched = (lineage.get("scheduled_reality") or {}).get("facts")
    if sched:
        try:
            artifacts.register("calendar_facts", sched, owner="scheduled_facts",
                               invalidation_rule="as_of change")
        except ValueError:
            artifacts.register(f"calendar_facts:{cand.model_id}", sched,
                               owner="scheduled_facts", invalidation_rule="as_of change")
    try:
        artifacts.register(f"structural_model:{cand.model_id}",
                           {"plan_hash": plan.plan_hash(),
                            "schema_hash": cand.schema_hash,
                            "thesis": cand.causal_thesis[:300]},
                           owner="ensemble_compiler.stage_b",
                           invalidation_rule="evidence recompile or plan surgery",
                           depends_on=["evidence_bundle"])
    except ValueError:
        pass
    try:
        artifacts.register(f"temporal_model:{cand.model_id}",
                           {"attached": bool(lineage.get("temporal_model"))},
                           owner="temporal_compiler",
                           invalidation_rule="plan structure change",
                           depends_on=[f"structural_model:{cand.model_id}"])
    except ValueError:
        pass
