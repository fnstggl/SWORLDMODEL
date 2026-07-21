"""The structural-ensemble runtime — the DEFAULT execution path for every World Model V2 simulation.

    question, context, intervention, as-of evidence
    → independent generation of several materially different causal models   (ensemble_compiler Stage A)
    → adversarial search for missing actors/institutions/constraints/mechanisms      (omission critics)
    → shared evidence gathered ONCE under one as-of boundary (union of recon requirements)
    → separate evidence-conditioned executable compilation per candidate     (ensemble_compiler Stage B)
    → conservative deduplication (deterministic first, blind judge second)
    → per-model conditioning: evidence recompile, ITS OWN posterior, fidelity, event-time, Phase 11
    → REAL pilot simulation of every plausible distinct model through the canonical funnel
    → uncertainty-aware conservative promotion
    → full per-model budgets — every promoted model gets AT LEAST the single-model production particle
      count, with its pilot particles reused as a deterministic prefix (never divided across models)
    → per-model trajectory distributions, honest aggregation, structural sensitivity, reversal
      conditions, structural value-of-information.

BUDGET INVARIANT (non-negotiable): if single-model production would give a plan N particles, every
promoted model receives ≥ N total particles. `three models × N/3` is prohibited and enforced by
EnsembleIntegrityError + tests. Cost is saved by sharing immutable evidence, content-addressed caching of
byte-identical LLM calls, common-random-number particle alignment across models, and pilot-prefix reuse —
never by weaker models, shallower execution, shorter horizons or divided budgets.

MODEL WEIGHTS: no LLM-minted probabilities anywhere. The default aggregate is an explicitly labeled
equal-weight compatibility mixture plus per-model distributions, robust ranges and a minimax view;
qualitative support classes come from evidence-fit critics and are never converted to numbers.
"""
from __future__ import annotations

import json
import math
import time as _time

from swm.world_model_v2.result import SimulationResult, ClarificationRequired, CompilerExecutionError
from swm.world_model_v2.structural_contracts import (
    EnsembleIntegrityError, classify_forecast_sensitivity, decompose_uncertainty)
from swm.world_model_v2.llm_call_cache import CachedLLM, CallLedger, ScopedActorCache

#: Pilot allocation (Section 10): a REAL fraction of the model's own full budget through the full
#: canonical runtime — never a screening simulator. The fraction (20%) sits at the top of the 10–20%
#: guidance because production budgets are small (N ∈ [12, 80] on the standard path, ≥200 on event-time
#: paths); the absolute floor of 8 particles is the smallest count that resolves a binary distribution at
#: 0.125 granularity while staying below the production minimum FULL budget of 12 (compiler._fidelity_plan
#: clamps at 12) — so a pilot is always cheaper than the smallest full run but never degenerate.
PILOT_FRACTION = 0.20
PILOT_MIN_PARTICLES = 8
#: When structural disagreement is material after full runs, promoted models receive additional particles
#: (Section 0.9 / 13) — 25% of the full budget, on top of (never instead of) the full budget.
DISAGREEMENT_EXTENSION_FRACTION = 0.25
#: Pilot indistinguishability (Section 12, the ONLY pilot-based non-promotion ground): distributions must
#: be this close AND the structures similar AND the support class weaker AND the pilot large enough that
#: the difference is not noise. A noisy pilot always promotes.
PILOT_INDISTINGUISHABLE_SPREAD = 0.02
PILOT_MIN_N_FOR_EXCLUSION = 12


def simulate_structural_ensemble(question: str, *, as_of: str, horizon: str = "", intervention: str = "",
                                 evidence: str = "",
                                 user_context=None, prior_checkpoint=None, compute_budget=None,
                                 seed: int = 0, llm=None, execution_policy: dict = None,
                                 trace_level: str = "standard", config=None,
                                 prebuilt_bundle=None) -> SimulationResult:
    """The default World Model V2 run: structural-model uncertainty simulated end-to-end."""
    from swm.world_model_v2 import unified_runtime as U
    from swm.world_model_v2 import ensemble_compiler as EC

    policy = execution_policy or {}
    drop = set(policy.get("drop_phases", []))
    t0 = _time.time()
    ledger = CallLedger()
    # COST-BENCHMARK-ONLY knobs (Section 27 arms). Both can only cost MORE — never less accuracy:
    # cache_mode=off disables identical-call reuse; pilot_reuse=off discards pilot particles and reruns
    # the full budget from index 0 (the manifest records the waste honestly).
    from swm.world_model_v2.llm_call_cache import NullStore
    cache_store: dict = NullStore() if policy.get("cache_mode") == "off" else {}
    pilot_reuse = policy.get("pilot_reuse", "on") != "off"
    gen_policy = dict(policy.get("generation_policy") or {})
    if compute_budget in ("maximum_capacity", "max", "max_capacity"):
        gen_policy.setdefault("max_capacity", True)

    # ---------- personal-reaction route: SAME structural-ensemble principles at the personal scale ------
    individual = _route_individual_reaction_ensemble(question, user_context, llm, as_of, seed, t0,
                                                     gen_policy, ledger, cache_store)
    if individual is not None:
        return individual

    def _fail(taxonomy, msg, extra=None):
        return SimulationResult(
            question=question, simulation_status="execution_failed", failure_taxonomy=taxonomy,
            limitations=[msg[:220]], latency_s=round(_time.time() - t0, 3),
            provenance={"runtime": U.RUNTIME_VERSION, "structural_mode": "ensemble",
                        "ensemble_cost_manifest": ledger.as_dict(), **(extra or {})})

    # ---------- Stage A: independent reconnaissance (loud on missing backend / broken independence) -----
    try:
        ens = EC.reconnoiter_structures(question, llm=llm, as_of=as_of, horizon=horizon,
                                        intervention=intervention, user_context=user_context,
                                        seed=seed, generation_policy=gen_policy, ledger=ledger,
                                        cache_store=cache_store)
    except CompilerExecutionError as e:
        return _fail(e.taxonomy, f"structural ensemble generation failed: {e}")
    except EnsembleIntegrityError as e:
        return _fail("invalid_execution_plan", f"ensemble integrity violation: {e}")

    # ---------- shared evidence: union of recon requirements, gathered ONCE under one as-of boundary ----
    bundle, evidence_text = None, ""
    if "phase2_evidence" not in drop and as_of:
        try:
            if prebuilt_bundle is not None:
                bundle = prebuilt_bundle           # sealed-replay injection (frozen, time-locked, recorded)
            else:
                from swm.world_model_v2.evidence_orchestrator import (
                    OrchestratorConfig, gather_evidence_with_escalation)
                reqs = EC.union_evidence_requirements(ens, as_of_iso=as_of)
                # ONE evidence-retry authority shared with the single-model path
                bundle, retry_rec = gather_evidence_with_escalation(
                    question, as_of=as_of, requirements=reqs, llm=llm,
                    config=config or OrchestratorConfig(), plan_hash=ens.ensemble_id, seed=seed)
                if retry_rec:
                    ens.generation_policy["evidence_retry"] = retry_rec
            evidence_text = (bundle.render(max_chars=2400) if hasattr(bundle, "render")
                             else str(evidence or "")[:2400])
            ens.shared_evidence_bundle_hash = bundle.bundle_hash() if hasattr(bundle, "bundle_hash") else ""
            ens.shared_evidence_as_of = as_of
        except Exception as e:  # noqa: BLE001 — evidence failure never blocks the forecast
            bundle, evidence_text = None, ""
            ens.generation_policy["evidence_error"] = f"{type(e).__name__}: {e}"[:160]

    # ---------- critics + adaptive expansion + Stage B compile + dedup + certificate ----------
    try:
        omission = EC.run_omission_critic(ens, llm=llm, evidence_text=evidence_text, ledger=ledger,
                                          cache_store=cache_store)
        contrast = EC.run_contrast_critic(ens, llm=llm, ledger=ledger, cache_store=cache_store)
        if EC.expansion_triggers(ens, omission, contrast):
            EC.expand_candidates(ens, omission, llm=llm, as_of=as_of, horizon=horizon,
                                 intervention=intervention, user_context=user_context,
                                 evidence_text=evidence_text, seed=seed, ledger=ledger,
                                 cache_store=cache_store)
        EC.run_candidate_critics(ens, llm=llm, evidence_text=evidence_text, ledger=ledger,
                                 cache_store=cache_store)
        EC.compile_candidates(ens, llm=llm, as_of=as_of, horizon=horizon, intervention=intervention,
                              evidence=bundle if bundle is not None else (evidence or ""), seed=seed,
                              ledger=ledger, cache_store=cache_store)
    except EnsembleIntegrityError as e:
        return _fail("invalid_execution_plan", f"ensemble integrity violation: {e}")
    executable = [c for c in ens.surviving() if c.executable_plan is not None]
    if not executable:
        reasons = [c.promotion_reason for c in ens.candidates]
        if reasons and all("ClarificationRequired" in r for r in reasons if r):
            return SimulationResult(
                question=question, simulation_status="clarification_required",
                clarification_reason=(reasons[0].split("ClarificationRequired:")[-1].strip()[:300]
                                      or "no coherent outcome contract"),
                latency_s=round(_time.time() - t0, 3),
                provenance={"runtime": U.RUNTIME_VERSION, "structural_mode": "ensemble",
                            "structural_ensemble_generation": ens.as_dict()})
        return _fail("invalid_execution_plan",
                     "no executable structural candidate remained after generation, critics and bounded "
                     "repair", {"structural_ensemble_generation": ens.as_dict()})
    EC.deduplicate_candidates(ens, llm=llm, ledger=ledger, cache_store=cache_store,
                              contrast_hints=contrast)
    EC.finalize_survivorship(ens, omission)
    try:
        ens.validate_integrity()
    except EnsembleIntegrityError as e:
        return _fail("invalid_execution_plan", f"ensemble integrity violation: {e}")

    # ---------- per-model conditioning + REAL pilots through the canonical funnel ----------
    cond_llm = CachedLLM(llm, ledger=ledger, stage="model_conditioning", store=cache_store)
    actor_cache = ScopedActorCache(llm, ledger=ledger, stage="actor_rollout")
    # §17: ONE model-family pool for the whole run — deterministic per-(particle, actor)
    # assignments, recorded transitions, honest monoculture reporting on the result
    try:
        from swm.world_model_v2.model_families import default_family_pool
        family_pool = default_family_pool(llm)
    except Exception:  # noqa: BLE001
        family_pool = None
    runs = {}                                     # model_id -> per-model run record
    for cand in ens.surviving():
        if cand.executable_plan is None:
            continue
        if family_pool is not None:
            cand.executable_plan._family_pool = family_pool
        rec = _condition_and_pilot_model(U, question, cand, bundle, as_of, horizon, intervention,
                                         seed, cond_llm, actor_cache, user_context, prior_checkpoint,
                                         drop, t0, evidence=evidence)
        runs[cand.model_id] = rec
        ens.pilot_models.append(cand.model_id)
    ens.candidates_rejected = sum(1 for c in ens.candidates
                                  if c.promotion_status in ("rejected", "failed"))

    # ---------- uncertainty-aware conservative promotion ----------
    _promote_models(ens, runs)

    # ---------- full per-model budgets: extend every promoted model to ≥ its OWN full particle count ----
    for cand in [c for c in ens.surviving() if c.promotion_status == "promoted"]:
        rec = runs[cand.model_id]
        if rec.get("error"):
            continue
        _extend_to_full(cand, rec, seed, actor_cache, reuse_pilot=pilot_reuse)
    # disagreement-driven extension AFTER first full pass (Section 0.9): more particles when structural
    # disagreement is material, never fewer
    dists0 = {m: r.get("full_distribution") or r.get("pilot_distribution") or {}
              for m, r in runs.items()
              if ens.by_id(m) is not None and ens.by_id(m).promotion_status == "promoted"}
    cls0 = classify_forecast_sensitivity(dists0, underidentified=ens.structurally_underidentified)
    if cls0["classification"] == "materially_structurally_sensitive":
        for cand in [c for c in ens.surviving() if c.promotion_status == "promoted"]:
            rec = runs[cand.model_id]
            if rec.get("error"):
                continue
            extra = max(1, math.ceil(rec["n_full"] * DISAGREEMENT_EXTENSION_FRACTION))
            _extend_to_full(cand, rec, seed, actor_cache, extra_particles=extra,
                            reason="material_structural_disagreement")

    # ---------- finalize each promoted model through the canonical result funnel ----------
    # exactly ONE model (the first promoted) commits the persistence checkpoint — one run advances
    # the actor-history lineage once, exactly as the single-model runtime did
    model_results = {}
    checkpoint_committed = False
    for cand in ens.surviving():
        rec = runs.get(cand.model_id)
        if rec is None or rec.get("error"):
            continue
        commit = cand.promotion_status == "promoted" and not checkpoint_committed
        res_m = _finalize_model(U, question, cand, rec, bundle, as_of, intervention, seed, t0,
                                commit_checkpoint=commit)
        if res_m is not None:
            model_results[cand.model_id] = res_m
            if commit:
                checkpoint_committed = True
    promoted = [c for c in ens.surviving() if c.promotion_status == "promoted"
                and c.model_id in model_results]
    if not promoted:
        return _fail("runtime_exception",
                     "every promoted structural model failed terminal projection",
                     {"structural_ensemble_generation": ens.as_dict(),
                      "per_model_finalize_errors": {m: {"error": r.get("error"),
                                                        "traceback": r.get("error_traceback")}
                                                    for m, r in runs.items() if r.get("error")}})

    # ---------- budget invariant: promoted models carry >= their full single-model budget ----------
    for cand in promoted:
        rec = runs[cand.model_id]
        if len(rec["branches"]) < rec["n_full"]:
            return _fail("invalid_execution_plan",
                         f"BUDGET VIOLATION: promoted model {cand.model_id} ran "
                         f"{len(rec['branches'])} < full budget {rec['n_full']} particles — budgets are "
                         f"never divided across models", {"structural_ensemble_generation": ens.as_dict()})
        cand.final_particles = len(rec["branches"])
        ens.full_models.append(cand.model_id)
        ens.simulation_manifest[cand.model_id] = {
            "pilot_particles": rec["n_pilot"], "final_particles": len(rec["branches"]),
            "full_budget_required": rec["n_full"],
            "pilot_reused_as_prefix": not rec.get("pilot_discarded", False),
            "extensions": rec.get("extensions", []), "status": "completed"}
    for cand in ens.candidates:
        if cand.promotion_status != "promoted" and cand.model_id in runs \
                and cand.model_id not in ens.simulation_manifest:
            rec = runs[cand.model_id]
            ens.simulation_manifest[cand.model_id] = {
                "pilot_particles": rec.get("n_pilot", 0), "final_particles": 0,
                "full_budget_required": rec.get("n_full", 0), "pilot_reused_as_prefix": False,
                "status": ("pilot_only:" + cand.promotion_status) if not rec.get("error")
                          else f"failed: {rec['error'][:120]}"}

    # ---------- honest aggregation + sensitivity + reversal + structural value-of-information ----------
    ens.cost_manifest = ledger.as_dict()
    return _assemble_ensemble_result(U, question, ens, runs, model_results, promoted, bundle,
                                     ledger, t0)


# ------------------------------------------------------------------ per-model pipeline
def _condition_and_pilot_model(U, question, cand, bundle, as_of, horizon, intervention, seed,
                               cond_llm, actor_cache, user_context, prior_checkpoint, drop, t0,
                               evidence=""):
    """Condition ONE model's own plan (evidence recompile → ITS OWN posterior → fidelity/event-time/
    Phase 11) and run its REAL pilot through the canonical persistence funnel. The pilot uses the full
    causal runtime — the model's actual plan, actual posterior, actual qualitative actors, actual
    institutions, the real event queue and the real horizon — only the particle count is reduced, and
    those particles are a deterministic PREFIX of the model's full run."""
    from swm.world_model_v2.phase8_pipeline import prepare_persistence_run, run_persistence_slice
    plan = cand.executable_plan
    manifest = {p: U._mani(available=True) for p in U._PHASES}
    manifest["phase1_compiler"].update(selected=True, executed=True, version="ensemble-compiler",
                                       reason=f"structural candidate {cand.model_id}")
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
        # ---------- §3-§6: EXPLICIT WORLD BOUNDARY per structural model (default-on) ----------
        # generation (actual LLM) → three independent critics → residual outside-world process →
        # bounded adversarial sensitivity. Never inferred solely from plan.entities; the plan is
        # cross-checked INTO the boundary. Results ride the plan into world construction (the
        # boundary monitor + outside-world arrivals) and into the §35.1 result sections.
        if "world_boundary" not in drop:
            try:
                from swm.world_model_v2.world_boundary import (
                    boundary_sensitivity_analysis, generate_world_boundary, run_boundary_critics)
                from swm.world_model_v2.outside_world import generate_outside_world
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
                try:
                    _opts = [str(o) for o in
                             (getattr(plan.outcome_contract, "options", None) or [])]
                except Exception:  # noqa: BLE001
                    _opts = []
                boundary_sensitivity_analysis(boundary, llm=model_llm, options=_opts)
                plan._world_boundary = boundary
                plan._outside_world = outside
                rec["boundary"] = boundary
                rec["outside_world"] = outside
                cand.world_boundary = json.dumps(boundary.boundary_answers(), default=str)[:800] \
                    if boundary.components else cand.world_boundary
                lineage["world_boundary"] = {"boundary_id": boundary.boundary_id,
                                             "boundary_hash": boundary.boundary_hash(),
                                             "support": boundary.support_classification,
                                             "n_components": len(boundary.components),
                                             "n_critic_findings": len(boundary.critic_findings),
                                             "outside_families": len(outside.families),
                                             "outside_unresolved":
                                                 len(outside.unresolved_external_risks)}
            except Exception as e:  # noqa: BLE001 — a failed boundary stage is a LOUD gap, not
                # a silent completion: recorded on the record; classification downgrades below
                rec["boundary_error"] = f"{type(e).__name__}: {e}"[:200]
                lineage["world_boundary"] = {"error": rec["boundary_error"]}
        posterior = U._phase3_block(question, plan, bundle, model_llm, seed, manifest, drop)
        rec["posterior_consumed"] = bool(posterior and posterior.n_effective_observations > 0)
        rec["posterior"] = posterior
        rec["prior_spec"] = getattr(plan, "_outcome_prior_spec", None)
        # evidence-sufficiency gate, PER structural model (each model has its own posterior):
        # a starved model must never be silently mistaken for evidence-driven
        rec["evidence_sufficiency"] = U._evidence_sufficiency_block(
            question, bundle, posterior, as_of=as_of, drop=drop, lineage=lineage)
        cand.posterior_diagnostics = ({"n_effective_observations": posterior.n_effective_observations,
                                       "outcome_rate_mean": getattr(posterior, "outcome_rate_mean", None)}
                                      if posterior is not None else {"n_effective_observations": 0})
        U._condition_plan(question, plan, bundle, as_of, horizon, seed, model_llm,
                          manifest, lineage, costs, drop,
                          user_context=user_context, intervention=intervention,
                          structural_model_id=cand.model_id, evidence=evidence)
        cand.plan_lineage = list(lineage["plan_hashes"]) or [plan.plan_hash()]
        # ---------- pilot through the canonical funnel (persistence-prepared, index-keyed slice) --------
        actor_cache.model_id = cand.model_id
        handle = prepare_persistence_run(question, plan, llm=actor_cache, context=user_context,
                                         actor_history=(prior_checkpoint or {}).get("actor_history")
                                         if prior_checkpoint else None)
        n_full = int(handle["n_particles"])
        n_pilot = min(n_full, max(PILOT_MIN_PARTICLES, math.ceil(n_full * PILOT_FRACTION)))
        branches = run_persistence_slice(handle, seed=seed, n_total=n_full, start=0, stop=n_pilot,
                                         particle_scope=actor_cache)
        rec.update(handle=handle, branches=branches, n_pilot=n_pilot, n_full=n_full)
        pilot_projection = handle["run"].project(list(branches))
        rec["pilot_distribution"] = dict(pilot_projection.get("distribution") or {})
        rec["pilot_projection"] = {k: v for k, v in pilot_projection.items() if k != "event_time"}
        cand.pilot_status = "completed"
        cand.pilot_particles = n_pilot
        cand.pilot_result = {"distribution": rec["pilot_distribution"],
                             "n_particles": n_pilot,
                             "unresolved_share": pilot_projection.get("unresolved_share"),
                             "binary_se": _binary_se(rec["pilot_distribution"], n_pilot)}
    except Exception as e:  # noqa: BLE001 — a failed model is recorded loudly, others continue
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
        cand.pilot_status = "failed"
        cand.promotion_status = "failed"
        cand.promotion_reason = f"pilot_failed: {rec['error']}"
    return rec


def _binary_se(dist: dict, n: int) -> float:
    """Sampling standard error of the leading option's frequency — the pilot noise measure."""
    if not dist or n <= 0:
        return 1.0
    p = max(dist.values())
    return round(math.sqrt(max(p * (1.0 - p), 1e-6) / n), 4)


def _promote_models(ens, runs):
    """Conservative, uncertainty-aware promotion (Section 12). Every surviving pilot model is promoted
    UNLESS a hard negative ground holds. Never rejected for a low probability, a bad-looking action, an
    inconvenient result, critic preference, or pilot-mean ranking. A noisy pilot always promotes."""
    survivors = [c for c in ens.surviving() if c.model_id in runs and not runs[c.model_id].get("error")]
    for cand in survivors:
        rec = runs[cand.model_id]
        others = [o for o in survivors if o is not cand]
        block = None
        # the ONLY pilot-based non-promotion ground: behaviorally indistinguishable from a STRONGER
        # model AND structurally near-identical AND the pilot is large enough that this is not noise
        if others and rec["n_pilot"] >= PILOT_MIN_N_FOR_EXCLUSION:
            for o in others:
                orec = runs[o.model_id]
                if orec.get("error") or orec["n_pilot"] < PILOT_MIN_N_FOR_EXCLUSION:
                    continue
                spread = _dist_spread(rec.get("pilot_distribution"), orec.get("pilot_distribution"))
                stronger = _support_rank(o.support_class) > _support_rank(cand.support_class)
                similar = _core_similarity(cand, o) >= 0.75
                if spread is not None and spread <= PILOT_INDISTINGUISHABLE_SPREAD and stronger and similar:
                    block = (f"pilot_indistinguishable_from_stronger_model:{o.model_id} "
                             f"(spread={spread:.3f} ≤ {PILOT_INDISTINGUISHABLE_SPREAD}, structures "
                             f"similar, support {cand.support_class} < {o.support_class}); structure "
                             f"adds no material sensitivity")
                    break
        if block:
            cand.promotion_status = "not_promoted"
            cand.promotion_reason = block
        else:
            cand.promotion_status = "promoted"
            cand.promotion_reason = _promotion_reason(cand, rec, survivors, runs)


def _promotion_reason(cand, rec, survivors, runs) -> str:
    if cand.support_class in ("strongly_supported", "plausible"):
        base = f"support_class={cand.support_class}"
    elif rec["n_pilot"] < PILOT_MIN_N_FOR_EXCLUSION:
        base = "pilot_too_small_to_exclude_safely"
    else:
        base = "materially_distinct_or_uncertain"
    spreads = [_dist_spread(rec.get("pilot_distribution"), runs[o.model_id].get("pilot_distribution"))
               for o in survivors if o is not cand and not runs[o.model_id].get("error")]
    spreads = [s for s in spreads if s is not None]
    if spreads and max(spreads) > PILOT_INDISTINGUISHABLE_SPREAD:
        base += f"; predicts_differently (max pilot spread {max(spreads):.3f})"
    return base


def _support_rank(sc: str) -> int:
    return {"contradicted": 0, "unresolved": 1, "weak_but_possible": 2, "plausible": 3,
            "strongly_supported": 4}.get(sc, 1)


def _dist_spread(a: dict, b: dict):
    if not a or not b:
        return None
    options = set(a) | set(b)
    return max(abs(float(a.get(o, 0.0)) - float(b.get(o, 0.0))) for o in options) if options else 0.0


def _core_similarity(a, b) -> float:
    from swm.world_model_v2.structural_contracts import structural_signature
    from swm.world_model_v2.ensemble_compiler import _sig_similarity
    if a.executable_plan is None or b.executable_plan is None:
        return 0.0
    return _sig_similarity(structural_signature(a.executable_plan),
                           structural_signature(b.executable_plan))["structural_min"]


def _extend_to_full(cand, rec, seed, actor_cache, extra_particles: int = 0, reason: str = "",
                    reuse_pilot: bool = True):
    """Progressive extension: continue the SAME prepared run from the pilot prefix to the full budget
    (plus any disagreement extension). Pilot particles are retained — identical worlds/seeds by particle
    index — so nothing is rerun and nothing is discarded. `reuse_pilot=False` exists ONLY for the cost
    benchmark's reuse-off arm: it reruns the full budget from index 0 (strictly more compute, identical
    result by index-keyed determinism) and records the discarded pilot honestly."""
    from swm.world_model_v2.phase8_pipeline import run_persistence_slice
    target = rec["n_full"] + int(extra_particles)
    if not reuse_pilot and rec["branches"] and not rec.get("pilot_discarded"):
        rec["pilot_discarded"] = True
        rec["extensions"].append({"from": len(rec["branches"]), "to": 0,
                                  "reason": "pilot_reuse_disabled_benchmark_arm: pilot discarded"})
        rec["branches"] = []
    start = len(rec["branches"])
    if start >= target:
        return
    actor_cache.model_id = cand.model_id
    new = run_persistence_slice(rec["handle"], seed=seed, n_total=target, start=start, stop=target,
                                particle_scope=actor_cache)
    rec["branches"] = list(rec["branches"]) + list(new)
    rec["extensions"].append({"from": start, "to": target,
                              "reason": reason or "promotion_to_full_budget"})
    cand.pilot_status = "reused_in_full" if reuse_pilot else "completed"


def _finalize_model(U, question, cand, rec, bundle, as_of, intervention, seed, t0,
                    commit_checkpoint: bool = True):
    """Terminal projection + result contract for ONE model over ALL its accumulated branches, through the
    canonical finalize funnel, with per-model phase supervision. `commit_checkpoint=False` neutralizes
    the persistence-checkpoint commit (history was already materialized at prepare time; only one model
    per run advances the lineage)."""
    from swm.world_model_v2.phase8_pipeline import finalize_persistence_run, run_persistence_slice
    handle = rec["handle"] if commit_checkpoint else {**rec["handle"], "actor_history": None}
    try:
        res, _artifacts = finalize_persistence_run(handle, rec["branches"],
                                                   intervention=intervention, t0=t0, seed=seed)
    except Exception as e:  # noqa: BLE001
        import traceback as _tb
        rec["error"] = f"finalize: {type(e).__name__}: {e}"[:200]
        rec["error_traceback"] = _tb.format_exc()[-1500:]
        print(f"[finalize {cand.model_id}] {rec['error']}\n{rec['error_traceback']}",
              flush=True)
        return None
    # ROLLOUT RETRY (per structural model): the persistence-aware rollout is stochastic; an
    # INTERMITTENT empty rollout (EXP-106 BoJ: one run's operator census came up empty → no absorber →
    # no forecast; the next run of the SAME plan rolled fully) recovers on a re-roll. Retry ONCE with a
    # fresh seed, recorded — honest unresolved results are exempt (see unified_runtime._no_forecast).
    if U._no_forecast(res):
        rec["lineage"]["rollout_retry"] = {"first_status": res.simulation_status,
                                           "first_taxonomy": res.failure_taxonomy}
        try:
            n = max(1, len(rec["branches"]))
            retry_branches = run_persistence_slice(rec["handle"], seed=seed + 1, n_total=n,
                                                   start=0, stop=n)
            res_retry, _ = finalize_persistence_run(handle, retry_branches,
                                                    intervention=intervention, t0=t0, seed=seed + 1)
            rec["lineage"]["rollout_retry"]["retry_status"] = res_retry.simulation_status
            rec["lineage"]["rollout_retry"]["recovered"] = not U._no_forecast(res_retry)
            if not U._no_forecast(res_retry):
                res = res_retry
                rec["branches"] = list(retry_branches)
        except Exception as e:  # noqa: BLE001 — the retry must never make things worse
            rec["lineage"]["rollout_retry"]["retry_error"] = f"{type(e).__name__}: {e}"[:120]
    rec["manifest"]["phase8_persistence"].update(selected=True, executed=True, version="phase8-1.0",
                                                 reason=f"{len(rec['branches'])} particles "
                                                        f"(pilot prefix {rec['n_pilot']} reused)")
    U._record_operator_phases(cand.executable_plan, res, rec["manifest"])
    U._attach_supervision(res, cand.executable_plan, as_of, bundle, rec["manifest"], rec["lineage"])
    # evidence-sufficiency surfacing + the no-silent-None guard, per structural model
    U._apply_result_guards(res, posterior=rec.get("posterior"), prior_spec=rec.get("prior_spec"),
                           evidence_sufficiency=rec.get("evidence_sufficiency"),
                           lineage=rec["lineage"])
    res.provenance["structural_model_id"] = cand.model_id
    res.provenance["active_component_manifest"] = rec["manifest"]
    res.provenance["plan_lineage"] = rec["lineage"]
    rec["full_distribution"] = dict(res.raw_distribution or {})
    return res


# ------------------------------------------------------------------ aggregation + result assembly
def _assemble_ensemble_result(U, question, ens, runs, model_results, promoted, bundle, ledger, t0):
    model_dists = {c.model_id: dict(model_results[c.model_id].raw_distribution or {})
                   for c in promoted}
    # HONESTY INVARIANT: any model that entered execution and died (pilot/conditioning failure) makes
    # the ensemble INCOMPLETE — including models no longer in surviving(). And an ensemble that shrank
    # to ONE promoted model through execution failures (rather than certified critic/merge survivorship)
    # is never "stable": a single survivor requires the generation-time convergence certificate.
    incomplete = any(r.get("error") for r in runs.values())
    if len(promoted) == 1 and ens.convergence_certificate is None:
        incomplete = True
    classification = classify_forecast_sensitivity(
        model_dists, underidentified=ens.structurally_underidentified, incomplete=incomplete)
    mixture = _equal_weight_mixture(model_dists)
    robust = _robust_range(model_dists)
    # §NAP: per-model honest-resolution accounting (unresolved mass, bounds, missing mechanisms,
    # numeric-causal-input manifests) rolls up into ONE ensemble resolution report
    per_model_resolution = {m: dict(model_results[m].resolution_report or {})
                            for m in model_results}
    unresolved_by_model = {m: float((per_model_resolution.get(m) or {}).get("unresolved_share")
                                    or 0.0) for m in model_dists}
    all_unresolved = bool(model_results) and all(
        model_results[m].simulation_status == "unresolved" for m in model_results)
    any_unresolved_mass = any(v > 0.0 for v in unresolved_by_model.values())
    # §NAP headline policy: when plausible structural models MATERIALLY disagree (or the ensemble
    # is underidentified/incomplete), their conditionals are NOT averaged into one headline
    # probability — per-model distributions + the robust range are the primary readout. The
    # equal-weight mixture survives only as a labeled diagnostic.
    material_disagreement = classification["classification"] in (
        "materially_structurally_sensitive", "structurally_underidentified",
        "ensemble_execution_incomplete") and len(model_dists) > 1
    headline = {} if material_disagreement else dict(mixture)
    from swm.world_model_v2.numeric_provenance import merge_manifests as _merge_manifests
    ensemble_resolution = {
        "per_model": per_model_resolution,
        "unresolved_share_by_model": {m: round(v, 4) for m, v in unresolved_by_model.items()},
        "robust_range": robust,
        "aggregation": ("per_model_conditionals_no_headline_average" if material_disagreement
                        else ("single_surviving_model" if len(model_dists) <= 1
                              else "agreeing_models_mixture")),
        "missing_mechanisms": [mm for m in sorted(per_model_resolution)
                               for mm in (per_model_resolution[m].get("missing_mechanisms")
                                          or [])][:20],
        "frequency_semantics": "simulated_scenario_frequency",
        "numeric_causal_inputs": _merge_manifests(
            *[(per_model_resolution.get(m) or {}).get("numeric_causal_inputs") or {}
              for m in sorted(per_model_resolution)]),
        "note": "unresolved mass is preserved per model and never normalized away; disagreeing "
                "structural models are reported as separate conditionals, never averaged into a "
                "headline probability (§NAP)"}
    within = {m: dict(model_results[m].uncertainty_decomposition or {}) for m in model_dists}
    decomposition = decompose_uncertainty(model_dists, within_model=within)
    reversal = _reversal_conditions(ens, promoted, model_dists, mixture)
    voi = _structural_value_of_information(ens, promoted, model_dists)

    # support grade: the most conservative promoted grade; material structural sensitivity caps it
    grades = [model_results[c.model_id].support_grade for c in promoted]
    order = ["highly_speculative", "exploratory", "transfer_supported", "empirically_supported"]
    grade = min(grades, key=lambda g: order.index(g) if g in order else 1) if grades else "exploratory"
    limitations = []
    for c in promoted:
        limitations.extend(model_results[c.model_id].limitations[:2])
    if classification["classification"] == "materially_structurally_sensitive":
        if order.index(grade) > order.index("exploratory"):
            grade = "exploratory"
        limitations.append(
            "materially structurally sensitive: plausible causal models disagree "
            f"(max spread {classification.get('max_spread')}); see structural_ensemble.reversal_conditions")
    if ens.structurally_underidentified:
        limitations.append("structurally underidentified: the generation ceiling was reached while the "
                           "omission critic still identified plausible missing structures "
                           "(structural_ensemble.unresolved_alternatives)")
    if incomplete:
        limitations.append("ensemble execution incomplete: structural model(s) failed during "
                           "conditioning/pilot execution (see structural_ensemble.rejected_and_merged "
                           "and simulation_manifest) — the surviving set is not a certified complete "
                           "ensemble")
    degraded = any(model_results[c.model_id].simulation_status == "completed_with_degradation"
                   for c in promoted)

    # ---------- §3/§5/§8/§17/§20/§21/§35: boundary, outside-world, family, truncation ----------
    boundaries = {m: runs[m]["boundary"] for m in model_dists
                  if runs.get(m, {}).get("boundary") is not None}
    outsides = {m: runs[m]["outside_world"] for m in model_dists
                if runs.get(m, {}).get("outside_world") is not None}
    under_subtypes, under_components = [], []
    for c in promoted:
        rec_c = runs.get(c.model_id, {})
        b = rec_c.get("boundary")
        if b is None:
            under_subtypes.append("under_modeled_boundary")
            under_components.append({
                "component": "(world boundary)", "kind": "boundary",
                "why": rec_c.get("boundary_error") or "no boundary was generated for this model",
                "sensitivity": "unknown", "model_id": c.model_id})
            continue
        if b.support_classification == "under_modeled_boundary":
            under_subtypes.append("under_modeled_boundary")
            for u in b.unresolved_decisive()[:6]:
                under_components.append({"component": u.get("name", ""),
                                         "kind": u.get("kind", ""),
                                         "why": u.get("why_unresolved", ""),
                                         "sensitivity": u.get("sensitivity", "decisive"),
                                         "model_id": c.model_id})
        for row in b.omitted_component_sensitivity:
            if row.get("kind") == "external_event_family" and \
                    (row.get("can_reverse_forecast_direction") or
                     row.get("can_change_best_action")):
                under_subtypes.append("under_modeled_external_process")
                under_components.append({"component": row.get("omitted_component", ""),
                                         "kind": "external_event_family",
                                         "why": "decisive external process without a defensible "
                                                "arrival model or mechanism",
                                         "sensitivity": "decisive", "model_id": c.model_id})
    # §28: broad-prior terminal resolutions the default runtime refused → the missing mechanism
    # is named and the run classifies under_modeled_nonhuman_mechanism (never a hidden draw)
    for c in promoted:
        trt = (model_results[c.model_id].provenance or {}).get("temporal_runtime") or {}
        for sup in (trt.get("mechanism_suppressions") or [])[:6]:
            under_subtypes.append("under_modeled_nonhuman_mechanism")
            under_components.append({
                "component": str(sup.get("outcome_var", sup.get("mechanism", ""))),
                "kind": "nonhuman_mechanism",
                "why": f"missing validated mechanism: {sup.get('mechanism', '')} — "
                       f"{str(sup.get('why', ''))[:160]}",
                "sensitivity": "decisive", "model_id": c.model_id})
    under_subtypes = sorted(set(under_subtypes))

    # §21 truncation aggregation: per-branch statuses from each promoted model's temporal stats
    from swm.world_model_v2.truncation import aggregate_branch_statuses, honest_note
    branch_rows, per_branch_modal = [], {}
    for c in promoted:
        trt = (model_results[c.model_id].provenance or {}).get("temporal_runtime") or {}
        n_b = int(trt.get("n_branches") or runs[c.model_id].get("n_full") or 0) or \
            len(runs[c.model_id].get("branches") or [])
        n_trunc = int(trt.get("n_branches_truncated") or 0)
        listed = ((trt.get("truncation") or {}).get("branches") or [])
        for i in range(n_b):
            bid = f"{c.model_id}/b{i:03d}"
            row = {"branch_id": bid, "truncated": False, "truncation": {},
                   "weight": 1.0 / max(1, n_b) / max(1, len(promoted))}
            branch_rows.append(row)
        for j, tb in enumerate(listed[:n_trunc] if listed else
                               [{"branch_status": "truncated_event_budget"}] * n_trunc):
            if j < n_b:
                branch_rows[len(branch_rows) - n_b + j].update(
                    truncated=True, truncation=dict(tb))
    trunc_agg = aggregate_branch_statuses(branch_rows)
    trunc_weight = float(trunc_agg.get("truncated_weight") or 0.0)
    bounds = {}
    # §21 bounds from the mixture: completed mass per option scaled by the non-truncated
    # share; the truncated share swings fully for/against each option (pure arithmetic)
    if mixture and trunc_weight > 0:
        comp = 1.0 - trunc_weight
        bounds = {o: {"lower": round(float(mixture.get(o, 0.0)) * comp, 4),
                      "upper": round(float(mixture.get(o, 0.0)) * comp + trunc_weight, 4)}
                  for o in mixture}
        modal_o = max(mixture, key=mixture.get)
        answer_settled = all(bounds[modal_o]["lower"] >= bounds[o]["upper"]
                             for o in mixture if o != modal_o)
    else:
        answer_settled = True
    truncation_report = {
        **trunc_agg, "bounds_under_truncation": bounds,
        "answer_settled_under_truncation": answer_settled,
        "note": honest_note(),
        "recommendation_rule": "a recommendation is withheld unless the leading action remains "
                               "best under all admissible truncated-branch completions (§21)"}

    # §17.4 model-family report (one pool serves the whole run)
    fam_pool = getattr(promoted[0].executable_plan, "_family_pool", None)
    model_family_report = fam_pool.report() if fam_pool is not None else {
        "model_family_monoculture": True,
        "monoculture_note": "no family pool constructed — single injected backend",
        "families": []}

    # §35.3 hybrid-mechanism report: what executed on the shared world, with unified
    # calibration statuses; unresolved/suppressed mechanisms named
    hybrid_mechanisms = {"shared_world_contract": "one typed WorldState, one clock, one event "
                                                  "queue, StateDelta everywhere",
                         "per_model": {}, "unresolved_mechanisms": [], "external_adapters": []}
    for c in promoted:
        provm = model_results[c.model_id].provenance or {}
        cr = provm.get("consequence_report") or {}
        trt = provm.get("temporal_runtime") or {}
        hybrid_mechanisms["per_model"][c.model_id] = {
            "mechanisms_invoked": cr.get("mechanisms_invoked"),
            "mechanism_successes": cr.get("mechanism_successes"),
            "mechanism_failures": cr.get("mechanism_failures"),
            "mechanism_unresolved": cr.get("mechanism_unresolved"),
            "outside_events_entered": cr.get("outside_events_entered"),
            "outside_entry_downgrades": (cr.get("outside_entry_downgrades") or [])[:6],
            "operator_delta_census": (provm.get("operator_delta_census") or [])[:12]
            if isinstance(provm.get("operator_delta_census"), list)
            else provm.get("operator_delta_census"),
            "accepted_mechanisms": [
                {k: m.get(k) for k in ("mech_id", "ontology_type", "operator",
                                       "calibration_status", "parameter_source")}
                for m in (getattr(c.executable_plan, "accepted_mechanisms", None) or [])
                if isinstance(m, dict)][:16],
        }
        for sup in (trt.get("mechanism_suppressions") or [])[:4]:
            hybrid_mechanisms["unresolved_mechanisms"].append(
                {"model_id": c.model_id, **{k: sup.get(k) for k in
                                            ("mechanism", "outcome_var", "why")}})
    try:
        from swm.world_model_v2.mechanism_spec import known_external_adapters
        hybrid_mechanisms["external_adapters"] = sorted(known_external_adapters())[:8]
    except Exception:  # noqa: BLE001 — adapter registry optional
        pass

    # §35.2 cognition report (stage architecture + per-model actor reports + bounded samples)
    cognition_report = {
        "pipeline": "observation→attention→working_memory→memory_retrieval→interpretation→"
                    "limited_action_search→one_choice→memory_update",
        "schema": "bounded.cognition.v1",
        "per_model_actor_policy": {m: (model_results[m].provenance or {}).get(
            "actor_policy_report") for m in model_dists},
        "sample_decision_records": [
            s for m in model_dists
            for s in ((model_results[m].provenance or {}).get("cognition_records_sample")
                      or [])][:8],
        "model_family_monoculture": model_family_report.get("model_family_monoculture", True),
    }

    # §29: does the ANSWER change because of each uncertainty axis? Measured where the run
    # actually varied the axis; honestly not-measured otherwise (never fabricated).
    answer_change_attribution = {
        "structural_model": {"measured": len(model_dists) > 1,
                             "changes_answer": classification["classification"] in
                             ("materially_structurally_sensitive",
                              "structurally_underidentified"),
                             "basis": classification.get("classification")},
        "world_boundary": {"measured": bool(boundaries),
                           "changes_answer": "under_modeled_boundary" in under_subtypes,
                           "basis": "boundary critics + sensitivity analysis"},
        "actor_cognition": {"measured": False,
                            "basis": "per-run cognition variants not run in-band; see the "
                                     "bounded-vs-oneshot ablation arm "
                                     "(artifacts/core_arch/ablation_report.json)"},
        "model_family": {"measured": not model_family_report.get("model_family_monoculture",
                                                                 True),
                         "changes_answer": None,
                         "basis": ("single-lineage monoculture — family disagreement is "
                                   "UNMEASURABLE this run and reported as risk"
                                   if model_family_report.get("model_family_monoculture", True)
                                   else "per-family decisions recorded in assignments")},
        "nonhuman_mechanism": {"measured": True,
                               "changes_answer": "under_modeled_nonhuman_mechanism"
                               in under_subtypes,
                               "basis": "mechanism suppressions + unresolved mechanisms"},
        "truncation": {"measured": True,
                       "changes_answer": (trunc_weight > 0 and not answer_settled),
                       "basis": "per-option bounds under admissible truncated completions"},
    }

    if under_subtypes:
        limitations.append(
            "UNDER-MODELED: unresolved high-sensitivity boundary components remain outside the "
            "represented world (see under_modeled_components); the distribution is conditional "
            "on the represented boundary")
    if trunc_weight > 0:
        limitations.append(
            f"truncated branch weight {round(trunc_weight, 4)} — {honest_note()}")

    primary = promoted[0]
    single_equivalent_calls = _single_model_equivalent_calls(ledger)
    ensemble_block = {
        "ensemble_id": ens.ensemble_id,
        "structural_mode": "ensemble",
        "generation_policy": ens.generation_policy,
        "n_independent_generation_calls": ens.independent_generation_calls(),
        "n_initial_candidates": sum(1 for g in ens.generation_manifest if g.get("independent")),
        "n_expansion_candidates": sum(1 for g in ens.generation_manifest if not g.get("independent")),
        "n_rejected": ens.candidates_rejected, "n_repaired": ens.candidates_repaired,
        "n_merged": ens.candidates_merged, "n_pilot_simulated": len(ens.pilot_models),
        "n_fully_simulated": len(ens.full_models),
        "models": [{
            "model_id": c.model_id, "generation_role": c.generation_role,
            "causal_thesis": c.causal_thesis, "plan_hash": c.plan_hash, "schema_hash": c.schema_hash,
            "decisive_actors": c.decisive_actors, "decisive_institutions": c.decisive_institutions,
            "decisive_constraints": c.decisive_constraints, "decisive_mechanisms": c.decisive_mechanisms,
            "world_boundary": c.world_boundary,
            "support_class": c.support_class, "support_basis": c.support_basis,
            "critic_findings": c.critic_findings[:6], "validation": c.validation,
            "promotion_status": c.promotion_status, "promotion_reason": c.promotion_reason,
            "pilot_particles": c.pilot_particles, "final_particles": c.final_particles,
            "plan_lineage": c.plan_lineage, "parent_ids": c.parent_ids,
            "posterior_diagnostics": c.posterior_diagnostics,
            "pilot_result": c.pilot_result,
            "prediction": (dict(model_results[c.model_id].raw_distribution or {})
                           if c.model_id in model_results else None),
            "trajectory_summary": _trajectory_summary(model_results.get(c.model_id)),
            "failure": runs.get(c.model_id, {}).get("error", ""),
        } for c in ens.surviving()],
        "rejected_and_merged": [{
            "model_id": c.model_id, "status": c.promotion_status, "reason": c.promotion_reason,
            "merge_record": c.merge_record} for c in ens.candidates
            if c.promotion_status in ("rejected", "merged", "failed")],
        "model_support": ens.model_support,
        "structural_coverage": ens.structural_coverage,
        "unresolved_alternatives": ens.unresolved_alternatives,
        "convergence_certificate": ens.convergence_certificate,
        "structurally_underidentified": ens.structurally_underidentified,
        "stopping_reason": ens.stopping_reason,
        "generation_manifest": ens.generation_manifest,
        "critic_manifest": ens.critic_manifest,
        "merge_manifest": ens.merge_manifest,
        "simulation_manifest": ens.simulation_manifest,
        "shared_evidence_bundle_hash": ens.shared_evidence_bundle_hash,
        "shared_evidence_as_of": ens.shared_evidence_as_of,
        "aggregation_method": ("single_surviving_model" if len(promoted) == 1
                               else ("per_model_conditionals_no_headline_average"
                                     if material_disagreement
                                     else "agreeing_models_equal_weight_mixture")),
        "aggregation_note": ("one surviving model with a recorded convergence certificate"
                             if len(promoted) == 1 else
                             ("§NAP: plausible structural models materially disagree — their "
                              "conditionals are NOT averaged into a headline probability; "
                              "per-model distributions and the robust range are the primary "
                              "readouts (the equal-weight mixture survives only as a labeled "
                              "diagnostic)" if material_disagreement else
                              "models agree within the exposed spread thresholds; the mixture "
                              "is served as the surviving shared conclusion, with the robust "
                              "range alongside")),
        "model_distributions": model_dists,
        "equal_weight_mixture_diagnostic": mixture,
        "robust_range": robust,
        "uncertainty_decomposition": decomposition,
        "structural_sensitivity": classification,
        "recommendation_stability": None,          # filled by Phase 13 when actions are evaluated
        "reversal_conditions": reversal,
        "structural_value_of_information": voi,
        "cost_manifest": {**ledger.as_dict(),
                          "single_model_equivalent_llm_calls": single_equivalent_calls,
                          "incremental_call_multiplier": (
                              round(ledger.total_calls() / single_equivalent_calls, 2)
                              if single_equivalent_calls else None),
                          "pilot_particles_reused": sum(
                              m.get("pilot_particles", 0) for m in ens.simulation_manifest.values()
                              if m.get("pilot_reused_as_prefix"))},
        "human_summary": _human_summary(question, mixture, classification, reversal, voi, promoted),
        "answer_change_attribution": answer_change_attribution,
    }

    # ---------- §8 status ladder (epistemic states, never engineering exceptions):
    # fully-unresolved dominates everything; under_modeled (an unresolved high-sensitivity
    # component) dominates truncated (compute stopped branches whose mass could change the
    # answer) dominates partially_resolved (explicit unresolved mass or material structural
    # disagreement) dominates degradation. STATUS DESCRIBES THE RUN — it never erases the
    # probability (forecast_recovery contract): per-model recovered probabilities aggregate
    # into a labeled headline below even when every model is unresolved or models disagree.
    if all_unresolved:
        sim_status = "unresolved"
        headline = {}
    elif under_subtypes:
        sim_status = "under_modeled"
    elif trunc_weight > 0 and not answer_settled:
        sim_status = "truncated"
    elif any_unresolved_mass or material_disagreement:
        sim_status = "partially_resolved"
    elif degraded or incomplete or trunc_weight > 0:
        sim_status = "completed_with_degradation"
    else:
        sim_status = "completed"
    res = SimulationResult(
        question=question,
        simulation_status=sim_status,
        support_grade=grade,
        raw_distribution=headline,
        raw_probability=(_binary_projection(headline, promoted[0].executable_plan)
                         if headline else None),
        resolution_report=ensemble_resolution,
        uncertainty_decomposition=decomposition,
        structural_disagreement={m: d for m, d in model_dists.items()},
        limitations=limitations[:10],
        interpretation_hypotheses=[{"model_id": c.model_id, "thesis": c.causal_thesis}
                                   for c in promoted],
        plan_hash=primary.plan_hash,
        structural_ensemble=ensemble_block,
        under_modeled_subtypes=under_subtypes,
        under_modeled_components=under_components[:16],
        world_boundaries={m: {**b.as_dict(), "answers": b.boundary_answers()}
                          for m, b in boundaries.items()},
        outside_world={m: ow.as_dict() for m, ow in outsides.items()},
        cognition_report=cognition_report,
        hybrid_mechanisms=hybrid_mechanisms,
        truncation_report=truncation_report,
        model_family_report=model_family_report,
        provenance={
            "runtime": U.RUNTIME_VERSION, "structural_mode": "ensemble",
            "ensemble_id": ens.ensemble_id,
            "active_component_manifest": runs[primary.model_id]["manifest"],
            "plan_lineage": {c.model_id: c.plan_lineage for c in promoted},
            "evidence_bundle_hash": ens.shared_evidence_bundle_hash,
            "posterior_consumed": any(runs[c.model_id].get("posterior_consumed") for c in promoted),
            "per_model_provenance": {m: model_results[m].provenance for m in model_dists},
            "calibration_compatibility": {
                "old_phase12_calibrator": "INCOMPATIBLE",
                "reason": "the structural-ensemble runtime changes the forecast distribution; refit "
                          "required before any calibrator may serve this runtime."}},
        latency_s=round(_time.time() - t0, 3))
    # ---------- ensemble-level forecast availability (forecast_recovery contract) ----------
    # Per-model results already carry their recovered probabilities/labels; when the ensemble
    # headline was suppressed (all-unresolved, or material disagreement), aggregate the
    # per-model probabilities into the labeled equal-weight mixture: the per-model conditionals
    # remain the primary readout in the resolution report, the headline is served with
    # weight_sensitive marked whenever plausible model-weight changes cross 0.5.
    if res.raw_probability is None:
        per_model_p = {m: model_results[m].raw_probability for m in model_results
                       if model_results[m].raw_probability is not None}
        if per_model_p:
            ps = list(per_model_p.values())
            res.raw_probability = round(sum(ps) / len(ps), 4)
            res.uncertainty_interval = [round(min(ps), 4), round(max(ps), 4)]
            res.weight_sensitive = (min(ps) < 0.5 < max(ps)) or len(per_model_p) < len(
                model_results)
            srcs = [model_results[m].probability_source for m in per_model_p]
            grades = [model_results[m].grounding_grade for m in per_model_p]
            order = ["grounded", "partially_grounded", "exploratory", "ungrounded"]
            res.probability_source = ("mixed:" + "+".join(sorted(set(s for s in srcs if s)))
                                      if len(set(srcs)) > 1 else (srcs[0] if srcs else ""))
            res.grounding_grade = max((g for g in grades if g), default="",
                                      key=lambda g: order.index(g) if g in order else 2)
            res.unresolved_mass = round(sum(
                float(model_results[m].unresolved_mass or 0.0) for m in per_model_p)
                / len(per_model_p), 4)
            res.confidence = "very_low" if res.weight_sensitive or res.grounding_grade in (
                "exploratory", "ungrounded") else "low"
            res.provenance["forecast_recovery_ensemble"] = {
                "per_model_probability": {m: round(p, 4) for m, p in per_model_p.items()},
                "per_model_source": {m: model_results[m].probability_source
                                     for m in per_model_p},
                "per_model_grade": {m: model_results[m].grounding_grade for m in per_model_p},
                "aggregation": "equal_weight_mixture_of_per_model_recovered_probabilities",
                "note": "the headline is served under the forecast-availability contract; "
                        "per-model conditionals remain the primary readout when models "
                        "materially disagree (see resolution_report.aggregation)"}
            res.limitations = (list(res.limitations or []) + [
                f"headline probability {res.raw_probability:.3f} aggregates per-model recovered "
                f"estimates (sources: {res.probability_source or 'n/a'}; grade "
                f"{res.grounding_grade or 'n/a'}); interval "
                f"[{res.uncertainty_interval[0]:.3f}, {res.uncertainty_interval[1]:.3f}]"
                + (" — WEIGHT-SENSITIVE: plausible model-weight changes cross 0.5"
                   if res.weight_sensitive else "")])
    # live handle for Phase 13: the SAME ensemble (its executable plans included) so decisions are
    # evaluated across these models by default. Deliberately a non-dataclass attribute — plans carry
    # callables and never belong in the serialized result dict.
    res._ensemble_handle = ens
    return res


def _single_model_equivalent_calls(ledger: CallLedger) -> int:
    """Honest denominator for the incremental-cost multiplier: the backend calls one single-model run
    would have made ≈ one compile + this run's per-model conditioning/actor averages. Computed from the
    observed ledger, never a promise."""
    by_stage = ledger.calls_by_stage
    n_models = max(1, len(ledger.calls_by_model))
    per_model = (by_stage.get("model_conditioning", 0) + by_stage.get("actor_rollout", 0)) / n_models
    return max(1, int(1 + per_model))


def _equal_weight_mixture(model_dists: dict) -> dict:
    if not model_dists:
        return {}
    options = sorted({o for d in model_dists.values() for o in d})
    n = len(model_dists)
    return {o: round(sum(float(d.get(o, 0.0)) for d in model_dists.values()) / n, 4) for o in options}


def _robust_range(model_dists: dict) -> dict:
    options = sorted({o for d in model_dists.values() for o in d})
    return {o: {"min": round(min(float(d.get(o, 0.0)) for d in model_dists.values()), 4),
                "max": round(max(float(d.get(o, 0.0)) for d in model_dists.values()), 4)}
            for o in options} if model_dists else {}


def _binary_projection(dist: dict, plan) -> float:
    try:
        options = [str(o) for o in (plan.outcome_contract.options or [])]
        if dist and len(options) == 2:
            return float(dist.get(options[0], dist.get("True", 0.0)))
    except Exception:  # noqa: BLE001
        pass
    return None


def _trajectory_summary(res) -> dict:
    if res is None:
        return None
    prov = res.provenance or {}
    return {"n_particles": (prov.get("n_particles") or
                            (prov.get("phase8") or {}).get("n_particles")),
            "event_time": (res.raw_distribution or {}).get("event_time"),
            "sensitivity_contributors": (res.sensitivity_contributors or [])[:4],
            "unresolved_share": prov.get("unresolved_share")}


def _reversal_conditions(ens, promoted, model_dists, mixture) -> list:
    """Assumptions/evidence that would REVERSE the answer — generated from ACTUAL model differences
    (modal-option flips and material spreads between concrete promoted models), never from an LLM
    impression."""
    out = []
    if not mixture:
        return out
    ens_modal = max(mixture, key=mixture.get)
    for c in promoted:
        d = model_dists.get(c.model_id) or {}
        if not d:
            continue
        modal = max(d, key=d.get)
        if modal != ens_modal:
            out.append({
                "model_id": c.model_id,
                "assumption": c.causal_thesis or f"the {c.generation_role} structure dominates",
                "decisive_structure": {"actors": c.decisive_actors[:5],
                                       "institutions": c.decisive_institutions[:4],
                                       "constraints": c.decisive_constraints[:4],
                                       "mechanisms": c.decisive_mechanisms[:4]},
                "if_true_answer_becomes": {modal: d[modal]},
                "ensemble_answer": {ens_modal: mixture[ens_modal]},
                "evidence_that_would_confirm": c.falsifiers[:3] or ["(no falsifiers recorded)"],
            })
    # material spread without a modal flip still names the driving disagreement
    if not out and len(model_dists) > 1:
        spreads = {}
        for o in mixture:
            vals = {m: float(d.get(o, 0.0)) for m, d in model_dists.items()}
            spreads[o] = (max(vals.values()) - min(vals.values()), vals)
        opt, (spread, vals) = max(spreads.items(), key=lambda kv: kv[1][0])
        if spread > 0.0:
            hi = max(vals, key=vals.get)
            lo = min(vals, key=vals.get)
            hi_c, lo_c = ens.by_id(hi), ens.by_id(lo)
            out.append({
                "model_pair": [hi, lo], "option": opt, "spread": round(spread, 4),
                "assumption": (f"whether {hi_c.causal_thesis[:160]!r} or "
                               f"{lo_c.causal_thesis[:160]!r} is the true structure"),
                "if_true_answer_becomes": {hi: round(vals[hi], 4), lo: round(vals[lo], 4)},
                "evidence_that_would_confirm": (hi_c.falsifiers[:2] + lo_c.falsifiers[:2]) or
                                               ["(no falsifiers recorded)"],
            })
    return out


def _structural_value_of_information(ens, promoted, model_dists) -> list:
    """Observations that would DISTINGUISH the surviving structural models — derived from the models'
    own falsifiers, evidence requirements and actual predicted differences. Decision relevance is
    computed (which model's answer the observation would select), not minted; no numeric EVSI is
    fabricated."""
    out = []
    pairs = []
    ids = [c.model_id for c in promoted]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            spread = _dist_spread(model_dists.get(ids[i]), model_dists.get(ids[j]))
            if spread is not None and spread > PILOT_INDISTINGUISHABLE_SPREAD:
                pairs.append((ids[i], ids[j], spread))
    pairs.sort(key=lambda p: -p[2])
    for a_id, b_id, spread in pairs[:4]:
        a, b = ens.by_id(a_id), ens.by_id(b_id)
        for falsifier in (a.falsifiers[:2] + b.falsifiers[:2]):
            owner, other = (a, b) if falsifier in a.falsifiers else (b, a)
            d_owner, d_other = model_dists.get(owner.model_id, {}), model_dists.get(other.model_id, {})
            out.append({
                "observation": falsifier,
                "distinguishes_models": [owner.model_id, other.model_id],
                "predicted_observation_by_model": {
                    owner.model_id: "observation ABSENT if this model is right (it is this model's "
                                    "falsifier)",
                    other.model_id: "observation compatible with this model"},
                "decision_relevance": (
                    f"observing it would shift the answer toward "
                    f"{max(d_other, key=d_other.get) if d_other else '?'} "
                    f"(models differ by {spread:.2f} on their distributions)"),
                "available_before_decision": "unknown — availability not established from the shared "
                                             "evidence bundle",
            })
        unmet = [r for r in (a.evidence_requirements + b.evidence_requirements)
                 if isinstance(r, dict) and r.get("status", "open") == "open"][:2]
        for r in unmet:
            out.append({
                "observation": r.get("claim_or_quantity", ""),
                "distinguishes_models": [a_id, b_id],
                "predicted_observation_by_model": {},
                "decision_relevance": "an open model-specific evidence requirement whose resolution "
                                      "bears on which structure holds",
                "available_before_decision": "unknown",
            })
    seen, deduped = set(), []
    for o in out:
        k = str(o["observation"])[:80]
        if k and k not in seen:
            seen.add(k)
            deduped.append(o)
    return deduped[:8]


def _human_summary(question, mixture, classification, reversal, voi, promoted) -> dict:
    """The five human-facing lines, in the mandated order: answer; survival across models; strongest
    competing explanation; reversing assumption; resolving information."""
    modal = max(mixture, key=mixture.get) if mixture else None
    competing = None
    if len(promoted) > 1:
        competing = promoted[1].causal_thesis or promoted[1].generation_role
    return {
        "answer": ({modal: mixture[modal]} if modal else None),
        "survives_across_models": classification["classification"] in
                                  ("structurally_stable", "mildly_structurally_sensitive"),
        "structural_sensitivity": classification["classification"],
        "most_important_competing_explanation": competing,
        "assumption_that_reverses_answer": (reversal[0].get("assumption") if reversal else None),
        "information_that_would_resolve": (voi[0].get("observation") if voi else None),
    }


# ------------------------------------------------------------------ personal-reaction ensemble route
def _route_individual_reaction_ensemble(question, user_context, llm, as_of, seed, t0,
                                        gen_policy, ledger, cache_store):
    """Personal-reaction questions get the SAME structural-ensemble treatment at the personal scale:
    several independently generated plausible causal frames of the reaction (relationship reading,
    attention/delivery, interpretation, incentives, competing obligations — whatever the generators
    surface), each simulated through the REAL qualitative-actor runtime with its OWN full sample budget,
    then compared. The old single-frame early return survives only inside the explicit
    single_structural_model ablation. Returns None when the question is not an individual reaction."""
    from swm.world_model_v2.actor_selection import is_individual_reaction_question
    from swm.world_model_v2 import ensemble_compiler as EC
    from swm.world_model_v2 import unified_runtime as U
    person = (user_context or {}).get("individual") if isinstance(user_context, dict) else None
    if not isinstance(person, dict) or not is_individual_reaction_question(question):
        return None
    if llm is None:
        return SimulationResult(
            question=question, simulation_status="execution_failed",
            failure_taxonomy="missing_required_operator",
            limitations=["individual-reaction route requires an LLM backend; none supplied — the "
                         "structural ensemble never fabricates a deterministic fallback"],
            provenance={"runtime": U.RUNTIME_VERSION, "structural_mode": "ensemble",
                        "route": "individual_reaction_ensemble",
                        "actor_policy_report": {"degraded": True, "actual_actor_policy_mode": "none",
                                                "construction_error": "no_llm_backend"}},
            latency_s=round(_time.time() - t0, 3))
    try:
        from swm.world_model_v2.compiler import parse_time
        from swm.world_model_v2.individual_reaction import simulate_individual_reaction
        # ---- Stage A at personal scale: independent frames (same adaptive machinery) ----
        ens = EC.reconnoiter_structures(question, llm=llm, as_of=as_of or "", horizon="",
                                        intervention="", user_context={"individual": {
                                            k: person.get(k) for k in
                                            ("role", "your_role", "relationship", "goals")
                                            if person.get(k)}},
                                        seed=seed, generation_policy=gen_policy, ledger=ledger,
                                        cache_store=cache_store)
        omission = EC.run_omission_critic(ens, llm=llm, ledger=ledger, cache_store=cache_store)
        contrast = EC.run_contrast_critic(ens, llm=llm, ledger=ledger, cache_store=cache_store)
        if EC.expansion_triggers(ens, omission, contrast):
            EC.expand_candidates(ens, omission, llm=llm, as_of=as_of or "", horizon="",
                                 intervention="", user_context=None, evidence_text="", seed=seed,
                                 ledger=ledger, cache_store=cache_store)
        # frame-level conservative dedup: identical thesis + decisive sets collapse (no plans here —
        # the executable unit at this scale is the qualitative reaction simulation itself)
        seen = {}
        for c in ens.candidates:
            key = (c.causal_thesis[:120], tuple(sorted(map(str, c.decisive_actors))),
                   tuple(sorted(map(str, c.decisive_constraints))))
            if key in seen:
                c.promotion_status = "merged"
                c.promotion_reason = f"identical_frame:{seen[key]}"
                c.merge_record = {"merged_into": seen[key], "method": "frame_exact",
                                  "confidence": "exact_structural_equality", "comparison": {}}
                ens.merge_manifest.append({"survivor": seen[key], "merged": c.model_id,
                                           "method": "frame_exact",
                                           "confidence": "exact_structural_equality",
                                           "judge_reasoning": "identical thesis and decisive sets",
                                           "structural_comparison": {},
                                           "information_preserved_from_merged": {
                                               "generation_role": c.generation_role}})
            else:
                seen[key] = c.model_id
        ens.candidates_merged = sum(1 for c in ens.candidates if c.promotion_status == "merged")
        EC.finalize_survivorship(ens, omission)
        ens.validate_integrity()
        now = parse_time(as_of) if as_of else _time.time()
        n_h = int(person.get("n_hypotheses", 3) or 3)
        n_s = int(person.get("samples_per_hypothesis", 2) or 2)
        model_artifacts, model_dists = {}, {}
        for cand in ens.surviving():
            frame = (f"{cand.causal_thesis} "
                     f"(decisive: {', '.join(map(str, (cand.decisive_actors + cand.decisive_constraints)[:5]))})"
                     ).strip()
            # FULL per-frame budget: every frame gets the complete n_hypotheses × samples budget the
            # single-frame route would have received (never divided), same seed = common random numbers
            artifact = simulate_individual_reaction(
                person_id=str(person.get("person_id", person.get("name", "the_person"))),
                stimulus=str(person.get("stimulus", question))[:800],
                context=person, llm=llm, n_hypotheses=n_h, samples_per_hypothesis=n_s,
                seed=seed, as_of=now, structural_frame=frame)
            model_artifacts[cand.model_id] = artifact
            model_dists[cand.model_id] = dict(artifact["raw_qualitative_simulation_distribution"])
            cand.pilot_status = "completed"
            cand.promotion_status = "promoted"
            cand.promotion_reason = "personal-scale frame: full sample budget"
            cand.final_particles = n_h * n_s
            ens.pilot_models.append(cand.model_id)
            ens.full_models.append(cand.model_id)
            ens.simulation_manifest[cand.model_id] = {
                "pilot_particles": 0, "final_particles": n_h * n_s,
                "full_budget_required": n_h * n_s, "pilot_reused_as_prefix": False,
                "status": "completed"}
        classification = classify_forecast_sensitivity(
            model_dists, underidentified=ens.structurally_underidentified)
        mixture = _equal_weight_mixture(model_dists)
        promoted = ens.promoted()
        reversal = _reversal_conditions(ens, promoted, model_dists, mixture)
        voi = _structural_value_of_information(ens, promoted, model_dists)
        ens.cost_manifest = ledger.as_dict()
        fallbacks = sum(int(a.get("n_excluded_numeric_fallbacks", 0)) for a in model_artifacts.values())
        # §33: personal-reaction results carry the SAME §35.2 family/cognition reporting as
        # every route — monoculture is never silently unreported at the personal scale
        try:
            from swm.world_model_v2.model_families import default_family_pool
            _pool = default_family_pool(llm)
            fam_report = _pool.report()
        except Exception:  # noqa: BLE001
            fam_report = {"model_family_monoculture": True, "families": [],
                          "monoculture_note": "family pool unavailable — single injected backend"}
        _cog_samples = [s.get("cognition") for a in model_artifacts.values()
                        for s in (a.get("samples") or []) if s.get("cognition")][:8]
        _nonresp = {}
        for a in model_artifacts.values():
            for k, v in (a.get("nonresponse_breakdown") or {}).items():
                _nonresp[k] = _nonresp.get(k, 0) + int(v or 0)
        cog_report = {
            "pipeline": "stimulus availability → attention → working_memory → memory_retrieval "
                        "→ interpretation → limited_action_search → one_choice → memory_update",
            "schema": "bounded.cognition.v1",
            "sample_decision_records": _cog_samples,
            "nonresponse_breakdown": _nonresp,
            "model_family_monoculture": fam_report.get("model_family_monoculture", True)}
        limitations = ["reaction distributions are counted from qualitative simulations per structural "
                       "frame and are unvalidated (no fitted calibrator)"]
        if classification["classification"] == "materially_structurally_sensitive":
            limitations.append("the predicted reaction depends materially on which causal frame of the "
                               "relationship/situation is true — see structural_ensemble")
        return SimulationResult(
            question=question, simulation_status="completed", support_grade="exploratory",
            raw_distribution=mixture,
            structural_disagreement=dict(model_dists),
            uncertainty_decomposition=decompose_uncertainty(model_dists),
            limitations=limitations,
            cognition_report=cog_report,
            model_family_report=fam_report,
            interpretation_hypotheses=[{"model_id": c.model_id, "thesis": c.causal_thesis}
                                       for c in promoted],
            structural_ensemble={
                "ensemble_id": ens.ensemble_id, "structural_mode": "ensemble",
                "route": "individual_reaction_ensemble",
                "generation_policy": ens.generation_policy,
                "n_independent_generation_calls": ens.independent_generation_calls(),
                "n_merged": ens.candidates_merged,
                "models": [{"model_id": c.model_id, "generation_role": c.generation_role,
                            "causal_thesis": c.causal_thesis,
                            "decisive_actors": c.decisive_actors,
                            "decisive_constraints": c.decisive_constraints,
                            "promotion_status": c.promotion_status,
                            "final_particles": c.final_particles,
                            "prediction": model_dists.get(c.model_id)}
                           for c in ens.surviving()],
                "model_distributions": model_dists,
                "equal_weight_mixture": mixture,
                "robust_range": _robust_range(model_dists),
                "aggregation_method": ("single_surviving_model" if len(promoted) == 1
                                       else "equal_weight_uncalibrated_structural_average"),
                "structural_sensitivity": classification,
                "reversal_conditions": reversal,
                "structural_value_of_information": voi,
                "convergence_certificate": ens.convergence_certificate,
                "stopping_reason": ens.stopping_reason,
                "generation_manifest": ens.generation_manifest,
                "critic_manifest": ens.critic_manifest,
                "merge_manifest": ens.merge_manifest,
                "simulation_manifest": ens.simulation_manifest,
                "cost_manifest": ens.cost_manifest,
                "per_frame_artifacts": model_artifacts,
                "human_summary": _human_summary(question, mixture, classification, reversal, voi,
                                                promoted)},
            provenance={"runtime": U.RUNTIME_VERSION, "structural_mode": "ensemble",
                        "route": "individual_reaction_ensemble",
                        "ensemble_id": ens.ensemble_id,
                        "actor_policy_report": {
                            "requested_actor_policy_mode": "hybrid_relevant_actor_policy",
                            "actual_actor_policy_mode": "persistent_qualitative_llm_policy",
                            "route": "individual_reaction_ensemble",
                            "reason": "reaction_is_the_question",
                            "degraded": False, "construction_error": "",
                            "fallbacks": fallbacks}},
            cost_usd=0.0, latency_s=round(_time.time() - t0, 3))
    except (CompilerExecutionError, EnsembleIntegrityError) as e:
        return SimulationResult(
            question=question, simulation_status="execution_failed",
            failure_taxonomy=getattr(e, "taxonomy", "invalid_execution_plan"),
            limitations=[f"individual-reaction ensemble failed loudly: {type(e).__name__}: {e}"[:220]],
            provenance={"runtime": U.RUNTIME_VERSION, "structural_mode": "ensemble",
                        "route": "individual_reaction_ensemble"},
            latency_s=round(_time.time() - t0, 3))
    except Exception as e:  # noqa: BLE001 — LOUD failure, never a silent fall-through
        return SimulationResult(
            question=question, simulation_status="execution_failed",
            failure_taxonomy="runtime_exception",
            limitations=[f"individual-reaction ensemble failed: {type(e).__name__}: {e}"[:220]],
            provenance={"runtime": U.RUNTIME_VERSION, "structural_mode": "ensemble",
                        "route": "individual_reaction_ensemble"},
            latency_s=round(_time.time() - t0, 3))
