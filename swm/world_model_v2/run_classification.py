"""Run self-classification + actor-mediated manifests + the product epistemic contract.

Every run must say WHAT KIND OF OUTPUT it produced — not as a vague support adjective but as a
declared class the product surfaces:

    full_numeric_forecast        numbers carry evidential support (evidence ran, the posterior
                                 was consumed); magnitudes are meant to be used
    rank_only                    ordering/relative comparison is supported; absolute magnitudes
                                 are broad-prior driven — do not read precision into them
    scenario_distribution        no admissible evidence conditioned the run: the distribution is
                                 a structured scenario weighting, not a numeric forecast
    structurally_underidentified the causal structure itself is not pinned down (competing
                                 structures dominate, mechanisms missing/rejected)
    execution_failed             a technical failure — no output class applies

Classification is honest labeling, never abstention: the strongest responsible output is still
returned; only its CLASS changes. The epistemic contract additionally states, in product-facing
terms, whether full recursive actor simulation ran, whether one-hop or numeric actor fallback
served, whether evidence retrieval degraded, which aggregate processes substituted for explicit
actors, whether unmodeled novel actions occurred, and whether calibration was available.
"""
from __future__ import annotations

RUN_CLASSES = ("full_numeric_forecast", "rank_only", "scenario_distribution",
               "structurally_underidentified", "execution_failed")
RUN_CLASSIFIER_VERSION = "run-class-1.0"


def collect_actor_mediated_manifests(branches) -> dict:
    """Aggregate the per-branch actor-mediated bookkeeping (event cascade, approximation
    stamps, demoted scalar writes, semantic events, tier promotions) for result provenance."""
    from swm.world_model_v2.actor_propagation import cascade_manifest
    worlds = [b.world for b in (branches or []) if getattr(b, "world", None) is not None]
    approximations, demoted, promotions, sem_events = [], [], [], 0
    for w in worlds:
        meta = getattr(w, "uncertainty_meta", None) or {}
        approximations += list(meta.get("approximation_manifest") or [])
        demoted += list(meta.get("demoted_scalar_writes") or [])
        promotions += list(meta.get("actor_tier_promotions") or [])
        sem_events += len(meta.get("semantic_events") or [])
    return {
        "event_cascade": cascade_manifest(worlds),
        "approximation_manifest": approximations[:200],
        "n_approximations": len(approximations),
        "demoted_scalar_writes": {"n": len(demoted),
                                  "kinds": sorted({d.get("kind", "") for d in demoted})},
        "actor_tier_promotions": promotions[:50],
        "n_semantic_events": sem_events,
    }


def classify_run(res, *, manifest: dict | None = None) -> dict:
    """One declared run class + machine-checkable reasons. `res` is a SimulationResult;
    `manifest` the active-component manifest when available."""
    reasons = []
    if not res.has_forecast():
        return {"run_class": "execution_failed", "version": RUN_CLASSIFIER_VERSION,
                "reasons": [f"simulation_status={res.simulation_status}",
                            f"failure_taxonomy={res.failure_taxonomy or 'n/a'}"]}
    manifest = manifest or (res.provenance or {}).get("active_component_manifest") or {}
    ev = manifest.get("phase2_evidence") or {}
    post = manifest.get("phase3_posterior") or {}
    evidence_ran = bool(ev.get("executed"))
    posterior_consumed = bool(post.get("executed")) and \
        bool((res.provenance or {}).get("posterior_consumed"))
    struct = res.structural_disagreement or {}
    n_struct = len([k for k, v in struct.items() if isinstance(v, (int, float)) and v > 0.05])
    mech_missing = [f for f in (res.fallbacks_used or [])
                    if isinstance(f, dict) and int(f.get("tier", 0) or 0) >= 7]
    integration_failures = (res.provenance or {}).get("phase_integration_failures") or []

    if n_struct >= 3 and not posterior_consumed:
        reasons.append(f"{n_struct} live structural hypotheses with no evidence-updated "
                       "posterior to weigh them")
        cls = "structurally_underidentified"
    elif len(mech_missing) >= 2 and not posterior_consumed:
        reasons.append(f"{len(mech_missing)} load-bearing tier-7 mechanism fallbacks")
        cls = "structurally_underidentified"
    elif not evidence_ran:
        reasons.append(f"evidence phase did not execute ({ev.get('reason', 'no as_of / error')})")
        cls = "scenario_distribution"
    elif not posterior_consumed:
        reasons.append("evidence ran but no admissible observation updated the posterior — "
                       "magnitudes are prior-driven; ordering is supported")
        cls = "rank_only"
    else:
        reasons.append("evidence executed and the posterior was consumed by the terminal path")
        cls = "full_numeric_forecast"
        if res.support_grade in ("exploratory", "highly_speculative"):
            reasons.append(f"support_grade={res.support_grade}: numeric class retained but "
                           "support is weak — see epistemic contract")
    if integration_failures:
        reasons.append(f"{len(integration_failures)} phase integration failure(s) recorded")
    return {"run_class": cls, "version": RUN_CLASSIFIER_VERSION, "reasons": reasons}


def epistemic_contract(res) -> dict:
    """The product-facing statement of what actually ran — every silent-downgrade channel made
    visible. Consumed verbatim by the serving layer."""
    prov = res.provenance or {}
    am = prov.get("actor_mediated") or {}
    cascade = (am.get("event_cascade") or {})
    n_reconsider = int(cascade.get("total_reconsiderations", 0) or 0)
    dists = prov.get("actor_decision_distributions") or {}
    n_fallback = sum(int(v.get("n_excluded_numeric_fallbacks", 0) or 0)
                     for v in dists.values() if isinstance(v, dict))
    n_qual = sum(int(v.get("n_qualitative_branches", 0) or 0)
                 for v in dists.values() if isinstance(v, dict))
    depth = max([r.get("max_depth_reached", 0) for r in cascade.get("branches", [])] or [0])
    if n_qual and n_reconsider and depth >= 2:
        actor_sim = "full_recursive_actor_simulation"
    elif n_qual and n_reconsider:
        actor_sim = "one_hop_actor_simulation"
    elif n_qual:
        actor_sim = "actor_decisions_without_cascade"
    elif dists or n_fallback:
        actor_sim = "numeric_actor_fallback"
    else:
        actor_sim = "no_actor_decisions_in_this_question"
    manifest = prov.get("active_component_manifest") or {}
    ev = manifest.get("phase2_evidence") or {}
    novel_unmodeled = [r for v in dists.values() if isinstance(v, dict)
                       for r in (v.get("rows") or []) if r.get("novel_action_unmodeled")]
    return {
        "run_class": (prov.get("run_classification") or {}).get("run_class", ""),
        "actor_simulation": actor_sim,
        "n_actor_reconsiderations": n_reconsider,
        "max_cascade_depth": depth,
        "n_numeric_actor_fallbacks": n_fallback,
        "numeric_actor_fallback_occurred": n_fallback > 0,
        "llm_actor_runtime_fallback": prov.get("actor_runtime_fallback") or [],
        "evidence_retrieval_degraded": (not ev.get("executed", False)) or
                                       "error" in str(ev.get("reason", "")),
        "aggregate_processes_used": (am.get("demoted_scalar_writes") or {}).get("kinds", []) +
                                    [a.get("approximation_type", "")
                                     for a in (am.get("approximation_manifest") or [])[:5]],
        "n_approximations_stamped": int(am.get("n_approximations", 0) or 0),
        "unmodeled_novel_actions": len(novel_unmodeled),
        "convergence": {"n_particles": prov.get("n_particles"),
                        "note": "particle count and terminal spread; see uncertainty_decomposition"},
        "calibration_available": res.calibrated_probability is not None,
        "scenario_only": (prov.get("run_classification") or {}).get("run_class")
                         in ("scenario_distribution", "structurally_underidentified"),
        "support_grade": res.support_grade,
    }
