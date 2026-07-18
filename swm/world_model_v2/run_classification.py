"""Run self-classification + the product-facing epistemic contract (audited port of PR #114's
run_classification, re-sourced onto the generated-world report surfaces).

Honest labeling, never abstention: the strongest responsible output is still returned — only
its CLASS changes. Every silent-downgrade channel (evidence skipped, posterior unconsumed,
actor budgets exhausted, mechanisms missing, schema recovery, numeric substitution) becomes a
first-class label the serving layer can show.

Classes:
    full_numeric_forecast        evidence ran AND the posterior was consumed; magnitudes usable
    rank_only                    ordering supported; magnitudes broad-prior driven
    scenario_distribution        no admissible evidence conditioned the run — a structured
                                 scenario weighting, not a forecast
    structurally_underidentified competing causal structures dominate, load-bearing mechanisms
                                 or the scenario schema itself are missing
    execution_failed             technical failure; no output class
"""
from __future__ import annotations

RUN_CLASS_VERSION = "run-class-1.0"
RUN_CLASSES = ("full_numeric_forecast", "rank_only", "scenario_distribution",
               "structurally_underidentified", "execution_failed")


def classify_run(res) -> dict:
    """SimulationResult -> {run_class, version, reasons[]}. Reads only fields the current
    result/provenance contract already carries."""
    reasons = []
    prov = getattr(res, "provenance", None) or {}
    if not res.has_forecast():
        return {"run_class": "execution_failed", "version": RUN_CLASS_VERSION,
                "reasons": [f"simulation_status={getattr(res, 'simulation_status', '?')}",
                            f"failure_taxonomy={getattr(res, 'failure_taxonomy', '')}"]}
    manifest = prov.get("active_component_manifest") or {}
    evidence_ran = bool((manifest.get("phase2_evidence") or {}).get("executed"))
    posterior_consumed = bool((manifest.get("phase3_posterior") or {}).get("executed")) \
        and bool(prov.get("posterior_consumed"))
    crep = prov.get("consequence_report") or {}
    if crep.get("structurally_underidentified") or (crep.get("scenario_schema_error")
                                                    and not crep.get("scenario_schema_id")):
        reasons.append("scenario schema unavailable — semantic consequences unmodeled "
                       f"({str(crep.get('scenario_schema_error', ''))[:120]})")
        return {"run_class": "structurally_underidentified",
                "version": RUN_CLASS_VERSION, "reasons": reasons}
    disagreement = getattr(res, "structural_disagreement", None) or {}
    n_struct = sum(1 for v in disagreement.values()
                   if isinstance(v, (int, float)) and v > 0.05)
    mech_missing = [f for f in (getattr(res, "fallbacks_used", None) or [])
                    if isinstance(f, dict) and int(f.get("tier", 0) or 0) >= 7]
    n_fail = len(prov.get("phase_integration_failures") or [])
    if n_fail:
        reasons.append(f"{n_fail} phase integration failure(s) recorded")
    if n_struct >= 3 and not posterior_consumed:
        reasons.append(f"{n_struct} structural hypotheses disagree materially and no "
                       f"evidence posterior arbitrates them")
        return {"run_class": "structurally_underidentified",
                "version": RUN_CLASS_VERSION, "reasons": reasons}
    if len(mech_missing) >= 2 and not posterior_consumed:
        reasons.append(f"{len(mech_missing)} load-bearing mechanisms served by broad-prior "
                       f"fallbacks with no posterior correction")
        return {"run_class": "structurally_underidentified",
                "version": RUN_CLASS_VERSION, "reasons": reasons}
    if not evidence_ran:
        reasons.append("no admissible evidence conditioned this run: "
                       + str((manifest.get("phase2_evidence") or {}).get("reason", ""))[:140])
        return {"run_class": "scenario_distribution", "version": RUN_CLASS_VERSION,
                "reasons": reasons}
    if not posterior_consumed:
        reasons.append("evidence retrieved but the outcome posterior was not consumed — "
                       "magnitudes are prior-driven; ordering is supported")
        return {"run_class": "rank_only", "version": RUN_CLASS_VERSION, "reasons": reasons}
    grade = str(getattr(res, "support_grade", "") or "")
    if grade in ("exploratory", "highly_speculative"):
        reasons.append(f"support_grade={grade}: treat magnitudes with corresponding width")
    return {"run_class": "full_numeric_forecast", "version": RUN_CLASS_VERSION,
            "reasons": reasons}


def collect_generated_manifests(branches) -> dict:
    """Aggregate the per-branch control-plane bookkeeping (cascade manifests, approximation
    stamps, tier promotions, semantic-event counts) into run provenance."""
    cascade = {"scheduled": 0, "max_depth_reached": 0, "suppressed_duplicate": 0,
               "suppressed_budget": 0, "suppressed_depth": 0, "suppressed_unobserved": 0,
               "quiescence_reasons": {}}
    approximations, promotions, n_events = [], [], 0
    for b in branches or []:
        world = getattr(b, "world", b)
        meta = getattr(world, "uncertainty_meta", None) or {}
        ec = meta.get("event_cascade") or {}
        for k in ("scheduled", "suppressed_duplicate", "suppressed_budget",
                  "suppressed_depth", "suppressed_unobserved"):
            cascade[k] += int(ec.get(k, 0) or 0)
        cascade["max_depth_reached"] = max(cascade["max_depth_reached"],
                                           int(ec.get("max_depth_reached", 0) or 0))
        q = str(ec.get("quiescence", "") or "")
        if q:
            cascade["quiescence_reasons"][q] = cascade["quiescence_reasons"].get(q, 0) + 1
        approximations.extend(meta.get("approximation_manifest") or [])
        promotions.extend(meta.get("actor_tier_promotions") or [])
        n_events += len(getattr(world, "semantic_log", []) or [])
    return {"event_cascade": cascade,
            "approximation_manifest": approximations[:200],
            "n_approximations": len(approximations),
            "actor_tier_promotions": promotions[:50],
            "n_semantic_events": n_events}


def epistemic_contract(res) -> dict:
    """The product-facing honesty surface: what actually ran, what substituted for what, and
    what kind of number (if any) the caller is holding."""
    prov = getattr(res, "provenance", None) or {}
    rc = prov.get("run_classification") or classify_run(res)
    crep = prov.get("consequence_report") or {}
    arep = prov.get("actor_policy_report") or {}
    dists = prov.get("actor_decision_distributions") or {}
    n_qual = sum(int(row.get("n_qualitative_branches", 0) or 0)
                 for row in dists.values() if isinstance(row, dict))
    n_fallback = sum(int(row.get("n_excluded_numeric_fallbacks", 0) or 0)
                     for row in dists.values() if isinstance(row, dict))
    n_reconsider = int(crep.get("actors_invoked", 0) or 0)
    depth = int(crep.get("recursive_cascade_depth", 0) or 0)
    manifests = prov.get("generated_manifests") or {}
    if n_qual > 0 and n_reconsider > 0 and depth >= 2:
        actor_simulation = "full_recursive_actor_simulation"
    elif n_qual > 0 and n_reconsider > 0:
        actor_simulation = "one_hop_actor_simulation"
    elif n_qual > 0:
        actor_simulation = "actor_decisions_without_cascade"
    elif dists or n_fallback > 0:
        actor_simulation = "numeric_actor_fallback"
    else:
        actor_simulation = "no_actor_decisions_in_this_question"
    unmodeled_novel = sum(
        1 for row in dists.values() if isinstance(row, dict)
        for r in (row.get("rows") or []) if r.get("novel_action_unmodeled")) + sum(
        1 for fr in (crep.get("fallback_reasons") or []) if isinstance(fr, dict)
        and fr.get("kind") in ("unmodeled_action_scaffolding", "action_semantics_unmodeled"))
    manifest = prov.get("active_component_manifest") or {}
    return {
        "run_class": rc.get("run_class"),
        "run_class_reasons": rc.get("reasons", []),
        "actor_simulation": actor_simulation,
        "n_actor_reconsiderations": n_reconsider,
        "max_cascade_depth": depth,
        "n_numeric_actor_fallbacks": n_fallback,
        "tier1_numeric_fallbacks": int(crep.get("tier1_numeric_fallbacks",
                                                arep.get("tier1_numeric_fallbacks", 0)) or 0),
        "tier2_numeric_fallbacks": int(crep.get("tier2_numeric_fallbacks",
                                                arep.get("tier2_numeric_fallbacks", 0)) or 0),
        "human_reactions_written_directly": int(
            crep.get("human_reactions_written_directly", 0) or 0),
        "fixed_ontology_uses": int(crep.get("fixed_ontology_uses", 0) or 0),
        "legacy_scalar_writes": int(crep.get("legacy_scalar_writes", 0) or 0),
        "llm_actor_runtime_fallback": bool(arep.get("degraded"))
        or bool(arep.get("construction_error")),
        "evidence_retrieval_degraded": not bool(
            (manifest.get("phase2_evidence") or {}).get("executed")),
        "scenario_schema": {"id": crep.get("scenario_schema_id", ""),
                            "version": crep.get("scenario_schema_version", ""),
                            "recovery": prov.get("scenario_schema_recovery", "")},
        "n_approximations_stamped": int(manifests.get("n_approximations", 0) or 0),
        "unmodeled_novel_actions": unmodeled_novel,
        "calibration_available": getattr(res, "calibrated_probability", None) is not None,
        "scenario_only": rc.get("run_class") in ("scenario_distribution",
                                                 "structurally_underidentified"),
        "support_grade": getattr(res, "support_grade", ""),
    }
