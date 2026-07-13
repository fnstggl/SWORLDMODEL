"""Ruthless nonlinearity audit — Phase 7, Part 0.

Scans every Phase-6 mechanism family (from the committed registry) and records, per family, whether the
current form is linear by assumption, which nonlinear phenomena are *possible* for its causal process, which
are *already modeled*, and a disposition (retain_linear / test_nonlinear / extend / split / quarantine /
reject). It is data-driven where it can be (status, code_ref, packs, ontology come straight from the
registry) and curated where judgment is needed (which phenomena a given causal process could plausibly show),
with the curation stated openly rather than hidden.

The output is the machine-readable `wmv2_phase7_audit.json`; the four docs summarize it, they do not restate
every row (Part 30).
"""
from __future__ import annotations

# curated per-family nonlinear assessment (from the Part-0 registry read; open, not hidden).
# fields: current_form, phenomena_possible, phenomena_modeled, disposition, priority, note
_CURATED = {
    "simple_contagion_hazard": dict(current_form="linear λ=q·k", assumes_linear=True,
        phenomena_possible=["saturation", "threshold", "reinforcement", "finite_population", "self_excitation"],
        phenomena_modeled=[], disposition="retain_linear_as_comparator", priority="high",
        note="the linear form Higgs proved WRONG; kept only as the baseline nonlinear forms must beat"),
    "complex_contagion_hazard": dict(current_form="Hill λ=exp(θ0)kᵅ/(cᵅ+kᵅ)", assumes_linear=False,
        phenomena_possible=["saturation", "threshold", "reinforcement", "tipping"],
        phenomena_modeled=["saturation", "threshold"], disposition="extend", priority="high",
        note="genuine nonlinear (α>1 superlinear onset + saturation); already implemented"),
    "exposure_response_hazard": dict(current_form="log-linear λ=exp(θ·x), x has log1p(k), k/deg",
        assumes_linear=False, phenomena_possible=["saturation", "interaction", "recency", "heterogeneity"],
        phenomena_modeled=["saturation", "interaction", "recency"], disposition="extend", priority="high",
        note="the validated winner (locally_validated); the ONE interaction (k/deg) in the whole registry"),
    "hawkes_self_excitation": dict(current_form="Hawkes μ+αωΣe^{−ω(t−tᵢ)}", assumes_linear=False,
        phenomena_possible=["self_excitation", "tipping", "recency"],
        phenomena_modeled=["self_excitation"], disposition="quarantine_preserved", priority="high",
        note="QUARANTINED — held-out MAE 1098.9 > Poisson 973.0 on Higgs; must NOT auto-promote"),
    "finite_population_saturation": dict(current_form="P=E_ε[1−exp(−ελW)]", assumes_linear=False,
        phenomena_possible=["saturation", "finite_population", "heterogeneity"],
        phenomena_modeled=["saturation", "finite_population", "heterogeneity"], disposition="extend",
        priority="medium", note="finite-pop saturation with frailty"),
    "bass_diffusion": dict(current_form="dN/dt=(p+q·N/M)(M−N)", assumes_linear=False,
        phenomena_possible=["saturation", "tipping", "finite_population"],
        phenomena_modeled=["saturation", "finite_population"], disposition="extend", priority="high",
        note="logistic-type adoption saturation; testbed = baby-name cultural adoption"),
    "attrition_dropout_hazard": dict(current_form="σ(b+Σw·z) additive logistic", assumes_linear=True,
        phenomena_possible=["saturation", "threshold", "interaction", "path_dependence"],
        phenomena_modeled=[], disposition="test_nonlinear", priority="high",
        note="tenure→churn is famously nonlinear (declining hazard); interaction contract×tenure. Testbed=telco"),
    "response_occurrence_hazard": dict(current_form="σ(b+Σw·z)", assumes_linear=True,
        phenomena_possible=["saturation", "interaction"], phenomena_modeled=[],
        disposition="test_nonlinear_expect_null", priority="medium",
        note="Phase-6 NULL; adversarial test that Phase 7 also honestly finds null. Testbed=stackexchange"),
    "argument_persuasion_success": dict(current_form="σ(b+Σw·z)", assumes_linear=True,
        phenomena_possible=["interaction", "backfire", "saturation"], phenomena_modeled=[],
        disposition="test_nonlinear_expect_null", priority="medium",
        note="Phase-6 NULL; test prior×argument interaction + guard against unsupported backfire. Testbed=CMV"),
    "content_response_click": dict(current_form="linear CTR + argmax winner-share", assumes_linear=True,
        phenomena_possible=["saturation", "inverted_u", "interaction", "heterogeneity"],
        phenomena_modeled=["heterogeneity"], disposition="test_nonlinear", priority="high",
        note="headline→CTR saturation + partial pooling across randomized A/B tests. Testbed=Upworthy"),
    "trust_formation": dict(current_form="piecewise-linear asymmetric gain/loss", assumes_linear=True,
        phenomena_possible=["hysteresis", "path_dependence", "saturation", "threshold"],
        phenomena_modeled=["path_dependence"], disposition="test_nonlinear", priority="medium",
        note="asymmetric gain/loss is proto-hysteresis; a true bistable band is unmodeled"),
    "voting_turnout": dict(current_form="σ(logit(base)+Σ coeffs)", assumes_linear=True,
        phenomena_possible=["threshold", "saturation", "fatigue", "interaction"], phenomena_modeled=[],
        disposition="test_nonlinear", priority="medium",
        note="social-pressure thresholds + repeated-contact fatigue"),
    "reinforcement_learning": dict(current_form="Q←Q+α(r−Q) linear", assumes_linear=True,
        phenomena_possible=["saturation", "regime", "habituation"], phenomena_modeled=[],
        disposition="test_nonlinear", priority="low", note="reinforcement saturation / habit"),
    "quantal_response_choice": dict(current_form="softmax p∝exp(λu)", assumes_linear=False,
        phenomena_possible=["regime", "heterogeneity"], phenomena_modeled=["saturation"],
        disposition="extend", priority="low", note="already nonlinear (softmax); heterogeneous rationality"),
    "position_bias_propensity": dict(current_form="(1/rank)^η power law", assumes_linear=False,
        phenomena_possible=["saturation", "interaction"], phenomena_modeled=["saturation"],
        disposition="retain", priority="low", note="already nonlinear power law"),
    "weak_tie_transmission": dict(current_form="inverted-U Gaussian bump", assumes_linear=False,
        phenomena_possible=["inverted_u", "backfire"], phenomena_modeled=["inverted_u"],
        disposition="structural_candidate", priority="low",
        note="non-monotone shape (research_encoded); magnitude broad — needs data to validate the U"),
}

# phenomena a causal ontology could plausibly show (fallback when a family is not individually curated)
_ONTOLOGY_PHENOMENA = {
    "diffusion": ["saturation", "threshold", "reinforcement", "finite_population", "self_excitation",
                  "recency", "tipping"],
    "attention": ["saturation", "fatigue", "habituation", "recency", "inverted_u"],
    "belief": ["interaction", "backfire", "saturation"],
    "relationship": ["hysteresis", "path_dependence", "saturation", "threshold"],
    "participation": ["threshold", "saturation", "fatigue", "interaction"],
    "decision": ["regime", "heterogeneity", "saturation"],
    "learning": ["saturation", "habituation", "regime"],
    "platform": ["saturation", "inverted_u", "interaction", "heterogeneity"],
    "memory": ["fatigue", "habituation", "recency"],
    "influence": ["threshold", "tipping", "path_dependence"],
    "network": ["inverted_u", "interaction", "heterogeneity"],
    "norm": ["threshold", "path_dependence"],
    "bargaining": ["saturation", "threshold"],
    "coalition": ["threshold", "tipping"],
    "resource": ["saturation", "threshold", "path_dependence"],
    "institutional": ["threshold", "regime"],
}


def run_audit() -> dict:
    """Produce the machine-readable Part-0 audit over the committed Phase-6 registry."""
    from swm.world_model_v2.registry.store import RegistryStore
    reg = RegistryStore.load()
    families = []
    disp_counts = {}
    for fid in sorted(reg.records):
        rec = reg.records[fid]
        cur = _CURATED.get(fid)
        onto = getattr(rec, "ontology_type", "")
        if cur:
            row = dict(cur)
        else:
            poss = _ONTOLOGY_PHENOMENA.get(onto, [])
            row = dict(current_form=(rec.formal_description or "")[:80], assumes_linear=None,
                       phenomena_possible=poss, phenomena_modeled=[],
                       disposition="test_nonlinear" if poss else "retain_linear", priority="low",
                       note="not individually curated — ontology-derived candidate phenomena")
        row.update({"family_id": fid, "ontology_type": onto, "status": rec.status,
                    "code_ref": rec.code_ref, "n_packs": len(rec.packs),
                    "has_validation": rec.has_validation(),
                    "answers_processes": list(getattr(rec.applicability, "answers_processes", []) or [])})
        # gap = possible phenomena not yet modeled
        row["nonlinear_gap"] = [p for p in row["phenomena_possible"] if p not in row["phenomena_modeled"]]
        families.append(row)
        disp_counts[row["disposition"]] = disp_counts.get(row["disposition"], 0) + 1
    # the three globally under-served phenomena (from the audit)
    underserved = ["interaction", "fatigue", "hysteresis"]
    served = {}
    for row in families:
        for p in row["phenomena_modeled"]:
            served[p] = served.get(p, 0) + 1
    return {"_meta": {"n_families": len(families),
                      "note": "Part-0 nonlinearity audit; dispositions curated openly, registry facts "
                              "data-driven. Under-served phenomena drive the Phase-7 build."},
            "disposition_counts": disp_counts,
            "phenomena_modeled_counts": served,
            "under_served_phenomena": underserved,
            "families": families}
