"""Forecast availability is separate from grounding quality — the layered probability contract.

For every coherent binary question the runtime returns its best DEFENSIBLE probability even when
evidence is incomplete, mechanisms are partially ungrounded, structural models disagree, rollout
mass stays unresolved, or the run classifies itself under-modeled. The execution status DESCRIBES
the forecast; it never erases it. What changes with weak support is the LABEL and the WIDTH:

    probability                     the headline best estimate (never an invented 0.5)
    probability_conditional_on_resolved   resolved-mass-only readout (weights preserved)
    unresolved_mass                 disclosed, never silently renormalized away
    probability_source              completed_rollouts | partial_rollouts |
                                    grounded_reference_prior | evidence_conditioned_prior |
                                    exploratory_model_estimate      (preferred in that order:
                                    simulation before priors; within priors the more-informed
                                    serves — an evidence-updated posterior subsumes its own
                                    grounded reference prior, both values are recorded)
    grounding_grade                 grounded | partially_grounded | exploratory | ungrounded
    confidence                      qualitative label derived from grade + mass + interval
    uncertainty_interval            worst/best-case bounds (unresolved mass swings fully)
    weight_sensitive                True when plausible weight changes cross 0.5

p = None survives ONLY when the question is malformed/non-probabilistic or absolutely no
defensible source exists: zero resolved mass, zero effective evidence, no grounded prior, and no
directional model estimate. A neutral no-information default is NOT a source — nothing here can
produce an automatic 0.5."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

RECOVERY_VERSION = "forecast-recovery-1.0"

GROUNDING_GRADES = ("grounded", "partially_grounded", "exploratory", "ungrounded")
PROBABILITY_SOURCES = ("completed_rollouts", "partial_rollouts", "grounded_reference_prior",
                       "evidence_conditioned_prior", "exploratory_model_estimate")
#: keys of a terminal distribution that are NOT resolved option mass
NON_OPTION_KEYS = ("unresolved_mechanism", "unresolved", "none_by_horizon_unresolved",
                   "truncated", "None", "no_choice")


@dataclass
class ForecastRecovery:
    probability: float = None
    probability_conditional_on_resolved: float = None
    unresolved_mass: float = 0.0
    probability_source: str = ""
    grounding_grade: str = ""
    confidence: str = ""
    uncertainty_interval: tuple = None            # (lower, upper) on P(yes)
    weight_sensitive: bool = False
    components: dict = field(default_factory=dict)  # every input that fed the estimate
    limitations: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _yes_no_mass(distribution: dict, options) -> tuple:
    """(yes_mass, resolved_mass, total_mass) with the SAME yes-key convention as the binary
    projection. Option masses keep their existing branch weights — nothing is re-weighted."""
    if not distribution:
        return 0.0, 0.0, 0.0
    yes_keys = [str(o) for o in (options or [])][:1] + ["True", "true", "yes", "Yes", "1"]
    yes_key = next((k for k in yes_keys if k in distribution), None)
    resolved = {k: float(v) for k, v in distribution.items()
                if k not in NON_OPTION_KEYS and isinstance(v, (int, float))}
    yes = float(resolved.get(yes_key, 0.0)) if yes_key is not None else None
    total = sum(float(v) for v in distribution.values() if isinstance(v, (int, float)))
    if yes is None and resolved:
        # no recognizable yes key: binary recovery is not defensible from this distribution
        return None, sum(resolved.values()), total
    return yes or 0.0, sum(resolved.values()), total


def recover_forecast(*, distribution: dict = None, options=None, unresolved_mass: float = None,
                     posterior_mean: float = None, posterior_n_eff: int = 0,
                     prior_mean: float = None, prior_source_class: str = "",
                     evidence_starved: bool = None) -> ForecastRecovery | None:
    """The layered recovery. Returns None ONLY when no defensible source exists."""
    rec = ForecastRecovery()
    yes, resolved, total = _yes_no_mass(distribution or {}, options)
    if unresolved_mass is None:
        unresolved_mass = max(0.0, (total or 0.0) - (resolved or 0.0))
    if total and total > 0:
        unresolved_share = min(1.0, max(0.0, unresolved_mass / total if unresolved_mass <= total
                                        else unresolved_mass))
        resolved_share = min(1.0, (resolved or 0.0) / total)
    else:
        unresolved_share, resolved_share = (1.0 if unresolved_mass else 0.0), 0.0
    rec.unresolved_mass = round(unresolved_share, 4)
    # the more-informed prior serves; both are recorded
    prior_grounded = str(prior_source_class or "").lower() not in ("", "lean", "llm_estimated",
                                                                   "generic", "none")
    if posterior_n_eff and posterior_mean is not None:
        prior_component, prior_src = float(posterior_mean), "evidence_conditioned_prior"
    elif prior_mean is not None and prior_grounded:
        prior_component, prior_src = float(prior_mean), "grounded_reference_prior"
    elif prior_mean is not None:
        prior_component, prior_src = float(prior_mean), "exploratory_model_estimate"
    else:
        prior_component, prior_src = None, ""
    rec.components = {"yes_mass": None if yes is None else round(yes, 4),
                      "resolved_mass": round(resolved or 0.0, 4),
                      "total_mass": round(total or 0.0, 4),
                      "posterior_mean": posterior_mean, "posterior_n_eff": posterior_n_eff,
                      "prior_mean": prior_mean, "prior_source_class": prior_source_class,
                      "prior_component_used": prior_component,
                      "prior_component_source": prior_src, "version": RECOVERY_VERSION}

    have_resolved = yes is not None and (resolved or 0.0) > 1e-9
    if have_resolved:
        p_cond = yes / resolved
        rec.probability_conditional_on_resolved = round(p_cond, 4)
        # worst/best case: the unresolved share swings entirely against/for yes
        lo = resolved_share * p_cond
        hi = resolved_share * p_cond + unresolved_share
        rec.uncertainty_interval = (round(lo, 4), round(min(1.0, hi), 4))
        if unresolved_share <= 0.001:
            rec.probability = round(p_cond, 4)
            rec.probability_source = "completed_rollouts"
            rec.grounding_grade = "grounded" if (posterior_n_eff and not evidence_starved) \
                else "partially_grounded"
        else:
            rec.probability_source = "partial_rollouts"
            if prior_component is not None:
                # the explicit unresolved-mass treatment: resolved mass keeps its simulated
                # frequency AND weights; unresolved mass takes the best prior — disclosed
                rec.probability = round(resolved_share * p_cond
                                        + unresolved_share * prior_component, 4)
                rec.limitations.append(
                    f"partial rollouts: {resolved_share:.0%} resolved mass simulated "
                    f"(conditional P={p_cond:.3f}); {unresolved_share:.0%} unresolved mass "
                    f"takes the {prior_src} ({prior_component:.3f}) — both disclosed, nothing "
                    f"renormalized away")
            else:
                rec.probability = round(p_cond, 4)
                rec.limitations.append(
                    f"partial rollouts with NO prior available: the headline conditions on the "
                    f"{resolved_share:.0%} resolved mass; {unresolved_share:.0%} unresolved "
                    f"mass is disclosed and bounds the interval")
            rec.grounding_grade = "partially_grounded" if prior_component is not None \
                else "exploratory"
        rec.weight_sensitive = rec.uncertainty_interval[0] < 0.5 < rec.uncertainty_interval[1]
    elif prior_component is not None:
        rec.probability = round(prior_component, 4)
        rec.probability_source = prior_src
        rec.grounding_grade = {"evidence_conditioned_prior": "exploratory",
                               "grounded_reference_prior": "exploratory",
                               "exploratory_model_estimate": "ungrounded"}[prior_src]
        rec.uncertainty_interval = (0.0, 1.0) if unresolved_share >= 0.999 else None
        rec.weight_sensitive = True
        rec.limitations.append(
            f"no resolved rollout mass: the headline is the {prior_src} "
            f"({prior_component:.3f}) — simulated-world confirmation is absent; treat as "
            f"{rec.grounding_grade}")
    else:
        return None                              # absolutely no defensible source — honest None
    if rec.probability is not None:
        rec.probability = min(1.0, max(0.0, rec.probability))
    rec.confidence = _confidence(rec)
    return rec


def _confidence(rec: ForecastRecovery) -> str:
    """Qualitative label, deterministically derived — a description, never a number."""
    if rec.grounding_grade == "grounded" and rec.unresolved_mass <= 0.05:
        return "moderate"                        # single-run simulation: never above moderate
    if rec.grounding_grade in ("grounded", "partially_grounded") and not rec.weight_sensitive:
        return "low"
    return "very_low"


def attach_recovery(res, rec: ForecastRecovery | None, *, override_probability: bool):
    """Write the recovery onto a SimulationResult. `override_probability=False` only fills the
    probability when the result has none (completed paths keep their existing numbers);
    the labels/intervals/mass fields are always attached. Status is NEVER changed here."""
    if rec is None:
        return res
    if override_probability or res.raw_probability is None:
        if rec.probability is not None:
            res.raw_probability = rec.probability
    res.probability_source = rec.probability_source
    res.grounding_grade = rec.grounding_grade
    res.confidence = rec.confidence
    res.unresolved_mass = rec.unresolved_mass
    res.probability_conditional_on_resolved = rec.probability_conditional_on_resolved
    res.uncertainty_interval = (list(rec.uncertainty_interval)
                                if rec.uncertainty_interval is not None else None)
    res.weight_sensitive = rec.weight_sensitive
    res.limitations = list(res.limitations or []) + rec.limitations[:3]
    res.provenance = {**(res.provenance or {}), "forecast_recovery": rec.as_dict()}
    return res


def plan_prior_inputs(plan) -> dict:
    """The posterior/prior inputs available on a conditioned plan (set by the phase-3 block)."""
    particles = getattr(plan, "posterior_rate_particles", None) or []
    posterior_mean = (sum(particles) / len(particles)) if particles else None
    spec = getattr(plan, "_outcome_prior_spec", None)
    return {"posterior_mean": (round(posterior_mean, 4) if posterior_mean is not None else None),
            "posterior_n_eff": (1 if particles else 0),
            "prior_mean": (round(float(spec.mean), 4) if spec is not None else None),
            "prior_source_class": (str(getattr(spec, "source_class", "")) if spec is not None
                                   else "")}
