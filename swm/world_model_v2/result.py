"""Production result semantics — honest epistemic statuses (Part A + §8/§20/§21/§35).

ACCURACY BEFORE APPARENT COMPLETENESS. Weak evidence, missing validated mechanisms, transport risk,
unfamiliar domains and uncertain hidden state do NOT prevent a forecast — they widen priors, add
competing hypotheses, lower the SUPPORT GRADE and add limitations. But apparent completeness is never
bought with silently over-claimed forecasts either: when the represented world demonstrably fails to
contain a high-sensitivity part of the real causal system, or when simulated branch mass was cut off
before causal resolution, the result SAYS SO in a first-class status instead of dressing up as an
ordinary completed forecast. Binary forecast abstention stays replaced by three INDEPENDENT concepts:

  SimulationStatus     — did the simulation run, and how completely does the represented world cover
                         the question?  completed / completed_with_degradation /
                         clarification_required / execution_failed / under_modeled / truncated
                         (+ the legacy alias temporally_truncated)
  SupportGrade         — how well-evidenced is it? empirically_supported / transfer_supported /
                         exploratory / highly_speculative
  RecommendationStatus — may we recommend an action? eligible / limited / withheld / not_requested

`under_modeled` (§35) and `truncated` (§20/§21) are EPISTEMIC/REPRESENTATIONAL states of the result —
NOT engineering exceptions and NOT clarification. under_modeled: the run executed but a
high-sensitivity component of the real causal system is known to be missing, unrepresentable or
unresolvably disputed in the represented world (the UNDER_MODELED_SUBTYPES name what kind; the
under_modeled_components name which pieces). truncated: simulated branch probability mass was cut off
by a budget or failure before resolution, and the truncation_report (§35.4) accounts for that mass —
truncated branch mass is unresolved simulation, not Monte Carlo error. Both statuses MAY carry a
partial exploratory distribution where mathematically defensible, but that distribution remains
EXPLICITLY CONDITIONAL on the modeled portion of the world (conditional_forecast_note rides with it);
it is never presented as an ordinary completed forecast over the full question.

`clarification_required` is reserved for genuinely incoherent questions (no simulable interpretation exists,
even after generating competing ones) and must be RARE. `execution_failed` is an engineering/architectural
failure (exception, unresolved reference, invalid plan, unbindable readout, missing operator) — NOT
epistemic abstention. Neither may be used to hide compiler incompleteness or missing mechanism coverage —
naming those gaps honestly is exactly what under_modeled exists for.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# ------------------------------------------------------------------ the three independent axes
# "truncated" is the FIRST-CLASS truncation status (§20/§21): new code emits it WITH a populated
# truncation_report (see swm/world_model_v2/truncation.py). "temporally_truncated" remains a readable
# LEGACY ALIAS status so historical artifacts and pre-§20 emitters keep loading unchanged; new
# emitters must prefer "truncated".
SIMULATION_STATUSES = ("completed", "completed_with_degradation", "clarification_required",
                       "execution_failed", "temporally_truncated", "under_modeled", "truncated")
SUPPORT_GRADES = ("empirically_supported", "transfer_supported", "exploratory", "highly_speculative")
RECOMMENDATION_STATUSES = ("eligible", "limited", "withheld", "not_requested")

#: §35 under-modeled subtypes — WHAT KIND of representational gap makes a result under_modeled.
#: An under_modeled result must name at least one subtype or one structured component; an unnamed
#: gap is not an honest gap.
UNDER_MODELED_SUBTYPES = (
    "under_modeled_boundary",                 # the world boundary excludes a high-sensitivity region (§35.1)
    "under_modeled_actor",                    # a decisive actor could not be represented
    "under_modeled_population",               # a decisive population/aggregate could not be represented
    "under_modeled_nonhuman_mechanism",       # a physical/algorithmic/biological mechanism is missing (§35.3)
    "under_modeled_external_process",         # an outside-world process drives the outcome (§35.1)
    "under_modeled_parameterization",         # structure exists but no defensible parameterization does
    "under_modeled_structural_disagreement",  # structural models disagree beyond what evidence resolves
)

#: The exact §35 conditionality sentence that rides with any partial under_modeled distribution.
CONDITIONAL_FORECAST_SENTENCE = ("This distribution is conditional on the represented world boundary "
                                 "and excludes unresolved high-sensitivity processes.")

#: engineering failure taxonomy — every execution_failed result names one of these
FAILURE_TAXONOMY = (
    "runtime_exception", "unresolved_reference", "invalid_execution_plan",
    "terminal_readout_unbindable", "missing_required_operator", "corrupt_evidence_artifact",
    "unavailable_service", "serialization_failure", "parser_failure_after_retries", "timeout")


class CompilerExecutionError(Exception):
    """A TECHNICAL compiler/execution failure (not epistemic). Carries a taxonomy code so the result
    contract can classify it as execution_failed rather than a forecast refusal."""
    def __init__(self, message: str, *, taxonomy: str = "runtime_exception"):
        super().__init__(message)
        self.taxonomy = taxonomy if taxonomy in FAILURE_TAXONOMY else "runtime_exception"


class ClarificationRequired(Exception):
    """The question is not coherent enough to define ANY simulable outcome contract, even after generating
    and evaluating competing interpretations. Must be rare; carries the interpretations that were tried."""
    def __init__(self, message: str, *, interpretations_tried: list = None):
        super().__init__(message)
        self.interpretations_tried = interpretations_tried or []


@dataclass
class SimulationResult:
    """The full user-facing result. Every completed/degraded result carries all epistemic fields and a
    forecast; under_modeled/truncated results carry their gap accounting (§35) and MAY carry a partial
    distribution that stays explicitly conditional on the modeled portion of the world."""
    question: str
    simulation_status: str                                  # SIMULATION_STATUSES
    support_grade: str = "exploratory"                      # SUPPORT_GRADES (set for completed/degraded)
    recommendation_status: str = "not_requested"            # RECOMMENDATION_STATUSES
    # forecast
    raw_distribution: dict = field(default_factory=dict)
    calibrated_distribution: dict = None
    raw_probability: float = None                           # binary convenience projection
    calibrated_probability: float = None
    # epistemics
    uncertainty_decomposition: dict = field(default_factory=dict)
    structural_disagreement: dict = None
    mechanism_disagreement: dict = None
    # STRUCTURAL-MODEL ENSEMBLE (level-A uncertainty; default-on). The full machine-readable record of
    # the independently generated causal models: generation/critic/merge manifests, per-model predictions
    # and trajectory summaries, support classes, pilot/full particle counts, aggregation method,
    # structural-sensitivity classification, reversal conditions, structural value-of-information, cost
    # manifest and stopping reason. None only on the explicit single_structural_model ablation and on
    # legacy phase-scoped science routes. Human-facing emphasis order: (1) the answer, (2) whether it
    # survives across models, (3) the strongest competing causal explanation, (4) what assumption
    # reverses the answer, (5) what information would resolve the disagreement.
    structural_ensemble: dict = None
    # Phase 3: evidence-updated posterior over hidden world-state (rate) + structure. Present only when the
    # posterior pipeline ran AND ≥1 effective (dependence-collapsed) observation actually updated it; the
    # decomposition names prior→posterior deltas, ESS, and the per-observation assimilation ledger so a
    # reviewer can trace every terminal-affecting number back to a verified claim. None on the prior-only path.
    posterior_inference: dict = None
    evidence_quality: str = ""
    limitations: list = field(default_factory=list)
    fallbacks_used: list = field(default_factory=list)      # [{process, tier, family, why}]
    mechanism_tiers: dict = field(default_factory=dict)     # {causal_process: tier}
    omitted_high_sensitivity_variables: list = field(default_factory=list)
    sensitivity_contributors: list = field(default_factory=list)
    interpretation_hypotheses: list = field(default_factory=list)
    # §35 honest under-modeling + §20/§21 truncation accounting (all additive; empty on ordinary
    # completed results — populating them never blocks a forecast, hiding them is what is forbidden)
    under_modeled_subtypes: list = field(default_factory=list)     # UNDER_MODELED_SUBTYPES members
    under_modeled_components: list = field(default_factory=list)   # [{component, kind, why, sensitivity}]
    conditional_forecast_note: str = ""                            # rides with any partial distribution
    world_boundaries: dict = field(default_factory=dict)           # per-structural-model §35.1 blocks
    outside_world: dict = field(default_factory=dict)              # §35.1 processes left OUTSIDE the boundary
    cognition_report: dict = field(default_factory=dict)           # §35.2 actor-cognition coverage/fidelity
    hybrid_mechanisms: dict = field(default_factory=dict)          # §35.3 nonhuman/hybrid mechanism coverage
    truncation_report: dict = field(default_factory=dict)          # §35.4 aggregated branch truncation (§21)
    model_family_report: dict = field(default_factory=dict)        # model families used + family failures
    # failure / clarification
    failure_taxonomy: str = ""                              # set iff execution_failed
    clarification_reason: str = ""                          # set iff clarification_required
    # provenance / accounting
    execution_trace_ref: str = ""
    plan_hash: str = ""
    provenance: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def __post_init__(self):
        if self.simulation_status not in SIMULATION_STATUSES:
            raise ValueError(f"bad simulation_status {self.simulation_status!r}")
        # under_modeled/truncated are epistemic states of a RUN result — like completed, they must
        # carry an honest support grade (they are graded claims about the world, not failures)
        if self.simulation_status in ("completed", "completed_with_degradation",
                                      "temporally_truncated", "under_modeled", "truncated"):
            if self.support_grade not in SUPPORT_GRADES:
                raise ValueError(f"{self.simulation_status} result needs a valid support_grade, "
                                 f"got {self.support_grade!r}")
        if self.simulation_status == "under_modeled" and not (self.under_modeled_subtypes
                                                              or self.under_modeled_components):
            raise ValueError("under_modeled requires at least one under_modeled subtype or structured "
                             "component — an unnamed representational gap is not an honest gap")
        if self.recommendation_status not in RECOMMENDATION_STATUSES:
            raise ValueError(f"bad recommendation_status {self.recommendation_status!r}")

    def has_forecast(self) -> bool:
        # temporally_truncated results carry a forecast — from an INCOMPLETE causal unfolding
        # (§12): usable, but the truncation record and lowered support ride with it
        if self.simulation_status in ("under_modeled", "truncated"):
            # §35/§21: these MAY carry a partial exploratory distribution where mathematically
            # defensible — a forecast exists iff that partial distribution does, and it stays
            # explicitly conditional on the modeled portion of the world
            return bool(self.raw_distribution)
        return self.simulation_status in ("completed", "completed_with_degradation",
                                          "temporally_truncated")

    def timing_narrative(self) -> dict:
        """The §27 human-facing timing explanation, rendered from the temporal-runtime block:
        what happens, when it is likely (DAY-level granularity — never false minute precision),
        why the timing looks that way, what could make it earlier or later, and whether the
        answer changes with the timing assumptions. Empty dict when no temporal block exists."""
        trt = (self.provenance or {}).get("temporal_runtime") or {}
        et = (self.provenance or {}).get("event_time") or {}
        if not trt and not et:
            return {}

        def _day(ts):
            if not isinstance(ts, (int, float)):
                return None
            import time as _t
            return _t.strftime("%Y-%m-%d", _t.gmtime(ts))
        qtl = et.get("first_passage_quantiles_ts") or {}
        out = {
            "what_happens": {"distribution": self.raw_distribution,
                             "modes": et.get("mode_distribution")},
            "when_likely": {"median_day": _day(qtl.get("0.5")),
                            "p10_day": _day(qtl.get("0.1")), "p90_day": _day(qtl.get("0.9")),
                            "p_not_by_horizon": et.get("p_censored"),
                            "precision_note": "day-level; sub-day timing is not claimed"},
            "why_this_timing": {
                "generated_processes": {
                    "channels": trt.get("generated_channels"),
                    "institutional_stages": trt.get("institutional_stage_processes"),
                    "continuous_processes": trt.get("continuous_processes")},
                "known_scheduled_facts": trt.get("known_scheduled_facts"),
                "decision_triggers_observed": trt.get("n_decision_triggers")},
            "could_be_earlier_if": ["a watched state change accelerates a hazard "
                                    "(re-projection preserves accumulated exposure)",
                                    "an urgent channel interrupts attention",
                                    "a deferred condition occurs sooner"],
            "could_be_later_if": ["attention cycles, sleep/work windows, or institutional "
                                  "queues defer the triggering events",
                                  "a provisional end-state collapses and the process resumes"],
            "sensitive_to_timing_assumptions": {
                "unresolved_mechanisms": trt.get("unresolved_timing_mechanisms"),
                "temporal_uncertainties": trt.get("temporal_uncertainties"),
                "support": trt.get("timing_support_classification"),
                "truncated": trt.get("temporally_truncated", False)},
        }
        return out

    def as_dict(self) -> dict:
        # §35: a partial under_modeled distribution is NEVER served without its conditionality sentence
        if (self.simulation_status == "under_modeled" and self.raw_distribution
                and not self.conditional_forecast_note):
            self.conditional_forecast_note = CONDITIONAL_FORECAST_SENTENCE
        d = asdict(self)
        # backward-compatible mirror fields for OLD readers (deprecated; never drives new logic).
        # A forecast that ran is NEVER an abstention; only a genuine clarification maps to the old flag.
        # under_modeled/truncated are epistemic/representational states, NOT clarification — they must
        # never mirror to abstain=True.
        d["abstain"] = self.simulation_status == "clarification_required"
        d["abstain_reason"] = self.clarification_reason if d["abstain"] else ""
        d["_semantics"] = "no_abstention_v2"                 # marks results produced under the new contract
        return d


# ------------------------------------------------------------------ migration reader for OLD artifacts
def migrate_legacy_result(old: dict) -> dict:
    """Read a pre-migration result dict (with `abstain`/`abstain_reason`) into the new axes WITHOUT
    editing the stored artifact — used by readers of historical forecast logs. An old abstention becomes
    `clarification_required` ONLY if its reason was genuinely about question coherence; old abstentions
    caused by 'no executable mechanism' / 'unresolved terminal' / 'dangling readout' are re-labeled
    `execution_failed` with the right taxonomy, because under the new semantics those are engineering
    gaps, not forecast refusals. The original text is preserved in provenance."""
    if not old.get("abstain"):
        return {**old, "simulation_status": old.get("simulation_status", "completed"),
                "_migrated": False}
    reason = str(old.get("abstain_reason", "")).lower()
    if any(k in reason for k in ("no executable", "no accepted mechanism", "mechanisms are missing",
                                 "no registry mechanism")):
        status, tax = "execution_failed", "missing_required_operator"
    elif any(k in reason for k in ("readout", "terminal", "unresolved")):
        status, tax = "execution_failed", "terminal_readout_unbindable"
    elif "unparseable" in reason or "parse" in reason:
        status, tax = "execution_failed", "parser_failure_after_retries"
    elif any(k in reason for k in ("ambiguous", "coherent", "cannot define", "interpret")):
        status, tax = "clarification_required", ""
    else:
        status, tax = "execution_failed", "runtime_exception"
    return {**old, "simulation_status": status, "failure_taxonomy": tax,
            "provenance": {**(old.get("provenance") or {}),
                           "migrated_from_legacy_abstention": old.get("abstain_reason", "")},
            "_migrated": True,
            "_migration_note": "legacy `abstain` re-labeled under no-abstention-v2; original preserved"}
