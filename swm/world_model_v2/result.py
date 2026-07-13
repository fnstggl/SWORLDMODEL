"""Production result semantics — the no-abstention contract (Part A).

The product simulates EVERY coherent social question. Weak evidence, missing validated mechanisms,
transport risk, unfamiliar domains and uncertain hidden state do NOT prevent a forecast — they widen priors,
add competing hypotheses, lower the SUPPORT GRADE and add limitations. Binary forecast abstention is
replaced by three INDEPENDENT concepts:

  SimulationStatus     — did the simulation run?  completed / completed_with_degradation /
                         clarification_required / execution_failed
  SupportGrade         — how well-evidenced is it? empirically_supported / transfer_supported /
                         exploratory / highly_speculative
  RecommendationStatus — may we recommend an action? eligible / limited / withheld / not_requested

`clarification_required` is reserved for genuinely incoherent questions (no simulable interpretation exists,
even after generating competing ones) and must be RARE. `execution_failed` is an engineering/architectural
failure (exception, unresolved reference, invalid plan, unbindable readout, missing operator) — NOT
epistemic abstention. Neither may be used to hide compiler incompleteness or missing mechanism coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# ------------------------------------------------------------------ the three independent axes
SIMULATION_STATUSES = ("completed", "completed_with_degradation", "clarification_required",
                       "execution_failed")
SUPPORT_GRADES = ("empirically_supported", "transfer_supported", "exploratory", "highly_speculative")
RECOMMENDATION_STATUSES = ("eligible", "limited", "withheld", "not_requested")

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
    """The full user-facing result. Every completed/degraded result carries all epistemic fields; a
    forecast is present whenever the simulation ran (status completed or completed_with_degradation)."""
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
    evidence_quality: str = ""
    limitations: list = field(default_factory=list)
    fallbacks_used: list = field(default_factory=list)      # [{process, tier, family, why}]
    mechanism_tiers: dict = field(default_factory=dict)     # {causal_process: tier}
    omitted_high_sensitivity_variables: list = field(default_factory=list)
    sensitivity_contributors: list = field(default_factory=list)
    interpretation_hypotheses: list = field(default_factory=list)
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
        if self.simulation_status in ("completed", "completed_with_degradation"):
            if self.support_grade not in SUPPORT_GRADES:
                raise ValueError(f"completed result needs a valid support_grade, got {self.support_grade!r}")
        if self.recommendation_status not in RECOMMENDATION_STATUSES:
            raise ValueError(f"bad recommendation_status {self.recommendation_status!r}")

    def has_forecast(self) -> bool:
        return self.simulation_status in ("completed", "completed_with_degradation")

    def as_dict(self) -> dict:
        d = asdict(self)
        # backward-compatible mirror fields for OLD readers (deprecated; never drives new logic).
        # A forecast that ran is NEVER an abstention; only a genuine clarification maps to the old flag.
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
