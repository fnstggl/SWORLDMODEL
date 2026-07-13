"""Mechanism registry — Phase 5's vocabulary. What the compiler is ALLOWED to instantiate.

Every entry declares: ontology type, causal role, required state, parameter source, temporal scale,
calibration status, domain applicability. The LLM proposes mechanisms BY REGISTRY ID (plus free-text
candidates it believes are missing — those become `candidate_experimental_mechanisms`, marked experimental,
never silently executed). Validated numerical kernels (v1's Poisson-arrival, whipcount, poll-error
aggregation, …) plug in here as mechanisms — behind the same interface, never the logistic-over-invented-
variables path for human behavior (unreachable from this package; pinned by test).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MechanismEntry:
    mech_id: str
    ontology_type: str                     # decision | belief | relationship | resource | diffusion |
    #                                        institutional | numerical | exogenous | measurement
    causal_role: str
    required_state: tuple = ()
    parameter_source: str = ""
    temporal_scale: str = "event"
    calibration_status: str = "uncalibrated"    # calibrated | prior | uncalibrated | experimental
    domains: tuple = ("*",)                # applicability ("*" = domain-general)
    operator: str = ""                     # transitions-registry operator that executes it
    experimental: bool = False


_REGISTRY: dict = {}


def register_mechanism(entry: MechanismEntry):
    _REGISTRY[entry.mech_id] = entry
    return entry.mech_id


def get_mechanism(mech_id: str) -> MechanismEntry:
    e = _REGISTRY.get(mech_id)
    if e is None:
        raise KeyError(f"unknown mechanism {mech_id!r} — the compiler must mark it experimental, not "
                       f"fabricate an implementation (known: {sorted(_REGISTRY)})")
    return e


def known_mechanisms() -> dict:
    return dict(_REGISTRY)


# ---------------- the foundational, domain-general set ----------------
for e in (
    MechanismEntry("agent_decision", "decision", "an actor chooses among typed institution-valid actions",
                   required_state=("entity", "information_set"), parameter_source="LLM policy (typed actions)",
                   operator="agent_decision", calibration_status="prior"),
    MechanismEntry("belief_update", "belief", "exposure moves an actor's typed belief, bounded",
                   required_state=("information", "entity.beliefs"),
                   parameter_source="rule core: credibility×trust×salience (broad priors)",
                   operator="belief_update", calibration_status="prior"),
    MechanismEntry("relationship_update", "relationship", "actions shift edge strength/trust, bounded",
                   required_state=("network",), parameter_source="bounded shifts |Δ|<=0.25",
                   operator="relationship_update", calibration_status="prior"),
    MechanismEntry("resource_update", "resource", "actions consume/create conserved resources",
                   required_state=("entity.resources",), parameter_source="conservation-checked",
                   operator="resource_update", calibration_status="calibrated"),
    MechanismEntry("institutional_vote", "institutional", "deterministic execution of voting rules",
                   required_state=("institutions",), parameter_source="rule execution",
                   operator="institutional_vote", calibration_status="calibrated"),
    MechanismEntry("background_dynamics", "exogenous", "attention drift + memory decay over elapsed time",
                   required_state=("entities",), parameter_source="broad priors (labeled)",
                   operator="background_dynamics", calibration_status="prior"),
    # the tier-6/7 FALLBACK: guarantees every coherent question can resolve an outcome from a broad prior
    # when no validated mechanism applies. NOT empirically validated — labeled exploratory/speculative.
    MechanismEntry("generic_outcome_prior", "numerical",
                   "resolve the terminal outcome from a BROAD prior (Beta/Normal) when no validated "
                   "mechanism applies; per-particle draw; qualitative lean only; never LLM-minted",
                   required_state=("quantities",), parameter_source="broad prior; tier 6/7 fallback",
                   temporal_scale="horizon", calibration_status="experimental", operator="generic_outcome_prior"),
    # v1's numerical kernels. poisson_arrival is PORTED (RareEventArrivalOperator). The other two are
    # NOT yet executable in V2 — marked experimental with empty operator so the compiler rejects them
    # LOUDLY instead of accepting silent no-ops (Tier A1; they were 2 of the audit's 3 dead entries).
    MechanismEntry("poisson_arrival", "numerical", "rare event by a deadline: P=1−exp(−λH)",
                   required_state=("quantities",), parameter_source="base-rate/observed rate (v1 sim_arrival)",
                   temporal_scale="horizon", calibration_status="prior", operator="poisson_arrival"),
    MechanismEntry("poll_error_aggregation", "measurement", "latent share vs threshold with poll error",
                   required_state=("quantities",), parameter_source="empirical poll error 3-6pt (v1); "
                   "NO V2 OPERATOR YET — v1 kernel unported",
                   temporal_scale="scheduled", calibration_status="experimental", experimental=True),
    MechanismEntry("whipcount_binomial", "institutional", "undecideds break at grounded lean (binomial)",
                   required_state=("institutions", "quantities"), parameter_source="whip counts (v1); "
                   "NO V2 OPERATOR YET — v1 kernel unported",
                   calibration_status="experimental", experimental=True),
):
    register_mechanism(e)
