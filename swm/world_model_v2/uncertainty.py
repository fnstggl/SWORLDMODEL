"""Phase 2B — universal uncertainty ontology + the four-stage mechanism compiler.

The fixed, universal ontology classifies every proposed uncertainty; concrete mechanism INSTANCES are
scenario-specific. Four independently-logged stages keep the legacy failure impossible:

    discovery (LLM proposes, TYPED)  →  parameter estimation (8-level source hierarchy; REJECTION is an
    outcome)  →  validation (registry status: validated | prior_backed | experimental)  →  execution
    (transitions.py operators — an experimental mechanism runs only if the plan explicitly enables it,
    with broadened uncertainty, and is independently ablatable)

An LLM-identified uncertainty is a HYPOTHESIS. It cannot execute until its causal role, parameter source,
uncertainty and temporal scale pass this pipeline. Monte Carlo never launders an unsupported guess into
scientific-looking precision: `estimate_parameter` either finds a defensible source or broadens/rejects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

UNCERTAINTY_TYPES = (
    "latent_initial_state", "background_stochastic_process", "exogenous_discrete_event",
    "endogenous_event", "information_arrival", "measurement_process", "observation_error",
    "institutional_transition", "resource_constraint", "parameter_uncertainty",
    "structural_model_uncertainty", "policy_model_uncertainty")

# the parameter-source hierarchy, best-first; "rejected" is a legitimate terminal outcome
PARAMETER_SOURCES = ("direct_observation", "deterministic_derivation", "fitted_domain_data",
                     "reference_class_statistics", "validated_registry_prior", "analogical_prior",
                     "broad_weak_prior", "rejected")

REGISTRY_STATUSES = ("validated", "prior_backed", "experimental")


@dataclass
class MechanismProposal:
    """Stage 1 output — TYPED discovery. The LLM names WHAT might matter and WHY; never numbers."""
    mechanism_id: str
    ontology_type: str                    # one of UNCERTAINTY_TYPES
    entities: list = field(default_factory=list)
    causal_path: list = field(default_factory=list)     # explicit chain to the outcome — no weighted sums
    required_parameters: list = field(default_factory=list)
    required_evidence: list = field(default_factory=list)
    relevance_confidence: str = "medium"  # low | medium | high (qualitative — the LLM may not emit numbers)

    def validate_typing(self) -> list:
        errs = []
        if self.ontology_type not in UNCERTAINTY_TYPES:
            errs.append(f"unknown ontology type {self.ontology_type!r}")
        if not self.causal_path or len(self.causal_path) < 2:
            errs.append("causal_path must chain >=2 steps to the outcome (no direct weighted-sum influence)")
        if self.relevance_confidence not in ("low", "medium", "high"):
            errs.append("relevance_confidence must be qualitative low/medium/high")
        return errs


@dataclass
class ParameterEstimate:
    """Stage 2 output — where the number came from, and how wide it honestly is."""
    name: str
    source: str                           # one of PARAMETER_SOURCES
    distribution: dict = None             # {mean, sd, lo, hi} or {value: p} — NEVER a bare point from an LLM
    why_this_source: str = ""
    evidence: list = field(default_factory=list)
    rejected_reason: str = ""


def estimate_parameter(name: str, *, observed=None, derive_fn=None, fitted=None, reference_class=None,
                       registry_prior=None, analogical=None, broad=None) -> ParameterEstimate:
    """Walk the hierarchy; first defensible source wins; nothing defensible → REJECTED (never fabricate).
    Every argument is (value_or_dist, evidence_list) or None."""
    for source, cand in (("direct_observation", observed), ("deterministic_derivation",
                          (derive_fn(),) if derive_fn else None),
                         ("fitted_domain_data", fitted), ("reference_class_statistics", reference_class),
                         ("validated_registry_prior", registry_prior), ("analogical_prior", analogical),
                         ("broad_weak_prior", broad)):
        if cand is None:
            continue
        val, ev = (cand if isinstance(cand, tuple) and len(cand) == 2 else (cand, []))
        dist = val if isinstance(val, dict) else {"mean": float(val), "sd": abs(float(val)) * 0.5 + 0.1}
        if source in ("analogical_prior", "broad_weak_prior") and "sd" in dist:
            dist = {**dist, "sd": dist["sd"] * 2.0}         # weaker sources are BROADENED, not trusted
        return ParameterEstimate(name=name, source=source, distribution=dist,
                                 why_this_source=f"first defensible level in the hierarchy",
                                 evidence=list(ev))
    return ParameterEstimate(name=name, source="rejected", rejected_reason="no defensible source at any "
                             "hierarchy level — the mechanism cannot be responsibly parameterized")


@dataclass
class ValidatedMechanism:
    """Stage 3 output — executable IF status permits. Experimental ⇒ broadened, labeled, ablatable,
    excluded from calibrated-fidelity claims."""
    proposal: MechanismProposal
    parameters: dict = field(default_factory=dict)        # name -> ParameterEstimate
    registry_status: str = "experimental"
    state_read_set: tuple = ()
    state_write_set: tuple = ()
    sensitivity: float = 0.5
    ablation_flag: str = ""               # unique flag to disable independently

    def executable(self, *, allow_experimental=False) -> tuple:
        rej = [p for p in self.parameters.values() if p.source == "rejected"]
        if rej:
            return False, f"parameters rejected: {[p.name for p in rej]}"
        if self.registry_status == "experimental" and not allow_experimental:
            return False, "experimental mechanism — plan must explicitly enable experimental execution"
        return True, ""


def validate_mechanism(proposal: MechanismProposal, parameters: dict, *, fitted_evidence=False,
                       prior_backed=False, sensitivity=0.5, reads=(), writes=()) -> ValidatedMechanism:
    errs = proposal.validate_typing()
    if errs:
        raise ValueError(f"proposal fails typing: {errs}")
    status = "validated" if fitted_evidence else ("prior_backed" if prior_backed else "experimental")
    # a mechanism whose every parameter came from weak sources cannot claim more than prior_backed
    if status == "validated" and any(p.source in ("analogical_prior", "broad_weak_prior")
                                     for p in parameters.values()):
        status = "prior_backed"
    return ValidatedMechanism(proposal=proposal, parameters=parameters, registry_status=status,
                              state_read_set=tuple(reads), state_write_set=tuple(writes),
                              sensitivity=sensitivity, ablation_flag=f"ablate:{proposal.mechanism_id}")


def detect_conflicts(mechanisms: list) -> list:
    """Duplicate/double-counted effects + circular dependencies over the declared causal paths."""
    problems, seen_writes = [], {}
    for m in mechanisms:
        key = (m.proposal.ontology_type, tuple(m.state_write_set))
        if m.state_write_set and key in seen_writes:
            problems.append(f"double-counted effect: {m.proposal.mechanism_id} and {seen_writes[key]} both "
                            f"write {m.state_write_set} as {m.proposal.ontology_type}")
        elif m.state_write_set:
            seen_writes[key] = m.proposal.mechanism_id
    # circularity: A reads what B writes and vice versa with no temporal separation declared
    for a in mechanisms:
        for b in mechanisms:
            if a is b:
                continue
            if (set(a.state_read_set) & set(b.state_write_set)
                    and set(b.state_read_set) & set(a.state_write_set)):
                pair = tuple(sorted([a.proposal.mechanism_id, b.proposal.mechanism_id]))
                msg = f"circular dependency: {pair[0]} ↔ {pair[1]} (needs explicit temporal ordering)"
                if msg not in problems:
                    problems.append(msg)
    return problems


def uncertainty_report(mechanisms: list, latents: list, residual: list = None) -> dict:
    """The required Phase-2B output block: accepted/rejected/residual + model-vs-world uncertainty split."""
    accepted, rejected = [], []
    for m in mechanisms:
        ok, why = m.executable(allow_experimental=True)
        entry = {"id": m.proposal.mechanism_id, "ontology_type": m.proposal.ontology_type,
                 "causal_role": " → ".join(m.proposal.causal_path[:5]),
                 "entities": m.proposal.entities,
                 "state_read_set": list(m.state_read_set), "state_write_set": list(m.state_write_set),
                 "parameter_source": {k: p.source for k, p in m.parameters.items()},
                 "confidence": m.proposal.relevance_confidence,
                 "registry_status": m.registry_status, "sensitivity": m.sensitivity,
                 "ablation_flag": m.ablation_flag}
        (accepted if ok else rejected).append(entry if ok else
                                              {"id": m.proposal.mechanism_id, "rejection_reason": why})
    return {"accepted_mechanisms": accepted, "rejected_mechanisms": rejected,
            "residual_uncertainty": residual or [],
            "model_uncertainty": {
                "parameter": [{"latent": l.path, "method": l.method} for l in latents
                              if l.method in ("llm", "prior")],
                "structural": [m.proposal.mechanism_id for m in mechanisms
                               if m.registry_status == "experimental"],
                "note": "world randomness lives in hazards/sampling; model uncertainty listed here"}}
