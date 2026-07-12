"""Production mechanism-registry records — Phase 6.

A mechanism FAMILY is a reusable structural hypothesis (complex contagion, quantal response, trust
repair, …) with an executable transition. A PARAMETER PACK binds a family to a domain/population with a
parameter posterior and its provenance. A SCENARIO INSTANTIATION (done by the compiler) binds a pack to
concrete state paths in one world.

Statuses form an explicit lifecycle and are ENFORCED at promotion time (see ingestion.py): a family may
not become production-eligible from a paper citation or a synthetic test alone — it needs an executable
transition, tests, evidence metadata, applicability rules, uncertainty, ≥1 parameter pack, and ≥1 held-out
or posterior-predictive validation recorded here. Failed validations are PRESERVED, never deleted.

Parameter-source vocabulary (rule 3 of the engineering contract — no unsupported precision):
  observed | inferred_from_data | fitted | published_research | reference_class_prior |
  experimental | assumed | unsupported
`unsupported` parameters must carry a broad distribution, be excluded with a logged omission, or force
abstention — never a bare point value.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field, asdict

STATUSES = ("proposed", "implemented", "locally_validated", "transfer_validated",
            "production_eligible", "quarantined", "rejected")

PARAMETER_SOURCES = ("observed", "inferred_from_data", "fitted", "published_research",
                     "reference_class_prior", "experimental", "assumed", "unsupported")

ONTOLOGY_TYPES = ("observation", "attention", "memory", "interpretation", "decision", "learning",
                  "belief", "relationship", "norm", "bargaining", "coalition", "participation",
                  "diffusion", "influence", "network", "platform", "resource", "institutional",
                  "measurement", "exogenous")


class RegistryError(ValueError):
    pass


@dataclass
class Citation:
    """A supporting study, with the exact limits of what it licenses."""
    ref: str                                  # short cite: "Centola & Macy 2007, AJS 113(3)"
    doi_or_url: str = ""
    study_population: str = ""                # who was studied
    study_period: str = ""                    # when
    finding: str = ""                         # the precise claim imported
    limits: str = ""                          # transport limits — what this does NOT license


@dataclass
class ValidationRecord:
    """One empirical check. kind: held_out | posterior_predictive | transfer | ablation |
    failed_replication | synthetic. Failed results are preserved (passed=False stays in history)."""
    kind: str
    dataset: str
    split: str                                # e.g. "time-forward test n=4000, seed 17"
    metric: str
    value: float | None
    baseline: str = ""
    baseline_value: float | None = None
    ci95: list = field(default_factory=list)
    passed: bool | None = None                # None = descriptive, not pass/fail
    artifact: str = ""                        # result file path
    at: str = ""
    note: str = ""


@dataclass
class ParameterSpec:
    """One named parameter of a family: its meaning, bounds, and REQUIRED source labeling."""
    name: str
    description: str
    lo: float | None = None
    hi: float | None = None
    default_source: str = "unsupported"       # PARAMETER_SOURCES

    def __post_init__(self):
        if self.default_source not in PARAMETER_SOURCES:
            raise RegistryError(f"{self.name}: bad parameter source {self.default_source!r}")


@dataclass
class ParameterPack:
    """A family bound to a domain/population: parameter values WITH uncertainty and provenance.
    values: {param: {"value": v, "sd": s, "lo": l, "hi": h, "source": <PARAMETER_SOURCES>,
                     "method": how, "dataset": where}}. A pack with any 'unsupported' source cannot
    ship a bare point — enforce_uncertainty() widens or raises."""
    pack_id: str
    family_id: str
    domain: str
    population: str
    values: dict
    fitted_on: str = ""                       # dataset + split the fit used (train only!)
    fit_method: str = ""
    time_scale: str = ""
    citations: list = field(default_factory=list)      # [Citation] for published packs
    validation: list = field(default_factory=list)     # [ValidationRecord]
    transport_note: str = ""                  # widening applied / assumptions when moving domains
    version: str = "1.0.0"
    created_at: str = ""

    def enforce_uncertainty(self):
        for name, v in self.values.items():
            src = v.get("source", "unsupported")
            if src not in PARAMETER_SOURCES:
                raise RegistryError(f"pack {self.pack_id}: param {name} bad source {src!r}")
            if src in ("assumed", "unsupported", "experimental") and not (
                    v.get("sd") or (v.get("lo") is not None and v.get("hi") is not None)):
                raise RegistryError(
                    f"pack {self.pack_id}: param {name} is {src} with no uncertainty — "
                    f"unsupported precision is banned (give sd or lo/hi, or exclude with a logged omission)")
        return self


@dataclass
class ApplicabilityRule:
    """Declared applicability/exclusion conditions the compiler scores against a scenario."""
    domains: list = field(default_factory=lambda: ["*"])      # domain tags this family fits
    excluded_domains: list = field(default_factory=list)
    requires_state: list = field(default_factory=list)        # world-state kinds needed (e.g. "network")
    requires_data: list = field(default_factory=list)         # evidence kinds needed to parameterize
    time_scales: list = field(default_factory=list)           # e.g. ["hours","days"]
    population_kinds: list = field(default_factory=list)      # e.g. ["online_social","organizational"]
    exclusion_conditions: list = field(default_factory=list)  # free-text, precise
    transport_risk: str = "high"              # low | medium | high — default pessimistic


@dataclass
class MechanismRecord:
    """The full production record for one mechanism family."""
    family_id: str                            # unique, stable
    version: str                              # semantic version of the IMPLEMENTATION
    ontology_type: str                        # ONTOLOGY_TYPES
    title: str
    formal_description: str                   # the equations/algorithm, precisely
    causal_inputs: list
    causal_outputs: list
    required_state: list
    action_dependencies: list = field(default_factory=list)
    temporal_scale: str = "event"
    parameters: list = field(default_factory=list)        # [ParameterSpec]
    applicability: ApplicabilityRule = field(default_factory=ApplicabilityRule)
    citations: list = field(default_factory=list)         # [Citation]
    packs: list = field(default_factory=list)             # [ParameterPack]
    validation: list = field(default_factory=list)        # [ValidationRecord] — family-level
    status: str = "proposed"
    status_reason: str = ""
    known_failure_modes: list = field(default_factory=list)
    owner: str = "wmv2"
    code_ref: str = ""                        # "module:callable" of the executable transition
    test_ref: str = ""                        # test file exercising the transition
    uncertainty_note: str = ""                # how parameter/structural uncertainty is represented
    implementation_note: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise RegistryError(f"{self.family_id}: bad status {self.status!r}")
        if self.ontology_type not in ONTOLOGY_TYPES:
            raise RegistryError(f"{self.family_id}: bad ontology_type {self.ontology_type!r}")

    # ---------- lifecycle checks (enforced, not aspirational) ----------
    def executable(self) -> bool:
        """True iff code_ref resolves to a callable right now."""
        try:
            mod, _, name = self.code_ref.partition(":")
            import importlib
            return callable(getattr(importlib.import_module(mod), name))
        except Exception:
            return False

    def has_validation(self, kinds=("held_out", "posterior_predictive", "transfer")) -> bool:
        recs = list(self.validation) + [v for p in self.packs for v in p.validation]
        return any(v.kind in kinds and v.passed is not None for v in recs)

    def failed_validations(self) -> list:
        recs = list(self.validation) + [v for p in self.packs for v in p.validation]
        return [v for v in recs if v.passed is False]

    def promotion_blockers(self, target: str) -> list:
        """Why this record cannot move to `target` status. Empty list = allowed."""
        order = {s: i for i, s in enumerate(
            ("proposed", "implemented", "locally_validated", "transfer_validated", "production_eligible"))}
        if target in ("quarantined", "rejected"):
            return []                          # demotion is always allowed (with a reason)
        blockers = []
        if target not in order:
            return [f"unknown target status {target!r}"]
        if order[target] >= order["implemented"]:
            if not self.executable():
                blockers.append(f"code_ref {self.code_ref!r} does not resolve to a callable")
            if not self.test_ref:
                blockers.append("no test_ref")
            if not self.formal_description.strip():
                blockers.append("no formal description")
        if order[target] >= order["locally_validated"]:
            if not self.packs:
                blockers.append("no parameter pack")
            if not self.has_validation(("held_out", "posterior_predictive")):
                blockers.append("no held-out or posterior-predictive validation record")
        if order[target] >= order["transfer_validated"]:
            if not self.has_validation(("transfer",)):
                blockers.append("no transfer validation record")
        if order[target] >= order["production_eligible"]:
            if not self.citations and not any(p.citations for p in self.packs):
                blockers.append("no supporting research/dataset citation")
            held = [v for v in self.validation +
                    [v for p in self.packs for v in p.validation]
                    if v.kind in ("held_out", "transfer") and v.passed]
            if not held:
                blockers.append("no PASSED held-out/transfer validation — paper citations and synthetic "
                                "tests alone cannot make a mechanism production eligible")
        return blockers

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def now_iso() -> str:
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
