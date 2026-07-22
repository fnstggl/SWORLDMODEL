"""First-class structural-ensemble contracts — structural-model uncertainty as a typed, default-on layer.

THE DISTINCTION THIS LAYER ENFORCES (three separate uncertainty levels, never collapsed):

  A. STRUCTURAL-MODEL uncertainty — which actors, institutions, constraints, mechanisms, boundaries and
     information routes determine the outcome AT ALL. Each structural model is its OWN independently
     generated, independently executable `WorldExecutionPlan`. NOT a random seed, NOT a parameter draw,
     NOT an entry in `plan.structural_hypotheses` (those are level-B hypotheses INSIDE one schema), NOT a
     narrative label on a shared schema.
  B. WITHIN-MODEL world uncertainty — hidden facts, private actor states, exogenous events, parameters and
     initial conditions inside ONE structural model. Represented (as before) by particles / coherent world
     hypotheses within that model's plan.
  C. BEHAVIORAL uncertainty — what actors do after seeing their own information. Represented (as before)
     by qualitative actor decisions across coherent particles.

A perfectly executed simulation of the wrong causal model is still wrong; level A is therefore simulated by
running EVERY promoted structural model through the full canonical runtime with its own complete particle
budget, then comparing the model-specific trajectory distributions. The budget invariant is absolute: a
promoted model receives AT LEAST the particle count one model receives in single-model production —
budgets are never divided across models.

MODEL SUPPORT IS NEVER A MINTED PROBABILITY. LLM critics assign only the qualitative SUPPORT_CLASSES below,
grounded in evidence fit; numeric model weights exist only when a defensible external basis exists. Absent
that, aggregation is an explicitly labeled equal-weight compatibility mixture plus robust (range / minimax)
analysis — never a silent precise prior.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

# ------------------------------------------------------------------ qualitative support classes (LLM-legal)
#: The ONLY support vocabulary an LLM critic may emit for a structural model. These labels are never
#: silently converted into numeric probabilities (enforced by tests).
SUPPORT_CLASSES = ("strongly_supported", "plausible", "weak_but_possible", "contradicted", "unresolved")

#: Candidate lifecycle. `merged` candidates survive inside their surviving model's merge record.
PROMOTION_STATUSES = ("generated", "repaired", "rejected", "merged", "pilot", "promoted",
                      "not_promoted", "failed")

#: Pilot lifecycle for a candidate.
PILOT_STATUSES = ("not_run", "running", "completed", "failed", "reused_in_full")

# ------------------------------------------------------------------ generation perspectives (Stage A)
#: General causal REASONING perspectives for independent reconnaissance — deliberately domain-free.
#: A perspective shapes what the generator actively investigates; every output must still be a COMPLETE
#: causal model of the question. These are not templates and not a fixed production ontology; the
#: adaptive policy may add adversarial/expansion roles beyond this list.
GENERATION_PERSPECTIVES = (
    ("actor_relationship", "the structure of strategic actors, their goals, capabilities and relationships"),
    ("institutional_procedural", "the institutions, formal procedures, authority chains and rule systems"),
    ("resource_constraint", "resource, capacity, operational, physical or timing constraints that bind"),
    ("information_distribution", "information flow, distribution channels, networks and attention"),
    ("exogenous_external", "exogenous market, legal, algorithmic, environmental or third-party systems"),
    ("adversarial_alternative", "the strongest alternative causal explanation the other lenses would miss"),
)

#: Default generation policy (Section 6): at least three independent attempts, normally four, adaptive
#: expansion up to a soft ceiling, a higher ceiling in maximum-capacity mode. The ceiling is NEVER treated
#: as proof the causal space is complete — reaching it with live critic findings marks the run
#: structurally_underidentified.
GENERATION_MIN_INDEPENDENT_CALLS = 3
GENERATION_TARGET_CALLS = 4
GENERATION_SOFT_CEILING = 8
GENERATION_MAX_CAPACITY_CEILING = 12

# ------------------------------------------------------------------ structural-sensitivity classification
STRUCTURAL_SENSITIVITY_CLASSES = (
    "structurally_stable",                  # answer survives across all promoted models
    "mildly_structurally_sensitive",        # models shift the answer within decision-irrelevant bounds
    "materially_structurally_sensitive",    # models disagree enough to change the answer/recommendation
    "structurally_underidentified",         # plausible alternatives remained unresolved at the ceiling
    "ensemble_execution_incomplete",        # a required ensemble stage failed — result is not a full ensemble
)

#: Classification thresholds on ACTUAL model-specific results (validated in
#: tests/test_structural_ensemble.py::test_sensitivity_thresholds_*; exposed here, never hidden).
#: For binary/categorical forecasts the driver is the max cross-model spread of any option's probability;
#: for decisions it is winner-change / regret (see swm/world_model_v2/phase13 integration).
SENSITIVITY_STABLE_MAX_SPREAD = 0.05        # spread below this: structurally stable
SENSITIVITY_MILD_MAX_SPREAD = 0.15          # spread below this: mild; at/above: material
#: For decisions: any winner change across models is at least MATERIAL; regret share of utility range
#: above this is material even without a winner change.
SENSITIVITY_DECISION_REGRET_SHARE = 0.15

AGGREGATION_METHODS = (
    "equal_weight_uncalibrated_structural_average",   # labeled compatibility mixture (no defensible weights)
    "externally_weighted",                            # defensible weights exist (documented source required)
    "single_surviving_model",                         # ensemble conservatively collapsed (certificate required)
)


def structural_signature(plan) -> dict:
    """Deterministic structural fingerprint of an executable plan — the level-A identity of a model.

    Compares ONLY causal elements that could change the result: world boundary (entity/population/
    institution sets), decisive actors, institutions, constraints (latents + quantities), mechanisms
    (accepted operator set), information routes (relations), intervention pathways (actor decisions +
    candidate action families), scheduled processes, and resolution dependencies (readout + outcome
    family/options). Prose (rationale, descriptions) is EXCLUDED so different narration of the same
    executable structure compares equal, and different structure never compares equal by prose accident.
    """
    def _ids(items, key="id"):
        out = []
        for it in items or []:
            if isinstance(it, dict) and it.get(key):
                out.append(str(it[key]))
            elif isinstance(it, str):
                out.append(it)
        return sorted(set(out))

    oc = plan.outcome_contract
    relations = sorted({f"{r.get('src')}-{r.get('rel')}->{r.get('dst')}"
                        for r in (plan.relations or []) if isinstance(r, dict)})
    action_paths = sorted({
        f"{d.get('actor')}:{a.get('family', a.get('name', '?'))}->{(a.get('target') or {}).get('target_id', '')}"
        for d in (plan.actor_decisions or []) if isinstance(d, dict)
        for a in (d.get("candidate_actions") or []) if isinstance(a, dict)})
    scheduled = sorted({str(e.get("etype")) for e in (plan.scheduled_events or [])
                        if isinstance(e, dict) and e.get("etype") != "resolve_outcome"})
    return {
        "entities": _ids(plan.entities),
        "populations": _ids(plan.populations),
        "institutions": _ids(plan.institutions),
        "mechanisms": sorted({str(m.get("mech_id")) for m in (plan.accepted_mechanisms or [])
                              if isinstance(m, dict)}),
        "relations": relations,
        "latents": sorted({str(getattr(l, "path", l)) for l in (plan.latents or [])}),
        "quantities": _ids(plan.quantities, key="name"),
        "action_pathways": action_paths,
        "scheduled_event_types": scheduled,
        "outcome_family": oc.family,
        "outcome_options": [str(o) for o in (oc.options or [])],
        "readout_var": oc.readout_var,
    }


def schema_hash(plan) -> str:
    """Content hash of the structural signature — two plans with equal schema_hash are structurally
    EQUIVALENT on every element the deterministic comparison inspects (necessary, not sufficient, for a
    merge: the conservative dedup additionally compares decisive assumptions and intervention response)."""
    return hashlib.sha1(json.dumps(structural_signature(plan), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class StructuralModelCandidate:
    """One independently generated causal model of the question — level-A identity + full lineage.

    Every candidate retains its OWN execution-plan identity, evidence-conditioned revisions, posterior,
    particles, event queue, actor private states, dynamic-recompilation lineage, simulation result and
    support classification. No mutable plan or world state is ever shared across candidates (shared
    IMMUTABLE evidence and content-addressed LLM caching are the only permitted sharing)."""
    model_id: str
    independent_generation_call_id: str = ""     # ties the candidate to its OWN generation LLM call
    generation_role: str = ""                    # perspective (GENERATION_PERSPECTIVES id / critic role)
    causal_thesis: str = ""                      # one-paragraph central causal claim
    decisive_actors: list = field(default_factory=list)
    decisive_institutions: list = field(default_factory=list)
    decisive_constraints: list = field(default_factory=list)
    decisive_mechanisms: list = field(default_factory=list)
    world_boundary: str = ""                     # what is inside/outside this model's world
    omitted_components: list = field(default_factory=list)
    falsifiers: list = field(default_factory=list)           # what would make this model wrong
    intervention_response: str = ""              # how an intervention propagates differently here
    evidence_requirements: list = field(default_factory=list)  # [EvidenceRequirement.as_dict()]
    executable_plan: object = None               # WorldExecutionPlan (its OWN; never shared)
    plan_hash: str = ""
    schema_hash: str = ""
    plan_lineage: list = field(default_factory=list)         # plan hashes through evidence/dynamic recompile
    parent_ids: list = field(default_factory=list)           # ancestry (splits / repairs / merges)
    provenance: dict = field(default_factory=dict)           # prompts hashes, seeds, stage, timings
    validation: dict = field(default_factory=dict)           # executability + boundary checks
    critic_findings: list = field(default_factory=list)      # [{critic, finding, action}]
    unresolved_mechanisms: list = field(default_factory=list)
    support_class: str = "unresolved"            # SUPPORT_CLASSES only — never a minted probability
    support_basis: str = ""                      # evidence-fit / consistency basis for the class
    promotion_status: str = "generated"          # PROMOTION_STATUSES
    promotion_reason: str = ""
    pilot_status: str = "not_run"                # PILOT_STATUSES
    pilot_particles: int = 0
    final_particles: int = 0
    pilot_result: dict = field(default_factory=dict)         # distribution + diagnostics from the pilot
    final_result: dict = field(default_factory=dict)         # model-specific full simulation summary
    posterior_diagnostics: dict = field(default_factory=dict)
    merge_record: dict = None                    # populated when this candidate was merged away

    def __post_init__(self):
        if self.support_class not in SUPPORT_CLASSES:
            raise ValueError(f"support_class must be one of {SUPPORT_CLASSES}, got {self.support_class!r}")
        if self.promotion_status not in PROMOTION_STATUSES:
            raise ValueError(f"bad promotion_status {self.promotion_status!r}")

    def summary(self) -> dict:
        """Blind-labelable summary used by critics and the equivalence judge (no model_id leakage)."""
        return {"causal_thesis": self.causal_thesis,
                "decisive_actors": list(self.decisive_actors),
                "decisive_institutions": list(self.decisive_institutions),
                "decisive_constraints": list(self.decisive_constraints),
                "decisive_mechanisms": list(self.decisive_mechanisms),
                "world_boundary": self.world_boundary,
                "intervention_response": self.intervention_response}

    def as_dict(self, include_plan: bool = False) -> dict:
        d = asdict(self)
        d.pop("executable_plan", None)
        if include_plan and self.executable_plan is not None:
            d["executable_plan_summary"] = structural_signature(self.executable_plan)
        return d


@dataclass
class StructuralModelEnsemble:
    """The ensemble-level record: everything generated, criticized, merged, piloted and promoted for one
    question, with the manifests that make cost and independence auditable."""
    question: str
    as_of: str = ""
    horizon: str = ""
    intervention: str = ""
    ensemble_id: str = ""
    generation_policy: dict = field(default_factory=dict)     # target/ceiling/mode actually applied
    candidates: list = field(default_factory=list)            # [StructuralModelCandidate] (ALL, any status)
    candidates_generated: int = 0
    candidates_repaired: int = 0
    candidates_rejected: int = 0
    candidates_merged: int = 0
    pilot_models: list = field(default_factory=list)          # model_ids that received a pilot
    full_models: list = field(default_factory=list)           # model_ids promoted to full simulation
    shared_evidence_bundle_hash: str = ""
    shared_evidence_as_of: str = ""
    model_support: dict = field(default_factory=dict)         # model_id -> support_class (qualitative)
    structural_coverage: dict = field(default_factory=dict)   # which causal axes are represented/missing
    unresolved_alternatives: list = field(default_factory=list)  # critic-identified missing models
    stopping_reason: str = ""                                 # why generation stopped
    convergence_certificate: dict = None                      # REQUIRED when only one model survives
    structurally_underidentified: bool = False
    generation_manifest: list = field(default_factory=list)   # per-call: role, prompt_hash, response_hash, ok
    critic_manifest: list = field(default_factory=list)       # per-critic-call trace
    merge_manifest: list = field(default_factory=list)        # per-merge: sources, comparison, confidence
    simulation_manifest: dict = field(default_factory=dict)   # model_id -> {pilot_n, final_n, reused, status}
    cost_manifest: dict = field(default_factory=dict)         # llm calls by stage/model, cache hits, tokens

    def __post_init__(self):
        if not self.ensemble_id:
            self.ensemble_id = "ens_" + hashlib.sha1(
                f"{self.question}|{self.as_of}|{self.horizon}|{self.intervention}".encode()
            ).hexdigest()[:12]

    # -------------------------------------------------------------- accessors
    def by_id(self, model_id: str):
        for c in self.candidates:
            if c.model_id == model_id:
                return c
        return None

    def surviving(self) -> list:
        """Validated, non-merged, non-rejected candidates (the pilot population)."""
        return [c for c in self.candidates
                if c.promotion_status not in ("rejected", "merged", "failed")]

    def promoted(self) -> list:
        return [c for c in self.candidates if c.promotion_status == "promoted"]

    def independent_generation_calls(self) -> int:
        """Count of ACTUAL independent Stage-A generation calls (not candidates — expansion candidates
        proposed by critics count via their own generation calls)."""
        return len([g for g in self.generation_manifest if g.get("independent")])

    def record_generation(self, *, role: str, prompt_hash: str, response_hash: str, ok: bool,
                          independent: bool = True, error: str = "", call_id: str = ""):
        self.generation_manifest.append({
            "call_id": call_id or f"gen_{len(self.generation_manifest)}", "role": role,
            "prompt_hash": prompt_hash, "response_hash": response_hash, "ok": ok,
            "independent": independent, "error": error})

    def validate_integrity(self):
        """Loud invariant checks (Section 23). Raises EnsembleIntegrityError on violation."""
        # distinct independent generation prompts — identical prompts mean the independence contract broke
        prompts = [g["prompt_hash"] for g in self.generation_manifest if g.get("independent") and g.get("ok")]
        if len(prompts) > 1 and len(set(prompts)) == 1:
            raise EnsembleIntegrityError(
                "all independent generation calls shared ONE prompt — generation was not independent")
        # no shared mutable plan objects across candidates
        seen = {}
        for c in self.candidates:
            if c.executable_plan is None:
                continue
            pid = id(c.executable_plan)
            if pid in seen and seen[pid] != c.model_id:
                raise EnsembleIntegrityError(
                    f"candidates {seen[pid]} and {c.model_id} SHARE one plan object — "
                    "model state must be isolated")
            seen[pid] = c.model_id
        # single-survivor runs require the convergence certificate
        if len(self.surviving()) == 1 and self.convergence_certificate is None:
            raise EnsembleIntegrityError(
                "ensemble collapsed to one model without a convergence certificate — a one-model result "
                "requires proof (independent attempts + alternatives invalid/equivalent + omission critic "
                "exhausted)")
        return self

    def as_dict(self) -> dict:
        d = asdict(self)
        d["candidates"] = [c.as_dict() for c in self.candidates]
        return d


class EnsembleIntegrityError(RuntimeError):
    """A structural-ensemble invariant was violated (independence, isolation, budget, or certificate).
    Always loud; never downgraded to a warning."""


def classify_forecast_sensitivity(model_distributions: dict, *, underidentified: bool = False,
                                  incomplete: bool = False) -> dict:
    """Structural-sensitivity classification from ACTUAL model-specific forecast distributions.

    `model_distributions`: {model_id: {option: prob}}. Uses max cross-model spread over options
    (distribution separation) plus outcome-direction change detection. Thresholds are the exposed
    module constants (validated by tests), never hidden cutoffs."""
    if incomplete:
        return {"classification": "ensemble_execution_incomplete", "max_spread": None,
                "basis": "a required ensemble stage failed; results do not form a complete ensemble"}
    dists = {m: d for m, d in (model_distributions or {}).items() if d}
    if len(dists) <= 1:
        cls = "structurally_underidentified" if underidentified else "structurally_stable"
        return {"classification": cls, "max_spread": 0.0,
                "basis": ("ceiling reached with unresolved alternatives" if underidentified else
                          "one surviving model with a recorded convergence certificate")}
    options = sorted({o for d in dists.values() for o in d})
    spreads = {}
    for o in options:
        vals = [float(d.get(o, 0.0)) for d in dists.values()]
        spreads[o] = round(max(vals) - min(vals), 4)
    max_spread = max(spreads.values()) if spreads else 0.0
    # outcome-direction change: does the modal option differ across models?
    modal = {m: max(d, key=d.get) for m, d in dists.items()}
    direction_change = len(set(modal.values())) > 1
    if underidentified:
        cls = "structurally_underidentified"
    elif direction_change or max_spread >= SENSITIVITY_MILD_MAX_SPREAD:
        cls = "materially_structurally_sensitive"
    elif max_spread >= SENSITIVITY_STABLE_MAX_SPREAD:
        cls = "mildly_structurally_sensitive"
    else:
        cls = "structurally_stable"
    return {"classification": cls, "max_spread": round(max_spread, 4), "per_option_spread": spreads,
            "modal_option_by_model": modal, "direction_change": direction_change,
            "thresholds": {"stable_max": SENSITIVITY_STABLE_MAX_SPREAD,
                           "mild_max": SENSITIVITY_MILD_MAX_SPREAD}}


def decompose_uncertainty(model_distributions: dict, within_model: dict = None) -> dict:
    """Decompose total uncertainty into between-model structural variance and within-model components.

    For each option: between-model variance of the per-model probabilities (equal-weight across models —
    the labeled compatibility convention) plus the mean of the supplied within-model variances (sampling /
    world / behavioral, as provided per model). Purely arithmetic — no minted numbers."""
    dists = {m: d for m, d in (model_distributions or {}).items() if d}
    if not dists:
        return {"between_model": {}, "within_model": dict(within_model or {}), "n_models": 0}
    options = sorted({o for d in dists.values() for o in d})
    n = len(dists)
    between = {}
    for o in options:
        vals = [float(d.get(o, 0.0)) for d in dists.values()]
        mean = sum(vals) / n
        between[o] = round(sum((v - mean) ** 2 for v in vals) / n, 6)
    return {"between_model": between, "within_model": dict(within_model or {}),
            "n_models": n, "weighting": "equal_weight_uncalibrated_structural_average"}
