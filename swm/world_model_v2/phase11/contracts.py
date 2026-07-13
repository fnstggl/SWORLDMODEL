"""Phase 11 — versioned, serializable dynamic-recompilation contracts (spec §8.1–8.8).

Every contract is a ``@dataclass`` carrying a semantic version + a deterministic content hash (via ``_serial``)
and typed fields. Names are adapted to this repository's conventions (``WorldExecutionPlan.plan_hash``,
``StateDelta``, ``SimulationResult`` statuses) but the required capability is present. These are pure data
records — detection/scoring/migration logic lives in the sibling modules and consumes/produces these.

Nothing here mints probabilities from an LLM; numeric fields are populated by the diagnostic/scoring/migration
code from real posteriors and evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.phase11._serial import Versioned

# ---- controlled vocabularies (typed, not free strings where it matters) --------------------------------
TRIGGER_FAMILIES = (
    "unexplained_residual", "impossible_event", "new_actor", "new_institution", "rule_change",
    "authority_change", "coalition_change", "network_restructuring", "exogenous_shock",
    "mechanism_precondition_failure", "sustained_predictive_failure", "particle_collapse",
    "evidence_contradiction", "outcome_space_change", "mechanism_regime_change", "parameter_drift",
)

RECOMPILE_ACTIONS = (
    "no_change", "observation_model_update", "parameter_update", "latent_state_update",
    "actor_revision", "relationship_revision", "local_network_recompile", "population_segment_revision",
    "institution_recompile", "mechanism_replacement", "structural_branch_addition",
    "structural_branch_pruning", "action_space_revision", "outcome_contract_revision", "full_recompile",
)

SCOPES = (
    "no_model_change", "observation_model", "parameter_only", "latent_state", "actor", "relationship",
    "local_network_region", "population_segment", "institution_ruleset", "mechanism_replacement",
    "structural_hypothesis", "action_space", "outcome_contract", "full_plan",
)

PLAN_DIFF_TARGETS = (
    "outcome_contract", "entities", "populations", "institutions", "relations", "network_layers",
    "state_variables", "structural_hypotheses", "mechanisms", "parameter_packs", "action_spaces",
    "observation_models", "event_schemas", "stopping_conditions", "terminal_readouts",
)

LINEAGE_STATUSES = ("active", "superseded", "rejected", "rolled_back", "candidate", "failed")


# ============================================================ 8.1 RecompileObservation
# How an observation entered the system. ONLY `external_evidence` observations are eligible to TRIGGER a
# recompile; a `simulation_internal` event (one the running rollout sampled from the ACTIVE plan) is executed
# normally through mechanisms + StateDelta and never, by itself, causes recompilation — even if surprising
# (that is ordinary Phase-3 posterior updating). `planned_expansion` marks a predefined structural branch /
# planned transition firing, which is normal execution, not model failure.
OBSERVATION_ORIGINS = ("external_evidence", "historical_replay", "simulation_internal", "planned_expansion",
                       "internal_diagnostic")


@dataclass
class RecompileObservation(Versioned):
    SCHEMA = "phase11.observation"
    SCHEMA_VERSION = "1.0.1"
    observation_id: str
    observation_type: str = ""                 # rollcall / rule_publication / actor_statement / ...
    origin: str = "external_evidence"          # ∈ OBSERVATION_ORIGINS — gates trigger eligibility
    representable: bool = True                 # can the ACTIVE plan represent this obs at all? (support check)
    planned: bool = False                      # a predefined structural branch / planned transition firing
    event_time: float = 0.0                    # when it happened (unix)
    ingestion_time: float = 0.0                # when the model saw it (unix; >= event_time)
    evidence_ids: list = field(default_factory=list)
    source_hashes: list = field(default_factory=list)
    actor_visibility: dict = field(default_factory=dict)   # {actor_id: bool}
    related_entities: list = field(default_factory=list)
    related_institutions: list = field(default_factory=list)
    related_network_region: list = field(default_factory=list)   # entity ids spanning the region
    expected_likelihood: float = 1.0           # predictive likelihood under the CURRENT plan
    residual: float = 0.0                       # surprise score (e.g. -log predictive density)
    contradiction_links: list = field(default_factory=list)      # evidence/fact ids contradicted
    mechanism_diagnostics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    temporal_validity: dict = field(default_factory=dict)        # {valid_from, valid_until}
    uncertainty: dict = field(default_factory=dict)


# ============================================================ 8.2 RecompileTriggerEvidence
@dataclass
class RecompileTriggerEvidence(Versioned):
    SCHEMA = "phase11.trigger_evidence"
    SCHEMA_VERSION = "1.0.0"
    trigger_evidence_id: str
    trigger_family: str = ""                   # ∈ TRIGGER_FAMILIES
    affected_scope_candidates: list = field(default_factory=list)  # ∈ SCOPES
    supporting_observations: list = field(default_factory=list)
    contradictory_observations: list = field(default_factory=list)
    severity: float = 0.0                       # [0,1] estimated magnitude of the change
    persistence: float = 0.0                    # [0,1] transient vs sustained
    expected_impact: float = 0.0                # [0,1] terminal sensitivity to the change
    evidence_independence: float = 1.0          # [0,1] 1=independent sources, <1=dependent/syndicated
    trigger_probability: float = 0.0            # posterior P(genuine structural change | evidence)
    alternative_explanations: list = field(default_factory=list)  # [{explanation, prob}]
    diagnostic_method: str = ""
    thresholds_version: str = ""
    cooldown_state: dict = field(default_factory=dict)            # {family: last_fired_at, on_cooldown}
    fingerprint: str = ""                       # dedup key across detectors/rounds
    provenance: dict = field(default_factory=dict)


# ============================================================ 8.3 RecompileDecision
@dataclass
class RecompileDecision(Versioned):
    SCHEMA = "phase11.decision"
    SCHEMA_VERSION = "1.0.0"
    decision_id: str
    current_plan_id: str = ""
    current_plan_hash: str = ""
    current_plan_version: int = 0
    decision_time: float = 0.0
    trigger_evidence: list = field(default_factory=list)         # [trigger_evidence_id]
    action: str = "no_change"                   # ∈ RECOMPILE_ACTIONS
    rationale: str = ""
    uncertainty: dict = field(default_factory=dict)
    expected_value_of_recompile: float = 0.0    # E[Δpredictive] − cost, in nats or scoring units
    compute_estimate: dict = field(default_factory=dict)         # {llm_calls, tokens, seconds}
    selected_scope: str = "no_model_change"     # ∈ SCOPES
    deferred_scope: list = field(default_factory=list)
    support_grade: str = "exploratory"
    limitations: list = field(default_factory=list)


# ============================================================ 8.4 PlanRevisionCandidate
@dataclass
class PlanRevisionCandidate(Versioned):
    SCHEMA = "phase11.candidate"
    SCHEMA_VERSION = "1.0.0"
    candidate_id: str
    parent_plan_id: str = ""
    changed_components: list = field(default_factory=list)       # [{target, op, detail}]
    unchanged_components: list = field(default_factory=list)
    causal_explanation: str = ""
    supporting_evidence: list = field(default_factory=list)
    contradictory_evidence: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    candidate_mechanisms: list = field(default_factory=list)
    mechanism_applicability: dict = field(default_factory=dict)
    state_schema_changes: list = field(default_factory=list)
    entity_mapping_requirements: list = field(default_factory=list)
    network_mapping_requirements: list = field(default_factory=list)
    institution_mapping_requirements: list = field(default_factory=list)
    pending_event_implications: list = field(default_factory=list)
    complexity: float = 0.0                     # # of structural changes (parsimony penalty input)
    expected_predictive_improvement: float = 0.0
    compute_cost: dict = field(default_factory=dict)
    llm_proposal_provenance: dict = field(default_factory=dict)  # {source, prompt_hash, grounded_claim_ids}
    static_validation: dict = field(default_factory=dict)        # {ok, problems:[...]}
    is_current_plan: bool = False               # the "no change" candidate must always be present


# ============================================================ 8.5 PlanDiff (typed)
@dataclass
class TypedPlanDiffEntry(Versioned):
    SCHEMA = "phase11.plan_diff_entry"
    SCHEMA_VERSION = "1.0.0"
    target: str = ""                            # ∈ PLAN_DIFF_TARGETS
    op: str = ""                                # added / removed / modified / reweighted / remapped
    component: str = ""
    before: str = ""
    after: str = ""
    detail: str = ""
    supporting_evidence: list = field(default_factory=list)


@dataclass
class TypedPlanDiff(Versioned):
    SCHEMA = "phase11.plan_diff"
    SCHEMA_VERSION = "1.0.0"
    source_plan_hash: str = ""
    dest_plan_hash: str = ""
    entries: list = field(default_factory=list)           # [TypedPlanDiffEntry.as_record()]
    n_structural_changes: int = 0
    touched_targets: list = field(default_factory=list)   # subset of PLAN_DIFF_TARGETS

    def add(self, entry: "TypedPlanDiffEntry"):
        self.entries.append(entry.as_record())
        if entry.target and entry.target not in self.touched_targets:
            self.touched_targets.append(entry.target)
        # every diff target except a pure terminal-lean tweak counts as structural
        self.n_structural_changes += 1
        return self


# ============================================================ 8.6 MigrationPlan
@dataclass
class MigrationPlan(Versioned):
    SCHEMA = "phase11.migration_plan"
    SCHEMA_VERSION = "1.0.0"
    source_plan_hash: str = ""
    dest_plan_hash: str = ""
    migration_version: str = "1.0.0"
    simulation_time: float = 0.0
    entity_mappings: list = field(default_factory=list)     # [{src, dst, transform, evidence, uncertainty}]
    split_mappings: list = field(default_factory=list)
    merge_mappings: list = field(default_factory=list)
    population_mappings: list = field(default_factory=list)
    institution_mappings: list = field(default_factory=list)
    network_node_mappings: list = field(default_factory=list)
    edge_mappings: list = field(default_factory=list)
    state_variable_mappings: list = field(default_factory=list)
    unit_transforms: list = field(default_factory=list)
    posterior_transforms: list = field(default_factory=list)
    parameter_transforms: list = field(default_factory=list)
    history_retention: dict = field(default_factory=dict)
    evidence_retention: dict = field(default_factory=dict)
    pending_event_transforms: list = field(default_factory=list)   # [{event, disposition, reason}]
    canceled_event_reasons: list = field(default_factory=list)
    newly_scheduled_events: list = field(default_factory=list)
    orphaned_state: list = field(default_factory=list)      # [{path, reason, terminal_sensitivity}]
    rollback_reference: str = ""                            # checkpoint hash
    invariants: dict = field(default_factory=dict)          # {name: bool} checked post-migration
    uncertainty: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


# ============================================================ 8.7 Plan lineage
@dataclass
class PlanLineageNode(Versioned):
    SCHEMA = "phase11.lineage_node"
    SCHEMA_VERSION = "1.0.0"
    plan_id: str = ""
    plan_hash: str = ""
    plan_version: int = 0
    parent_plan_ids: list = field(default_factory=list)
    revision_reason: str = ""
    trigger_id: str = ""
    migration_id: str = ""
    evidence_bundle_hash: str = ""
    state_checkpoint_hash: str = ""
    posterior_weight: float = 1.0
    creation_time: float = 0.0
    simulation_time: float = 0.0
    code_commit: str = ""
    compiler_version: str = ""
    mechanism_registry_version: str = ""
    status: str = "candidate"                   # ∈ LINEAGE_STATUSES
    failure_reason: str = ""


@dataclass
class PlanLineageEdge(Versioned):
    SCHEMA = "phase11.lineage_edge"
    SCHEMA_VERSION = "1.0.0"
    parent_plan_id: str = ""
    child_plan_id: str = ""
    trigger_id: str = ""
    migration_id: str = ""
    reason: str = ""


# ============================================================ 8.8 RecompilationTrace
@dataclass
class RecompilationTrace(Versioned):
    SCHEMA = "phase11.trace"
    SCHEMA_VERSION = "1.0.0"
    trace_id: str = ""
    simulation_id: str = ""
    simulation_time: float = 0.0
    observations: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    trigger_posterior: dict = field(default_factory=dict)       # {family: prob} fused
    selected_scope: str = ""
    scope_alternatives: list = field(default_factory=list)
    candidates: list = field(default_factory=list)              # [candidate.as_record()]
    scores: list = field(default_factory=list)                  # [{candidate_id, components, total, weight}]
    rejected_candidates: list = field(default_factory=list)
    decision: dict = field(default_factory=dict)                # RecompileDecision.as_record()
    migration_report: dict = field(default_factory=dict)
    before_state_summary: dict = field(default_factory=dict)
    after_state_summary: dict = field(default_factory=dict)
    event_queue_diff: dict = field(default_factory=dict)
    plan_mixture: list = field(default_factory=list)            # [{plan_hash, weight}]
    lineage: dict = field(default_factory=dict)                 # {nodes, edges}
    continued_rollout: dict = field(default_factory=dict)
    terminal_effect: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    checksums: dict = field(default_factory=dict)
    events_emitted: list = field(default_factory=list)          # [recompile_* event records]
