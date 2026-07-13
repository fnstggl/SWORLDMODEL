"""Phase 8 — persistent-state contracts, variable registry, and the Phase-4 integration surface.

This module defines the TYPED vocabulary of persistent, longitudinal state (Part 1 + the parallel-branch
integration contract). It deliberately owns only the contracts — the durable event log lives in
``phase8_events``, sequential inference in ``phase8_filtering``, in-world transitions in
``phase8_transitions``, checkpoint/lineage in ``phase8_service``, and the WorldState bridge in
``phase8_materialize``.

Design stance (anti-scaffolding): a persistent variable is NOT a scalar with a decay constant. Every
variable declares (i) its semantic meaning and scope, (ii) the evidence and events that update it, (iii)
the observation model and transition FAMILY that carries it forward, (iv) an explicit posterior
representation, and (v) which actor-visible field it materializes into so a mechanism actually consumes it.
A spec with no ``materializes_into`` and no downstream consumer is refused registration — an ornamental
variable cannot enter the registry.

The Phase-4 contract (``persistent_features_for_policy`` / ``PolicyFeedbackEvent``) is a stable, typed
boundary so Phase-4 actor-policy learning can read persistent posteriors and emit outcomes back WITHOUT
Phase 8 rewriting Phase-4-owned code (the branches evolve in parallel).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

SCHEMA_VERSION = "phase8-persistence-1.0"

# ------------------------------------------------------------------ scope + posterior families
SCOPES = ("actor", "dyad", "relationship", "population", "institution", "network_edge", "world")
POSTERIOR_FAMILIES = ("beta_bernoulli", "beta_binomial", "gaussian_state", "dirichlet_count",
                      "categorical_stage", "particle", "point")
#: transition families implemented in phase8_transitions (Part 5). Registration validates against this set.
TRANSITION_FAMILIES = (
    "reinforcement", "asymmetric_reward_punishment", "decay", "mean_reversion", "bayesian_belief",
    "habit_accumulation", "trust_asymmetric", "trust_repair", "relationship_strengthen",
    "relationship_decay", "reputation_accrual", "commitment_create", "commitment_fulfill",
    "commitment_violate", "resource_flow", "risk_adaptation", "learned_strategy",
    "institutional_stage", "hysteresis", "path_dependence", "memory_consolidation",
    "memory_interference", "retrieval_strengthening", "forgetting")

IDENTIFIABILITY = ("identified", "weakly_identified", "unidentified")
PROVENANCE_STATUS = ("observed", "fitted", "reference_pack", "hierarchical_prior", "broad_prior")


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def logit(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, float(p)))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x < -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ------------------------------------------------------------------ Part 1: the typed variable spec
@dataclass(frozen=True)
class PersistentStateKey:
    """Addresses one persistent variable instance in one scenario. Stable across a scenario's whole
    trajectory and across checkpoints — it is the join key between the event log, the posterior store,
    and the materialized WorldState field."""
    world_id: str
    scenario_id: str
    scope: str                              # SCOPES
    entity_id: str                          # actor / dyad "a|b" / institution / edge "src|layer|dst" / "world"
    variable_id: str                        # e.g. "engagement_propensity", "trust", "institutional_stage"

    def as_tuple(self):
        return (self.world_id, self.scenario_id, self.scope, self.entity_id, self.variable_id)

    def token(self) -> str:
        return "::".join(str(x) for x in self.as_tuple())

    def __str__(self):
        return self.token()


@dataclass
class PersistentVariableSpec:
    """The full typed declaration of a persistent variable (Part 1). Numbers do NOT live here — this is the
    STRUCTURE (semantics, causal graph, observation/transition families, materialization target). The
    posterior VALUES live in ``PersistentStatePosterior``; the transition PARAMETERS live in a fitted /
    reference / broad parameter posterior referenced by ``transition_param_source``."""
    variable_id: str
    definition: str                          # what the value MEANS (whose, over what, in what units)
    scope: str                               # SCOPES
    support: str = "unit_interval"           # unit_interval | nonneg_real | real | categorical | count | simplex
    units: str = ""
    posterior_family: str = "beta_bernoulli"  # POSTERIOR_FAMILIES
    transition_family: str = "reinforcement"  # TRANSITION_FAMILIES
    transition_param_source: str = "broad_prior"  # PROVENANCE_STATUS — where the transition params come from
    observation_model: str = ""              # how events are read as observations (evidence class)
    prior_source: str = "broad_prior"        # PROVENANCE_STATUS
    causal_parents: tuple = ()               # variable_ids that drive this one
    causal_children: tuple = ()              # variable_ids this one drives
    evidence_dependencies: tuple = ()        # event types / evidence classes that update it
    update_triggers: tuple = ()              # event types that trigger an update
    reset_conditions: tuple = ()             # event types that reset it (e.g. regime change)
    expected_timescale: str = "event"        # event | hours | days | weeks | months
    actor_visibility: str = "self"           # self | public | dyad | private | institutional
    memory_accessibility: str = "recallable"  # recallable | summarized | inaccessible
    terminal_sensitivity: float = 0.5        # expected influence on the terminal outcome (0..1)
    identifiability: str = "weakly_identified"  # IDENTIFIABILITY
    materializes_into: str = ""              # WorldState field path the value writes to (REQUIRED, non-ornamental)
    consumed_by: tuple = ()                  # mechanisms/policy families that READ the materialized field
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> "PersistentVariableSpec":
        if self.scope not in SCOPES:
            raise ValueError(f"unknown scope {self.scope!r} (known: {SCOPES})")
        if self.posterior_family not in POSTERIOR_FAMILIES:
            raise ValueError(f"unknown posterior_family {self.posterior_family!r}")
        if self.transition_family not in TRANSITION_FAMILIES:
            raise ValueError(f"unknown transition_family {self.transition_family!r}")
        # ANTI-ORNAMENTAL: a persistent variable MUST name a WorldState field it materializes into AND a
        # consumer that reads it. A variable that changes nothing downstream is not persistence — it is
        # storage, and Phase 8 explicitly refuses to count storage-only state.
        if not self.materializes_into:
            raise ValueError(f"persistent variable {self.variable_id!r} declares no materializes_into target "
                             "— an unconsumed variable is ornamental (Phase 8 anti-scaffolding rule)")
        if not self.consumed_by:
            raise ValueError(f"persistent variable {self.variable_id!r} names no consumer — storage that no "
                             "mechanism reads does not count as persistence")
        return self

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def spec_hash(self) -> str:
        return _hash(self.as_dict())


# ------------------------------------------------------------------ Part 4: the posterior over a variable
@dataclass
class PersistentStatePosterior:
    """The sequential posterior over ONE persistent variable AFTER assimilating events up to ``as_of``.

    Carries both the value posterior (mean/sd + a compact representation) AND the transition-parameter
    posterior, plus the full sequential-inference diagnostics the acceptance gates require (prior→posterior,
    likelihood, ESS, resample decision, lineage, seed). This is what a checkpoint stores and what
    materialization reads."""
    key: PersistentStateKey
    variable_id: str
    posterior_family: str
    mean: float = 0.5
    sd: float = 0.29
    representation: dict = field(default_factory=dict)   # family-specific sufficient stats (e.g. {"a":..,"b":..})
    prior_mean: float = 0.5
    transition_params: dict = field(default_factory=dict)  # e.g. {"decay": .., "gain": .., "loss": ..} + source
    n_events_assimilated: int = 0
    n_effective_observations: float = 0.0
    ess: float = 0.0
    resampled: bool = False
    as_of: float = 0.0
    method: str = ""
    seed: int = 0
    lineage: list = field(default_factory=list)          # [PersistentUpdateRecord.as_dict()] — the update trail
    support_grade: str = "exploratory"
    diagnostics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def value(self) -> float:
        return self.mean

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["key"] = self.key.token()
        return d

    def posterior_hash(self) -> str:
        return _hash({"key": self.key.token(), "mean": round(self.mean, 6), "sd": round(self.sd, 6),
                      "rep": {k: round(v, 6) if isinstance(v, float) else v
                              for k, v in self.representation.items()},
                      "n": self.n_events_assimilated})


@dataclass
class PersistentUpdateRecord:
    """One step in the sequential filter's lineage: prior → observation → posterior. Every filtered
    variable accumulates these so a reviewer can trace the terminal-affecting value back to a specific
    event, and so smoothing vs filtering is auditable."""
    at: float
    event_id: str
    prior_mean: float
    obs_value: object
    obs_loglik: float
    posterior_mean: float
    ess_after: float
    mode: str = "filter"                    # filter | smooth — leakage-critical distinction

    def as_dict(self):
        return {"at": self.at, "event_id": self.event_id, "prior_mean": round(self.prior_mean, 6),
                "obs_value": self.obs_value, "obs_loglik": round(self.obs_loglik, 6),
                "posterior_mean": round(self.posterior_mean, 6), "ess_after": round(self.ess_after, 3),
                "mode": self.mode}


# ------------------------------------------------------------------ the actor-visible projection
@dataclass
class PersistentStateView:
    """An actor's VISIBLE slice of persistent state at a time (Part 6/7). The omniscient log holds
    everything; an actor sees only variables whose ``actor_visibility`` admits them AND (for episodic
    memory) only what they can plausibly retrieve. This is the object Phase-4 policy code reads through
    ``persistent_features_for_policy`` — it never receives the raw event log or another actor's private
    posteriors."""
    actor_id: str
    as_of: float
    beliefs: dict = field(default_factory=dict)          # variable_id -> posterior mean (self-visible)
    trust: dict = field(default_factory=dict)            # other_actor -> trust posterior mean
    commitments: list = field(default_factory=list)
    habits: dict = field(default_factory=dict)           # action -> strength
    reputation: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    risk_tolerance: float | None = None
    institutional_stage: dict = field(default_factory=dict)
    recalled_events: list = field(default_factory=list)  # actor-retrievable memory traces
    uncertainty: dict = field(default_factory=dict)      # variable_id -> sd
    provenance: dict = field(default_factory=dict)

    def as_dict(self):
        return self.__dict__.copy()

    def view_hash(self) -> str:
        return _hash(self.as_dict())


@dataclass
class MemoryTrace:
    """An actor-specific, retrievable memory of a persistent event (Part 6). Salience/recency/interference
    determine retrieval probability; recall is not guaranteed. Wraps the ``swm.memory`` episodic substrate
    with the persistent-state key so a recalled event is joined back to the variable it updated."""
    entity_id: str
    at: float
    text: str
    salience: float = 0.5
    source_event_id: str = ""
    variable_id: str = ""
    retrieval_prob: float = 1.0

    def as_dict(self):
        return self.__dict__.copy()


# ------------------------------------------------------------------ the explicit mutation record
@dataclass
class PersistentStateDelta:
    """The machine-readable record of one persistent-state mutation (mirrors ``transitions.StateDelta`` but
    for the persistence plane). EVERY change to persistent state — whether a sequential-filter update or an
    in-world transition — emits one of these. No silent in-place mutation is permitted (Part 7)."""
    at: float
    variable_id: str
    key: str
    transition_family: str
    before: object
    after: object
    driven_by_event_id: str = ""
    reason_codes: list = field(default_factory=list)
    uncertainty: dict = field(default_factory=dict)
    evidence_deps: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def as_dict(self):
        return self.__dict__.copy()


@dataclass
class PersistentLineage:
    """The provenance chain from the event log through the posterior to the materialized world (Part 9).
    A checkpoint stores this so a value can be traced to its genesis events, transition params, and code
    versions — the audit trail that makes 'delete history → change execution' verifiable."""
    key: str
    event_watermark: str = ""               # running hash of the event log at checkpoint time
    genesis_event_ids: list = field(default_factory=list)
    posterior_hash: str = ""
    transition_param_source: str = ""
    code_versions: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def as_dict(self):
        return self.__dict__.copy()


# ------------------------------------------------------------------ the variable registry (canonical specs)
_VARIABLE_REGISTRY: dict = {}


def register_persistent_variable(spec: PersistentVariableSpec) -> PersistentVariableSpec:
    """Register a typed persistent variable after validating it is non-ornamental. Idempotent by
    variable_id; re-registering the same id replaces the spec (used by tests / scenario packs)."""
    spec.validate()
    _VARIABLE_REGISTRY[spec.variable_id] = spec
    return spec


def get_persistent_variable(variable_id: str) -> PersistentVariableSpec:
    if variable_id not in _VARIABLE_REGISTRY:
        raise KeyError(f"no persistent variable {variable_id!r} registered "
                       f"(known: {sorted(_VARIABLE_REGISTRY)})")
    return _VARIABLE_REGISTRY[variable_id]


def registered_variables() -> dict:
    return dict(_VARIABLE_REGISTRY)


# ---- canonical specs: the core persistent families, each wired to a real WorldState consumer ----------
# Every spec below names materializes_into + consumed_by, so it flows through the existing ActorView→policy
# path (habit reads action_history; reinforcement reads policy_state Q-values; reciprocity reads relationship
# trust; utility reads beliefs/resources). This is what makes removing history change the action distribution.
_CANONICAL = [
    PersistentVariableSpec(
        variable_id="engagement_propensity", scope="actor", support="unit_interval",
        definition="P(this actor takes a positive engagement action on a passive exposure), a latent that "
                   "persists across their real event sequence (bursts/refractoriness), NOT an i.i.d. draw",
        posterior_family="beta_bernoulli", transition_family="reinforcement",
        transition_param_source="fitted",
        observation_model="binary acted/not-acted on each passive exposure event",
        prior_source="hierarchical_prior", evidence_dependencies=("passive_exposure",),
        update_triggers=("passive_exposure",), expected_timescale="days",
        actor_visibility="self", terminal_sensitivity=0.8, identifiability="identified",
        materializes_into="entity.latent_state[phase4_policy_value:engage]",
        consumed_by=("phase4_policy.reinforcement_learning", "phase4_policy.habit",
                     "generic_outcome_readout")),
    PersistentVariableSpec(
        variable_id="habit_strength", scope="actor", support="nonneg_real",
        definition="accumulated tendency to repeat a specific action from its own reinforcement history",
        posterior_family="dirichlet_count", transition_family="habit_accumulation",
        transition_param_source="broad_prior", observation_model="count of past occurrences of the action",
        prior_source="broad_prior", update_triggers=("actor_action",), expected_timescale="weeks",
        actor_visibility="self", terminal_sensitivity=0.5,
        materializes_into="entity.past_actions", consumed_by=("phase4_policy.habit",)),
    PersistentVariableSpec(
        variable_id="trust", scope="dyad", support="unit_interval",
        definition="actor A's trust in actor B for a class of action/information, updated asymmetrically "
                   "(slow gain, fast loss) with an explicit repair path",
        posterior_family="beta_bernoulli", transition_family="trust_asymmetric",
        transition_param_source="reference_pack",
        observation_model="promise-kept / promise-broken / cooperative / defect events between A and B",
        prior_source="broad_prior", causal_children=("relationship_strength",),
        evidence_dependencies=("promise_fulfilled", "promise_violated", "cooperative_act", "defection"),
        update_triggers=("promise_fulfilled", "promise_violated", "cooperative_act", "defection"),
        reset_conditions=("relationship_reset",), expected_timescale="weeks", actor_visibility="dyad",
        terminal_sensitivity=0.7, materializes_into="network.edge.trust",
        consumed_by=("phase4_policy.reciprocity", "relationship_update")),
    PersistentVariableSpec(
        variable_id="relationship_strength", scope="relationship", support="unit_interval",
        definition="tie strength between two actors, strengthened by interaction and decayed by inactivity",
        posterior_family="beta_bernoulli", transition_family="relationship_strengthen",
        transition_param_source="broad_prior", observation_model="interaction / inactivity events",
        prior_source="broad_prior", causal_parents=("trust",),
        update_triggers=("interaction", "inactivity"), expected_timescale="months",
        actor_visibility="dyad", terminal_sensitivity=0.5, materializes_into="network.edge.strength",
        consumed_by=("phase4_policy.social_proof", "relationship_update")),
    PersistentVariableSpec(
        variable_id="reputation", scope="actor", support="unit_interval",
        definition="public standing accumulated from observed outcomes; recovers slowly after damage",
        posterior_family="beta_bernoulli", transition_family="reputation_accrual",
        transition_param_source="broad_prior", observation_model="publicly observed success/failure/sanction",
        prior_source="broad_prior", update_triggers=("public_outcome", "sanction"),
        expected_timescale="months", actor_visibility="public", terminal_sensitivity=0.5,
        materializes_into="entity.beliefs[reputation]", consumed_by=("phase4_policy.limited_depth_reasoning",)),
    PersistentVariableSpec(
        variable_id="commitment", scope="actor", support="categorical",
        definition="lifecycle state of a promise/obligation (open→fulfilled|violated), gating later actions",
        posterior_family="categorical_stage", transition_family="commitment_create",
        transition_param_source="observed", observation_model="commitment create/fulfill/violate events",
        prior_source="observed", update_triggers=("commitment_created", "promise_fulfilled",
                                                   "promise_violated"),
        expected_timescale="weeks", actor_visibility="self", terminal_sensitivity=0.6,
        materializes_into="entity.commitments", consumed_by=("phase4_feasibility", "phase4_policy.obligation")),
    PersistentVariableSpec(
        variable_id="resource_level", scope="actor", support="nonneg_real", units="unit",
        definition="depletable/accruable resource stock constraining feasible actions",
        posterior_family="gaussian_state", transition_family="resource_flow",
        transition_param_source="observed", observation_model="resource gain/spend events",
        prior_source="observed", update_triggers=("resource_gain", "resource_spend"),
        expected_timescale="days", actor_visibility="self", terminal_sensitivity=0.5,
        materializes_into="entity.resources", consumed_by=("phase4_feasibility", "resource_update")),
    PersistentVariableSpec(
        variable_id="risk_tolerance", scope="actor", support="unit_interval",
        definition="adaptive risk appetite that shifts toward/away from risk after gains/losses",
        posterior_family="gaussian_state", transition_family="risk_adaptation",
        transition_param_source="broad_prior", observation_model="realized gain/loss events",
        prior_source="broad_prior", update_triggers=("realized_gain", "realized_loss"),
        expected_timescale="weeks", actor_visibility="self", terminal_sensitivity=0.4,
        materializes_into="entity.beliefs[risk_tolerance]", consumed_by=("phase4_policy.risk_sensitive",)),
    PersistentVariableSpec(
        variable_id="institutional_stage", scope="institution", support="categorical",
        definition="the process stage/queue position an actor's case has reached in an institution, "
                   "path-dependent (a stage reached after appeal differs from the same stage reached directly)",
        posterior_family="categorical_stage", transition_family="institutional_stage",
        transition_param_source="observed", observation_model="stage-transition / decision / appeal events",
        prior_source="observed", update_triggers=("stage_transition", "decision", "appeal"),
        expected_timescale="weeks", actor_visibility="institutional", terminal_sensitivity=0.7,
        materializes_into="entity.latent_state[institutional_stage]",
        consumed_by=("phase4_feasibility", "institutional_vote")),
]
for _spec in _CANONICAL:
    register_persistent_variable(_spec)


# ------------------------------------------------------------------ Part 4/anti-scaffolding: the Phase-4 contract
@dataclass
class PolicyFeedbackEvent:
    """A typed outcome Phase-4 emits back into Phase 8 after an action (Part 4). Phase 8 converts it into a
    ``PersistentEvent`` and a sequential update, so the next decision sees changed persistent state. This is
    the closed loop: persistent state → action distribution → outcome → persistent update → next action."""
    at: float
    actor_id: str
    action_name: str
    outcome: str                            # reward | failure | sanction | promise_fulfilled | promise_violated
    #                                         | response_received | institutional_decision | resource_change
    #                                         | trust_event | relationship_event
    reward: float = 0.0
    target_id: str = ""
    magnitude: float = 0.0
    source_event_id: str = ""
    payload: dict = field(default_factory=dict)

    def as_dict(self):
        return self.__dict__.copy()


def persistent_features_for_policy(view: PersistentStateView) -> dict:
    """The stable read surface Phase-4 policy code consumes (Part 4). Returns a flat, typed feature dict from
    an actor's VISIBLE persistent state — belief/trust/commitment/habit/reputation/resource/risk/stage plus
    per-feature uncertainty and provenance. Phase-4 never touches the event log or another actor's private
    posteriors; it reads only this projection. Deliberately additive: new persistent variables surface here
    without changing Phase-4's signature."""
    feats = {}
    for vid, m in (view.beliefs or {}).items():
        feats[f"belief:{vid}"] = float(m)
    for other, t in (view.trust or {}).items():
        feats[f"trust:{other}"] = float(t)
    for action, s in (view.habits or {}).items():
        feats[f"habit:{action}"] = float(s)
    for k, v in (view.reputation or {}).items():
        feats[f"reputation:{k}"] = float(v)
    for k, v in (view.resources or {}).items():
        feats[f"resource:{k}"] = float(v)
    if view.risk_tolerance is not None:
        feats["risk_tolerance"] = float(view.risk_tolerance)
    for k, v in (view.institutional_stage or {}).items():
        feats[f"stage:{k}"] = v
    feats["_n_open_commitments"] = float(len(view.commitments or []))
    feats["_n_recalled_events"] = float(len(view.recalled_events or []))
    return {"features": feats, "uncertainty": dict(view.uncertainty or {}),
            "provenance": dict(view.provenance or {}), "as_of": view.as_of,
            "schema_version": SCHEMA_VERSION}
