"""Phase 13 decision contracts — the typed `DecisionProblem` and `DecisionResult` (Parts 2, 21, 24).

The forecast question is NOT the objective: "will the agreement pass?" is a prediction target; "what
should the negotiator do?" needs its own utility, constraints and authority. A `DecisionProblem` that
cannot say whose decision it is, what they control, and what they want, is UNDERSPECIFIED — the layer
then returns a Pareto frontier + missing-preference report + abstention, never a fabricated scalar.

Every reported effect carries a causal-claim label (Part 24): simulated lift is never presented as
identified real lift.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from swm.world_model_v2.state import parse_time

# ---------------------------------------------------------------- causal-claim labels (Part 24)
CAUSAL_CLAIM_LABELS = (
    "predictive_association",
    "simulated_mechanism_counterfactual",       # what the world model produces
    "identified_experimental_effect",           # RCT / A/B
    "identified_quasi_experimental_effect",     # DiD / RD / IV / natural experiment
    "off_policy_estimate",                      # IPS / DR on logged decisions
    "observational_estimate_with_assumptions",
    "unsupported_hypothesis",
)


def claim_label_valid(label: str) -> bool:
    return label in CAUSAL_CLAIM_LABELS


# ---------------------------------------------------------------- stakeholders + utility spec (Part 3)
@dataclass
class Stakeholder:
    """One party whose welfare enters the objective. `utility_fn(outcome_vector) -> float` maps the
    per-particle outcome readout to this stakeholder's utility; weight is used by weighted aggregation;
    `floor` is a minimum-guarantee (lexicographic: any action violating a floor is dominated regardless
    of aggregate gain); `rights` are noncompensable constraints (predicate on outcome; violation is a
    hard rejection, not a penalty)."""
    stakeholder_id: str
    utility_fn: object = None                       # callable(outcome: dict) -> float
    weight: float = 1.0
    floor: float = None                             # minimum acceptable utility (None = none)
    rights: list = field(default_factory=list)      # [callable(outcome)->bool] True = right respected

    def utility(self, outcome: dict) -> float:
        return float(self.utility_fn(outcome)) if self.utility_fn else 0.0


AGGREGATIONS = ("weighted_sum", "lexicographic", "maximin", "nash_social_welfare",
                "minimax_regret", "cvar", "chance_constrained", "pareto_only")


@dataclass
class UtilitySpec:
    """The decision-maker's explicit objective. `provenance` records WHERE the utility came from
    (user-supplied | domain-default | underspecified) — a Phase-13 result always shows its utility
    decomposition, never one undocumented scalar."""
    stakeholders: list = field(default_factory=list)     # [Stakeholder]
    aggregation: str = "weighted_sum"
    cvar_alpha: float = 0.2                              # for cvar aggregation / risk report
    discount_per_day: float = 0.0                        # exponential discounting of dated utilities
    provenance: str = "user_supplied"

    def validate(self) -> list:
        errs = []
        if self.aggregation not in AGGREGATIONS:
            errs.append(f"unknown aggregation {self.aggregation!r} (valid: {AGGREGATIONS})")
        if not self.stakeholders:
            errs.append("no stakeholders declared — utility is underspecified")
        for s in self.stakeholders:
            if s.utility_fn is None:
                errs.append(f"stakeholder {s.stakeholder_id!r} has no utility_fn")
        return errs


# ---------------------------------------------------------------- constraints (Part 2)
@dataclass
class ConstraintSpec:
    """Hard constraints reject an action outright (typed reason); soft constraints enter the utility as
    documented penalties; chance constraints bound P(bad event) over the matched particle set."""
    constraint_id: str
    kind: str = "hard"                              # hard | soft | chance
    description: str = ""
    # hard/soft: predicate over the ACTION (pre-simulation) or the terminal outcome (post-simulation)
    action_pred: object = None                      # callable(ActionSchema)->bool  True = SATISFIED
    outcome_pred: object = None                     # callable(outcome)->bool       True = SATISFIED
    penalty: float = 0.0                            # soft: subtracted per unit violation
    max_prob: float = 0.0                           # chance: max allowed P(outcome_pred is False)


@dataclass
class RiskSpec:
    """Risk / robustness preferences. `downside_limit` is the utility level counted as material harm;
    `robustness` selects the ranking objective (expected | cvar | lower_confidence | minimax_regret |
    worst_hypothesis)."""
    tolerance: str = "neutral"                      # averse | neutral | seeking
    robustness: str = "expected"
    downside_limit: float = None
    lower_confidence: float = 0.2                   # q for lower-confidence ranking
    ambiguity_aversion: float = 0.0                 # weight on worst-structural-hypothesis value


# ---------------------------------------------------------------- the decision problem (Part 2)
@dataclass
class DecisionProblem:
    """The complete typed decision contract. Compiled BEFORE any action generation or simulation.

    Authority/resources say what the decision-maker can actually do; `generated_action_permission`
    gates affordance-based candidate generation (user-supplied candidates are always allowed through
    feasibility); `prohibited` are hard exclusions checked before simulation. Utility is NEVER inferred
    silently from the forecast question — if `utility.provenance == 'underspecified'` the layer returns
    a Pareto frontier + abstention from single-action selection."""
    decision_id: str
    decision_maker: str                              # actor id in the world
    role: str = ""                                   # decision-maker's role (provenance for authority)
    authority: list = field(default_factory=list)    # capability strings, e.g. "send_message", "set:price"
    controllable_resources: dict = field(default_factory=dict)   # name -> amount available
    context: str = ""                                # free-text decision context (recorded)
    as_of: str = ""                                  # RFC3339 information cutoff
    horizon: str = ""                                # RFC3339 end of evaluation window
    deadlines: list = field(default_factory=list)    # [{"what":..., "ts": rfc3339}]
    decision_points: list = field(default_factory=list)  # rfc3339 times where a policy may act
    information_set: dict = field(default_factory=dict)  # what the decision-maker observes now
    private_information: dict = field(default_factory=dict)
    candidate_actions: list = field(default_factory=list)  # user-supplied ActionSchema (optional)
    generated_action_permission: bool = True
    prohibited: list = field(default_factory=list)   # operation names / predicates never allowed
    constraints: list = field(default_factory=list)  # [ConstraintSpec]
    utility: UtilitySpec = field(default_factory=UtilitySpec)
    risk: RiskSpec = field(default_factory=RiskSpec)
    implementation_costs: dict = field(default_factory=dict)  # action_id/operation -> cost in utility units
    switching_cost: float = 0.0
    reversibility_required: bool = False             # if True, irreversible actions are infeasible
    information_gathering_allowed: bool = True
    human_approval_required: bool = True             # recommendations are never auto-executed (Part 23)
    output: dict = field(default_factory=dict)       # output requirements (e.g. {"pareto": True})

    # ---- governance -------------------------------------------------------------
    def validate(self) -> list:
        """Typed contract validation. Returns a list of defects (empty = valid)."""
        errs = []
        if not self.decision_id:
            errs.append("decision_id required")
        if not self.decision_maker:
            errs.append("decision_maker (actor id) required — a decision needs a decider")
        if self.as_of:
            try:
                parse_time(self.as_of)
            except ValueError as e:
                errs.append(f"as_of unparseable: {e}")
        if self.horizon:
            try:
                parse_time(self.horizon)
            except ValueError as e:
                errs.append(f"horizon unparseable: {e}")
        errs.extend(self.utility.validate())
        for c in self.constraints:
            if c.kind not in ("hard", "soft", "chance"):
                errs.append(f"constraint {c.constraint_id}: unknown kind {c.kind!r}")
            if c.kind == "chance" and not (0.0 <= c.max_prob <= 1.0):
                errs.append(f"chance constraint {c.constraint_id}: max_prob must be in [0,1]")
        return errs

    def underspecification(self) -> list:
        """What is MISSING for a single-action recommendation (drives Pareto+abstention, Part 2)."""
        missing = []
        if self.utility.provenance == "underspecified" or not self.utility.stakeholders:
            missing.append("utility: no stakeholder utility supplied — cannot rank on one scalar")
        if len(self.utility.stakeholders) > 1 and self.utility.aggregation == "pareto_only":
            missing.append("aggregation: multiple stakeholders with no aggregation rule — Pareto only")
        if not self.authority and not self.candidate_actions:
            missing.append("authority: decision-maker has no declared authority and no candidate actions")
        return missing

    def contract_hash(self) -> str:
        payload = json.dumps({
            "id": self.decision_id, "maker": self.decision_maker, "authority": sorted(self.authority),
            "resources": self.controllable_resources, "as_of": self.as_of, "horizon": self.horizon,
            "prohibited": [str(p) for p in self.prohibited],
            "aggregation": self.utility.aggregation,
            "stakeholders": [s.stakeholder_id for s in self.utility.stakeholders],
            "risk": [self.risk.tolerance, self.risk.robustness]}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- abstention (Part 23)
@dataclass
class Abstention:
    """A principled refusal to pick one action, WITH what is needed to proceed. This is a first-class
    result, not an error."""
    reasons: list = field(default_factory=list)          # typed reason codes + text
    needed: list = field(default_factory=list)           # information/authorization needed to proceed
    partial: dict = field(default_factory=dict)          # whatever WAS computable (e.g. Pareto frontier)

    def as_dict(self):
        return {"abstained": True, "reasons": self.reasons, "needed": self.needed,
                "partial": {k: v for k, v in self.partial.items() if k != "evaluations"}}


# ---------------------------------------------------------------- the result (Part 21)
@dataclass
class DecisionResult:
    """Everything a caller needs to trust or challenge the recommendation — never only a string."""
    decision_id: str
    contract_hash: str = ""
    runtime_fingerprint: dict = field(default_factory=dict)
    reference_action: str = ""                        # explicit baseline (do_nothing unless overridden)
    evaluated: list = field(default_factory=list)     # [per-action evaluation dicts (robust.py schema)]
    policies: list = field(default_factory=list)      # [per-policy evaluation dicts]
    recommended: str = None                           # action_id | policy_id | None (abstained)
    recommendation_kind: str = "action"               # action | policy | pareto | abstain | gather_information
    pareto_frontier: list = field(default_factory=list)
    feasibility: list = field(default_factory=list)   # [FeasibilityVerdict.as_dict()] incl. rejections
    counterfactual: dict = field(default_factory=dict)  # paired-difference block vs reference
    value_of_information: dict = field(default_factory=dict)
    search: dict = field(default_factory=dict)        # method, evaluations, budget, diagnostics
    causal_claim: str = "simulated_mechanism_counterfactual"
    empirical_validation: str = "not_validated_on_this_decision"
    support_grade: str = "exploratory"
    # §31 recommendation axis (same vocabulary as result.RECOMMENDATION_STATUSES): "withheld"
    # when an under-modeled subtype is present on the ensemble result, when the winner does not
    # survive every admissible completion of truncated branch mass (§21), or when >1 model
    # family was configured but the recommendation was exercised under only one. Existing gates
    # (abstention / pareto / gather_information) keep deciding `recommendation_kind`; this axis
    # rides alongside them and never silently relaxes them.
    recommendation_status: str = "not_requested"
    abstention: dict = None
    active_phases: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)    # crn manifest, plan hashes, delta counts, seeds
    cost: dict = field(default_factory=dict)
    latency_s: float = 0.0
    seed: int = 0

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "decision_id", "contract_hash", "runtime_fingerprint", "reference_action", "evaluated",
            "policies", "recommended", "recommendation_kind", "pareto_frontier", "feasibility",
            "counterfactual", "value_of_information", "search", "causal_claim",
            "empirical_validation", "support_grade", "recommendation_status", "abstention",
            "active_phases", "provenance", "cost", "latency_s", "seed")}
        return d

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), default=_json_default, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "DecisionResult":
        obj = json.loads(s)
        r = cls(decision_id=obj.get("decision_id", ""))
        for k, v in obj.items():
            if hasattr(r, k):
                setattr(r, k, v)
        return r

    def result_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def _json_default(o):
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if callable(o):
        return f"<fn:{getattr(o, '__name__', 'lambda')}>"
    return str(o)
