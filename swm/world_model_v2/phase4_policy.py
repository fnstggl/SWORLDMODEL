"""Phase 4 production actor-policy contracts.

This module is deliberately domain-general.  A compiler may propose semantic action
descriptions, but it cannot mint probabilities or bypass the contracts below:

    WorldExecutionPlan -> posterior WorldState particles -> ActorView
    -> scenario ActionSpace -> perceived/actual feasibility
    -> policy-family posterior -> calibrated action posterior

The execution side lives in :mod:`phase4_execution`; learning and artifact handling
live in :mod:`phase4_learning`.  The split keeps policy computation actor-local and
makes it difficult to accidentally hand an omniscient ``WorldState`` to a fitted
numeric policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

from swm.world_model_v2.state import StateField, register_entity_extension


SCHEMA_VERSION = "4.0.0"

ACTION_FAMILIES = (
    "messaging", "negotiation", "participation", "platform", "institutional",
    "organizational_market", "generic",
)

ACTION_ONTOLOGY = {
    "messaging": ("reply_now", "reply_later", "acknowledge", "clarify", "delegate", "ignore",
                  "follow_up", "escalate_message", "reveal_information", "withhold_information"),
    "negotiation": ("accept", "reject", "counteroffer", "concede", "hold_position", "delay",
                    "escalate", "reveal", "conceal", "exit", "seek_mediator"),
    "participation": ("support", "oppose", "abstain", "volunteer", "donate", "persuade",
                      "mobilize", "defect", "coordinate", "protest", "strike", "withdraw"),
    "platform": ("ignore", "view", "click", "like", "comment", "share", "report", "follow",
                 "unfollow", "create_content", "delete_content"),
    "institutional": ("approve", "reject", "amend", "defer", "veto", "refer", "escalate",
                      "enforce", "appeal", "schedule", "place_on_agenda", "allocate_resource"),
    "organizational_market": ("hire", "fire", "recommend", "authorize", "purchase", "sell",
                              "launch", "delay_launch", "acquire", "withdraw_offer",
                              "allocate_budget", "request_approval"),
    "generic": ("act", "wait", "abstain", "exit"),
}

KNOWN_ACTIONS = {name: family for family, names in ACTION_ONTOLOGY.items() for name in names}

FEASIBILITY_STATUSES = (
    "feasible", "feasible_with_uncertainty", "temporarily_unavailable",
    "institutionally_prohibited", "outside_authority", "physically_impossible", "unaffordable",
    "insufficient_information", "precondition_unmet", "binding_commitment_conflict",
    "unknown_to_actor", "unsupported_action_semantics",
)

POLICY_FAMILIES = (
    "random_utility", "multinomial_logit", "nested_discrete_choice", "quantal_response",
    "satisficing", "bounded_search", "habit", "reinforcement_learning", "ewa",
    "belief_planning", "norm_compliance", "obligation", "reciprocity", "imitation",
    "social_proof", "institutional_obedience", "risk_sensitive", "loss_aversion",
    "strategic_anticipation", "limited_depth_reasoning", "delay_hazard", "regime_mixture",
)

PRIVATE_ACTOR_FIELDS = {"private_information", "latent_state"}
SIMULATOR_ONLY_FIELDS = {
    "uncertainty_meta", "versions", "evidence_hash", "omissions", "parent_version",
    "terminal_probability", "resolution_outcome", "future_events", "posterior_truth",
}

register_entity_extension("phase4_actor_policy", fields={
    "incentives": "actor-visible situational incentives with provenance",
    "obligations": "actor-visible obligations distinct from stable preferences",
    "expected_reactions": "actor-subjective beliefs about likely reactions",
    "workload_pressure": "actor-observed normalized workload",
}, entity_types=("person", "institution"))


def _hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _value(value, default=None):
    if isinstance(value, StateField):
        return value.value if value.value is not None else value.dist
    return default if value is None else value


def _uncertainty(value) -> dict:
    if isinstance(value, StateField):
        return {
            "distribution": value.dist,
            "confidence": value.prov.confidence,
            "status": value.prov.status,
            "method": value.prov.method,
            "sources": list(value.prov.sources),
        }
    return {"status": "untyped", "confidence": 0.0}


@dataclass(frozen=True)
class ActionTarget:
    target_type: str = "none"
    target_id: str = ""


@dataclass
class TypedAction:
    """Versioned executable actor action.

    ``possible_consequences`` is semantic mechanism configuration, not a promise
    that the consequence will happen.  The execution engine verifies it and emits
    a separate StateDelta for every actual change.
    """

    action_id: str
    actor_id: str
    actor_role: str
    action_family: str
    action_name: str
    target: ActionTarget = field(default_factory=ActionTarget)
    parameters: dict = field(default_factory=dict)
    preconditions: list = field(default_factory=list)
    information_requirements: list = field(default_factory=list)
    institutional_permissions: list = field(default_factory=list)
    authority_requirements: list = field(default_factory=list)
    resource_requirements: dict = field(default_factory=dict)
    resource_costs: dict = field(default_factory=dict)
    available_from: float | None = None
    available_until: float | None = None
    expected_duration_s: float = 0.0
    observability: dict = field(default_factory=dict)
    reversible: bool = True
    commitments_created: list = field(default_factory=list)
    possible_consequences: list = field(default_factory=list)
    possible_delayed_consequences: list = field(default_factory=list)
    mechanisms_triggered: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)
    compiler_inclusion_reason: str = ""
    support_status: str = "broad_prior"
    semantic_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.action_family not in ACTION_FAMILIES:
            raise ValueError(f"unknown action family {self.action_family!r}")
        if not self.action_name or not self.action_id or not self.actor_id:
            raise ValueError("action_id, actor_id, and action_name are required")
        if self.action_name not in KNOWN_ACTIONS and not self.mechanisms_triggered:
            raise ValueError("new scenario actions require at least one executable mechanism")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TypedAction":
        d = dict(data)
        target = d.get("target") or {}
        d["target"] = target if isinstance(target, ActionTarget) else ActionTarget(**target)
        return cls(**d)


@dataclass
class FeasibilityDecision:
    action_id: str
    perceived_status: str
    actual_status: str
    perceived_reasons: list = field(default_factory=list)
    actual_reasons: list = field(default_factory=list)
    uncertainty: float = 0.0

    @property
    def perceived_feasible(self) -> bool:
        return self.perceived_status in ("feasible", "feasible_with_uncertainty")

    @property
    def actually_feasible(self) -> bool:
        return self.actual_status in ("feasible", "feasible_with_uncertainty")


@dataclass
class ActorView:
    schema_version: str
    actor_id: str
    actor_role: str
    observed_time: float
    observed_events: list = field(default_factory=list)
    remembered_events: list = field(default_factory=list)
    perceived_actions: list = field(default_factory=list)
    beliefs: dict = field(default_factory=dict)
    beliefs_about_actors: dict = field(default_factory=dict)
    relationships: list = field(default_factory=list)
    network_position: dict = field(default_factory=dict)
    institution_rules: list = field(default_factory=list)
    authority: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    preferences: dict = field(default_factory=dict)
    incentives: dict = field(default_factory=dict)
    commitments: list = field(default_factory=list)
    obligations: list = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    workload: float | None = None
    attention: float | None = None
    risk_beliefs: dict = field(default_factory=dict)
    action_history: list = field(default_factory=list)
    policy_state: dict = field(default_factory=dict)
    expected_reactions: dict = field(default_factory=dict)
    information_credibility: dict = field(default_factory=dict)
    observed_evidence_ids: list = field(default_factory=list)
    uncertainty: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    hidden_fields_excluded: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def view_hash(self) -> str:
        return _hash(self.as_dict())


class ActorViewBuilder:
    """Fail-closed projection from omniscient state into actor-local state."""

    def build(self, world, actor_id: str, *, observed_events: Iterable | None = None,
              institution_boundaries: dict | None = None) -> ActorView:
        actor = world.entity(actor_id)
        now = world.clock.now
        fields = actor.fields
        excluded = sorted(PRIVATE_ACTOR_FIELDS | SIMULATOR_ONLY_FIELDS)

        def own(name, default=None):
            return _value(fields.get(name), default)

        roles = own("roles", []) or []
        role = str(roles[0] if isinstance(roles, list) and roles else roles or actor.entity_type)
        beliefs = self._mapping(fields.get("beliefs"))
        resources = self._mapping(fields.get("resources"))
        prefs = self._mapping(fields.get("preferences"))
        uncertainty = {
            name: _uncertainty(value) for name, value in fields.items()
            if name not in PRIVATE_ACTOR_FIELDS
        }

        info_rows, evidence_ids, credibility = [], [], {}
        if world.information is not None:
            for item, exposure in world.information.visible_to(actor_id, at=now):
                published = getattr(item, "published_at", None)
                if published is not None and published > now:
                    continue
                iid = str(getattr(item, "item_id", ""))
                info_rows.append({
                    "event_id": iid, "at": getattr(exposure, "observed_at", now),
                    "kind": "information", "source": getattr(item, "source", ""),
                    "content": getattr(item, "content", ""),
                    "about": getattr(item, "about", ""),
                })
                evidence_ids.append(iid)
                credibility[iid] = float(getattr(item, "credibility", 0.5))

        for event in observed_events or []:
            row = event if isinstance(event, dict) else getattr(event, "__dict__", {})
            ts = float(row.get("ts", row.get("at", now)) or now)
            if ts <= now and self._event_visible(row, actor_id):
                info_rows.append({k: v for k, v in row.items() if k not in SIMULATOR_ONLY_FIELDS})

        relations = []
        if world.network is not None:
            for edge in world.network.edges:
                if actor_id not in (edge.src, edge.dst):
                    continue
                if edge.visibility == "private" and actor_id not in (edge.src, edge.dst):
                    continue
                if edge.visibility == "src_only" and actor_id != edge.src:
                    continue
                if edge.visibility == "dst_only" and actor_id != edge.dst:
                    continue
                relations.append({
                    "relation": edge.rel, "other_actor": edge.dst if edge.src == actor_id else edge.src,
                    "direction": "out" if edge.src == actor_id else "in",
                    "strength": _value(edge.strength, 0.5), "trust": edge.trust,
                    "channel": edge.channel, "uncertainty": _uncertainty(edge.strength),
                })

        rules, authority = [], list(own("authority", []) or [])
        for inst_id, inst in (world.institutions or {}).items():
            boundary = (institution_boundaries or {}).get(inst_id)
            for rule in getattr(inst, "rules", []):
                params = dict(getattr(rule, "params", {}) or {})
                info_class = params.pop("info_class", "public")
                if boundary is not None and not boundary.visible_to(role, info_class):
                    excluded.append(f"institutions[{inst_id}].rules[{rule.rule_id}]")
                    continue
                rules.append({"institution_id": inst_id, "rule_id": rule.rule_id,
                              "kind": rule.kind, "params": params})

        history = list(own("past_actions", []) or [])
        memory = list(own("memory", []) or [])
        latent = fields.get("latent_state") or {}
        policy_state = ({k: _value(v) for k, v in latent.items() if str(k).startswith("phase4_policy_")}
                        if isinstance(latent, dict) else {})
        return ActorView(
            schema_version=SCHEMA_VERSION, actor_id=actor_id, actor_role=role, observed_time=now,
            observed_events=sorted(info_rows, key=lambda x: float(x.get("at", now) or now)),
            remembered_events=memory, perceived_actions=[h for h in history if h.get("public", True)],
            beliefs=beliefs, beliefs_about_actors={k: v for k, v in beliefs.items() if str(k).startswith("actor:")},
            relationships=relations,
            network_position={"visible_degree": len(relations),
                              "reachable_actor_ids": sorted({r["other_actor"] for r in relations})},
            institution_rules=rules, authority=authority,
            goals=list(own("goals", []) or []), preferences=prefs,
            incentives=self._mapping(fields.get("incentives")),
            commitments=list(own("commitments", []) or []),
            obligations=list(own("obligations", []) or []), resources=resources,
            workload=self._number(own("workload_pressure")), attention=self._number(own("attention")),
            risk_beliefs={k: v for k, v in beliefs.items() if "risk" in str(k).lower()},
            action_history=history, policy_state=policy_state,
            expected_reactions=self._mapping(fields.get("expected_reactions")),
            information_credibility=credibility, observed_evidence_ids=sorted(evidence_ids),
            uncertainty=uncertainty,
            provenance={"world_id": world.world_id, "world_branch_id": world.branch_id,
                        "projection": "ActorViewBuilder:4.0.0"},
            hidden_fields_excluded=sorted(set(excluded)),
        )

    @staticmethod
    def _mapping(value) -> dict:
        if isinstance(value, dict):
            return {str(k): _value(v) for k, v in value.items()}
        raw = _value(value, {})
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _event_visible(event: dict, actor_id: str) -> bool:
        vis = event.get("visibility", "public")
        participants = event.get("participants", []) or []
        return vis == "public" or actor_id in participants or actor_id in (event.get("visible_to") or [])


class ActionSpaceBuilder:
    """Construct scenario actions from compiled semantic proposals and live capabilities.

    There is no question-keyword router.  Actions enter through compiler proposals,
    executable institution rules, actor authority/resources, or visible network paths.
    """

    def build(self, plan, world, view: ActorView, *, decision: dict | None = None) -> list[TypedAction]:
        decision = decision or {}
        proposals = list(decision.get("candidate_actions") or decision.get("actions") or [])
        proposals += list(decision.get("scenario_actions") or [])
        explicit_names = {str(p if isinstance(p, str) else (p.get("name") or p.get("type") or ""))
                          for p in proposals}
        out = []
        for i, proposal in enumerate(proposals):
            if isinstance(proposal, str):
                proposal = {"name": proposal}
            try:
                out.append(self._from_proposal(proposal, view, decision, i))
            except (TypeError, ValueError):
                continue

        mentioned = set()
        for rule in view.institution_rules:
            params = rule.get("params", {})
            # A constraint mentioning an action does not itself grant that action.  Only
            # explicit decision rights held by this actor (or actor authority already in
            # the view) are capability evidence.
            names = params.get("actions", []) or []
            if rule.get("kind") == "decision_right":
                if view.actor_id not in (params.get("holders") or []):
                    continue
                mentioned.update(str(name) for name in names)
            else:
                mentioned.update(str(name) for name in names if str(name) in set(view.authority))
        for i, name in enumerate(sorted(mentioned)):
            if name in explicit_names:
                continue
            proposal = {"name": name, "family": KNOWN_ACTIONS.get(name, "institutional"),
                        "institutional_permissions": [name],
                        "mechanisms_triggered": ["institution_processing"],
                        "inclusion_reason": "executable institution rule permits or constrains action"}
            try:
                out.append(self._from_proposal(proposal, view, decision, len(out) + i))
            except ValueError:
                pass

        if not out:
            # No-forecast-abstention: broad structural hypotheses remain executable.
            for name in ("wait", "abstain"):
                out.append(self._from_proposal(
                    {"name": name, "family": "generic", "mechanisms_triggered": ["record_action"],
                     "support_status": "tier_7_broad_prior",
                     "inclusion_reason": "cold-start no-abstention fallback"}, view, decision, len(out)))

        unique = {}
        for action in out:
            key = (action.action_name, action.target.target_type, action.target.target_id,
                   json.dumps(action.parameters, sort_keys=True, default=str))
            unique.setdefault(key, action)
        return list(unique.values())

    def _from_proposal(self, proposal: dict, view: ActorView, decision: dict, index: int) -> TypedAction:
        name = str(proposal.get("name") or proposal.get("type") or "").strip()
        family = str(proposal.get("family") or KNOWN_ACTIONS.get(name, "generic"))
        mechanisms = list(proposal.get("mechanisms_triggered") or self._default_mechanisms(name))
        target = proposal.get("target") or {}
        if isinstance(target, str):
            target = {"target_type": "actor", "target_id": target}
        aid = proposal.get("action_id") or _hash({"actor": view.actor_id, "name": name, "target": target,
                                                  "parameters": proposal.get("parameters", {}),
                                                  "index": index})[:20]
        return TypedAction(
            action_id=aid, actor_id=view.actor_id, actor_role=view.actor_role,
            action_family=family, action_name=name,
            target=ActionTarget(target_type=str(target.get("target_type", "none")),
                                target_id=str(target.get("target_id", ""))),
            parameters=dict(proposal.get("parameters") or {}),
            preconditions=list(proposal.get("preconditions") or []),
            information_requirements=list(proposal.get("information_requirements") or []),
            institutional_permissions=list(proposal.get("institutional_permissions") or []),
            authority_requirements=list(proposal.get("authority_requirements") or []),
            resource_requirements=dict(proposal.get("resource_requirements") or {}),
            resource_costs=dict(proposal.get("resource_costs") or {}),
            available_from=proposal.get("available_from"), available_until=proposal.get("available_until"),
            expected_duration_s=float(proposal.get("expected_duration_s", 0.0) or 0.0),
            observability=dict(proposal.get("observability") or {"default": "participants"}),
            reversible=bool(proposal.get("reversible", True)),
            commitments_created=list(proposal.get("commitments_created") or []),
            possible_consequences=list(proposal.get("possible_consequences") or []),
            possible_delayed_consequences=list(proposal.get("possible_delayed_consequences") or []),
            mechanisms_triggered=mechanisms,
            provenance={"source": proposal.get("source", "compiler_semantic_proposal"),
                        "plan_hash": plan_hash(decision.get("plan"))},
            uncertainty=dict(proposal.get("uncertainty") or {"semantic": 0.5}),
            compiler_inclusion_reason=str(proposal.get("inclusion_reason") or "compiler candidate"),
            support_status=str(proposal.get("support_status", "broad_prior")),
        )

    @staticmethod
    def _default_mechanisms(name: str) -> list:
        if name in ("wait", "abstain", "ignore", "withhold_information", "conceal", "hold_position"):
            return ["record_action"]
        if name in ACTION_ONTOLOGY["messaging"] or name in ("comment", "share", "create_content"):
            return ["message_delivery", "reaction_scheduling"]
        if name in ACTION_ONTOLOGY["institutional"]:
            return ["institution_processing", "reaction_scheduling"]
        return ["record_action", "reaction_scheduling"]


def plan_hash(plan) -> str:
    if plan is None:
        return ""
    try:
        return plan.plan_hash()
    except AttributeError:
        return _hash(plan)[:12]


class FeasibilityEngine:
    """Classify perceived feasibility and omniscient actual feasibility separately."""

    KNOWN_IMPOSSIBLE = {
        "temporarily_unavailable", "institutionally_prohibited", "outside_authority",
        "physically_impossible", "unaffordable", "insufficient_information", "precondition_unmet",
        "binding_commitment_conflict", "unknown_to_actor", "unsupported_action_semantics",
    }

    def classify(self, action: TypedAction, view: ActorView, world) -> FeasibilityDecision:
        perceived, preasons = self._perceived(action, view)
        actual, areasons = self._actual(action, world)
        uncertainty = max(float(action.uncertainty.get("feasibility", 0.0) or 0.0),
                          0.35 if perceived == "feasible_with_uncertainty" else 0.0)
        return FeasibilityDecision(action.action_id, perceived, actual, preasons, areasons, uncertainty)

    def _perceived(self, action: TypedAction, view: ActorView):
        if action.available_from is not None and view.observed_time < float(action.available_from):
            return "temporarily_unavailable", ["not_yet_available"]
        if action.available_until is not None and view.observed_time > float(action.available_until):
            return "temporarily_unavailable", ["deadline_passed"]
        missing = [i for i in action.information_requirements
                   if i not in view.observed_evidence_ids and i not in view.beliefs]
        if missing:
            return "insufficient_information", [f"missing:{x}" for x in missing]
        if action.authority_requirements and not set(action.authority_requirements) <= set(view.authority):
            return "outside_authority", ["actor_perceives_missing_authority"]
        for resource, amount in action.resource_requirements.items():
            if float(view.resources.get(resource, 0.0) or 0.0) < float(amount):
                return "unaffordable", [f"perceived_{resource}_shortfall"]
        for commitment in view.commitments:
            if not isinstance(commitment, dict):
                continue
            if commitment.get("binding") and action.action_name in (commitment.get("prohibits") or []):
                return "binding_commitment_conflict", [str(commitment.get("id", "binding_commitment"))]
        if action.support_status in ("unsupported", "quarantined"):
            return "unsupported_action_semantics", [f"support={action.support_status}"]
        if action.uncertainty.get("feasibility", 0.0):
            return "feasible_with_uncertainty", ["actor_uncertain"]
        return "feasible", []

    def _actual(self, action: TypedAction, world):
        if action.target.target_id and action.target.target_type in ("actor", "institution"):
            known = action.target.target_id in world.entities or action.target.target_id in world.institutions
            if not known:
                return "physically_impossible", ["unresolved_target"]
        raw = self.to_world_action(action)
        reasons = []
        for inst in (world.institutions or {}).values():
            ok, why = inst.validate_action(world, raw)
            if not ok:
                reasons.extend(why)
        if reasons:
            return "institutionally_prohibited", reasons
        actor = world.entities.get(action.actor_id)
        if actor is None:
            return "physically_impossible", ["actor_missing"]
        resources = actor.get("resources") or {}
        for resource, amount in action.resource_requirements.items():
            sf = resources.get(resource) if isinstance(resources, dict) else None
            have = float(_value(sf, 0.0) or 0.0)
            if have < float(amount):
                return "unaffordable", [f"actual_{resource}_shortfall"]
        return "feasible", []

    @staticmethod
    def to_world_action(action: TypedAction) -> dict:
        return {"actor": action.actor_id, "type": action.action_name,
                "target": action.target.target_id, **action.parameters}

    @classmethod
    def perceived_mask(cls, actions: list[TypedAction], decisions: list[FeasibilityDecision]) -> list[TypedAction]:
        by_id = {d.action_id: d for d in decisions}
        return [a for a in actions if by_id[a.action_id].perceived_feasible]


@dataclass
class ConsequenceDistribution:
    action_id: str
    outcomes: list = field(default_factory=list)
    expected_resource_delta: dict = field(default_factory=dict)
    expected_relationship_delta: float = 0.0
    expected_belief_delta: float = 0.0
    success_probability: float = 0.5
    reaction_distribution: dict = field(default_factory=dict)
    uncertainty: float = 0.5
    provenance: dict = field(default_factory=dict)


class SubjectiveConsequenceModel:
    """Build consequence beliefs exclusively from ActorView and parameter packs."""

    def predict(self, action: TypedAction, view: ActorView, parameters: dict) -> ConsequenceDistribution:
        reaction_prior = parameters.get("reaction_priors", {}).get(action.action_name, {})
        if not reaction_prior:
            n_targets = 1 if action.target.target_id else 0
            reaction_prior = {"observe": 0.5, "respond": 0.25 if n_targets else 0.1,
                              "ignore": 0.25 if n_targets else 0.4}
        z = sum(max(0.0, float(v)) for v in reaction_prior.values()) or 1.0
        reactions = {k: max(0.0, float(v)) / z for k, v in reaction_prior.items()}
        costs = {k: -abs(float(v)) for k, v in action.resource_costs.items()}
        rel = float(parameters.get("relationship_effects", {}).get(action.action_name, 0.0) or 0.0)
        belief = float(parameters.get("belief_effects", {}).get(action.action_name, 0.0) or 0.0)
        success = float(parameters.get("success_priors", {}).get(action.action_name, 0.5) or 0.5)
        uncertainty = max(0.05, float(parameters.get("consequence_uncertainty", 0.5) or 0.5))
        return ConsequenceDistribution(
            action_id=action.action_id,
            outcomes=list(action.possible_consequences) + list(action.possible_delayed_consequences),
            expected_resource_delta=costs, expected_relationship_delta=rel,
            expected_belief_delta=belief, success_probability=min(0.99, max(0.01, success)),
            reaction_distribution=reactions, uncertainty=uncertainty,
            provenance={"parameter_pack": parameters.get("pack_id", "tier_7_reference"),
                        "actor_view_hash": view.view_hash()},
        )


@dataclass
class UtilityComponent:
    name: str
    mean: float
    sd: float
    source: str
    tier: int


@dataclass
class UtilityPosterior:
    action_id: str
    components: list
    expected_utility: float
    utility_sd: float


class UtilityInference:
    """Hierarchical utility inference with explicit shrinkage and provenance."""

    COMPONENTS = ("success", "resources", "relationship", "belief", "effort", "delay", "risk",
                  "norm", "obligation", "reputation", "commitment")

    def infer(self, action: TypedAction, view: ActorView, consequence: ConsequenceDistribution,
              parameters: dict) -> UtilityPosterior:
        actor_params = parameters.get("actors", {}).get(view.actor_id, {})
        role_params = parameters.get("roles", {}).get(view.actor_role, {})
        group_params = parameters.get("global", {})
        n_actor = int(actor_params.get("n", 0) or 0)
        shrink = n_actor / (n_actor + float(parameters.get("partial_pool_strength", 10.0) or 10.0))
        comps = []
        signals = {
            "success": consequence.success_probability,
            "resources": sum(consequence.expected_resource_delta.values()),
            "relationship": consequence.expected_relationship_delta,
            "belief": consequence.expected_belief_delta,
            "effort": -float(action.parameters.get("effort", action.expected_duration_s / 3600.0) or 0.0),
            "delay": -float(action.expected_duration_s / 86400.0),
            "risk": -consequence.uncertainty,
            "norm": float(action.parameters.get("norm_alignment", 0.0) or 0.0),
            "obligation": float(action.parameters.get("obligation_alignment", 0.0) or 0.0),
            "reputation": float(action.parameters.get("reputation_effect", 0.0) or 0.0),
            "commitment": float(len(action.commitments_created)) * 0.1,
        }
        for name in self.COMPONENTS:
            g = group_params.get(name, {"mean": 0.0, "sd": 1.0})
            r = role_params.get(name, g)
            a = actor_params.get(name, r)
            gm = float(g.get("mean", 0.0) if isinstance(g, dict) else g)
            rm = float(r.get("mean", gm) if isinstance(r, dict) else r)
            am = float(a.get("mean", rm) if isinstance(a, dict) else a)
            coef = shrink * am + (1.0 - shrink) * rm
            sd = float((a if isinstance(a, dict) else {}).get("sd", (r if isinstance(r, dict) else {}).get("sd", 1.0)))
            value = coef * signals[name]
            tier = 1 if n_actor >= 5 else (2 if view.actor_role in parameters.get("roles", {}) else 7)
            source = "actor_history" if tier == 1 else ("role_parameter_pack" if tier == 2 else "broad_prior")
            comps.append(UtilityComponent(name, value, abs(signals[name]) * sd, source, tier))
        eu = sum(c.mean for c in comps)
        usd = math.sqrt(sum(c.sd ** 2 for c in comps) + consequence.uncertainty ** 2)
        return UtilityPosterior(action.action_id, comps, eu, usd)


@dataclass
class PolicyFamilySpec:
    family_id: str
    assumptions: str
    required_state: tuple
    compatible_families: tuple = ACTION_FAMILIES
    parameter_schema: dict = field(default_factory=dict)
    fit_method: str = "hierarchical likelihood"
    applicability: str = "state requirements satisfied"
    exclusions: str = ""
    transport_limits: str = "parameters must widen outside fitted domain"
    validation_status: str = "implemented"
    failure_modes: list = field(default_factory=list)


def default_policy_registry() -> dict[str, PolicyFamilySpec]:
    specs = {}
    for family in POLICY_FAMILIES:
        required = ("action_history",) if family in ("habit", "reinforcement_learning", "ewa") else ()
        if family in ("strategic_anticipation", "limited_depth_reasoning", "belief_planning"):
            required = ("beliefs", "expected_reactions")
        specs[family] = PolicyFamilySpec(
            family, assumptions=family.replace("_", " ") + " is a plausible decision process",
            required_state=required,
            parameter_schema={"precision": "distribution", "weight": "posterior_probability"},
            exclusions="exclude when required actor-visible state is absent",
            failure_modes=["transport shift", "unidentified family", "sparse actor history"],
        )
    return specs


@dataclass
class PolicyFamilyPosterior:
    weights: dict
    provenance: dict = field(default_factory=dict)
    structural_particles: list = field(default_factory=list)


@dataclass
class ActionPosterior:
    schema_version: str
    actor_id: str
    feasible_actions: list
    action_probabilities: dict
    unnormalized_scores: dict
    expected_utilities: dict
    expected_consequences: dict
    policy_family_posterior: PolicyFamilyPosterior
    parameter_uncertainty: dict
    credible_intervals: dict
    entropy: float
    feasibility_diagnostics: list
    support_grade: str
    fallbacks_used: list
    sensitivity_contributors: list
    provenance: dict
    model_version: str
    parameter_pack_versions: list

    def as_dict(self) -> dict:
        d = asdict(self)
        d["policy_family_posterior"] = asdict(self.policy_family_posterior)
        return d

    def sample(self, rng: random.Random) -> str:
        r, acc = rng.random(), 0.0
        for action_id, probability in self.action_probabilities.items():
            acc += probability
            if r <= acc:
                return action_id
        return next(reversed(self.action_probabilities))


class TemperatureCalibrator:
    def __init__(self, temperature: float = 1.0, *, fitted_on: str = "unfitted"):
        self.temperature = max(0.05, float(temperature))
        self.fitted_on = fitted_on

    def apply(self, probabilities: dict) -> dict:
        if not probabilities:
            return {}
        logits = {k: math.log(max(1e-12, v)) / self.temperature for k, v in probabilities.items()}
        m = max(logits.values())
        weights = {k: math.exp(v - m) for k, v in logits.items()}
        z = sum(weights.values()) or 1.0
        return {k: v / z for k, v in weights.items()}


class ActorPolicyModel:
    """Mixture-of-regimes policy over actor-visible state and posterior particles."""

    def __init__(self, parameter_pack: dict | None = None, *, calibrator=None,
                 family_registry: dict | None = None, model_version: str = SCHEMA_VERSION):
        self.parameter_pack = parameter_pack or self._broad_pack()
        self.calibrator = calibrator or TemperatureCalibrator(1.0)
        self.registry = family_registry or default_policy_registry()
        self.model_version = model_version
        self.utility = UtilityInference()
        self.consequences = SubjectiveConsequenceModel()

    def decide(self, views: list[ActorView], actions: list[TypedAction],
               feasibility: list[list[FeasibilityDecision]], *, seed: int = 0) -> ActionPosterior:
        if not views:
            raise ValueError("at least one posterior ActorView particle is required")
        if len(feasibility) != len(views):
            raise ValueError("one feasibility vector is required per ActorView particle")
        family_post = self._family_posterior(views[0], actions)
        per_particle, utility_rows, consequence_rows = [], {}, {}
        feasible_union = []
        for view, decisions in zip(views, feasibility):
            by_id = {d.action_id: d for d in decisions}
            feasible = [a for a in actions if by_id[a.action_id].perceived_feasible]
            if not feasible:
                continue
            feasible_union.extend(a.action_id for a in feasible)
            scores = {a.action_id: 0.0 for a in feasible}
            for action in feasible:
                consequence = self.consequences.predict(action, view, self.parameter_pack)
                utility = self.utility.infer(action, view, consequence, self.parameter_pack)
                utility_rows.setdefault(action.action_id, []).append(utility)
                consequence_rows.setdefault(action.action_id, []).append(consequence)
                for family, weight in family_post.weights.items():
                    scores[action.action_id] += weight * self._family_score(
                        family, action, view, utility, consequence)
            per_particle.append(self._softmax(scores, float(self.parameter_pack.get("precision", 1.0))))

        if not per_particle:
            raise ValueError("actor perceives no feasible action; action-space fallback contract was violated")
        probs = {aid: sum(p.get(aid, 0.0) for p in per_particle) / len(per_particle)
                 for aid in sorted(set(feasible_union))}
        probs = self.calibrator.apply(probs)
        # Known-impossible in every actor particle receives exactly zero and is omitted from sampling.
        intervals = {}
        for aid in probs:
            vals = sorted(p.get(aid, 0.0) for p in per_particle)
            intervals[aid] = [vals[max(0, int(0.05 * len(vals)) - 1)],
                              vals[min(len(vals) - 1, int(0.95 * len(vals)))]]
        eu = {aid: sum(u.expected_utility for u in rows) / len(rows) for aid, rows in utility_rows.items()}
        scores = {aid: math.log(max(1e-12, p)) for aid, p in probs.items()}
        entropy = -sum(p * math.log(max(1e-12, p)) for p in probs.values())
        fallbacks = list(self.parameter_pack.get("fallbacks", []))
        return ActionPosterior(
            schema_version=SCHEMA_VERSION, actor_id=views[0].actor_id,
            feasible_actions=sorted(probs), action_probabilities=probs,
            unnormalized_scores=scores, expected_utilities=eu,
            expected_consequences={aid: asdict(rows[0]) for aid, rows in consequence_rows.items()},
            policy_family_posterior=family_post,
            parameter_uncertainty=dict(self.parameter_pack.get("uncertainty", {})),
            credible_intervals=intervals, entropy=entropy,
            feasibility_diagnostics=[asdict(d) for row in feasibility for d in row],
            support_grade=str(self.parameter_pack.get("support_grade", "highly_speculative")),
            fallbacks_used=fallbacks,
            sensitivity_contributors=self._sensitivity(views, utility_rows),
            provenance={"actor_view_hashes": [v.view_hash() for v in views],
                        "numeric_source": self.parameter_pack.get("source", "reference_class_prior"),
                        "llm_probability_minting": False, "seed": seed},
            model_version=self.model_version,
            parameter_pack_versions=[str(self.parameter_pack.get("pack_id", "tier7:4.0.0"))],
        )

    def _family_posterior(self, view: ActorView, actions: list[TypedAction]) -> PolicyFamilyPosterior:
        prior = dict(self.parameter_pack.get("policy_family_weights") or {})
        if not prior:
            prior = {"random_utility": 0.25, "quantal_response": 0.2, "satisficing": 0.15,
                     "habit": 0.1, "obligation": 0.1, "risk_sensitive": 0.1,
                     "limited_depth_reasoning": 0.1}
        valid = {}
        for family, weight in prior.items():
            spec = self.registry.get(family)
            if spec is None:
                continue
            if "action_history" in spec.required_state and not view.action_history:
                weight *= 0.25
            if "beliefs" in spec.required_state and not view.beliefs:
                weight *= 0.25
            valid[family] = max(0.0, float(weight))
        z = sum(valid.values()) or 1.0
        weights = {k: v / z for k, v in valid.items()}
        particles = [{"family": k, "weight": v, "preserved_structure": True} for k, v in weights.items()]
        return PolicyFamilyPosterior(weights, {"source": self.parameter_pack.get("source"),
                                               "pack_id": self.parameter_pack.get("pack_id")}, particles)

    def _family_score(self, family: str, action: TypedAction, view: ActorView,
                      utility: UtilityPosterior, consequence: ConsequenceDistribution) -> float:
        u = utility.expected_utility
        if family in ("random_utility", "multinomial_logit", "nested_discrete_choice", "quantal_response"):
            return u
        if family in ("satisficing", "bounded_search"):
            threshold = float(self.parameter_pack.get("satisficing_threshold", 0.0) or 0.0)
            return 1.0 if u >= threshold else u - threshold
        if family == "habit":
            count = sum(1 for h in view.action_history if h.get("action") == action.action_name)
            return u + math.log1p(count) * float(self.parameter_pack.get("habit_strength", 0.2) or 0.2)
        if family in ("reinforcement_learning", "ewa"):
            learned = view.policy_state.get(f"phase4_policy_value:{action.action_name}")
            q = float(learned if learned is not None else
                      self.parameter_pack.get("action_values", {}).get(action.action_name, 0.0) or 0.0)
            return 0.5 * u + 0.5 * q
        if family in ("norm_compliance", "obligation", "institutional_obedience"):
            aligned = action.action_name in view.authority or bool(action.institutional_permissions)
            return u + (0.5 if aligned else -0.25)
        if family in ("reciprocity", "imitation", "social_proof"):
            social = sum(float(r.get("strength", 0.5) or 0.5) for r in view.relationships)
            return u + 0.05 * social
        if family in ("risk_sensitive", "loss_aversion"):
            return u - utility.utility_sd * float(self.parameter_pack.get("risk_aversion", 0.5) or 0.5)
        if family in ("strategic_anticipation", "limited_depth_reasoning", "belief_planning"):
            return u + math.log(max(1e-6, consequence.success_probability))
        if family == "delay_hazard":
            return u - action.expected_duration_s / 86400.0
        return u

    @staticmethod
    def _softmax(scores: dict, precision: float) -> dict:
        m = max(scores.values())
        weights = {k: math.exp(max(-40.0, min(40.0, precision * (v - m)))) for k, v in scores.items()}
        z = sum(weights.values()) or 1.0
        return {k: v / z for k, v in weights.items()}

    @staticmethod
    def _sensitivity(views, rows) -> list:
        out = []
        for aid, utilities in rows.items():
            sd = sum(u.utility_sd for u in utilities) / len(utilities)
            out.append({"component": f"utility:{aid}", "score": round(sd, 6)})
        out.append({"component": "world_particle_disagreement", "score": len(views)})
        return sorted(out, key=lambda x: -float(x["score"]))[:8]

    @staticmethod
    def _broad_pack():
        return {
            "pack_id": "phase4:tier7-reference:4.0.0", "source": "reference_class_prior",
            "support_grade": "highly_speculative", "precision": 0.7, "partial_pool_strength": 10.0,
            "global": {name: {"mean": 0.0 if name != "success" else 1.0, "sd": 1.0}
                       for name in UtilityInference.COMPONENTS},
            "policy_family_weights": {"random_utility": 0.3, "quantal_response": 0.2,
                                      "satisficing": 0.2, "risk_sensitive": 0.15,
                                      "limited_depth_reasoning": 0.15},
            "consequence_uncertainty": 0.8,
            "uncertainty": {"preference": "broad", "policy_family": "mixture",
                            "transport": "maximally widened"},
            "fallbacks": [{"tier": 7, "reason": "no validated local policy pack",
                           "uncertainty_widening": 2.0}],
        }


@dataclass
class DecisionTrace:
    trace_id: str
    question_id: str
    plan_hash: str
    world_particle_ids: list
    actor_id: str
    decision_time: float
    actor_view_hashes: list
    observed_evidence_ids: list
    hidden_fields_excluded: list
    candidate_actions: list
    feasibility_decisions: list
    policy_family_posterior: dict
    parameter_pack_versions: list
    utility_components: dict
    subjective_consequences: dict
    calibrated_action_distribution: dict
    sampled_action_id: str
    random_seed: int
    resulting_event_ids: list = field(default_factory=list)
    resulting_state_delta_ids: list = field(default_factory=list)
    downstream_reactions: list = field(default_factory=list)
    latency_ms: float = 0.0
    cost: dict = field(default_factory=lambda: {"llm_calls": 0, "usd": 0.0})
    warnings: list = field(default_factory=list)
    support_grade: str = "highly_speculative"
    fallback_tier: int = 7
    checksum: str = ""

    def seal(self) -> "DecisionTrace":
        self.checksum = _hash({k: v for k, v in asdict(self).items() if k != "checksum"})
        return self

    def verify(self) -> bool:
        expected = _hash({k: v for k, v in asdict(self).items() if k != "checksum"})
        return bool(self.checksum) and self.checksum == expected

    def as_dict(self) -> dict:
        return asdict(self)


def build_trace(*, question_id: str, plan, worlds: list, views: list[ActorView], actions: list[TypedAction],
                feasibility: list[list[FeasibilityDecision]], posterior: ActionPosterior,
                selected_action_id: str, seed: int, started_at: float | None = None) -> DecisionTrace:
    started_at = started_at or time.monotonic()
    utility_components = {}
    for aid, value in posterior.expected_utilities.items():
        utility_components[aid] = {"expected_utility": value}
    tier = min([int(x.get("tier", 7)) for x in posterior.fallbacks_used] or [1])
    return DecisionTrace(
        trace_id=_hash({"plan": plan_hash(plan), "actor": views[0].actor_id,
                        "time": views[0].observed_time, "seed": seed})[:24],
        question_id=question_id, plan_hash=plan_hash(plan),
        world_particle_ids=[f"{w.world_id}:{w.branch_id}" for w in worlds], actor_id=views[0].actor_id,
        decision_time=views[0].observed_time, actor_view_hashes=[v.view_hash() for v in views],
        observed_evidence_ids=sorted({x for v in views for x in v.observed_evidence_ids}),
        hidden_fields_excluded=sorted({x for v in views for x in v.hidden_fields_excluded}),
        candidate_actions=[a.as_dict() for a in actions],
        feasibility_decisions=[asdict(d) for row in feasibility for d in row],
        policy_family_posterior=asdict(posterior.policy_family_posterior),
        parameter_pack_versions=posterior.parameter_pack_versions,
        utility_components=utility_components,
        subjective_consequences=posterior.expected_consequences,
        calibrated_action_distribution=posterior.action_probabilities,
        sampled_action_id=selected_action_id, random_seed=seed,
        latency_ms=(time.monotonic() - started_at) * 1000.0, support_grade=posterior.support_grade,
        fallback_tier=tier,
    ).seal()
