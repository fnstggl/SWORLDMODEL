"""Phase 13 universal action ontology (Part 4) — typed actor-authorized transformations, not a catalog.

An action is `(acting_actor, authority_basis, operation, controlled_object, params, timing, ...)` where
`operation` comes from an EXTENSIBLE registry seeded with the nine cross-domain families (resources,
time, information, relationships, negotiation, institutional, operations, policy-control, meta). Domain
adapters register new operations through the same door (`register_operation`) — there is no global
switch statement to edit, and the registry is data the feasibility engine and intervention layer read.

Every action knows how it enters the world: `event_repr()` yields canonical Events for the shared queue
(intervention semantics live in interventions.py). Nothing here mutates terminal probabilities.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# ---------------------------------------------------------------- operation registry
OPERATION_FAMILIES = ("resources", "time", "information", "relationships", "negotiation",
                      "institutional", "operations", "policy_control", "meta")

_OPERATIONS: dict = {}


def register_operation(name: str, *, family: str, description: str = "",
                       required_authority: str = "", requires_resources: bool = False,
                       reversible: bool = True, event_type: str = "decision_action",
                       param_schema: dict = None, feasibility_checks: tuple = ()):
    """Register an operation the ontology can express. `required_authority` is the capability string an
    actor must hold (empty = any actor may attempt; feasibility still checks object control).
    `feasibility_checks` name extra typed checks the engine runs (e.g. 'target_exists')."""
    if family not in OPERATION_FAMILIES:
        raise ValueError(f"unknown family {family!r} (valid: {OPERATION_FAMILIES})")
    _OPERATIONS[name] = {"family": family, "description": description,
                         "required_authority": required_authority,
                         "requires_resources": requires_resources, "reversible": reversible,
                         "event_type": event_type, "param_schema": dict(param_schema or {}),
                         "feasibility_checks": tuple(feasibility_checks)}
    return name


def operation_registered(name: str) -> bool:
    return name in _OPERATIONS


def operation_spec(name: str) -> dict:
    if name not in _OPERATIONS:
        raise KeyError(f"unregistered operation {name!r} — register_operation() first")
    return _OPERATIONS[name]


def operations_in_family(family: str) -> list:
    return sorted(n for n, s in _OPERATIONS.items() if s["family"] == family)


# seed the nine families (Part 4's grammar). Domain adapters extend; they never replace.
for _fam, _ops in (
    ("resources", ("allocate", "transfer", "reserve", "release", "invest", "consume",
                   "withhold", "acquire", "exchange")),
    ("time", ("begin", "delay", "accelerate", "sequence", "schedule", "cancel", "pause",
              "resume", "wait", "stop")),
    ("information", ("observe", "investigate", "query", "request", "disclose", "publish",
                     "signal", "communicate", "clarify", "verify", "commit_disclosure")),
    ("relationships", ("contact", "connect", "endorse", "coordinate", "ally", "form_coalition",
                       "dissolve_coalition", "delegate", "mediate", "sanction", "block",
                       "change_access", "change_structure")),
    ("negotiation", ("propose", "counteroffer", "accept", "reject", "condition", "bundle",
                     "unbundle", "concede", "guarantee", "insure", "escrow", "renegotiate",
                     "withdraw")),
    ("institutional", ("submit", "nominate", "call_vote", "amend", "veto", "approve", "appeal",
                       "authorize", "prohibit", "enforce", "set_procedure", "change_agenda",
                       "establish_rule", "invoke_authority")),
    ("operations", ("create", "modify", "deploy", "procure", "hire", "assign", "route", "price",
                    "prioritize", "scale", "shut_down", "maintain")),
    ("policy_control", ("set_parameter", "choose_policy", "set_threshold", "set_contingent_rule",
                        "establish_trigger", "establish_stop_rule", "set_monitoring_policy",
                        "choose_escalation_policy")),
    ("meta", ("gather_information", "defer", "commit", "preserve_option", "delegate_decision",
              "request_approval", "run_experiment", "choose_observation", "choose_decision_point",
              "do_nothing")),
):
    for _op in _ops:
        register_operation(_op, family=_fam,
                           requires_resources=(_fam == "resources"),
                           reversible=(_op not in ("consume", "shut_down", "commit", "transfer",
                                                   "dissolve_coalition", "withdraw")))


# ---------------------------------------------------------------- the typed action
@dataclass
class ActionSchema:
    """A typed, actor-authorized transformation over the causal world (Part 4's full field list).
    `content` carries free-form realization (e.g. message text / strategy vector) — semantics still flow
    through operation + params; content alone never bypasses feasibility."""
    action_id: str
    actor: str                                       # acting actor (must be the decision-maker or delegate)
    operation: str                                   # registered operation name
    object: str = ""                                 # controlled object: entity id, quantity, event id...
    params: dict = field(default_factory=dict)
    timing_ts: float = 0.0                           # unix ts the action fires (0 = at decision time)
    duration_days: float = 0.0
    authority_basis: str = ""                        # which held capability authorizes it
    preconditions: list = field(default_factory=list)   # [callable(world)->bool] state-dependent gates
    required_resources: dict = field(default_factory=dict)  # name -> amount consumed
    institutional_permission: str = ""               # institution id whose procedure must allow it
    observability: str = "public"                    # public | private | targeted
    recipients: list = field(default_factory=list)
    reversible: bool = None                          # None = inherit from operation spec
    direct_cost: float = 0.0                         # in utility units (implementation cost)
    indirect_cost: float = 0.0
    failure_prob: float = 0.0                        # implementation-failure probability (CRN-sampled)
    failure_modes: list = field(default_factory=list)
    content: dict = field(default_factory=dict)      # realization payload (message text, offer terms...)
    provenance: str = "user"                         # user | affordance | llm_proposer
    meta: dict = field(default_factory=dict)

    def spec(self) -> dict:
        return operation_spec(self.operation)

    def is_reversible(self) -> bool:
        return self.spec()["reversible"] if self.reversible is None else bool(self.reversible)

    def event_type(self) -> str:
        return self.spec()["event_type"]

    def semantic_key(self) -> str:
        """Dedup/normalization key: same (operation, object, bucketed params, recipients) collapses
        wording-only variants; `content['variant']` keeps deliberate content alternatives distinct."""
        def _bucket(v):
            if isinstance(v, (int, float)):
                return round(float(v), 3)
            return str(v)[:60]
        payload = {"op": self.operation, "obj": self.object,
                   "params": {k: _bucket(v) for k, v in sorted(self.params.items())},
                   "rcpt": sorted(self.recipients), "variant": self.content.get("variant", "")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def as_dict(self) -> dict:
        return {"action_id": self.action_id, "actor": self.actor, "operation": self.operation,
                "family": self.spec()["family"], "object": self.object,
                "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)[:80])
                           for k, v in self.params.items()},
                "timing_ts": self.timing_ts, "duration_days": self.duration_days,
                "authority_basis": self.authority_basis,
                "required_resources": self.required_resources,
                "institutional_permission": self.institutional_permission,
                "observability": self.observability, "recipients": self.recipients,
                "reversible": self.is_reversible(), "direct_cost": self.direct_cost,
                "indirect_cost": self.indirect_cost, "failure_prob": self.failure_prob,
                "content": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)[:120])
                            for k, v in self.content.items()},
                "provenance": self.provenance, "semantic_key": self.semantic_key()}


def do_nothing(actor: str, action_id: str = "do_nothing") -> ActionSchema:
    """The explicit status-quo reference (Part 9): enters evaluation like any action, changes nothing."""
    return ActionSchema(action_id=action_id, actor=actor, operation="do_nothing",
                        provenance="baseline", direct_cost=0.0)


def defer(actor: str, until_ts: float, action_id: str = "defer") -> ActionSchema:
    return ActionSchema(action_id=action_id, actor=actor, operation="defer",
                        params={"until_ts": until_ts}, provenance="baseline")


def dedupe(actions: list) -> tuple:
    """Semantic dedup preserving first occurrence and DIVERSE families. Returns (kept, dropped_keys)."""
    seen, kept, dropped = set(), [], []
    for a in actions:
        k = a.semantic_key()
        if k in seen:
            dropped.append({"action_id": a.action_id, "duplicate_of_key": k})
            continue
        seen.add(k)
        kept.append(a)
    return kept, dropped
