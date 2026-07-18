"""First-class SEMANTIC EVENTS — the causal object an executed action creates in the world.

A scalar coupling (`accept → cooperative_agreement +0.15`) erases exactly what social causality
runs on: who said what, to whom, through which channel, with what commitments, ambiguity and
provenance. The `SemanticEvent` contract preserves that causal content so downstream actors can
OBSERVE and INTERPRET the actual event rather than receive an anonymous scalar nudge.

Contract boundaries:
  * A semantic event is DATA about the world, produced by typed execution — it never mutates
    state itself. Delivery (who observes it) belongs to `observation_delivery.ObservationRouter`;
    consequences belong to the recipients' own decisions plus explicit structural mechanisms.
  * Compilation is deterministic: `compile_semantic_events(action, world, ...)` maps ANY executed
    TypedAction — menu, known-ontology, or LLM-novel — into one or more semantic events. A novel
    multi-target communication ("privately ask two wavering members whether they would defect
    together") compiles into one private communication event PER TARGET with no ontology
    coefficient anywhere.
  * Events ride the canonical queue as payloads of registered event types; they are never a
    second event system.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

SEMANTIC_EVENT_SCHEMA = "semantic.event.v1"

#: semantic event types with their default observability class. A registry, not a closed world:
#: unknown action shapes fall back to the generic communication/action types, never to silence.
SEMANTIC_EVENT_TYPES = {
    "public_statement": "public",
    "public_endorsement": "public",
    "public_opposition": "public",
    "private_communication": "participants",
    "institutional_submission": "institutional",
    "formal_vote_cast": "institutional",
    "proposal": "participants",
    "threat": "participants",
    "concession": "participants",
    "order_issued": "participants",
    "request": "participants",
    "refusal": "participants",
    "information_release": "public",
    "leak": "public",
    "coordination_attempt": "participants",
    "resource_commitment": "participants",
    "observed_action": "public",          # a public act with no message content
    "private_action": "participants",     # a non-public act others may still learn of later
}

#: ontology action → semantic event type (deterministic; family-independent names resolve once)
_ACTION_EVENT_TYPES = {
    "support": "public_endorsement", "oppose": "public_opposition",
    "endorse": "public_endorsement", "defect": "public_opposition",
    "accept": "concession", "concede": "concession", "counteroffer": "proposal",
    "reject": "refusal", "refuse": "refusal", "veto": "formal_vote_cast",
    "approve": "formal_vote_cast", "threaten": "threat", "escalate": "threat",
    "reveal": "information_release", "reveal_information": "information_release",
    "leak": "leak", "coordinate": "coordination_attempt",
    "request_approval": "request", "clarify": "request", "seek_mediator": "request",
    "donate": "resource_commitment", "allocate_budget": "resource_commitment",
    "authorize": "order_issued", "enforce": "order_issued",
}


def _hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class SemanticEvent:
    """Versioned semantic representation of one social/causal event.

    `exact_content` is the literal language where one exists (a statement, a message); the
    qualitative `semantic_content` carries commitments/intent/ambiguity. `intended_audience`
    is the SOURCE's intent; `actual_recipients` is filled by the observation router — the two
    may legitimately differ (leaks, missed deliveries, summaries)."""

    event_id: str
    event_type: str
    actor_id: str
    targets: list = field(default_factory=list)
    exact_content: str | None = None
    semantic_content: dict = field(default_factory=dict)
    channel: str = "direct"
    intended_audience: list = field(default_factory=list)
    actual_recipients: list = field(default_factory=list)
    observability: str = "participants"        # public | participants | institutional | private
    timestamp: float = 0.0
    provenance: dict = field(default_factory=dict)
    credibility_context: dict = field(default_factory=dict)
    institutional_context: dict = field(default_factory=dict)
    linked_action_id: str = ""
    parent_event_ids: list = field(default_factory=list)
    world_hypothesis_id: str = ""
    schema_version: str = SEMANTIC_EVENT_SCHEMA

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticEvent":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}  # type: ignore[attr-defined]
        return cls(**known)

    def semantic_signature(self) -> str:
        """Duplicate-detection signature: same actor, same semantic act, same targets, same
        content gist ⇒ same signature. Used by quiescence control — two actors repeatedly
        acknowledging each other must not generate an infinite cascade."""
        return _hash({"a": self.actor_id, "t": self.event_type,
                      "targets": sorted(self.targets),
                      "gist": (self.exact_content or "")[:80],
                      "name": self.semantic_content.get("action_name", "")})[:16]


def semantic_event_type(action) -> str:
    """Deterministic action → semantic event type. Named mappings first; then structural
    derivation from observability + target + content."""
    name = str(getattr(action, "action_name", ""))
    if name in _ACTION_EVENT_TYPES:
        return _ACTION_EVENT_TYPES[name]
    obs = (getattr(action, "observability", None) or {}).get("default", "participants")
    target = getattr(getattr(action, "target", None), "target_id", "")
    target_type = getattr(getattr(action, "target", None), "target_type", "none")
    has_content = bool((getattr(action, "parameters", None) or {}).get("content")
                       or (getattr(action, "parameters", None) or {}).get("intended_effect"))
    if target_type == "institution":
        return "institutional_submission"
    if obs == "public":
        return "public_statement" if has_content else "observed_action"
    if target:
        return "private_communication"
    return "private_action"


def _extra_targets(action, world) -> list:
    """Additional explicit targets beyond `action.target`: `parameters.additional_targets`,
    plus targets named in the decision's linked actions as `<act>@<entity>` when the entity is
    a real world entity. This is how a single chosen action fans out into per-target events."""
    params = getattr(action, "parameters", None) or {}
    out = [str(t) for t in (params.get("additional_targets") or []) if str(t)]
    for part in (params.get("linked_actions") or []):
        s = str(part)
        if "@" in s:
            cand = s.split("@", 1)[1].strip()
            if cand in (getattr(world, "entities", None) or {}):
                out.append(cand)
    seen, uniq = set(), []
    primary = getattr(getattr(action, "target", None), "target_id", "")
    for t in out:
        if t and t != primary and t not in seen and t in (world.entities or {}):
            seen.add(t)
            uniq.append(t)
    return uniq


def compile_semantic_events(action, world, *, decision=None, parent_event_ids=None,
                            depth: int = 0) -> list:
    """EXECUTED TypedAction → one or more SemanticEvents (deterministic, no LLM, no ontology
    coefficient required). Multi-target communications compile into one event per target so
    each recipient's observation and reaction are independent."""
    params = dict(getattr(action, "parameters", None) or {})
    if decision is not None:
        params.setdefault("linked_actions", list(getattr(decision, "linked_actions", []) or []))
    action = _with_params(action, params)
    etype = semantic_event_type(action)
    obs = (action.observability or {}).get("default", "participants")
    obs = obs if obs in ("public", "participants", "institutional", "private") else "participants"
    content = (params.get("content") or params.get("novel_description")
               or params.get("intended_effect") or "")
    semantic = {
        "action_name": action.action_name,
        "action_family": action.action_family,
        "intended_effect": str(params.get("intended_effect", ""))[:400],
        "commitments": [c for c in (action.commitments_created or []) if isinstance(c, dict)],
        "ambiguity": ("explicit" if params.get("content") else "inferred_from_action"),
        "timing": str(params.get("timing", "immediate")),
    }
    if decision is not None and getattr(decision, "decision_summary", ""):
        semantic["decision_summary"] = str(decision.decision_summary)[:300]
    primary_target = action.target.target_id
    target_list = ([primary_target] if primary_target else []) + _extra_targets(action, world)
    channel = ("public_broadcast" if obs == "public" else
               ("institutional_channel" if action.target.target_type == "institution"
                else "direct_private"))
    wh = (getattr(world, "uncertainty_meta", None) or {}).get("joint_world_hypothesis") or {}
    base = {
        "event_type": etype, "actor_id": action.actor_id,
        "exact_content": str(content)[:800] or None, "semantic_content": semantic,
        "channel": channel, "observability": obs,
        "timestamp": float(world.clock.now),
        "provenance": {"compiled_from_action": action.action_id,
                       "compiler": "semantic-events-1.0",
                       "action_source": (action.provenance or {}).get("source", ""),
                       "cascade_depth": int(depth)},
        "credibility_context": {"source_actor": action.actor_id,
                                "source_role": action.actor_role,
                                "first_hand": True},
        "institutional_context": ({"institution_id": primary_target}
                                  if action.target.target_type == "institution" else {}),
        "linked_action_id": action.action_id,
        "parent_event_ids": list(parent_event_ids or []),
        "world_hypothesis_id": str(wh.get("hypothesis_id", "")),
    }
    events = []
    if obs in ("public", "institutional") or not target_list:
        audience = ["*"] if obs == "public" else list(target_list)
        events.append(SemanticEvent(
            event_id="sev_" + _hash({**base, "targets": target_list})[:20],
            targets=list(target_list), intended_audience=audience, **base))
    else:
        # private/participants multi-target: one event PER TARGET — each recipient observes
        # their own copy; whether they compare notes is their own later decision.
        for t in target_list:
            events.append(SemanticEvent(
                event_id="sev_" + _hash({**base, "targets": [t]})[:20],
                targets=[t], intended_audience=[t], **base))
    return events


def _with_params(action, params):
    action.parameters = params
    return action
