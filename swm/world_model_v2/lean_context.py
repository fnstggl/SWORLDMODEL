"""Decision-relevant context projection — the exactness standard of the lean profile.

The lean actor cache never compares whole worlds and never uses fuzzy similarity. It compares a
DETERMINISTIC PROJECTION of the actor's current decision situation: two invocations are equivalent
iff their projections are EXACTLY equal (`DecisionContextSignature`), where the projection

  * includes every field that could materially affect the current choice (identity, role,
    authority, trigger + causal payload, private-state cohort and branch-local private state,
    beliefs/goals/commitments/relationships, institutional rules and pressures, the information
    actually available to the actor, working/episodic memory content, deadline relations (day
    granularity — the decision prompt itself renders dates by day), the feasible action set with
    targets/content/timing, resources, prior decisions and their invalidation conditions, the
    actor-local structural frame, prompt/schema versions, backend fingerprint, and the behavioral
    replicate index), and
  * structurally EXCLUDES implementation identity that cannot affect the decision (branch/particle
    ids, trace/event UUIDs, dict ordering, exact sub-day timestamps, provenance ordering,
    unrelated world objects, other actors' private states) — excluded by never being READ, not by
    being read and dropped.

A false cache miss costs one provider call; a false hit can corrupt a simulated world — every
ambiguous field is therefore INCLUDED. Canonical evidence: observation content is canonicalized
(whitespace-collapsed, id-stripped) into stable fact records so duplicate deliveries and re-ordered
bundles project identically, while any wording difference that survives canonicalization is treated
as potentially material (a miss). No LLM judge, no embeddings, no nearest-neighbour anywhere in
the lookup path."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict

CONTEXT_SCHEMA_VERSION = "lean.context.v1"

#: Components every projection MUST carry — the conservative floor of any DecisionDependencySpec.
#: (Names mirror §5 of the lean contract; tests enforce the floor.)
MANDATORY_COMPONENTS = (
    "actor_identity", "actor_role", "authority", "trigger", "private_state",
    "beliefs_goals_commitments", "relationships", "institution_rules", "information_available",
    "memory", "deadline_relation", "feasible_actions", "resources", "prior_decision",
    "structural_assumptions", "versions",
)

_WS = re.compile(r"\s+")


def _norm_text(s, cap: int = 800) -> str:
    """Whitespace-collapsed text — the canonical wording. No case folding, no stemming: wording
    differences beyond whitespace are conservatively treated as material."""
    return _WS.sub(" ", str(s or "").strip())[:cap]


def canonical_fact_id(content, channel: str = "") -> str:
    """Stable id for one proposition as delivered to an actor: hash of the canonical wording plus
    the channel CLASS (not the delivery instance). Duplicate deliveries of the same fact map to the
    same id; genuinely different wording maps to different ids (conservative)."""
    return hashlib.sha256(f"{_norm_text(channel, 60)}\x00{_norm_text(content)}".encode()) \
        .hexdigest()[:24]


def canonicalize(value, *, depth: int = 0, max_depth: int = 8):
    """Deterministic canonical form: dicts sorted by key, text normalized, bounded depth. Never
    called on raw world objects — only on already-projected material."""
    if depth > max_depth:
        return None
    if isinstance(value, str):
        return _norm_text(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [canonicalize(v, depth=depth + 1) for v in list(value)[:48]]
    if isinstance(value, dict):
        items = sorted(((str(k)[:80], v) for k, v in value.items()), key=lambda kv: kv[0])[:48]
        return {k: canonicalize(v, depth=depth + 1) for k, v in items}
    return _norm_text(repr(value), 120)


def _canon_dict(d: dict) -> dict:
    return {str(k)[:80]: canonicalize(v, depth=1) for k, v in sorted((d or {}).items(),
                                                                     key=lambda kv: str(kv[0]))}


@dataclass
class DecisionDependencySpec:
    """WHICH projection components this decision depends on. A compilation stage may PROPOSE a
    narrowed spec; `validate()` conservatively re-adds every mandatory component — narrowing below
    the floor is structurally impossible. v1 ships the full floor for every decision (include when
    uncertain); recorded so future narrowing is auditable."""
    trigger_etype: str = ""
    components: tuple = MANDATORY_COMPONENTS
    proposed_by: str = "conservative_default"
    narrowed: list = field(default_factory=list)          # components a proposal tried to drop

    def validate(self) -> "DecisionDependencySpec":
        missing = [c for c in MANDATORY_COMPONENTS if c not in self.components]
        if missing:
            self.narrowed = sorted(set(self.narrowed) | set(missing))
            self.components = tuple(dict.fromkeys(list(self.components) + missing))
        return self

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionRelevantContext:
    """The compact, deterministic projection of one actor invocation. Every field is already
    canonical (built only by `DecisionRelevantContextBuilder`)."""
    schema_version: str = CONTEXT_SCHEMA_VERSION
    actor_id: str = ""
    actor_role: str = ""
    authority: list = field(default_factory=list)
    cohort_id: str = ""
    private_state: dict = field(default_factory=dict)      # branch-local qualitative state, canonical
    trigger: dict = field(default_factory=dict)            # {etype, situation, payload_facts}
    observations: list = field(default_factory=list)       # canonical fact records, sorted
    working_memory: list = field(default_factory=list)     # content sequence (recency order is causal)
    memories: list = field(default_factory=list)           # accessible episodic content, deterministic
    commitments: list = field(default_factory=list)
    stances: list = field(default_factory=list)
    relationships: dict = field(default_factory=dict)
    institution_rules: list = field(default_factory=list)
    resources: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
    feasible_actions: list = field(default_factory=list)   # menu lines, builder order (causal)
    day: str = ""                                          # decision-day (deadline relation lives here
    public_facts_hash: str = ""                            # + in the shared scheduled-reality facts)
    prior_decision: dict = field(default_factory=dict)
    structural_frame_hash: str = ""
    prompt_version: str = ""
    backend_fingerprint: str = ""
    replicate_index: int = 0
    dependency_spec: dict = field(default_factory=dict)
    obstacle: str = ""                                     # revision-round context (perceived block)

    def signature(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def as_dict(self) -> dict:
        return asdict(self)


#: field-level labels used by difference reports and the equivalence certificate
_DIFF_FIELDS = ("actor_id", "actor_role", "authority", "cohort_id", "private_state", "trigger",
                "observations", "working_memory", "memories", "commitments", "stances",
                "relationships", "institution_rules", "resources", "action_history",
                "feasible_actions", "day", "public_facts_hash", "prior_decision",
                "structural_frame_hash", "prompt_version", "backend_fingerprint",
                "replicate_index", "obstacle", "schema_version")


@dataclass
class DecisionContextDifference:
    """Component-wise difference between two projections — powers tests, the miss audit and
    `explain_equivalence`."""
    equal: bool
    differing_components: list = field(default_factory=list)

    @classmethod
    def between(cls, a: DecisionRelevantContext, b: DecisionRelevantContext
                ) -> "DecisionContextDifference":
        diffs = [f for f in _DIFF_FIELDS if getattr(a, f) != getattr(b, f)]
        return cls(equal=not diffs, differing_components=diffs)


@dataclass
class DecisionEquivalenceCertificate:
    """Why one branch's decision may serve another: the exact matched projection, what was ignored
    (implementation identity never read into the projection), and the receiving-branch
    revalidation results. Auditable without exposing chain-of-thought."""
    context_hash: str
    actor_id: str
    cohort_id: str
    source_branch: str
    receiving_branch: str
    matched_components: list
    ignored_differences: list = field(default_factory=lambda: [
        {"field": "branch_id/particle_index", "why": "implementation identity — never projected"},
        {"field": "event/trace UUIDs", "why": "delivery identity, not content — never projected"},
        {"field": "sub-day timestamps", "why": "decision prompts render day granularity; only the "
                                               "deadline relation is decision-relevant"},
        {"field": "dict/provenance ordering", "why": "canonicalized before hashing"},
        {"field": "unrelated world objects / other actors' private states",
         "why": "outside the actor's information boundary — never projected"}])
    revalidation: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class DecisionRelevantContextBuilder:
    """Builds the projection from the SAME inputs the decision path itself consumes (view, state,
    situation, menu, delivered bundle) — nothing more, nothing less. The builder reads no branch
    ids, no particle indices, no UUIDs, no raw world entities."""

    def __init__(self, *, prompt_version: str, backend_fingerprint: str,
                 structural_frame: str = "", public_facts=None):
        self.prompt_version = str(prompt_version)
        self.backend_fingerprint = str(backend_fingerprint)
        self.structural_frame_hash = hashlib.sha256(
            _norm_text(structural_frame, 2000).encode()).hexdigest()[:16] if structural_frame \
            else ""
        pf = json.dumps(canonicalize(list(public_facts or [])), sort_keys=True, default=str)
        self.public_facts_hash = hashlib.sha256(pf.encode()).hexdigest()[:16]

    # -- component projections ------------------------------------------------------------
    @staticmethod
    def observation_facts(bundle: list) -> list:
        """Delivered bundle → sorted canonical fact records. Sorting is safe AND required: the
        bundle's arrival order varies with queue internals, while the actor's information set is
        order-free; the lean prompt renders the same sorted order, so signature and prompt agree."""
        facts, seen = [], set()
        for it in bundle or []:
            if not isinstance(it, dict):
                continue
            content = it.get("content", it.get("summary", ""))
            channel = str(it.get("channel", ""))[:40]
            fid = canonical_fact_id(content, channel)
            if fid in seen:
                continue                                   # duplicate delivery of the same fact
            seen.add(fid)
            facts.append({"fact_id": fid, "channel": _norm_text(channel, 40),
                          "source": _norm_text(it.get("source", ""), 60),
                          "content": _norm_text(content, 600),
                          "urgency": _norm_text(it.get("urgency", ""), 20),
                          "interrupting": bool(it.get("interrupting"))})
        return sorted(facts, key=lambda f: (f["channel"], f["fact_id"]))

    @staticmethod
    def _state_projection(state) -> dict:
        if state is None:
            return {}
        d = state.as_dict() if hasattr(state, "as_dict") else dict(state or {})
        d.pop("revision_log", None)                        # provenance ordering — not content
        return _canon_dict(d)

    @staticmethod
    def _memory_projection(world, actor_id: str) -> tuple:
        """Working-memory content sequence (recency order is causal) + accessible episodic
        content. Item ids and timestamps are dropped; day-relative staleness is decision-relevant
        only through what retrieval later exposes deterministically."""
        try:
            from swm.world_model_v2.bounded_cognition import load_memory, load_working_memory
            wm = load_working_memory(world, actor_id)
            mem = load_memory(world, actor_id)
            wm_seq = [{"kind": _norm_text(i.kind, 40), "content": _norm_text(i.content, 400)}
                      for i in wm.active()]
            epi = [{"content": _norm_text(m.content, 300),
                    "salience": _norm_text(getattr(m, "salience", ""), 12)}
                   for m in mem.episodic if getattr(m, "accessible", True)][:16]
            tasks = [{"task": _norm_text(t.get("task", ""), 200)}
                     for t in mem.unresolved_tasks[:8] if isinstance(t, dict)]
            return wm_seq, epi + tasks
        except Exception:  # noqa: BLE001 — a world without memory stores projects empty memory
            return [], []

    def build(self, *, view, state, situation: str, menu: list, decision: dict,
              day: str, replicate_index: int = 0, prior_decision: dict = None,
              world=None, obstacle: str = "") -> DecisionRelevantContext:
        spec = DecisionDependencySpec(trigger_etype=str((decision or {}).get("etype", ""))) \
            .validate()
        wm_seq, epi = self._memory_projection(world, view.actor_id) if world is not None \
            else ([], [])
        return DecisionRelevantContext(
            actor_id=str(view.actor_id),
            actor_role=_norm_text(view.actor_role, 80),
            authority=sorted(_norm_text(a, 80) for a in (view.authority or [])),
            cohort_id=_norm_text(getattr(state, "hypothesis_id", ""), 60),
            private_state=self._state_projection(state),
            trigger={"etype": _norm_text((decision or {}).get("etype", ""), 60),
                     "situation": _norm_text(situation, 400),
                     "payload_facts": self.observation_facts(
                         (decision or {}).get("observation_bundle") or [])},
            observations=self.observation_facts(
                [e for e in (view.observed_events or []) if isinstance(e, dict)]),
            working_memory=wm_seq,
            memories=epi,
            commitments=[_canon_dict(c) for c in (view.commitments or [])
                         if isinstance(c, dict)][:12],
            stances=[_canon_dict(s) for s in (view.stances or []) if isinstance(s, dict)][:12],
            relationships={_norm_text(r.get("other_actor", ""), 60):
                           _norm_text(r.get("relation", ""), 80)
                           for r in (view.relationships or []) if isinstance(r, dict)},
            institution_rules=sorted(
                (json.dumps(_canon_dict(r), sort_keys=True, default=str)[:300]
                 for r in (view.institution_rules or []) if isinstance(r, dict))),
            resources=sorted(_norm_text(k, 60) for k in (view.resources or {})),
            action_history=[_norm_text(h.get("action", ""), 60)
                            for h in (view.action_history or [])[-6:] if isinstance(h, dict)],
            feasible_actions=[_norm_text(m.get("line", ""), 200)
                              for m in (menu or []) if isinstance(m, dict)],
            day=str(day)[:10],
            public_facts_hash=self.public_facts_hash,
            prior_decision=_canon_dict(prior_decision or {}),
            structural_frame_hash=self.structural_frame_hash,
            prompt_version=self.prompt_version,
            backend_fingerprint=self.backend_fingerprint,
            replicate_index=int(replicate_index),
            dependency_spec=spec.as_dict(),
            obstacle=_norm_text(obstacle, 240))


def context_rng_seed(signature: str, *, replicate_index: int = 0) -> int:
    """The lean behavioral-replicate seed law: stochastic actor-local draws (memory-retrieval
    failure, tie-breaks) are seeded by the DECISION CONTEXT, not the particle index — equivalent
    situations behave identically at replicate 0, and additional behavioral replicates are an
    explicit, indexed choice (§replicate policy), never accidental provider noise."""
    return int(hashlib.sha256(f"{signature}\x00r{int(replicate_index)}".encode())
               .hexdigest()[:12], 16)
