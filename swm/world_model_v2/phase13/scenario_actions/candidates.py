"""Concrete actions and plans — exact causal content, no verb labels, no parameter buckets.

A candidate preserves everything that could distinguish materially different interventions:
exact intent text, exact targets, exact content, exact terms, channel, audience,
observability, timing, conditions, dependencies, resource commitments, authority basis,
implementation steps, fallback, stop conditions, assumptions, provenance, ancestry. Fields
that do not apply to a scenario stay empty — nothing forces them.

Semantic identity is the full causal intervention, hashed without truncation or bucketing.
Merging is CONSERVATIVE: exact identity merges automatically (recorded); anything else
merges only through an explicit judge whose claim, evidence, and method are recorded on the
survivor. False merges are worse than duplicates, so the default judge refuses.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

CANDIDATE_KIND = "scenario.concrete.action.v1"


def _hash(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class ConditionSpec:
    """An observation-predicate a contingent step is gated on. Evaluated ONLY against the
    decision-maker's observable projection (their visible records + delivered information) —
    structurally never against hidden simulator state."""
    kind: str = "record"                # record | information | time | resource
    record_type: str = ""
    field: str = ""
    op: str = "exists"                  # exists | eq | ne | in | gte | lte | contains
    value: object = None
    description: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanStep:
    """One concrete thing the decision-maker attempts. `intent` is their exact words for it;
    `exact_content` is the artifact/message/terms text when one exists. `compiled_ops` is
    filled ONCE by the direct-effect compiler and replayed identically across matched worlds."""
    step_id: str
    intent: str = ""                                     # the exact intended act, verbatim
    target_ids: list = field(default_factory=list)
    channel: str = ""
    exact_content: str = ""
    terms: dict = field(default_factory=dict)            # structured terms (amounts, dates, offers)
    timing_ts: float = None                              # None = at decision time / after deps
    after_steps: list = field(default_factory=list)      # step ids this one waits for
    conditions: list = field(default_factory=list)       # [ConditionSpec] — all must hold to fire
    max_condition_checks: int = 3
    visibility: str = "participants"                     # public | participants | private
    audience: list = field(default_factory=list)
    resource_commitments: dict = field(default_factory=dict)   # name -> amount
    authority_basis: str = ""
    duration_s: float = 0.0
    reversible: bool = None                              # None = unstated (reported as such)
    compiled_ops: list = field(default_factory=list)     # generated-world kernel ops (compiler fills)
    compile_meta: dict = field(default_factory=dict)
    unresolved: list = field(default_factory=list)       # causal steps the compiler could not model
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["conditions"] = [c.as_dict() if isinstance(c, ConditionSpec) else c
                           for c in self.conditions]
        return d

    def causal_payload(self) -> dict:
        """Everything that could make this step a materially different intervention. No
        truncation, no numeric bucketing — exact content and exact terms are identity."""
        return {"intent": self.intent, "targets": sorted(map(str, self.target_ids)),
                "channel": self.channel, "content": self.exact_content,
                "terms": {str(k): self.terms[k] for k in sorted(self.terms)},
                "timing": self.timing_ts, "after": sorted(self.after_steps),
                "conditions": [c.as_dict() if isinstance(c, ConditionSpec) else c
                               for c in self.conditions],
                "visibility": self.visibility, "audience": sorted(map(str, self.audience)),
                "resources": {str(k): self.resource_commitments[k]
                              for k in sorted(self.resource_commitments)}}


@dataclass
class ConcreteAction:
    """A candidate intervention: one step, a multi-step plan, or (with conditions) a
    contingent plan. `strategy_class` names the causal theory it instantiates — used for
    diversity protection during search, never for execution semantics."""
    candidate_id: str
    actor_id: str
    title: str = ""
    strategy_class: str = ""                            # the causal theory, in scenario terms
    causal_theory: str = ""                             # how these steps are supposed to reach the goal
    steps: list = field(default_factory=list)           # [PlanStep]
    stop_conditions: list = field(default_factory=list)  # [ConditionSpec] plan halts when any holds
    fallback: str = ""                                   # stated behavior when a step fails
    assumptions: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    source: str = "user"                                 # user | goal_backward | affordance | orthogonal | revision
    parent_ids: list = field(default_factory=list)       # ancestry (revisions/crossover)
    revision_reason: str = ""
    original_text: str = ""                              # user-supplied natural language, verbatim
    schema_id: str = ""
    language_hash: str = ""
    unresolved: list = field(default_factory=list)
    implementation_support: str = "unstated"             # unstated | user_assumption | evidence | scenario_mechanism
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [s.as_dict() if isinstance(s, PlanStep) else s for s in self.steps]
        d["stop_conditions"] = [c.as_dict() if isinstance(c, ConditionSpec) else c
                                for c in self.stop_conditions]
        return d

    def identity(self) -> str:
        """The semantic identity of the complete causal intervention."""
        return _hash({"actor": self.actor_id,
                      "steps": [s.causal_payload() for s in self.steps],
                      "stops": [c.as_dict() if isinstance(c, ConditionSpec) else c
                                for c in self.stop_conditions]})[:16]

    def is_contingent(self) -> bool:
        return any(getattr(s, "conditions", None) for s in self.steps)

    def all_unresolved(self) -> list:
        out = list(self.unresolved)
        for s in self.steps:
            out.extend(s.unresolved)
        return out


def single_step_action(candidate_id: str, actor_id: str, intent: str, *, target_ids=(),
                       exact_content: str = "", channel: str = "", terms: dict = None,
                       timing_ts: float = None, visibility: str = "participants",
                       source: str = "user", original_text: str = "") -> ConcreteAction:
    return ConcreteAction(
        candidate_id=candidate_id, actor_id=actor_id, title=intent[:80], source=source,
        original_text=original_text or intent,
        steps=[PlanStep(step_id=f"{candidate_id}_s1", intent=intent,
                        target_ids=list(target_ids), exact_content=exact_content,
                        channel=channel, terms=dict(terms or {}), timing_ts=timing_ts,
                        visibility=visibility)])


def do_nothing_action(actor_id: str) -> ConcreteAction:
    """The explicit status-quo reference: a candidate with zero steps. It changes nothing and
    is always feasible — the baseline every recommendation is paired against."""
    return ConcreteAction(candidate_id="do_nothing", actor_id=actor_id,
                          title="do nothing (status quo)", strategy_class="status_quo",
                          causal_theory="the world evolves without intervention",
                          source="baseline")


def defer_action(actor_id: str, until_ts: float, *, reconsider: str = "") -> ConcreteAction:
    """Deliberate waiting as a first-class candidate: no world writes now; the value shows up
    through the matched simulation of the un-acted-on world plus a later decision point."""
    a = do_nothing_action(actor_id)
    a.candidate_id = "defer"
    a.title = f"defer until {until_ts}"
    a.strategy_class = "wait_for_information"
    a.causal_theory = reconsider or "conditions may resolve before acting"
    a.provenance["defer_until_ts"] = float(until_ts)
    return a


# ---------------------------------------------------------------- conservative merging
def merge_equivalent(candidates: list, *, judge=None, trace=None) -> tuple:
    """Merge ONLY provably or explicitly-judged equivalent candidates.

    Tier 1 (automatic): identical semantic identity — the complete causal payload matches
    exactly. Tier 2 (optional, off by default): `judge(a_dict, b_dict) -> (equivalent, evidence)`
    may merge surface paraphrases; every merge records both originals, the claim, the
    evidence, and the method. Returns (kept, merges). No judge => no tier-2 merges."""
    kept, merges, by_identity = [], [], {}
    for c in candidates:
        ident = c.identity()
        if ident in by_identity:
            survivor = by_identity[ident]
            rec = {"kept": survivor.candidate_id, "merged": c.candidate_id,
                   "claim": "identical causal content",
                   "evidence": f"semantic identity {ident} equal over the full causal payload",
                   "method": "exact", "equivalence": "exact"}
            merges.append(rec)
            survivor.provenance.setdefault("merged_candidates", []).append(rec)
            if trace is not None:
                trace.record(stage="dedup", role="merge", prompt="", response="",
                             parsed=rec, accepted=True)
            continue
        by_identity[ident] = c
        kept.append(c)
    if judge is not None:
        survivors = []
        for c in kept:
            merged_into = None
            for s in survivors:
                try:
                    equivalent, evidence = judge(s.as_dict(), c.as_dict())
                except Exception:  # noqa: BLE001 — a failing judge merges nothing
                    equivalent, evidence = False, "judge failed; refusing merge"
                if equivalent:
                    rec = {"kept": s.candidate_id, "merged": c.candidate_id,
                           "claim": "surface paraphrase of the same intervention",
                           "evidence": str(evidence)[:300], "method": "llm_judged",
                           "equivalence": "llm_judged"}
                    merges.append(rec)
                    s.provenance.setdefault("merged_candidates", []).append(rec)
                    if trace is not None:
                        trace.record(stage="dedup", role="merge_judge", prompt="",
                                     response=str(evidence)[:300], parsed=rec, accepted=True)
                    merged_into = s
                    break
            if merged_into is None:
                survivors.append(c)
        kept = survivors
    return kept, merges
