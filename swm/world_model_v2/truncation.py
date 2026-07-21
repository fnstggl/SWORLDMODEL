"""First-class branch truncation — §20/§21: truncated branch mass is unresolved simulation, not noise.

THE PROBLEM THIS MODULE MAKES IMPOSSIBLE TO HIDE: a branch that hits a safety budget, loses its LLM
provider, times out or reaches a mechanism with no executable model did NOT resolve. Historically such
branches either vanished from the terminal projection (silently renormalizing their probability mass
into whichever branches happened to finish) or were folded into sampling noise. Both are dishonest:
truncated branch mass is unresolved SIMULATION, not Monte Carlo error — more particles never make it
go away; only more compute or more coverage does.

THE CONTRACT (§20 statuses, §21 honest aggregation):

  1. Every branch ends in EXACTLY ONE first-class status from BRANCH_STATUSES. Truncation statuses
     name WHICH budget or failure cut the branch off; `invalid` is reserved for corrupt/inconsistent
     branch state — excluded from inference with its weight REPORTED, never silently renormalized.
  2. Aggregation preserves the weight identity by construction:
         completed_weight + truncated_weight + invalid_weight == total_weight
     — truncated weight is never renormalized away (`aggregate_branch_statuses`).
  3. Terminal distributions over the completed mass are reported WITH truncation bounds
     (`truncation_bounds`): for every option, the lower bound assumes ALL truncated mass resolves
     against it and the upper bound assumes ALL of it resolves toward it. Pure arithmetic over the
     recorded per-branch terminal options — nothing minted.
  4. A recommendation is ELIGIBLE only if the leading action remains best under EVERY admissible
     completion of the truncated mass (worst case: the truncated mass scores the leader at the score
     floor and a rival at the ceiling — `recommendation_eligibility`). Otherwise the recommendation
     MUST be withheld (§21).

Runtime truncation KINDS — the strings temporal_runtime / generated_world / phase4_execution record
today ("safety_max_events_reached", "actor_llm_budget_exhausted", ...) — map onto the status
vocabulary via `map_truncation_kind`. Unknown kinds map conservatively to `truncated_event_budget`
WITH a recorded note; they are never dropped and never promoted to a completed status.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# ------------------------------------------------------------------ §20 first-class branch statuses
BRANCH_STATUSES = (
    "active",                       # still unfolding (transient only; never terminal in a report)
    "completed",                    # reached causal resolution of the readout
    "absorbed",                     # entered an absorbing state before the horizon
    "quiescent",                    # natural causal quiescence (no pending in-horizon events)
    "truncated_actor_budget",       # actor-cognition/invocation budget cut the branch off
    "truncated_event_budget",       # event/safety budget cut the branch off
    "truncated_context_budget",     # context budget cut the branch off
    "truncated_boundary_budget",    # world-boundary expansion budget cut the branch off
    "truncated_missing_mechanism",  # a required mechanism has no executable model
    "truncated_provider_failure",   # LLM provider / cognition-stage failure cut the branch off
    "truncated_timeout",            # wall-clock timeout cut the branch off
    "invalid",                      # corrupt/inconsistent state — excluded, weight reported
)

#: statuses whose mass counts as RESOLVED simulation
COMPLETED_BRANCH_STATUSES = ("completed", "absorbed", "quiescent")
#: statuses whose mass is UNRESOLVED simulation (never renormalized away, never Monte Carlo error)
TRUNCATED_BRANCH_STATUSES = tuple(s for s in BRANCH_STATUSES if s.startswith("truncated_"))

#: existing runtime truncation kinds → first-class §20 branch status
TRUNCATION_KIND_TO_STATUS = {
    "actor_llm_budget_exhausted": "truncated_actor_budget",
    "invocation_safety_budget_reached": "truncated_actor_budget",
    "safety_max_events_reached": "truncated_event_budget",
    "provider_failure_all_families": "truncated_provider_failure",
    "cognition_stage_failure": "truncated_provider_failure",
    "missing_mechanism": "truncated_missing_mechanism",
    "context_budget": "truncated_context_budget",
    "boundary_budget": "truncated_boundary_budget",
    "timeout": "truncated_timeout",
}

#: cap on the pending-event union surfaced in the §21 report (mirrors queue.peek_pending(30))
PENDING_EVENTS_REPORT_CAP = 30


def honest_note() -> str:
    """The one-line doctrine every §21 truncation report carries verbatim."""
    return "truncated branch mass is unresolved simulation, not Monte Carlo error"


def map_truncation_kind(kind: str, *, notes: list = None) -> str:
    """Map a runtime truncation KIND string (as recorded in branch stats today) onto the first-class
    §20 branch status. Unknown kinds are NEVER dropped and NEVER promoted toward completion — they map
    conservatively to `truncated_event_budget`, and when a `notes` list is supplied the unknown kind
    is recorded there so the report shows the vocabulary gap instead of hiding it."""
    status = TRUNCATION_KIND_TO_STATUS.get(str(kind or ""))
    if status is None:
        status = "truncated_event_budget"
        if notes is not None:
            notes.append(f"unknown truncation kind {kind!r} mapped conservatively to "
                         "truncated_event_budget")
    return status


@dataclass
class BranchTruncationRecord:
    """One truncated branch, first-class: which branch, which §20 status, why, when, and what was left
    UNRESOLVED — the pending events that never ran, the actors they name, and the decision trigger
    that never fired. `weight` is the branch's probability mass; the 0.0 default is an equal-weight
    placeholder that aggregation fills (never a claim that the branch is weightless)."""
    branch_id: str
    status: str
    reason: str = ""
    at_ts: float = None
    pending_events: list = field(default_factory=list)
    affected_actors: list = field(default_factory=list)
    unresolved_decision_trigger: dict = field(default_factory=dict)
    weight: float = 0.0                      # equal-weight placeholder; filled by aggregation

    def __post_init__(self):
        if self.status not in BRANCH_STATUSES:
            raise ValueError(f"bad branch status {self.status!r}; must be one of {BRANCH_STATUSES}")


def _event_key(e) -> str:
    """Stable identity for the pending-event union (dict events compare by content, not object id)."""
    if isinstance(e, dict):
        try:
            return json.dumps(e, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(sorted((str(k), str(v)) for k, v in e.items()))
    return str(e)


def aggregate_branch_statuses(branches_stats: list, *, weights: dict = None) -> dict:
    """The §21 truncation report over per-branch stats — the accounting a truncated result carries.

    `branches_stats`: [{branch_id, truncated (bool), truncation (dict), weight?, invalid?, status?}] —
    the shape the temporal runtime already records (`truncation` carries reason/kind, at_ts,
    pending_events, actors_not_processed / affected_actors / actor). `weights` optionally maps
    branch_id → probability mass; otherwise a branch's own `weight` is used, else equal weight 1/n.

    THE IDENTITY THIS FUNCTION GUARANTEES BY CONSTRUCTION:

        completed_weight + truncated_weight + invalid_weight == total_weight

    Truncated weight is NEVER renormalized away and invalid weight is NEVER hidden — both are reported
    next to the completed mass, with the reason histogram, the earliest truncation timestamp, the
    actors affected, and the capped union of pending high-sensitivity events."""
    stats = list(branches_stats or [])
    n = len(stats)
    completed_w = truncated_w = invalid_w = 0.0
    reasons, actors, pending, truncated_ids, notes = {}, set(), [], [], []
    seen_pending = set()
    earliest = None
    for i, b in enumerate(stats):
        b = b or {}
        bid = str(b.get("branch_id", i))
        if weights and bid in weights:
            w = float(weights[bid])
        elif b.get("weight") is not None:
            w = float(b["weight"])
        else:
            w = 1.0 / n if n else 0.0
        trunc = b.get("truncation") or {}
        status = str(b.get("status", ""))
        is_invalid = bool(b.get("invalid")) or status == "invalid"
        is_truncated = (not is_invalid) and (bool(b.get("truncated")) or bool(trunc)
                                             or status in TRUNCATED_BRANCH_STATUSES)
        if is_invalid:
            invalid_w += w
        elif is_truncated:
            truncated_w += w
            truncated_ids.append(bid)
            kind = str(trunc.get("reason") or trunc.get("kind") or status or "unknown")
            reasons[kind] = reasons.get(kind, 0) + 1
            map_truncation_kind(kind, notes=notes)           # records vocabulary gaps, never drops
            ts = trunc.get("at_ts", trunc.get("ts"))
            if isinstance(ts, (int, float)):
                earliest = ts if earliest is None else min(earliest, ts)
            for a in (trunc.get("affected_actors") or trunc.get("actors_not_processed") or []):
                actors.add(str(a))
            if trunc.get("actor"):
                actors.add(str(trunc["actor"]))
            for e in (trunc.get("pending_events") or []):
                key = _event_key(e)
                if key not in seen_pending and len(pending) < PENDING_EVENTS_REPORT_CAP:
                    seen_pending.add(key)
                    pending.append(e)
        else:
            completed_w += w
    total = completed_w + truncated_w + invalid_w            # the identity holds by construction
    return {
        "total_weight": total,
        "completed_weight": completed_w,
        "truncated_weight": truncated_w,
        "invalid_weight": invalid_w,
        "truncated_branch_share": round(truncated_w / total, 6) if total else 0.0,
        "truncation_reasons": reasons,
        "earliest_truncation_ts": earliest,
        "actors_affected": sorted(actors),
        "pending_high_sensitivity_events": pending,
        "truncated_branch_ids": truncated_ids,
        "mapping_notes": notes,
        "honest_note": honest_note(),
    }


def truncation_bounds(distribution_by_branch: dict, truncated_weight: float, options: list) -> dict:
    """§21 bounds: what COULD the answer be once every truncated branch resolves?

    `distribution_by_branch`: {branch_id: terminal_option} for completed branches that resolved to one
    option, or {branch_id: {option: prob}} where a branch carries its own terminal distribution. The
    completed branches share the completed mass (1 − truncated_weight) equally, each contributing its
    share to its terminal option (or spreading it per its own normalized distribution). For each option:

        lower = completed mass on that option              (ALL truncated mass resolves AGAINST it)
        upper = completed mass + truncated_weight, ≤ 1     (ALL truncated mass resolves TOWARD it)

    Pure arithmetic over recorded terminal options — nothing minted, nothing renormalized. With
    truncated_weight == 0 the bounds collapse to the point distribution (lower == upper)."""
    t = max(0.0, min(1.0, float(truncated_weight)))
    branches = dict(distribution_by_branch or {})
    completed_total = 1.0 - t
    mass = {str(o): 0.0 for o in (options or [])}
    if branches:
        per_branch = completed_total / len(branches)
        for terminal in branches.values():
            if isinstance(terminal, dict):
                z = sum(float(p) for p in terminal.values()) or 1.0
                for o, p in terminal.items():
                    mass[str(o)] = mass.get(str(o), 0.0) + per_branch * float(p) / z
            else:
                mass[str(terminal)] = mass.get(str(terminal), 0.0) + per_branch
    return {o: {"lower": round(m, 9), "upper": round(min(1.0, m + t), 9)}
            for o, m in sorted(mass.items())}


def recommendation_eligibility(candidate_scores_completed: dict, truncated_weight: float,
                               score_range: tuple) -> dict:
    """§21 gate: may the leading action be recommended AT ALL while truncated mass is unresolved?

    `candidate_scores_completed`: {action: mean score over the COMPLETED branch mass}. `score_range`
    = (lo, hi) bounds any admissible per-branch score. The worst admissible completion of the
    truncated mass t scores the leader at `lo` and a rival at `hi`; the leader survives iff

        (1 − t)·(score_leader − score_rival) ≥ t·(hi − lo)   for EVERY rival,

    i.e. the completed-mass margin must cover the truncated span. Returns {eligible, leader,
    margin_worst_case, why} (+ challenger / completed_margin / truncated_span for audit). eligible ==
    False means the recommendation MUST be withheld (§21): an admissible completion of the unresolved
    mass could flip the choice, and pretending otherwise would be minting a certainty."""
    t = max(0.0, min(1.0, float(truncated_weight)))
    lo, hi = float(score_range[0]), float(score_range[1])
    span = t * (hi - lo)
    scores = {str(a): float(s) for a, s in (candidate_scores_completed or {}).items()}
    if not scores:
        return {"eligible": False, "leader": None, "margin_worst_case": None,
                "why": "no candidate actions were scored on completed mass; nothing may be recommended"}
    leader = max(scores, key=lambda a: (scores[a], a))       # deterministic tie-break by name
    rivals = {a: s for a, s in scores.items() if a != leader}
    if not rivals:
        return {"eligible": True, "leader": leader, "margin_worst_case": None,
                "truncated_span": round(span, 9),
                "why": "single candidate action; no rival can overtake under any truncated completion"}
    challenger = max(rivals, key=lambda a: (rivals[a], a))
    completed_margin = (1.0 - t) * (scores[leader] - rivals[challenger])
    margin_worst_case = completed_margin - span
    eligible = margin_worst_case >= 0.0
    if eligible:
        why = (f"leader {leader!r} stays best even if all truncated mass ({t:g}) scores it at {lo:g} "
               f"and {challenger!r} at {hi:g}")
    else:
        why = (f"an admissible completion of the truncated mass ({t:g}) lets {challenger!r} overtake "
               f"{leader!r}; recommendation withheld — {honest_note()}")
    return {"eligible": eligible, "leader": leader, "challenger": challenger,
            "margin_worst_case": round(margin_worst_case, 9),
            "completed_margin": round(completed_margin, 9), "truncated_span": round(span, 9),
            "why": why}
