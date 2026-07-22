"""Decision invalidation, deterministic prechecks, duplicate suppression, execution
classification and the causal-frontier gate.

Two distinct deterministic layers run BEFORE any actor LLM call:

  INVOCATION layer (`should_invoke`) — the causal-frontier gate at the operator seam: an actor is
  considered only when they observed/received the event, it touches something relevant to them,
  they have authority or a feasible response, and their reaction could still matter. Dynamic
  promotion is preserved (the gate never blocks the runtime's own consequential-promotion path);
  there is no fixed actor-count truncation here.

  DECISION layer (`precheck` + `PriorDecisionValidity`) — once a view/menu exists: skip the call
  when no genuine human choice remains (no authority AND no feasible act, duplicate notification
  carrying no new fact, an unchanged valid prior decision, an already-settled choice). Time
  invalidates a decision only through a MATERIAL temporal condition (deadline window entered/
  passed, scheduled event started, commitment expired) — the projection's day/deadline fields
  change and the context differs; no periodic reconsideration interval exists anywhere.

The deterministic layer may conclude that NO decision exists; it may never decide a human choice:
likelihood, base rates, personality labels and 'only one plausible option' are structurally
inadmissible here (nothing in this module reads them). Every avoided call records its exact
reason; every transition records an execution classification
(fully_mechanical | mechanical_shell_with_uncertain_human_inputs | human_discretion_required |
mixed | under_modeled) inside the normal typed runtime — never a separate shortcut forecaster."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from swm.world_model_v2.lean_context import DecisionContextDifference, DecisionRelevantContext

INVALIDATION_VERSION = "lean.invalidation.v1"

EXECUTION_CLASSES = ("fully_mechanical", "mechanical_shell_with_uncertain_human_inputs",
                     "human_discretion_required", "mixed", "under_modeled")

#: §23 avoided-call reason taxonomy (manifest keys)
AVOIDED_REASONS = ("shared_cohort", "equivalent_decision_context", "unchanged_prior_decision",
                   "duplicate_notification", "irrelevant_trigger", "actor_lacked_authority",
                   "no_observed_event", "outside_causal_frontier",
                   "mechanically_determined_transition", "provider_prefix_reuse_only", "other")


@dataclass
class ExecutionClassification:
    """One transition's §classification record, attached to the branch trace."""
    classification: str
    grounded_inputs: list = field(default_factory=list)
    deterministic_rule: str = ""
    remaining_human_decisions: list = field(default_factory=list)
    actor_calls_avoided: int = 0
    terminal_relevant_field: str = ""
    provenance: str = ""
    as_of: str = ""

    def __post_init__(self):
        if self.classification not in EXECUTION_CLASSES:
            raise ValueError(f"unknown execution classification {self.classification!r}")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PriorDecisionValidity:
    """Branch-local record of the actor's standing decision: the projection it was made under,
    the choice, and the conditions that would invalidate it. Stored on the branch's own actor
    state (worlds are deep copies — never shared)."""
    context_signature: str
    chosen_action: str
    act_or_wait: str
    decided_day: str
    revisit: dict = field(default_factory=dict)
    processed_fact_ids: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PriorDecisionValidity":
        return cls(**{k: d.get(k) for k in
                      ("context_signature", "chosen_action", "act_or_wait", "decided_day",
                       "revisit", "processed_fact_ids") if k in d})


@dataclass
class DecisionMaterialChange:
    """WHY a prior decision no longer settles the situation — the §7 material-change detector's
    positive verdict, with the exact differing components."""
    changed: bool
    components: list = field(default_factory=list)
    named_condition_met: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


LEAN_PRIOR_KEY = "lean_prior_decision"


def load_prior(state) -> PriorDecisionValidity | None:
    raw = getattr(state, "_lean_prior", None) if state is not None else None
    if isinstance(raw, dict) and raw.get("context_signature"):
        return PriorDecisionValidity.from_dict(raw)
    return None


def store_prior(state, prior: PriorDecisionValidity):
    if state is not None:
        state._lean_prior = prior.as_dict()      # rides the branch-local qualitative state object


class DecisionInvalidationPolicy:
    """Deterministic material-change detection between the prior decision's projection and the
    current one. Compares full projections (conservative); the ONLY differences that never count
    as material are the trigger's delivery identity when the payload facts are already processed
    (duplicate notification) — everything else that differs is material by definition of the
    projection."""

    #: projection components whose change is always material (mirrors §7's list)
    MATERIAL_COMPONENTS = ("observations", "private_state", "relationships", "commitments",
                           "stances", "institution_rules", "feasible_actions", "authority",
                           "day", "resources", "trigger", "working_memory", "memories",
                           "prior_decision", "structural_frame_hash", "obstacle")

    def material_change(self, prior: PriorDecisionValidity, prior_ctx_sig: str,
                        ctx: DecisionRelevantContext,
                        prior_ctx: DecisionRelevantContext = None) -> DecisionMaterialChange:
        if prior is None:
            return DecisionMaterialChange(changed=True, components=["no_prior_decision"])
        # a condition the actor NAMED at decision time reopens the choice regardless of the diff
        cond = (prior.revisit or {}).get("condition") or {}
        etype = str(cond.get("etype", "")).strip().lower()
        if etype and etype == str(ctx.trigger.get("etype", "")).strip().lower():
            return DecisionMaterialChange(changed=True,
                                          named_condition_met=f"revisit condition etype={etype}")
        if ctx.signature() == prior.context_signature:
            return DecisionMaterialChange(changed=False)
        if prior_ctx is not None:
            diff = DecisionContextDifference.between(prior_ctx, ctx)
            material = [c for c in diff.differing_components if c in self.MATERIAL_COMPONENTS]
            if not material and self._only_duplicate_trigger(prior, ctx, diff):
                return DecisionMaterialChange(changed=False,
                                              components=diff.differing_components)
            return DecisionMaterialChange(changed=True, components=material
                                          or diff.differing_components)
        # no stored prior projection: conservative — any signature change is material
        return DecisionMaterialChange(changed=True, components=["context_signature_changed"])

    @staticmethod
    def _only_duplicate_trigger(prior: PriorDecisionValidity, ctx: DecisionRelevantContext,
                                diff: DecisionContextDifference) -> bool:
        """True when the only difference is a re-delivery of already-processed facts."""
        if diff.differing_components not in (["trigger"],):
            return False
        payload = ctx.trigger.get("payload_facts") or []
        processed = set(prior.processed_fact_ids or [])
        return bool(payload) and all(f.get("fact_id") in processed for f in payload)


@dataclass
class PrecheckVerdict:
    skip: bool
    reason: str = ""                              # one of AVOIDED_REASONS when skip
    detail: str = ""
    classification: ExecutionClassification = None


def precheck(*, ctx: DecisionRelevantContext, state, view, menu: list, decision: dict,
             policy: DecisionInvalidationPolicy, prior_ctx: DecisionRelevantContext = None,
             as_of: str = "") -> PrecheckVerdict:
    """The DECISION-layer deterministic gate. Order matters: cheapest, least-assuming first.
    It may skip only situations where NO genuine human choice remains — it never predicts one."""
    prior = load_prior(state)
    payload = ctx.trigger.get("payload_facts") or []
    # 1. duplicate notification: every delivered fact already processed AND the standing decision
    #    remains valid under the material-change detector
    if prior is not None and payload:
        processed = set(prior.processed_fact_ids or [])
        if all(f.get("fact_id") in processed for f in payload):
            mc = policy.material_change(prior, prior.context_signature, ctx, prior_ctx)
            if not mc.changed:
                return PrecheckVerdict(
                    skip=True, reason="duplicate_notification",
                    detail=f"{len(payload)} delivered fact(s) all previously processed; "
                           f"standing decision {prior.chosen_action!r} remains valid",
                    classification=ExecutionClassification(
                        classification="fully_mechanical",
                        grounded_inputs=[f["fact_id"] for f in payload],
                        deterministic_rule="duplicate-delivery suppression: no new information "
                                           "and no invalidating condition",
                        actor_calls_avoided=1, provenance="lean_precheck", as_of=as_of))
    # 2. unchanged valid prior decision (same projection, or immaterial drift only)
    if prior is not None:
        mc = policy.material_change(prior, prior.context_signature, ctx, prior_ctx)
        if not mc.changed:
            return PrecheckVerdict(
                skip=True, reason="unchanged_prior_decision",
                detail=f"standing decision {prior.chosen_action!r} (decided {prior.decided_day}) "
                       f"remains valid: no material change",
                classification=ExecutionClassification(
                    classification="fully_mechanical",
                    deterministic_rule="prior-decision validity: identical decision-relevant "
                                       "projection and no named invalidating condition",
                    actor_calls_avoided=1, provenance="lean_precheck", as_of=as_of))
    # 3. nothing observed: no delivered payload AND no observations in view — the actor has
    #    nothing to react to (availability is empty, not merely unnoticed)
    if not payload and not ctx.observations and not ctx.working_memory:
        return PrecheckVerdict(
            skip=True, reason="no_observed_event",
            detail="empty availability set: no delivered bundle, no observed events, nothing "
                   "active in memory",
            classification=ExecutionClassification(
                classification="fully_mechanical",
                deterministic_rule="no-observation suppression: an unobserved event cannot "
                                   "trigger cognition", actor_calls_avoided=1,
                provenance="lean_precheck", as_of=as_of))
    # 4. no authority AND nothing feasible beyond waiting, where waiting is not itself a choice
    #    (a menu of only wait/no-op lines with no authority and no resources)
    lines = [ln.lower() for ln in ctx.feasible_actions]
    only_noop = bool(lines) and all(("wait" in ln or "do nothing" in ln or "no_action" in ln)
                                    for ln in lines)
    if only_noop and not ctx.authority and not ctx.resources:
        return PrecheckVerdict(
            skip=True, reason="actor_lacked_authority",
            detail="no authority, no resources, and the feasible set contains only no-ops — no "
                   "genuine choice remains",
            classification=ExecutionClassification(
                classification="fully_mechanical",
                deterministic_rule="authority/feasibility suppression: empty effective action "
                                   "space", actor_calls_avoided=1, provenance="lean_precheck",
                as_of=as_of))
    # otherwise: a genuine human decision — classify and CALL
    return PrecheckVerdict(
        skip=False,
        classification=ExecutionClassification(
            classification="human_discretion_required",
            remaining_human_decisions=[f"{ctx.actor_id}: {ctx.trigger.get('situation', '')[:80]}"],
            provenance="lean_precheck", as_of=as_of))


def should_invoke(world, event, actor_id: str) -> tuple:
    """INVOCATION-layer causal-frontier gate (operator seam). Conservative: any doubt → invoke.
    Returns (invoke: bool, reason: str). Never reads likelihoods or personality — only structural
    observability/participation facts."""
    try:
        participants = set(getattr(event, "participants", None) or [])
        payload = getattr(event, "payload", None) or {}
        if actor_id in participants:
            return True, "participant"
        bundle = payload.get("observation_bundle")
        if bundle:
            return True, "delivered_bundle"
        visibility = str(getattr(event, "visibility", "") or payload.get("visibility", ""))
        if visibility in ("participants", "private") and actor_id not in participants:
            return False, "no_observed_event"
        return True, "public_or_unknown_visibility"
    except Exception:  # noqa: BLE001 — the gate must fail OPEN (invoke) on any surprise
        return True, "gate_error_fail_open"


class AvoidedCallLedger:
    """Every avoided actor call, by §23 reason — the measured (never projected) savings record."""

    def __init__(self):
        self.by_reason: dict[str, int] = {}
        self.records: list[dict] = []
        self.classifications: list[dict] = []

    def record(self, *, reason: str, actor_id: str, detail: str = "", branch_id: str = ""):
        reason = reason if reason in AVOIDED_REASONS else "other"
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1
        if len(self.records) < 400:
            self.records.append({"reason": reason, "actor_id": actor_id,
                                 "detail": detail[:200], "branch_id": branch_id})

    def record_classification(self, c: ExecutionClassification):
        if len(self.classifications) < 400:
            self.classifications.append(c.as_dict())

    def total(self) -> int:
        return sum(self.by_reason.values())

    def as_dict(self) -> dict:
        by_class: dict[str, int] = {}
        for c in self.classifications:
            k = c.get("classification", "")
            by_class[k] = by_class.get(k, 0) + 1
        return {"version": INVALIDATION_VERSION, "avoided_calls_total": self.total(),
                "avoided_by_reason": dict(sorted(self.by_reason.items())),
                "execution_classifications": by_class,
                "sample_records": self.records[:60],
                "sample_classifications": self.classifications[:60]}
