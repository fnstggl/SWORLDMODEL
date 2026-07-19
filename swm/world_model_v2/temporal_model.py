"""The SCENARIO TEMPORAL MODEL — first-class, per-scenario generated temporal structure (§2–§3
of the event-driven temporal architecture).

Real timestamps were already load-bearing in World Model V2; what was missing is a model of WHY
things happen when they happen. This module is the typed home for that model:

    real initial time and real calendar
    → real scheduled facts, commitments, deadlines, and process stages
    → scenario-generated temporal model            (this module; compiled in temporal_compiler)
    → events occur when their real causal triggers occur          (temporal_runtime)
    → information travels through actual channels                 (ChannelTemporalModel)
    → actors notice information according to their situation      (ActorTemporalProfile,
                                                                   AttentionProcess)
    → actors reconsider only when something creates a reason      (DecisionTrigger)
    → decisions and actions take situation-specific time          (ResponseProcess, TimingSpec)
    → institutions advance through scenario-specific stages       (InstitutionalProcessModel)
    → continuous processes evolve over the exact elapsed interval (ContinuousProcessSpec)
    → simultaneous events interact without insertion-order artifacts (temporal_runtime batches)
    → the world advances event by event until the real horizon

Nothing here is a fixed global calendar, one decision-cadence number, or a table of generic
channel delays. Every field is generated per scenario (temporal_compiler), carries provenance,
and keeps unknown timing UNKNOWN (TimingSpec.kind == "unresolved") instead of minting precision.

CRN NOTE: everything sampled per world particle (regime durations, latent temporal states,
hazard thresholds) seeds from the PARTICLE ROOT of the branch id (`particle_rng`), not the full
branch id — matched counterfactual arms (`b3` vs `b3:armA`) share one temporal reality except
where an action causally changes it (§21, §23).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, asdict

SCHEMA_VERSION = "temporal-model-1.0"

#: Qualitative timing regimes — the DOCUMENTED sampling semantics of the compiler's qualitative
#: vocabulary, as (median_s, lo80_s, hi80_s). These are deliberately WIDE bands (honest about
#: what "within hours" means), not fitted claims and not per-scenario constants: a compiler that
#: knows more must emit a range or an exact time instead. Log-normal sampling per particle.
TIMING_REGIMES = {
    "immediate":    (5.0,          1.0,          60.0),
    "minutes":      (300.0,        60.0,         1800.0),
    "within_hour":  (1500.0,       300.0,        5400.0),
    "hours":        (3.0 * 3600,   40.0 * 60,    10.0 * 3600),
    "same_day":     (6.0 * 3600,   1.0 * 3600,   16.0 * 3600),
    "next_day":     (24.0 * 3600,  16.0 * 3600,  40.0 * 3600),
    "days":         (3.0 * 86400,  1.0 * 86400,  7.0 * 86400),
    "week":         (7.0 * 86400,  4.0 * 86400,  12.0 * 86400),
    "weeks":        (18.0 * 86400, 8.0 * 86400,  35.0 * 86400),
    "months":       (60.0 * 86400, 25.0 * 86400, 150.0 * 86400),
}
_Z80 = 1.2816


def particle_rng(world, salt: str) -> random.Random:
    """Deterministic stream seeded by the PARTICLE ROOT of the branch id + salt. Matched
    counterfactual arms clone `b3` into `b3:armA` / `b3:armB`; this stream is identical across
    those arms, so sampled temporal realities (schedules, thresholds, regimes) are SHARED unless
    an action causally changes the inputs (§21/§23). Contrast `phase_consumers._branch_rng`,
    which seeds by the full branch id and therefore diverges across arms."""
    root = str(getattr(world, "branch_id", "")).split(":", 1)[0]
    seed = int.from_bytes(hashlib.sha256(f"{root}|temporal|{salt}".encode()).digest()[:8], "big")
    return random.Random(seed)


def _lognormal(rng: random.Random, med: float, lo: float, hi: float,
               clamp_lo: float = None, clamp_hi: float = None) -> float:
    med = max(1e-9, float(med))
    sigma = (math.log(max(hi, 1e-9)) - math.log(max(lo, 1e-9))) / (2 * _Z80) if hi > lo else 0.0
    v = med * math.exp(sigma * rng.gauss(0.0, 1.0)) if sigma > 0 else med
    if clamp_lo is not None:
        v = max(clamp_lo, v)
    if clamp_hi is not None:
        v = min(clamp_hi, v)
    return v


# ---------------------------------------------------------------- TimingSpec: the "when" value
@dataclass
class TimingSpec:
    """The universal representation of WHEN something happens. Exactly one kind:

      exact        ts                        — a known real timestamp (scheduled fact, deadline)
      range        lo_s..hi_s after ref      — bounded duration uncertainty (log-uniformish)
      regime       one of TIMING_REGIMES     — qualitative band, sampled per particle
      calendar     calendar expression       — resolved in a CivilCalendar (tz-aware)
      after_event  dependency + lag          — fires when the referenced event/condition occurs
      unresolved   description only          — the honest "we do not know"; NEVER sampled

    provenance: evidence claim id / "user_context" / "scenario_generated" / "model_knowledge".
    An unresolved spec never silently becomes a number — the runtime carries it as an unresolved
    timing mechanism on the result (§11, §27)."""
    kind: str
    ts: float = None
    lo_s: float = None
    hi_s: float = None
    regime: str = ""
    calendar_expr: str = ""
    calendar_of: str = ""                  # actor/institution whose CivilCalendar resolves it
    depends_on: str = ""                   # event id / condition id for after_event
    lag: dict = None                       # nested TimingSpec dict for after_event lag
    description: str = ""
    provenance: str = "scenario_generated"
    confidence: float = 0.5

    def resolved(self) -> bool:
        return self.kind != "unresolved"

    def sample_duration_s(self, rng: random.Random) -> float:
        """A concrete duration for range/regime kinds (per-particle; caller persists it).
        exact/calendar/after_event resolve elsewhere; unresolved raises — it must never be
        silently coerced into a number."""
        if self.kind == "range":
            lo = max(0.0, float(self.lo_s or 0.0))
            hi = max(lo, float(self.hi_s if self.hi_s is not None else lo))
            if hi <= 0.0:
                return 0.0
            if lo <= 0.0:
                return rng.uniform(0.0, hi)
            return math.exp(rng.uniform(math.log(lo), math.log(hi)))
        if self.kind == "regime":
            med, lo, hi = TIMING_REGIMES.get(self.regime, TIMING_REGIMES["hours"])
            return _lognormal(rng, med, lo, hi, clamp_lo=0.0)
        if self.kind == "exact":
            return 0.0
        raise ValueError(f"TimingSpec kind {self.kind!r} has no samplable duration "
                         f"(unresolved timing must stay unresolved)")

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [])}

    @classmethod
    def from_dict(cls, d: dict) -> "TimingSpec":
        if not isinstance(d, dict):
            return cls(kind="unresolved", description=f"unparseable timing spec: {d!r}")
        known = {f for f in cls.__dataclass_fields__}
        kind = str(d.get("kind", "") or "").strip() or "unresolved"
        if kind not in ("exact", "range", "regime", "calendar", "after_event", "unresolved"):
            return cls(kind="unresolved", description=f"unknown timing kind {kind!r}",
                       provenance=str(d.get("provenance", "scenario_generated")))
        kw = {k: d[k] for k in d if k in known}
        kw["kind"] = kind
        if kind == "regime" and str(kw.get("regime", "")) not in TIMING_REGIMES:
            return cls(kind="unresolved",
                       description=f"unknown timing regime {kw.get('regime')!r}",
                       provenance=str(d.get("provenance", "scenario_generated")))
        try:
            return cls(**kw)
        except TypeError:
            return cls(kind="unresolved", description=f"malformed timing spec: {d!r}")


# ---------------------------------------------------------------- actor temporal profiles (§8)
@dataclass
class ActorTemporalProfile:
    """One individually simulated actor's scenario-specific temporal situation. Unknown fields
    stay None/empty — unknown-ness is honest state, not a gap to fill with invented schedules.
    `latent_hypotheses` lists mutually exclusive current-state hypotheses (available / asleep /
    traveling / in_meetings / crisis_workload …) with priors; ONE is sampled per particle and
    PERSISTS on the world (temporal_runtime.sample_temporal_latents)."""
    actor_id: str
    timezone: str = ""                                    # IANA; "" = unknown
    location_assumption: str = ""
    sleep_window: tuple = None                            # (local_start_hour, local_end_hour)
    active_window: tuple = None                           # work/active local hours
    calendar_commitments: list = field(default_factory=list)   # [{label, ts|rule, source}]
    workload_regime: str = ""                             # e.g. "normal" | "crunch" | unknown ""
    channel_checking: dict = field(default_factory=dict)  # channel_id -> TimingSpec dict (check gap)
    urgency_interrupt: dict = field(default_factory=dict) # channel_id -> {threshold: 0..1, why}
    relationship_priority: dict = field(default_factory=dict)  # sender_id -> 0..1
    pending_obligations: list = field(default_factory=list)
    deadline_awareness: list = field(default_factory=list)
    batching_habit: str = ""
    response_expectation: dict = field(default_factory=dict)   # channel -> TimingSpec dict
    latent_hypotheses: list = field(default_factory=list)      # [{state, prior, why, source}]
    temporal_evidence: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    provenance: str = "scenario_generated"

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- channels (§9–§10)
#: Stage vocabulary for information travel. Not every channel has every stage — the compiler
#: names the stages that exist for THIS channel in THIS scenario.
CHANNEL_STAGES = ("transmitted", "delivered", "available", "exposed", "noticed", "read")


@dataclass
class ChannelTemporalModel:
    """One communication/information channel's scenario-specific temporal process (§10).
    Every stage delay is a TimingSpec (possibly exact-0 for technically instant transmission,
    possibly unresolved). `modifiers` captures the situation-dependence the task requires
    (urgency, relationship, time of day, weekday, workload …) as multiplicative regime shifts
    or overrides — explicit, not a fixed per-channel constant."""
    channel_id: str
    kind: str = ""                                        # direct_message|email|call|public_post|filing|...
    transmission: dict = None                             # TimingSpec dict: initiation → transmitted
    delivery: dict = None                                 # TimingSpec dict: transmitted → delivered/available
    moderation: dict = None                               # optional gate before availability
    exposure: dict = None                                 # availability → a given actor CAN see it
    #                                                       (for broadcast channels: spread process)
    requires_attention: bool = True                       # delivered ≠ noticed (§9); False only for
    #                                                       channels that literally interrupt (a siren)
    failure: dict = field(default_factory=dict)           # {p_fail, mode, source}
    relay: list = field(default_factory=list)             # intermediary hops [{via, lag: TimingSpec}]
    modifiers: list = field(default_factory=list)         # [{when, effect, factor|override, source}]
    provenance: str = "scenario_generated"
    evidence: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- decision triggers (§6)
#: OPEN vocabulary — documented examples, not an exhaustive ontology. The compiler generates the
#: scenario's own triggers; the runtime accepts any string type but every trigger instance must
#: carry causal parents + why-now.
TRIGGER_TYPE_EXAMPLES = (
    "newly_noticed_information", "direct_request", "deadline_approaching", "scheduled_obligation",
    "institutional_stage_reached", "promised_follow_up", "action_completed", "action_failed",
    "observable_state_change", "threshold_crossed", "new_option_available", "condition_became_true",
    "recurring_responsibility", "self_scheduled_revisit")


@dataclass
class DecisionTrigger:
    """The reason an actor is being asked to decide RIGHT NOW. Every actor decision event must
    carry one (§6, invariant 12). No trigger → no decision event → no actor call."""
    trigger_id: str
    trigger_type: str
    actor_id: str
    causal_parent_events: list = field(default_factory=list)
    observed: str = ""                                    # what the actor observed
    decision_relevance: str = ""                          # why it is decision-relevant
    why_now: str = ""                                     # why at this timestamp
    calendar_constraints: str = ""
    temporal_uncertainty: str = ""
    provenance: str = "scenario_generated"

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- institutions (§17)
@dataclass
class InstitutionalStage:
    """One stage of one institution's process. Scheduled ONLY when its entry condition is met at
    runtime (invariant 22) — compile time declares the machine, never pre-schedules the run."""
    stage_id: str
    institution_id: str
    entry_condition: str = ""                             # condition expression / parent stage id
    responsible: str = ""                                 # actor/institution holding this stage
    required_inputs: list = field(default_factory=list)
    earliest_start: dict = None                           # TimingSpec dict
    duration: dict = None                                 # TimingSpec dict
    working_calendar: str = ""                            # calendar id (ScenarioTemporalModel.calendars)
    dependencies: list = field(default_factory=list)
    possible_delays: list = field(default_factory=list)
    possible_accelerations: list = field(default_factory=list)
    deadline: dict = None                                 # TimingSpec dict
    output: str = ""
    next_stages: list = field(default_factory=list)       # stage ids entered on completion
    creates_decision_for: str = ""                        # actor who receives a decision trigger
    provenance: str = "scenario_generated"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class InstitutionalProcessModel:
    process_id: str
    institution_id: str
    stages: list = field(default_factory=list)            # [InstitutionalStage]
    initial_stages: list = field(default_factory=list)    # entered when the process starts
    started_by: str = ""                                  # what starts the process (condition/event)
    provenance: str = "scenario_generated"
    evidence: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"process_id": self.process_id, "institution_id": self.institution_id,
                "stages": [s.as_dict() for s in self.stages],
                "initial_stages": list(self.initial_stages), "started_by": self.started_by,
                "provenance": self.provenance, "evidence": list(self.evidence)}


# ---------------------------------------------------------------- continuous processes (§14)
@dataclass
class ContinuousProcessSpec:
    """A process that evolves over ELAPSED TIME between events — updated over the exact interval
    by temporal_runtime.advance_interval, never by daily ticks. `form` names the analytic update:

      exponential_decay   x' = x * exp(-rate_per_day * dt_days)   toward `floor`
      linear_drift        x' = x + rate_per_day * dt_days          clamped [floor, ceil]
      exponential_approach x' = target + (x-target)*exp(-rate*dt)  relaxation toward target
      logistic            dx/dt = rate * x * (1 - x/ceil)          adaptive internal steps

    active_when: "" = always; else a condition name evaluated on the world (registered by the
    integration layer). Discontinuities and unknown calibration are declared, not hidden."""
    process_id: str
    reads: list = field(default_factory=list)
    writes: str = ""
    form: str = "exponential_decay"
    rate_per_day: float = 0.0
    target: float = None
    floor: float = 0.0
    ceil: float = 1.0
    active_when: str = ""
    units: str = "share"
    discontinuities: list = field(default_factory=list)
    provenance: str = "scenario_generated"
    calibration_status: str = "documented_prior_unfitted"

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- stochastic arrivals (§15–§16)
@dataclass
class StochasticArrivalSpec:
    """A continuous-time hazard/arrival process executed by first-passage threshold crossing
    (temporal_hazards.CumulativeHazardState): per-branch persistent Exp(1) threshold, cumulative
    intensity integrated over real elapsed time, event scheduled at the projected crossing,
    re-projected (accumulated hazard preserved) whenever a declared read field changes."""
    process_id: str
    etype: str
    base_rate_per_day: float = 0.0
    rate_curve: list = field(default_factory=list)        # optional piecewise [ (frac_lo, frac_hi, rate) ]
    state_reads: list = field(default_factory=list)       # fields whose change re-projects the crossing
    participants: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    provenance: str = "scenario_generated"

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- the model
@dataclass
class ScenarioTemporalModel:
    """The per-scenario generated temporal model (§3). One per structural world model — a
    structural ensemble carries one of these per member (§22), sharing only immutable facts."""
    scenario_id: str
    structural_model_id: str = ""
    as_of: float = 0.0
    horizon_ts: float = 0.0
    timezones: dict = field(default_factory=dict)         # actor/institution id -> IANA tz
    calendars: dict = field(default_factory=dict)         # calendar id -> CivilCalendar kwargs dict
    actor_profiles: dict = field(default_factory=dict)    # actor_id -> ActorTemporalProfile
    channels: dict = field(default_factory=dict)          # channel_id -> ChannelTemporalModel
    response_processes: list = field(default_factory=list)  # [{actor, situation, timing: TimingSpec}]
    institutional_processes: list = field(default_factory=list)  # [InstitutionalProcessModel]
    operational_processes: list = field(default_factory=list)    # [{process, stages…}] free-form typed
    continuous_processes: list = field(default_factory=list)     # [ContinuousProcessSpec]
    scheduled_facts: list = field(default_factory=list)   # exact events (ts + label + source) — EXACT
    deadlines: list = field(default_factory=list)         # [{label, ts|TimingSpec, binds, source}]
    dependencies: list = field(default_factory=list)      # [{event, depends_on, why}]
    recurring_obligations: list = field(default_factory=list)   # [RecurrenceRule.as_dict()] — §5 gated
    stochastic_arrivals: list = field(default_factory=list)     # [StochasticArrivalSpec]
    decision_trigger_sources: list = field(default_factory=list)  # generated trigger structure
    simultaneity_rules: list = field(default_factory=list)  # [{events, rule, mechanism}]
    temporal_uncertainties: list = field(default_factory=list)  # [{about, why_unknown, impact}]
    correlated_latents: list = field(default_factory=list)  # [{latent_id, affects: [...], hypotheses}]
    evidence_sources: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    unresolved_mechanisms: list = field(default_factory=list)   # honest unknowns (§27)
    support_classification: str = "scenario_generated_unvalidated"
    schema_version: str = SCHEMA_VERSION
    compilation_trace: list = field(default_factory=list)  # [TemporalTrace.as_dict()] — every LLM call
    critic_findings: list = field(default_factory=list)
    degraded: str = ""                                    # "" | reason the compiler could not run

    def __deepcopy__(self, memo):
        """Immutable-after-compile: per-particle world clones share ONE temporal model object
        (branch-local temporal state lives on the world — latents in quantities, attention
        buffers, hazard states — never on this model)."""
        return self

    # -- hashing ---------------------------------------------------------------------------
    def temporal_model_hash(self) -> str:
        payload = json.dumps(self.as_dict(include_trace=False), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self, *, include_trace: bool = True) -> dict:
        d = {
            "scenario_id": self.scenario_id, "structural_model_id": self.structural_model_id,
            "as_of": self.as_of, "horizon_ts": self.horizon_ts,
            "timezones": dict(self.timezones), "calendars": dict(self.calendars),
            "actor_profiles": {k: (v.as_dict() if hasattr(v, "as_dict") else v)
                               for k, v in self.actor_profiles.items()},
            "channels": {k: (v.as_dict() if hasattr(v, "as_dict") else v)
                         for k, v in self.channels.items()},
            "response_processes": list(self.response_processes),
            "institutional_processes": [p.as_dict() if hasattr(p, "as_dict") else p
                                        for p in self.institutional_processes],
            "operational_processes": list(self.operational_processes),
            "continuous_processes": [p.as_dict() if hasattr(p, "as_dict") else p
                                     for p in self.continuous_processes],
            "scheduled_facts": list(self.scheduled_facts),
            "deadlines": list(self.deadlines),
            "dependencies": list(self.dependencies),
            "recurring_obligations": list(self.recurring_obligations),
            "stochastic_arrivals": [p.as_dict() if hasattr(p, "as_dict") else p
                                    for p in self.stochastic_arrivals],
            "decision_trigger_sources": list(self.decision_trigger_sources),
            "simultaneity_rules": list(self.simultaneity_rules),
            "temporal_uncertainties": list(self.temporal_uncertainties),
            "correlated_latents": list(self.correlated_latents),
            "evidence_sources": list(self.evidence_sources),
            "assumptions": list(self.assumptions),
            "unresolved_mechanisms": list(self.unresolved_mechanisms),
            "support_classification": self.support_classification,
            "schema_version": self.schema_version,
            "critic_findings": list(self.critic_findings),
            "degraded": self.degraded,
        }
        if include_trace:
            d["compilation_trace"] = list(self.compilation_trace)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioTemporalModel":
        m = cls(scenario_id=str(d.get("scenario_id", "")),
                structural_model_id=str(d.get("structural_model_id", "")),
                as_of=float(d.get("as_of", 0.0) or 0.0),
                horizon_ts=float(d.get("horizon_ts", 0.0) or 0.0))
        m.timezones = dict(d.get("timezones") or {})
        m.calendars = dict(d.get("calendars") or {})
        for aid, p in (d.get("actor_profiles") or {}).items():
            if isinstance(p, dict):
                known = {f for f in ActorTemporalProfile.__dataclass_fields__}
                kw = {k: v for k, v in p.items() if k in known}
                kw["actor_id"] = str(kw.get("actor_id", aid) or aid)
                kw = {k: (tuple(v) if k in ("sleep_window", "active_window")
                          and isinstance(v, list) else v) for k, v in kw.items()}
                try:
                    m.actor_profiles[str(aid)] = ActorTemporalProfile(**kw)
                except TypeError:
                    continue
        for cid, c in (d.get("channels") or {}).items():
            if isinstance(c, dict):
                known = {f for f in ChannelTemporalModel.__dataclass_fields__}
                kw = {k: v for k, v in c.items() if k in known}
                kw["channel_id"] = str(kw.get("channel_id", cid) or cid)
                try:
                    m.channels[str(cid)] = ChannelTemporalModel(**kw)
                except TypeError:
                    continue
        m.response_processes = list(d.get("response_processes") or [])
        for p in (d.get("institutional_processes") or []):
            if isinstance(p, dict):
                stages = []
                for s in (p.get("stages") or []):
                    if isinstance(s, dict):
                        known = {f for f in InstitutionalStage.__dataclass_fields__}
                        kw = {k: v for k, v in s.items() if k in known}
                        kw.setdefault("stage_id", str(s.get("stage_id", "")))
                        kw.setdefault("institution_id", str(p.get("institution_id", "")))
                        try:
                            stages.append(InstitutionalStage(**kw))
                        except TypeError:
                            continue
                m.institutional_processes.append(InstitutionalProcessModel(
                    process_id=str(p.get("process_id", "")),
                    institution_id=str(p.get("institution_id", "")), stages=stages,
                    initial_stages=list(p.get("initial_stages") or []),
                    started_by=str(p.get("started_by", "")),
                    provenance=str(p.get("provenance", "scenario_generated")),
                    evidence=list(p.get("evidence") or [])))
        m.operational_processes = list(d.get("operational_processes") or [])
        for p in (d.get("continuous_processes") or []):
            if isinstance(p, dict):
                known = {f for f in ContinuousProcessSpec.__dataclass_fields__}
                kw = {k: v for k, v in p.items() if k in known}
                if kw.get("process_id"):
                    try:
                        m.continuous_processes.append(ContinuousProcessSpec(**kw))
                    except TypeError:
                        continue
        m.scheduled_facts = list(d.get("scheduled_facts") or [])
        m.deadlines = list(d.get("deadlines") or [])
        m.dependencies = list(d.get("dependencies") or [])
        m.recurring_obligations = list(d.get("recurring_obligations") or [])
        for p in (d.get("stochastic_arrivals") or []):
            if isinstance(p, dict):
                known = {f for f in StochasticArrivalSpec.__dataclass_fields__}
                kw = {k: v for k, v in p.items() if k in known}
                if kw.get("process_id") and kw.get("etype"):
                    try:
                        m.stochastic_arrivals.append(StochasticArrivalSpec(**kw))
                    except TypeError:
                        continue
        m.decision_trigger_sources = list(d.get("decision_trigger_sources") or [])
        m.simultaneity_rules = list(d.get("simultaneity_rules") or [])
        m.temporal_uncertainties = list(d.get("temporal_uncertainties") or [])
        m.correlated_latents = list(d.get("correlated_latents") or [])
        m.evidence_sources = list(d.get("evidence_sources") or [])
        m.assumptions = list(d.get("assumptions") or [])
        m.unresolved_mechanisms = list(d.get("unresolved_mechanisms") or [])
        m.support_classification = str(d.get("support_classification",
                                             "scenario_generated_unvalidated"))
        m.critic_findings = list(d.get("critic_findings") or [])
        m.compilation_trace = list(d.get("compilation_trace") or [])
        m.degraded = str(d.get("degraded", ""))
        return m
