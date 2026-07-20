"""The EVENT-DRIVEN TEMPORAL RUNTIME (§6–§21) — the default execution engine for every World
Model V2 rollout.

    pop all events at the earliest timestamp                     (EventQueue.pop_batch)
    → determine explicit causal dependencies                     (parent/dependency ids)
    → group independent simultaneous events                      (canonical content order —
                                                                  insertion order carries NO
                                                                  semantics, invariant 32)
    → evaluate independents from the same pre-batch state        (declared-read deferral +
                                                                  write-path ledger)
    → resolve genuine write conflicts explicitly                 (simultaneity rules / LOUD
                                                                  unmodeled-conflict record,
                                                                  invariant 34)
    → schedule same-time causal descendants into the next microstep (invariant 33)
    → advance continuous processes over the EXACT elapsed interval  (advance_interval, §14)
    → re-project state-dependent hazards, preserving accumulated hazard and thresholds (§16)
    → deliver availability ≠ attention: actors notice per their situation (§9), see the full
      available bundle at one attention event (§20), and decide only on a real trigger (§6)
    → the world advances event by event until the real horizon; exhaustion of a safety budget
      is a RECORDED temporal truncation, never a fake quiescence (§12)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from swm.world_model_v2.events import Event
from swm.world_model_v2.temporal_calendar import (CivilCalendar, next_time_in_window,
                                                  resolve_calendar_expression)
from swm.world_model_v2.temporal_hazards import crossing_is_current, schedule_crossing
from swm.world_model_v2.temporal_model import (DecisionTrigger, ScenarioTemporalModel,
                                               TimingSpec, particle_rng)

#: Latent temporal states → documented attention-cadence semantics. These are the MEANING of
#: the latent-state vocabulary (like TIMING_REGIMES), not scenario constants: a sampled state
#: multiplies the actor's own generated channel-check gap, or defers noticing to a real window.
LATENT_STATE_CHECK_FACTOR = {"available": 1.0, "in_meetings": 3.0, "traveling": 5.0,
                             "crisis_workload": 6.0, "asleep_offset": 1.0}

# §NAP: the numeric stance-watch layer (0.08 material-change hysteresis, 0.30/0.70
# exhaustion/ripeness thresholds over 0-1 progress bars) is removed with the progress bars it
# watched. Stances change through the actors' OWN cognition at their real decision triggers.


@dataclass
class TemporalRunStats:
    """Per-branch temporal accounting for the §27 result contract."""
    event_counts: dict = field(default_factory=dict)
    actor_invocations: dict = field(default_factory=dict)     # actor -> {trigger_type: n}
    decision_triggers: list = field(default_factory=list)     # bounded trigger records
    delivery_to_attention_s: list = field(default_factory=list)
    attention_to_decision_s: list = field(default_factory=list)
    decision_to_action_s: list = field(default_factory=list)
    action_to_completion_s: list = field(default_factory=list)
    same_time_batches: int = 0
    max_batch_size: int = 1
    max_microsteps: int = 0
    simultaneity_conflicts: list = field(default_factory=list)
    events_canceled: int = 0                                  # stale first-passage generations
    events_rescheduled: int = 0                               # hazard re-projections
    pending_at_horizon: list = field(default_factory=list)
    temporally_truncated: bool = False
    truncation: dict = field(default_factory=dict)
    safety_limits: dict = field(default_factory=dict)
    unresolved_timing: list = field(default_factory=list)
    interval_advances: int = 0
    attention_batches: list = field(default_factory=list)     # bounded: bundle sizes
    #: §20 branch-halt contract: set by _record_truncation(halt=True); the branch loop stops
    #: at the current timestamp and the branch carries a first-class truncation status
    branch_halted: bool = False
    branch_status: str = ""                                   # truncation.BRANCH_STATUSES value
    #: §28: broad-prior resolutions the default runtime REFUSED (generic outcome prior /
    #: prior-beta institutional members / prior-beta aggregates) — under_modeled trail
    mechanism_suppressions: list = field(default_factory=list)

    def count(self, etype: str):
        self.event_counts[etype] = self.event_counts.get(etype, 0) + 1

    def as_dict(self) -> dict:
        def _q(xs):
            if not xs:
                return None
            s = sorted(xs)
            return {"n": len(s), "p10": round(s[int(0.1 * (len(s) - 1))], 1),
                    "p50": round(s[len(s) // 2], 1), "p90": round(s[int(0.9 * (len(s) - 1))], 1)}
        return {"event_counts": dict(self.event_counts),
                "actor_invocations": {a: dict(t) for a, t in self.actor_invocations.items()},
                "n_decision_triggers": len(self.decision_triggers),
                "decision_triggers": self.decision_triggers[:40],
                "delivery_to_attention_delays_s": _q(self.delivery_to_attention_s),
                "attention_to_decision_delays_s": _q(self.attention_to_decision_s),
                "decision_to_action_delays_s": _q(self.decision_to_action_s),
                "action_to_completion_delays_s": _q(self.action_to_completion_s),
                "same_time_batches": self.same_time_batches,
                "max_batch_size": self.max_batch_size, "max_microsteps": self.max_microsteps,
                "simultaneity_conflicts": self.simultaneity_conflicts[:20],
                "n_simultaneity_conflicts": len(self.simultaneity_conflicts),
                "events_canceled": self.events_canceled,
                "events_rescheduled": self.events_rescheduled,
                "pending_at_horizon": self.pending_at_horizon[:30],
                "temporally_truncated": self.temporally_truncated,
                "truncation": self.truncation or None,
                "branch_halted": self.branch_halted,
                "branch_status": self.branch_status or ("truncated_event_budget"
                                                        if self.temporally_truncated else
                                                        "completed"),
                "safety_limits": self.safety_limits or None,
                "unresolved_timing": self.unresolved_timing[:20],
                "interval_advances": self.interval_advances,
                "attention_batch_sizes": self.attention_batches[:40],
                "mechanism_suppressions": self.mechanism_suppressions[:20]}


def get_stats(world) -> TemporalRunStats:
    st = getattr(world, "temporal_stats", None)
    if st is None:
        st = TemporalRunStats()
        world.temporal_stats = st
    return st


def temporal_model_of(world) -> ScenarioTemporalModel:
    m = getattr(world, "temporal_model", None)
    return m if isinstance(m, ScenarioTemporalModel) else None


# ---------------------------------------------------------------- §21: temporal latents
def sample_temporal_latents(world, model: ScenarioTemporalModel) -> dict:
    """Sample ONE coherent temporal reality per world particle and PERSIST it: correlated
    latents (a holiday/outage/crisis affecting several parties) sample once per latent group;
    per-actor current-state hypotheses sample once per actor. Seeded from the PARTICLE ROOT so
    matched counterfactual arms share the same temporal reality (§21/§23). Never redrawn
    mid-branch — an actor who is traveling stays traveling until an event changes it."""
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    register_quantity_type("temporal_latent", units="state")
    drawn = {}
    if model is None:
        return drawn

    def _draw(name, hyps, salt):
        qname = f"temporal_latent:{name}"
        q = world.quantities.get(qname)
        if q is not None and getattr(q, "value", None):
            drawn[name] = str(q.value)
            return
        rng = particle_rng(world, f"latent:{salt}")
        z = sum(max(0.0, float(h.get("prior", 0.0) or 0.0)) for h in hyps) or 1.0
        r, acc, state = rng.random() * z, 0.0, str(hyps[-1].get("state", "available"))
        for h in hyps:
            acc += max(0.0, float(h.get("prior", 0.0) or 0.0))
            if r <= acc:
                state = str(h.get("state", "available"))
                break
        world.quantities[qname] = Quantity(name=qname, qtype="temporal_latent", value=state,
                                           timestamp=world.clock.now)
        drawn[name] = state

    for cl in (model.correlated_latents or []):
        if cl.get("hypotheses"):
            _draw(str(cl["latent_id"]), cl["hypotheses"], f"corr:{cl['latent_id']}")
    for aid, prof in (model.actor_profiles or {}).items():
        hyps = getattr(prof, "latent_hypotheses", None) or []
        if hyps:
            _draw(f"actor:{aid}", hyps, f"actor:{aid}")
    return drawn


def actor_latent_state(world, actor_id: str) -> str:
    q = world.quantities.get(f"temporal_latent:actor:{actor_id}")
    return str(getattr(q, "value", "") or "")


# ---------------------------------------------------------------- timing resolution
def _actor_calendar(model: ScenarioTemporalModel, actor_id: str) -> CivilCalendar:
    if model is not None:
        for cal_id, kw in (model.calendars or {}).items():
            if cal_id == actor_id and isinstance(kw, dict):
                try:
                    return CivilCalendar(**{k: (tuple(v) if isinstance(v, list) else v)
                                            for k, v in kw.items()})
                except TypeError:
                    break
        tz = (model.timezones or {}).get(actor_id, "")
        if tz:
            return CivilCalendar(tz=str(tz), provenance="temporal_model_timezone")
    return CivilCalendar(provenance="unknown_tz_utc_default")


def resolve_timing_spec(spec, *, world, model, ref_ts: float, calendar_of: str = "",
                        salt: str = "", stats: TemporalRunStats = None):
    """TimingSpec dict → concrete timestamp, or None for unresolved/after_event (the caller
    keeps those conditional — never coerced to an invented number, §11). range/regime durations
    sample from the PARTICLE stream keyed by `salt` (persistent per particle, matched across
    arms)."""
    if spec in (None, "", {}):
        return None
    ts_spec = TimingSpec.from_dict(spec if isinstance(spec, dict) else {})
    if ts_spec.kind == "exact" and isinstance(ts_spec.ts, (int, float)):
        return float(ts_spec.ts)
    if ts_spec.kind in ("range", "regime"):
        rng = particle_rng(world, f"timing:{salt or ts_spec.regime or 'range'}")
        return float(ref_ts) + ts_spec.sample_duration_s(rng)
    if ts_spec.kind == "calendar":
        cal = _actor_calendar(model, str(ts_spec.calendar_of or calendar_of))
        out = resolve_calendar_expression(ts_spec.calendar_expr, ref_ts, cal)
        if out is None and stats is not None:
            stats.unresolved_timing.append(
                {"kind": "calendar", "expr": ts_spec.calendar_expr, "at": ref_ts})
        return out
    if stats is not None and ts_spec.kind == "unresolved":
        stats.unresolved_timing.append({"kind": "unresolved",
                                        "description": ts_spec.description[:160]})
    return None


# ---------------------------------------------------------------- §9/§10: delivery timing
def channel_delivery_ts(world, model, *, channel_id: str, sent_ts: float, urgency: float = 0.0,
                        recipient: str = "", salt: str = "",
                        stats: TemporalRunStats = None) -> tuple:
    """When a message through this channel becomes technically AVAILABLE to the recipient:
    transmission + delivery (+ moderation, + exposure for broadcast reach). Returns
    (available_ts, provenance). Without a channel model the delivery falls back to a WIDE
    documented regime band sampled per particle — a labeled unknown, not a 60-second constant."""
    ch = (model.channels or {}).get(channel_id) if model is not None else None
    if ch is None:
        rng = particle_rng(world, f"chan_fallback:{channel_id}:{salt}")
        dur = TimingSpec(kind="regime", regime="within_hour",
                         provenance="unmodeled_channel_broad_band").sample_duration_s(rng)
        if stats is not None:
            stats.unresolved_timing.append({"kind": "unmodeled_channel", "channel": channel_id})
        return sent_ts + dur, "unmodeled_channel_broad_band"
    t = float(sent_ts)
    prov = [f"channel:{channel_id}"]
    for stage_name in ("transmission", "delivery", "moderation", "exposure"):
        spec = getattr(ch, stage_name, None)
        if not spec:
            continue
        stage_ts = resolve_timing_spec(spec, world=world, model=model, ref_ts=t,
                                       calendar_of=recipient,
                                       salt=f"{channel_id}:{stage_name}:{salt}", stats=stats)
        if stage_ts is not None and stage_ts > t:
            t = stage_ts
            prov.append(stage_name)
    for mod in (ch.modifiers or []):
        try:
            f = mod.get("factor")
            if f and urgency >= 0.75 and "urgen" in str(mod.get("when", "")).lower():
                t = sent_ts + max(0.0, (t - sent_ts)) * max(0.05, min(1.0, float(f)))
                prov.append("urgency_modifier")
        except (TypeError, ValueError):
            continue
    return max(t, sent_ts), "+".join(prov)


# ---------------------------------------------------------------- §9: attention (notice) time
def compute_notice_ts(world, model, *, actor_id: str, channel_id: str, available_ts: float,
                      urgency: float = 0.0, sender: str = "",
                      stats: TemporalRunStats = None) -> tuple:
    """When THIS actor plausibly NOTICES an item that became available on this channel:
      * urgency above the actor's interrupt threshold → noticed at availability (a call that
        wakes them; the channel must be one they declared interruptible);
      * else: next channel check after availability — the actor's own generated checking gap,
        stretched by their sampled latent state (traveling/crisis multiplies the gap);
      * sleep/active windows defer noticing into the actor's real waking window (tz-aware);
      * relationship priority tightens the gap for senders the actor prioritizes.
    Returns (notice_ts, provenance). Without any profile: a labeled broad regime band."""
    prof = (model.actor_profiles or {}).get(actor_id) if model is not None else None
    cal = _actor_calendar(model, actor_id)
    if prof is not None:
        thr = (prof.urgency_interrupt or {}).get(channel_id, {})
        if thr and urgency >= float(thr.get("threshold", 1.1)):
            return float(available_ts), "urgency_interrupt"
        gap_spec = (prof.channel_checking or {}).get(channel_id) \
            or (prof.channel_checking or {}).get("*")
        if gap_spec:
            rng = particle_rng(world, f"check:{actor_id}:{channel_id}")
            try:
                gap = TimingSpec.from_dict(gap_spec).sample_duration_s(rng)
            except ValueError:
                gap = None
            if gap is not None:
                factor = LATENT_STATE_CHECK_FACTOR.get(actor_latent_state(world, actor_id), 1.0)
                prio = (prof.relationship_priority or {}).get(sender, 0.0)
                factor *= max(0.25, 1.0 - 0.5 * prio)          # prioritized senders get seen sooner
                # the actor checks this channel in a CYCLE: per-particle anchor phase + gap.
                # The next check after availability is the next cycle point — so several items
                # arriving before one check naturally coalesce into one bundle (§20), and the
                # cycle persists coherently within the particle (§21)
                import math as _m
                eff_gap = max(1.0, gap * factor)
                phase = particle_rng(world,
                                     f"phase:{actor_id}:{channel_id}").uniform(0.0, eff_gap)
                anchor_ts = float(getattr(world.clock, "as_of", available_ts)) + phase
                k = _m.ceil(max(0.0, available_ts - anchor_ts) / eff_gap)
                notice = anchor_ts + k * eff_gap
                if notice <= available_ts:
                    notice += eff_gap
                notice = _defer_to_waking(notice, prof, cal)
                return notice, "profile_channel_checking_cycle"
        if prof.sleep_window or prof.active_window:
            notice = _defer_to_waking(available_ts, prof, cal)
            if notice > available_ts:
                return notice, "profile_waking_window"
    rng = particle_rng(world, f"notice_fallback:{actor_id}:{channel_id}:{round(available_ts)}")
    dur = TimingSpec(kind="regime", regime="hours",
                     provenance="unmodeled_attention_broad_band").sample_duration_s(rng)
    if stats is not None:
        stats.unresolved_timing.append({"kind": "unmodeled_attention", "actor": actor_id,
                                        "channel": channel_id})
    return available_ts + dur, "unmodeled_attention_broad_band"


def _defer_to_waking(ts: float, prof, cal: CivilCalendar) -> float:
    win = prof.active_window
    if win is None and prof.sleep_window is not None:
        s0, s1 = prof.sleep_window
        win = (float(s1), float(s0) if float(s0) > float(s1) else 24.0)
    if not win:
        return ts
    try:
        return next_time_in_window(ts, cal, start_hour=float(win[0]), end_hour=float(win[1]))
    except (ValueError, TypeError):
        return ts


# ---------------------------------------------------------------- §9/§20: availability→attention
def record_available_observation(world, *, recipient: str, item: dict, available_ts: float,
                                 channel: str, stats: TemporalRunStats = None) -> None:
    """An item became technically available to an actor — NOT yet noticed, NOT yet in their
    information set (invariants 17/18). It waits in the actor's availability buffer until an
    attention event collects it."""
    store = getattr(world, "temporal_attention", None)
    if store is None:
        store = {}
        world.temporal_attention = store
    buf = store.setdefault(recipient, {"available": [], "scheduled_attention": {}})
    buf["available"].append({**item, "available_ts": float(available_ts),
                             "channel": str(channel)})


def schedule_attention(world, model, *, actor_id: str, channel_id: str, available_ts: float,
                       urgency: float = 0.0, sender: str = "",
                       stats: TemporalRunStats = None):
    """Schedule (or coalesce into) the actor's next attention event covering this item. If an
    attention event is already pending at or before the computed notice time, DO NOT schedule
    another — the pending one will collect the whole bundle (§20). Returns an Event or None."""
    notice_ts, prov = compute_notice_ts(world, model, actor_id=actor_id, channel_id=channel_id,
                                        available_ts=available_ts, urgency=urgency,
                                        sender=sender, stats=stats)
    notice_ts = max(float(notice_ts), float(available_ts))
    store = getattr(world, "temporal_attention", None) or {}
    buf = store.setdefault(actor_id, {"available": [], "scheduled_attention": {}})
    sched = buf["scheduled_attention"]
    pending = sched.get(channel_id)
    if pending is not None and float(pending) <= notice_ts:
        return None                                            # existing check collects the bundle
    sched[channel_id] = notice_ts
    return Event(ts=notice_ts, etype="ctrl_attention", participants=[actor_id],
                 payload={"actor_id": actor_id, "channel": channel_id,
                          "notice_provenance": prov, "scheduled_for": notice_ts},
                 visibility="participants", source="endogenous:temporal_attention")


def collect_attention_bundle(world, *, actor_id: str, now_ts: float, channel: str = "",
                             stats: TemporalRunStats = None) -> list:
    """At an attention event: everything available to the actor by NOW (on the checked channel,
    or all channels for a general check) becomes NOTICED as one bundle — one actor view, one
    invocation, ordered by availability time with sources/channels preserved (§20). Items enter
    the actor's information set HERE, not at delivery (invariant 18)."""
    store = getattr(world, "temporal_attention", None) or {}
    buf = store.get(actor_id)
    if not buf:
        return []
    take, keep = [], []
    for it in buf["available"]:
        same_channel = (not channel) or (it.get("channel") == channel)
        if it["available_ts"] <= now_ts and same_channel:
            take.append(it)
        else:
            keep.append(it)
    buf["available"] = keep
    if channel:
        buf["scheduled_attention"].pop(channel, None)
    else:
        buf["scheduled_attention"].clear()
    take.sort(key=lambda x: (x["available_ts"], str(x.get("iid", ""))))
    if take and world.information is not None:
        for it in take:
            if it.get("iid"):
                try:
                    world.information.expose(actor_id, it["iid"], now_ts,
                                             channel=str(it.get("channel", ""))[:24])
                except KeyError:
                    continue                                # availability without a ledger item
    if stats is not None and take:
        stats.attention_batches.append(len(take))
        for it in take:
            stats.delivery_to_attention_s.append(max(0.0, now_ts - it["available_ts"]))
    return take


# ---------------------------------------------------------------- §6: decision triggers
def make_trigger(*, trigger_type: str, actor_id: str, parents=(), observed: str = "",
                 relevance: str = "", why_now: str = "", provenance: str = "temporal_runtime",
                 uncertainty: str = "") -> dict:
    t = DecisionTrigger(
        trigger_id=f"trg_{abs(hash((trigger_type, actor_id, tuple(parents), observed))) % 10**10:010d}",
        trigger_type=str(trigger_type), actor_id=str(actor_id),
        causal_parent_events=[str(p) for p in parents], observed=str(observed)[:300],
        decision_relevance=str(relevance)[:200], why_now=str(why_now)[:200],
        temporal_uncertainty=str(uncertainty)[:160], provenance=str(provenance))
    return t.as_dict()


def record_invocation(stats: TemporalRunStats, *, actor_id: str, trigger: dict):
    tt = str((trigger or {}).get("trigger_type", "untyped"))
    stats.actor_invocations.setdefault(actor_id, {})
    stats.actor_invocations[actor_id][tt] = stats.actor_invocations[actor_id].get(tt, 0) + 1
    if len(stats.decision_triggers) < 200:
        stats.decision_triggers.append({"actor": actor_id, **{k: v for k, v in
                                        (trigger or {}).items() if k != "actor_id"}})


# ---------------------------------------------------------------- §14: interval evolution
def advance_interval(world, start_ts: float, end_ts: float, temporal_context=None, rng=None,
                     *, operators=(), branch_log=None, stats: TemporalRunStats = None) -> int:
    """Advance every ACTIVE continuous process over the EXACT interval [start_ts, end_ts]:
      * scenario-generated ContinuousProcessSpecs (analytic forms; adaptive internal stepping
        for the logistic form — internal only, no world events, no actor decisions);
      * legacy background operators (memory/salience decay), fed the exact elapsed interval —
        the daily-threshold tick is GONE: a 10-day gap is one exact 10-day update, not ten
        synthetic daily events.
    Returns the number of state writes."""
    elapsed_s = float(end_ts) - float(start_ts)
    if elapsed_s <= 0:
        return 0
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    from swm.world_model_v2.transitions import StateDelta
    n_writes = 0
    model = temporal_model_of(world)
    dt_days = elapsed_s / 86400.0
    if stats is not None:
        stats.interval_advances += 1
    # ---- scenario-generated continuous processes ----
    import math as _m
    for spec in (model.continuous_processes if model is not None else []):
        if spec.active_when and not _condition_active(world, spec.active_when):
            continue
        q = world.quantities.get(spec.writes)
        x = getattr(q, "value", None)
        if not isinstance(x, (int, float)):
            continue
        x = float(x)
        if spec.form == "exponential_decay":
            x2 = spec.floor + (x - spec.floor) * _m.exp(-max(0.0, spec.rate_per_day) * dt_days)
        elif spec.form == "linear_drift":
            x2 = x + spec.rate_per_day * dt_days
        elif spec.form == "exponential_approach":
            tgt = spec.target if spec.target is not None else spec.ceil
            x2 = tgt + (x - tgt) * _m.exp(-max(0.0, spec.rate_per_day) * dt_days)
        elif spec.form == "logistic":
            # adaptive INTERNAL stepping: refine until the answer stabilizes; the grid is
            # invisible — no events, no actor decisions (invariant 25)
            x2, err = _logistic_integrate(x, spec.rate_per_day, spec.ceil or 1.0, dt_days)
            if stats is not None and err > 1e-6:
                stats.unresolved_timing.append({"kind": "integration_error_bound",
                                                "process": spec.process_id, "bound": err})
        else:
            continue
        x2 = max(spec.floor, min(spec.ceil if spec.ceil is not None else float("inf"), x2))
        if abs(x2 - x) > 1e-12:
            register_quantity_type(getattr(q, "qtype", spec.writes), units=spec.units)
            # full float precision: a chain of short intervals must equal one long interval
            # exactly (invariant 23) — rounding would break elapsed-time additivity
            world.quantities[spec.writes] = Quantity(name=spec.writes,
                                                     qtype=getattr(q, "qtype", spec.writes),
                                                     value=x2, timestamp=end_ts)
            n_writes += 1
            if branch_log is not None:
                d = StateDelta(at=end_ts, event_type="interval_evolution",
                               operator=f"continuous:{spec.process_id}",
                               reason_codes=[f"elapsed_days={round(dt_days, 4)}",
                                             f"form={spec.form}"],
                               uncertainty={"provenance": spec.provenance,
                                            "calibration": spec.calibration_status})
                branch_log.append(d.change(f"quantities[{spec.writes}]",
                                           round(x, 6), round(x2, 6)))
    # §NAP: the contested-attrition capacity drain (0.0007/day on a unit-less 0-1 "capacity")
    # was an invented psychology parameter — removed. Exhaustion reaches actors the way it
    # reaches real ones: through typed world state their own cognition observes.
    # ---- legacy background operators over the exact interval ----
    if operators:
        bg = Event(ts=end_ts, etype="background_tick",
                   payload={"elapsed_days": dt_days, "interval": [start_ts, end_ts],
                            "exact_interval": True})
        for op in operators:
            try:
                if op.applicable(world, bg):
                    delta, _ = op.run(world, bg, rng or random.Random(0))
                    if delta is not None:
                        n_writes += 1
                        if branch_log is not None:
                            branch_log.append(delta)
            except Exception:  # noqa: BLE001 — a failing background op must not kill the branch
                continue
    return n_writes


def _logistic_integrate(x0: float, rate: float, ceil: float, dt_days: float,
                        tol: float = 1e-9) -> tuple:
    """Adaptive Heun integration of dx/dt = r x (1 - x/K), refined until step-halving changes
    the answer < tol. Returns (x_end, error_bound). Internal only."""
    def _step(x, h):
        k1 = rate * x * (1.0 - x / max(1e-9, ceil))
        xp = x + h * k1
        k2 = rate * xp * (1.0 - xp / max(1e-9, ceil))
        return x + h * 0.5 * (k1 + k2)

    n = 4
    prev = None
    for _ in range(12):
        x, h = x0, dt_days / n
        for _i in range(n):
            x = _step(x, h)
        if prev is not None and abs(x - prev) < tol:
            return x, abs(x - prev)
        prev, n = x, n * 2
    return prev, abs(prev - x0) * 1e-3


def _condition_active(world, condition: str) -> bool:
    """Minimal condition evaluation for continuous processes: `quantity_name` truthy /
    `quantity_name>0.5` style. Unknown conditions default ACTIVE (recorded upstream)."""
    c = str(condition).strip()
    if not c:
        return True
    for op in (">=", "<=", ">", "<"):
        if op in c:
            name, _, rhs = c.partition(op)
            q = world.quantities.get(name.strip())
            v = getattr(q, "value", None)
            try:
                rv = float(rhs)
            except ValueError:
                return True
            if not isinstance(v, (int, float)):
                return False
            return {" >= ": v >= rv, "<=": v <= rv, ">": v > rv, "<": v < rv}.get(op, True) \
                if op != ">=" else v >= rv
    q = world.quantities.get(c)
    return bool(getattr(q, "value", None))


# ---------------------------------------------------------------- §16: hazard re-projection
def reproject_hazards(world, queue, written_paths: set, *, stats: TemporalRunStats = None):
    """After a batch: any first-passage process whose declared read fields were written gets
    accumulated-to-now under the OLD rate, its modulation recomputed, and its crossing
    re-projected — threshold and accumulated hazard preserved (invariants 26/27)."""
    store = getattr(world, "temporal_hazards", None)
    if not store or not written_paths:
        return 0
    n = 0
    for pid, st in store.items():
        if st.fired or not st.reads:
            continue
        touched = any(any(r in p for p in written_paths) for r in st.reads)
        if not touched:
            continue
        from swm.world_model_v2.temporal_hazards import resolve_modulation
        new_mod = resolve_modulation(world, st)
        if abs(new_mod - st.modulation) < 1e-9:
            st.accumulate_to(world.clock.now)
            continue
        st.on_state_change(world.clock.now, new_mod)
        ev = schedule_crossing(queue, world, st,
                               etype=str(st.payload.get("etype", "first_passage")),
                               participants=st.payload.get("participants") or ())
        n += 1
        if stats is not None:
            stats.events_rescheduled += 1
            if ev is None:
                stats.count("first_passage_beyond_horizon")
    return n


# ---------------------------------------------------------------- §13 → §NAP: stance dynamics
def emit_stance_relevant_changes(world, queue, written_paths: set,
                                 *, stats: TemporalRunStats = None) -> int:
    """§NAP: the numeric stance-change monitor is GONE. It watched 0-1 progress bars and a
    unit-less capacity resource against invented thresholds (0.30/0.70/±0.08) and fed a rule
    table that moved stance labels mechanically. Neither the bars nor the rule table exist in
    production. Stances now change the way real stances change: the actor's OWN situated
    cognition, invoked at its real decision triggers (temporal_compiler), observes the concrete
    typed state the batch wrote and rewrites its own stance record. This stub returns 0 and
    exists so the batch loop keeps a single, documented seam."""
    return 0


# ---------------------------------------------------------------- §11: conditional deferrals
def register_conditional(world, *, condition: dict, event_spec: dict, actor_id: str = "",
                         provenance: str = "actor_deferral"):
    """An actor deferred to a CONDITION (once legal replies / when the other party follows up /
    after the board meeting). Stored as a watched conditional — resolved when the condition's
    event occurs, NEVER converted to an invented duration (§11)."""
    conds = getattr(world, "temporal_conditionals", None)
    if conds is None:
        conds = []
        world.temporal_conditionals = conds
    conds.append({"condition": dict(condition), "event_spec": dict(event_spec),
                  "actor_id": str(actor_id), "provenance": str(provenance),
                  "registered_at": world.clock.now, "fired": False})


def check_conditionals(world, queue, batch_events, *, stats: TemporalRunStats = None) -> int:
    """After a batch: fire any registered conditional whose condition matches an event in the
    batch (etype match + optional participant match)."""
    conds = getattr(world, "temporal_conditionals", None)
    if not conds:
        return 0
    n = 0
    for c in conds:
        if c.get("fired"):
            continue
        cond = c.get("condition") or {}
        want_etype = str(cond.get("etype", ""))
        want_part = str(cond.get("participant", ""))
        for ev in batch_events:
            if want_etype and ev.etype != want_etype:
                continue
            if want_part and want_part not in [str(p) for p in ev.participants]:
                continue
            spec = dict(c.get("event_spec") or {})
            spec.setdefault("ts", world.clock.now)
            try:
                queue.schedule(Event(
                    ts=max(float(spec.get("ts", world.clock.now)), world.clock.now),
                    etype=str(spec.get("etype", "conditional_trigger")),
                    participants=list(spec.get("participants") or [c.get("actor_id", "")]),
                    payload={**(spec.get("payload") or {}),
                             "condition_met": {"etype": want_etype, "by_event": ev.event_id},
                             "provenance": c.get("provenance")},
                    parent_ids=[ev.event_id],
                    trigger=make_trigger(trigger_type="condition_became_true",
                                         actor_id=str(c.get("actor_id", "")),
                                         parents=[ev.event_id],
                                         observed=f"condition {want_etype} occurred",
                                         why_now="the awaited condition occurred",
                                         provenance=c.get("provenance", "actor_deferral"))))
                c["fired"] = True
                n += 1
                if stats is not None:
                    stats.count("conditional_trigger_fired")
            except Exception:  # noqa: BLE001
                continue
            break
    return n


def register_state_watch(world, *, match_substrings, event_spec: dict, max_fires: int = 24,
                         provenance: str = "condition_watch",
                         stats: TemporalRunStats = None):
    """Subscribe an event to STATE CHANGES (§14 — replace polling with change triggers): when
    a batch writes any path containing one of `match_substrings`, the event fires at that
    timestamp (next microstep). `max_fires` bounds re-fires; exhaustion leaves the watch
    dormant (recorded), never a poll loop."""
    watches = getattr(world, "temporal_state_watches", None)
    if watches is None:
        watches = []
        world.temporal_state_watches = watches
    watches.append({"match": tuple(str(m) for m in match_substrings if m),
                    "event_spec": dict(event_spec), "max_fires": int(max_fires),
                    "fires": 0, "provenance": str(provenance),
                    "registered_at": world.clock.now})
    if stats is not None:
        stats.count("state_watch_registered")


def check_state_watches(world, queue, written_paths: set,
                        *, stats: TemporalRunStats = None) -> int:
    watches = getattr(world, "temporal_state_watches", None)
    if not watches or not written_paths:
        return 0
    n = 0
    for wch in watches:
        if wch["fires"] >= wch["max_fires"]:
            continue
        if not any(m in p for p in written_paths for m in wch["match"]):
            continue
        wch["fires"] += 1
        spec = wch["event_spec"]
        try:
            queue.schedule(Event(
                ts=max(float(spec.get("ts", world.clock.now)), world.clock.now),
                etype=str(spec.get("etype", "conditional_trigger")),
                participants=list(spec.get("participants") or []),
                payload=dict(spec.get("payload") or {}),
                source=f"endogenous:{wch['provenance']}"))
            n += 1
            if stats is not None:
                stats.count("state_watch_fired")
        except Exception:  # noqa: BLE001
            continue
    return n


# ---------------------------------------------------------------- §19: the batch engine
def _effective_reads(ev) -> set:
    reads = set(str(p) for p in (ev.read_set or ()))
    for m in (ev.payload or {}).get("consume", []) or []:
        if isinstance(m, dict) and m.get("var"):
            reads.add(f"quantities[{m['var']}]")
    return reads


def process_timestamp(world, queue, operators, rng, branch_log, *,
                      stats: TemporalRunStats, rng_for=None, horizon_ts=None) -> int:
    """Process EVERYTHING at one timestamp: the initial batch plus every same-time causal
    descendant, in microsteps. Within a microstep events run in canonical content order
    (insertion-order invariant); an event whose declared reads intersect paths already written
    THIS microstep is deferred one microstep (same-time causal ordering); two events writing
    the same path in one microstep is an EXPLICIT simultaneity conflict — resolved by a
    scenario simultaneity rule when one names the mechanism, else recorded loudly as
    unmodeled (canonical order applies, never silent queue order). Returns events processed."""
    batch = queue.pop_batch(rng=rng, world=world)
    if not batch:
        return 0
    now = batch[0].ts
    stats.same_time_batches += 1
    stats.max_batch_size = max(stats.max_batch_size, len(batch))
    model = temporal_model_of(world)
    microstep, n_processed = 0, 0
    written_this_ts: set = set()
    batch_all: list = []
    current = list(batch)
    while current:
        # ---- layer by explicit causal dependency: an event whose parents are in THIS layer
        #      moves to the next microstep (invariant 33)
        layer_ids = {e.event_id for e in current}
        ready, deferred = [], []
        for e in current:
            dep = set(map(str, (e.parent_ids or []) + (e.dependency_ids or [])))
            (deferred if dep & layer_ids else ready).append(e)
        if not ready:                                          # dependency cycle — break loudly
            ready, deferred = current, []
            stats.simultaneity_conflicts.append(
                {"at": now, "kind": "dependency_cycle",
                 "events": [e.etype for e in current[:6]]})
        ready.sort(key=lambda e: e.content_key())              # canonical, insertion-invariant
        written_this_step: dict = {}                           # path -> event_id
        for ev in ready:
            # stale first-passage crossings (superseded by re-projection) are canceled
            if ev.payload.get("hazard_process_id") and not crossing_is_current(world, ev):
                stats.events_canceled += 1
                continue
            reads = _effective_reads(ev)
            if reads and any(r in written_this_step for r in reads):
                ev.microstep = microstep + 1
                deferred.append(ev)                            # same-time causal descendant
                continue
            ev.microstep = microstep
            stats.count(ev.etype)
            n_processed += 1
            batch_all.append(ev)
            for op in operators:
                try:
                    applicable = op.applicable(world, ev)
                except Exception:  # noqa: BLE001
                    continue
                if not applicable:
                    continue
                op_rng = rng_for(op, ev) if rng_for is not None else rng
                delta, vr = op.run(world, ev, op_rng)
                if delta is None:
                    if vr is not None and not vr.ok:
                        branch_log.append(_rejection(world, ev, op, vr))
                    continue
                # provenance stamp: lets the CRN pairing check separate exogenous shocks
                # (hazard-sourced) from action-caused activity across matched arms
                delta.uncertainty.setdefault("event_source", ev.source)
                branch_log.append(delta)
                for ch in delta.changes:
                    path = str(ch.get("path", ""))
                    if not path:
                        continue
                    prior = written_this_step.get(path)
                    if prior is not None and prior != ev.event_id:
                        _record_conflict(world, model, stats, now=now, path=path,
                                         first=prior, second=ev)
                    written_this_step[path] = ev.event_id
                    written_this_ts.add(path)
                for fu in delta.follow_up_events:
                    _schedule_follow_up(world, queue, ev, op, fu, branch_log, stats)
        microstep += 1
        # same-ts events scheduled during this microstep (descendants) join the next one
        current = list(deferred)
        while queue.events and queue.events[0].ts <= now:
            nxt = queue.next_event(rng=rng, world=world)
            if nxt is None:
                break
            current.append(nxt)
        if microstep > 64:                                      # runaway same-time loop guard
            stats.simultaneity_conflicts.append({"at": now, "kind": "microstep_overflow",
                                                 "deferred": len(current)})
            for e in current:
                e.ts = now + 1.0
                queue.schedule(e)
            break
    stats.max_microsteps = max(stats.max_microsteps, microstep)
    # ---- post-timestamp monitors: hazards re-project, stances react, watches + conditionals fire ----
    reproject_hazards(world, queue, written_this_ts, stats=stats)
    emit_stance_relevant_changes(world, queue, written_this_ts, stats=stats)
    check_state_watches(world, queue, written_this_ts, stats=stats)
    check_conditionals(world, queue, batch_all, stats=stats)
    # a monitor may have scheduled same-ts events (stance_relevant_change) — drain them now
    if queue.events and queue.events[0].ts <= now:
        n_processed += process_timestamp(world, queue, operators, rng, branch_log,
                                         stats=stats, rng_for=rng_for, horizon_ts=horizon_ts)
    return n_processed


def _record_conflict(world, model, stats, *, now, path, first, second):
    rule = None
    for r in (model.simultaneity_rules if model is not None else []) or []:
        if str(r.get("relation")) == "conflict":
            rule = r
            break
    rec = {"at": now, "path": path, "first_writer": first, "second_writer": second.event_id,
           "second_etype": second.etype,
           "resolution": (f"scenario_rule:{str(rule.get('mechanism'))[:80]}" if rule
                          else "unmodeled_simultaneity_conflict:canonical_order_applied_loudly")}
    stats.simultaneity_conflicts.append(rec)


def _schedule_follow_up(world, queue, parent_ev, op, fu, branch_log, stats):
    try:
        fev = Event(ts=max(float(fu.get("ts", world.clock.now)), world.clock.now),
                    etype=str(fu["etype"]),
                    participants=list(fu.get("participants") or []),
                    payload=dict(fu.get("payload") or {}),
                    source=f"endogenous:{getattr(op, 'name', 'op')}",
                    parent_ids=[parent_ev.event_id] + list(fu.get("parent_ids") or []),
                    trigger=dict(fu.get("trigger") or {}))
    except (KeyError, TypeError, ValueError) as e:
        from swm.world_model_v2.transitions import StateDelta
        branch_log.append(StateDelta(
            at=world.clock.now, event_type=parent_ev.etype, operator=getattr(op, "name", "op"),
            reason_codes=["action_rejected", f"invalid follow-up event: {e}"[:120]]))
        return
    queue.schedule(fev)


def _rejection(world, ev, op, vr):
    from swm.world_model_v2.transitions import StateDelta
    return StateDelta(at=world.clock.now, event_type=ev.etype, operator=getattr(op, "name", "op"),
                      reason_codes=["action_rejected"] + list(vr.reasons)[:3])


# ---------------------------------------------------------------- the branch loop
def run_branch_temporal(world, queue, operators, *, seed: int = 0,
                        safety_max_events: int = 2000, rng_for=None, branch=None,
                        queue_rng=None):
    """One world, event by event, to the real horizon. Stopping conditions are REAL: queue
    quiescence or horizon. The safety budget protects the service; exhausting it marks the
    branch simulation_status=temporally_truncated with pending events recorded — never a fake
    natural completion (§12). `rng_for(op, ev)` and `queue_rng` let the matched-counterfactual
    engine route randomness through named streams (CRN)."""
    from swm.world_model_v2.state import WorldBranch
    rng = queue_rng if queue_rng is not None else random.Random(seed)
    if branch is None:
        branch = WorldBranch(branch_id=world.branch_id, world=world)
    stats = get_stats(world)
    stats.safety_limits = {"safety_max_events": safety_max_events}
    model = temporal_model_of(world)
    if model is not None:
        sample_temporal_latents(world, model)
    n = 0
    while True:
        if n >= safety_max_events:
            stats.temporally_truncated = True
            stats.truncation = {
                "reason": "safety_max_events_reached", "limit": safety_max_events,
                "at_ts": world.clock.now,
                "pending_events": queue.peek_pending(30),
                "actors_not_processed": sorted({str(p) for e in queue.events
                                                for p in (e.participants or [])
                                                if e.ts <= queue.horizon_ts})[:20],
                "note": "additional compute would process the pending causal chains; results "
                        "from this branch are temporally truncated, not naturally quiescent"}
            break
        if not queue.events or queue.empty():
            break                                              # causal quiescence
        next_ts = min(e.ts for e in queue.events)
        if next_ts > queue.horizon_ts:
            break
        if next_ts > world.clock.now:
            advance_interval(world, world.clock.now, next_ts, rng=rng,
                             operators=operators, branch_log=branch.log, stats=stats)
            world.clock.advance_to(next_ts)
        n += process_timestamp(world, queue, operators, rng, branch.log,
                               stats=stats, rng_for=rng_for, horizon_ts=queue.horizon_ts)
        if getattr(stats, "branch_halted", False):
            # §20: an actor decision could not execute (budget/provider/parse/cognition) — the
            # branch STOPS at this exact timestamp with full world state, pending events and
            # the unresolved decision trigger preserved. No substitute action; no further
            # advancement; never presented as having reached the horizon.
            stats.temporally_truncated = True
            trunc = stats.truncation if isinstance(stats.truncation, dict) else {}
            trunc.setdefault("reason", getattr(stats, "branch_status", "truncated_actor_budget"))
            trunc["branch_status"] = getattr(stats, "branch_status", "truncated_actor_budget")
            trunc["at_ts"] = world.clock.now
            trunc["pending_events"] = queue.peek_pending(30)
            trunc["halted"] = True
            stats.truncation = trunc
            break
    # horizon reporting: pending in-horizon events that never ran (§27)
    stats.pending_at_horizon = queue.peek_pending(30)
    branch.terminal = True
    branch.temporal_stats = stats
    return branch


def aggregate_temporal_stats(branches) -> dict:
    """Cross-particle §27 temporal-runtime block: event counts, invocations by actor and
    trigger, delay quantiles, batching, conflicts, truncations, pending events, unresolved
    timing. Attached to every run result."""
    agg = TemporalRunStats()
    n_truncated, models = 0, set()
    for b in branches:
        st = getattr(b, "temporal_stats", None) or getattr(getattr(b, "world", None),
                                                           "temporal_stats", None)
        if st is None:
            continue
        for k, v in st.event_counts.items():
            agg.event_counts[k] = agg.event_counts.get(k, 0) + v
        for a, trigs in st.actor_invocations.items():
            slot = agg.actor_invocations.setdefault(a, {})
            for t, v in trigs.items():
                slot[t] = slot.get(t, 0) + v
        if len(agg.decision_triggers) < 200:
            agg.decision_triggers.extend(st.decision_triggers[:10])
        agg.delivery_to_attention_s.extend(st.delivery_to_attention_s[:200])
        agg.attention_to_decision_s.extend(st.attention_to_decision_s[:200])
        agg.decision_to_action_s.extend(st.decision_to_action_s[:200])
        agg.action_to_completion_s.extend(st.action_to_completion_s[:200])
        agg.same_time_batches += st.same_time_batches
        agg.max_batch_size = max(agg.max_batch_size, st.max_batch_size)
        agg.max_microsteps = max(agg.max_microsteps, st.max_microsteps)
        agg.simultaneity_conflicts.extend(st.simultaneity_conflicts[:5])
        agg.events_canceled += st.events_canceled
        agg.events_rescheduled += st.events_rescheduled
        agg.interval_advances += st.interval_advances
        agg.attention_batches.extend(st.attention_batches[:20])
        if st.temporally_truncated:
            n_truncated += 1
            if not agg.truncation:
                agg.truncation = dict(st.truncation)
            # §21: every truncated branch stays individually visible in aggregation
            agg.truncation.setdefault("branches", []).append({
                "branch_id": str(getattr(b, "branch_id",
                                         getattr(getattr(b, "world", None), "branch_id", ""))),
                "branch_status": (st.truncation or {}).get("branch_status",
                                                           st.branch_status or
                                                           "truncated_event_budget"),
                "reason": (st.truncation or {}).get("reason", ""),
                "at_ts": (st.truncation or {}).get("at_ts"),
                "actors": list((st.truncation or {}).get("actors_not_processed", [])),
                "unresolved_decision_trigger":
                    (st.truncation or {}).get("unresolved_decision_trigger", {}),
            })
        if len(agg.pending_at_horizon) < 30:
            agg.pending_at_horizon.extend(st.pending_at_horizon[:5])
        agg.unresolved_timing.extend(st.unresolved_timing[:5])
        agg.mechanism_suppressions.extend(
            getattr(st, "mechanism_suppressions", [])[:5])
        if st.safety_limits:
            agg.safety_limits.update(st.safety_limits)
        mw = getattr(getattr(b, "world", None), "temporal_model", None)
        if mw is not None:
            models.add(mw.temporal_model_hash())
    agg.temporally_truncated = n_truncated > 0
    out = agg.as_dict()
    out["n_branches"] = len(branches)
    out["n_branches_truncated"] = n_truncated
    out["truncated_branch_share"] = round(n_truncated / max(1, len(branches)), 4)
    out["temporal_model_hashes"] = sorted(models)
    return out
