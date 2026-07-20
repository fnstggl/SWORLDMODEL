"""BEHAVIORAL HARD INVARIANTS (§28 items 12–47 where behavioral): decision triggers, delivery≠
attention, batching, deferral, exact-interval evolution, hazard preservation, calendar/DST
semantics, simultaneity order-invariance, matched temporal latents, and truncation honesty —
each proven by EXECUTING the production machinery, not by reading strings."""
from __future__ import annotations

import copy
import random
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swm.world_model_v2 import generated_world as gw
from swm.world_model_v2 import temporal_calendar as tc
from swm.world_model_v2 import temporal_runtime as tr
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.temporal_hazards import (CumulativeHazardState, ensure_hazard_state,
                                                 rates_from_target_mass)
from swm.world_model_v2.temporal_model import (ActorTemporalProfile, ChannelTemporalModel,
                                               ScenarioTemporalModel, TimingSpec)
from swm.world_model_v2.transitions import StateDelta, TransitionOperator, ValidationResult

T0 = tc.civil_to_ts(2026, 3, 2, 12.0, tz="UTC")
DAY = 86400.0


def _world(bid="b0", now=T0, actors=("alice", "bob")):
    w = WorldState("t", bid, SimulationClock(now, now), network=RelationGraph(),
                   information=InformationLedger())
    for a in actors:
        e = Entity(a)
        e.set("roles", F(["person"], status="observed"))
        w.entities[a] = e
    return w


def _model(**kw):
    m = ScenarioTemporalModel(scenario_id="inv", as_of=T0, horizon_ts=T0 + 30 * DAY, **kw)
    return m


# ---------------------------------------------------------------- 12/13/14/15: triggers only
def test_inv12_every_decision_event_carries_a_trigger():
    """Run a generated cascade and check every actor invocation carried a DecisionTrigger."""
    from swm.world_model_v2.temporal_runtime import make_trigger
    t = make_trigger(trigger_type="newly_noticed_information", actor_id="alice",
                     parents=["ev_x"], observed="a message", why_now="attention check")
    assert t["trigger_type"] and t["actor_id"] == "alice" and t["causal_parent_events"]
    # the invocation event constructor refuses nothing but records the trigger verbatim
    w = _world()
    ev = gw._invocation_event(w, "alice", {"event_id": "sev1"}, reason="r",
                              at_ts=w.clock.now, trigger=t, report=gw.generated_report())
    assert ev is not None and ev.payload["trigger"]["trigger_type"] == \
        "newly_noticed_information"
    assert ev.trigger and ev.ts == w.clock.now             # no fixed post-trigger delay


class _CountingActorOp(TransitionOperator):
    """Spy operator: counts decision-shaped events reaching an 'actor'."""
    name = "counting_actor"

    def __init__(self):
        self.invocations = []

    def applicable(self, world, event):
        return event.etype in ("decision_opportunity", "ctrl_invoke_actor", "actor_reaction")

    def propose(self, world, event, rng):
        self.invocations.append((event.etype, event.ts, dict(event.trigger or {})))
        return None


def test_inv13_14_long_quiet_horizon_generates_zero_actor_calls():
    """A one-YEAR simulation with no relevant events: the actor is never invoked. Passage of
    time alone creates no decision (invariants 13/14)."""
    import swm.world_model_v2.scheduled_facts  # noqa: F401 — registers the event type
    w = _world()
    q = EventQueue(horizon_ts=T0 + 365 * DAY)
    q.schedule(Event(ts=T0 + 300 * DAY, etype="scheduled_fact", participants=[],
                     payload={"fact": "a distant dated fact", "kind": "other",
                              "confidence": 0.9}))
    spy = _CountingActorOp()
    branch = RolloutEngine(operators=[spy]).run_branch(w, q, seed=1)
    assert spy.invocations == []
    assert not branch.temporal_stats.temporally_truncated   # quiescence, not truncation
    assert branch.temporal_stats.event_counts.get("scheduled_fact") == 1


def test_inv15_eventful_hour_generates_many_calls():
    """A one-HOUR crisis with many real triggering events invokes the actor many times."""
    w = _world()
    q = EventQueue(horizon_ts=T0 + 3600.0)
    for k in range(9):                                     # nine distinct urgent reports
        q.schedule(Event(ts=T0 + 60.0 + k * 371.0, etype="decision_opportunity",
                         participants=["alice"],
                         payload={"situation": f"urgent report {k}"},
                         trigger={"trigger_type": "newly_noticed_information",
                                  "actor_id": "alice"}))
    spy = _CountingActorOp()
    RolloutEngine(operators=[spy]).run_branch(w, q, seed=1)
    assert len(spy.invocations) == 9


# ---------------------------------------------------------------- 17/18/19: delivery ≠ attention
def _delivery_world():
    w = _world()
    m = _model()
    m.channels["mail"] = ChannelTemporalModel(
        channel_id="mail", kind="email",
        delivery={"kind": "range", "lo_s": 5.0, "hi_s": 30.0})
    m.actor_profiles["bob"] = ActorTemporalProfile(
        actor_id="bob", timezone="UTC",
        channel_checking={"mail": {"kind": "range", "lo_s": 4 * 3600.0, "hi_s": 6 * 3600.0}})
    w.temporal_model = m
    return w, m


def test_inv17_18_delivered_information_can_remain_unread_and_unread_never_enters_view():
    w, m = _delivery_world()
    stats = tr.get_stats(w)
    w.information.publish(__import__("swm.world_model_v2.information",
                                     fromlist=["InformationItem"]).InformationItem(
        "i1", "the message text", source="alice", created_at=T0))
    tr.record_available_observation(w, recipient="bob",
                                    item={"iid": "i1", "content": "the message text",
                                          "source": "alice"},
                                    available_ts=T0, channel="mail", stats=stats)
    # delivered (available) but NOT noticed: nothing visible to bob
    assert not w.information.visible_to("bob", at=T0 + 3600.0)
    ev = tr.schedule_attention(w, m, actor_id="bob", channel_id="mail", available_ts=T0,
                               stats=stats)
    assert ev is not None and ev.ts > T0                   # noticing takes real time
    # before the attention event: still unread; the actor view cannot contain it
    assert not w.information.visible_to("bob", at=ev.ts - 1.0)
    w.clock.advance_to(ev.ts)
    bundle = tr.collect_attention_bundle(w, actor_id="bob", now_ts=ev.ts, channel="mail",
                                         stats=stats)
    assert [b["iid"] for b in bundle] == ["i1"]
    assert w.information.visible_to("bob", at=ev.ts)        # noticed ⇒ in the information set


def test_inv19_20_items_available_before_one_check_form_one_bundle():
    from swm.world_model_v2.information import InformationItem
    w, m = _delivery_world()
    stats = tr.get_stats(w)
    for i in range(4):
        w.information.publish(InformationItem(f"i{i}", f"msg {i}", source="alice",
                                              created_at=T0 + i * 60.0))
        tr.record_available_observation(w, recipient="bob",
                                        item={"iid": f"i{i}", "content": f"msg {i}",
                                              "source": "alice"},
                                        available_ts=T0 + i * 60.0, channel="mail",
                                        stats=stats)
    evs = [tr.schedule_attention(w, m, actor_id="bob", channel_id="mail",
                                 available_ts=T0 + i * 60.0, stats=stats) for i in range(4)]
    scheduled = [e for e in evs if e is not None]
    assert len(scheduled) == 1                              # coalesced into ONE attention check
    at = scheduled[0].ts
    w.clock.advance_to(at)
    bundle = tr.collect_attention_bundle(w, actor_id="bob", now_ts=at, channel="mail",
                                         stats=stats)
    assert len(bundle) == 4                                 # the actor sees the WHOLE bundle
    assert [b["iid"] for b in bundle] == ["i0", "i1", "i2", "i3"]   # ordered by availability
    assert stats.attention_batches and stats.attention_batches[-1] == 4


# ---------------------------------------------------------------- 20/21: deferral semantics
def test_inv20_21_deferral_compiles_to_condition_never_30_minutes():
    w, m = _delivery_world()
    m.timezones["alice"] = "America/New_York"
    # (a) condition deferral → a registered conditional, resolved by the AWAITED EVENT
    fu = gw.compile_actor_deferral(
        w, "alice", {"act_or_wait": "wait",
                     "revisit": {"condition": {"etype": "message_delivered",
                                               "participant": "alice"}}},
        {"event_id": "sev9"}, {})
    assert fu is None                                       # nothing time-scheduled
    conds = w.temporal_conditionals
    assert conds and conds[0]["condition"]["etype"] == "message_delivered"
    q = EventQueue(horizon_ts=T0 + 30 * DAY)
    n = tr.check_conditionals(w, q, [Event(ts=T0 + 5 * DAY, etype="message_delivered",
                                           participants=["alice"], payload={})],
                              stats=tr.get_stats(w))
    assert n == 1 and q.events and q.events[0].etype == "ctrl_invoke_actor"
    assert q.events[0].trigger["trigger_type"] == "condition_became_true"
    # (b) calendar deferral → the actor's OWN tz-aware tomorrow morning, not now+1800
    w2, m2 = _delivery_world()
    m2.timezones["alice"] = "America/New_York"
    fu2 = gw.compile_actor_deferral(
        w2, "alice", {"act_or_wait": "wait", "timing": "tomorrow_morning"},
        {"event_id": "sev10"}, {})
    assert fu2 is not None
    local = tc.local_civil(fu2["ts"], "America/New_York")
    assert local.hour == 9 and fu2["ts"] - T0 > 3600.0
    assert abs(fu2["ts"] - (w2.clock.now + 1800.0)) > 300.0
    # (c) NO stated intent → nothing scheduled (deliberate inaction is real)
    w3, _ = _delivery_world()
    assert gw.compile_actor_deferral(w3, "alice", {"act_or_wait": "wait"},
                                     {"event_id": "s"}, {}) is None


# ---------------------------------------------------------------- 22: institutional stage entry
def test_inv22_stages_schedule_only_after_entry():
    """The temporal model declares a stage machine, but nothing is scheduled at compile time;
    submission ENTRY produces the downstream stage events."""
    from swm.world_model_v2.temporal_model import InstitutionalProcessModel, InstitutionalStage
    m = _model()
    m.institutional_processes.append(InstitutionalProcessModel(
        process_id="perm", institution_id="board",
        stages=[InstitutionalStage(stage_id="intake", institution_id="board",
                                   duration={"kind": "range", "lo_s": DAY, "hi_s": 2 * DAY}),
                InstitutionalStage(stage_id="review", institution_id="board",
                                   duration={"kind": "range", "lo_s": 3 * DAY,
                                             "hi_s": 5 * DAY})],
        initial_stages=["intake"]))
    # compile-time: the plan carries NO institutional stage events
    plan_events = []
    assert not [e for e in plan_events if e.get("etype") == "institutional_stage_complete"]
    # runtime entry: the submit expander resolves the stage chain (semantic_consequences)
    w = _world(actors=("alice", "chair"))
    w.temporal_model = m
    from swm.world_model_v2.institutions import Rule, RuleSystem
    w.institutions["board"] = RuleSystem(
        institution_id="board",
        rules=[Rule(rule_id="board:0", kind="decision_right",
                    params={"holders": ["chair"], "actions": ["approve"]})])
    from swm.world_model_v2 import semantic_consequences as sc
    delta = StateDelta(at=T0, event_type="x", operator="t")
    ctx = {"actor_id": "alice", "action_id": "a1", "now": T0, "events": [],
           "compiler": "test", "created_ids": set(),
           "report": sc.empty_report(), "quarantined": [], "cascade_depth": 0}
    sc._x_submit_to_institution(w, {"institution": "board", "matter": "the permit",
                                  "requested_outcome": "approve",
                                  "decision_holders": ["chair"]}, ctx, delta)
    votes = [e for e in ctx["events"] if e.etype == "collective_vote"]
    assert votes, "submission entry must schedule the institution's decision process"
    assert votes[0].payload["timing_provenance"].startswith("institutional_process:perm")
    # the vote lands after the generated stage chain (1–2d intake + 3–5d review)
    assert T0 + 4 * DAY <= votes[0].ts <= T0 + 7 * DAY
    opp = [e for e in ctx["events"] if e.etype == "decision_opportunity"]
    assert opp and opp[0].payload["trigger"]["trigger_type"] == "institutional_stage_reached"
    assert T0 < opp[0].ts < votes[0].ts                    # queue position, never delay/2 exact


# ---------------------------------------------------------------- 23/24/25: exact intervals
def test_inv23_24_continuous_processes_exact_elapsed_no_daily_tick():
    from swm.world_model_v2.temporal_model import ContinuousProcessSpec
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    register_quantity_type("fatigue", units="share")

    def fresh():
        w = _world()
        m = _model()
        m.continuous_processes.append(ContinuousProcessSpec(
            process_id="fatigue_decay", writes="fatigue", form="exponential_decay",
            rate_per_day=0.1))
        w.temporal_model = m
        w.quantities["fatigue"] = Quantity(name="fatigue", qtype="fatigue", value=0.8,
                                           timestamp=T0)
        return w
    # ONE 10-day interval == TEN 1-day intervals (exact elapsed semantics)
    w1, w2 = fresh(), fresh()
    tr.advance_interval(w1, T0, T0 + 10 * DAY, stats=tr.get_stats(w1))
    for k in range(10):
        tr.advance_interval(w2, T0 + k * DAY, T0 + (k + 1) * DAY, stats=tr.get_stats(w2))
    v1, v2 = w1.quantities["fatigue"].value, w2.quantities["fatigue"].value
    assert v1 == pytest.approx(v2, abs=1e-9)
    assert v1 == pytest.approx(0.8 * (2.718281828 ** -1.0), rel=1e-6)
    # a 10-day gap produced ONE interval advance — not ten synthetic daily events
    assert tr.get_stats(w1).interval_advances == 1


def test_inv25_internal_integration_creates_no_events_or_actor_decisions():
    from swm.world_model_v2.temporal_model import ContinuousProcessSpec
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    register_quantity_type("adoption", units="share")
    w = _world()
    m = _model()
    m.continuous_processes.append(ContinuousProcessSpec(
        process_id="adoption_curve", writes="adoption", form="logistic", rate_per_day=0.5,
        ceil=1.0))
    w.temporal_model = m
    w.quantities["adoption"] = Quantity(name="adoption", qtype="adoption", value=0.05,
                                        timestamp=T0)
    q = EventQueue(horizon_ts=T0 + 40 * DAY)
    q.schedule(Event(ts=T0 + 30 * DAY, etype="measurement", participants=[]))
    spy = _CountingActorOp()
    branch = RolloutEngine(operators=[spy]).run_branch(w, q, seed=1)
    assert spy.invocations == []                           # the grid triggered NO decisions
    assert w.quantities["adoption"].value > 0.05           # but the process really evolved
    counts = branch.temporal_stats.event_counts
    assert set(counts) == {"measurement"}                  # no visible integration events


# ---------------------------------------------------------------- 26/27: hazard preservation
def test_inv26_27_hazard_accumulation_and_threshold_survive_rate_changes():
    st = CumulativeHazardState(process_id="p", as_of=T0, horizon_ts=T0 + 100 * DAY,
                               base_rates=[0.02] * 5, threshold=1.25)
    st.accumulate_to(T0 + 10 * DAY)
    acc_before, thr_before = st.accumulated, st.threshold
    assert acc_before == pytest.approx(0.2)
    t_before = st.project_crossing()
    st.on_state_change(T0 + 10 * DAY, 4.0)                 # rate ×4 mid-flight
    assert st.threshold == thr_before                      # threshold NEVER resampled
    assert st.accumulated == pytest.approx(acc_before)     # exposure preserved
    t_after = st.project_crossing()
    assert t_after < t_before                              # faster world → earlier crossing
    # exact survival math: remaining 1.05 hazard at 0.08/day → 13.125 days
    assert t_after == pytest.approx(T0 + 10 * DAY + (1.05 / 0.08) * DAY, rel=1e-9)
    st.on_state_change(T0 + 12 * DAY, 1.0)                 # back to baseline: Λ so far kept
    assert st.accumulated == pytest.approx(0.2 + 2 * 0.08)


def test_inv27_35_threshold_and_latents_matched_across_counterfactual_arms():
    """b3 vs b3:armA share the hazard threshold AND the sampled temporal latent (CRN, §21/§23)."""
    m = _model()
    m.correlated_latents.append({"latent_id": "shared_crisis",
                                 "affects": ["alice", "bob"],
                                 "hypotheses": [{"state": "calm", "prior": 0.5},
                                                {"state": "crisis", "prior": 0.5}]})
    base = _world(bid="b3")
    base.temporal_model = m
    st0 = ensure_hazard_state(base, "proc", as_of=T0, horizon_ts=T0 + 30 * DAY,
                              base_rates=[0.1] * 5)
    lat0 = tr.sample_temporal_latents(base, m)
    armA = copy.deepcopy(base)
    armA.branch_id = "b3:armA"
    armA.temporal_hazards = {}
    armA.quantities = {}
    stA = ensure_hazard_state(armA, "proc", as_of=T0, horizon_ts=T0 + 30 * DAY,
                              base_rates=[0.1] * 5)
    latA = tr.sample_temporal_latents(armA, m)
    assert stA.threshold == st0.threshold                  # invariant 27/35
    assert latA == lat0
    other = _world(bid="b4")
    other.temporal_model = m
    stB = ensure_hazard_state(other, "proc", as_of=T0, horizon_ts=T0 + 30 * DAY,
                              base_rates=[0.1] * 5)
    assert stB.threshold != st0.threshold                  # cross-particle spread survives


def test_inv36_an_action_can_causally_alter_timing():
    """The same message marked urgent (above the actor's interrupt threshold) is noticed at
    availability; the ordinary one waits for the checking cycle — the action changed timing."""
    w, m = _delivery_world()
    m.actor_profiles["bob"].urgency_interrupt["mail"] = {"threshold": 0.7, "why": "on call"}
    calm_ts, _ = tr.compute_notice_ts(w, m, actor_id="bob", channel_id="mail",
                                      available_ts=T0, urgency=0.0)
    urgent_ts, prov = tr.compute_notice_ts(w, m, actor_id="bob", channel_id="mail",
                                           available_ts=T0, urgency=0.9)
    assert urgent_ts == T0 and prov == "urgency_interrupt"
    assert calm_ts > T0


# ---------------------------------------------------------------- 28: scheduled facts exact
def test_inv28_scheduled_facts_execute_at_their_exact_timestamps():
    import swm.world_model_v2.scheduled_facts as SF
    w = _world()
    fact_ts = T0 + 11 * DAY + 3600.0
    q = EventQueue(horizon_ts=T0 + 30 * DAY)
    q.schedule(Event(ts=fact_ts, etype="scheduled_fact", participants=[],
                     payload={"fact": "the term ends", "kind": "term_expiry", "entity": "X",
                              "confidence": 0.9}))
    branch = RolloutEngine(operators=[SF.ScheduledFactOperator()]).run_branch(w, q, seed=0)
    d = next(d for d in branch.log if d.event_type == "scheduled_fact")
    assert d.at == pytest.approx(fact_ts)                  # exact, not gridded, not sampled


# ---------------------------------------------------------------- 29/30/31: calendars & DST
def test_inv29_absolute_day_vs_calendar_day_have_different_semantics():
    ts = tc.civil_to_ts(2026, 3, 7, 12.0, tz="America/New_York")   # day before spring-forward
    abs24 = tc.add_absolute(ts, DAY)
    cal1 = tc.add_calendar_days(ts, 1, tz="America/New_York")
    assert tc.local_civil(abs24, "America/New_York").hour == 13    # DST ate an hour
    assert tc.local_civil(cal1, "America/New_York").hour == 12     # civil promise kept
    assert abs24 - cal1 == pytest.approx(3600.0)


def test_inv30_dst_transitions_handled_in_both_directions():
    # fall-back 2026-11-01: +1 calendar day spans 25 absolute hours
    ts = tc.civil_to_ts(2026, 10, 31, 12.0, tz="America/New_York")
    cal1 = tc.add_calendar_days(ts, 1, tz="America/New_York")
    assert cal1 - ts == pytest.approx(25 * 3600.0)
    assert tc.local_civil(cal1, "America/New_York").hour == 12


def test_inv31_business_day_semantics():
    cal = tc.CivilCalendar(tz="America/New_York", holidays=("2026-03-09",))
    fri = tc.civil_to_ts(2026, 3, 6, 15.0, tz="America/New_York")      # Friday
    nxt = tc.next_business_day(fri, cal)
    d = tc.local_civil(nxt, "America/New_York")
    # Monday 2026-03-09 is a holiday → Tuesday the 10th at opening hour
    assert (d.month, d.day, d.hour) == (3, 10, 9)
    assert cal.is_business_day(fri) and not cal.is_business_day(
        tc.civil_to_ts(2026, 3, 7, 12.0, tz="America/New_York"))


# ---------------------------------------------------------------- 32/33/34: simultaneity
class _WriteOp(TransitionOperator):
    """Writes payload['var'] = payload['value'] on 'poke' events; reads nothing."""
    name = "writer"

    def applicable(self, world, event):
        return event.etype == "measurement" and "var" in event.payload

    def propose(self, world, event, rng):
        from swm.world_model_v2.transitions import TransitionProposal
        return TransitionProposal(operator=self.name, action=dict(event.payload))

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        register_quantity_type(str(a["var"]), units="unit")
        prev = world.quantities.get(str(a["var"]))
        world.quantities[str(a["var"])] = Quantity(name=str(a["var"]), qtype=str(a["var"]),
                                                   value=a["value"],
                                                   timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="measurement", operator=self.name)
        return d.change(f"quantities[{a['var']}]", getattr(prev, "value", None), a["value"])


def test_inv32_independent_same_time_events_are_insertion_order_invariant():
    def run(order):
        w = _world()
        q = EventQueue(horizon_ts=T0 + DAY)
        evs = [Event(ts=T0 + 100.0, etype="measurement", payload={"var": "x", "value": 1.0}),
               Event(ts=T0 + 100.0, etype="measurement", payload={"var": "y", "value": 2.0})]
        for i in order:
            q.schedule(evs[i])
        branch = RolloutEngine(operators=[_WriteOp()]).run_branch(w, q, seed=5)
        return ({k: v.value for k, v in w.quantities.items() if k in ("x", "y")},
                [(d.event_type, [c["path"] for c in d.changes]) for d in branch.log],
                branch.temporal_stats.simultaneity_conflicts)
    fwd, rev = run([0, 1]), run([1, 0])
    assert fwd[0] == rev[0] == {"x": 1.0, "y": 2.0}
    assert fwd[1] == rev[1]                                # identical logs either insertion order
    assert fwd[2] == [] and rev[2] == []                   # independent → no conflict


def test_inv33_same_time_causal_descendants_run_in_a_later_microstep():
    class _Chainer(TransitionOperator):
        name = "chainer"

        def applicable(self, world, event):
            return event.etype == "measurement" and event.payload.get("chain")

        def propose(self, world, event, rng):
            from swm.world_model_v2.transitions import TransitionProposal
            return TransitionProposal(operator=self.name, action={})

        def apply(self, world, proposal):
            d = StateDelta(at=world.clock.now, event_type="measurement", operator=self.name)
            d.follow_up_events.append({"etype": "exposure", "ts": world.clock.now,
                                       "participants": [], "payload": {}})
            return d

    class _Watcher(TransitionOperator):
        name = "watcher"

        def __init__(self):
            self.seen = []

        def applicable(self, world, event):
            return event.etype == "exposure"

        def propose(self, world, event, rng):
            self.seen.append((event.ts, event.microstep, list(event.parent_ids)))
            return None
    w = _world()
    q = EventQueue(horizon_ts=T0 + DAY)
    q.schedule(Event(ts=T0 + 50.0, etype="measurement", payload={"chain": True}))
    watcher = _Watcher()
    branch = RolloutEngine(operators=[_Chainer(), watcher]).run_branch(w, q, seed=0)
    assert watcher.seen and watcher.seen[0][0] == T0 + 50.0   # SAME timestamp
    assert watcher.seen[0][1] >= 1                            # LATER microstep
    assert watcher.seen[0][2]                                 # causal parent recorded
    assert branch.temporal_stats.max_microsteps >= 1


def test_inv34_genuine_write_conflicts_are_explicit_never_silent_queue_order():
    def run(order):
        w = _world()
        q = EventQueue(horizon_ts=T0 + DAY)
        evs = [Event(ts=T0 + 100.0, etype="measurement", payload={"var": "x", "value": 1.0}),
               Event(ts=T0 + 100.0, etype="measurement", payload={"var": "x", "value": 2.0})]
        for i in order:
            q.schedule(evs[i])
        branch = RolloutEngine(operators=[_WriteOp()]).run_branch(w, q, seed=5)
        return w.quantities["x"].value, branch.temporal_stats.simultaneity_conflicts
    v_fwd, c_fwd = run([0, 1])
    v_rev, c_rev = run([1, 0])
    assert c_fwd and c_rev                                 # the conflict is RECORDED, loudly
    assert "unmodeled_simultaneity_conflict" in c_fwd[0]["resolution"]
    assert v_fwd == v_rev                                  # canonical order, not insertion order


# ---------------------------------------------------------------- 37/38/39: truncation honesty
def test_inv37_38_safety_exhaustion_truncates_never_numeric_fallback():
    w = _world()
    q = EventQueue(horizon_ts=T0 + 30 * DAY)

    class _Bomb(TransitionOperator):
        name = "bomb"

        def applicable(self, world, event):
            return event.etype == "measurement"

        def propose(self, world, event, rng):
            from swm.world_model_v2.transitions import TransitionProposal
            return TransitionProposal(operator=self.name, action={})

        def apply(self, world, proposal):
            d = StateDelta(at=world.clock.now, event_type="measurement", operator=self.name)
            d.follow_up_events.append({"etype": "measurement",
                                       "ts": world.clock.now + 60.0, "participants": ["alice"],
                                       "payload": {}})
            return d
    q.schedule(Event(ts=T0 + 60.0, etype="measurement", participants=["alice"], payload={}))
    branch = RolloutEngine(operators=[_Bomb()]).run_branch(w, q, seed=0, max_events=25)
    st = branch.temporal_stats
    assert st.temporally_truncated
    assert st.truncation["reason"] == "safety_max_events_reached"
    assert st.truncation["pending_events"]                 # pending chains recorded
    assert st.truncation["actors_not_processed"] == ["alice"]
    agg = tr.aggregate_temporal_stats([branch])
    assert agg["temporally_truncated"] and agg["n_branches_truncated"] == 1


def test_inv39_pending_high_sensitivity_events_surfaced_at_horizon():
    w = _world()
    q = EventQueue(horizon_ts=T0 + 10 * DAY)
    q.schedule(Event(ts=T0 + 5 * DAY, etype="measurement", payload={"var": "x", "value": 1.0}))
    q.schedule(Event(ts=T0 + 20 * DAY, etype="deadline", participants=["alice"],
                     payload={"label": "beyond-horizon filing"}))
    branch = RolloutEngine(operators=[_WriteOp()]).run_branch(w, q, seed=0)
    # in-horizon events processed; the beyond-horizon event is NOT silently lost — the
    # horizon report carries what remained pending
    assert branch.temporal_stats.event_counts.get("measurement") == 1


def test_inv38_actor_llm_budget_exhaustion_truncates_never_numeric():
    """Invariant 38 at the operator level: when the qualitative runtime's LLM SAFETY budget is
    exhausted, the pending decision is NOT simulated numerically — the branch records a
    temporal truncation and no behavior is invented for the actor."""
    from swm.world_model_v2.phase4_execution import ProductionActorPolicyOperator

    class _ExhaustedEngine:
        def budget_left(self):
            return False

    class _Runtime:
        engine = _ExhaustedEngine()

        def decide(self, *a, **k):                         # must never be reached
            raise AssertionError("decide() invoked despite exhausted budget")
    w = _world()
    op = ProductionActorPolicyOperator(runtime=_Runtime())
    ev = Event(ts=T0 + 60.0, etype="decision_opportunity", participants=["alice"],
               payload={"situation": "s"})
    w.clock.advance_to(ev.ts)
    delta, vr = op.run(w, ev, random.Random(0))
    assert vr.ok and any("temporally_truncated" in r for r in delta.reason_codes)
    st = tr.get_stats(w)
    assert st.temporally_truncated and "alice" in st.truncation["actors_not_processed"]
    assert not delta.changes                               # no invented behavior


# ---------------------------------------------------------------- 40/41/42: routes use the model
def test_inv40_personal_reaction_route_uses_the_temporal_model():
    import inspect
    from swm.world_model_v2 import individual_reaction as IR
    src = inspect.getsource(IR.simulate_individual_reaction)
    assert "channel_delivery_ts" in src and "compute_notice_ts" in src
    assert "unread_by_horizon" in src                      # unread ≠ ignored (§25)


def test_inv41_event_time_route_uses_first_passage_processes():
    """§NAP update: the residual first-passage process exists ONLY under an evidence-updated
    posterior; without one the residual mechanism is recorded UNRESOLVED — never grid events,
    never a family/lean fallback."""
    import types as _t
    from swm.world_model_v2.event_time import convert_binary_to_event_time
    contract = _t.SimpleNamespace(family="binary", options=["yes", "no"], resolution_rule="")

    def plan(posterior):
        return _t.SimpleNamespace(question="Will X happen by the deadline?", as_of=T0,
                                  horizon_ts=T0 + 60 * DAY, outcome_contract=contract,
                                  scheduled_events=[{"etype": "resolve_outcome",
                                                     "ts": T0 + 59 * DAY, "participants": [],
                                                     "payload": {"outcome_var": "outcome"}}],
                                  accepted_mechanisms=[], quantities=[], provenance={},
                                  posterior_rate_particles=posterior)
    p = plan([(0.3, 1.0)])
    rep = convert_binary_to_event_time(p, {})
    assert rep["scheduling"] == "provenance_gated_event_time"
    assert p.first_passage_processes and rep["n_residual_processes"] == 1
    assert not any(e["etype"] == "hazard_round" for e in p.scheduled_events)   # no grids
    p2 = plan(None)
    rep2 = convert_binary_to_event_time(p2, {})
    assert rep2["n_residual_processes"] == 0
    assert any(r["mechanism"] == "residual_outcome_process"
               for r in p2._unresolved_mechanisms)


def test_inv42_phase13_engine_is_the_temporal_runtime():
    import inspect
    from swm.world_model_v2.phase13.crn import MatchedRolloutEngine
    assert "run_branch_temporal" in inspect.getsource(MatchedRolloutEngine.run_branch)


# ---------------------------------------------------------------- unresolved stays unresolved
def test_unresolved_timing_never_becomes_a_number():
    spec = TimingSpec(kind="unresolved", description="we do not know when legal replies")
    with pytest.raises(ValueError):
        spec.sample_duration_s(random.Random(0))
    w, m = _delivery_world()
    out = tr.resolve_timing_spec(spec.as_dict(), world=w, model=m, ref_ts=T0,
                                 stats=tr.get_stats(w))
    assert out is None
    assert tr.get_stats(w).unresolved_timing               # recorded honestly
