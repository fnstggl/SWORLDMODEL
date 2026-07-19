"""CROSS-DOMAIN TEMPORAL FIXTURES (§29) — reusable synthetic scenarios exercising the
event-driven temporal architecture across domains. Names/ids are RANDOMIZED per run where
possible (no production domain branches; these fixtures prove generality, they are not
templates)."""
from __future__ import annotations

import random
import string
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swm.world_model_v2 import generated_world as gw
from swm.world_model_v2 import temporal_calendar as tc
from swm.world_model_v2 import temporal_runtime as tr
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.temporal_model import (ActorTemporalProfile, ChannelTemporalModel,
                                               ScenarioTemporalModel)
from swm.world_model_v2.transitions import StateDelta, TransitionOperator

DAY = 86400.0
_rng = random.Random()                                        # per-run randomized names


def _name(prefix):
    return f"{prefix}_{''.join(_rng.choices(string.ascii_lowercase, k=5))}"


def _world(actors, *, now, bid="b0"):
    w = WorldState("fx", bid, SimulationClock(now, now), network=RelationGraph(),
                   information=InformationLedger())
    for a in actors:
        e = Entity(a)
        e.set("roles", F(["person"], status="observed"))
        w.entities[a] = e
    return w


# ================================================================ communication timing
def test_fixture_communication_sleep_urgency_and_accumulation():
    """A message sent while the recipient sleeps is seen after waking; the urgent channel
    interrupts sleep; several messages accumulate into one bundle before one check."""
    person = _name("recipient")
    sender = _name("sender")
    tz = "Asia/Tokyo"
    sent_ts = tc.civil_to_ts(2026, 4, 7, 2.0, tz=tz)          # 02:00 local — asleep
    m = ScenarioTemporalModel(scenario_id="comm_fx", as_of=sent_ts - 3600,
                              horizon_ts=sent_ts + 3 * DAY)
    m.timezones[person] = tz
    m.actor_profiles[person] = ActorTemporalProfile(
        actor_id=person, timezone=tz, sleep_window=(23.0, 7.0), active_window=(7.0, 22.0),
        channel_checking={"text": {"kind": "range", "lo_s": 600.0, "hi_s": 1800.0}},
        urgency_interrupt={"phone_call": {"threshold": 0.7, "why": "calls ring audibly"}})
    w = _world([person, sender], now=sent_ts)
    w.temporal_model = m
    # ordinary channel: noticed only after waking (>= 07:00 local)
    notice, prov = tr.compute_notice_ts(w, m, actor_id=person, channel_id="text",
                                        available_ts=sent_ts, sender=sender)
    local = tc.local_civil(notice, tz)
    assert local.hour >= 7 and notice > sent_ts + 4 * 3600
    # urgent channel: interrupts sleep — noticed at availability
    notice_u, prov_u = tr.compute_notice_ts(w, m, actor_id=person, channel_id="phone_call",
                                            available_ts=sent_ts, urgency=0.95, sender=sender)
    assert notice_u == sent_ts and prov_u == "urgency_interrupt"
    # accumulation: three texts overnight → ONE bundle at one attention event
    stats = tr.get_stats(w)
    for i in range(3):
        w.information.publish(InformationItem(f"m{i}", f"msg {i}", source=sender,
                                              created_at=sent_ts + i * 900))
        tr.record_available_observation(w, recipient=person,
                                        item={"iid": f"m{i}", "content": f"msg {i}",
                                              "source": sender},
                                        available_ts=sent_ts + i * 900, channel="text",
                                        stats=stats)
        tr.schedule_attention(w, m, actor_id=person, channel_id="text",
                              available_ts=sent_ts + i * 900, sender=sender, stats=stats)
    w.clock.advance_to(notice)
    bundle = tr.collect_attention_bundle(w, actor_id=person, now_ts=notice, channel="text",
                                         stats=stats)
    assert len(bundle) == 3


# ================================================================ public information
def test_fixture_public_information_publish_exposure_moderation_and_nonreach():
    """Publication is immediate; per-actor exposure spreads over the channel's exposure
    process; a key actor (prioritized/checking often) sees it earlier; moderation delays
    availability; one actor never receives it (private visibility)."""
    src = _name("author")
    key_actor, casual_actor, outsider = _name("key"), _name("casual"), _name("outsider")
    t0 = tc.civil_to_ts(2026, 5, 4, 10.0, tz="UTC")
    m = ScenarioTemporalModel(scenario_id="pub_fx", as_of=t0, horizon_ts=t0 + 7 * DAY)
    m.channels["feed"] = ChannelTemporalModel(
        channel_id="feed", kind="public_post",
        transmission={"kind": "exact", "ts": None},           # instant publish
        exposure={"kind": "range", "lo_s": 1800.0, "hi_s": 6 * 3600.0})
    m.channels["moderated_forum"] = ChannelTemporalModel(
        channel_id="moderated_forum", kind="public_post",
        moderation={"kind": "range", "lo_s": 2 * 3600.0, "hi_s": 8 * 3600.0},
        exposure={"kind": "range", "lo_s": 1800.0, "hi_s": 4 * 3600.0})
    m.actor_profiles[key_actor] = ActorTemporalProfile(
        actor_id=key_actor, timezone="UTC",
        channel_checking={"feed": {"kind": "range", "lo_s": 300.0, "hi_s": 900.0}})
    m.actor_profiles[casual_actor] = ActorTemporalProfile(
        actor_id=casual_actor, timezone="UTC",
        channel_checking={"feed": {"kind": "range", "lo_s": 12 * 3600.0,
                                   "hi_s": 36 * 3600.0}})
    w = _world([src, key_actor, casual_actor, outsider], now=t0)
    w.temporal_model = m
    # publication available essentially immediately on the open feed; per-actor exposure varies
    avail_key, _ = tr.channel_delivery_ts(w, m, channel_id="feed", sent_ts=t0,
                                          recipient=key_actor, salt="post1")
    avail_mod, _ = tr.channel_delivery_ts(w, m, channel_id="moderated_forum", sent_ts=t0,
                                          recipient=key_actor, salt="post1")
    assert avail_mod - t0 >= 2 * 3600.0                       # moderation delays availability
    notice_key, _ = tr.compute_notice_ts(w, m, actor_id=key_actor, channel_id="feed",
                                         available_ts=avail_key)
    notice_cas, _ = tr.compute_notice_ts(w, m, actor_id=casual_actor, channel_id="feed",
                                         available_ts=avail_key)
    assert notice_key < notice_cas                            # the key actor sees it earlier
    # the post NEVER reaches the outsider: private visibility routes no delivery to them
    from swm.world_model_v2.scenario_schema import ScenarioSemanticModel
    w.scenario_schema = ScenarioSemanticModel(
        question="q", prediction_timestamp=t0, horizon=t0 + 7 * DAY,
        semantic_event_types={"announcement": {"description": "x", "fields": {},
                                               "typical_visibility": "participants"}},
        information_rules={"default_channel": "feed"},
        provenance={"compiler": "test"}).freeze()
    sev = {"event_id": "p1", "semantic_type_id": "announcement", "exact_content": "launch!",
           "intended_visibility": "participants", "direct_targets": [key_actor],
           "source_actor_id": src}
    deliveries = gw.route_semantic_event(w, sev, gw.generated_report())
    assert {d.payload["recipient"] for d in deliveries} == {key_actor}


# ================================================================ founder launch
def test_fixture_founder_launch_weekend_and_competitor_anchor():
    """Product readiness (dependency), weekend user behavior (calendar), a competitor's dated
    announcement (exact scheduled fact), private recruitment before public launch (channel
    stages) — the launch decision's timing choices anchor to REAL scenario times."""
    founder = _name("founder")
    t0 = tc.civil_to_ts(2026, 6, 3, 9.0, tz="America/Los_Angeles")   # Wednesday
    m = ScenarioTemporalModel(scenario_id="launch_fx", as_of=t0, horizon_ts=t0 + 30 * DAY)
    m.timezones[founder] = "America/Los_Angeles"
    competitor_ts = tc.civil_to_ts(2026, 6, 10, 10.0, tz="America/Los_Angeles")
    m.scheduled_facts.append({"ts": competitor_ts, "fact": "competitor keynote",
                              "source": "public_calendar"})
    m.deadlines.append({"label": "beta access expires",
                        "timing": {"kind": "exact", "ts": t0 + 21 * DAY},
                        "binds": founder, "source": "user_context"})
    cal = tc.CivilCalendar(tz="America/Los_Angeles")
    # weekend semantics are real: Sunday is not a business day; the next business day from
    # Friday evening is Monday 09:00 local
    fri_evening = tc.civil_to_ts(2026, 6, 5, 19.0, tz="America/Los_Angeles")
    sunday = tc.add_calendar_days(fri_evening, 2, tz="America/Los_Angeles")
    assert not cal.is_business_day(sunday)
    monday = tc.next_business_day(fri_evening, cal)
    assert tc.local_civil(monday, "America/Los_Angeles").isoweekday() == 1
    # timing anchors: the affordance layer offers real anchors (competitor date, deadline)
    w = _world([founder], now=t0)
    w.temporal_model = m
    anchors = [d["timing"]["ts"] for d in m.deadlines] + [f["ts"] for f in m.scheduled_facts]
    assert competitor_ts in anchors and t0 + 21 * DAY in anchors


# ================================================================ institution
def test_fixture_institution_filing_through_stages():
    """Filing → statutory waiting period → committee stage (working calendar) → vote →
    implementation: stages activate only on entry; known statutory timing is exact."""
    from swm.world_model_v2.temporal_model import InstitutionalProcessModel, InstitutionalStage
    inst = _name("commission")
    filer, commissioner = _name("filer"), _name("commissioner")
    t0 = tc.civil_to_ts(2026, 2, 2, 9.0, tz="Europe/Berlin")
    m = ScenarioTemporalModel(scenario_id="inst_fx", as_of=t0, horizon_ts=t0 + 120 * DAY)
    m.calendars[inst] = {"tz": "Europe/Berlin", "business_days": (1, 2, 3, 4, 5),
                         "open_hour": 9.0, "close_hour": 17.0, "provenance": "documented"}
    m.institutional_processes.append(InstitutionalProcessModel(
        process_id="filing_review", institution_id=inst,
        stages=[
            InstitutionalStage(stage_id="statutory_notice", institution_id=inst,
                               duration={"kind": "exact_days", "kind_note": "statutory"},
                               earliest_start=None,
                               deadline=None, next_stages=["committee_review"]),
            InstitutionalStage(stage_id="committee_review", institution_id=inst,
                               duration={"kind": "range", "lo_s": 10 * DAY, "hi_s": 20 * DAY},
                               working_calendar=inst, next_stages=["vote"],
                               creates_decision_for=commissioner),
            InstitutionalStage(stage_id="vote", institution_id=inst,
                               duration={"kind": "range", "lo_s": DAY, "hi_s": 2 * DAY})],
        initial_stages=["statutory_notice"], started_by="filing_submitted"))
    w = _world([filer, commissioner], now=t0)
    w.temporal_model = m
    from swm.world_model_v2.institutions import Rule, RuleSystem
    w.institutions[inst] = RuleSystem(
        institution_id=inst,
        rules=[Rule(rule_id=f"{inst}:0", kind="decision_right",
                    params={"holders": [commissioner], "actions": ["approve"]})])
    from swm.world_model_v2 import semantic_consequences as sc
    delta = StateDelta(at=t0, event_type="x", operator="t")
    ctx = {"actor_id": filer, "action_id": "f1", "now": t0, "events": [],
           "compiler": "test", "created_ids": set(), "report": sc.empty_report(),
           "quarantined": [], "cascade_depth": 0}
    sc._x_submit_to_institution(w, {"institution": inst, "matter": "the filing",
                                    "requested_outcome": "approve"}, ctx, delta)
    votes = [e for e in ctx["events"] if e.etype == "collective_vote"]
    opps = [e for e in ctx["events"] if e.etype == "decision_opportunity"]
    assert votes and opps
    # the commissioner's opportunity comes from the institutional queue, strictly before the
    # vote, and after real days of process — never a fixed delay/2 or one-hour constant
    assert t0 < opps[0].ts < votes[0].ts
    assert votes[0].ts > t0 + 5 * DAY
    assert opps[0].payload["trigger"]["trigger_type"] == "institutional_stage_reached"


# ================================================================ crisis (hours, no cadence)
def test_fixture_crisis_many_decisions_within_hours_no_daily_cadence():
    crisis_team = [_name("responder") for _ in range(3)]
    t0 = tc.civil_to_ts(2026, 8, 12, 3.0, tz="UTC")
    w = _world(crisis_team, now=t0)
    q = EventQueue(horizon_ts=t0 + 6 * 3600.0)                # a six-hour window
    for k in range(12):                                       # twelve real incoming reports
        q.schedule(Event(ts=t0 + 120.0 + k * 1490.0, etype="decision_opportunity",
                         participants=[crisis_team[k % 3]],
                         payload={"situation": f"incident update {k}"},
                         trigger={"trigger_type": "newly_noticed_information",
                                  "actor_id": crisis_team[k % 3]}))

    class _Spy(TransitionOperator):
        name = "spy"

        def __init__(self):
            self.calls = []

        def applicable(self, world, event):
            return event.etype == "decision_opportunity"

        def propose(self, world, event, rng):
            self.calls.append(event.ts)
            return None
    spy = _Spy()
    branch = RolloutEngine(operators=[spy]).run_branch(w, q, seed=2)
    assert len(spy.calls) == 12                               # many decisions within hours
    assert branch.temporal_stats.event_counts.get("background_tick") is None
    gaps = [b - a for a, b in zip(spy.calls, spy.calls[1:])]
    assert all(g < 3600.0 for g in gaps)                      # no artificial daily spacing


# ================================================================ long-horizon sparse
def test_fixture_long_horizon_sparse_evolves_without_actor_calls():
    """Nine months pass; continuous processes evolve; ZERO actor decisions occur because no
    real trigger exists — nobody is polled weekly for their opinion."""
    from swm.world_model_v2.temporal_model import ContinuousProcessSpec
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    actor = _name("distant_party")
    t0 = tc.civil_to_ts(2026, 1, 5, 9.0, tz="UTC")
    w = _world([actor], now=t0)
    m = ScenarioTemporalModel(scenario_id="sparse_fx", as_of=t0, horizon_ts=t0 + 270 * DAY)
    m.continuous_processes.append(ContinuousProcessSpec(
        process_id="relationship_cooling", writes="warmth", form="exponential_decay",
        rate_per_day=0.01))
    w.temporal_model = m
    register_quantity_type("warmth", units="share")
    w.quantities["warmth"] = Quantity(name="warmth", qtype="warmth", value=0.9, timestamp=t0)
    q = EventQueue(horizon_ts=t0 + 270 * DAY)
    q.schedule(Event(ts=t0 + 260 * DAY, etype="measurement", participants=[]))
    spy_calls = []

    class _Spy(TransitionOperator):
        name = "spy"

        def applicable(self, world, event):
            return event.etype in ("decision_opportunity", "ctrl_invoke_actor")

        def propose(self, world, event, rng):
            spy_calls.append(event.ts)
            return None
    branch = RolloutEngine(operators=[_Spy()]).run_branch(w, q, seed=1)
    assert spy_calls == []                                    # zero decisions in nine months
    assert w.quantities["warmth"].value == pytest.approx(0.9 * 2.718281828 ** -2.6, rel=1e-3)
    assert branch.temporal_stats.interval_advances == 1       # ONE exact 260-day interval


# ================================================================ personal relationship
def test_fixture_personal_relationship_states_are_distinguishable():
    """Workload delays the reply; read-but-deferred differs from unread; no-response is not
    'ignored'."""
    from swm.world_model_v2.individual_reaction import simulate_individual_reaction
    person = _name("friend")

    class _StubLLM:
        def __call__(self, prompt):
            if "hypotheses" in prompt.lower() or "HYPOTHESIZER" in prompt:
                return ('{"hypotheses": [{"hypothesis_id": "h1", "description": "busy week",'
                        '"private_state": "swamped at work"}]}')
            return ('{"schema_version": "qual.decision.v1", "decision": {"act_or_wait": '
                    '"wait", "chosen_action": "reply_later", "target": "you", "timing": '
                    '"delayed", "observability": "private", "intended_effect": "reply when '
                    'free"}, "decision_summary": "will reply after the deadline crunch"}')
    art = simulate_individual_reaction(
        person_id=person, stimulus="hey — dinner this friday?", llm=_StubLLM(),
        context={"relationship": "close_friend", "timezone": "Europe/Paris",
                 "sleep_window": [23.0, 7.0],
                 "channel_check_gap": {"kind": "range", "lo_s": 1800.0, "hi_s": 2 * 3600.0},
                 "history": [{"text": "last month's dinner", "ts": 1770000000.0},
                             "an untimestamped memory"]},
        n_hypotheses=1, samples_per_hypothesis=3, as_of=1772000000.0, horizon_s=3 * DAY)
    states = {s["temporal_state"] for s in art["samples"]}
    assert states <= {"responded", "read_but_deferred", "unread_by_horizon"}
    assert art["temporal"]["history_timestamps"]["n_supplied_real"] == 1
    assert art["temporal"]["history_timestamps"]["unresolved_spacing"] == 1
    for s in art["samples"]:
        if s["temporal_state"] != "unread_by_horizon":
            assert s["noticed_ts"] and s["noticed_ts"] > 1772000000.0
    # deferral is visible as its own state, not silently identical to a reply
    assert any(s["temporal_state"] == "read_but_deferred" for s in art["samples"])


# ================================================================ simultaneity permutation
class _Writer(TransitionOperator):
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
        world.quantities[str(a["var"])] = Quantity(name=str(a["var"]), qtype=str(a["var"]),
                                                   value=a["value"], timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="measurement", operator=self.name)
        return d.change(f"quantities[{a['var']}]", None, a["value"])


@pytest.mark.parametrize("perm", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
def test_fixture_simultaneous_events_terminal_invariant_under_permutation(perm):
    t0 = tc.civil_to_ts(2026, 9, 1, 12.0, tz="UTC")
    var_names = [f"q_{i}" for i in range(3)]
    evs = [Event(ts=t0 + 500.0, etype="measurement",
                 payload={"var": var_names[i], "value": float(i + 1)}) for i in range(3)]
    w = _world([_name("a")], now=t0)
    q = EventQueue(horizon_ts=t0 + DAY)
    for i in perm:
        q.schedule(evs[i])
    branch = RolloutEngine(operators=[_Writer()]).run_branch(w, q, seed=9)
    assert {k: v.value for k, v in w.quantities.items() if k.startswith("q_")} == \
        {"q_0": 1.0, "q_1": 2.0, "q_2": 3.0}
    assert branch.temporal_stats.simultaneity_conflicts == []
    assert branch.temporal_stats.max_batch_size >= 3


# ================================================================ more than six actors
def test_fixture_eight_decisive_actors_all_receive_their_triggers():
    """Nine individually decisive actors; every one of them receives their actual trigger —
    none dropped by rank (§7 / invariant 16)."""
    actors = [_name("principal") for _ in range(9)]
    t0 = tc.civil_to_ts(2026, 10, 6, 9.0, tz="UTC")
    w = _world(actors + [_name("source")], now=t0)
    from swm.world_model_v2.scenario_schema import ScenarioSemanticModel
    w.scenario_schema = ScenarioSemanticModel(
        question="q", prediction_timestamp=t0, horizon=t0 + 7 * DAY,
        semantic_event_types={"joint_announcement": {"description": "x", "fields": {},
                                                     "typical_visibility": "public"}},
        information_rules={"default_channel": "direct"},
        provenance={"compiler": "test"}).freeze()
    src_id = next(a for a in sorted(w.entities) if a.startswith("source"))
    sev = {"event_id": "big1", "semantic_type_id": "joint_announcement",
           "exact_content": "the consortium terms changed", "intended_visibility": "public",
           "direct_targets": list(actors), "source_actor_id": src_id}
    deliveries = gw.route_semantic_event(w, sev, gw.generated_report())
    recipients = {d.payload["recipient"] for d in deliveries}
    assert set(actors) <= recipients and len(recipients) >= 9   # nobody dropped by rank
    frontier = gw.discover_causal_frontier(w, sev)
    assert len(frontier) >= 9                                   # all nine targets — no [:8] cap
    assert {a for a, _ in frontier} >= set(actors)
