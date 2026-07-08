"""Future-event model — outcome sampling, normalization, resolution, surprise hazard, calendar windows."""
import random

from swm.transition.future_events import (EventCalendar, EventImpactJudge, EventOutcome, FutureEvent,
                                          SurpriseHazard, events_from_records)


def test_outcome_probabilities_are_normalized():
    ev = FutureEvent("cpi", 2.0, [EventOutcome("hot", 2.0, impact=-0.2), EventOutcome("cool", 2.0, impact=0.2)])
    assert abs(sum(o.prob for o in ev.outcomes) - 1.0) < 1e-9   # 2:2 -> 0.5/0.5


def test_sample_respects_probabilities():
    ev = FutureEvent("cpi", 2.0, [EventOutcome("hot", 0.8, impact=-0.2), EventOutcome("cool", 0.2, impact=0.2)])
    rng = random.Random(0)
    labels = [ev.sample(rng, 0.5)[0] for _ in range(4000)]
    hot = labels.count("hot") / len(labels)
    assert 0.75 < hot < 0.85                                    # ~0.8


def test_impact_vs_resolve_effects():
    imp = FutureEvent("cpi", 1.0, [EventOutcome("hot", 1.0, impact=-0.3)])
    res = FutureEvent("vote", 1.0, [EventOutcome("pass", 1.0, resolves=1.0)])
    rng = random.Random(1)
    assert imp.sample(rng, 0.5) == ("hot", ("impact", -0.3))
    assert res.sample(rng, 0.5) == ("pass", ("resolve", 1.0))


def test_from_belief_resolves_bernoulli():
    ev = FutureEvent("fomc", 4.0, from_belief=True)
    rng = random.Random(0)
    yes = sum(1 for _ in range(4000) if ev.sample(rng, 0.7)[1] == ("resolve", 1.0))
    assert 0.66 < yes / 4000 < 0.74                            # ~P(yes)=belief=0.7


def test_surprise_hazard_zero_rate_is_silent():
    assert SurpriseHazard(rate=0.0).sample_impacts(10.0, random.Random(0)) == []


def test_surprise_hazard_rate_scales_with_time():
    h = SurpriseHazard(rate=1.0, shock_sd=0.05)
    rng = random.Random(0)
    total = sum(len(h.sample_impacts(1.0, rng)) for _ in range(2000))
    assert 1800 < total < 2200                                 # ~1 shock/unit-time over 2000 unit intervals


def test_calendar_window_is_half_open_and_ordered():
    cal = events_from_records([
        {"name": "a", "time": 1.0, "outcomes": [{"label": "x", "prob": 1.0, "impact": 0.0}]},
        {"name": "b", "time": 2.0, "outcomes": [{"label": "x", "prob": 1.0, "impact": 0.0}]},
        {"name": "c", "time": 3.0, "outcomes": [{"label": "x", "prob": 1.0, "impact": 0.0}]},
    ])
    win = cal.scheduled_in(1.0, 3.0)                            # (1,3] -> b,c (not a at t=1)
    assert [e.name for e in win] == ["b", "c"]


def test_events_from_records_parses_hazard_and_resolves():
    cal = events_from_records([
        {"name": "fomc", "time": 4.0, "outcomes": [{"label": "cut", "prob": 0.4, "resolves": 1.0},
                                                   {"label": "hold", "prob": 0.6, "resolves": 0.0}]},
        {"hazard": {"rate": 0.5, "shock_sd": 0.03}},
    ])
    assert cal.hazard.rate == 0.5 and cal.hazard.shock_sd == 0.03
    assert len(cal.resolving_events()) == 1


def test_event_impact_judge_parses_llm_json():
    def fake(prompt):
        return ('{"events":[{"name":"cpi","time":2,"outcomes":['
                '{"label":"hot","prob":0.5,"impact":-0.2},{"label":"cool","prob":0.5,"impact":0.2}]}]}')
    cal = EventImpactJudge(fake).build("Will the Fed cut?", "timeline...", horizon=5)
    assert cal.event_names() == ["cpi"] and len(cal.events[0].outcomes) == 2
