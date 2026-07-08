"""Future-event model — the DISCRETE / JUMP term of the forward operator (the branching-realities core).

EXP-033's verdict, in the repo's own words: *"you cannot forecast forward without forecasting the events …
long-horizon forecasting is a future-event-forecasting problem, not a better-dynamics problem."* Two sibling
models already cover the *continuous* term — `swm/transition/event_model.py` (EXP-035, a smooth
heteroskedastic diffusion of a scalar belief) and `swm/simulation/event_model.py` (EXP-077, calibrated
continuous-Gaussian-JUMP variance placement for interval coverage) — and the calibrated transition-operator
work sharpens the between-event drift/volatility. This module is the complementary DISCRETE-CATEGORICAL term
none of those can express: the pivotal FUTURE EVENTS that fork reality into distinct branches — a CPI print, an FOMC decision, an election, a playoff game — each with a *distribution over its
outcomes* and a *jump* each outcome applies to the belief. A diffusion smears variance smoothly and stays
unimodal; discrete events place variance at KNOWN DATES and make the future genuinely MULTIMODAL ("25% if
the Fed holds, 85% if it cuts"). That multimodality is the object that lets us *model past the horizon
without faking a point*.

Three pieces, all mechanism-agnostic and dependency-free:
  - `FutureEvent`   — a dated event with a categorical outcome distribution; each outcome either JUMPS the
                      belief (`impact`) or RESOLVES the question (`resolves`, terminating the trajectory).
  - `SurpriseHazard`— a base-rate Poisson process of UNscheduled shocks (the unknown-unknowns variance).
  - `EventCalendar` — the known events over a horizon + the hazard; the object the branching rollout walks.

The `event→impact` mapping (what a real dated event does to the belief, and with what outcome odds) is the
LLM's job in production — `events_from_records` takes structured events (for tests / a resolved calendar),
and `EventImpactJudge` wraps a pluggable judge_fn (reusing the EXP-030 channel) to author them from text.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EventOutcome:
    """One branch of an event. Exactly one of `impact` (a signed jump on the belief) or `resolves` (a
    terminal outcome value that ends the trajectory) is the effect; `prob` is this branch's probability."""
    label: str
    prob: float
    impact: float = 0.0                 # signed jump applied to the tracked belief if this branch fires
    resolves: float | None = None       # if set, the trajectory RESOLVES to this value (e.g. 1.0 yes / 0.0 no)


@dataclass
class FutureEvent:
    """A dated event whose discrete outcome forks the future. `time` is in the horizon's own units.
    `from_belief=True` marks a *resolving* event whose yes-probability is the belief itself at that moment
    (the honest martingale-respecting resolution: the belief IS P(outcome)); its `outcomes` are then
    ignored and it resolves 1.0 w.p. belief, else 0.0."""
    name: str
    time: float
    outcomes: list = field(default_factory=list)     # list[EventOutcome]
    from_belief: bool = False

    def sample(self, rng, belief: float):
        """Draw an outcome. Returns (label, effect) where effect is ("impact", x) or ("resolve", value)."""
        if self.from_belief:
            yes = rng.random() < belief
            return ("yes" if yes else "no"), ("resolve", 1.0 if yes else 0.0)
        r = rng.random()
        cum = 0.0
        chosen = self.outcomes[-1]
        for oc in self.outcomes:
            cum += oc.prob
            if r <= cum:
                chosen = oc
                break
        if chosen.resolves is not None:
            return chosen.label, ("resolve", float(chosen.resolves))
        return chosen.label, ("impact", float(chosen.impact))

    def labels(self) -> list:
        return ["yes", "no"] if self.from_belief else [oc.label for oc in self.outcomes]

    def __post_init__(self):
        if not self.from_belief and self.outcomes:
            s = sum(oc.prob for oc in self.outcomes)
            if s > 0 and abs(s - 1.0) > 1e-9:            # tolerate un-normalized odds from an LLM
                for oc in self.outcomes:
                    oc.prob = oc.prob / s


@dataclass
class SurpriseHazard:
    """Base-rate UNscheduled shocks — the unknown-unknowns. Poisson `rate` shocks per unit time, each a
    mean-zero jump with std `shock_sd`. This is the irreducible surprise floor that widens every horizon."""
    rate: float = 0.0
    shock_sd: float = 0.05

    def sample_impacts(self, dt: float, rng) -> list:
        if self.rate <= 0 or dt <= 0:
            return []
        # number of shocks in dt ~ Poisson(rate*dt); Knuth's algorithm (dependency-free)
        lam = self.rate * dt
        L = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            p *= rng.random()
            if p <= L:
                break
            k += 1
        return [rng.gauss(0.0, self.shock_sd) for _ in range(k)]


@dataclass
class EventCalendar:
    """The known dated events over a horizon + a surprise hazard. Walked by the branching rollout."""
    events: list = field(default_factory=list)          # list[FutureEvent]
    hazard: SurpriseHazard = field(default_factory=SurpriseHazard)

    def scheduled_in(self, t0: float, t1: float) -> list:
        """Events firing in the half-open interval (t0, t1], in time order."""
        return sorted((e for e in self.events if t0 < e.time <= t1 + 1e-12), key=lambda e: e.time)

    def resolving_events(self) -> list:
        return [e for e in self.events
                if e.from_belief or any(oc.resolves is not None for oc in e.outcomes)]

    def event_names(self) -> list:
        return [e.name for e in self.events]


# ---- builders ---------------------------------------------------------------------------------------
def events_from_records(records: list) -> EventCalendar:
    """Build a calendar from structured event dicts (tests / a pre-resolved calendar):
      {"name","time","outcomes":[{"label","prob","impact"?,"resolves"?}], "from_belief"?}
    plus an optional trailing hazard dict {"hazard": {"rate","shock_sd"}}."""
    events, hazard = [], SurpriseHazard()
    for rec in records:
        if "hazard" in rec:
            h = rec["hazard"]
            hazard = SurpriseHazard(rate=float(h.get("rate", 0.0)), shock_sd=float(h.get("shock_sd", 0.05)))
            continue
        outs = [EventOutcome(label=str(o.get("label", "")), prob=float(o.get("prob", 0.0)),
                             impact=float(o.get("impact", 0.0)),
                             resolves=(None if o.get("resolves") is None else float(o["resolves"])))
                for o in rec.get("outcomes", [])]
        events.append(FutureEvent(name=str(rec["name"]), time=float(rec["time"]), outcomes=outs,
                                  from_belief=bool(rec.get("from_belief", False))))
    return EventCalendar(events=events, hazard=hazard)


@dataclass
class EventImpactJudge:
    """PRODUCTION event→impact: an LLM reads the retrieved future timeline and authors the calendar — for
    each dated pivotal event, its outcome odds and the jump each outcome applies to the belief. Pluggable
    `judge_fn(prompt) -> JSON` (the EXP-030 channel). Kept behind the same interface the rest of the system
    uses; `events_from_records` is the offline/test path."""
    judge_fn: object

    def build(self, question: str, timeline: str, horizon: float) -> EventCalendar:
        import json
        prompt = (
            "You forecast the FUTURE EVENTS that will move the answer to a question over a horizon.\n"
            f"QUESTION: {question}\nHORIZON (time units): {horizon}\n"
            f"KNOWN UPCOMING TIMELINE:\n{timeline}\n\n"
            "List the pivotal dated events between now and the horizon. For each: its time (0..horizon), "
            "its possible outcomes with probabilities that sum to 1, and for each outcome EITHER an "
            "'impact' (signed change in the probability-of-YES, -1..1) OR 'resolves' (1 if this outcome "
            "makes the answer YES, 0 if NO) for the event that settles the question. Return ONLY JSON: "
            '{"events":[{"name","time","from_belief":false,"outcomes":[{"label","prob","impact"?,"resolves"?}]}]}')
        raw = self.judge_fn(prompt)
        obj = raw if isinstance(raw, dict) else json.loads(str(raw)[str(raw).find("{"):str(raw).rfind("}") + 1])
        return events_from_records(obj.get("events", []))
