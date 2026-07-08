"""IndividualAgent — ONE person modeled as an agent with internal STATE that acts on their response.

Level 1 of the social world model. The other levels simulate *groups* (a committee deliberating, an
electorate cascading); this one simulates a *single person* as a little dynamical system — the thing the
thesis has been missing. A person here is NOT a static variable vector; they are:

  - WHO THEY ARE — stable variables carried in a `VariableMap` (dispositions, incentives, relational ties,
    persona traits): openness, skepticism, goal-alignment, trust in the source, and so on.
  - HOW THEY ARE RIGHT NOW — a mutable STATE that evolves as they are contacted and as they respond:
    attention/busyness, mood, cognitive load, how recently they were contacted, and the sense of owing a
    reply. This is the "incentives, busyness, mood, and everything else acting on them" made concrete.

Simulating the person is reading their response to a message THROUGH that state, then letting the message
and the act of responding UPDATE the state — so the identical message lands differently on a fresh,
receptive person than on a depleted, irritated one, and a sequence of asks has momentum. That state
feedback is what makes this a simulation of the person rather than a lookup of a fixed probability.

The response probability itself comes from a pluggable `response_fn` (structured / grounded, validated on
real persuasion data in EXP-060; or an LLM reasoning AS the person in production) — exactly the
pluggable-backend pattern used everywhere else in the system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.variables.variable_map import VariableMap

# schema variables that constitute the transient, evolving STATE (everything else is stable "who they are")
STATE_VARS = ("attention_availability", "mood_valence", "cognitive_load", "recency_of_contact",
              "reciprocity_debt", "urgency_fit")
# neutral baselines the state relaxes back toward between contacts (busyness/mood recover; contact fades)
STATE_BASELINE = {"attention_availability": 0.6, "mood_valence": 0.0, "cognitive_load": 0.2,
                  "recency_of_contact": 0.0, "reciprocity_debt": 0.0, "urgency_fit": 0.5}


def _clamp(v, signed=False):
    return max(-1.0, min(1.0, v)) if signed else max(0.0, min(1.0, v))


@dataclass
class IndividualAgent:
    """A person as a stateful agent. `variables` = who they are (stable); `state` = how they are now;
    `memory` (optional) = their episodic stream — as contacts land, they are written to memory so a
    situation-conditioned response_fn can recall how this person reacted to similar messages before."""
    agent_id: str
    variables: VariableMap = field(default_factory=VariableMap)
    state: dict = field(default_factory=dict)
    history: list = field(default_factory=list)      # log of (message, responded, p, state-snapshot)
    memory: object = None                            # optional swm.memory.MemoryStream
    clock: float = 0.0                               # step clock; ts fallback when a message carries no _as_of

    def __post_init__(self):
        for k, base in STATE_BASELINE.items():
            self.state.setdefault(k, self.variables.get(k, base) if k in self.variables.vars else base)

    def remember(self, text: str, *, responded: bool, ts=None, topic: str = "", importance: float = None):
        """Write a contact episode to this agent's memory stream (no-op if the agent has no memory)."""
        if self.memory is None:
            return None
        return self.memory.record_contact(ts=self.clock if ts is None else ts, text=text,
                                          responded=responded, topic=topic, importance=importance)

    # ---- reading a response THROUGH the current state ------------------------------------------------
    def response_p(self, message: dict, response_fn) -> dict:
        """P(this person responds favorably to `message` right now) via the pluggable response_fn."""
        return response_fn(self.variables, dict(self.state), dict(message))

    # ---- the message and the act of responding ACT ON the person (the dynamics) ----------------------
    def apply(self, message: dict, responded: bool, p: float = None) -> "IndividualAgent":
        """Update the transient state after a contact: this is what makes the person a dynamical system.
        Grounded, first-principles moves — no fitted magic, just the direction each force pushes:
          - being contacted spends a little attention and raises the sense of owing a reply (recency up);
          - a pushy / disrespectful ask sours mood; a respectful, well-fit one lifts it;
          - a high-effort ask raises cognitive load (and load lingers);
          - actually responding discharges the reciprocity debt and costs attention."""
        s = self.state
        eff = float(message.get("effort_cost", 0.4))
        push = float(message.get("pushiness", 0.3))
        polite = float(message.get("politeness_disposition", message.get("respectfulness", 0.5)))
        fit = float(message.get("personalization", 0.5))

        s["recency_of_contact"] = 1.0
        s["reciprocity_debt"] = _clamp(s["reciprocity_debt"] + 0.25, signed=True)
        s["attention_availability"] = _clamp(s["attention_availability"] - 0.15 * eff)
        s["cognitive_load"] = _clamp(s["cognitive_load"] + 0.20 * eff)
        s["mood_valence"] = _clamp(s["mood_valence"] + 0.25 * (polite + fit - push - 0.5), signed=True)
        if responded:
            s["reciprocity_debt"] = _clamp(s["reciprocity_debt"] - 0.6, signed=True)
            s["attention_availability"] = _clamp(s["attention_availability"] - 0.15)
        self.history.append({"message": message, "responded": responded,
                             "p": None if p is None else round(p, 4), "state": dict(s)})
        if self.memory is not None:                  # accrue episodic memory of the contact + outcome
            ts = message.get("_as_of", self.clock)
            self.memory.record_contact(ts=ts, text=str(message.get("text", "")), responded=bool(responded),
                                       topic=str(message.get("topic", "")))
        self.clock += 1
        return self

    def relax(self, steps: int = 1, rate: float = 0.34) -> "IndividualAgent":
        """Time passing between contacts: the state decays back toward baseline (busyness eases, a
        recent contact fades, mood settles). Lets a thread have realistic spacing/momentum."""
        for _ in range(max(0, steps)):
            for k, base in STATE_BASELINE.items():
                signed = k in ("mood_valence", "reciprocity_debt")
                self.state[k] = _clamp(self.state[k] + rate * (base - self.state[k]), signed=signed)
            self.clock += 1
        return self

    def snapshot(self) -> dict:
        return {"agent_id": self.agent_id, "state": {k: round(v, 3) for k, v in self.state.items()}}
