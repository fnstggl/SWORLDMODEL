"""IndividualSimulator — the Level-1 front door: simulate ONE person, and choose the best action on them.

The product surface for individual questions:

  - `predict_response(person, message)` — "how will person A respond to THIS email?" Reads the person's
    response through their current state, with an auditable breakdown of the drivers.
  - `best_message(person, candidates)` — "what is the best email to send person A for outcome X?" Ranks
    candidate messages by simulating each on the person and returns the argmax. This is the
    InterventionSelector's do(x) choice, but GROUNDED in this specific person's variable map rather than a
    generic judgment — the whole point of Level 1.
  - `simulate_thread(person, messages)` — roll the person forward through a sequence of contacts, letting
    each message and response UPDATE their state (mood, busyness, load, reciprocity), so a follow-up lands
    on the person the earlier messages left behind — not a fresh copy each time.

`response_fn` is pluggable (structured/grounded — validated in EXP-060; or the LLM `llm_response_fn` in
production), so the same object that scores on real data serves real questions.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.simulation.individual_agent import IndividualAgent
from swm.variables.variable_map import VariableMap


def _to_agent(person) -> IndividualAgent:
    if isinstance(person, IndividualAgent):
        return person
    if isinstance(person, VariableMap):
        return IndividualAgent(agent_id=person.entity_id or "person", variables=person)
    if isinstance(person, dict):                       # {name: value} convenience
        vm = VariableMap(entity_id="person")
        for n, v in person.items():
            vm.set(n, v, provenance="user", confidence=0.9)
        return IndividualAgent(agent_id="person", variables=vm)
    raise TypeError(f"person must be IndividualAgent | VariableMap | dict, got {type(person)}")


@dataclass
class IndividualSimulator:
    response_fn: object                                # callable(variables, state, message) -> {"p", ...}

    def predict_response(self, person, message: dict, *, threshold: float = 0.5) -> dict:
        agent = _to_agent(person)
        out = agent.response_p(message, self.response_fn)
        p = out["p"]
        return {"agent_id": agent.agent_id, "p_respond": round(p, 4), "will_respond": p >= threshold,
                "drivers": out.get("drivers", {}), "state": agent.snapshot()["state"]}

    def best_message(self, person, candidates: list, *, labels: list = None) -> dict:
        """Rank candidate messages by simulating each on a FRESH copy of the person; return the argmax.
        Each candidate is scored independently (the choice is which single message to send)."""
        agent = _to_agent(person)
        ranked = []
        for i, msg in enumerate(candidates):
            probe = IndividualAgent(agent_id=agent.agent_id, variables=agent.variables, state=dict(agent.state))
            p = probe.response_p(msg, self.response_fn)["p"]
            ranked.append({"index": i, "label": (labels[i] if labels else f"option_{i}"),
                           "p_respond": round(p, 4), "message": msg})
        ranked.sort(key=lambda r: -r["p_respond"])
        best = ranked[0] if ranked else None
        lift = round(best["p_respond"] - sum(r["p_respond"] for r in ranked) / len(ranked), 4) if ranked else None
        return {"best": best, "ranking": ranked, "lift_over_mean": lift}

    def simulate_thread(self, person, messages: list, *, threshold: float = 0.5, gap_steps: int = 1) -> dict:
        """Roll the person forward through a sequence of contacts. State carries over: each message and
        the (simulated) response update mood/busyness/load/reciprocity before the next one lands."""
        agent = _to_agent(person)
        steps = []
        for k, msg in enumerate(messages):
            if k > 0:
                agent.relax(steps=gap_steps)           # time passes between contacts
            out = agent.response_p(msg, self.response_fn)
            p = out["p"]
            responded = p >= threshold
            steps.append({"turn": k, "p_respond": round(p, 4), "responded": responded,
                          "drivers": out.get("drivers", {}),
                          "state_before": {kk: round(vv, 3) for kk, vv in agent.state.items()}})
            agent.apply(msg, responded, p)             # the message + response act on the person
        return {"agent_id": agent.agent_id, "turns": steps,
                "final_state": agent.snapshot()["state"],
                "any_response": any(s["responded"] for s in steps)}
