"""GenerativeSimulator — the full generative loop as one simulate(question) call.

The capstone assembly. For an arbitrary question it:
  (1) IDENTIFIES the deciding agents and maps each one's known + inferred variables from accessible context
      (an LLM call — "who decides this, and what are their relevant variables/influence?");
  (2) INSTANTIATES them as `PersonaAgent`s and assigns each an initial POSITION by having the LLM reason
      *as / about that persona* given the question and context (the generative-agent `position_fn`);
  (3) RUNS `AgentSociety` forward — the agents deliberate and interact, so the outcome EMERGES;
  (4) READS the emergent outcome with a full audit trail (each agent's persona, initial and final position,
      the interaction trajectory, and the value drivers).

Everything the LLM does is behind a pluggable backend, exactly like `semantic_stance` / `intervention_
selector`: production supplies `anthropic_*` backends; a dev/replay run supplies cached judgments. The
STRUCTURED `position_fn` path (a value-match, no LLM) reproduces the validated EXP-055 result, so the
assembly is verifiable independent of any LLM call; swapping in the LLM `position_fn` is the general path.

This is the difference the whole project was chasing: not `sum(ps)/n`, but a society of grounded agents
reasoning and interacting until an outcome falls out.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.simulation.agent_society import AgentSociety, PersonaAgent, independent_outcome


@dataclass
class AgentSpec:
    agent_id: str
    variables: dict = field(default_factory=dict)   # known + inferred (ideology, expertise, stance, ...)
    influence: float = 1.0
    openness: float = 0.3
    conviction: float = 0.4


def parse_agents(raw) -> list:
    """Parse an LLM agent-identification payload into AgentSpecs (tolerant of raw JSON strings)."""
    obj = raw if isinstance(raw, (list, dict)) else json.loads(str(raw)[str(raw).find("["):str(raw).rfind("]") + 1])
    items = obj.get("agents", []) if isinstance(obj, dict) else obj
    out = []
    for a in items:
        out.append(AgentSpec(agent_id=str(a.get("id", a.get("name", f"agent{len(out)}"))),
                             variables={k: float(v) for k, v in (a.get("variables") or {}).items()
                                        if _isnum(v)},
                             influence=float(a.get("influence", 1.0)),
                             openness=float(a.get("openness", 0.3)),
                             conviction=float(a.get("conviction", 0.4))))
    return out


def _isnum(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def build_identify_prompt(question: str, context: str = "") -> str:
    return (f"QUESTION: {question}\n{('CONTEXT: ' + context) if context else ''}\n"
            "Identify the AGENTS whose decisions determine this outcome (named individuals for an "
            "institution — justices, FOMC members, voters; or representative segments for a population). "
            "For each, give the variables that drive their position (e.g. ideology 0..1, expertise, "
            "stakes), their influence (how much they sway others), openness (how much they update), and "
            "conviction (resistance to change).\n"
            'Return ONLY JSON: {"agents":[{"id","variables":{...},"influence","openness","conviction"}]}')


def build_position_prompt(question: str, spec: AgentSpec, context: str = "") -> str:
    return (f"QUESTION: {question}\n{('CONTEXT: ' + context) if context else ''}\n"
            f"AGENT {spec.agent_id} — variables: {json.dumps(spec.variables)}.\n"
            "Reasoning AS this agent given who they are and the situation, what is their initial position "
            "on the question resolving YES? Return ONLY JSON: {\"position\": <0..1>}. Judge from the "
            "persona and context; do not use hindsight of the actual outcome.")


@dataclass
class GenerativeForecast:
    p_outcome: float
    passes: bool
    n_agents: int
    agents: list = field(default_factory=list)       # audit: id, initial+final position, influence
    trajectory: list = field(default_factory=list)
    independent_p: float = None                       # composite mean (the sum(ps)/n baseline)
    independent_vote_share: float = None              # composite VOTE count (fraction initially > threshold)
    independent_passes: bool = None                   # composite majority-vote outcome (for contrast)

    def as_dict(self):
        return {"p_outcome": round(self.p_outcome, 4), "passes": self.passes, "n_agents": self.n_agents,
                "independent_p": round(self.independent_p, 4) if self.independent_p is not None else None,
                "independent_passes": self.independent_passes,
                "trajectory": [round(x, 3) for x in self.trajectory], "agents": self.agents}


@dataclass
class GenerativeSimulator:
    """One simulate(question) call: identify agents -> assign positions -> deliberate -> emergent outcome."""
    society: AgentSociety = field(default_factory=lambda: AgentSociety(homophily=0.5, consensus_pull=0.3, rounds=6))
    identify_fn: object = None       # callable(question, context) -> list[AgentSpec] (LLM or cached)
    position_fn: object = None       # callable(question, spec, context) -> float in [0,1] (LLM or structured)

    def simulate(self, question: str, *, context: str = "", agents: list = None,
                 threshold: float = 0.5) -> GenerativeForecast:
        specs = agents if agents is not None else self.identify_fn(question, context)
        personas = [PersonaAgent(s.agent_id, dict(s.variables), influence=s.influence,
                                 openness=s.openness, conviction=s.conviction) for s in specs]
        spec_by_id = {s.agent_id: s for s in specs}

        def pf(agent, q):
            return self.position_fn(question, spec_by_id[agent.agent_id], context)

        indep = independent_outcome(personas, pf, question, threshold)
        init = {a.agent_id: pf(a, question) for a in personas}
        sim = self.society.simulate(question, personas, pf, threshold=threshold)
        audit = [{"id": a.agent_id, "initial": round(init[a.agent_id], 3), "final": round(a.position, 3),
                  "influence": a.influence} for a in personas]
        return GenerativeForecast(p_outcome=sim["p_outcome"], passes=sim["passes"], n_agents=len(personas),
                                  agents=audit, trajectory=sim["trajectory"], independent_p=indep["p_outcome"],
                                  independent_vote_share=indep["vote_share"], independent_passes=indep["passes"])
