"""AgentSociety — general agent-based social simulation (the architecture the thesis actually wants).

The audit's verdict was that the shipped pipeline is a mean of independent regressions. This is the real
thing: a population of PERSONA AGENTS — each carrying known + inferred variables (a value profile, stance,
expertise, status, openness) — who take a position on a proposition, then INTERACT over rounds
(influence-weighted, similarity-gated deliberation with an institutional consensus pull), so the outcome
EMERGES from their coupled behavior. It is general over agent types: the same machinery runs a handful of
named institutional agents (FOMC members, Supreme Court justices, a committee) or a sample of population
segments — you supply the agents and a `position_fn`; the society does the rest.

Why this is not compositing: each agent's update depends on every other agent (through influence +
similarity weighting), so `∂positionᵢ/∂positionⱼ ≠ 0`. An influential minority can flip the outcome;
ideological blocs form via homophily; deliberation can drive consensus. None of these are reachable by
averaging independent per-agent predictions — they are emergent properties of the interaction.

The `position_fn` is where the persona's variables meet the question: a structured value-match, or an LLM
roleplaying the agent given its persona + the situation (the generative-agent path). Pluggable, so the
society is general and testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class PersonaAgent:
    agent_id: str
    variables: dict = field(default_factory=dict)   # known + inferred (value profile, ideology, ...)
    position: float = 0.5                            # current stance on the proposition, [0,1]
    influence: float = 1.0                           # how hard they sway others (status/expertise/size)
    openness: float = 0.3                            # how much they update per round (from the VariableMap)
    conviction: float = 0.4                          # resistance to change (entrenchment)

    def value_vector(self) -> dict:
        return self.variables


def _similarity(a: PersonaAgent, b: PersonaAgent) -> float:
    """Homophily kernel over shared variable keys (1 = identical values, 0 = maximally distant)."""
    keys = set(a.variables) & set(b.variables)
    if not keys:
        return 0.5
    d = sum(abs(float(a.variables[k]) - float(b.variables[k])) for k in keys) / len(keys)
    return max(0.0, 1.0 - d)


@dataclass
class AgentSociety:
    """A coupled deliberation over persona agents. Positions interact; the outcome emerges."""
    homophily: float = 0.5          # 0 = influence ignores similarity; 1 = only listen to the similar (echo)
    consensus_pull: float = 0.0     # institutional pressure toward the whole-body mean (deliberative unanimity)
    confidence_bound: float = 0.0   # bounded confidence: ignore agents with similarity below this (0 = off)
    rounds: int = 6

    def set_initial_positions(self, agents, position_fn, proposition):
        for a in agents:
            a.position = _clamp(position_fn(a, proposition))

    def _deliberate_step(self, agents):
        n = len(agents)
        body_mean = sum(a.influence * a.position for a in agents) / (sum(a.influence for a in agents) or 1.0)
        new = [0.0] * n
        for i, a in enumerate(agents):
            num = den = 0.0
            for j, b in enumerate(agents):
                if i == j:
                    continue
                sim = _similarity(a, b)
                if sim < self.confidence_bound:              # bounded confidence: too dissimilar -> ignored
                    continue
                w = b.influence * ((1 - self.homophily) + self.homophily * sim)   # who a listens to
                num += w * b.position
                den += w
            social = (num / den) if den else a.position
            target = (1 - self.consensus_pull) * social + self.consensus_pull * body_mean
            step = a.openness * (1 - a.conviction) * (target - a.position)         # gated update
            new[i] = _clamp(a.position + step)
        for a, p in zip(agents, new):
            a.position = p

    def simulate(self, proposition, agents, position_fn, *, threshold: float = 0.5):
        """Run the coupled deliberation; return the emergent outcome + trajectory + final positions."""
        self.set_initial_positions(agents, position_fn, proposition)
        init_share = sum(int(a.position > threshold) for a in agents) / len(agents)
        traj = [self.outcome_share(agents, threshold)]
        for _ in range(self.rounds):
            self._deliberate_step(agents)
            traj.append(self.outcome_share(agents, threshold))
        return {"p_outcome": self.outcome_mean(agents),
                "vote_share": self.outcome_share(agents, threshold),
                "initial_vote_share": init_share,
                "passes": self.outcome_share(agents, threshold) > 0.5,
                "trajectory": traj,
                "final_positions": [round(a.position, 3) for a in agents]}

    def outcome_mean(self, agents) -> float:
        return sum(a.position for a in agents) / len(agents)

    def outcome_share(self, agents, threshold=0.5) -> float:
        return sum(int(a.position > threshold) for a in agents) / len(agents)


def independent_outcome(agents, position_fn, proposition, threshold=0.5) -> dict:
    """The COMPOSITE baseline: each agent's position from its own variables, no interaction — `sum/n`.
    This is exactly what `simulate_population` does; AgentSociety must beat it to earn the word simulate."""
    pos = [_clamp(position_fn(a, proposition)) for a in agents]
    return {"p_outcome": sum(pos) / len(pos),
            "vote_share": sum(int(p > threshold) for p in pos) / len(pos),
            "passes": (sum(int(p > threshold) for p in pos) / len(pos)) > 0.5}
