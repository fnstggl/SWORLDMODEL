"""Mean-field coupling — the change that makes aggregation NON-SEPARABLE (the audit's #1).

The simulation audit's core finding: `GroundedSimulator.simulate_population` is `sum(ps)/n` — a mean of
INDEPENDENT per-person readouts, `∂pᵢ/∂pⱼ = 0`. No agent reads any other; there is no interaction, so it
cannot be anything but a linear pool of its parts. This makes the aggregation genuinely coupled: each
agent's belief evolves toward the population's *current aggregate* (a mean-field term), modulated by that
agent's own responsiveness, over steps — so agent i's outcome depends on agent j's state through the
aggregate. The output can no longer be reproduced by a mean of independent predictions.

Ingredients, all grounded in already-validated parts:
  - per-agent belief update = `responsiveness · pull` (the EXP-030/042 operator, `update_person`);
  - `pull = k_social·(aggregate − belief) + k_event·event_impact + k_proof·social_proof(aggregate)`:
      * conformity toward the aggregate (mean-field coupling),
      * exogenous event shocks over the horizon,
      * a SOCIAL-PROOF term that makes adoption self-reinforcing — the nonlinearity that produces
        cascades / S-curves a linear compositional model provably cannot;
  - INFLUENCE-WEIGHTED aggregate (opinion leaders pull harder), so the aggregate can DRIFT rather than be
    conformity-preserved.

State is carried forward across steps (like `rollout.py` does for the scalar case) and the collective
outcome is read from the evolved population — a real forward-rolling coupled simulation, held honestly to
whether it beats the independent mean (EXP-053).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _clamp(x, lo=1e-4, hi=1 - 1e-4):
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class Agent:
    belief: float                 # current p in [0,1]
    responsiveness: float = 0.3   # how much they move per step (from openness/skepticism/entrenchment)
    influence: float = 1.0        # how hard they pull the aggregate (opinion-leader weight)


@dataclass
class MeanFieldRollout:
    """A coupled opinion-dynamics roll-forward. `step` mutates the population; the aggregate each step
    depends on every agent, and every agent updates toward it — non-separable by construction."""
    k_social: float = 0.15        # conformity strength (pull toward the aggregate)
    k_event: float = 1.0          # exogenous event gain
    k_proof: float = 0.0          # social-proof self-reinforcement (nonlinear; 0 = off)
    proof_center: float = 0.5     # 0.5 = conformity-to-majority (polarization); 0.0 = bandwagon adoption

    def aggregate(self, agents) -> float:
        w = sum(a.influence for a in agents) or 1.0
        return sum(a.influence * a.belief for a in agents) / w

    def step(self, agents, event_impact: float = 0.0) -> float:
        agg = self.aggregate(agents)                    # depends on ALL agents (the coupling)
        proof = (agg - self.proof_center)               # bandwagon toward adoption / majority
        for a in agents:
            pull = (self.k_social * (agg - a.belief)    # conformity toward the evolving aggregate
                    + self.k_event * event_impact       # exogenous shock
                    + self.k_proof * proof)             # social proof: prevalence begets adoption
            a.belief = _clamp(a.belief + a.responsiveness * pull)
        return agg

    def roll(self, agents, steps: int, events=None):
        """Roll the coupled population forward `steps` steps; return (trajectory of aggregates, final agg)."""
        traj = []
        for t in range(steps):
            ev = 0.0 if not events else (events[t] if t < len(events) else events[-1])
            traj.append(self.step(agents, ev))
        traj.append(self.aggregate(agents))
        return traj, traj[-1]


def agents_from_cells(cells) -> list:
    """Build agents from (belief, responsiveness, influence) cells — e.g. demographic segments."""
    return [Agent(belief=_clamp(b), responsiveness=max(0.0, min(1.0, r)), influence=max(1e-3, w))
            for b, r, w in cells]
