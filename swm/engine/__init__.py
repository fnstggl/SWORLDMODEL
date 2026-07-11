"""swm.engine — the grounded-agent world engine (the ONE mechanism).

THE CONSTITUTION (do not regress — these are the failures this package exists to kill):
  1. NO variables-with-weights. Never decompose a question into abstract latent scalars and press them
     through a logistic with LLM-invented coefficients. The LLM is asked "who are the real actors and how
     would THIS person react given who they are" — never "emit an elasticity of 0.4 ± 0.25".
  2. NO grounding theater. A fact is grounded only if it came from a real, dated passage with provenance.
     A value we could not ground is reported MISSING and, if it is a deciding fact, the engine ABSTAINS
     loudly. Never a neutral prior dressed up with source="retrieval".
  3. NATIVE answer types. "Who wins" returns a distribution over NAMED candidates. "Best headline" returns
     ACTUAL ranked headline texts. "Will X reply to this email" returns a scenario-specific p for THAT
     message and THAT person. The simulation's state space IS the answer space.
  4. THE AGENT'S COGNITION IS REASONING, NOT A SCALAR. Each round, each agent is the LLM reasoning from
     that agent's grounded dossier + what just happened publicly — not a position float nudged by invented
     ODE constants (homophily/consensus_pull are gone).
  5. REAL CALENDAR TIME. Rounds are dated; one week of rollout is one week of plausible change. The
     simulation stops at the real resolution date.
  6. GRADE-OR-ABSTAIN. A distribution ships as confident ONLY if its question-class carries a backtested
     calibration grade (swm/engine/calibrate.py). No ungraded simulation ever ships as a confident number
     — "no ungraded logistics" generalized.
  7. NEVER the base rate. An individual-response prediction is always conditioned on WHO the person is and
     THE exact message; if we cannot ground who they are, we abstain — we do not return the global reply rate.

Pipeline (the vision, clause by clause):
  ground the situation (retrieval.py + grounding.py)  →  cast the agents (casting.py)  →
  roll the society forward under uncertainty (agents.py + society.py / individual.py)  →
  calibrated native-typed distribution + best action (outcome.py + calibrate.py + actions.py),
  all behind one call: front_door.AgentWorldModel.simulate(question).
"""
from swm.engine.front_door import AgentWorldModel, agent_world_model, hybrid_world_model

__all__ = ["AgentWorldModel", "agent_world_model", "hybrid_world_model"]
