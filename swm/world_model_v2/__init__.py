"""WorldModelV2 — a probabilistic, hybrid, event-driven social world model. THE WORLD IS NOT THE LLM.

Built in parallel to the v1 engine (swm/engine — preserved as baselines); NOT wired into production until the
acceptance suite (tests/test_world_model_v2.py) passes and a reference world beats the fair grounded baseline
on held-out outcomes.

The constitution (enforced by the anti-cheating tests):
 1. The world exists as EXECUTABLE TYPED STATE outside any LLM context (state.py) — beliefs, resources,
    relationships, commitments, information sets are data, not prose.
 2. Every transition produces a MACHINE-READABLE StateDelta (transitions.py); prose is never the only output.
 3. Time is REAL CALENDAR TIME via an event queue (events.py) — no "round 1/round 2".
 4. Hidden state is a DISTRIBUTION, sampled into coherent particles (init_state.py) — never a fabricated
    certainty, never an arbitrary LLM coefficient.
 5. Different actors have DIFFERENT information sets (information.py) — no universal dossier.
 6. Institutional rules are EXECUTABLE and can reject invalid actions (institutions.py).
 7. The LLM proposes (mechanisms, decompositions, hypotheses, action choices among TYPED options); the
    compiler validates; the executor runs — discovery, parameterization and execution are separated.
 8. The answer is READ FROM TERMINAL STATES (rollout.py) — never asked of an LLM after the fact.
 9. Counterfactuals run on CLONED matched worlds with shared exogenous randomness (rollout.py).
10. Every state field carries PROVENANCE (source, status, confidence, method, timestamp).
11. Human cognition never runs through arbitrary-variable logistic/ODE code; numerical mechanisms are
    fitted, mechanistic, or explicitly uncertain (mechanisms.py registry).
12. Generality = universal ontology + validated mechanism registry + scenario compilation — never a
    per-question free-form role-play prompt, never a new top-level scenario branch.
"""
from swm.world_model_v2.state import (Provenance, StateField, Entity, WorldState, WorldBranch,
                                      WorldTrajectory, SimulationClock, register_entity_extension)
from swm.world_model_v2.events import Event, ScheduledEvent, StochasticHazard, EventQueue
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)
