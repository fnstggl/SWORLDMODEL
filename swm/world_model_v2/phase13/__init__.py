"""Phase 13 — the universal best-action / decision layer over the canonical World Model V2 runtime.

Phase 13 does NOT fork the simulator. It compiles a typed `DecisionProblem`, generates a feasible,
authority-checked action space from world affordances, represents every action as an intervention that
enters the canonical event queue (never a direct terminal mutation), evaluates alternatives on MATCHED
posterior particles under common random numbers through the same rollout funnel (all phase operators
fire), and returns a `DecisionResult` with utility distributions, risk, regret, robustness, information
value, causal-claim labels, and abstention — or a principled abstention instead of fabricated certainty.

Public API (canonical, Part 37): `swm.world_model_v2.phase13.api`
    recommend_action(decision_problem, world_context, budget)
    evaluate_actions(decision_problem, actions, world_context, budget)
    optimize_policy(decision_problem, world_context, budget)
    value_of_information(decision_problem, candidate_observations, budget)
"""
from swm.world_model_v2.phase13.contracts import (DecisionProblem, Stakeholder, UtilitySpec,
                                                  ConstraintSpec, RiskSpec, DecisionResult,
                                                  Abstention, CAUSAL_CLAIM_LABELS)
from swm.world_model_v2.phase13.ontology import (ActionSchema, register_operation, operation_registered,
                                                 OPERATION_FAMILIES)
from swm.world_model_v2.phase13.api import (recommend_action, evaluate_actions, optimize_policy,
                                            value_of_information)

__all__ = ["DecisionProblem", "Stakeholder", "UtilitySpec", "ConstraintSpec", "RiskSpec",
           "DecisionResult", "Abstention", "CAUSAL_CLAIM_LABELS", "ActionSchema",
           "register_operation", "operation_registered", "OPERATION_FAMILIES",
           "recommend_action", "evaluate_actions", "optimize_policy", "value_of_information"]
