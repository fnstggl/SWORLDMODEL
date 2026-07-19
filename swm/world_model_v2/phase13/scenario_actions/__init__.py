"""Scenario-generated action layer — the production generated-mode decision architecture.

The fixed Phase 13 operation catalog (`ontology._OPERATIONS`, nine families, verb-keyed
interventions) is LEGACY, reachable only through an explicit baseline/ablation request. In
generated mode, every candidate action is:

  scenario world  →  generated ScenarioActionLanguage (what THIS decision-maker can attempt here)
                 →  ConcreteAction / ConcretePlan (exact content, targets, terms, timing — no verb labels)
                 →  deterministic feasibility + authority validation (per world hypothesis)
                 →  compiled ONCE into scenario-native direct effects (generated-world kernel ops)
                 →  executed through the canonical runtime (control plane routes observations;
                     affected actors react through their own simulations)
                 →  matched counterfactual rollouts → goal-contract evaluation → diagnosis →
                     revision → robust recommendation, Pareto set, or principled abstention.

No module in this package may import the legacy operation registry; an AST enforcement test
rejects any new global verb catalog. Missing generated semantics fail loudly or classify the
run structurally under-modeled — they never route back through fixed-v1 verbs.
"""
from swm.world_model_v2.phase13.scenario_actions.language import (  # noqa: F401
    ActionLanguageGenerator, ScenarioActionLanguage, validate_action_language,
)
from swm.world_model_v2.phase13.scenario_actions.candidates import (  # noqa: F401
    ConcreteAction, PlanStep, merge_equivalent,
)
from swm.world_model_v2.phase13.scenario_actions.goals import (  # noqa: F401
    GoalContract, GoalContractGenerator, evaluate_goal_on_arm,
)
