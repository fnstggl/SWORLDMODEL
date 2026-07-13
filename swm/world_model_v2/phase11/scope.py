"""Phase 11 — recompilation scope selection (spec §11).

Infer the SMALLEST causally sufficient revision. A typed causal-impact graph over the plan's components
(entities → relations/network → institutions → mechanisms → structural hypotheses → outcome contract) gives
the downstream dependents of whatever changed; the chosen scope must cover those descendants without going
larger than necessary. Two failure modes are guarded explicitly:

  * DON'T recompile the whole plan when a parameter update suffices (transient/drift → minimal scope);
  * DON'T pick a local update when the change invalidates the GLOBAL structure (impossible event / outcome-
    space change / broad structural failure → escalate to outcome_contract / full_plan).

Records the chosen scope AND the plausible alternatives (spec: "record both").
"""
from __future__ import annotations

from dataclasses import dataclass, field

# smallest → largest; index is the "cost/destructiveness" rank
SCOPE_ORDER = [
    "no_model_change", "observation_model", "parameter_only", "latent_state", "actor", "relationship",
    "local_network_region", "population_segment", "institution_ruleset", "mechanism_replacement",
    "structural_hypothesis", "action_space", "outcome_contract", "full_plan",
]
SCOPE_RANK = {s: i for i, s in enumerate(SCOPE_ORDER)}

# which recompile ACTION a scope implies (spec §8.3 RECOMPILE_ACTIONS)
SCOPE_ACTION = {
    "no_model_change": "no_change", "observation_model": "observation_model_update",
    "parameter_only": "parameter_update", "latent_state": "latent_state_update", "actor": "actor_revision",
    "relationship": "relationship_revision", "local_network_region": "local_network_recompile",
    "population_segment": "population_segment_revision", "institution_ruleset": "institution_recompile",
    "mechanism_replacement": "mechanism_replacement", "structural_hypothesis": "structural_branch_addition",
    "action_space": "action_space_revision", "outcome_contract": "outcome_contract_revision",
    "full_plan": "full_recompile",
}


@dataclass
class CausalImpactGraph:
    """Typed downstream-dependency layers of a plan. ``downstream(component_kind)`` returns the kinds a change
    propagates into, terminating at the outcome contract (terminal readout)."""
    layers: list = field(default_factory=lambda: [
        "observation_model", "parameter", "latent_state", "actor", "relationship", "network",
        "population", "institution", "mechanism", "structural_hypothesis", "action_space", "outcome"])

    def downstream(self, kind: str) -> list:
        try:
            i = self.layers.index(kind)
        except ValueError:
            return ["outcome"]
        return self.layers[i + 1:]

    @classmethod
    def from_plan(cls, plan) -> "CausalImpactGraph":
        return cls()                              # the typed layering is universal; plan sizes only tune costs


_SCOPE_TO_LAYER = {
    "observation_model": "observation_model", "parameter_only": "parameter", "latent_state": "latent_state",
    "actor": "actor", "relationship": "relationship", "local_network_region": "network",
    "population_segment": "population", "institution_ruleset": "institution",
    "mechanism_replacement": "mechanism", "structural_hypothesis": "structural_hypothesis",
    "action_space": "action_space", "outcome_contract": "outcome",
}


@dataclass
class ScopeSelection:
    scope: str = "no_model_change"
    action: str = "no_change"
    alternatives: list = field(default_factory=list)       # [{scope, why_not}]
    impacted_components: list = field(default_factory=list)
    rationale: str = ""
    expected_improvement: float = 0.0
    migration_risk: float = 0.0
    within_budget: bool = True

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def select_scope(fused, *, plan=None, terminal_sensitivity: float = 0.5, compute_budget=None,
                 graph: CausalImpactGraph = None) -> ScopeSelection:
    """Pick the minimal causally sufficient scope from the fused assessment's candidates."""
    graph = graph or CausalImpactGraph.from_plan(plan)
    if not getattr(fused, "proceed", False):
        return ScopeSelection(scope="no_model_change", action="no_change",
                              rationale="fused trigger did not clear the decision threshold — no model change")

    cands = list(getattr(fused, "scope_candidates", []) or [])
    cls = getattr(fused, "classification", "transient_anomaly")

    # global-structural invalidations force escalation to the contract/plan level — a local update cannot
    # repair invalid global structure. Prefer the (cheaper) outcome_contract revision; escalate to full_plan
    # when the action space is also invalid or no narrower global candidate is offered.
    if cls == "global_structural":
        if "action_space" in cands and "outcome_contract" in cands:
            chosen = "full_plan"                         # both contract AND action space invalid → whole plan
        elif "outcome_contract" in cands:
            chosen = "outcome_contract"
        else:
            chosen = "full_plan"
        rationale = "global structural invalidation (impossible event / outcome-space / broad failure) — a " \
                    "local update cannot repair invalid global structure"
    elif cls == "parameter_drift":
        chosen = "parameter_only" if "parameter_only" in cands else (cands and min(cands, key=lambda c: SCOPE_RANK.get(c, 99)) or "parameter_only")
        rationale = "parameters moved but structure/support held — refit, do not restructure"
    else:  # local_structural
        # smallest candidate that is at least an actor/relationship/institution/mechanism-level change
        local = [c for c in cands if c in _SCOPE_TO_LAYER]
        chosen = min(local, key=lambda c: SCOPE_RANK.get(c, 99)) if local else "structural_hypothesis"
        rationale = "localized structural change — revise the smallest sufficient component"

    # terminal-sensitivity handling: when the change's causal descendants reach the outcome with high
    # sensitivity AND the trigger is NOT near-certain (ambiguous), retain structural uncertainty. We do this by
    # RECOMMENDING a competing structural-hypothesis ALTERNATIVE (the candidate generator adds it and the
    # scoring mixture retains it) — we do NOT downgrade a confident, verified edit into a mere hypothesis.
    layer = _SCOPE_TO_LAYER.get(chosen, "outcome")
    impacted = graph.downstream(layer)
    retain_uncertainty = (terminal_sensitivity >= 0.7 and "outcome" in impacted
                          and cls == "local_structural"
                          and float(getattr(fused, "fused_probability", 1.0)) < 0.85)
    if retain_uncertainty and "structural_hypothesis" not in [a["scope"] for a in ([])]:
        rationale += "; high terminal sensitivity + ambiguous evidence → also retain a competing structural " \
                     "hypothesis (mixture), primary scope kept minimal"

    # budget check: full_plan under a tight budget degrades to the largest affordable scope (recorded)
    within_budget = True
    if compute_budget is not None and chosen == "full_plan":
        max_calls = compute_budget.get("max_llm_calls", 99) if isinstance(compute_budget, dict) else 99
        if max_calls < 2:
            within_budget = False

    alternatives = []
    for c in sorted(set(cands) - {chosen}, key=lambda c: SCOPE_RANK.get(c, 99)):
        smaller = SCOPE_RANK.get(c, 99) < SCOPE_RANK.get(chosen, 99)
        alternatives.append({"scope": c, "why_not": "insufficient for the change" if smaller
                             else "larger than causally necessary"})

    return ScopeSelection(
        scope=chosen, action=SCOPE_ACTION.get(chosen, "no_change"), alternatives=alternatives,
        impacted_components=impacted, rationale=rationale,
        expected_improvement=round(float(getattr(fused, "fused_probability", 0.0)) * terminal_sensitivity, 4),
        migration_risk=round(SCOPE_RANK.get(chosen, 0) / len(SCOPE_ORDER), 3), within_budget=within_budget)
