"""QUARANTINED LEGACY NUMERIC REALITY — explicitly NOT the production path (§NAP).

This module is the single burial ground for every ARBITRARY NUMERICAL REPRESENTATION OF SOCIAL
REALITY that used to parameterize the default World Model V2 runtime: label→number maps, unfitted
"documented priors", hand-authored action magnitudes, invented capacity/persistence constants, and
the qualitative-lean shift table. None of them was ever fitted, held-out validated, or transported;
each existed because a developer needed a convenient scalar proxy for an uncertain qualitative
concept. Production no longer imports ANY name from this module (enforced by
tests/test_numeric_reality_enforcement.py); the tables remain ONLY so explicit old-vs-new benchmark
ablations can reproduce the historical behavior under the acknowledgement token.

Access contract: call `legacy_numeric_table(name, acknowledge=ABLATION_TOKEN)`. Anything else gets
a loud PermissionError. There is no environment-variable door.

What is buried here and what replaced it in production:

  PROCESS_STATE_LEVELS      dormant/exploratory/active/advanced/imminent → 0.15…0.85. Replaced by
                            scenario-generated TYPED process records (qualitative state + concrete
                            stage-entry events); no generic progress bar exists in production.
  STANCE_ORIENTATION        stance label → signed weight −0.9…+0.9. Replaced by the stance record
                            itself conditioning the actor's OWN situated LLM cognition.
  RELIABILITY_SHRINK /      3-level label → multiplier. Reliability/capability stay QUALITATIVE
  CAPABILITY_SHRINK         facts on the record; they are never coefficients.
  CONTROL_WEIGHTS           graded-control label → multiplier. Control now determines actual
                            authority/feasibility through institutional mechanisms only.
  ENDOGENOUS_STANCE_SPLIT   0.6 residual share of a direct stance→hazard channel that no longer
                            exists in production.
  INTENTION_HR_PRIORS       stance level → hazard ratio (0.55…2.10), never fitted. There is no
                            production stance→hazard channel; a fitted, eligibility-passing pack
                            could restore one through the provenance gate (numeric_provenance).
  COUPLING_PRIORS (+clamps) pathway_step 0.04, own/cross/world weights 1.0/0.25/0.35, attrition,
                            persistence survival 0.75/0.85, contested suppression … — the entire
                            unfitted behavior→state→hazard coupling layer.
  CAPACITY_INIT / EFFORTFUL_ACTION_COST / EXHAUSTION / RIPENESS / BANDWAGON thresholds
                            normalized "capacity" psychology with no real unit.
  ACTION_PATHWAY_EFFECTS    hand-authored action → pathway magnitudes (accept +1.0, reject −0.7 …)
                            and the ≥0.5 prohibition threshold. Replaced by literal binding rules +
                            qualitative contradiction judgment (resolution_criteria) and by the
                            generated causal-boundary consequence architecture.
  STANCE_AGGREGATION_WEIGHTS  _STANCE_WEIGHT 0.95/0.75/0.4/0.6/0.85 → the `actor_intentions`
                            share quantity. Removed; stances are per-actor qualitative state.
  LEAN_SHIFT                hypothesis lean → ±0.2/±0.4 multiplicative posterior shifts. Removed;
                            competing structures stay separate models, never numeric nudges.
  LEAN_BETA                 qualitative lean → broad Beta(a,b). The table itself still lives in
                            fallback.py behind the §28 generic-prior quarantine; the EVENT-TIME
                            fallback rungs that consumed it are gone.
"""
from __future__ import annotations

ABLATION_TOKEN = "I_UNDERSTAND_THIS_IS_A_LEGACY_ABLATION_NOT_PRODUCTION"

# --------------------------------------------------------------------------------- the tables
_PROCESS_STATE_LEVELS = {"dormant": 0.15, "exploratory": 0.3, "active": 0.5,
                         "advanced": 0.7, "imminent": 0.85}

_STANCE_ORIENTATION = {"committed_to_prevent": -0.9, "conditionally_opposed": -0.55,
                       "weakly_opposed": -0.25, "neutral": 0.0, "inclined_toward": 0.35,
                       "actively_pursuing": 0.7, "formally_committed": 0.9}

_RELIABILITY_SHRINK = {"high": 1.0, "medium": 0.6, "low": 0.3}
_CAPABILITY_SHRINK = {"high": 1.0, "medium": 0.75, "low": 0.4}
_CONTROL_WEIGHTS = {"sole_authority": 1.0, "veto": 1.0, "agenda_setting": 0.75,
                    "partial_implementation": 0.6, "coalition_member": 0.5,
                    "operational_capability": 0.5, "informal_influence": 0.3, "none": 0.25}
_ENDOGENOUS_STANCE_SPLIT = 0.6

_INTENTION_HR_PRIORS = {
    "committed_to_prevent": (0.55, 0.30, 0.90),
    "conditionally_opposed": (0.78, 0.50, 1.10),
    "weakly_opposed": (0.90, 0.65, 1.15),
    "neutral": (1.00, 0.80, 1.25),
    "inclined_toward": (1.35, 1.00, 1.90),
    "actively_pursuing": (1.70, 1.15, 2.50),
    "formally_committed": (2.10, 1.30, 3.20),
}

_COUPLING_PRIORS = {
    "pathway_step":            (0.04, 0.02, 0.08),
    "endogenous_stance_split": (0.60, 0.45, 0.80),
    "own_pathway_weight":      (1.00, 0.70, 1.40),
    "cross_pathway_weight":    (0.25, 0.12, 0.50),
    "world_state_weight":      (0.35, 0.20, 0.60),
    "contested_suppression":   (0.50, 0.30, 0.80),
    "nonprincipal_step_share": (0.50, 0.30, 0.80),
    "attrition_rate_per_day":  (0.0007, 0.0004, 0.0015),
    "persistence_survival_shared":   (0.75, 0.55, 0.90),
    "persistence_survival_default":  (0.85, 0.70, 0.95),
}
_COUPLING_CLAMPS = {"pathway_step": (0.005, 0.15), "endogenous_stance_split": (0.2, 1.0),
                    "own_pathway_weight": (0.3, 2.0), "cross_pathway_weight": (0.0, 1.0),
                    "world_state_weight": (0.0, 1.0), "contested_suppression": (0.1, 1.0),
                    "nonprincipal_step_share": (0.1, 1.0), "attrition_rate_per_day": (0.0, 0.003),
                    "persistence_survival_shared": (0.2, 0.99),
                    "persistence_survival_default": (0.3, 0.99)}

_CAPACITY_INIT = {"high": 0.85, "medium": 0.6, "low": 0.35}
_EFFORTFUL_ACTION_COST = 0.02
_EXHAUSTION_THRESHOLD = 0.30
_RIPENESS_THRESHOLD = 0.70
_BANDWAGON_THRESHOLD = 0.70
_STANCE_MATERIAL_HYSTERESIS = 0.08

_ACTION_PATHWAY_EFFECTS = {
    ("negotiation", "accept"): {"cooperative_agreement": 1.0},
    ("negotiation", "concede"): {"cooperative_agreement": 0.6},
    ("negotiation", "counteroffer"): {"cooperative_agreement": 0.35},
    ("negotiation", "seek_mediator"): {"cooperative_agreement": 0.4},
    ("negotiation", "reveal"): {"cooperative_agreement": 0.15},
    ("negotiation", "delay"): {"cooperative_agreement": -0.2},
    ("negotiation", "hold_position"): {"cooperative_agreement": -0.3},
    ("negotiation", "conceal"): {"cooperative_agreement": -0.1},
    ("negotiation", "reject"): {"cooperative_agreement": -0.7},
    ("negotiation", "exit"): {"cooperative_agreement": -1.0},
    ("negotiation", "escalate"): {"cooperative_agreement": -0.5, "unilateral_action": 0.4,
                                  "competitive_interaction": 0.3},
    ("participation", "mobilize"): {"unilateral_action": 0.5, "competitive_interaction": 0.3},
    ("participation", "strike"): {"unilateral_action": 0.35},
    ("participation", "protest"): {"institutional_procedure": -0.2, "unilateral_action": 0.2},
    ("participation", "support"): {"institutional_procedure": 0.35},
    ("participation", "oppose"): {"institutional_procedure": -0.35},
    ("participation", "coordinate"): {"cooperative_agreement": 0.3},
    ("participation", "defect"): {"cooperative_agreement": -0.4},
    ("participation", "persuade"): {"cooperative_agreement": 0.2},
    ("participation", "withdraw"): {"cooperative_agreement": -0.3, "unilateral_action": -0.2},
    ("institutional", "approve"): {"institutional_procedure": 0.8},
    ("institutional", "reject"): {"institutional_procedure": -0.8},
    ("institutional", "veto"): {"institutional_procedure": -1.0},
    ("institutional", "amend"): {"institutional_procedure": 0.15},
    ("institutional", "defer"): {"institutional_procedure": -0.3},
    ("institutional", "refer"): {"institutional_procedure": 0.1},
    ("institutional", "schedule"): {"institutional_procedure": 0.3},
    ("institutional", "place_on_agenda"): {"institutional_procedure": 0.3},
    ("institutional", "enforce"): {"institutional_procedure": 0.3, "operational_execution": 0.3},
    ("institutional", "appeal"): {"institutional_procedure": -0.2},
    ("institutional", "escalate"): {"institutional_procedure": 0.2},
    ("organizational_market", "launch"): {"operational_execution": 0.9},
    ("organizational_market", "delay_launch"): {"operational_execution": -0.6},
    ("organizational_market", "authorize"): {"operational_execution": 0.5},
    ("organizational_market", "allocate_budget"): {"operational_execution": 0.3,
                                                   "resource_depletion": 0.2},
    ("organizational_market", "request_approval"): {"institutional_procedure": 0.3},
    ("organizational_market", "acquire"): {"operational_execution": 0.4,
                                           "market_aggregation": 0.2},
    ("organizational_market", "hire"): {"operational_execution": 0.2},
    ("organizational_market", "fire"): {"operational_execution": -0.2},
    ("organizational_market", "purchase"): {"market_aggregation": 0.15},
    ("organizational_market", "sell"): {"market_aggregation": -0.15},
    ("organizational_market", "withdraw_offer"): {"cooperative_agreement": -0.7},
    ("messaging", "escalate_message"): {"cooperative_agreement": -0.15},
    ("messaging", "reveal_information"): {"cooperative_agreement": 0.1},
    ("messaging", "withhold_information"): {"cooperative_agreement": -0.1},
}

_STANCE_AGGREGATION_WEIGHTS = {
    "committed_to_prevent": 0.95, "conditionally_opposed": 0.75, "weakly_opposed": 0.4,
    "neutral": 0.0, "inclined_toward": 0.6, "actively_pursuing": 0.85,
    "formally_committed": 0.95,
    "categorical_refusal": 0.95, "conditional_refusal": 0.75, "weak_opposition": 0.4,
    "openness_to_agreement": 0.6, "formal_commitment_toward_agreement": 0.95}

_LEAN_SHIFT = {"strong_no": -0.4, "weak_no": -0.2, "neutral": 0.0,
               "weak_yes": 0.2, "strong_yes": 0.4}

_TABLES = {
    "PROCESS_STATE_LEVELS": _PROCESS_STATE_LEVELS,
    "STANCE_ORIENTATION": _STANCE_ORIENTATION,
    "RELIABILITY_SHRINK": _RELIABILITY_SHRINK,
    "CAPABILITY_SHRINK": _CAPABILITY_SHRINK,
    "CONTROL_WEIGHTS": _CONTROL_WEIGHTS,
    "ENDOGENOUS_STANCE_SPLIT": _ENDOGENOUS_STANCE_SPLIT,
    "INTENTION_HR_PRIORS": _INTENTION_HR_PRIORS,
    "COUPLING_PRIORS": _COUPLING_PRIORS,
    "COUPLING_CLAMPS": _COUPLING_CLAMPS,
    "CAPACITY_INIT": _CAPACITY_INIT,
    "EFFORTFUL_ACTION_COST": _EFFORTFUL_ACTION_COST,
    "EXHAUSTION_THRESHOLD": _EXHAUSTION_THRESHOLD,
    "RIPENESS_THRESHOLD": _RIPENESS_THRESHOLD,
    "BANDWAGON_THRESHOLD": _BANDWAGON_THRESHOLD,
    "STANCE_MATERIAL_HYSTERESIS": _STANCE_MATERIAL_HYSTERESIS,
    "ACTION_PATHWAY_EFFECTS": _ACTION_PATHWAY_EFFECTS,
    "STANCE_AGGREGATION_WEIGHTS": _STANCE_AGGREGATION_WEIGHTS,
    "LEAN_SHIFT": _LEAN_SHIFT,
}


def legacy_numeric_table(name: str, *, acknowledge: str = ""):
    """The ONLY access path to a buried table. Requires the literal acknowledgement token so no
    code path can drift into using these as a quiet default; every call is an explicit statement
    that the caller is running a legacy ablation, not production."""
    if acknowledge != ABLATION_TOKEN:
        raise PermissionError(
            f"legacy numeric table {name!r} is QUARANTINED (§NAP): arbitrary numerical proxies for "
            f"social reality may not serve production. Pass acknowledge=ABLATION_TOKEN only from an "
            f"explicitly named old-vs-new ablation/benchmark.")
    if name not in _TABLES:
        raise KeyError(f"unknown legacy numeric table {name!r}; known: {sorted(_TABLES)}")
    v = _TABLES[name]
    return dict(v) if isinstance(v, dict) else v


def legacy_actions_advancing_pathway(pathway: str, *, min_effect: float = 0.5,
                                     acknowledge: str = "") -> list:
    """The historical prohibition-set derivation (ACTION_PATHWAY_EFFECTS ≥ threshold) — ablation
    only. Production binding prohibitions come from literal binding rules + qualitative
    contradiction judgment (resolution_criteria)."""
    table = legacy_numeric_table("ACTION_PATHWAY_EFFECTS", acknowledge=acknowledge)
    pw = str(pathway).strip().lower()
    out = set()
    for (_fam, name), eff in table.items():
        if pw == "any":
            if any(v >= min_effect for v in eff.values()):
                out.add(name)
        elif eff.get(pw, 0.0) >= min_effect:
            out.add(name)
    return sorted(out)
