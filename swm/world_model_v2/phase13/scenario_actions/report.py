"""The §17 result contract — everything a caller needs to trust or challenge the outcome.

Human-facing summary stays small and honest (no fake precision); the complete machine-
readable detail rides in `full`. Support classifications are separate, never conflated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ScenarioDecisionReport:
    decision_id: str
    decision_maker: str = ""
    stated_goal: str = ""
    horizon: str = ""
    constraints: list = field(default_factory=list)
    risk_posture: dict = field(default_factory=dict)
    world_support: str = "generated_scenario_world"
    action_language_summary: dict = field(default_factory=dict)
    goal_contract: dict = field(default_factory=dict)
    user_supplied_actions: list = field(default_factory=list)
    generated_strategy_classes: list = field(default_factory=list)
    candidates: list = field(default_factory=list)            # full ConcreteAction dicts
    candidate_ancestry: dict = field(default_factory=dict)
    feasibility: dict = field(default_factory=dict)
    unresolved_semantics: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    compiled_effects: dict = field(default_factory=dict)      # candidate -> kernel ops
    simulation_coverage: dict = field(default_factory=dict)
    matched_world_proof: dict = field(default_factory=dict)   # CRN manifest
    trajectory_summaries: dict = field(default_factory=dict)
    evaluations: dict = field(default_factory=dict)
    revisions: list = field(default_factory=list)
    finalist_reasons: dict = field(default_factory=dict)      # why each finalist wins/loses
    pareto: list = field(default_factory=list)
    recommended: str = None
    recommendation_kind: str = "action"
    recommended_implementation: dict = field(default_factory=dict)  # the exact plan
    assumptions: list = field(default_factory=list)
    reversal_conditions: list = field(default_factory=list)
    highest_value_information: str = ""
    finalists_distinguishable: bool = False
    human_approval_required: bool = True
    causal_claim: str = "simulated_mechanism_counterfactual"
    real_world_validation: str = "not_validated_on_this_decision"
    diversity: dict = field(default_factory=dict)
    critic_findings: list = field(default_factory=list)
    missing_preferences: list = field(default_factory=list)
    stop_reason: str = ""
    trace_summary: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def human_summary(self) -> dict:
        """What a person reads: the recommendation (or honest non-recommendation), its exact
        implementation, why, what would reverse it, and what is unresolved — counted
        evidence, no invented percentages of anything."""
        top = None
        for c in self.candidates:
            if c.get("candidate_id") == self.recommended:
                top = c
        return {
            "decision": self.decision_id,
            "recommendation_kind": self.recommendation_kind,
            "recommended": self.recommended,
            "recommended_plan": ({"title": top.get("title"),
                                  "steps": [{"intent": s.get("intent"),
                                             "targets": s.get("target_ids"),
                                             "content": (s.get("exact_content") or "")[:300],
                                             "timing_ts": s.get("timing_ts")}
                                            for s in top.get("steps", [])]}
                                 if top else None),
            "why": self.finalist_reasons.get(self.recommended or "", ""),
            "support": ("best-supported among the considered feasible actions under the "
                        "stated goal, constraints, world hypotheses, and simulation support "
                        "— not 'objectively best'"),
            "finalists_distinguishable": self.finalists_distinguishable,
            "pareto_set": self.pareto if self.recommendation_kind == "pareto" else [],
            "missing_preferences": self.missing_preferences,
            "reversal_conditions": self.reversal_conditions[:4],
            "highest_value_information": self.highest_value_information,
            "unresolved": self.unresolved_semantics[:6],
            "human_approval_required": self.human_approval_required,
            "causal_claim": self.causal_claim,
            "real_world_validation": self.real_world_validation,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), default=_default, indent=1)


def _default(o):
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if callable(o):
        return f"<fn:{getattr(o, '__name__', 'lambda')}>"
    return str(o)


def build_report(*, problem, goal, language, planner_out, search_report, goal_text,
                 user_candidates, trace, recommended, recommendation_kind,
                 finalist_reasons, reversal_conditions, distinguishable) -> ScenarioDecisionReport:
    cands_by_id = {}
    for c in planner_out.candidates:
        cands_by_id[c.candidate_id] = c
    rep = ScenarioDecisionReport(
        decision_id=problem.decision_id, decision_maker=problem.decision_maker,
        stated_goal=goal_text, horizon=problem.horizon,
        constraints=[getattr(c, "description", str(c))[:120]
                     for c in (problem.constraints or [])],
        risk_posture={"tolerance": problem.risk.tolerance,
                      "robustness": problem.risk.robustness},
        action_language_summary=language.summary(),
        goal_contract=goal.as_dict(),
        user_supplied_actions=[{"candidate_id": c.candidate_id,
                                "original_text": c.original_text}
                               for c in (user_candidates or [])],
        generated_strategy_classes=sorted({c.strategy_class for c in planner_out.candidates
                                           if c.strategy_class}),
        candidates=[c.as_dict() for c in planner_out.candidates],
        candidate_ancestry={c.candidate_id: {"source": c.source, "parents": c.parent_ids,
                                             "revision_reason": c.revision_reason}
                            for c in planner_out.candidates},
        feasibility={c.candidate_id: c.provenance.get("feasibility", {})
                     for c in planner_out.candidates},
        unresolved_semantics=[{"candidate_id": c.candidate_id, "unresolved": c.unresolved}
                              for c in planner_out.candidates if c.unresolved],
        rejected=search_report.screened_out,
        compiled_effects={c.candidate_id: [{"step": s.step_id, "ops": s.compiled_ops}
                                           for s in c.steps]
                          for c in planner_out.candidates if c.steps},
        simulation_coverage=search_report.coverage,
        trajectory_summaries={k: (v.as_dict() if hasattr(v, "as_dict") else v)
                              for k, v in search_report.diagnoses.items()},
        evaluations={k: {kk: vv for kk, vv in v.items() if kk != "per_particle"}
                     for k, v in search_report.evaluations.items()},
        revisions=search_report.revisions,
        finalist_reasons=finalist_reasons,
        pareto=search_report.pareto,
        recommended=recommended, recommendation_kind=recommendation_kind,
        recommended_implementation=(cands_by_id[recommended].as_dict()
                                    if recommended in cands_by_id else {}),
        assumptions=sorted({a for c in planner_out.candidates for a in c.assumptions})[:10],
        reversal_conditions=reversal_conditions,
        highest_value_information=search_report.adjudication.get(
            "highest_value_information", ""),
        finalists_distinguishable=distinguishable,
        human_approval_required=bool(problem.human_approval_required),
        diversity=planner_out.diversity,
        critic_findings=planner_out.critic_findings,
        missing_preferences=list(goal.missing_preferences)
        + list(goal.unresolved_tradeoffs),
        stop_reason=search_report.stop_reason,
        trace_summary={"n_llm_calls": trace.n_calls(), "by_role": trace.by_role()},
        provenance={"language_hash": language.language_hash(), "goal_hash": goal.goal_hash(),
                    "schema_id": language.schema_id,
                    "schema_version": language.schema_version})
    return rep
