"""Phase 13 canonical public API (Part 37) — one interface, one semantics, no parallel paths.

    recommend_action(problem, world_context, budget)     — full pipeline → DecisionResult
    evaluate_actions(problem, actions, world_context, …) — evaluate SUPPLIED actions only
    optimize_policy(problem, policies, world_context, …) — sequential policies through the same funnel
    value_of_information(problem, candidate_observations, world_context, …)

`world_context` is either a compiled `WorldExecutionPlan` (the canonical runtime path — all phase
operators the plan names fire in every rollout) or a dict of raw runtime pieces
(initial/queue_builder/operators/contract) for controlled tasks. Simulation, recommendation, approval
and execution are SEPARATE: this API only simulates and recommends; nothing here executes an action in
the real world, and `human_approval_required` is stamped on every result (Part 23)."""
from __future__ import annotations

import time as _time

from swm.world_model_v2.phase13.abstain import abstention_check
from swm.world_model_v2.phase13.affordances import generate_actions
from swm.world_model_v2.phase13.contracts import DecisionProblem, DecisionResult
from swm.world_model_v2.phase13.counterfactual import MatchedEvaluator
from swm.world_model_v2.phase13.feasibility import FeasibilityEngine
from swm.world_model_v2.phase13.ontology import do_nothing
from swm.world_model_v2.phase13.robust import evaluate_bundle
from swm.world_model_v2.phase13.search import SearchBudget, select_and_run
from swm.world_model_v2.phase13.utility import evaluate_utility, pareto_frontier
from swm.world_model_v2.phase13.voi import information_report


class SingleModelContextError(TypeError):
    """A bare single WorldExecutionPlan was passed where the structural ensemble is the default.
    Phase 13 evaluates every serious action ACROSS the surviving structural models; comparing actions
    inside only one causal model requires the explicit `allow_single_structural_model=True` ablation."""


def _evaluator(world_context, *, n_particles=None, seed=0, llm=None,
               allow_single_structural_model=False) -> MatchedEvaluator:
    from swm.world_model_v2.compiler import WorldExecutionPlan
    if isinstance(world_context, WorldExecutionPlan):
        if not allow_single_structural_model:
            raise SingleModelContextError(
                "Phase 13 evaluates actions across the structural ensemble by default. Pass the "
                "ensemble (a StructuralModelEnsemble, the SimulationResult from the default runtime, "
                "or {model_id: plan}) — or set allow_single_structural_model=True for the explicit "
                "single-model ablation.")
        return MatchedEvaluator.from_plan(world_context, llm=llm,
                                          n_particles=n_particles, seed=seed)
    if isinstance(world_context, dict) and "initial" in world_context:
        # the CONTROLLED-BENCHMARK context: synthetic runtime pieces with known ground truth — a test
        # harness, not a compiled world model, so the structural-ensemble default does not apply.
        return MatchedEvaluator(initial=world_context["initial"],
                                queue_builder=world_context["queue_builder"],
                                operators=list(world_context.get("operators", [])),
                                contract=world_context.get("contract"),
                                n_particles=n_particles or world_context.get("n_particles", 60),
                                seed=seed, hypotheses=world_context.get("hypotheses"),
                                max_events=world_context.get("max_events", 500))
    raise TypeError("world_context must be a structural ensemble, a WorldExecutionPlan (explicit "
                    "single-model ablation only), or a dict of runtime pieces "
                    "(initial/queue_builder/operators/contract) — Phase 13 never forks the runtime")


def _ensemble_context(world_context) -> dict:
    """{model_id: {plan, meta}} when the context is ensemble-shaped, else {}."""
    from swm.world_model_v2.phase13.ensemble import extract_ensemble_models
    return extract_ensemble_models(world_context)


def _guard_single_plan(world_context, allow_single_structural_model: bool) -> None:
    """The structural-ensemble default applies to the generated route too: a BARE single plan
    (even a generated-world one) is the explicit single-model ablation. The ensemble's own
    per-model re-entry passes allow_single_structural_model=True."""
    from swm.world_model_v2.compiler import WorldExecutionPlan
    if isinstance(world_context, WorldExecutionPlan) and not allow_single_structural_model:
        raise SingleModelContextError(
            "Phase 13 evaluates actions across the structural ensemble by default. Pass the "
            "ensemble (a StructuralModelEnsemble, the SimulationResult from the default runtime, "
            "or {model_id: plan}) — or set allow_single_structural_model=True for the explicit "
            "single-model ablation.")


def _fingerprint() -> dict:
    try:
        from swm.world_model_v2.runtime_fingerprint import runtime_fingerprint
        fp = runtime_fingerprint()
        return {"fingerprint_hash": fp.get("fingerprint_hash", ""), "phase13": "phase13-1.0"}
    except Exception:  # noqa: BLE001
        return {"phase13": "phase13-1.0"}


def recommend_action(problem: DecisionProblem, world_context, *, budget: str = "standard",
                     seed: int = 0, n_particles: int = None, llm=None,
                     candidate_observations: list = None, mode: str = "auto",
                     goal_text: str = "", trace_path: str = "",
                     forensic_dir: str = "",
                     allow_single_structural_model: bool = False) -> DecisionResult:
    """The full Phase-13 pipeline. Abstains (with what is needed) instead of fabricating certainty.

    STRUCTURAL-ENSEMBLE DEFAULT: an ensemble-shaped `world_context` (StructuralModelEnsemble /
    default-runtime SimulationResult / {model_id: plan}) evaluates every action across ALL surviving
    structural models with per-model budgets and cross-model robustness (see phase13.ensemble); each
    per-model evaluation re-enters this function with that model's own context, so a generated-world
    model routes through the scenario-generated action layer INSIDE its model. A bare single plan is
    the explicit ablation and requires `allow_single_structural_model=True`.

    mode="auto" (default): a generated scenario world routes through the scenario-generated
    action layer (scenario_actions.api) — goal-backward discovery, scenario action language,
    kernel-compiled candidates, matched simulation, diagnosis-driven revision. The fixed-v1
    operation catalog runs ONLY for non-generated world contexts (controlled benchmarks) or
    an explicit mode="legacy_fixed_v1" baseline/ablation request; generated worlds can never
    silently fall back to it."""
    from swm.world_model_v2.phase13.scenario_actions.api import (discover_best_action,
                                                                 is_generated_context)
    if mode not in ("auto", "legacy_fixed_v1"):
        raise ValueError(f"unknown mode {mode!r} (valid: auto | legacy_fixed_v1)")
    models = _ensemble_context(world_context)
    if models:
        from swm.world_model_v2.phase13.ensemble import recommend_action_across_models
        return recommend_action_across_models(problem, models, budget=budget, seed=seed,
                                              n_particles=n_particles, llm=llm,
                                              candidate_observations=candidate_observations,
                                              mode=mode, goal_text=goal_text)
    if mode == "auto" and is_generated_context(world_context):
        _guard_single_plan(world_context, allow_single_structural_model)
        return discover_best_action(goal_text or problem.context, world_context,
                                    problem=problem, budget=budget, seed=seed,
                                    n_particles=n_particles, llm=llm,
                                    trace_path=trace_path, forensic_dir=forensic_dir)
    if mode == "legacy_fixed_v1" and is_generated_context(world_context):
        # reachable ONLY by explicit request — stamped as a baseline, never a default
        pass
    t0 = _time.time()
    res = DecisionResult(decision_id=problem.decision_id, contract_hash=problem.contract_hash(),
                        runtime_fingerprint=_fingerprint(), seed=seed)

    # 1. contract validation + underspecification (Part 2)
    defects = problem.validate()
    missing = problem.underspecification()
    ev = _evaluator(world_context, n_particles=n_particles, seed=seed, llm=llm,
                    allow_single_structural_model=allow_single_structural_model)
    base_world = ev.particles()[0] if ev.n_particles else None

    # 2. action space: affordances + user candidates + baselines (Part 5)
    gen = generate_actions(base_world, problem, llm=llm)
    res.provenance["action_generation"] = gen.as_dict()

    # 3. feasibility with typed rejections (Part 6) — infeasible actions are NEVER simulated
    fe = FeasibilityEngine()
    verdicts = fe.check_bundle(base_world, gen.candidates, problem)
    res.feasibility = [v.as_dict() for v in verdicts]
    feasible = [a for a, v in zip(gen.candidates, verdicts) if v.feasible]
    if not any(a.action_id == "do_nothing" for a in feasible):
        feasible.append(do_nothing(problem.decision_maker))

    # 4. abstention gates that need no simulation (Part 23)
    ab = abstention_check(problem, defects=defects, missing=missing, feasible=feasible)
    if ab is not None and not ab.partial.get("continue_for_pareto"):
        res.abstention = ab.as_dict()
        res.recommendation_kind = "abstain"
        res.latency_s = round(_time.time() - t0, 3)
        return res

    # 5. matched counterfactual search/evaluation through the canonical funnel (Parts 8–10, 17–18)
    bundle, diag = select_and_run(ev, feasible, problem, budget=SearchBudget.tiered(budget))
    res.search = diag.as_dict()
    res.reference_action = bundle.reference
    res.provenance["crn_manifest"] = bundle.crn_manifest
    res.provenance["hypothesis_assignment"] = sorted(set(bundle.hypothesis_assignment))
    res.active_phases = _active_phases(bundle)
    # the causal-boundary report for decision actions: attempts vs deliveries, mechanisms
    # invoked/succeeded/failed/unresolved, rejected directness claims — same contract as
    # ordinary forecasts (§causal truth boundary)
    for op in ev.operators:
        if getattr(op, "name", "") == "decision_action" and getattr(op, "report", None):
            res.provenance["causal_consequence_report"] = {
                k: v for k, v in op.report.items() if k != "causal_action_reports"}
            res.provenance["causal_action_reports"] = \
                list(op.report.get("causal_action_reports") or [])[:12]
            break

    # 6. robust evaluation (Parts 15–16)
    evals = evaluate_bundle(bundle, feasible, problem)
    res.evaluated = [v for k, v in evals.items() if not k.startswith("_")]
    res.counterfactual = {"reference": bundle.reference,
                          "paired": {e["action_id"]: e.get("paired_vs_reference", {})
                                     for e in res.evaluated}}

    # 7. Pareto frontier (always computed for multi-stakeholder contracts)
    if len(problem.utility.stakeholders) > 1:
        bds = [evaluate_utility(aid, bundle.arms[aid].outcomes, problem.utility)
               for aid in bundle.arms]
        res.pareto_frontier = pareto_frontier(bds)

    # 8. value of information (Part 14) — from the SAME matched utility matrix
    agg = {e["action_id"]: None for e in res.evaluated}
    agg = _agg_matrix(bundle, feasible, problem)
    if problem.information_gathering_allowed:
        res.value_of_information = information_report(agg, candidate_observations or [])

    # 9. recommendation or principled non-selection
    ranking = evals["_ranking"]["order"]
    viable = [r for r in ranking if r["score"] != "excluded"]
    if ab is not None:                                     # underspecified utility → Pareto only
        res.abstention = ab.as_dict()
        res.recommendation_kind = "pareto"
        res.recommended = None
    elif res.value_of_information.get("recommend_gathering"):
        res.recommendation_kind = "gather_information"
        res.recommended = "gather_information"
    elif viable:
        res.recommended = viable[0]["action_id"]
        res.recommendation_kind = "action"
    else:
        from swm.world_model_v2.phase13.contracts import Abstention
        res.abstention = Abstention(reasons=[{"code": "no_viable_action",
                                              "detail": "every action violates rights/floors/chance "
                                                        "constraints"}],
                                    needed=["relax a constraint or supply new actions"]).as_dict()
        res.recommendation_kind = "abstain"
    res.provenance["ranking"] = evals["_ranking"]
    res.provenance["minimax_regret_action"] = evals["_regret"]["minimax_regret_action"]
    res.causal_claim = "simulated_mechanism_counterfactual"
    res.support_grade = "exploratory"
    res.cost = {"arm_evaluations": res.search.get("n_evaluated", 0),
                "rollouts": res.search.get("n_evaluated", 0) * ev.n_particles}
    res.latency_s = round(_time.time() - t0, 3)
    return res


def evaluate_actions(problem: DecisionProblem, actions: list, world_context, *,
                     budget: str = "standard", seed: int = 0, n_particles: int = None,
                     llm=None, mode: str = "auto", goal_text: str = "",
                     trace_path: str = "", forensic_dir: str = "",
                     allow_single_structural_model: bool = False) -> DecisionResult:
    """Evaluate ONLY the supplied actions (plus the mandatory do-nothing reference).
    Differs from recommend_action ONLY in where candidates come from. The caller's
    DecisionProblem is never mutated.

    Ensemble-shaped contexts evaluate the SAME actions inside every structural model (per-model
    budgets, cross-model robustness); a bare single plan requires the explicit ablation flag."""
    from swm.world_model_v2.phase13.scenario_actions.api import (evaluate_actions_generated,
                                                                 is_generated_context)
    if mode not in ("auto", "legacy_fixed_v1"):
        raise ValueError(f"unknown mode {mode!r} (valid: auto | legacy_fixed_v1)")
    models = _ensemble_context(world_context)
    if models:
        from swm.world_model_v2.phase13.ensemble import recommend_action_across_models
        return recommend_action_across_models(problem, models, budget=budget, seed=seed,
                                              n_particles=n_particles, llm=llm,
                                              actions=list(actions), mode=mode,
                                              goal_text=goal_text)
    if mode == "auto" and is_generated_context(world_context):
        _guard_single_plan(world_context, allow_single_structural_model)
        return evaluate_actions_generated(problem, actions, world_context,
                                          goal_text=goal_text or problem.context,
                                          budget=budget, seed=seed,
                                          n_particles=n_particles, llm=llm,
                                          trace_path=trace_path,
                                          forensic_dir=forensic_dir)
    import copy as _copy
    p2 = _copy.copy(problem)                            # NEVER mutate the caller's contract
    p2.candidate_actions = list(actions)
    p2.generated_action_permission = False
    return recommend_action(p2, world_context, budget=budget, seed=seed,
                            n_particles=n_particles, llm=llm, mode=mode,
                            allow_single_structural_model=allow_single_structural_model)


def optimize_policy(problem: DecisionProblem, policies: list, world_context, *,
                    seed: int = 0, n_particles: int = None, llm=None, mode: str = "auto",
                    goal_text: str = "", trace_path: str = "", forensic_dir: str = "",
                    allow_single_structural_model: bool = False) -> DecisionResult:
    """Sequential policies (Part 12): each Policy rolls through the SAME matched particles with
    decision points scheduled; a do-nothing policy is the reference. Policies act on belief state only
    (policies.belief_state — the canonical observable-view boundary)."""
    from swm.world_model_v2.phase13.scenario_actions.api import (is_generated_context,
                                                                 optimize_policy_generated)
    if mode not in ("auto", "legacy_fixed_v1"):
        raise ValueError(f"unknown mode {mode!r} (valid: auto | legacy_fixed_v1)")
    models = _ensemble_context(world_context)
    if models:
        from swm.world_model_v2.phase13.ensemble import optimize_policy_across_models
        return optimize_policy_across_models(problem, policies, models, seed=seed,
                                             n_particles=n_particles, llm=llm, mode=mode,
                                             goal_text=goal_text or problem.context)
    if mode == "auto" and is_generated_context(world_context):
        # contingent/sequential plans on a generated world route through the scenario layer:
        # ConcreteActions with observation-gated conditional steps, executed on the maker's
        # observable projection through the same canonical funnel
        _guard_single_plan(world_context, allow_single_structural_model)
        return optimize_policy_generated(problem, policies, world_context,
                                         goal_text=goal_text or problem.context, seed=seed,
                                         n_particles=n_particles, llm=llm,
                                         trace_path=trace_path,
                                         forensic_dir=forensic_dir)
    from swm.world_model_v2.phase13.policies import (Policy, PolicyExecutionOperator,
                                                     schedule_decision_points)
    from swm.world_model_v2.phase13.counterfactual import MatchedBundle, paired_report
    t0 = _time.time()
    res = DecisionResult(decision_id=problem.decision_id, contract_hash=problem.contract_hash(),
                        runtime_fingerprint=_fingerprint(), seed=seed)
    ev = _evaluator(world_context, n_particles=n_particles, seed=seed, llm=llm,
                    allow_single_structural_model=allow_single_structural_model)
    from swm.world_model_v2.state import parse_time
    points = []
    for dp in problem.decision_points or []:
        try:
            points.append(parse_time(dp))
        except (ValueError, TypeError):
            continue
    if not points:
        # no decision points supplied: the policy's ONE real trigger is its own initiation —
        # the evaluation's start instant (the intervention's initiation event), explicitly
        # labeled; never a synthetic offset pretending to be a schedule
        points = [float(ev.particles()[0].clock.now)]
        res.provenance["decision_points_note"] = (
            "no decision_points supplied — the policy decides once at its own initiation "
            "(intervention start), not on a synthetic schedule")
    all_policies = list(policies)
    if not any(p.policy_id == "do_nothing" for p in all_policies):
        all_policies.append(Policy(policy_id="do_nothing", decide=lambda belief: None,
                                   description="never act — the status-quo policy"))
    bundle = MatchedBundle(n_particles=ev.n_particles, seed=seed, reference="do_nothing")
    base_qb = ev.queue_builder
    for pol in all_policies:
        op = PolicyExecutionOperator(pol, problem.decision_maker, problem)
        ev_ops = list(ev.operators) + [op]
        arm_eval = MatchedEvaluator(initial=ev.initial,
                                    queue_builder=schedule_decision_points(base_qb, points,
                                                                           problem.decision_maker),
                                    operators=ev_ops, contract=ev.contract,
                                    n_particles=ev.n_particles, seed=seed,
                                    hypotheses=ev.hypotheses, max_events=ev.max_events)
        arm_eval._particles = ev.particles()               # SHARED particle identity across policies
        arm_eval._assignment = ev._assignment
        bundle.arms[pol.policy_id] = arm_eval.evaluate_arm(pol.policy_id, None)
    bundle.hypothesis_assignment = list(ev._assignment)
    bundle.crn_manifest = ev.crn_manifest(bundle)
    # evaluate POLICY arms as themselves: a zero-cost stand-in ActionSchema per policy id, so
    # robust evaluation ranks the actual policy outcomes (audit finding: a placeholder
    # [do_nothing] list made every policy report cost=0/operation=''/reversible=True under
    # do_nothing's identity instead of its own)
    from swm.world_model_v2.phase13.ontology import ActionSchema
    stand_ins = [ActionSchema(action_id=pol.policy_id, actor=problem.decision_maker,
                              operation="choose_policy", provenance="policy")
                 for pol in all_policies]
    evals = evaluate_bundle(bundle, stand_ins, problem)
    pols = {k: v for k, v in evals.items() if not k.startswith("_")}
    res.policies = list(pols.values())
    ranking = evals["_ranking"]["order"]
    viable = [r for r in ranking if r["score"] != "excluded"]
    res.recommended = viable[0]["action_id"] if viable else None
    res.recommendation_kind = "policy"
    res.reference_action = "do_nothing"
    res.provenance["crn_manifest"] = bundle.crn_manifest
    res.provenance["decision_points"] = points
    res.causal_claim = "simulated_mechanism_counterfactual"
    res.latency_s = round(_time.time() - t0, 3)
    return res


def value_of_information(problem: DecisionProblem, candidate_observations: list, world_context, *,
                         seed: int = 0, n_particles: int = None, llm=None,
                         mode: str = "auto", goal_text: str = "",
                         trace_path: str = "", forensic_dir: str = "") -> dict:
    """Standalone VOI (Part 14). On a generated world (mode='auto') this routes through the
    scenario layer: information-gathering is a first-class strategy class competing in the
    same matched simulation, and the report carries the adjudicated
    highest-value-information finding — no legacy detour."""
    r = recommend_action(problem, world_context, budget="standard", seed=seed,
                         n_particles=n_particles, llm=llm,
                         candidate_observations=candidate_observations, mode=mode,
                         goal_text=goal_text, trace_path=trace_path,
                         forensic_dir=forensic_dir)
    voi = r.value_of_information
    sr = (r.provenance or {}).get("scenario_report")
    if sr is not None:
        gather = [c for c in sr.get("candidates", [])
                  if "information" in str(c.get("strategy_class", "")).lower()
                  or "gather" in str(c.get("title", "")).lower()]
        voi = {"route": "scenario_generated",
               "highest_value_information": sr.get("highest_value_information", ""),
               "information_gathering_candidates": [c.get("candidate_id") for c in gather],
               "gathering_recommended": r.recommended in {c.get("candidate_id")
                                                          for c in gather}}
    return {"decision_id": problem.decision_id, "value_of_information": voi,
            "recommended": r.recommended, "recommendation_kind": r.recommendation_kind}


def _agg_matrix(bundle, actions, problem) -> dict:
    by_id = {a.action_id: a for a in actions}
    out = {}
    for aid, arm in bundle.arms.items():
        bd = evaluate_utility(aid, arm.outcomes, problem.utility)
        a = by_id.get(aid)
        cost = (a.direct_cost + a.indirect_cost) if a is not None else 0.0
        out[aid] = [u - cost for u in bd.aggregate]
    return out


def _active_phases(bundle) -> dict:
    """Which operators actually produced StateDeltas in this evaluation (per-arm census)."""
    census = {}
    for aid, arm in bundle.arms.items():
        for b in arm.branches:
            for d in b.log:
                census[d.operator] = census.get(d.operator, 0) + 1
    return {"operator_delta_census": census}
