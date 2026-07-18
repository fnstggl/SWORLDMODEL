"""Phase 13 across the structural ensemble — the DEFAULT decision path (Section 18 of the ensemble
contract).

    candidate action or policy
    → the same intended intervention compiled for each compatible structural model (the typed
      ActionSchema is model-independent; each model's OWN operators/feasibility/institutions decide how
      it executes — the LLM never rewrites the action per model)
    → matched within-model rollouts (full canonical Phase-13 pipeline per model, per-model budgets)
    → model-specific trajectory + utility results and action rankings (preserved, never blended away)
    → cross-model robustness: winner-per-model, feasibility-per-model, worst-model downside, minimax
      regret across models, labeled equal-mixture summary
    → recommendation stability or structural sensitivity, reversal conditions, and the information that
      would distinguish the models.

If different plausible models recommend different actions, the result is a conditional strategy plus an
information-gathering recommendation (or a robust/Pareto set) — never one average utility hiding the
disagreement. No LLM ever mints a model probability; the mixture view is explicitly labeled uncalibrated
equal weighting.
"""
from __future__ import annotations

import copy
import time as _time

from swm.world_model_v2.phase13.contracts import Abstention, DecisionProblem, DecisionResult

#: winner-change across models is at least material; regret share (of cross-action utility range) above
#: this is material even without a winner change. Mirrors structural_contracts thresholds; validated by
#: tests/test_structural_ensemble.py.
DECISION_REGRET_MATERIAL_SHARE = 0.15


def extract_ensemble_models(world_context) -> dict:
    """Normalize any ensemble-shaped world context into {model_id: {"plan", "meta"}}.

    Accepted: a StructuralModelEnsemble; a SimulationResult from the default runtime (carries the live
    ensemble handle); {model_id: WorldExecutionPlan}; [WorldExecutionPlan, ...]. Returns {} when the
    context is not ensemble-shaped."""
    from swm.world_model_v2.compiler import WorldExecutionPlan
    try:
        from swm.world_model_v2.structural_contracts import StructuralModelEnsemble
    except Exception:  # noqa: BLE001
        StructuralModelEnsemble = ()  # noqa: N806
    handle = getattr(world_context, "_ensemble_handle", None)
    if handle is not None:
        world_context = handle
    if isinstance(world_context, StructuralModelEnsemble):
        out = {}
        for c in world_context.surviving():
            if c.executable_plan is not None and c.promotion_status in ("promoted", "generated",
                                                                        "repaired", "pilot"):
                out[c.model_id] = {"plan": c.executable_plan,
                                   "meta": {"causal_thesis": c.causal_thesis,
                                            "generation_role": c.generation_role,
                                            "support_class": c.support_class,
                                            "falsifiers": list(c.falsifiers),
                                            "decisive_actors": list(c.decisive_actors),
                                            "decisive_mechanisms": list(c.decisive_mechanisms)}}
        return out
    if isinstance(world_context, dict) and world_context and \
            all(isinstance(v, WorldExecutionPlan) for v in world_context.values()):
        return {str(k): {"plan": v, "meta": {}} for k, v in world_context.items()}
    if isinstance(world_context, (list, tuple)) and world_context and \
            all(isinstance(v, WorldExecutionPlan) for v in world_context):
        return {f"model_{i}": {"plan": v, "meta": {}} for i, v in enumerate(world_context)}
    return {}


def recommend_action_across_models(problem: DecisionProblem, models: dict, *, budget: str = "standard",
                                   seed: int = 0, n_particles: int = None, llm=None,
                                   candidate_observations: list = None,
                                   actions: list = None) -> DecisionResult:
    """Run the FULL canonical Phase-13 pipeline inside every structural model (matched rollouts,
    feasibility, robust evaluation, per-model ranking — each model with its OWN particle budget, the
    same seed for common-random-number alignment), then synthesize cross-model robustness. Per-model
    results are preserved verbatim in provenance."""
    from swm.world_model_v2.phase13.api import evaluate_actions, recommend_action
    t0 = _time.time()
    per_model: dict = {}
    for mid, entry in models.items():
        p_m = copy.deepcopy(problem)
        try:
            if actions is not None:
                r = evaluate_actions(p_m, list(actions), entry["plan"], budget=budget, seed=seed,
                                     n_particles=n_particles, llm=llm,
                                     allow_single_structural_model=True)
            else:
                r = recommend_action(p_m, entry["plan"], budget=budget, seed=seed,
                                     n_particles=n_particles, llm=llm,
                                     candidate_observations=candidate_observations,
                                     allow_single_structural_model=True)
            per_model[mid] = {"result": r, "meta": entry.get("meta", {}), "error": ""}
        except Exception as e:  # noqa: BLE001 — a failed model is recorded loudly, never hidden
            per_model[mid] = {"result": None, "meta": entry.get("meta", {}),
                              "error": f"{type(e).__name__}: {e}"[:200]}
    ok = {m: v for m, v in per_model.items() if v["result"] is not None}
    res = DecisionResult(decision_id=problem.decision_id, contract_hash=problem.contract_hash(),
                         runtime_fingerprint={"phase13": "phase13-ensemble-1.0"}, seed=seed)
    if not ok:
        res.abstention = Abstention(
            reasons=[{"code": "ensemble_execution_incomplete",
                      "detail": "every structural model failed Phase-13 evaluation"}],
            needed=[v["error"] for v in per_model.values()][:4]).as_dict()
        res.recommendation_kind = "abstain"
        res.provenance["structural_ensemble"] = {"per_model_errors":
                                                 {m: v["error"] for m, v in per_model.items()}}
        res.latency_s = round(_time.time() - t0, 3)
        return res

    synthesis = _synthesize(ok, problem)
    # ---------- the combined result: stability decides the recommendation form ----------
    res.evaluated = synthesis["mixture_evaluated"]
    res.reference_action = next(iter(ok.values()))["result"].reference_action
    res.feasibility = synthesis["feasibility_union"]
    res.counterfactual = {"per_model_reference": {m: v["result"].reference_action
                                                  for m, v in ok.items()}}
    res.value_of_information = synthesis["information"]
    if synthesis["stable_winner"] is not None:
        res.recommended = synthesis["stable_winner"]
        res.recommendation_kind = "action"
    elif synthesis["information"].get("model_discriminating_observations"):
        res.recommended = "gather_information"
        res.recommendation_kind = "gather_information"
        res.abstention = Abstention(
            reasons=[{"code": "structurally_sensitive_recommendation",
                      "detail": "plausible structural models recommend different actions; the "
                                "conditional strategy and discriminating observations are in "
                                "provenance.structural_ensemble"}],
            needed=[str(o.get("observation", ""))[:160] for o in
                    synthesis["information"]["model_discriminating_observations"][:3]]).as_dict()
    else:
        res.recommendation_kind = "pareto"
        res.recommended = None
        res.abstention = Abstention(
            reasons=[{"code": "structurally_sensitive_recommendation",
                      "detail": "models disagree and no discriminating observation was identified; "
                                "the robust (minimax-regret) set is reported"}],
            needed=["evidence distinguishing the surviving causal models"]).as_dict()
        res.pareto_frontier = synthesis["robust_set"]
    res.causal_claim = "simulated_mechanism_counterfactual"
    res.support_grade = "exploratory"
    res.cost = {"per_model": {m: v["result"].cost for m, v in ok.items()},
                "n_models": len(ok)}
    res.provenance["structural_ensemble"] = {
        "structural_mode": "ensemble",
        "n_models_evaluated": len(ok),
        "per_model_results": {m: v["result"].as_dict() for m, v in ok.items()},
        "per_model_errors": {m: v["error"] for m, v in per_model.items() if v["error"]},
        "model_meta": {m: v["meta"] for m, v in per_model.items()},
        **{k: synthesis[k] for k in ("winner_by_model", "ranking_by_model", "infeasible_by_model",
                                     "expected_utility_matrix", "worst_model_downside",
                                     "minimax_regret_across_models", "robust_action",
                                     "equal_mixture_ranking", "recommendation_stability",
                                     "reversal_conditions", "conditional_strategy",
                                     "dominant_model_check")},
        "aggregation_note": "equal-mixture views are UNCALIBRATED structural averaging; per-model "
                            "rankings are the primary readouts",
    }
    res.latency_s = round(_time.time() - t0, 3)
    return res


def _synthesize(ok: dict, problem) -> dict:
    """Deterministic cross-model synthesis from per-model DecisionResults. No minted numbers: every
    quantity is arithmetic over ACTUAL per-model expected utilities and rankings."""
    winner_by_model, ranking_by_model, infeasible_by_model, eu = {}, {}, {}, {}
    for mid, v in ok.items():
        r = v["result"]
        winner_by_model[mid] = r.recommended
        order = ((r.provenance.get("ranking") or {}).get("order")) or []
        ranking_by_model[mid] = [{"action_id": o.get("action_id"), "score": o.get("score")}
                                 for o in order]
        infeasible_by_model[mid] = [f.get("action_id") for f in (r.feasibility or [])
                                    if isinstance(f, dict) and not f.get("feasible", True)]
        for e in (r.evaluated or []):
            aid = e.get("action_id")
            if aid is not None and isinstance(e.get("expected_utility"), (int, float)):
                eu.setdefault(aid, {})[mid] = float(e["expected_utility"])
    model_ids = list(ok)
    shared_actions = [a for a, per in eu.items() if len(per) == len(model_ids)]

    # winner stability: STABLE only when EVERY model names the SAME winner. A model that abstains or
    # finds the actions infeasible counts against stability — "wins under one model, unresolved under
    # the others" is exactly the fragility Section 18 requires surfacing, never a stable winner.
    winners_named = {m: w for m, w in winner_by_model.items()
                     if w and w != "gather_information"}
    stable_winner = (next(iter(set(winners_named.values())))
                     if (len(winners_named) == len(model_ids) and
                         len(set(winners_named.values())) == 1) else None)

    # minimax regret across models over shared actions
    regret_by_action = {}
    for a in shared_actions:
        regrets = []
        for m in model_ids:
            best_m = max((eu[b][m] for b in shared_actions), default=0.0)
            regrets.append(best_m - eu[a][m])
        regret_by_action[a] = {"max_regret_across_models": round(max(regrets), 4) if regrets else None,
                               "regret_by_model": {m: round(r, 4)
                                                   for m, r in zip(model_ids, regrets)}}
    robust_action = (min(regret_by_action,
                         key=lambda a: regret_by_action[a]["max_regret_across_models"])
                     if regret_by_action else None)
    worst_model_downside = {a: {"worst_model": min(per, key=per.get),
                                "worst_expected_utility": round(min(per.values()), 4)}
                            for a, per in eu.items() if per}
    mixture = {a: round(sum(per.values()) / len(per), 4)
               for a, per in eu.items() if len(per) == len(model_ids)}
    mixture_ranking = sorted(mixture.items(), key=lambda kv: -kv[1])

    # stability classification from actual results (winner change / regret share of utility range)
    spread = 0.0
    for a, per in eu.items():
        if len(per) > 1:
            spread = max(spread, max(per.values()) - min(per.values()))
    utils_all = [u for per in eu.values() for u in per.values()]
    util_range = (max(utils_all) - min(utils_all)) if len(utils_all) > 1 else 0.0
    if stable_winner is not None:
        sel_regret = (regret_by_action.get(stable_winner, {}).get("max_regret_across_models") or 0.0)
        material = util_range > 0 and (sel_regret / util_range) > DECISION_REGRET_MATERIAL_SHARE
        stability = ("mildly_structurally_sensitive" if material else "structurally_stable")
    elif winners_named and len(set(winners_named.values())) == 1:
        # one action wins wherever a winner exists, but some models abstained/found it infeasible —
        # unopposed is not confirmed
        stability = "mildly_structurally_sensitive"
    else:
        stability = "materially_structurally_sensitive"

    reversal = []
    if stable_winner is None and winner_by_model:
        for mid, w in winner_by_model.items():
            meta = ok[mid]["meta"]
            reversal.append({
                "model_id": mid, "winning_action": w,
                "assumption": meta.get("causal_thesis") or f"structure of {mid}",
                "evidence_that_would_confirm": (meta.get("falsifiers") or [])[:3]})
    conditional = [{"if_model": mid,
                    "assumption": (ok[mid]["meta"].get("causal_thesis") or mid),
                    "then_action": w}
                   for mid, w in winner_by_model.items() if w]

    # dominance check: does one model's ranking fully determine the mixture ranking?
    dominant = None
    for mid in model_ids:
        order_m = [r["action_id"] for r in ranking_by_model.get(mid, [])]
        order_mix = [a for a, _ in mixture_ranking]
        if order_m and order_mix and order_m[:len(order_mix)] == order_mix:
            dominant = mid
            break

    # discriminating information: falsifiers of models whose winners differ
    discriminating = []
    if stable_winner is None:
        for mid, v in ok.items():
            for f in (v["meta"].get("falsifiers") or [])[:2]:
                discriminating.append({"observation": f, "model_id": mid,
                                       "decision_relevance":
                                           f"bears on whether {winner_by_model.get(mid)!r} "
                                           f"is the right action (it tests model {mid})"})
    information = {"model_discriminating_observations": discriminating,
                   "note": "structural VOI is derived from actual model differences; no numeric EVSI "
                           "is minted without a measurement basis"}

    mixture_evaluated = [{"action_id": a, "expected_utility": u,
                          "aggregation": "equal_weight_uncalibrated_structural_average",
                          "per_model_expected_utility": {m: round(x, 4)
                                                         for m, x in eu.get(a, {}).items()}}
                         for a, u in mixture_ranking]
    feas_union = []
    for mid, v in ok.items():
        for f in (v["result"].feasibility or []):
            feas_union.append({**f, "model_id": mid})
    robust_set = [{"action_id": a, **regret_by_action[a]} for a in
                  sorted(regret_by_action,
                         key=lambda a: regret_by_action[a]["max_regret_across_models"])[:4]]
    return {"winner_by_model": winner_by_model, "ranking_by_model": ranking_by_model,
            "infeasible_by_model": infeasible_by_model, "expected_utility_matrix": eu,
            "worst_model_downside": worst_model_downside,
            "minimax_regret_across_models": regret_by_action, "robust_action": robust_action,
            "equal_mixture_ranking": mixture_ranking, "stable_winner": stable_winner,
            "recommendation_stability": stability, "reversal_conditions": reversal,
            "conditional_strategy": conditional, "dominant_model_check": dominant,
            "information": information, "mixture_evaluated": mixture_evaluated,
            "feasibility_union": feas_union, "robust_set": robust_set}


def optimize_policy_across_models(problem: DecisionProblem, policies: list, models: dict, *,
                                  seed: int = 0, n_particles: int = None, llm=None) -> DecisionResult:
    """Policy optimization across structural models: the full canonical policy evaluation inside every
    model (same seed → CRN alignment), then the same cross-model synthesis over policy values."""
    from swm.world_model_v2.phase13.api import optimize_policy
    t0 = _time.time()
    per_model = {}
    for mid, entry in models.items():
        try:
            r = optimize_policy(copy.deepcopy(problem), policies, entry["plan"], seed=seed,
                                n_particles=n_particles, llm=llm,
                                allow_single_structural_model=True)
            per_model[mid] = {"result": r, "meta": entry.get("meta", {}), "error": ""}
        except Exception as e:  # noqa: BLE001
            per_model[mid] = {"result": None, "meta": entry.get("meta", {}),
                              "error": f"{type(e).__name__}: {e}"[:200]}
    ok = {m: v for m, v in per_model.items() if v["result"] is not None}
    res = DecisionResult(decision_id=problem.decision_id, contract_hash=problem.contract_hash(),
                         runtime_fingerprint={"phase13": "phase13-ensemble-1.0"}, seed=seed)
    if not ok:
        res.recommendation_kind = "abstain"
        res.abstention = Abstention(reasons=[{"code": "ensemble_execution_incomplete",
                                              "detail": "every model failed policy evaluation"}],
                                    needed=[]).as_dict()
        res.latency_s = round(_time.time() - t0, 3)
        return res
    winner_by_model = {m: v["result"].recommended for m, v in ok.items()}
    values = {}
    for mid, v in ok.items():
        for p in (v["result"].policies or []):
            aid = p.get("action_id")
            if aid and isinstance(p.get("expected_utility"), (int, float)):
                values.setdefault(aid, {})[mid] = float(p["expected_utility"])
    winners = {w for w in winner_by_model.values() if w}
    stable = winners.pop() if len({w for w in winner_by_model.values() if w}) == 1 else None
    res.policies = [{"policy_id": a, "per_model_expected_utility": per,
                     "equal_mixture": round(sum(per.values()) / len(per), 4)}
                    for a, per in values.items()]
    res.recommended = stable
    res.recommendation_kind = "policy" if stable else "pareto"
    res.provenance["structural_ensemble"] = {
        "winner_by_model": winner_by_model,
        "recommendation_stability": ("structurally_stable" if stable
                                     else "materially_structurally_sensitive"),
        "per_model_results": {m: v["result"].as_dict() for m, v in ok.items()},
        "per_model_errors": {m: v["error"] for m, v in per_model.items() if v["error"]}}
    res.latency_s = round(_time.time() - t0, 3)
    return res
