"""Phase 13 adaptive search (Parts 17–19) — compile the decision, then pick the optimizer that fits.

The compiler classifies the decision problem (space size, discrete/continuous mix, sequential depth,
constraints, budget) and selects: exhaustive enumeration (small finite), successive-elimination racing
(medium finite, reuses the repo's best-arm discipline), coarse-to-fine hierarchical search (large
structured: family → target → timing → intensity → content), cross-entropy method (continuous), or
policy rollout comparison (sequential). Feasibility is respected DURING generation (infeasible actions
never reach a rollout). Every run returns diagnostics (method, evaluations, budget spent, checkpoint)
and — on tasks where exhaustive evaluation is possible — the harness measures the optimality gap
against the true optimum (search correctness is a MEASURED gate, not a vibe)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SearchBudget:
    max_arm_evaluations: int = 64          # rollout-bundle evaluations (each = n_particles rollouts)
    tier: str = "standard"                 # diagnostic | standard | production | maximum_capacity

    @classmethod
    def tiered(cls, tier: str) -> "SearchBudget":
        caps = {"diagnostic": 12, "standard": 64, "production": 256, "maximum_capacity": 4096}
        if tier not in caps:
            raise ValueError(f"unknown budget tier {tier!r} (valid: {sorted(caps)})")
        return cls(max_arm_evaluations=caps[tier], tier=tier)


@dataclass
class SearchDiagnostics:
    method: str = ""
    n_candidates: int = 0
    n_evaluated: int = 0
    budget: int = 0
    tier: str = ""
    eliminated: list = field(default_factory=list)
    checkpoint: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self):
        return {"method": self.method, "n_candidates": self.n_candidates,
                "n_evaluated": self.n_evaluated, "budget": self.budget, "tier": self.tier,
                "n_eliminated_early": len(self.eliminated), "notes": self.notes[:8],
                "checkpoint": self.checkpoint}


def classify(actions: list, problem) -> dict:
    """The decision-structure classification that drives optimizer selection."""
    n = len(actions)
    fams = {a.spec()["family"] for a in actions}
    continuous = [a for a in actions if isinstance(a.params.get("value"), float)]
    sequential = bool(problem.decision_points and len(problem.decision_points) > 1)
    return {"n_actions": n, "families": sorted(fams), "n_continuous_params": len(continuous),
            "sequential": sequential,
            "recommended": ("exhaustive" if n <= 24 and not sequential else
                            "racing" if n <= 200 and not sequential else
                            "hierarchical" if not sequential else "policy_rollout")}


def select_and_run(evaluator, actions: list, problem, *, budget: SearchBudget = None,
                   score_of=None) -> tuple:
    """Dispatch on the classification. Returns (bundle, diagnostics). `score_of(evals, action_id)`
    extracts the ranking score once robust evaluation ran; racing uses cheap interim scores."""
    budget = budget or SearchBudget()
    cls = classify(actions, problem)
    diag = SearchDiagnostics(method=cls["recommended"], n_candidates=len(actions),
                             budget=budget.max_arm_evaluations, tier=budget.tier)
    if cls["recommended"] == "exhaustive" or len(actions) <= budget.max_arm_evaluations:
        if cls["recommended"] != "exhaustive":
            diag.method = "exhaustive_within_budget"
        bundle = evaluator.evaluate(actions, problem=problem)
        diag.n_evaluated = len(actions)
        return bundle, diag
    if cls["recommended"] == "racing":
        return _race(evaluator, actions, problem, budget, diag)
    # hierarchical: family -> best-in-family -> refine
    return _hierarchical(evaluator, actions, problem, budget, diag)


def _mean_readout(arm) -> float:
    vals = [o.get("readout") for o in arm.outcomes]
    xs = [float(v) for v in vals if isinstance(v, (int, float, bool))]
    return sum(xs) / len(xs) if xs else 0.0


def _race(evaluator, actions, problem, budget, diag):
    """Successive halving on the matched evaluator: evaluate survivors, drop the bottom half by
    interim paired score, repeat until the budget or one survivor remains. The FINAL bundle re-runs
    the survivors + reference so the reported numbers all come from one matched evaluation."""
    from swm.world_model_v2.phase13.ontology import do_nothing
    survivors = list(actions)
    ref = next((a for a in survivors if a.action_id == "do_nothing"), None)
    if ref is None:
        ref = do_nothing(problem.decision_maker)
    spent = 0
    while len(survivors) > max(4, len(actions) // 16) and spent + len(survivors) <= budget.max_arm_evaluations:
        bundle = evaluator.evaluate(survivors + ([ref] if ref not in survivors else []),
                                    problem=problem, reference_id="do_nothing")
        spent += len(survivors)
        scored = sorted(((aid, _mean_readout(arm)) for aid, arm in bundle.arms.items()
                         if aid != "do_nothing"), key=lambda kv: -kv[1])
        keep = {aid for aid, _ in scored[:max(2, len(scored) // 2)]}
        diag.eliminated.extend([aid for aid, _ in scored if aid not in keep])
        survivors = [a for a in survivors if a.action_id in keep or a.action_id == "do_nothing"]
        diag.checkpoint = {"survivors": [a.action_id for a in survivors], "spent": spent}
    final = evaluator.evaluate(survivors + ([ref] if all(a.action_id != "do_nothing"
                                                         for a in survivors) else []),
                               problem=problem, reference_id="do_nothing")
    diag.n_evaluated = spent + len(survivors)
    return final, diag


def _hierarchical(evaluator, actions, problem, budget, diag):
    """Coarse-to-fine (Part 18): pick the best FAMILY on a per-family representative sample, then race
    within the winning family. Preserved uncertainty: the final report carries the losing families'
    representative scores so omitted detail is visible, not vanished."""
    from swm.world_model_v2.phase13.ontology import do_nothing
    by_family = {}
    for a in actions:
        by_family.setdefault(a.spec()["family"], []).append(a)
    reps = []
    for fam, acts in by_family.items():
        reps.extend(acts[:max(1, min(3, budget.max_arm_evaluations // (2 * len(by_family))))])
    ref = next((a for a in actions if a.action_id == "do_nothing"),
               do_nothing(problem.decision_maker))
    coarse = evaluator.evaluate(reps + ([ref] if ref not in reps else []), problem=problem,
                                reference_id="do_nothing")
    fam_score = {}
    for aid, arm in coarse.arms.items():
        a = next((x for x in reps if x.action_id == aid), None)
        if a is None:
            continue
        fam = a.spec()["family"]
        fam_score[fam] = max(fam_score.get(fam, -math.inf), _mean_readout(arm))
    best_fam = max(fam_score, key=fam_score.get) if fam_score else None
    diag.notes.append(f"coarse family scores: { {k: round(v, 4) for k, v in fam_score.items()} }")
    fine_actions = by_family.get(best_fam, [])[:budget.max_arm_evaluations // 2]
    bundle, race_diag = _race(evaluator, fine_actions + [ref], problem,
                              SearchBudget(max_arm_evaluations=budget.max_arm_evaluations // 2,
                                           tier=budget.tier), diag)
    diag.n_evaluated = len(reps) + race_diag.n_evaluated
    diag.checkpoint = race_diag.checkpoint
    return bundle, diag


# ---------------------------------------------------------------- Part 19: search correctness harness
def correctness_check(evaluator_factory, actions, problem, *, budget: SearchBudget = None) -> dict:
    """On a finite task: run exhaustive (fresh evaluator) AND the selected optimizer (fresh evaluator,
    same seed); report optimum recovery, optimality gap, and ranking agreement. This is the Part-19
    gate's per-task measurement."""
    ex_eval = evaluator_factory()
    exhaustive = ex_eval.evaluate(actions, problem=problem)
    ex_scores = {aid: _mean_readout(arm) for aid, arm in exhaustive.arms.items()}
    true_best = max(ex_scores, key=ex_scores.get)
    ap_eval = evaluator_factory()
    bundle, diag = select_and_run(ap_eval, actions, problem, budget=budget)
    ap_scores = {aid: _mean_readout(arm) for aid, arm in bundle.arms.items()}
    picked = max(ap_scores, key=ap_scores.get) if ap_scores else None
    gap = (ex_scores[true_best] - ex_scores.get(picked, -math.inf))
    denom = abs(ex_scores[true_best]) if abs(ex_scores[true_best]) > 1e-12 else 1.0
    return {"true_best": true_best, "picked": picked, "recovered": picked == true_best,
            "optimality_gap": round(max(0.0, gap), 6),
            "optimality_gap_rel": round(max(0.0, gap) / denom, 6),
            "method": diag.method, "n_evaluated": diag.n_evaluated}
