"""Phase 13 strategic & equilibrium reasoning (Part 11) — bounded, explicit, convergence-reported.

Other actors already respond DYNAMICALLY inside every rollout (Phase-4 policy operators, institutions,
populations, diffusion react to the decision_action's follow-up events). This module adds the layer for
decisions where OPPONENTS OPTIMIZE TOO: iterated best response / level-k over the matched evaluator.
Each iteration re-evaluates the focal actor's actions against the opponents' current strategy profile
(carried in the world via entity `planned_actions`), then best-responds each opponent using THEIR
utility. Convergence (a fixed point = pure Nash within the discretized action sets) or non-convergence
after max_iters is REPORTED, never claimed silently; quantal response replaces argmax when a precision
λ is supplied (McKelvey–Palfrey, via the canonical `policy.logit_choice`)."""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.policy import logit_choice


@dataclass
class StrategicActor:
    actor_id: str
    actions: list                                   # [ActionSchema] this opponent can play
    utility_fn: object                              # callable(outcome) -> float (THEIR utility)
    level: int = 1                                  # level-k depth for this actor
    precision: float = None                         # None = best reply; else quantal response λ


def iterated_best_response(evaluate_profile, focal_actions, opponents, *, max_iters: int = 6,
                           tol: float = 1e-9, rng=None) -> dict:
    """`evaluate_profile(focal_action, {opponent_id: action}) -> {"focal": u, opponent_id: u, ...}`
    (expected utilities from a MATCHED evaluation under that joint profile). Returns the profile
    trajectory, the final profile, and an explicit convergence statement with sensitivity handles."""
    profile = {o.actor_id: o.actions[0] for o in opponents}
    focal = focal_actions[0]
    history = []
    converged = False
    for it in range(max_iters):
        prev = (focal.action_id, {k: v.action_id for k, v in profile.items()})
        # focal best-responds to the current opponent profile
        focal_utils = [evaluate_profile(fa, profile)["focal"] for fa in focal_actions]
        focal = focal_actions[max(range(len(focal_actions)), key=lambda i: focal_utils[i])]
        # each opponent best-responds (or quantal-responds) to the updated profile
        for o in opponents:
            others = {k: v for k, v in profile.items() if k != o.actor_id}
            utils = []
            for oa in o.actions:
                prof = dict(others)
                prof[o.actor_id] = oa
                utils.append(evaluate_profile(focal, prof)[o.actor_id])
            if o.precision is not None and rng is not None:
                ps = logit_choice(utils, o.precision)
                r, acc, pick = rng.random(), 0.0, len(utils) - 1
                for i, p in enumerate(ps):
                    acc += p
                    if r <= acc:
                        pick = i
                        break
            else:
                pick = max(range(len(utils)), key=lambda i: utils[i])
            profile[o.actor_id] = o.actions[pick]
        cur = (focal.action_id, {k: v.action_id for k, v in profile.items()})
        history.append({"iter": it, "focal": cur[0], "opponents": cur[1]})
        if cur == prev:
            converged = True
            break
    return {"focal_action": focal, "profile": {k: v.action_id for k, v in profile.items()},
            "converged": converged, "iterations": len(history), "trajectory": history,
            "equilibrium_claim": ("pure best-response fixed point within the discretized action sets"
                                  if converged else
                                  "NOT converged — no equilibrium claim; report is the final iterate"),
            "sensitivity": {"reasoning_depth": "re-run with different max_iters/level",
                            "opponent_model": "utilities are the modeled opponents', not observed",
                            "policy_uncertainty": "wrap evaluate_profile over posterior particles"}}


def level_k_response(evaluate_profile, focal_actions, opponents, *, k: int = 2) -> dict:
    """Level-k: level-0 opponents play their FIRST listed action (the declared naive anchor); each
    higher level best-responds to the level below. Returns the focal level-k action + the ladder."""
    ladder = []
    profile = {o.actor_id: o.actions[0] for o in opponents}
    for lvl in range(1, max(1, k) + 1):
        focal_utils = [evaluate_profile(fa, profile)["focal"] for fa in focal_actions]
        focal = focal_actions[max(range(len(focal_actions)), key=lambda i: focal_utils[i])]
        nxt = {}
        for o in opponents:
            utils = []
            for oa in o.actions:
                prof = {kk: vv for kk, vv in profile.items()}
                prof[o.actor_id] = oa
                utils.append(evaluate_profile(focal, prof)[o.actor_id])
            nxt[o.actor_id] = o.actions[max(range(len(utils)), key=lambda i: utils[i])]
        ladder.append({"level": lvl, "focal": focal.action_id,
                       "opponents": {a: x.action_id for a, x in nxt.items()}})
        profile = nxt
    return {"focal_action": focal, "ladder": ladder, "k": k,
            "anchor": "level-0 = first listed opponent action (declared, not hidden)"}


def apply_profile_to_world(world, profile: dict):
    """Carry the opponents' current strategy into the shared world (entity planned_actions) so the
    rollout's own operators see the strategic context — the profile is world state, not a side channel."""
    from swm.world_model_v2.state import F
    for actor_id, action in profile.items():
        ent = (world.entities or {}).get(actor_id)
        if ent is not None:
            ent.set("planned_actions", F(action.operation, status="assumed",
                                         method="phase13:strategic_profile"), key=action.action_id)
