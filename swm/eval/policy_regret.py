"""The action layer's scoreboard — policy regret + CATE-sign + off-policy value, NOT log-loss.

A decision layer is graded on whether ACTING on it beats not acting, not on reconstruction accuracy (the
lesson of EXP-054: reconstruction did not transfer to intervention skill). The honest metrics:

  - PRECISION@1 — how often the model's top-ranked action is a true-best (the best-message KPI, EXP-060).
  - POLICY REGRET — the utility left on the table vs an oracle that always picks the best action.
  - CATE-SIGN — does the model get the SIGN of a pairwise treatment effect right (does B beat A)? Ranking
    arms correctly is the whole game; this is the cleanest test of it.
  - OFF-POLICY VALUE (IPS / doubly-robust) — from logged (action, reward, propensity) data, the value the
    model's policy WOULD have achieved, without running it live. Unbiased under known propensities (IPS);
    lower-variance with a reward model (DR).
"""
from __future__ import annotations


def precision_at_1(chosen, oracle) -> float:
    """Fraction of instances where the model's chosen action equals a best action. `chosen`/`oracle` are
    per-instance labels; `oracle[i]` may be a single label or a set/list of tied-best labels."""
    n = len(chosen)
    if not n:
        return float("nan")
    hit = 0
    for c, o in zip(chosen, oracle):
        ok = (c in o) if isinstance(o, (set, list, tuple)) else (c == o)
        hit += 1 if ok else 0
    return hit / n


def policy_regret(chosen_rewards, oracle_rewards) -> float:
    """Mean utility gap vs an oracle: mean(oracle - chosen). 0 = optimal; larger = more left on the table."""
    n = len(chosen_rewards)
    return sum(o - c for c, o in zip(chosen_rewards, oracle_rewards)) / n if n else float("nan")


def cate_sign_accuracy(pred_deltas, true_deltas, *, tol=0.0) -> float:
    """Fraction of pairwise contrasts whose SIGN the model gets right (does B beat A?), over contrasts whose
    true effect exceeds `tol` in magnitude (ties excluded)."""
    num = den = 0
    for p, t in zip(pred_deltas, true_deltas):
        if abs(t) <= tol:
            continue
        den += 1
        if (p > 0) == (t > 0):
            num += 1
    return num / den if den else float("nan")


def ips_value(logged_actions, rewards, behavior_probs, target_actions) -> float:
    """Inverse-propensity-scored value of a deterministic target policy. `target_actions[i]` is the action the
    policy would take for context i; contributes reward/propensity only where it matches the logged action.
    Unbiased when behavior_probs are the true logging propensities and all > 0."""
    n = len(rewards)
    if not n:
        return float("nan")
    tot = 0.0
    for a, r, p, ta in zip(logged_actions, rewards, behavior_probs, target_actions):
        if a == ta and p > 0:
            tot += r / p
    return tot / n


def doubly_robust_value(logged_actions, rewards, behavior_probs, target_actions, q_target, q_logged) -> float:
    """Doubly-robust value: the reward-model estimate at the target action plus an IPS correction on the
    logged action. `q_target[i]` = model's reward estimate for the target action; `q_logged[i]` = for the
    logged action. Consistent if EITHER the propensities or the reward model is correct (hence 'doubly')."""
    n = len(rewards)
    if not n:
        return float("nan")
    tot = 0.0
    for a, r, p, ta, qt, ql in zip(logged_actions, rewards, behavior_probs, target_actions, q_target, q_logged):
        tot += qt + ((r - ql) / p if a == ta and p > 0 else 0.0)
    return tot / n
