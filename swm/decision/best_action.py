"""The general best-action layer — argmax_a E[U(outcome) | do(a)], as nested Monte-Carlo + best-arm racing.

This is the interventional core the whole product leans on: rather than forecasting from what we know now,
we PLACE each candidate action in the world, simulate what happens, and find the action that best reaches a
desired outcome. Two nested loops:

  INNER  — for a fixed action, draw utility samples from the rollout under do(action) (a compiled Sampler).
  OUTER  — best-arm identification by successive elimination with confidence intervals: pour samples into
           the arms still in contention, eliminate the confidently-dominated (their CI upper bound falls
           below the leader's lower bound), and STOP with either a confident winner or, at the budget, an
           honest "tie within noise" set. This is what keeps it from being fragile: a fixed-N argmax crowns
           whichever loser got lucky and can never say "too close to call"; racing does both.

`best_action` is generic over `outcome_fn(action, rng) -> (outcome, factors)`, so it works on any compiled
mechanism (via swm.api.action_simulate) and on any hand-built model today. The winner is returned as a
NavigableOutcome (distribution + reducible/irreducible + pivotal worlds) with a confidence statement and the
contrast versus doing nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from swm.decision.utility import Mean
from swm.report.navigable import navigable_from_samples


@dataclass
class ArmResult:
    label: str
    action: object
    value: float
    ci: tuple
    n: int

    def as_dict(self):
        return {"label": self.label, "value": self.value, "ci": [round(self.ci[0], 4), round(self.ci[1], 4)],
                "n": self.n}


@dataclass
class DecisionResult:
    best: ArmResult
    ranking: list                      # list[ArmResult], value-desc
    decided: bool                      # True: confident winner; False: tie set at budget
    tie_set: list                      # labels indistinguishable from the best at the budget
    win_prob: float                    # P(best beats runner-up) by bootstrap
    objective: str = "mean"
    navigable: object = None           # NavigableOutcome for the best action
    contrast: dict = None              # deltas vs runner-up and vs baseline (do-nothing)
    total_samples: int = 0
    provenance: dict = None            # identifiability honesty: is this ranking VALIDATED or a HYPOTHESIS?

    def grade(self) -> str:
        """A calibration/confidence grade for the recommendation — never a naked 'do B'. Combines whether a
        winner was resolved (vs a tie), how decisively (win_prob), how much of the outcome spread is
        irreducible (from the navigable), and whether the ranking is validated on a real domain (provenance).
        A = confident + validated; through D = a tie / mostly-irreducible / unvalidated hypothesis."""
        if not self.decided:
            return "D (tie within noise — get more data or accept the coin flip)"
        validated = bool(self.provenance and self.provenance.get("status") == "validated")
        forecastable = getattr(self.navigable, "forecastable", None)
        if self.win_prob >= 0.9 and validated and forecastable is not False:
            return "A (decisive, validated)"
        if self.win_prob >= 0.9:
            return "B (decisive; unvalidated domain)" if not validated else "B"
        if self.win_prob >= 0.7:
            return "C (leaning; not decisive)"
        return "D (weak edge)"

    def as_dict(self):
        return {"best": self.best.as_dict(), "decided": self.decided, "tie_set": self.tie_set,
                "win_prob": self.win_prob, "objective": self.objective, "grade": self.grade(),
                "provenance": self.provenance,
                "ranking": [a.as_dict() for a in self.ranking], "contrast": self.contrast,
                "navigable": self.navigable.as_dict() if self.navigable else None,
                "total_samples": self.total_samples}

    def summary(self) -> str:
        head = f"BEST: {self.best.label} ({self.objective}={self.best.value:.4f})"
        if not self.decided:
            return head + f" — but TIE within noise at budget: {self.tie_set} (get more data / more signal)"
        head += f", beats runner-up w.p. {self.win_prob:.0%}  [grade {self.grade()}]"
        if self.contrast and self.contrast.get("vs_baseline"):
            head += f"; {self.contrast['vs_baseline']['delta']:+.4f} vs do-nothing"
        if self.provenance:
            head += f"\n  provenance: {self.provenance.get('note', self.provenance.get('status', '?'))}"
        if self.navigable:
            head += f"\n  {self.navigable.summary()}"
        return head


def _win_prob(a_samples, b_samples, objective, rng, resamples=200):
    """Bootstrap P(objective(a) > objective(b))."""
    na, nb = len(a_samples), len(b_samples)
    if na < 2 or nb < 2:
        return 1.0 if objective.value(a_samples) >= objective.value(b_samples) else 0.0
    wins = 0
    for _ in range(resamples):
        ra = objective.value([a_samples[rng.randrange(na)] for _ in range(na)])
        rb = objective.value([b_samples[rng.randrange(nb)] for _ in range(nb)])
        wins += 1 if ra > rb else 0
    return wins / resamples


def race(actions, sample_fn, objective, *, batch=48, max_per_arm=3000, conf=0.9, seed=0):
    """Successive-elimination best-arm identification. `sample_fn(action, rng) -> utility_value`. Returns
    (buffers, ranking, alive, decided). Confidence is Bonferroni-corrected across arms so eliminations hold
    jointly at ~`conf`."""
    rng = Random(seed)
    arms = {a.label: a for a in actions}
    buf = {lab: [] for lab in arms}
    alive = list(arms)
    per_arm_conf = 1 - (1 - conf) / max(1, len(arms))
    decided = False
    while True:
        for lab in alive:
            a = arms[lab]
            buf[lab].extend(sample_fn(a, rng) for _ in range(batch))
        alive.sort(key=lambda l: -objective.value(buf[l]))
        leader = alive[0]
        best_lo, _ = objective.ci(buf[leader], per_arm_conf, rng)
        survivors = [leader]
        for lab in alive[1:]:
            _, hi = objective.ci(buf[lab], per_arm_conf, rng)
            if hi >= best_lo:                                     # still could beat the leader — keep racing
                survivors.append(lab)
        alive = survivors
        if len(alive) == 1:
            decided = True
            break
        if min(len(buf[l]) for l in alive) >= max_per_arm:
            decided = False                                       # budget exhausted; survivors are a tie set
            break
    ranking = sorted(arms, key=lambda l: -objective.value(buf[l]))
    return buf, ranking, alive, decided


def _navigable_for(outcome_fn, action, utility, n, seed):
    rng = Random(seed)
    samples = [outcome_fn(action, rng) for _ in range(n)]
    return navigable_from_samples(samples, target=(utility.fn, utility.desc))


def best_action(outcome_fn, actions, utility, *, objective=None, baseline=None, batch=48, max_per_arm=3000,
                conf=0.9, seed=0, navigate=True, n_navigable=4000, provenance=None) -> DecisionResult:
    """Choose argmax over `actions` of `objective` of `utility(outcome | do(action))`, adaptively.
    `outcome_fn(action, rng) -> (outcome, factors)`. Returns the winner as a navigable object with a
    confidence statement and contrast vs do-nothing. `provenance` (e.g. `validated_domain('CMV persuasion',
    0.74)`) rides into the result so the recommendation carries whether its ranking is validated or a
    hypothesis (Component 7)."""
    objective = objective or Mean()
    acts = list(actions)
    if not acts:
        raise ValueError("no candidate actions")

    def sfn(a, rng):
        return utility(outcome_fn(a, rng)[0])

    buf, ranking, alive, decided = race(acts, sfn, objective, batch=batch, max_per_arm=max_per_arm,
                                        conf=conf, seed=seed)
    arms = {a.label: a for a in acts}
    ci_rng = Random(seed + 7)

    def arm(lab):
        return ArmResult(lab, arms[lab], round(objective.value(buf[lab]), 4),
                         objective.ci(buf[lab], conf, ci_rng), len(buf[lab]))

    ranked = [arm(l) for l in ranking]
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    wp = _win_prob(buf[best.label], buf[runner.label], objective, Random(seed + 11)) if runner else 1.0

    nav = _navigable_for(outcome_fn, best.action, utility, n_navigable, seed + 3) if navigate else None

    contrast = {}
    if runner:
        contrast["vs_runner_up"] = {"label": runner.label, "delta": round(best.value - runner.value, 4),
                                    "win_prob": round(wp, 4)}
    if baseline is not None:
        base_samples = [utility(outcome_fn(baseline, r)[0]) for r in (Random(seed + 5) for _ in range(1))
                        for _ in range(max(1, n_navigable // 2))]
        base_val = objective.value(base_samples)
        bp = _win_prob(buf[best.label], base_samples, objective, Random(seed + 13))
        contrast["vs_baseline"] = {"label": getattr(baseline, "label", "baseline"),
                                   "value": round(base_val, 4), "delta": round(best.value - base_val, 4),
                                   "win_prob": round(bp, 4)}

    return DecisionResult(best=best, ranking=ranked, decided=decided,
                          tie_set=(alive if not decided else [best.label]), win_prob=round(wp, 4),
                          objective=objective.name, navigable=nav, contrast=contrast,
                          total_samples=sum(len(b) for b in buf.values()), provenance=provenance)


def validated_domain(name, metric=None, *, status="validated") -> dict:
    """Provenance tag for a recommendation: `validated_domain('CMV persuasion', 0.74)` ->
    {'status':'validated','note':'validated on CMV persuasion (skill 0.74)'}. Use `status='hypothesis'` for
    a domain the layer has NOT been backtested on, so the recommendation says so honestly."""
    note = f"{'validated on' if status == 'validated' else 'HYPOTHESIS on'} {name}"
    if metric is not None:
        note += f" (skill {metric})"
    return {"status": status, "domain": name, "metric": metric, "note": note}


def best_continuous(outcome_fn_for, var, lo, hi, utility, *, objective=None, rounds=3, steps=7, shrink=0.34,
                    baseline=None, seed=0, **kw) -> DecisionResult:
    """Continuous-parameter optimization (pricing, discount, timing): grid → race → NARROW the range around
    the winner → repeat (Component 2's 'grid → local refine'). `outcome_fn_for(value) -> outcome_fn` builds
    the rollout for a parameter value (e.g. sets price=value on the spec). Each round races a fresh grid
    inside the shrinking window; the final DecisionResult is the last round's, with the refined optimum."""
    lo, hi = float(lo), float(hi)
    result = None
    for r in range(max(1, rounds)):
        vals = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)] if steps > 1 else [lo]
        actions = [_ValueArm(var, v, outcome_fn_for(v)) for v in vals]

        def of(arm, rng):
            return arm.outcome_fn(rng)
        result = best_action(of, actions, utility, objective=objective, baseline=baseline, seed=seed + r,
                             navigate=(r == rounds - 1), **kw)
        best_v = result.best.action.value
        half = (hi - lo) * shrink / 2.0                      # shrink the window around the current optimum
        lo, hi = max(lo, best_v - half), min(hi, best_v + half)
    return result


def best_action_generative(outcome_fn, propose_fn, utility, *, mutate_fn=None, rounds=3, k=6, keep=2,
                           objective=None, baseline=None, seed=0, **kw) -> DecisionResult:
    """Open-ended action search: PROPOSE k candidate actions (the LLM generator — it has read how people
    pitch/price/negotiate), SCORE them by simulation (best-arm race), then MUTATE the top `keep` ('shorter',
    'lead with the ask') and re-score, for `rounds` rounds. Generation is the LLM; selection is always
    simulation. `propose_fn(seed) -> list[Action]`; `mutate_fn(list[Action], seed) -> list[Action]` (an LLM
    refinement seam; if None, the pool is fixed and this reduces to one race over the proposals)."""
    pool = list(propose_fn(seed))
    result = None
    for r in range(max(1, rounds)):
        result = best_action(outcome_fn, pool, utility, objective=objective, baseline=baseline, seed=seed + r,
                             navigate=(r == rounds - 1), **kw)
        if mutate_fn is None or r == rounds - 1:
            break
        survivors = [a.action for a in result.ranking[:keep]]
        pool = survivors + list(mutate_fn(survivors, seed + r + 1))    # keep the best, mutate around them
    return result


@dataclass
class _ValueArm:
    var: str
    value: float
    outcome_fn: object

    @property
    def label(self):
        return f"{self.var}={round(self.value, 4)}"


def compare_actions(outcome_fn, a, b, utility, *, n=4000, seed=0, objective=None, conf=0.9) -> dict:
    """do(A) vs do(B) with COMMON RANDOM NUMBERS: each sample draws the same underlying world for both
    actions, so the paired difference isolates the action effect (variance reduction). Returns the mean
    delta with a CI and the preferred action."""
    from statistics import NormalDist
    objective = objective or Mean()
    ua, ub, diffs = [], [], []
    for i in range(n):
        s = (seed << 20) ^ i
        va = utility(outcome_fn(a, Random(s))[0])
        vb = utility(outcome_fn(b, Random(s))[0])            # SAME seed => shared world (CRN pairing)
        ua.append(va); ub.append(vb); diffs.append(va - vb)
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    h = NormalDist().inv_cdf(1 - (1 - conf) / 2) * (var / n) ** 0.5
    return {"a": a.label, "b": b.label, "a_value": round(objective.value(ua), 4),
            "b_value": round(objective.value(ub), 4), "delta_mean": round(m, 4),
            "ci": [round(m - h, 4), round(m + h, 4)], "paired": True,
            "prefer": a.label if m >= 0 else b.label,
            "significant": (m - h > 0) or (m + h < 0)}
