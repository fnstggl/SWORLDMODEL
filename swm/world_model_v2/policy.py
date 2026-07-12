"""Universal learned actor-policy layer — Phase 4 (production).

The contract the Enron/BehaviorBench/Upworthy rounds forced:

    actor-observable state + beliefs + goals + constraints + typed feasible actions
    → calibrated DISTRIBUTION over typed actions

The LLM never mints behavioral probabilities. It may parse intent, propose structure, extract semantic
features (which enter only through validated feature registries — see semantic_registry.py). The numbers
come from: (a) typed utility models over action consequences, (b) noisy-rationality choice rules with
FITTED precision, (c) population preference distributions fitted hierarchically (population → domain/game
→ person shrinkage where repeated measures exist), (d) empirically fitted calibration layers.

Pieces here are domain-general; game/domain structure (action spaces, payoff consequences) comes from the
compiler or a reference world. Mechanism families registered from registry/families/choice.py.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------- typed actions + feasibility
@dataclass
class TypedAction:
    action_id: str
    value: object = None                     # numeric level for graded actions (offer=30, contribute=12)
    payload: dict = field(default_factory=dict)


@dataclass
class ActionSpace:
    """A typed, enumerable action space with feasibility masking. Institutional/resource constraints are
    applied by `feasible()` BEFORE any policy sees the actions — infeasible actions get exactly 0 mass."""
    actions: list                             # [TypedAction]
    masks: list = field(default_factory=list)  # [callable(world, actor_id, action) -> bool]

    def feasible(self, world=None, actor_id: str = "") -> list:
        out = []
        for a in self.actions:
            if all(m(world, actor_id, a) for m in self.masks):
                out.append(a)
        return out

    @classmethod
    def numeric_grid(cls, lo: float, hi: float, step: float, action_id: str = "choose") -> "ActionSpace":
        n = int(round((hi - lo) / step))
        return cls(actions=[TypedAction(action_id, value=lo + i * step) for i in range(n + 1)])


# ---------------------------------------------------------------- noisy-rationality choice rules
def logit_choice(utilities: list, precision: float) -> list:
    """Quantal response (McKelvey & Palfrey 1995): p_a ∝ exp(λ·u_a). λ=0 → uniform; λ→∞ → best reply.
    Numerically stable; utilities on the caller's payoff scale (fit λ on the same scale)."""
    if not utilities:
        return []
    m = max(utilities)
    ws = [math.exp(max(-40.0, precision * (u - m))) for u in utilities]
    z = sum(ws)
    return [w / z for w in ws]


def tremble_mix(p: list, eps: float) -> list:
    """Uniform tremble: (1−ε)p + ε·uniform — the standard agnostic error layer."""
    n = len(p)
    return [(1 - eps) * pi + eps / n for pi in p]


# ---------------------------------------------------------------- population preference layer
@dataclass
class PreferenceAtom:
    """One point in a discrete population-preference mixture (e.g. a Fehr-Schmidt (α,β) type)."""
    params: dict
    weight: float


@dataclass
class PopulationPreferences:
    """Discrete mixture over preference types. Provenance REQUIRED: published pack, fitted pack, or
    prior — never invisible defaults. Fit with fit_mixture_weights()."""
    atoms: list                                # [PreferenceAtom]
    source: str = "unsupported"
    fitted_on: str = ""

    def normalized(self) -> "PopulationPreferences":
        z = sum(a.weight for a in self.atoms) or 1.0
        for a in self.atoms:
            a.weight /= z
        return self

    def sample(self, rng: random.Random) -> dict:
        r, acc = rng.random(), 0.0
        for a in self.atoms:
            acc += a.weight
            if r <= acc:
                return a.params
        return self.atoms[-1].params


def mixture_action_dist(pop: PopulationPreferences, per_type_dist) -> dict:
    """Population choice distribution: P(a) = Σ_types w_t · P_t(a). per_type_dist(params) → {value: p}."""
    out = {}
    for atom in pop.atoms:
        d = per_type_dist(atom.params)
        for v, p in d.items():
            out[v] = out.get(v, 0.0) + atom.weight * p
    z = sum(out.values()) or 1.0
    return {v: p / z for v, p in out.items()}


# ---------------------------------------------------------------- distribution utilities (exact, discrete)
def dist_to_cdf(dist: dict) -> list:
    """{value: p} → [(value, cum_p)] sorted by value."""
    acc, out = 0.0, []
    for v in sorted(dist):
        acc += dist[v]
        out.append((v, acc))
    return out


def w1_dist_sample(dist: dict, sample: list, lo: float, hi: float) -> float:
    """Exact Wasserstein-1 between a discrete predicted distribution and an empirical sample:
    ∫|F_pred − F_emp| dx over the union grid."""
    if not dist or not sample:
        return float("inf")
    xs = sorted(set(list(dist.keys()) + list(sample) + [lo, hi]))
    svals = sorted(sample)
    n = len(svals)
    cdf_p = dist_to_cdf(dist)
    area, j_p = 0.0, 0
    import bisect
    for i in range(len(xs) - 1):
        x = xs[i]
        while j_p < len(cdf_p) and cdf_p[j_p][0] <= x:
            j_p += 1
        Fp = cdf_p[j_p - 1][1] if j_p > 0 else 0.0
        Fe = bisect.bisect_right(svals, x) / n
        area += abs(Fp - Fe) * (xs[i + 1] - x)
    return area


def sample_from_dist(dist: dict, rng: random.Random, n: int) -> list:
    vals = list(dist.keys())
    cum, acc = [], 0.0
    for v in vals:
        acc += dist[v]
        cum.append(acc)
    out = []
    for _ in range(n):
        r = rng.random() * acc
        import bisect
        out.append(vals[min(len(vals) - 1, bisect.bisect_left(cum, r))])
    return out


# ---------------------------------------------------------------- hierarchical mixture fitting
def fit_mixture_weights(train_samples: dict, per_game_type_dist, atoms: list, *,
                        lo_hi: dict, iters=200, lr=0.5, pool_strength=0.0, seed=0) -> list:
    """Fit mixture weights over preference atoms by minimizing Σ_games W1(pred_g, train_g) with
    exponentiated-gradient updates (finite differences — W1 is piecewise linear in w). PARTIAL POOLING:
    one weight vector serves all games in `train_samples` (population-level parameters); game-level
    response params (λ, tremble) are fitted separately by profile. pool_strength>0 shrinks toward uniform.

    train_samples: {game: [values]}; per_game_type_dist: (game, params) → {value: p};
    atoms: [dict params]. Returns fitted weights (list, sums to 1)."""
    k = len(atoms)
    w = [1.0 / k] * k

    def objective(wv):
        tot = 0.0
        for g, sample in train_samples.items():
            pop = PopulationPreferences(
                atoms=[PreferenceAtom(a, wi) for a, wi in zip(atoms, wv)]).normalized()
            d = mixture_action_dist(pop, lambda prm: per_game_type_dist(g, prm))
            lo, hi = lo_hi[g]
            tot += w1_dist_sample(d, sample, lo, hi) / (hi - lo)
        if pool_strength > 0:
            tot += pool_strength * sum((wi - 1.0 / k) ** 2 for wi in wv)
        return tot

    f0 = objective(w)
    for it in range(iters):
        # finite-difference gradient in log-weight space
        g = [0.0] * k
        h = 0.02
        base = objective(w)
        for i in range(k):
            wp = list(w)
            wp[i] *= math.exp(h)
            z = sum(wp)
            wp = [x / z for x in wp]
            g[i] = (objective(wp) - base) / h
        # exponentiated gradient step
        w = [wi * math.exp(-lr * gi) for wi, gi in zip(w, g)]
        z = sum(w)
        w = [wi / z for wi in w]
        if it % 50 == 49:
            lr *= 0.7
    return w


def profile_scalar(objective, grid) -> tuple:
    """1-D profile fit: return (best_value, {value: objective})."""
    scores = {v: objective(v) for v in grid}
    best = min(scores, key=scores.get)
    return best, scores


# ---------------------------------------------------------------- distributional calibration
def pit_calibration(dist: dict, test_sample: list) -> dict:
    """Probability-integral-transform check of a predicted distribution against held-out points:
    PIT_i = F_pred(x_i) (randomized for discrete ties). Uniform PITs = calibrated. Reports KS distance
    from U(0,1) and central-interval coverage (50%, 90%)."""
    if not dist or not test_sample:
        return {"ks": None}
    cdf = dist_to_cdf(dist)
    rng = random.Random(7)
    pits = []
    for x in test_sample:
        f_below = 0.0
        f_at = 0.0
        for v, c in cdf:
            if v < x:
                f_below = c
            elif v == x:
                f_at = c - f_below
                break
        pits.append(min(1.0, f_below + rng.random() * f_at if f_at > 0 else f_below))
    pits.sort()
    n = len(pits)
    ks = max(max(abs((i + 1) / n - p), abs(i / n - p)) for i, p in enumerate(pits))
    cov50 = sum(1 for p in pits if 0.25 <= p <= 0.75) / n
    cov90 = sum(1 for p in pits if 0.05 <= p <= 0.95) / n
    return {"ks": round(ks, 4), "coverage_50": round(cov50, 3), "coverage_90": round(cov90, 3), "n": n}
