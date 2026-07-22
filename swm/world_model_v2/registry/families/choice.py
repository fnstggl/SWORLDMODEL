"""Decision/choice mechanism families — Phase 4/6: utility-based choice, quantal response,
social preferences (Fehr-Schmidt inequity aversion), Poisson cognitive hierarchy, conditional
cooperation — with the BehaviorBench game structures as typed scenario instantiations.

The design replaces per-game hand-crafted decision rules with ONE population-preference model shared
across games: a discrete mixture over Fehr-Schmidt types (α = disadvantageous-inequity aversion,
β = advantageous-inequity aversion) drives every two-player game through the SAME utility machinery;
games differ only in their typed action space and payoff consequences (scenario structure available to
every arm, including the LLM prompts). Choice noise is quantal response (logit) with per-game precision
fitted on train; cross-game transfer uses payoff-scale-normalized pooled precision.

Cross-game INTERACTION is structural: the proposer's acceptance beliefs come from the SAME preference
mixture's responder thresholds; the investor's return beliefs from the banker's QRE distribution.

Published parameter packs (with exact study limits):
  * Fehr & Schmidt 1999 (QJE 114:817-868) calibrated type distribution:
      α ∈ {0,0.5,1,4} weights (.3,.3,.3,.1); β ∈ {0,0.25,0.6} weights (.3,.3,.4)
    — calibrated on 1990s lab ultimatum data (mostly student subjects), NOT a universal law.
  * Camerer, Ho & Chong 2004 (QJE 119:861-898) Poisson-CH τ ≈ 1.5 (median across games/samples).
  * Fischbacher, Gächter & Fehr 2001 (Econ Letters 71:397-404): ~50% conditional cooperators,
    ~30% free riders; conditional slope slightly below 1 (self-serving bias).
All transported packs carry widened uncertainty; local fitted packs are preferred where train data exists.
"""
from __future__ import annotations

import math
import random

from swm.world_model_v2.policy import (ActionSpace, PopulationPreferences, PreferenceAtom,
                                       logit_choice, mixture_action_dist, profile_scalar,
                                       fit_mixture_weights, w1_dist_sample)

# ---------------------------------------------------------------- Fehr-Schmidt utility core
def fs_utility(own: float, other: float, alpha: float, beta: float) -> float:
    """u = x_own − α·max(x_other−x_own, 0) − β·max(x_own−x_other, 0)  (two-player FS 1999)."""
    return own - alpha * max(other - own, 0.0) - beta * max(own - other, 0.0)


FS_ALPHAS = (0.0, 0.5, 1.0, 4.0)
FS_BETAS = (0.0, 0.25, 0.6)
FS_PUBLISHED_W = {"alpha": {0.0: 0.3, 0.5: 0.3, 1.0: 0.3, 4.0: 0.1},
                  "beta": {0.0: 0.3, 0.25: 0.3, 0.6: 0.4}}


def fs_atoms() -> list:
    """The 12-point (α,β) product grid. FS 1999 treated α,β as comonotonic; the product grid is a
    structural choice logged in the registry record (limits: independence assumption)."""
    return [{"alpha": a, "beta": b} for a in FS_ALPHAS for b in FS_BETAS]


def fs_published_pop() -> PopulationPreferences:
    atoms = [PreferenceAtom({"alpha": a, "beta": b},
                            FS_PUBLISHED_W["alpha"][a] * FS_PUBLISHED_W["beta"][b])
             for a in FS_ALPHAS for b in FS_BETAS]
    return PopulationPreferences(atoms=atoms, source="published_research (Fehr-Schmidt 1999 QJE)",
                                 fitted_on="1990s lab ultimatum data — student subjects").normalized()


# ---------------------------------------------------------------- game structures (typed scenarios)
GAME_GRID = {
    "dictator":            (0.0, 100.0, 5.0),
    "ultimatum_responder": (0.0, 100.0, 5.0),
    "ultimatum_proposer":  (0.0, 100.0, 5.0),
    "trust_investor":      (0.0, 100.0, 5.0),
    "trust_banker":        (0.0, 150.0, 5.0),
    "guessing":            (0.0, 100.0, 1.0),
    "public_goods":        (0.0, 20.0, 1.0),
}
INTERACTION_GAMES = ("ultimatum_proposer", "trust_investor", "guessing", "public_goods")


def responder_threshold(alpha: float) -> float:
    """FS min-acceptable offer of 100: accept o iff o − α(100−2o) ≥ 0 → t* = 100α/(1+2α)."""
    return 100.0 * alpha / (1.0 + 2.0 * alpha)


def _gauss_dist_on_grid(mu: float, sd: float, lo: float, hi: float, step: float) -> dict:
    """Discretized truncated normal on the action grid (closed form via erf)."""
    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    out = {}
    v = lo
    while v <= hi + 1e-9:
        a, b = v - step / 2, v + step / 2
        p = Phi((min(hi, b) - mu) / sd) - Phi((max(lo, a) - mu) / sd)
        out[round(v, 4)] = max(0.0, p)
        v += step
    z = sum(out.values()) or 1.0
    return {k: p / z for k, p in out.items()}


class GamePolicyModel:
    """The shared-preference QRE policy over every BehaviorBench game. All response params carry
    provenance; `context` holds cross-game beliefs (responder thresholds, banker returns, PG belief)."""

    def __init__(self, pop: PopulationPreferences, *, lam: dict, sd: dict,
                 tau_ch: float = 1.5, tau_src: str = "published_research (CHC 2004)",
                 pg_cond_share: float = 0.5, pg_slope: float = 0.8,
                 pg_src: str = "published_research (FGF 2001)"):
        self.pop = pop.normalized()
        self.lam = lam                        # per-game QRE precision {game: λ} (fitted or pooled)
        self.sd = sd                          # per-game report/tremble sd {game: σ}
        self.tau_ch = tau_ch
        self.tau_src = tau_src
        self.pg_cond_share = pg_cond_share
        self.pg_slope = pg_slope
        self.pg_src = pg_src
        self._cache = {}

    # -------- per-type QRE choice distributions (closed form on the action grid) --------
    def type_dist(self, game: str, prm: dict) -> dict:
        lo, hi, step = GAME_GRID[game]
        a, b = prm["alpha"], prm["beta"]
        lam = self.lam[game]
        if game == "dictator":
            grid = _grid(lo, hi, step)
            us = [fs_utility(100.0 - g, g, a, b) for g in grid]
            return dict(zip(grid, logit_choice(us, lam)))
        if game == "ultimatum_responder":
            t = responder_threshold(a)
            return _gauss_dist_on_grid(t, max(2.0, self.sd[game]), lo, hi, step)
        if game == "trust_banker":
            grid = _grid(lo, hi, step)
            us = [fs_utility(150.0 - r, 50.0 + r, a, b) for r in grid]
            return dict(zip(grid, logit_choice(us, lam)))
        if game == "ultimatum_proposer":
            p_acc = self._acceptance_curve()
            grid = _grid(lo, hi, step)
            us = [p_acc[o] * fs_utility(100.0 - o, o, a, b) for o in grid]
            return dict(zip(grid, logit_choice(us, lam)))
        if game == "trust_investor":
            phis = self._banker_return_fractions()
            grid = _grid(lo, hi, step)
            us = []
            for x in grid:
                eu = 0.0
                for phi, w in phis:
                    back = 3.0 * x * phi
                    eu += w * fs_utility(100.0 - x + back, 3.0 * x - back, a, b)
                us.append(eu)
            return dict(zip(grid, logit_choice(us, lam)))
        if game == "guessing":
            return self._ch_guessing_dist()
        if game == "public_goods":
            return self._pg_type_dist(a, b)
        raise KeyError(game)

    def population_dist(self, game: str) -> dict:
        key = ("pop", game)
        if key not in self._cache:
            self._cache[key] = mixture_action_dist(self.pop, lambda prm: self.type_dist(game, prm))
        return self._cache[key]

    # -------- cross-game interaction structures --------
    def _acceptance_curve(self) -> dict:
        """P(accept | offer o) from the SAME preference mixture's responder thresholds + report noise —
        the proposer's simulated partner (interaction mechanism)."""
        key = "acc_curve"
        if key not in self._cache:
            lo, hi, step = GAME_GRID["ultimatum_proposer"]
            grid = _grid(lo, hi, step)
            sd = max(2.0, self.sd.get("ultimatum_responder", 8.0))
            curve = {o: 0.0 for o in grid}
            for atom in self.pop.atoms:
                t = responder_threshold(atom.params["alpha"])
                for o in grid:
                    z = (o - t) / (sd * math.sqrt(2.0))
                    curve[o] += atom.weight * 0.5 * (1.0 + math.erf(z))
            self._cache[key] = curve
        return self._cache[key]

    def _banker_return_fractions(self, n_pts: int = 9) -> list:
        """(φ, weight) support of the banker's return fraction implied by the banker QRE distribution —
        the investor's simulated partner. Transport limit: banker behavior observed at x=50 only;
        assuming fraction-invariance across x is a logged assumption."""
        key = "banker_phi"
        if key not in self._cache:
            d = self.population_dist("trust_banker")
            pts = sorted(d.items())
            phis = [(min(1.0, r / 150.0), p) for r, p in pts]
            agg = {}
            for phi, p in phis:
                bucket = round(phi * (n_pts - 1)) / (n_pts - 1)
                agg[bucket] = agg.get(bucket, 0.0) + p
            self._cache[key] = sorted(agg.items())
        return self._cache[key]

    def _ch_guessing_dist(self) -> dict:
        """Poisson cognitive hierarchy (CHC 2004) on the 2/3-average game: level-0 uniform; level-k
        best-responds to the Poisson(τ)-truncated distribution of lower levels; report noise σ."""
        key = ("ch", self.tau_ch)
        if key not in self._cache:
            lo, hi, step = GAME_GRID["guessing"]
            tau = self.tau_ch
            max_k = 6
            pk = [math.exp(-tau) * tau ** k / math.factorial(k) for k in range(max_k + 1)]
            z = sum(pk)
            pk = [p / z for p in pk]
            means = [50.0]                         # level-0 mean
            for k in range(1, max_k + 1):
                wz = sum(pk[:k]) or 1.0
                mean_lower = sum(pk[j] * means[j] for j in range(k)) / wz
                means.append((2.0 / 3.0) * mean_lower)
            sd = max(1.5, self.sd.get("guessing", 6.0))
            out = {}
            for k, w in enumerate(pk):
                d = _gauss_dist_on_grid(means[k] if k > 0 else 50.0, 29.0 if k == 0 else sd,
                                        lo, hi, step)
                for v, p in d.items():
                    out[v] = out.get(v, 0.0) + w * p
            zz = sum(out.values()) or 1.0
            self._cache[key] = {v: p / zz for v, p in out.items()}
        return self._cache[key]

    def _pg_type_dist(self, alpha: float, beta: float) -> dict:
        """Public goods (n=4, MPCR=0.5): free-ride if β low (selfish dominant: own return 0.5<1);
        conditional cooperators (β high) match believed others' mean × slope (FGF 2001). Belief =
        self-consistent fixed point of the population's own mean contribution."""
        lo, hi, step = GAME_GRID["public_goods"]
        belief = self._pg_belief()
        sd = max(1.0, self.sd.get("public_goods", 3.0))
        if beta >= 0.5:                            # conditional cooperator type
            mu = min(hi, self.pg_slope * belief)
            return _gauss_dist_on_grid(mu, sd, lo, hi, step)
        if beta >= 0.25:                           # weak reciprocator: half-matching
            mu = min(hi, 0.5 * self.pg_slope * belief)
            return _gauss_dist_on_grid(mu, sd, lo, hi, step)
        return _gauss_dist_on_grid(0.0, sd, lo, hi, step)   # free rider

    def _pg_belief(self, iters: int = 10) -> float:
        key = "pg_belief"
        if key not in self._cache:
            m = 10.0
            for _ in range(iters):
                tot, wz = 0.0, 0.0
                for atom in self.pop.atoms:
                    b = atom.params["beta"]
                    if b >= 0.5:
                        c = self.pg_slope * m
                    elif b >= 0.25:
                        c = 0.5 * self.pg_slope * m
                    else:
                        c = 0.0
                    tot += atom.weight * c
                    wz += atom.weight
                m = tot / max(1e-9, wz)
            self._cache[key] = m
        return self._cache[key]

    def invalidate(self):
        self._cache = {}


def _grid(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


# ---------------------------------------------------------------- fitting (train only)
#: payoff scale per game — λ transfers across games as λ·scale = const (precision is payoff-relative)
PAYOFF_SCALE = {"dictator": 100.0, "ultimatum_responder": 100.0, "ultimatum_proposer": 100.0,
                "trust_investor": 100.0, "trust_banker": 150.0, "guessing": 100.0, "public_goods": 20.0}

LAM_GRID = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)     # on a 100-point payoff scale
SD_FRAC_GRID = (0.03, 0.06, 0.10, 0.16, 0.25)   # tremble sd as fraction of range


def fit_game_policy(train_by_game: dict, *, games=None, seed=0,
                    fit_weights=True, base_pop: PopulationPreferences | None = None) -> GamePolicyModel:
    """Hierarchical fit: (1) game-level response params (λ, σ, τ, PG slope) profiled per game on train;
    (2) population-level FS mixture weights fitted JOINTLY across the given games (partial pooling).
    Games absent from train_by_game keep pooled/published response params — that is the cold-start path
    used by leave-one-game-out transfer."""
    games = games or [g for g in GAME_GRID if g in train_by_game]
    pop = (base_pop or fs_published_pop())
    # λ stated on a 100-point payoff scale; per-game λ_g = λ100 · 100/scale_g (payoff-relative precision)
    lam = {g: LAM_GRID[2] * 100.0 / PAYOFF_SCALE[g] for g in GAME_GRID}
    sd = {g: 0.10 * (GAME_GRID[g][1] - GAME_GRID[g][0]) for g in GAME_GRID}
    model = GamePolicyModel(pop, lam=lam, sd=sd)

    def score(g):
        lo, hi, _ = GAME_GRID[g]
        return w1_dist_sample(model.population_dist(g), train_by_game[g], lo, hi) / (hi - lo)

    # ---- stage 1: per-game response params (with published/pooled preference weights) ----
    for g in games:
        if g not in train_by_game:
            continue
        lo, hi, _ = GAME_GRID[g]

        def obj_lam(lv):
            model.lam[g] = lv * 100.0 / PAYOFF_SCALE[g]
            model.invalidate()
            return score(g)

        best_l, _ = profile_scalar(obj_lam, LAM_GRID)
        model.lam[g] = best_l * 100.0 / PAYOFF_SCALE[g]

        def obj_sd(fr):
            model.sd[g] = fr * (hi - lo)
            model.invalidate()
            return score(g)

        best_f, _ = profile_scalar(obj_sd, SD_FRAC_GRID)
        model.sd[g] = best_f * (hi - lo)
        model.invalidate()
    if "guessing" in games and "guessing" in train_by_game:
        def obj_tau(tv):
            model.tau_ch = tv
            model.invalidate()
            return score("guessing")
        model.tau_ch, _ = profile_scalar(obj_tau, (0.5, 1.0, 1.5, 2.0, 3.0))
        model.tau_src = "fitted (guessing train)"
        model.invalidate()
    if "public_goods" in games and "public_goods" in train_by_game:
        def obj_slope(sv):
            model.pg_slope = sv
            model.invalidate()
            return score("public_goods")
        model.pg_slope, _ = profile_scalar(obj_slope, (0.4, 0.6, 0.8, 1.0))
        model.pg_src = "fitted (public_goods train)"
        model.invalidate()

    # ---- stage 2: population-level FS mixture (partial pooling across FS-structural games) ----
    # two outer rounds: fitted weights update the cross-game BELIEF structures (acceptance curve,
    # banker returns, PG fixed point), which the second round's fit then sees — belief consistency.
    if fit_weights:
        fs_games = [g for g in games if g in train_by_game
                    and g in ("dictator", "ultimatum_responder", "trust_banker",
                              "ultimatum_proposer", "trust_investor", "public_goods")]
        if fs_games:
            atoms = fs_atoms()
            for _round in range(2):
                w = fit_mixture_weights(
                    {g: train_by_game[g] for g in fs_games},
                    lambda g, prm: _type_dist_with(model, g, prm),
                    atoms, lo_hi={g: (GAME_GRID[g][0], GAME_GRID[g][1]) for g in fs_games},
                    iters=40, lr=0.4, pool_strength=0.02, seed=seed)
                model.pop = PopulationPreferences(
                    atoms=[PreferenceAtom(a, wi) for a, wi in zip(atoms, w)],
                    source="fitted (BehaviorBench train, joint across games, EG/W1)",
                    fitted_on=",".join(fs_games)).normalized()
                model.invalidate()
    return model


def _type_dist_with(model: GamePolicyModel, game: str, prm: dict) -> dict:
    return model.type_dist(game, prm)


# ---------------------------------------------------------------- world-runtime sampling path
def sample_policy_action(model: GamePolicyModel, game: str, rng: random.Random) -> float:
    """One actor's draw through the population policy (type draw → QRE draw) — used by the typed
    decision operator in the world-execution path."""
    prm = model.pop.sample(rng)
    d = model.type_dist(game, prm)
    r, acc = rng.random(), 0.0
    for v, p in d.items():
        acc += p
        if r <= acc:
            return v
    return list(d)[-1]
