"""The transition operator — calibrated forward DYNAMICS, the fidelity frontier past state grounding.

State grounding fixes the INITIAL CONDITION (what the world is now). But a forecast is initial_condition +
TRAJECTORY, and the Monte-Carlo has been rolling variables forward with GUESSED drift and volatility. A
perfectly-grounded present rolled forward by a naive random walk only ever MATCHES persistence — markets and
persistence already price the present. The edge over the crowd, and the edge GROWING with horizon, comes from
simulating the forward EVOLUTION better than the crowd. That is this object.

It learns the conditional transition Δstate = B·φ(state) + ε, ε ~ N(0, Σ), from historical trajectories —
exactly the philosophy used for the weights:

  - φ(state) is a feature map of the current state. The default is linear (an intercept + the centered
    levels) => a VAR(1): Δx = b + A·(x − c). A's off-diagonals are the CROSS-VARIABLE COUPLING (inflation
    pushing the policy rate; unemployment pulling inflation), its diagonal is MEAN-REVERSION vs momentum. A
    richer basis (e.g. a quadratic self-term) lets it learn a saturating / logistic drift — an S-curve.
  - The coefficients are fit by ridge-to-PERSISTENCE: the prior is B = 0, i.e. Δ = 0, i.e. a random walk. So
    the honest null is persistence, and structure only emerges where the data pays for it. An
    empirical-Bayes temper (held-out one-step error) sets the shrinkage, so THIN data collapses back to
    persistence (no hallucinated drift) and RICH data lets the coupling through — the n-adaptive discipline.
  - Σ is the innovation covariance from the residuals (correlated shocks across variables). Rolling H unit
    steps composes H innovations, so the forecast spread grows like √H automatically — calibrated time with
    no assumed scaling.

If the true process is a random walk, A → 0 and the operator IS persistence (no harm). If it mean-reverts or
couples, the operator tracks the trajectory persistence cannot — and the gap widens with the horizon.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.variables.bayes_logistic import _cholesky, _matinv


def linear_basis(centered, raw):
    """VAR(1): intercept + centered levels. Δx = b + A·(x − c); A is drift + cross-variable coupling."""
    return [1.0] + list(centered)


def quadratic_self_basis(centered, raw):
    """Intercept + centered levels + each raw level squared — lets a variable's drift SATURATE (learn the
    logistic Δx ≈ r·x·(1 − x) as b + a·x + q·x², the S-curve a random walk misses)."""
    return [1.0] + list(centered) + [r * r for r in raw]


def _ridge_solve(phi, y, prior_prec):
    """Ridge-to-zero linear least squares: argmin_w ||phi·w − y||² + Σ_k prior_prec[k]·w_k². Closed form
    w = (ΦᵀΦ + diag(prior_prec))⁻¹ Φᵀy — the prior mean is 0, so every coefficient is shrunk toward the
    persistence null (no drift) and only survives where the data supports it."""
    p = len(phi[0])
    ata = [[sum(phi[t][i] * phi[t][j] for t in range(len(phi))) + (prior_prec[i] if i == j else 0.0)
            for j in range(p)] for i in range(p)]
    aty = [sum(phi[t][i] * y[t] for t in range(len(phi))) for i in range(p)]
    inv = _matinv(ata)
    return [sum(inv[i][j] * aty[j] for j in range(p)) for i in range(p)]


@dataclass
class TransitionOperator:
    names: list                              # variable order (state vector layout)
    basis: object = linear_basis             # (centered_vec, raw_vec) -> feature vector
    l2: float = 1.0                          # base ridge strength (scaled to feature magnitude, x EB temper)
    temper_grid: tuple = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    los: list = None                         # per-variable clamp bounds (default: unbounded)
    his: list = None
    center_window: int = None                # if set, revert toward a TRAILING-mean level (window W), not a
    #                                          fixed global mean — the honest choice for NON-STATIONARY series
    #                                          (a stale global center reverts toward a dead level and hurts).
    # fitted:
    centers: list = None
    coef: list = None                        # p x n_vars  (feature-major): Δ_i = Σ_k coef[k][i]·φ_k
    Sigma: list = None                       # n_vars x n_vars innovation covariance
    L: list = None                           # cholesky(Sigma)
    temper: float = 1.0
    n_pairs: int = 0
    p_feats: int = 0

    def _pairs(self, trajectories):
        """(x_t, Δ_t = x_{t+1} − x_t, center_t) triples, built WITHIN each trajectory only. center_t is the
        trailing-mean level (window W) as of t when `center_window` is set (a leakage-free local reversion
        target), else None (the global mean is used)."""
        X, D, C = [], [], []
        W = self.center_window
        for series in trajectories:
            for t in range(len(series) - 1):
                x = [float(series[t][n]) for n in self.names]
                nx = [float(series[t + 1][n]) for n in self.names]
                X.append(x)
                D.append([nx[j] - x[j] for j in range(len(self.names))])
                if W:
                    win = series[max(0, t - W + 1):t + 1]
                    C.append([sum(float(r[n]) for r in win) / len(win) for n in self.names])
        return X, D, (C if W else None)

    def trailing_center(self, history):
        """The leakage-free local reversion target at a forecast origin: the trailing-mean level over the last
        W observations of the KNOWN history (a list of state-dicts up to and including the origin)."""
        W = self.center_window or len(history)
        win = history[-W:]
        return {n: sum(float(r[n]) for r in win) / len(win) for n in self.names}

    def _phi(self, x, center=None):
        c = center if center is not None else self.centers
        return self.basis([x[j] - c[j] for j in range(len(x))], x)

    def _fit_coef(self, Phi, D, prec):
        nv = len(self.names)
        return [_ridge_solve(Phi, [D[t][i] for t in range(len(D))], prec) for i in range(nv)]  # nv x p

    def fit(self, trajectories, *, tune=True, seed=0):
        X, D, C = self._pairs(trajectories)
        self.n_pairs = len(X)
        nv = len(self.names)
        self.centers = [sum(x[j] for x in X) / len(X) for j in range(nv)]   # global mean (report + fallback)
        Phi = [self._phi(X[t], C[t] if C else None) for t in range(len(X))]
        self.p_feats = p = len(Phi[0])
        # ridge precision SCALED to each feature's TOTAL ΦᵀΦ diagonal (Σ_t φ_k²), so temper is a meaningful
        # knob relative to the data term: temper≈1 halves a coefficient, temper→large zeros it ⇒ the operator
        # collapses to PERSISTENCE (the honest null). Scaling by the mean (not the sum) would make the penalty
        # ~n× too weak — the operator could never shrink out a spurious dynamic. Intercept barely penalized.
        gram = [sum(Phi[t][k] ** 2 for t in range(len(Phi))) for k in range(p)]
        prec0 = [self.l2 * max(gram[k], 1e-9) for k in range(p)]
        prec0[0] *= 0.01
        self.temper = self._eb_temper(Phi, D, prec0, seed) if tune else 1.0
        prec = [q * self.temper for q in prec0]
        coefs = self._fit_coef(Phi, D, prec)             # nv x p
        self.coef = [[coefs[i][k] for i in range(nv)] for k in range(p)]   # -> p x nv (feature-major)
        # innovation covariance from residuals
        resid = [[D[t][i] - sum(coefs[i][k] * Phi[t][k] for k in range(p)) for i in range(nv)]
                 for t in range(len(D))]
        m = len(resid)
        self.Sigma = [[sum(resid[t][a] * resid[t][b] for t in range(m)) / max(1, m - p)
                       for b in range(nv)] for a in range(nv)]
        for i in range(nv):
            self.Sigma[i][i] = max(self.Sigma[i][i], 1e-9)
        self.L = _cholesky(self.Sigma)
        return self

    def _eb_temper(self, Phi, D, prec0, seed):
        """Pick the ridge multiplier minimizing held-out one-step MSE (last 25% of pairs held out in time)."""
        m = len(Phi)
        cut = max(3, int(0.75 * m))
        if cut >= m:
            return 1.0
        nv = len(self.names)
        best_t, best_e = 1.0, float("inf")
        for t in self.temper_grid:
            prec = [q * t for q in prec0]
            coefs = [_ridge_solve(Phi[:cut], [D[r][i] for r in range(cut)], prec) for i in range(nv)]
            err = sum((D[r][i] - sum(coefs[i][k] * Phi[r][k] for k in range(len(prec)))) ** 2
                      for r in range(cut, m) for i in range(nv))
            if err < best_e:
                best_t, best_e = t, err
        return best_t

    def _clamp(self, vec):
        if self.los is None and self.his is None:
            return vec
        out = list(vec)
        for j in range(len(out)):
            if self.los is not None:
                out[j] = max(self.los[j], out[j])
            if self.his is not None:
                out[j] = min(self.his[j], out[j])
        return out

    def step(self, x, rng, *, noise=True, center=None, gain=None):
        """One calibrated forward step from state vector x -> next state vector. `center` overrides the
        reversion target (the origin's trailing level for a non-stationary series). `gain` (per-variable
        multiplier) scales the learned drift SHAPE by a series-specific rate grounded from recent data —
        keeping the pooled curvature but the entity's own velocity (see `ground_gain`)."""
        phi = self._phi(x, center)
        nv = len(self.names)
        drift = [sum(self.coef[k][i] * phi[k] for k in range(self.p_feats)) for i in range(nv)]
        if gain is not None:
            drift = [drift[i] * gain[i] for i in range(nv)]
        if noise:
            z = [rng.gauss(0, 1) for _ in range(nv)]
            shock = [sum(self.L[i][k] * z[k] for k in range(i + 1)) for i in range(nv)]
        else:
            shock = [0.0] * nv
        return self._clamp([x[i] + drift[i] + shock[i] for i in range(nv)])

    def _center_vec(self, center):
        if center is None:
            return None
        return [float(center[n]) for n in self.names] if isinstance(center, dict) else center

    def _gain_vec(self, gain):
        if gain is None:
            return None
        return [float(gain[n]) for n in self.names] if isinstance(gain, dict) else gain

    def ground_gain(self, history, *, window=6, center=None, clamp=(0.0, 6.0)):
        """GROUND the per-series growth RATE from recent trajectory — the dynamics analog of state grounding.
        The pooled operator supplies the drift SHAPE d_pool(x) (the saturation curvature, transferable across
        series); this measures the entity's OWN rate as the scalar gain γ that best matches its recent observed
        transitions: γ = Σ(Δ_obs · d_pool) / Σ(d_pool²) over the last `window` steps of the KNOWN history
        (leakage-free). Then Δ = γ·d_pool(x) climbs at the series' own velocity but still bends to saturation.
        γ→1 recovers the pooled rate; a near-zero pooled drift (random walk) yields γ=1 (a no-op)."""
        c = self._center_vec(center)
        hist = history[-(window + 1):]
        nv = len(self.names)
        if len(hist) < 2:
            return {n: 1.0 for n in self.names}
        num = [0.0] * nv
        den = [0.0] * nv
        for t in range(len(hist) - 1):
            x = [float(hist[t][n]) for n in self.names]
            phi = self._phi(x, c)
            for i in range(nv):
                d = sum(self.coef[k][i] * phi[k] for k in range(self.p_feats))
                num[i] += (float(hist[t + 1][self.names[i]]) - x[i]) * d
                den[i] += d * d
        return {self.names[i]: (min(clamp[1], max(clamp[0], num[i] / den[i])) if den[i] > 1e-12 else 1.0)
                for i in range(nv)}

    def _relaxed_gain(self, g, k, relax):
        """Gain at rollout step k, relaxing the grounded per-series gain toward 1 (the pooled rate) as
        γ_k = 1 + (γ − 1)·relax^k. relax<1 means: trust the entity's OWN measured rate near-term, fall back
        to the transferable pooled rate long-term (a short window's rate is most informative near-term)."""
        if g is None or relax is None:
            return g
        f = relax ** k
        return [1.0 + (g[j] - 1.0) * f for j in range(len(g))]

    def mean_path(self, state0, steps, *, center=None, gain=None, gain_relax=None):
        """Deterministic (noise-free) H-step trajectory of the state MEAN — the point forecast."""
        c, g = self._center_vec(center), self._gain_vec(gain)
        x = self._clamp([float(state0[n]) for n in self.names])
        path = [dict(zip(self.names, x))]
        for k in range(steps):
            x = self.step(x, None, noise=False, center=c, gain=self._relaxed_gain(g, k, gain_relax))
            path.append(dict(zip(self.names, x)))
        return path

    def rollout(self, state0, steps, *, n=2000, seed=0, center=None, gain=None, gain_relax=None):
        """Monte-Carlo H-step forward simulation. Returns, per variable, the predictive mean + 90% interval
        at the terminal step (spread grows like √H by composing H one-step innovations — calibrated time).
        `center` (dict/list) is the fixed reversion target for the horizon — pass the origin's trailing level
        for a non-stationary series (leakage-free; computed from KNOWN history via `trailing_center`). `gain`
        (dict/list) scales the drift to the entity's own grounded rate (see `ground_gain`); `gain_relax` (<1)
        blends that grounded rate back toward the pooled rate over the horizon (grounded-short, pooled-long)."""
        rng = random.Random(seed)
        c, g = self._center_vec(center), self._gain_vec(gain)
        nv = len(self.names)
        terminal = [[] for _ in range(nv)]
        for _ in range(n):
            x = self._clamp([float(state0[nm]) for nm in self.names])
            for k in range(steps):
                x = self.step(x, rng, center=c, gain=self._relaxed_gain(g, k, gain_relax))
            for j in range(nv):
                terminal[j].append(x[j])
        out = {}
        for j, nm in enumerate(self.names):
            s = sorted(terminal[j])
            out[nm] = {"mean": sum(s) / len(s), "p05": s[max(0, int(0.05 * len(s)))],
                       "p95": s[min(len(s) - 1, int(0.95 * len(s)))]}
        return out

    def coupling_report(self):
        """The learned linear coupling A (row = variable that CHANGES, col = variable DRIVING the change) +
        the intercept drift. Only meaningful for a basis whose features 1..n are the centered levels."""
        nv = len(self.names)
        rep = {"intercept_drift": {self.names[i]: round(self.coef[0][i], 5) for i in range(nv)},
               "coupling": {}, "temper": self.temper, "n_pairs": self.n_pairs}
        if self.p_feats >= nv + 1:
            for i in range(nv):                          # Δ names[i]  <-  level of names[j]
                rep["coupling"][self.names[i]] = {self.names[j]: round(self.coef[1 + j][i], 5)
                                                  for j in range(nv)}
        return rep


def persistence_rollout(state0, names, steps):
    """The baseline the operator must beat: the world stays exactly as grounded (Δ = 0 forever)."""
    return {n: {"mean": float(state0[n]), "p05": float(state0[n]), "p95": float(state0[n])} for n in names}
