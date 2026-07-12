"""Diffusion mechanism families — Phase 7 (nonlinear, context-dependent contagion).

The Higgs round proved a linear-in-exposure hazard is WRONG (fitted logistic learned concave exposure
response + negative degree effect; the linear world was significantly worse, Δ+0.00234 [0.00117,0.00349]).
These families make the hazard a fitted, nonlinear, heterogeneous intensity that still EXECUTES inside the
shared world: activation is a survival process λ_i(t) integrated event-by-event through the window, with
exposure k_i(t) evolving as in-sample followees activate (rollout), exposures aging (information aging),
and per-user frailty (susceptibility heterogeneity). Statistical fits estimate transition hazards — an
allowed internal role; the prediction is still read from terminal survival state, never from a bypass.

Families here:
  simple_contagion_hazard   λ = q·k                      (the prior linear form — kept as comparator)
  complex_contagion_hazard  λ = exp(θ0)·k^α/(c^α+k^α)    (Hill: threshold-like for α>1, saturating)
  exposure_response_hazard  λ = exp(θ·x), x = [1, log1p k, log1p deg, k/deg, recency]
                            (log-linear intensity: concavity + degree-conditioned susceptibility)
  susceptibility_frailty    multiplicative lognormal frailty, σ FITTED by profile likelihood
  information_aging         age-weighted exposure k_eff = Σ_j exp(−age_j/τ), τ fitted on a grid
  hawkes_self_excitation    λ(t) = μ + α·ω·Σ exp(−ω(t−t_i)) on the aggregate activity stream

All parameters are labeled `fitted` (train split only). Transport limits: fits are per-cascade
(this campaign, this network); moving to another cascade requires refit or transport widening.
"""
from __future__ import annotations

import math
import random

from swm.world_model_v2.registry.ingestion import (fit_bernoulli_hazard, hazard_lambda,
                                                   marginal_window_p, profile_frailty)

_GH = [(-2.3506049736745, 0.019111580500770), (-1.3358490740137, 0.13383774880098),
       (-0.4360774119276, 0.44648878212421), (0.4360774119276, 0.44648878212421),
       (1.3358490740137, 0.13383774880098), (2.3506049736745, 0.019111580500770)]


# ------------------------------------------------------------------ hazard forms (the equations)
def feats_hazard(k: float, deg: float, recency_h: float) -> list:
    """The exposure-response feature map. x[0]=1 intercept; recency in HOURS since last new exposure."""
    return [1.0, math.log1p(max(0.0, k)), math.log1p(max(0.0, deg)),
            min(5.0, k / max(1.0, deg)), math.exp(-max(0.0, recency_h) / 24.0)]


class LinearHazard:
    """λ = q·k — simple contagion, independent per-exposure transmission (comparator; prior V2 form)."""
    form_id = "simple_contagion_hazard"

    def __init__(self, q: float):
        self.q = q

    def lam(self, k, deg, recency_h):
        return self.q * max(0.0, k)

    def params(self):
        return {"q": self.q}


class HillHazard:
    """λ = exp(θ0)·k^α/(c^α+k^α): α>1 → complex contagion (superlinear onset, needs social
    reinforcement); saturates at high k (diminishing returns). Fitted by grid+profile."""
    form_id = "complex_contagion_hazard"

    def __init__(self, theta0: float, alpha: float, c: float):
        self.theta0, self.alpha, self.c = theta0, alpha, c

    def lam(self, k, deg, recency_h):
        k = max(0.0, k)
        if k <= 0:
            return 0.0
        ka = k ** self.alpha
        return math.exp(self.theta0) * ka / (self.c ** self.alpha + ka)

    def params(self):
        return {"theta0": self.theta0, "alpha": self.alpha, "c": self.c}


class LogLinearHazard:
    """λ = exp(θ·x) over feats_hazard — concave exposure response (via log1p), degree-conditioned
    susceptibility, recency effect. The GLM estimates the transition hazard; the world integrates it."""
    form_id = "exposure_response_hazard"

    def __init__(self, theta: list):
        self.theta = list(theta)

    def lam(self, k, deg, recency_h):
        return hazard_lambda(self.theta, feats_hazard(k, deg, recency_h))

    def params(self):
        return {"theta": [round(t, 5) for t in self.theta]}


# ------------------------------------------------------------------ fits (train only)
def fit_linear_q(train_rows, window_days: float) -> LinearHazard:
    """Moment-match mean(1−exp(−q·k·W)) = train rate (the prior fit, kept identical)."""
    rate = sum(r["y"] for r in train_rows) / max(1, len(train_rows))
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        pred = sum(1 - math.exp(-mid * r["k"] * window_days) for r in train_rows) / len(train_rows)
        if pred < rate:
            lo = mid
        else:
            hi = mid
    return LinearHazard((lo + hi) / 2)


def fit_hill(train_rows, window_days: float, *, alphas=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
             cs=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0)) -> HillHazard:
    """Grid over (α, c); θ0 profiled by bisection to match the train rate at each grid point; pick the
    grid point maximizing train Bernoulli log-likelihood."""
    ys = [r["y"] for r in train_rows]
    best = None
    for a in alphas:
        for c in cs:
            g = [(max(1e-9, r["k"]) ** a) / (c ** a + max(1e-9, r["k"]) ** a) if r["k"] > 0 else 0.0
                 for r in train_rows]
            rate = sum(ys) / len(ys)
            lo, hi = -12.0, 2.0
            for _ in range(50):
                mid = (lo + hi) / 2
                pred = sum(1 - math.exp(-math.exp(mid) * gi * window_days) for gi in g) / len(g)
                if pred < rate:
                    lo = mid
                else:
                    hi = mid
            th0 = (lo + hi) / 2
            ll = 0.0
            for gi, y in zip(g, ys):
                p = min(1 - 1e-9, max(1e-9, 1 - math.exp(-math.exp(th0) * gi * window_days)))
                ll += y * math.log(p) + (1 - y) * math.log(1 - p)
            if best is None or ll > best[0]:
                best = (ll, HillHazard(th0, a, c))
    return best[1]


def fit_loglinear(train_rows, window_days: float) -> LogLinearHazard:
    X = [feats_hazard(r["k"], r["deg"], r["recency_h"]) for r in train_rows]
    Y = [r["y"] for r in train_rows]
    return LogLinearHazard(fit_bernoulli_hazard(X, Y, window_days))


def fit_frailty_sigma(hz, train_rows, window_days: float, *, k_key: str = "k"):
    """Susceptibility-heterogeneity sd, FITTED by profile likelihood (never assumed). `k_key` selects the
    exposure column the hazard was fitted on ("k" raw, "k_eff0" age-weighted)."""
    X = [feats_hazard(r.get(k_key, r["k"]), r["deg"], r["recency_h"]) for r in train_rows]
    Y = [r["y"] for r in train_rows]
    if isinstance(hz, LogLinearHazard):
        return profile_frailty(hz.theta, X, Y, window_days)
    # generic: evaluate hz directly
    lls = {}
    for s in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        ll = 0.0
        for r, y in zip(train_rows, Y):
            p = marginal_window_p(hz.lam(r.get(k_key, r["k"]), r["deg"], r["recency_h"]), window_days, s)
            p = min(1 - 1e-9, max(1e-9, p))
            ll += y * math.log(p) + (1 - y) * math.log(1 - p)
        lls[s] = ll
    best = max(lls, key=lls.get)
    return best, {str(k): round(v, 2) for k, v in lls.items()}


def fit_aging_tau(train_rows_aged, window_days: float, *, taus=(6.0, 12.0, 24.0, 48.0)):
    """Information aging: choose exposure-age half-life τ (hours) on train by refitting the log-linear
    hazard with k replaced by the age-weighted k_τ and comparing train likelihood. Rows must carry
    r['k_tau'][τ]. Returns (best_tau, fitted LogLinearHazard on k_τ, {tau: ll})."""
    Y = [r["y"] for r in train_rows_aged]
    out = {}
    best = None
    for tau in taus:
        X = [feats_hazard(r["k_tau"][tau], r["deg"], r["recency_h"]) for r in train_rows_aged]
        th = fit_bernoulli_hazard(X, Y, window_days)
        ll = 0.0
        for x, y in zip(X, Y):
            p = 1 - math.exp(-min(50.0, hazard_lambda(th, x) * window_days))
            p = min(1 - 1e-9, max(1e-9, p))
            ll += y * math.log(p) + (1 - y) * math.log(1 - p)
        out[tau] = round(ll, 2)
        if best is None or ll > best[0]:
            best = (ll, tau, LogLinearHazard(th))
    return best[1], best[2], out


# ------------------------------------------------------------------ the executable world transition
def contagion_window_predict(rows, hz, window_days: float, followers_of, *,
                             n_particles=30, seed=0, rollout=True, frailty_sigma=0.0,
                             aging_tau_h=None, latent_scale_sd=0.0):
    """The shared-world execution of ANY hazard form: event-driven survival over the window.
    Per particle: optional global hazard-scale draw (parameter uncertainty, lognormal sd
    `latent_scale_sd`); per user: optional frailty draw ε_i ~ LN(−σ²/2, σ) (heterogeneity).
    Each of 12 steps: hazard λ_i(k_t, deg, recency_t) accumulates into Rao-Blackwellized log-survival;
    sampled activations propagate exposure to in-sample followers (k_t += 1, recency resets, aged
    exposures decay by exp(−dt/τ)). Terminal readout: 1 − survival, averaged over particles.

    This function IS the mechanism execution (typed local state: k, recency, active, log_surv per user;
    machine-readable per-step transitions are the (k, recency) updates). A full-WorldState parity path
    runs in tests on a subsample (see tests/test_diffusion_families.py) to pin equivalence."""
    n_steps = 12
    dt = window_days / n_steps
    dt_h = dt * 24.0
    idx = {r["u"]: i for i, r in enumerate(rows)}
    p_acc = [0.0] * len(rows)
    rng = random.Random(seed)
    decay = math.exp(-dt_h / aging_tau_h) if aging_tau_h else 1.0
    k_key = "k_eff0" if aging_tau_h else "k"
    for pi in range(n_particles):
        scale = math.exp(rng.gauss(0.0, latent_scale_sd)) if latent_scale_sd > 0 else 1.0
        frail = [math.exp(rng.gauss(-frailty_sigma ** 2 / 2, frailty_sigma)) if frailty_sigma > 0 else 1.0
                 for _ in rows]
        k = [float(r.get(k_key, r["k"])) for r in rows]
        rec = [float(r["recency_h"]) for r in rows]
        active = [False] * len(rows)
        log_surv = [0.0] * len(rows)
        for _ in range(n_steps):
            newly = []
            for i, r in enumerate(rows):
                if active[i]:
                    continue
                lam = scale * frail[i] * hz.lam(k[i], r["deg"], rec[i])
                haz = lam * dt
                log_surv[i] -= haz
                if rng.random() < 1.0 - math.exp(-haz):
                    active[i] = True
                    newly.append(r["u"])
            # exposures age; recency grows (information aging + recency dynamics)
            if aging_tau_h:
                for i in range(len(rows)):
                    if not active[i]:
                        k[i] *= decay
            for i in range(len(rows)):
                if not active[i]:
                    rec[i] += dt_h
            if rollout:
                for u in newly:                       # new activations add fresh exposure in-sample
                    for a in followers_of.get(u, []):
                        j = idx.get(a)
                        if j is not None and not active[j]:
                            k[j] += 1.0
                            rec[j] = 0.0
        for i in range(len(rows)):
            p_acc[i] += 1.0 - math.exp(log_surv[i])
    return [min(0.97, max(1e-4, p / n_particles)) for p in p_acc]


def closed_form_window_p(rows, hz, window_days: float, *, frailty_sigma=0.0, aging_tau_h=None):
    """No-rollout ablation: P_i = E_ε[1 − exp(−ε·λ_i·W)] with k frozen at t0 (matches the fit's own
    assumption). Deterministic — isolates what rollout adds."""
    out = []
    k_key = "k_eff0" if aging_tau_h else "k"
    for r in rows:
        lam = hz.lam(float(r.get(k_key, r["k"])), r["deg"], r["recency_h"])
        out.append(min(0.97, max(1e-4, marginal_window_p(lam, window_days, frailty_sigma))))
    return out


# ------------------------------------------------------------------ Hawkes on the activity stream
def fit_hawkes(times, t_start, t_end, *, omegas=(1 / 600.0, 1 / 1800.0, 1 / 3600.0, 1 / 7200.0),
               iters=200):
    """Self-exciting point process on the aggregate activity stream: λ(t) = μ + αω Σ_{t_i<t} e^{−ω(t−t_i)}.
    Exponential-kernel MLE via EM (fixed ω grid; best by likelihood). Pure Python, O(n) per iter via
    recursive kernel sum. Returns (mu, alpha, omega, train_ll)."""
    ts = sorted(t for t in times if t_start <= t < t_end)
    n = len(ts)
    T = t_end - t_start
    if n < 10:
        raise ValueError("too few events to fit a Hawkes process")
    best = None
    for om in omegas:
        mu = 0.5 * n / T
        al = 0.3
        for _ in range(iters):
            # E-step: p_i = prob event i is background; recursive A_i = Σ_{j<i} e^{−ω(t_i−t_j)}
            A = 0.0
            p_bg_sum, p_ex_sum, weighted_dt = 0.0, 0.0, 0.0
            prev_t = None
            for t in ts:
                if prev_t is not None:
                    A = (A + 1.0) * math.exp(-om * (t - prev_t))
                lam_ex = al * om * A
                lam = mu + lam_ex
                p_bg = mu / lam
                p_bg_sum += p_bg
                p_ex_sum += 1.0 - p_bg
                prev_t = t
            # M-step
            mu = p_bg_sum / T
            # branching ratio: expected offspring per event ≈ p_ex_sum / n (kernel integrates to α)
            al = min(0.95, p_ex_sum / n)
        # final log-likelihood
        A, ll, prev_t = 0.0, 0.0, None
        for t in ts:
            if prev_t is not None:
                A = (A + 1.0) * math.exp(-om * (t - prev_t))
            lam = mu + al * om * A
            ll += math.log(max(1e-12, lam))
            prev_t = t
        # compensator ∫λ ≈ μT + α·(n − Σ e^{−ω(T_end−t_i)}) — tail-corrected
        tail = sum(math.exp(-om * (t_end - t)) for t in ts)
        ll -= mu * T + al * (n - tail)
        if best is None or ll > best[3]:
            best = (mu, al, om, ll)
    return best


def hawkes_forecast_counts(mu, al, om, history, t_from, t_to, n_bins, *, n_sims=200, seed=0):
    """Held-out validation harness: simulate the fitted process forward by Ogata thinning and return
    expected counts per bin; the caller compares against actual counts and a Poisson-rate baseline.
    A(t) = Σ e^{−ω(t−t_i)} is maintained recursively (exact for the exponential kernel), so the upper
    bound λ⁺ = μ + αω·A(t⁺) is valid over the decaying interval to the next candidate point."""
    rng = random.Random(seed)
    binw = (t_to - t_from) / n_bins
    acc = [0.0] * n_bins
    hist = sorted(t for t in history if t < t_from)
    A0 = sum(math.exp(-om * (t_from - ti)) for ti in hist[-5000:])
    for _ in range(n_sims):
        t, A = t_from, A0
        while True:
            lam_bar = mu + al * om * A
            dt = rng.expovariate(lam_bar)
            if t + dt >= t_to:
                break
            A *= math.exp(-om * dt)                    # kernel decay over the jump
            t += dt
            lam_t = mu + al * om * A
            if rng.random() < lam_t / lam_bar:         # accept
                A += 1.0
                b = int((t - t_from) / binw)
                if 0 <= b < n_bins:
                    acc[b] += 1.0 / n_sims
    return acc
