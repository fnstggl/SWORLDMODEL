"""Phase-3 posterior propagation through nonlinear forms — Phase 7, Part 12.

THE central correctness hazard of this whole phase: for a nonlinear f, E[f(X)] ≠ f(E[X]). Collapsing the
Phase-3 posterior to its mean and pushing the mean through a Hill / threshold / logistic curve gives the
WRONG answer (Jensen gap), and the error grows exactly where the curvature — the thing Phase 7 exists to
model — is largest. The Phase-3 audit confirmed no generic per-particle evaluator exists in the codebase;
this module adds it.

What it provides:
  * `ParamPosterior` — a thin, uniform adapter over the several Phase-3 posterior representations
    (`PosteriorResult.outcome_rate_particles`, a fitted grid, a {mean,sd,lo,hi} envelope, raw samples), so a
    mechanism can `.sample(n, rng)` any latent the same way.
  * `propagate(form, param_particles, inputs)` — evaluate the form ONCE PER PARTICLE, then aggregate. Returns
    the posterior-correct E[f(X)], its spread, quantiles, AND the Jensen gap vs the naive f(E[X]) so the error
    of the shortcut is measured, never assumed away.
  * `delta_method_gap` — a cheap curvature estimate of the same gap, for the justified-approximation path
    when per-particle evaluation is too expensive; the reported approximation error is the difference.

Nothing here fits or invents a value; it only moves an existing Phase-3 posterior through an existing form.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _weighted_stats(pairs):
    """pairs: [(value, weight)] → (mean, sd, [p10,p50,p90])."""
    z = sum(w for _, w in pairs) or 1.0
    mean = sum(v * w for v, w in pairs) / z
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / z
    xs = sorted(pairs)

    def q(t):
        acc = 0.0
        for v, w in xs:
            acc += w / z
            if acc >= t:
                return v
        return xs[-1][0]
    return mean, math.sqrt(max(0.0, var)), [q(0.1), q(0.5), q(0.9)]


@dataclass
class ParamPosterior:
    """Uniform adapter over a named latent's posterior. Exactly one representation is populated.

    particles : [(value, weight)]     — e.g. PosteriorResult.outcome_rate_particles (correlation-preserving
                                         when drawn jointly; see `joint`)
    grid      : ([values],[weights])  — a fitted 1-D posterior grid (ContinuousBetaRepresentation.grid/.w)
    envelope  : {mean,sd,lo,hi}       — a marginal Gaussian/Beta envelope (ParameterEstimate.distribution)
    samples   : [value,...]           — raw equal-weight draws (compositional Dirichlet particles)
    """
    name: str
    particles: list | None = None
    grid: tuple | None = None
    envelope: dict | None = None
    samples: list | None = None

    def mean(self) -> float:
        if self.particles:
            return _weighted_stats(self.particles)[0]
        if self.grid:
            vs, ws = self.grid
            z = sum(ws) or 1.0
            return sum(v * w for v, w in zip(vs, ws)) / z
        if self.samples:
            return sum(self.samples) / len(self.samples)
        if self.envelope:
            return float(self.envelope.get("mean", 0.0))
        return 0.0

    def sample(self, n: int, rng) -> list:
        """Draw n realizations (weighted where the representation carries weights)."""
        if self.particles:
            return [self._wdraw(self.particles, rng) for _ in range(n)]
        if self.grid:
            vs, ws = self.grid
            pairs = list(zip(vs, ws))
            return [self._wdraw(pairs, rng) for _ in range(n)]
        if self.samples:
            return [self.samples[rng.randrange(len(self.samples))] for _ in range(n)]
        if self.envelope:
            e = self.envelope
            out = []
            for _ in range(n):
                v = rng.gauss(float(e.get("mean", 0.0)), float(e.get("sd", 0.0)))
                if e.get("lo") is not None:
                    v = max(float(e["lo"]), v)
                if e.get("hi") is not None:
                    v = min(float(e["hi"]), v)
                out.append(v)
            return out
        return [0.0] * n

    @staticmethod
    def _wdraw(pairs, rng):
        z = sum(w for _, w in pairs) or 1.0
        r, acc = rng.random() * z, 0.0
        for v, w in pairs:
            acc += w
            if r <= acc:
                return v
        return pairs[-1][0]

    @classmethod
    def from_posterior_result(cls, name, pr):
        """Adapt a Phase-3 PosteriorResult's outcome-rate cloud."""
        parts = getattr(pr, "outcome_rate_particles", None)
        if parts:
            return cls(name=name, particles=[(float(r), float(w)) for r, w in parts])
        return cls(name=name, envelope={"mean": getattr(pr, "outcome_rate_mean", 0.5),
                                        "sd": getattr(pr, "outcome_rate_sd", 0.29), "lo": 0.0, "hi": 1.0})

    @classmethod
    def point(cls, name, value):
        return cls(name=name, samples=[float(value)])


@dataclass
class PropagationResult:
    mean: float                          # E[f(X)] — the posterior-correct value
    sd: float
    quantiles: list                      # [p10, p50, p90]
    naive: float                         # f(E[X]) — the WRONG shortcut, kept to measure the gap
    jensen_gap: float                    # E[f(X)] − f(E[X])
    n_particles: int
    samples: list = field(default_factory=list)

    def as_dict(self):
        return {"mean": round(self.mean, 6), "sd": round(self.sd, 6),
                "quantiles": [round(q, 6) for q in self.quantiles], "naive_f_of_mean": round(self.naive, 6),
                "jensen_gap": round(self.jensen_gap, 6), "n_particles": self.n_particles}


def propagate(form, param_posteriors: dict, inputs: dict, *, n: int = 200, rng=None,
              param_map=None) -> PropagationResult:
    """Evaluate `form` once per posterior particle and aggregate → E[f(X)], with the Jensen gap measured.

    param_posteriors : {latent_name: ParamPosterior} — the uncertain quantities.
    param_map        : callable(sampled: dict) -> params dict for form.eval  (default: pass through as params)
    inputs           : the (deterministic) form inputs (x, features, window_days, …).

    The per-particle loop is what makes this correct for curved f. `naive` runs the form at the posterior
    MEAN of every latent (the shortcut) so the two are directly comparable and the gap is a number, not a
    hope."""
    import random
    rng = rng or random.Random(0)
    names = list(param_posteriors)
    draws = {nm: param_posteriors[nm].sample(n, rng) for nm in names}
    pm = param_map or (lambda s: s)
    vals = []
    for i in range(n):
        sampled = {nm: draws[nm][i] for nm in names}
        params = pm(sampled)
        vals.append((form.eval(params, inputs), 1.0 / n))
    mean, sd, qs = _weighted_stats(vals)
    naive = form.eval(pm({nm: param_posteriors[nm].mean() for nm in names}), inputs)
    return PropagationResult(mean=mean, sd=sd, quantiles=qs, naive=naive, jensen_gap=mean - naive,
                             n_particles=n, samples=[v for v, _ in vals])


def delta_method_gap(form, param_posteriors: dict, inputs: dict, *, param_map=None, eps=1e-3) -> dict:
    """Second-order (delta-method) estimate of the Jensen gap ≈ ½ Σ f''(μ)·Var, for the justified-
    approximation path. Returns the estimated gap AND flags when |gap| is large enough that per-particle
    propagation is mandatory (curvature × variance is not negligible)."""
    pm = param_map or (lambda s: s)
    mu = {nm: param_posteriors[nm].mean() for nm in param_posteriors}
    gap = 0.0
    for nm, post in param_posteriors.items():
        # marginal variance
        import random
        s = post.sample(64, random.Random(7))
        m = sum(s) / len(s)
        var = sum((v - m) ** 2 for v in s) / max(1, len(s) - 1)
        base = mu[nm]
        hi = dict(mu); hi[nm] = base + eps
        lo = dict(mu); lo[nm] = base - eps
        f0 = form.eval(pm(mu), inputs)
        fh = form.eval(pm(hi), inputs)
        fl = form.eval(pm(lo), inputs)
        f2 = (fh - 2 * f0 + fl) / (eps * eps)
        gap += 0.5 * f2 * var
    return {"delta_method_gap": round(gap, 6),
            "per_particle_required": abs(gap) > 0.01,
            "note": "if per_particle_required, do NOT use f(E[X]); evaluate per posterior particle (Part 12)"}
