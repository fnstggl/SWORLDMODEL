"""Joint posterior over outcome-rate + structural hypotheses — Phase 3 (Parts H, I, J, K).

`infer_posterior` consumes the Phase-2 bundle's tagged claims and produces a NUMERIC, likelihood-updated
posterior:

  * outcome-rate posterior — N rate particles in [0,1] with log-weights; prior from the plan's qualitative
    lean (a broad Beta, labeled generic/weakly-informative), each dependence-collapsed claim reweighting the
    particles through DirectionalRateModel; ESS-triggered systematic resampling keeps it non-degenerate;
  * structural posterior — over the plan's competing hypotheses; prior = compiler structural prior, updated
    by StructuralDetectionModel log-likelihoods summed over dependence-collapsed claims.

All arithmetic is log-space and guarded (no NaN, no all-zero collapse without a loud log). Deterministic under
a fixed seed. The result carries an assimilation ledger (per effective observation: ESS before/after,
resample decision) and the prior→posterior deltas, so a reviewer can trace every number to a claim.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.world_model_v2.fallback import LEAN_BETA
from swm.world_model_v2.phase3_observation import (DirectionalRateModel, StructuralDetectionModel,
                                                   collapse_by_dependence)


@dataclass
class PosteriorResult:
    outcome_rate_particles: list = field(default_factory=list)   # [(rate, weight)] normalized
    outcome_rate_mean: float = 0.5
    outcome_rate_sd: float = 0.29
    outcome_rate_prior_mean: float = 0.5
    structural_prior: dict = field(default_factory=dict)
    structural_posterior: dict = field(default_factory=dict)
    n_effective_observations: int = 0
    n_claims_collapsed: int = 0
    rate_ess: float = 0.0
    structural_ess: float = 0.0
    assimilation_ledger: list = field(default_factory=list)
    prior_provenance: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def as_dict(self):
        d = self.__dict__.copy()
        d["outcome_rate_particles"] = [[round(r, 5), round(w, 6)] for r, w in self.outcome_rate_particles]
        return d


def _beta_sample(rng, a, b):
    ga, gb = _gamma(rng, a), _gamma(rng, b)
    return ga / (ga + gb) if (ga + gb) > 0 else 0.5


def _gamma(rng, k):
    if k < 1:
        return _gamma(rng, k + 1) * (rng.random() ** (1.0 / k))
    d = k - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0, 1)
        v = (1 + c * x) ** 3
        if v <= 0:
            continue
        u = rng.random()
        if u < 1 - 0.0331 * x ** 4 or math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
            return d * v


def _ess(weights) -> float:
    s = sum(w * w for w in weights)
    return (1.0 / s) if s > 0 else 0.0


def _systematic_resample(particles, weights, rng):
    n = len(particles)
    cums, acc = [], 0.0
    for w in weights:
        acc += w
        cums.append(acc)
    start = rng.random() / n
    out, idx = [], 0
    for i in range(n):
        pos = start + i / n
        while idx < n - 1 and cums[idx] < pos:
            idx += 1
        out.append(particles[idx])
    return out


def infer_posterior(plan, bundle, tags, *, n_rate_particles: int = 400, seed: int = 0,
                    resample_threshold: float = 0.5, use_dependence: bool = True,
                    use_structural: bool = True, prior_spec=None) -> PosteriorResult:
    """Assimilate the tagged claims into a joint posterior. `use_dependence`/`use_structural` toggles exist
    for the ablations. `prior_spec` (a phase3_priors.PriorSpec) overrides the default lean-Beta with a
    provenance-carrying, transport-risk-inflated reference-class prior. Deterministic under `seed`."""
    rng = random.Random(seed * 104729 + 17)
    res = PosteriorResult()

    # ---- prior: outcome-rate Beta (reference-class + transport-inflated if a PriorSpec is supplied, else the
    #      fixed qualitative-lean broad Beta — both broad, weakly-informative, and explicitly labeled) ----
    lean = str((plan.provenance or {}).get("outcome_lean", "neutral"))
    if prior_spec is not None:
        a0, b0 = float(prior_spec.alpha), float(prior_spec.beta)
        res.prior_provenance = {"outcome_rate": prior_spec.as_dict(),
                                "structural": {"source": "compiler structural prior", "class": "compiler"}}
    else:
        a0, b0 = LEAN_BETA.get(lean, (1.0, 1.0))
        res.prior_provenance = {"outcome_rate": {"family": "beta", "alpha": a0, "beta": b0,
                                                 "source": f"qualitative lean '{lean}' → fixed broad Beta",
                                                 "class": "generic_weakly_informative",
                                                 "transport_risk": "high (no held-out-validated reference class)"},
                                "structural": {"source": "compiler structural prior", "class": "compiler"}}
    res.outcome_rate_prior_mean = a0 / (a0 + b0)
    rate_particles = [_beta_sample(rng, a0, b0) for _ in range(n_rate_particles)]
    log_w = [0.0] * n_rate_particles

    # ---- dependence correction (Part D) ----
    eff_tags = collapse_by_dependence(tags) if use_dependence else list(tags)
    res.n_claims_collapsed = len(tags)
    res.n_effective_observations = len(eff_tags)

    # ---- structural prior ----
    hyps = [h for h in (plan.structural_hypotheses or []) if isinstance(h, dict)]
    struct_log = {}
    if use_structural and len(hyps) >= 2:
        z = sum(max(1e-6, float(h.get("prior", 1.0) or 1.0)) for h in hyps) or 1.0
        for h in hyps:
            hid = str(h.get("id"))
            res.structural_prior[hid] = round(max(1e-6, float(h.get("prior", 1.0) or 1.0)) / z, 4)
            struct_log[hid] = math.log(res.structural_prior[hid])

    # ---- assimilate each effective (dependence-collapsed) observation ----
    rate_model, struct_model = DirectionalRateModel(), StructuralDetectionModel()
    for t in eff_tags:
        ess_before = _ess(_normalized(log_w))
        # outcome-rate update
        if t.outcome_direction != "neutral":
            for i, r in enumerate(rate_particles):
                log_w[i] += math.log(rate_model.likelihood(t, r))
        # structural update
        for hid in struct_log:
            struct_log[hid] += struct_model.loglik_for_hypothesis(t, hid)
        w = _normalized(log_w)
        ess_after = _ess(w)
        resampled = False
        if ess_after / max(1, n_rate_particles) < resample_threshold:
            rate_particles = _systematic_resample(rate_particles, w, rng)
            # small rejuvenation jitter to fight impoverishment (bounded to [0,1])
            rate_particles = [min(1.0, max(0.0, r + rng.gauss(0, 0.02))) for r in rate_particles]
            log_w = [0.0] * n_rate_particles
            resampled = True
        res.assimilation_ledger.append({
            "obs": t.claim_id, "direction": t.outcome_direction, "strength": t.strength,
            "is_strategic": t.is_strategic, "reliability": round(t.reliability, 3),
            "n_collapsed": getattr(t, "n_collapsed", 1),
            "ess_before": round(ess_before, 1), "ess_after": round(ess_after, 1), "resampled": resampled})

    # ---- finalize outcome-rate posterior ----
    w = _normalized(log_w)
    res.outcome_rate_particles = list(zip(rate_particles, w))
    mean = sum(r * wi for r, wi in res.outcome_rate_particles)
    var = sum(wi * (r - mean) ** 2 for r, wi in res.outcome_rate_particles)
    res.outcome_rate_mean = round(mean, 5)
    res.outcome_rate_sd = round(math.sqrt(max(0.0, var)), 5)
    res.rate_ess = round(_ess(w), 2)

    # ---- finalize structural posterior ----
    if struct_log:
        m = max(struct_log.values())
        exps = {k: math.exp(v - m) for k, v in struct_log.items()}
        zz = sum(exps.values()) or 1.0
        res.structural_posterior = {k: round(v / zz, 4) for k, v in
                                    sorted(exps.items(), key=lambda kv: -kv[1])}
        res.structural_ess = round(1.0 / sum((v) ** 2 for v in
                                             (x / zz for x in exps.values())), 3)
    res.diagnostics = {"n_rate_particles": n_rate_particles, "seed": seed,
                       "rate_ess_frac": round(res.rate_ess / max(1, n_rate_particles), 3),
                       "posterior_shifted": abs(res.outcome_rate_mean - res.outcome_rate_prior_mean) > 0.02,
                       "structural_updated": res.structural_posterior != res.structural_prior}
    if res.rate_ess < 0.1 * n_rate_particles and eff_tags:
        res.warnings.append("low rate ESS — posterior may be degenerate; treat as high-uncertainty")
    return res


def _normalized(log_w):
    m = max(log_w) if log_w else 0.0
    exps = [math.exp(lw - m) for lw in log_w]
    z = sum(exps) or 1.0
    return [e / z for e in exps]
