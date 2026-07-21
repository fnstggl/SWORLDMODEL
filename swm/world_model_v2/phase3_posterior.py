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


# ============================================================ Phase 9: compositional (simplex) posterior
# The SAME particle engine as the outcome-rate posterior, extended to a compositional latent (segment-weight
# vector on the K-simplex). Prior = Dirichlet(alpha); each survey/count observation reweights the simplex
# particles by its (reliability-tempered) multinomial log-likelihood; ESS-triggered systematic resampling with
# a Dirichlet jitter keeps it non-degenerate. Weights are non-negative and sum to one BY CONSTRUCTION (each
# particle is a simplex point). NOT independent scalars. Reuses _ess / _systematic_resample / _gamma / the
# dependence-collapse discipline / the assimilation-ledger format — one posterior engine, new representation.
@dataclass
class CompositionalPosteriorResult:
    segment_ids: list = field(default_factory=list)
    particles: list = field(default_factory=list)          # [(simplex_vector, weight)]
    posterior_mean: list = field(default_factory=list)      # K-vector, sums to 1
    posterior_sd: list = field(default_factory=list)
    prior_mean: list = field(default_factory=list)
    conjugate_alpha: list = field(default_factory=list)     # exact Dirichlet(alpha+counts) for the count-only case
    ess: float = 0.0
    n_effective_observations: int = 0
    assimilation_ledger: list = field(default_factory=list)
    prior_provenance: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def as_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "particles"}
        d["n_particles"] = len(self.particles)
        return d


def _dirichlet(rng, alpha):
    g = [_gamma(rng, max(1e-3, a)) for a in alpha]
    z = sum(g) or 1.0
    return [x / z for x in g]


def _multinomial_loglik(counts, probs):
    ll = 0.0
    for c, p in zip(counts, probs):
        if c > 0:
            ll += c * math.log(max(1e-9, p))
    return ll


def infer_compositional_posterior(segment_ids, prior_alpha, count_observations, *, n_particles: int = 400,
                                  seed: int = 0, resample_threshold: float = 0.5,
                                  use_dependence: bool = True, prior_provenance: dict = None):
    """Posterior over a segment-weight simplex. `count_observations` = list of dicts
    {counts: {seg: n}, reliability, dependence_group, source, method}. Deterministic under seed.

    Dependence collapse: observations sharing a dependence_group are merged (counts summed once, most-reliable
    source) so a survey re-reported N times does not N-count. The reliability tempers the multinomial
    log-likelihood (a low-reliability frame contributes a fractional effective sample)."""
    rng = random.Random(seed * 100003 + 11)
    K = len(segment_ids)
    res = CompositionalPosteriorResult(segment_ids=list(segment_ids))
    idx = {s: i for i, s in enumerate(segment_ids)}
    a0 = [float(prior_alpha[i]) for i in range(K)]
    z0 = sum(a0) or 1.0
    res.prior_mean = [a / z0 for a in a0]
    res.prior_provenance = prior_provenance or {"family": "dirichlet", "alpha": a0,
                                                "class": "generic_weakly_informative"}
    # dependence collapse
    obs = _collapse_count_obs(count_observations) if use_dependence else list(count_observations)
    res.n_effective_observations = len(obs)
    # Dirichlet is conjugate to multinomial counts: the posterior is Dirichlet(alpha0 + Σ reliability-tempered
    # counts) — EXACT, and it avoids the importance-sampling degeneracy that a diffuse Dirichlet proposal
    # suffers under a peaked count likelihood. The per-observation ledger below still records the (importance-
    # sampling) ESS + dependence collapse for the audit trail; the FINAL particles are drawn from the exact
    # conjugate posterior so materialized populations are true posterior samples.
    conj = list(a0)
    diffuse = [_dirichlet(rng, a0) for _ in range(min(64, n_particles))]   # a small proposal set, for the ESS trace
    dlog = [0.0] * len(diffuse)
    for o in obs:
        counts = [float(o.get("counts", {}).get(s, 0.0)) for s in segment_ids]
        rel = max(0.0, min(1.0, float(o.get("reliability", 0.8))))
        eff = [c * rel for c in counts]                     # reliability-tempered effective counts
        ess_before = _ess(_normalized(dlog))
        for i, p in enumerate(diffuse):
            dlog[i] += _multinomial_loglik(eff, p)
        for j in range(K):
            conj[j] += eff[j]
        ess_after = _ess(_normalized(dlog))
        res.assimilation_ledger.append({
            "source": o.get("source", "survey"), "method": o.get("method", "counts"),
            "n_total": sum(counts), "reliability": round(rel, 3),
            "n_collapsed": o.get("n_collapsed", 1),
            "ess_before": round(ess_before, 1), "ess_after": round(ess_after, 1),
            "conjugate_update": {segment_ids[j]: round(eff[j], 1) for j in range(K) if eff[j] > 0}})
    res.conjugate_alpha = conj
    A = sum(conj) or 1.0
    res.posterior_mean = [a / A for a in conj]              # exact Dirichlet posterior mean
    res.posterior_sd = [math.sqrt(max(0.0, a * (A - a) / (A * A * (A + 1)))) for a in conj]
    # materialize N exact posterior samples (Dirichlet(conj)) — non-degenerate, true composition draws
    particles = [_dirichlet(rng, conj) for _ in range(n_particles)]
    res.particles = [(p, 1.0 / n_particles) for p in particles]
    res.ess = float(n_particles)                            # exact posterior samples are equally weighted
    res.diagnostics = {"n_particles": n_particles, "seed": seed, "K": K,
                       "posterior_family": "dirichlet_conjugate",
                       "effective_prior_plus_data_N": round(A, 2),
                       "sum_to_one": round(sum(res.posterior_mean), 6),
                       "moved_from_prior": round(sum(abs(a - b) for a, b in
                                                     zip(res.posterior_mean, res.prior_mean)), 4)}
    return res


def _dirichlet_jitter(rng, p, conc: float = 200.0):
    """Rejuvenation: resample a simplex point near p (Dirichlet centered at p with concentration `conc`)."""
    return _dirichlet(rng, [max(1e-3, conc * x) for x in p])


def _collapse_count_obs(observations):
    from collections import defaultdict
    groups, singletons = defaultdict(list), []
    for o in observations:
        g = o.get("dependence_group", "")
        if g:
            groups[g].append(o)
        else:
            singletons.append(o)
    out = list(singletons)
    for g, members in groups.items():
        # syndicated copies of ONE survey → count it ONCE (the most-reliable member's counts), NOT the sum
        # (summing would N-count the same underlying sample — the over-counting dependence correction prevents).
        best = max(members, key=lambda o: float(o.get("reliability", 0.8)))
        out.append({"counts": dict(best.get("counts") or {}), "reliability": float(best.get("reliability", 0.8)),
                    "dependence_group": g, "source": best.get("source", "survey"),
                    "method": best.get("method", "counts"), "n_collapsed": len(members)})
    return out


# ============================================================ Phase 9: edge-existence posterior (log-odds)
# A real Bayesian update on edge existence: posterior logit = prior logit + Σ log-likelihood-ratio of the
# typed edge observations (dependence-collapsed first). Reuses phase3_observation.edge_loglik + the
# collapse discipline; produces a per-edge assimilation ledger. Existence is Bernoulli (the right
# representation), NOT a scalar strength score.
@dataclass
class EdgePosterior:
    src: str
    dst: str
    layer: str
    prior_p: float
    posterior_p: float
    log_odds_shift: float
    n_observations: int
    observed_status: str                                    # observed | inferred | hypothesized
    ledger: list = field(default_factory=list)

    def as_dict(self):
        return {"src": self.src, "dst": self.dst, "layer": self.layer,
                "prior_p": round(self.prior_p, 4), "posterior_p": round(self.posterior_p, 4),
                "log_odds_shift": round(self.log_odds_shift, 4), "n_observations": self.n_observations,
                "observed_status": self.observed_status, "ledger": self.ledger}


def _logit(p):
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1 - p))


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class ExposureObservation:
    """An edge observation WITH exposure (Part 4, informative absence): out of `n_opportunities` chances for a
    typed interaction to have been recorded, `n_observed` were seen. Non-observations (n_opportunities −
    n_observed) are INFORMATIVE — many opportunities with few records is evidence the edge does not exist. A
    Binomial detection model under exposure. Distinguishes 'no opportunity to observe' (n_opportunities=0 →
    uninformative) from 'opportunity existed but nothing occurred'."""
    evidence_class: str
    n_opportunities: int
    n_observed: int
    reliability: float = 0.85
    dependence_group: str = ""


def infer_edge_posterior_exposure(src, dst, layer, exposures, *, prior_p: float = 0.1) -> "EdgePosterior":
    """Edge-existence posterior under an OBSERVATION EXPOSURE model (Part 4). Each ExposureObservation
    contributes a Binomial log-likelihood ratio: k observed of N opportunities with per-opportunity detect
    (edge) / false (no edge) rates. When N=0 (no opportunity to observe) the term is 0 (uninformative). This is
    the calibrated fix to the present-only overconfidence — absence is informative in proportion to exposure."""
    from swm.world_model_v2.phase3_observation import _edge_rates
    lo = _logit(prior_p)
    ledger, total = [], 0.0
    for ex in exposures:
        N, k = max(0, int(ex.n_opportunities)), max(0, int(ex.n_observed))
        k = min(k, N)
        if N == 0:
            ledger.append({"evidence_class": ex.evidence_class, "n_opportunities": 0, "n_observed": 0,
                           "log_lr": 0.0, "note": "no_opportunity_uninformative"})
            continue
        detect, false = _edge_rates(ex.evidence_class, "strong", ex.reliability)
        ll_exists = k * math.log(detect) + (N - k) * math.log(1 - detect)
        ll_absent = k * math.log(false) + (N - k) * math.log(1 - false)
        lr = ll_exists - ll_absent
        lo += lr
        total += lr
        ledger.append({"evidence_class": ex.evidence_class, "n_opportunities": N, "n_observed": k,
                       "log_lr": round(lr, 4)})
    status = "observed" if any(e.get("n_observed", 0) > 0 for e in ledger) else (
        "inferred" if exposures else "hypothesized")
    return EdgePosterior(src=src, dst=dst, layer=layer, prior_p=prior_p, posterior_p=_sigmoid(lo),
                         log_odds_shift=total, n_observations=len(exposures), observed_status=status,
                         ledger=ledger)


def infer_edge_posterior(src, dst, layer, edge_observations, *, prior_p: float = 0.1,
                         use_dependence: bool = True) -> EdgePosterior:
    """Per-edge existence posterior. `edge_observations` are phase3_observation.EdgeObservation for THIS edge.
    posterior_logit = prior_logit + Σ [logP(obs|exists) − logP(obs|absent)]. Deterministic (no sampling)."""
    from swm.world_model_v2.phase3_observation import collapse_edge_observations, edge_loglik
    obs = collapse_edge_observations(edge_observations) if use_dependence else list(edge_observations)
    lo = _logit(prior_p)
    ledger, total_shift = [], 0.0
    strongest = "hypothesized"
    for o in obs:
        lr = edge_loglik(o, True) - edge_loglik(o, False)
        lo += lr
        total_shift += lr
        cls = o.evidence_class
        if cls in ("direct_communication_record", "org_chart_relationship", "formal_authority_record",
                   "resource_transfer", "conflict_record", "co_membership"):
            strongest = "observed"
        elif strongest != "observed":
            strongest = "inferred"
        ledger.append({"evidence_class": cls, "strength": o.strength,
                       "reliability": round(o.reliability, 3), "n_collapsed": o.n_collapsed,
                       "log_lr": round(lr, 4)})
    return EdgePosterior(src=src, dst=dst, layer=layer, prior_p=prior_p, posterior_p=_sigmoid(lo),
                         log_odds_shift=total_shift, n_observations=len(obs),
                         observed_status=(strongest if obs else "hypothesized"), ledger=ledger)
