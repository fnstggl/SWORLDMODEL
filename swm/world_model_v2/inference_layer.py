"""Evidence → posterior world-state inference — Phase 3 (Tier C).

The gap audit's RC2: nothing anywhere mapped evidence to a posterior over hidden state — every latent was
a hand prior, and `WorldBranch.weight` stayed 1.0 forever. This module supplies the missing layer:

  1. HIERARCHICAL SHRINKAGE ESTIMATORS — beta-binomial / normal partial pooling
     (person ← segment ← population), returning posteriors WITH uncertainty, never point rates;
  2. EVIDENCE-CONDITIONED LATENTS — LatentVariableRecords built from observation counts/values with
     evidence dependencies, provenance, and confidence from posterior concentration;
  3. STRUCTURAL HYPOTHESES — competing world structures (mechanism variants, parameter packs, rule
     regimes) carried as per-particle assignments with prior weights; model uncertainty is reported
     separately from within-world randomness and updates by assimilation;
  4. FILTERED ROLLOUT — the loop the audit found missing: roll → hit an observation time → reweight
     branches through the observation model → (resample) → continue; weights genuinely change.

Everything is pure Python, deterministic under seeds, and validated by recovery tests
(tests/test_inference_layer.py): posteriors must beat priors on synthetic ground truth, and hypothesis
weights must concentrate on the generating mechanism.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.world_model_v2.init_state import LatentVariableRecord
from swm.world_model_v2.observation import observation_model_for
from swm.world_model_v2.posterior import ParticlePosterior, _read_path
from swm.world_model_v2.state import rfc3339


# ------------------------------------------------------------------ 1. hierarchical shrinkage
@dataclass
class RatePosterior:
    """Beta-binomial posterior over a rate: alpha/beta carry the pooled prior + local evidence."""
    alpha: float
    beta: float
    n_local: int
    prior_mean: float
    method: str

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sd(self) -> float:
        a, b = self.alpha, self.beta
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def credible90(self):
        """Central ~90% interval by normal approx, clamped (adequate at the sample sizes we shrink at)."""
        m, s = self.mean(), self.sd()
        return max(0.0, m - 1.645 * s), min(1.0, m + 1.645 * s)


def shrunk_rate(k: int, n: int, *, prior_mean: float, prior_strength: float,
                method: str = "beta_binomial_shrinkage") -> RatePosterior:
    """Posterior for a local rate under a pooled prior worth `prior_strength` pseudo-observations."""
    pm = min(1 - 1e-6, max(1e-6, prior_mean))
    return RatePosterior(alpha=pm * prior_strength + k, beta=(1 - pm) * prior_strength + (n - k),
                         n_local=n, prior_mean=pm, method=method)


def hierarchical_rates(groups: dict, *, population_prior: float | None = None) -> dict:
    """Empirical-Bayes partial pooling: groups = {gid: (k, n)}. The pooled prior strength comes from the
    method-of-moments estimate of between-group variance — heavy shrinkage when groups look alike, light
    when they genuinely differ. Returns {gid: RatePosterior} plus '_pool' metadata."""
    ks = {g: k for g, (k, n) in groups.items() if n > 0}
    ns = {g: n for g, (k, n) in groups.items() if n > 0}
    if not ns:
        return {"_pool": {"prior_mean": population_prior or 0.5, "prior_strength": 2.0,
                          "note": "no data — population prior only"}}
    total_k, total_n = sum(ks.values()), sum(ns.values())
    pooled = population_prior if population_prior is not None else total_k / total_n
    # method-of-moments between-group variance of observed rates (weighted)
    rates = {g: ks[g] / ns[g] for g in ns}
    mean_r = sum(rates[g] * ns[g] for g in ns) / total_n
    var_between = sum(ns[g] * (rates[g] - mean_r) ** 2 for g in ns) / max(1, total_n)
    var_within = max(1e-9, mean_r * (1 - mean_r))
    # shrinkage prior strength: within/between ratio (bounded) — classic EB precision estimate
    strength = min(200.0, max(2.0, var_within / max(1e-6, var_between - var_within / (total_n / len(ns)))
                              if var_between > var_within / (total_n / len(ns)) else 200.0))
    out = {g: shrunk_rate(ks.get(g, 0), ns.get(g, 0), prior_mean=pooled, prior_strength=strength)
           for g in groups}
    out["_pool"] = {"prior_mean": round(pooled, 5), "prior_strength": round(strength, 2),
                    "var_between": round(var_between, 6), "n_groups": len(ns),
                    "method": "empirical-Bayes method-of-moments partial pooling"}
    return out


@dataclass
class NormalPosterior:
    mu: float
    sd: float
    n_local: int
    method: str = "normal_shrinkage"


def shrunk_mean(values: list, *, prior_mu: float, prior_sd: float, obs_sd: float) -> NormalPosterior:
    """Conjugate normal posterior for a latent mean under a population prior."""
    n = len(values)
    if n == 0:
        return NormalPosterior(mu=prior_mu, sd=prior_sd, n_local=0, method="prior_only")
    xbar = sum(values) / n
    prec = 1.0 / (prior_sd ** 2) + n / (obs_sd ** 2)
    mu = (prior_mu / (prior_sd ** 2) + n * xbar / (obs_sd ** 2)) / prec
    return NormalPosterior(mu=mu, sd=math.sqrt(1.0 / prec), n_local=n)


# ------------------------------------------------------------------ 2. evidence-conditioned latents
def latent_from_rate_evidence(path: str, k: int, n: int, *, pool: dict,
                              evidence_ids: list, lo=0.0, hi=1.0) -> LatentVariableRecord:
    """Build an evidence-conditioned latent record from count evidence under a fitted pool. Confidence
    scales with posterior concentration; the record keeps its evidence dependencies for the trace."""
    rp = shrunk_rate(k, n, prior_mean=pool.get("prior_mean", 0.5),
                     prior_strength=pool.get("prior_strength", 2.0))
    m, s = rp.mean(), rp.sd()
    return LatentVariableRecord(
        path=path, method="dataset",
        candidates={"mean": min(hi, max(lo, m * (hi - lo) + lo)), "sd": max(1e-4, s * (hi - lo)),
                    "lo": lo, "hi": hi},
        evidence=list(evidence_ids)[:8],
        confidence=min(0.95, max(0.05, 1.0 - 2.0 * s)),
        calibrated=False)


def latent_from_llm_with_floor(llm, question: str, path: str, *, lo=0.0, hi=1.0,
                               min_sd_frac: float = 0.15) -> LatentVariableRecord:
    """LLM-proposed latent distribution with an enforced uncertainty FLOOR: whatever the LLM claims,
    the sd never drops below min_sd_frac×range without corroborating data (no unsupported precision)."""
    from swm.world_model_v2.init_state import llm_distribution
    rec = llm_distribution(llm, question, path, lo=lo, hi=hi)
    if "sd" in rec.candidates:
        rec.candidates["sd"] = max(rec.candidates["sd"], min_sd_frac * (hi - lo))
    rec.confidence = min(rec.confidence, 0.6)               # an LLM read is never high-confidence alone
    return rec


# ------------------------------------------------------------------ 3. structural hypotheses
@dataclass
class StructuralHypothesis:
    """One competing world structure: operator overrides and/or parameter overrides, with a prior weight.
    Example: H_A 'manager can approve directly' vs H_B 'finance approval required' differ in their
    institutions/rules; H_linear vs H_hill differ in the diffusion hazard pack."""
    hypothesis_id: str
    prior: float
    describe: str = ""
    operators: list = None                    # None = plan default; else instantiated operator list
    world_patch: object = None                # callable(world) -> None applied to each particle's world
    param_overrides: dict = field(default_factory=dict)


@dataclass
class HypothesisSet:
    hypotheses: list

    def normalized(self):
        z = sum(h.prior for h in self.hypotheses) or 1.0
        for h in self.hypotheses:
            h.prior /= z
        return self

    def assign(self, n_particles: int, rng: random.Random) -> list:
        """Stratified assignment of hypotheses to particles proportional to priors (every hypothesis with
        prior ≥ 1/n gets at least one particle — structural uncertainty must not silently vanish)."""
        self.normalized()
        counts = {h.hypothesis_id: max(1, round(h.prior * n_particles)) if h.prior >= 1.0 / n_particles
                  else 0 for h in self.hypotheses}
        # trim/pad to exactly n_particles
        order = sorted(self.hypotheses, key=lambda h: -h.prior)
        total = sum(counts.values())
        i = 0
        while total > n_particles:
            hid = order[i % len(order)].hypothesis_id
            if counts[hid] > 1:
                counts[hid] -= 1
                total -= 1
            i += 1
        while total < n_particles:
            counts[order[i % len(order)].hypothesis_id] += 1
            total += 1
            i += 1
        out = []
        for h in self.hypotheses:
            out += [h] * counts[h.hypothesis_id]
        rng.shuffle(out)
        return out[:n_particles]


# ------------------------------------------------------------------ 4. filtered rollout (assimilation)
def TimedObservation(obs_id: str, of_path: str, value, reported_at: float, *,
                     at: float | None = None, reliability: float = 0.9):
    """A real-world measurement to assimilate mid-rollout — a thin constructor over observation.Observation
    (of_path names the measured latent/quantity; reported_at is when it becomes visible, as-of enforced)."""
    from swm.world_model_v2.observation import Observation
    return Observation(obs_id=obs_id, of_path=of_path, value=value,
                       at=at if at is not None else reported_at, reported_at=reported_at,
                       reliability=reliability)


def run_filtered(initial, queue_builder, operators, contract, observations, *,
                 n_particles=30, seed=0, hypotheses: HypothesisSet | None = None,
                 resample_threshold=0.5) -> dict:
    """The missing filter loop (audit cap 24): sample particles (optionally with structural-hypothesis
    assignment), roll ALL branches observation-window by observation-window, reweight through the
    observation models at each reported_at, resample on ESS collapse, continue to the horizon, then
    project the contract over the WEIGHTED branches. Model (hypothesis) posterior reported separately.
    """
    from swm.world_model_v2.rollout import RolloutEngine
    from swm.world_model_v2.state import WorldBranch
    contract.validate()
    rng = random.Random(seed)
    worlds = initial.sample_particles(n_particles, seed=seed)
    assignment = None
    if hypotheses is not None:
        assignment = hypotheses.assign(n_particles, rng)
        for w, h in zip(worlds, assignment):
            w.uncertainty_meta.setdefault("model", {})["hypothesis"] = h.hypothesis_id
            if h.world_patch is not None:
                h.world_patch(w)
    branches = [WorldBranch(branch_id=w.branch_id, world=w) for w in worlds]
    queues = [queue_builder(w) for w in worlds]
    obs = sorted(observations, key=lambda o: o.reported_at)
    post = ParticlePosterior.from_worlds([b.world for b in branches],
                                         resample_threshold=resample_threshold)
    # operator sets follow the WORLD's recorded hypothesis (survives resampling — the hypothesis id
    # rides in uncertainty_meta, which clones with the world), never the branch index.
    hyp_by_id = {h.hypothesis_id: h for h in (hypotheses.hypotheses if hypotheses else [])}

    def ops_for(branch):
        hid = branch.world.uncertainty_meta.get("model", {}).get("hypothesis")
        h = hyp_by_id.get(hid)
        if h is not None and h.operators is not None:
            return h.operators
        return operators

    horizon = contract.horizon_ts
    checkpoints = [o.reported_at for o in obs if o.reported_at < horizon] + [horizon]
    oi = 0
    for cp in checkpoints:
        for i, b in enumerate(branches):
            if b.world.clock.now < cp:
                engine = RolloutEngine(operators=ops_for(b))
                q = queues[i]
                saved_horizon = q.horizon_ts
                q.horizon_ts = min(saved_horizon, cp)
                nb = engine.run_branch(b.world, q, seed=seed * 7919 + i)
                b.log.extend(nb.log)
                q.horizon_ts = saved_horizon
                if b.world.clock.now < cp:
                    b.world.clock.advance_to(cp)
        while oi < len(obs) and obs[oi].reported_at <= cp:
            o = obs[oi]
            # posterior particles wrap the SAME world objects as branches; sync both directions
            for p, b in zip(post.particles, branches):
                p.world = b.world
                p.weight = b.weight if b.weight > 0 else p.weight
            post.assimilate(o, rng=rng)
            resampled = bool(post.log) and post.log[-1].get("event") == "resample"
            for i, (p, b) in enumerate(zip(post.particles, branches)):
                b.world = p.world
                b.weight = p.weight
                if resampled:
                    # a resampled branch carries a cloned world; rebuild its queue from that world —
                    # already-past scheduled events are skipped by the engine's stale check, hazards
                    # re-arm from the world's current clock (documented approximation: in-flight queue
                    # state that isn't derivable from the world does not survive resampling)
                    queues[i] = queue_builder(b.world)
            oi += 1
    for b in branches:
        b.terminal = True
    result = contract.project(branches)
    result["n_deltas"] = sum(len(b.log) for b in branches)
    result["readout"] = "terminal_states"
    result["assimilation"] = {"n_observations": len(obs), "log": post.log[-6:],
                              "ess": round(post.ess(), 2)}
    if assignment is not None:
        hyp_w = {}
        for b, h in zip(branches, assignment):
            hid = b.world.uncertainty_meta.get("model", {}).get("hypothesis", h.hypothesis_id)
            hyp_w[hid] = hyp_w.get(hid, 0.0) + b.weight
        z = sum(hyp_w.values()) or 1.0
        result["structural_posterior"] = {k: round(v / z, 4) for k, v in
                                          sorted(hyp_w.items(), key=lambda kv: -kv[1])}
        result["structural_prior"] = {h.hypothesis_id: round(h.prior, 4)
                                      for h in hypotheses.hypotheses}
    return result, branches
