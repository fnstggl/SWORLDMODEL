"""Production population inference — Phase 9 (Parts A–E).

A population is NOT eight arbitrary personas with uniform weights and independent traits. It is a typed target
universe whose segment composition is a COMPOSITIONAL posterior (a simplex, inferred via the Phase-3 engine),
whose traits are CORRELATED with segment (segment-conditional rate posteriors, not a single marginal), and
whose aggregate behavior is a POSTSTRATIFIED integral over that posterior — so different posterior particles
produce different aggregate outcomes.

Everything numeric comes from a prior × likelihood update (Phase 3): the LLM may propose the segmentation
variables, never the weights, covariances, or behavior rates. Every population latent is a Phase-3
`LatentVariableSpec` (evidence-linked + consumer-declared) — a population that never alters an aggregate
outcome is ornamental and is rejected.

Reuses `phase3_posterior.infer_compositional_posterior` for the weight simplex and a conjugate Beta-binomial
for segment-conditional rates. No second posterior engine.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.world_model_v2.phase3_latent_spec import LatentVariableSpec
from swm.world_model_v2.phase3_posterior import infer_compositional_posterior


@dataclass
class PopulationSpec:
    """The typed population (Part A). `segments` are ids; the WEIGHT posterior is compositional and inferred —
    it is not stored here as scalars. Provenance/scope make the target universe explicit."""
    population_id: str
    definition: str = ""
    segments: list = field(default_factory=list)              # segment ids
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    geographic_scope: str = ""
    temporal_scope: str = ""
    target_universe: str = ""
    source_frame: str = ""
    frame_coverage_risk: str = "moderate"                     # none|low|moderate|high|severe
    nonresponse_risk: str = "moderate"
    transport_risk: str = "moderate"
    representation: str = "compositional_simplex"
    sensitivity: float = 0.7
    consumed_by: list = field(default_factory=list)

    def as_dict(self):
        return self.__dict__.copy()


@dataclass
class SegmentRatePosterior:
    """Conjugate Beta-binomial posterior for a behavior/opinion rate WITHIN one segment (segment-conditional →
    captures the segment↔trait correlation a single marginal rate throws away)."""
    segment_id: str
    alpha: float
    beta: float
    n: int = 0

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def sd(self):
        a, b = self.alpha, self.beta
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def sample(self, rng):
        from swm.world_model_v2.phase3_posterior import _beta_sample
        return _beta_sample(rng, self.alpha, self.beta)

    def as_dict(self):
        return {"segment_id": self.segment_id, "alpha": round(self.alpha, 3), "beta": round(self.beta, 3),
                "mean": round(self.mean, 4), "sd": round(self.sd, 4), "n": self.n}


def infer_segment_rates(segment_counts, *, prior_a: float = 1.0, prior_b: float = 1.0) -> dict:
    """Per-segment conjugate rate posteriors from {segment: (successes, total)}. Segment-conditional, so the
    correlation between segment membership and the trait is preserved (Part D)."""
    out = {}
    for s, (succ, tot) in segment_counts.items():
        succ, tot = float(succ), float(tot)
        out[s] = SegmentRatePosterior(segment_id=s, alpha=prior_a + succ, beta=prior_b + max(0.0, tot - succ),
                                      n=int(tot))
    return out


def infer_population(pop: PopulationSpec, *, prior_alpha, survey_observations, seed: int = 0,
                     n_particles: int = 400):
    """Infer the compositional segment-weight posterior for a population (Part B/C). Returns the Phase-3
    compositional posterior. `survey_observations` = count observations (see infer_compositional_posterior)."""
    prov = {"family": "dirichlet", "alpha": list(prior_alpha), "population": pop.population_id,
            "target_universe": pop.target_universe, "source_frame": pop.source_frame,
            "frame_coverage_risk": pop.frame_coverage_risk, "class": "reference_or_survey_prior",
            "llm_role": "proposed segmentation variables ONLY; weights inferred by likelihood"}
    return infer_compositional_posterior(pop.segments, prior_alpha, survey_observations, seed=seed,
                                         n_particles=n_particles, prior_provenance=prov)


@dataclass
class PopulationParticle:
    """One posterior draw of a whole population: a segment-weight simplex + per-segment trait rates. Different
    particles = different worlds (Part U). weights sum to one by construction."""
    weights: dict                                            # segment -> weight (sums to 1)
    segment_rates: dict = field(default_factory=dict)        # trait -> {segment: rate}


def materialize_population_particles(comp_posterior, trait_rate_posteriors: dict, *, n: int = 40,
                                     seed: int = 0) -> list:
    """Draw `n` posterior-weighted population particles: each draws a weight simplex from the compositional
    posterior and a rate per (trait, segment) from its Beta-binomial posterior. These materialize into
    WorldState and drive poststratified aggregates."""
    rng = random.Random(seed * 7919 + 3)
    segs = comp_posterior.segment_ids
    parts, wts = comp_posterior.particles, [w for _, w in comp_posterior.particles]
    total = sum(wts) or 1.0
    out = []
    for _ in range(n):
        r, acc, chosen = rng.random() * total, 0.0, parts[-1][0]
        for vec, w in parts:
            acc += w
            if r <= acc:
                chosen = vec
                break
        weights = {s: chosen[i] for i, s in enumerate(segs)}
        srates = {}
        for trait, per_seg in trait_rate_posteriors.items():
            srates[trait] = {s: (per_seg[s].sample(rng) if s in per_seg else 0.5) for s in segs}
        out.append(PopulationParticle(weights=weights, segment_rates=srates))
    return out


def poststratified_estimate(particles: list, trait: str) -> dict:
    """Aggregate a trait rate by poststratification: per particle, aggregate = Σ_s weight_s · rate_s; report
    the posterior-predictive mean + sd across particles (Part C causal-consumption). This is the number that a
    population mechanism produces — different posterior compositions yield different aggregates."""
    vals = []
    for p in particles:
        rates = p.segment_rates.get(trait, {})
        agg = sum(p.weights.get(s, 0.0) * rates.get(s, 0.5) for s in p.weights)
        vals.append(agg)
    if not vals:
        return {"mean": None, "sd": None, "n": 0}
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return {"mean": round(m, 5), "sd": round(sd, 5), "n": len(vals),
            "lo": round(min(vals), 5), "hi": round(max(vals), 5)}


def independent_trait_estimate(trait_rate_posteriors: dict, trait: str, segment_marginal: dict) -> float:
    """The INDEPENDENT-traits baseline (Part D ablation): ignore segment↔trait correlation and use a single
    pooled marginal rate weighted by the (fixed) marginal composition — the thing poststratification corrects."""
    per_seg = trait_rate_posteriors.get(trait, {})
    if not per_seg:
        return None
    pooled = sum(rp.mean * rp.n for rp in per_seg.values()) / max(1, sum(rp.n for rp in per_seg.values()))
    return round(pooled, 5)


def population_latent_specs(pop: PopulationSpec, *, evidence_ids=None) -> list:
    """Phase-3 latent specs for the population latents (Part A + anti-ornamental). Both are evidence-linked and
    consumer-declared; unconsumed specs are dropped by `.measurable()`."""
    ev = list(evidence_ids or [])
    specs = [
        LatentVariableSpec(
            variable_id=f"{pop.population_id}:segment_weights",
            definition=f"compositional segment-weight vector for {pop.population_id}",
            measurable_interpretation="the share of the target universe in each segment",
            support_type="correlated_multivariate", categories=list(pop.segments),
            evidence_claim_ids=ev, observation_models=["compositional_multinomial"],
            prior_source="survey_or_reference_dirichlet", posterior_representation="dirichlet",
            inference_method="compositional_particle_assimilation", sensitivity=pop.sensitivity,
            consumed_by=["poststratified_estimate", "materialize_population_particles"],
            transport_risk=pop.transport_risk),
        LatentVariableSpec(
            variable_id=f"{pop.population_id}:segment_conditional_rate",
            definition="behavior/opinion rate conditional on segment (correlated with segment)",
            measurable_interpretation="within-segment fraction exhibiting the behavior",
            support_type="bounded_continuous", scope=pop.population_id,
            evidence_claim_ids=ev, observation_models=["beta_binomial_counts"],
            prior_source="beta_binomial", posterior_representation="beta",
            inference_method="conjugate", sensitivity=pop.sensitivity,
            consumed_by=["poststratified_estimate"], transport_risk=pop.transport_risk),
    ]
    return [s for s in specs if s.measurable()]
