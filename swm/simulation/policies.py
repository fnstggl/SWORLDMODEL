"""Simulation policies: exposure, front-page transition, social proof, and their FITTING (Phase 4).

These are the transition dynamics of HN score formation — how a submission gains exposure, crosses
the front-page threshold, and cascades via social proof — expressed as a small set of parameters
that are FIT from training data (not hand-tuned to a result). Kept here so the engine stays a clean
event loop and the learnable knobs are auditable in one place.

Also holds fast stdlib samplers (Poisson/binomial) so the engine can draw sampled reaction counts
without numpy.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict


# ------------------------------------------------------------------ samplers (no numpy)
def sample_poisson(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    if lam < 20:                                  # Knuth
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= L:
                return k - 1
    # normal approximation for large lambda
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


# ------------------------------------------------------------------ fittable dynamics parameters
@dataclass
class PolicyParams:
    """The learnable transition dynamics. Defaults are priors; `fit` tunes them to train outcomes."""
    n_steps: int = 4
    fp_window: int = 2                      # front-page can only be reached in the first 2 steps
    new_page_exposure: float = 45.0         # audience size in /new before front page
    frontpage_threshold: float = 8.0        # early points needed to have a shot at front page
    frontpage_noise: float = 2.0            # stochasticity of the threshold (luck)
    frontpage_multiplier: float = 30.0      # exposure jump on reaching front page
    frontpage_decay: float = 0.5            # exposure decay per step after the peak
    social_proof_scale: float = 40.0        # score at which social proof ~saturates
    novelty_decay: float = 0.5              # novelty multiplier per step
    author_rep_gain: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyParams":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


def frontpage_prob(early_score: float, p: PolicyParams) -> float:
    """Stochastic front-page transition: logistic in (early_score - threshold), scaled by noise.
    This is the crux nonlinearity — a heavy-tailed, near-bimodal split between posts that die in
    /new and posts that cross over and cascade."""
    z = (early_score - p.frontpage_threshold) / max(1e-6, p.frontpage_noise)
    return 1.0 / (1.0 + math.exp(-z))


def social_proof(score: float, p: PolicyParams) -> float:
    """Bandwagon signal in [0,1): rises with accumulated score, saturating."""
    return 1.0 - math.exp(-score / max(1e-6, p.social_proof_scale))
