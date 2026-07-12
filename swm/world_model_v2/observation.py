"""The observation model — the missing fourth core object. Latent truth ≠ what anyone measures.

    true latent state → measurement/generation process → noisy, delayed, incomplete OBSERVATION
    → actor-specific access → simulator evidence → posterior update (posterior.py)

An `ObservationModel` owns the forward direction (generate what a measurement WOULD show, given latent state)
and the inverse (the LIKELIHOOD of an actual observation under a hypothesized latent state — what particle
reweighting consumes). Supports: measurement error, missingness, selection bias, reporting delay, source
reliability, false/misleading observations. Every generated observation records its latent input, noise
model, delay, visibility, provenance and calibration status. Polling, email-open tracking, platform
analytics are INSTANCES of this one contract — registered, not hardcoded.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.world_model_v2.state import Provenance, rfc3339


@dataclass
class Observation:
    obs_id: str
    of_path: str                          # which latent state this measures (entity.field or quantity)
    value: object                         # what the measurement SHOWED (None = missing/nonresponse)
    at: float                             # when the measurement was TAKEN
    reported_at: float                    # when it became visible (delay = reported_at - at)
    noise_model: str = ""
    visibility: str = "public"            # who can see it (public | actor:<id>)
    reliability: float = 0.8              # source reliability (a false source has low reliability)
    misleading: bool = False              # a deliberately false observation (its likelihood uses reliability)
    prov: Provenance = field(default_factory=lambda: Provenance(status="observed", method="measurement"))

    def as_dict(self):
        return {"obs_id": self.obs_id, "of": self.of_path, "value": self.value,
                "at": rfc3339(self.at), "reported_at": rfc3339(self.reported_at),
                "noise_model": self.noise_model, "reliability": self.reliability,
                "misleading": self.misleading}


class ObservationModel:
    """The contract. Subclasses implement generate() (forward) and likelihood() (inverse). Registered
    per-quantity/field; the compiler attaches instances, never bakes measurement into transitions."""
    name = "abstract"
    calibration_status = "prior"

    def generate(self, latent_value, *, at, rng, **kw) -> Observation:
        raise NotImplementedError

    def likelihood(self, observation: Observation, hypothesized_latent) -> float:
        """P(observation.value | latent = hypothesized). The particle-reweighting kernel."""
        raise NotImplementedError


@dataclass
class GaussianMeasurement(ObservationModel):
    """Continuous measurement with error sd, optional selection bias (shift), missingness and delay.
    The polling instance: latent share → published poll with ~3-6pt error, house effect, days of delay."""
    name: str = "gaussian_measurement"
    sd: float = 0.05
    bias: float = 0.0                     # selection/house effect (added to what's shown)
    p_missing: float = 0.0                # nonresponse / never published
    delay_days: float = 0.0
    reliability: float = 0.85
    calibration_status: str = "prior"

    def generate(self, latent_value, *, at, rng, of_path="", obs_id="") -> Observation:
        missing = rng.random() < self.p_missing
        val = None if missing else float(latent_value) + self.bias + rng.gauss(0, self.sd)
        return Observation(obs_id=obs_id or f"obs@{at:.0f}", of_path=of_path, value=val, at=at,
                           reported_at=at + self.delay_days * 86400.0,
                           noise_model=f"gauss(sd={self.sd},bias={self.bias},miss={self.p_missing})",
                           reliability=self.reliability)

    def likelihood_pure(self, observation, hypothesized_latent) -> float:
        """The raw noise-model density (no reliability flattening) — posterior predictive checks use this,
        so an impossible observation is flagged instead of being masked by the unreliability floor."""
        if observation.value is None:
            return max(1e-6, self.p_missing) if self.p_missing else 0.5
        mu = float(hypothesized_latent) + self.bias
        z = (float(observation.value) - mu) / max(1e-6, self.sd)
        return math.exp(-0.5 * z * z) / (self.sd * math.sqrt(2 * math.pi))

    def likelihood(self, observation, hypothesized_latent) -> float:
        like = self.likelihood_pure(observation, hypothesized_latent)
        # unreliable/misleading sources flatten toward uninformative
        return observation.reliability * like + (1 - observation.reliability) * 0.5


@dataclass
class BernoulliDetection(ObservationModel):
    """Binary detection with false-negative/false-positive rates and delay. The email instance:
    a reply HAPPENED (latent) but tracking only shows it with p_detect; silence is ambiguous."""
    name: str = "bernoulli_detection"
    p_detect: float = 0.95                # P(observe positive | latent positive)
    p_false: float = 0.02                 # P(observe positive | latent negative)
    delay_days: float = 0.0
    reliability: float = 0.9
    calibration_status: str = "prior"

    def generate(self, latent_value, *, at, rng, of_path="", obs_id="") -> Observation:
        pos = bool(latent_value)
        shown = (rng.random() < self.p_detect) if pos else (rng.random() < self.p_false)
        return Observation(obs_id=obs_id or f"obs@{at:.0f}", of_path=of_path, value=bool(shown), at=at,
                           reported_at=at + self.delay_days * 86400.0,
                           noise_model=f"bern(detect={self.p_detect},false={self.p_false})",
                           reliability=self.reliability)

    def likelihood(self, observation, hypothesized_latent) -> float:
        if observation.value is None:
            return 0.5
        pos = bool(hypothesized_latent)
        p_obs_true = self.p_detect if pos else self.p_false
        like = p_obs_true if observation.value else (1 - p_obs_true)
        return observation.reliability * like + (1 - observation.reliability) * 0.5


_OBS_MODELS: dict = {}


def register_observation_model(of_path: str, model: ObservationModel):
    """Attach a measurement process to a latent path. The compiler does this; transitions never measure."""
    _OBS_MODELS[of_path] = model
    return model


def observation_model_for(of_path: str) -> ObservationModel:
    m = _OBS_MODELS.get(of_path)
    if m is None:
        raise KeyError(f"no observation model registered for {of_path!r} — a latent without a measurement "
                       f"process cannot generate observations (and cannot be assimilated)")
    return m
