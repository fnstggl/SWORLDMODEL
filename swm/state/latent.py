"""Generic latent variables with hierarchical partial pooling (audit C.2, spec state layer).

`swm/state/state.py:Posterior` is the atom (mean + evidence weight). This module adds the two
things the general world model needs on top of it:

1. `HierarchicalPosterior` — the person <- segment <- population shrinkage chain, generalized off
   `swm/entities/persona.py` so BOTH regimes use one estimator:
     - no individual evidence  -> you get the segment prior, wide uncertainty
     - some evidence           -> you shrink toward the individual
     - strong evidence         -> you trust the individual posterior
   This is the single most important object for reconciling aggregate and individual prediction:
   an aggregate query reads the segment/population level; an individual query reads the person level;
   they are the same math at different pooling depths.

2. `LatentField` — a named bag of latents (belief/attention/stance/incentive) with uncertainty,
   so a state can carry an arbitrary, ablation-filtered set of latent scalars without a bespoke
   dataclass per domain.

Everything is a POSTERIOR, never a bare value, so uncertainty propagates through transitions.
No dependencies beyond the stdlib and `state.Posterior`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.state.state import Posterior


@dataclass
class HierarchicalPosterior:
    """A latent estimated by partial pooling toward a segment mean.

    The posterior mean is a precision-weighted blend of the individual evidence and the segment
    prior. `prior_strength` is how many pseudo-observations the segment prior is worth (the
    shrinkage knob). With n_obs=0 you get exactly the segment mean and the widest interval; each
    observation moves you toward the individual empirical mean and narrows the interval.

    This is deliberately the same Beta/Normal-conjugate logic as persona.py, exposed generically so
    aggregate and individual code share one estimator.
    """
    segment_mean: float
    prior_strength: float = 4.0
    _sum: float = 0.0          # sum of observed values (individual evidence)
    _w: float = 0.0            # total observation weight (individual evidence)
    population_mean: float | None = None   # optional deeper level for two-stage pooling

    def observe(self, value: float, weight: float = 1.0) -> "HierarchicalPosterior":
        self._sum += value * weight
        self._w += weight
        return self

    @property
    def n_obs(self) -> float:
        return self._w

    @property
    def _prior_mean(self) -> float:
        # two-stage: the segment prior can itself be shrunk toward the population mean.
        if self.population_mean is None:
            return self.segment_mean
        # segment carries its own weight = prior_strength; pool it half-way to population.
        return (self.segment_mean * self.prior_strength
                + self.population_mean * self.prior_strength) / (2 * self.prior_strength)

    @property
    def mean(self) -> float:
        num = self._prior_mean * self.prior_strength + self._sum
        den = self.prior_strength + self._w
        return num / den if den > 0 else self._prior_mean

    @property
    def n_effective(self) -> float:
        return self.prior_strength + self._w

    @property
    def shrinkage(self) -> float:
        """Fraction of the estimate coming from the segment prior (1 = cold, 0 = fully individual)."""
        return self.prior_strength / self.n_effective

    @property
    def uncertainty(self) -> float:
        """Std-error-like width; shrinks ~1/sqrt(n_effective)."""
        return 1.0 / math.sqrt(max(1e-9, self.n_effective))

    def interval(self, z: float = 1.28) -> tuple[float, float]:
        m, u = self.mean, self.uncertainty
        return (m - z * u, m + z * u)

    def as_posterior(self) -> Posterior:
        return Posterior(mean=self.mean, n=self.n_effective)


@dataclass
class BetaHierarchical:
    """Partial pooling for a RATE (a probability in [0,1]): Beta-Binomial with a segment prior.

    Same contract as HierarchicalPosterior but correct for bounded rates (reply prob, hit rate,
    conversion): the prior is a Beta centered at the segment rate with mass `prior_strength`.
    """
    segment_rate: float
    prior_strength: float = 4.0
    successes: float = 0.0
    failures: float = 0.0

    def observe(self, success: bool | float, weight: float = 1.0) -> "BetaHierarchical":
        s = float(success)
        self.successes += s * weight
        self.failures += (1.0 - s) * weight
        return self

    @property
    def alpha(self) -> float:
        return self.segment_rate * self.prior_strength + self.successes

    @property
    def beta(self) -> float:
        return (1.0 - self.segment_rate) * self.prior_strength + self.failures

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n_effective(self) -> float:
        return self.alpha + self.beta

    @property
    def n_obs(self) -> float:
        return self.successes + self.failures

    @property
    def shrinkage(self) -> float:
        return self.prior_strength / self.n_effective

    def interval(self, z: float = 1.28) -> tuple[float, float]:
        a, b = self.alpha, self.beta
        m = a / (a + b)
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        s = math.sqrt(max(var, 1e-12))
        return (max(0.0, m - z * s), min(1.0, m + z * s))


@dataclass
class LatentField:
    """A named collection of latent scalars, each a Posterior. The domain-agnostic carrier for
    belief / attention / stance / incentive latents that ablation decides to keep."""
    values: dict[str, Posterior] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        p = self.values.get(key)
        return p.mean if p is not None else default

    def observe(self, key: str, value: float, weight: float = 1.0,
                prior: float = 0.0) -> None:
        p = self.values.get(key)
        if p is None:
            p = Posterior(mean=prior, n=1.0)
            self.values[key] = p
        p.observe(value, weight)

    def uncertainty(self, key: str) -> float:
        p = self.values.get(key)
        return p.uncertainty if p is not None else 1.0
