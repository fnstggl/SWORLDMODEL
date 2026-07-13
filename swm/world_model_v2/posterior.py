"""The particle POSTERIOR — Phase 3 upgrade from "equal samples, count at the end" to a weighted, updatable
P(World | evidence).

Machinery: normalized particle weights · observation-likelihood reweighting (via observation.py models) ·
effective sample size · threshold-triggered systematic resampling with DIVERSITY PRESERVATION (small jitter on
sampled-status latents only) · traceable ancestry · posterior predictive checks. Provenance semantics are
ENFORCED here, not just labeled: `observed` fields are never perturbed by rejuvenation; `sampled`/`inferred`
fields may be; `assumed` fields flag when they dominate sensitivity. Parameter particles and structural-model
particles ride the same mechanism (a latent whose path names a parameter/mechanism choice) — model uncertainty
stays separately reported (uncertainty_meta) from world randomness.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from swm.world_model_v2.observation import observation_model_for
from swm.world_model_v2.state import StateField, WorldState


def _read_path(world: WorldState, path: str):
    if path in world.quantities:
        return world.quantities[path].value
    eid, _, fpath = path.partition(".")
    ent = world.entities.get(eid)
    if ent is None:
        return None
    fname, _, key = fpath.partition("[")
    return ent.value(fname, key=key.rstrip("]") or None)


@dataclass
class Particle:
    world: WorldState
    weight: float
    ancestry: list = field(default_factory=list)    # [(generation, parent_branch_id)]


@dataclass
class ParticlePosterior:
    particles: list = field(default_factory=list)   # [Particle]
    generation: int = 0
    resample_threshold: float = 0.5                 # resample when ESS/N < this
    log: list = field(default_factory=list)         # assimilation history (append-only)

    @classmethod
    def from_worlds(cls, worlds, **kw):
        n = len(worlds) or 1
        return cls(particles=[Particle(world=w, weight=1.0 / n) for w in worlds], **kw)

    # ---------------- weights ----------------
    def normalize(self):
        z = sum(p.weight for p in self.particles)
        if z <= 0:                                   # all mass destroyed → uniform (and log it loudly)
            self.log.append({"event": "weight_collapse", "generation": self.generation})
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)
            return self
        for p in self.particles:
            p.weight /= z
        return self

    def ess(self) -> float:
        """Effective sample size: 1/Σw² — degeneracy detector."""
        s = sum(p.weight ** 2 for p in self.particles)
        return (1.0 / s) if s > 0 else 0.0

    # ---------------- observation assimilation ----------------
    def assimilate(self, observation, *, rng=None) -> dict:
        """Reweight every particle by P(obs | its latent value). Append-only log; observed-status fields in
        each particle are untouched (we reweight worlds, we don't edit them); auto-resample on low ESS."""
        rng = rng or random.Random(self.generation)
        model = observation_model_for(observation.of_path)
        # missing-latent policy (Tier A1): a particle in which the observed path does not exist is
        # penalized relative to particles that CAN explain the observation — using the observation
        # model's own floor rather than a flat 0.5 that could dominate real densities. The penalty is
        # the minimum likelihood across particles that do carry the latent (or a hard floor).
        likes = []
        for p in self.particles:
            latent = _read_path(p.world, observation.of_path)
            likes.append(model.likelihood(observation, latent) if latent is not None else None)
        present = [l for l in likes if l is not None]
        missing_pen = min(present) if present else 1e-6
        for p, like in zip(self.particles, likes):
            p.weight *= max(1e-12, like if like is not None else missing_pen)
        self.normalize()
        self.generation += 1
        rec = {"event": "assimilate", "obs": observation.as_dict(), "generation": self.generation,
               "ess": round(self.ess(), 2), "n": len(self.particles)}
        self.log.append(rec)
        if self.ess() / len(self.particles) < self.resample_threshold:
            self.resample(rng)
        return rec

    # ---------------- resampling (diversity-preserving) ----------------
    def resample(self, rng):
        """Systematic resampling + rejuvenation jitter applied ONLY to sampled/inferred-status latent fields
        (never observed ones — provenance semantics are executable here). Ancestry recorded."""
        n = len(self.particles)
        positions = [(rng.random() + i) / n for i in range(n)]
        cum, idx, chosen = 0.0, 0, []
        weights = [p.weight for p in self.particles]
        cums = []
        acc = 0.0
        for w in weights:
            acc += w
            cums.append(acc)
        for pos in positions:
            while idx < n - 1 and cums[idx] < pos:
                idx += 1
            chosen.append(idx)
        new = []
        for i in chosen:
            src = self.particles[i]
            w = src.world.clone(branch_id=f"{src.world.branch_id}~g{self.generation}")
            self._rejuvenate(w, rng)
            new.append(Particle(world=w, weight=1.0 / n,
                                ancestry=src.ancestry + [(self.generation, src.world.branch_id)]))
        self.particles = new
        self.log.append({"event": "resample", "generation": self.generation,
                         "unique_parents": len(set(chosen))})
        return self

    @staticmethod
    def _rejuvenate(world: WorldState, rng, jitter: float = 0.05):
        """Diversity preservation: perturb ONLY fields whose provenance status is sampled/inferred and whose
        value is numeric. `observed` and `derived` fields are NEVER touched. Jitter is RANGE-RELATIVE
        (Tier A1 fix: a fixed 0.05 assumed every latent lives in [0,1]; fields without a declared range
        jitter relative to their own magnitude and are NOT clamped into a fabricated [0,1] box)."""
        for ent in world.entities.values():
            for fname, sf in list(ent.fields.items()):
                items = sf.items() if isinstance(sf, dict) else [(None, sf)]
                for key, f in items:
                    if not isinstance(f, StateField):
                        continue
                    if f.prov.status not in ("sampled", "inferred"):
                        continue                          # provenance semantics, executable
                    if isinstance(f.value, float):
                        if f.dist and ("lo" in f.dist or "hi" in f.dist):
                            lo = float(f.dist.get("lo", 0.0))
                            hi = float(f.dist.get("hi", 1.0))
                            span = max(1e-9, hi - lo)
                            f.value = min(hi, max(lo, f.value + rng.gauss(0, jitter * span)))
                        else:                             # no declared range: scale-relative, unclamped
                            scale = max(1e-3, abs(f.value))
                            f.value = f.value + rng.gauss(0, jitter * scale)

    # ---------------- posterior readout & checks ----------------
    def expectation(self, path: str) -> float:
        vals = [(p.weight, _read_path(p.world, path)) for p in self.particles]
        num = [(w, v) for w, v in vals if isinstance(v, (int, float, bool))]
        z = sum(w for w, _ in num) or 1.0
        return sum(w * float(v) for w, v in num) / z

    def distribution(self, path: str) -> dict:
        out = {}
        for p in self.particles:
            v = str(_read_path(p.world, path))
            out[v] = out.get(v, 0.0) + p.weight
        return {k: round(v, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}

    def posterior_predictive_check(self, observation) -> dict:
        """Would the posterior have predicted this observation? Low predictive density on many observations
        = the model family is wrong, not just the weights (structural-model uncertainty signal)."""
        model = observation_model_for(observation.of_path)
        like_fn = getattr(model, "likelihood_pure", model.likelihood)   # pure density: no reliability floor
        dens = sum(p.weight * like_fn(observation, _read_path(p.world, observation.of_path))
                   for p in self.particles)
        return {"obs": observation.obs_id, "posterior_predictive_density": round(dens, 6),
                "suspect_model_family": dens < 1e-3}
