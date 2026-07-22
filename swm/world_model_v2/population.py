"""Population state — Phase 1.3. Segments with weights, WITHIN-segment distributions, and particles.

A segment weight is the portion of the relevant decision population it represents (adjusted for eligibility/
turnout/exposure/voting power where the compiler establishes those). Heterogeneity is a DISTRIBUTION per
field, not one average persona; the compiler infers WHICH dimensions members differ on (causally relevant +
estimable) — political demographics are not assumed for non-political populations. Sample allocation follows
weight × within-segment uncertainty × outcome sensitivity, with floors so small-but-pivotal groups survive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.state import F, Provenance, StateField


@dataclass
class PopulationSegment:
    segment_id: str
    description: str = ""
    weight: StateField = None             # share of the decision population (a StateField: value or dist)
    heterogeneity: dict = field(default_factory=dict)   # field_name -> StateField with a DISTRIBUTION
    sensitivity: float = 0.5              # estimated outcome sensitivity (fidelity planner input; 0..1)

    def uncertainty(self) -> float:
        """Mean spread of the heterogeneous fields — drives sample allocation."""
        spreads = []
        for sf in self.heterogeneity.values():
            d = sf.dist or {}
            if "sd" in d:
                spreads.append(min(1.0, float(d["sd"])))
            elif d:
                probs = list(d.values())
                spreads.append(1.0 - max(probs) / (sum(probs) or 1.0))
        return sum(spreads) / len(spreads) if spreads else 0.3


@dataclass
class PopulationParticle:
    """One sampled concrete member: a coherent draw from a segment's heterogeneity distributions."""
    segment_id: str
    weight: float                         # this particle's share of the population
    traits: dict = field(default_factory=dict)   # field_name -> sampled value
    prov: Provenance = field(default_factory=lambda: Provenance(status="sampled", method="particle"))


@dataclass
class Population:
    population_id: str
    segments: list = field(default_factory=list)      # [PopulationSegment]
    construction_prov: Provenance = field(default_factory=Provenance)   # how weights were established

    def normalized_weights(self) -> dict:
        w = {s.segment_id: (s.weight.value if isinstance(s.weight, StateField) else (s.weight or 0.0))
             for s in self.segments}
        z = sum(v for v in w.values() if isinstance(v, (int, float))) or 1.0
        return {k: (v / z if isinstance(v, (int, float)) else 0.0) for k, v in w.items()}

    def allocate(self, budget: int, *, floor: int = 2, cap: int = None) -> dict:
        """Samples per segment ∝ weight × uncertainty × sensitivity, floored and capped. Deterministic
        (largest remainder)."""
        segs = self.segments
        if not segs:
            return {}
        cap = cap or max(floor, budget)
        w = self.normalized_weights()
        raw = {s.segment_id: max(1e-9, w[s.segment_id]) * (0.5 + s.uncertainty()) * (0.5 + s.sensitivity)
               for s in segs}
        z = sum(raw.values())
        alloc = {sid: floor for sid in raw}
        rem = max(0, budget - floor * len(segs))
        shares = {sid: rem * v / z for sid, v in raw.items()}
        for sid, sh in shares.items():
            alloc[sid] += int(sh)
        left = rem - sum(int(sh) for sh in shares.values())
        for sid, _ in sorted(shares.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:max(0, left)]:
            alloc[sid] += 1
        return {sid: min(cap, n) for sid, n in alloc.items()}

    def sample_particles(self, budget: int, rng, *, floor: int = 2) -> list:
        """Draw coherent particles per the allocation; each carries its share of population weight."""
        alloc = self.allocate(budget, floor=floor)
        w = self.normalized_weights()
        out = []
        for s in self.segments:
            n = alloc.get(s.segment_id, 0)
            for _ in range(n):
                traits = {fname: sf.sample(rng) for fname, sf in s.heterogeneity.items()}
                out.append(PopulationParticle(segment_id=s.segment_id,
                                              weight=w[s.segment_id] / max(1, n), traits=traits))
        return out

    def check_marginals(self, particles: list, observed: dict, *, tol: float = 0.15) -> dict:
        """Validate simulated marginal distributions against observed population data:
        observed = {field: {value: share}}. Returns {field: {simulated, observed, ok}}."""
        report = {}
        for fname, obs in observed.items():
            got = {}
            wz = sum(p.weight for p in particles) or 1.0
            for p in particles:
                v = p.traits.get(fname)
                got[v] = got.get(v, 0.0) + p.weight / wz
            report[fname] = {v: {"simulated": round(got.get(v, 0.0), 3), "observed": round(sh, 3),
                                 "ok": abs(got.get(v, 0.0) - sh) <= tol} for v, sh in obs.items()}
        return report
