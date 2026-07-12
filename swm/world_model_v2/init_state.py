"""Probabilistic state initialization — Phase 2. Evidence → P(WorldState_t0 | evidence by as-of).

The estimation hierarchy, in strict order of preference:
  1. OBSERVED   — directly from grounded evidence (a poll number, a public statement, a filed budget);
  2. DERIVED    — deterministic rules over observed values;
  3. DATASET    — estimated from structured data (base-rate tables, survey marginals);
  4. LLM        — hypothesis WITH EXPLICIT UNCERTAINTY (candidate values + probabilities, prompt hash kept);
  5. PRIOR      — a broad labeled prior when evidence is absent.
The LLM may propose distributions; it may NOT silently mint precise point values — an LLM-supplied value with
no distribution is coerced to a wide one and flagged. Every latent gets a LatentVariableRecord.

Particles: each trajectory starts from ONE coherent sampled world — correlated latents draw jointly through
declared correlation rules (workload↔attention, trust↔persuadability, …), then coherence validators reject or
repair logically incompatible draws. Model uncertainty (which mechanism/parameters) is recorded separately
from in-world randomness (uncertainty_meta.model vs .aleatory).
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from swm.world_model_v2.state import F, Provenance, StateField, WorldState


@dataclass
class LatentVariableRecord:
    path: str                              # e.g. "recipient.attention"
    candidates: dict                       # {value: prob} or {"mean","sd","lo","hi"}
    evidence: list = field(default_factory=list)
    method: str = "prior"                  # observed|derived|dataset|llm|prior
    prompt_hash: str = ""
    confidence: float = 0.5
    sensitivity: float = 0.5               # expected outcome sensitivity (fidelity planner input)
    calibrated: bool = False


def llm_distribution(llm, question: str, path: str, *, lo=0.0, hi=1.0) -> LatentVariableRecord:
    """Ask the LLM for a DISTRIBUTION over a latent — never a point. Unparseable → broad prior, flagged."""
    from swm.engine.grounding import parse_json
    prompt = (f"Estimate the latent variable '{path}' for: {question}\n"
              f"You may NOT give one precise value. Give 2-4 candidate values in [{lo},{hi}] with "
              f"probabilities summing to 1, from the evidence you were shown.\n"
              f'Return ONLY JSON: {{"candidates": {{"<value>": <prob>, ...}}, "confidence": <0..1>}}')
    ph = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    raw = parse_json(llm(prompt)) if llm is not None else None
    if raw and isinstance(raw.get("candidates"), dict) and len(raw["candidates"]) >= 2:
        cands = {}
        for k, v in raw["candidates"].items():
            try:
                cands[float(k)] = max(0.0, float(v))
            except (TypeError, ValueError):
                continue
        z = sum(cands.values())
        if z > 0 and len(cands) >= 2:
            return LatentVariableRecord(path=path, candidates={k: v / z for k, v in cands.items()},
                                        method="llm", prompt_hash=ph,
                                        confidence=min(1.0, max(0.0, float(raw.get("confidence", 0.5)))))
    mid = (lo + hi) / 2.0
    return LatentVariableRecord(path=path, candidates={"mean": mid, "sd": (hi - lo) / 4.0, "lo": lo, "hi": hi},
                                method="prior", prompt_hash=ph, confidence=0.2)


@dataclass
class CorrelationRule:
    """Joint structure between two latents: after sampling `src`, shift `dst`'s draw by strength×(src-mid).
    Declares the known couplings (high workload ↔ low attention, trust ↔ persuadability, …)."""
    src: str
    dst: str
    strength: float                        # -1..1 (negative = anticorrelated)

    def adjust(self, src_val, dst_val, *, lo=0.0, hi=1.0):
        try:
            mid = (lo + hi) / 2.0
            return min(hi, max(lo, float(dst_val) + self.strength * (float(src_val) - mid)))
        except (TypeError, ValueError):
            return dst_val


@dataclass
class CoherenceRule:
    """Reject/repair incompatible draws: check(sample: {path: value}) -> (ok, repaired_sample_or_None)."""
    name: str
    check: object


@dataclass
class InitialStateModel:
    """The posterior over t0 worlds: a base world + latent records + correlations + coherence rules."""
    base_world: WorldState
    latents: list = field(default_factory=list)          # [LatentVariableRecord]
    correlations: list = field(default_factory=list)     # [CorrelationRule]
    coherence: list = field(default_factory=list)        # [CoherenceRule]

    def _set_path(self, world, path: str, value, rec: LatentVariableRecord):
        eid, _, fpath = path.partition(".")
        ent = world.entities.get(eid)
        if ent is None:
            return
        fname, _, key = fpath.partition("[")
        key = key.rstrip("]") or None
        ent.set(fname, F(value, status="sampled", method=f"init:{rec.method}",
                         confidence=rec.confidence, sources=rec.evidence,
                         updated_at=world.clock.now, calibrated=rec.calibrated), key=key)

    def sample_particle(self, branch_id: str, rng) -> WorldState:
        """One coherent world draw. Latents sample in declaration order; correlations adjust downstream
        draws; coherence rules get up to 3 repair attempts, else the particle re-draws."""
        for _ in range(5):
            world = self.base_world.clone(branch_id=branch_id)
            sample = {}
            for rec in self.latents:
                sf = StateField(dist=rec.candidates)
                v = sf.sample(rng)
                for cr in self.correlations:
                    if cr.dst == rec.path and cr.src in sample:
                        v = cr.adjust(sample[cr.src], v)
                sample[rec.path] = v
            ok = True
            for rule in self.coherence:
                good, repaired = rule.check(sample)
                if not good and repaired is None:
                    ok = False
                    break
                if not good:
                    sample = repaired
            if not ok:
                continue
            for rec in self.latents:
                self._set_path(world, rec.path, sample[rec.path], rec)
            world.uncertainty_meta = {"latents": {r.path: {"method": r.method, "confidence": r.confidence}
                                                  for r in self.latents},
                                      "sampled": {k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in sample.items()},
                                      "model": {"note": "mechanism/parameter uncertainty recorded per-record"},
                                      "aleatory": {"seed_branch": branch_id}}
            return world
        raise RuntimeError("could not sample a coherent world in 5 attempts — coherence rules too strict "
                           "or latents contradictory (this is a modeling error to surface, not paper over)")

    def sample_particles(self, n: int, *, seed: int = 0) -> list:
        rng = random.Random(seed)
        return [self.sample_particle(f"b{i:03d}", rng) for i in range(n)]
