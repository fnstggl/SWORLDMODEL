"""State-sufficiency gate + hybrid mixture-of-experts (Phase 4/7).

The honest lesson from the benchmark: the simulation/world-model wins where state is rich (repeat
authors, strong domains, high context) and the raw LLM wins where it is sparse (cold-start,
semantics-dominant). So don't force one to win everywhere — GATE between them by how much state we
actually have:

    prediction = gate * world_model + (1 - gate) * calibrated_llm_prior

`StateSufficiencyGate` maps evidence (entity-history depth, domain evidence, retrieval quality,
epistemic uncertainty) to a gate in [0,1]. `HybridModel` combines a world-model predictor and a
calibrated-LLM predictor by the gate, and records which branch drove each prediction so the report
can say when the world model was actually used.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StateSufficiencyGate:
    """Gate = evidence / (evidence + softness). Cold entity/domain -> ~0 (trust LLM); deep -> ~1
    (trust world model). `domain_weight` down-weights domain evidence vs. entity evidence."""
    softness: float = 6.0
    domain_weight: float = 0.5
    uncertainty_penalty: float = 1.0

    def evidence(self, *, author_depth: float = 0.0, domain_depth: float = 0.0,
                retrieval_quality: float = 1.0, epistemic_uncertainty: float = 0.0) -> float:
        raw = (author_depth + self.domain_weight * domain_depth) * retrieval_quality
        return max(0.0, raw - self.uncertainty_penalty * epistemic_uncertainty)

    def gate(self, **kw) -> float:
        e = self.evidence(**kw)
        return e / (e + self.softness)


@dataclass
class HybridModel:
    """Mixture of a world-model branch and a calibrated-LLM branch, combined by the gate."""
    gate: StateSufficiencyGate = field(default_factory=StateSufficiencyGate)
    usage: dict = field(default_factory=lambda: {"world_model": 0, "llm": 0, "mixed": 0})

    def predict(self, *, world_p: float, llm_p: float, author_depth: float = 0.0,
                domain_depth: float = 0.0, retrieval_quality: float = 1.0,
                epistemic_uncertainty: float = 0.0) -> dict:
        g = self.gate.gate(author_depth=author_depth, domain_depth=domain_depth,
                           retrieval_quality=retrieval_quality,
                           epistemic_uncertainty=epistemic_uncertainty)
        p = g * world_p + (1 - g) * llm_p
        branch = "world_model" if g > 0.66 else "llm" if g < 0.34 else "mixed"
        self.usage[branch] += 1
        return {"p": p, "gate": round(g, 4), "branch": branch,
                "world_p": world_p, "llm_p": llm_p}

    def usage_report(self) -> dict:
        tot = sum(self.usage.values()) or 1
        return {k: {"n": v, "frac": round(v / tot, 3)} for k, v in self.usage.items()}
