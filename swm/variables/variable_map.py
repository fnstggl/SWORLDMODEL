"""VariableMap — the universal state object every simulation conditions on.

A `VariableMap` is the mapped set of behavioral variables for ONE (entity, action, context) instance:
each variable is a `Variable(value, confidence, provenance, evidence)`. This is the WorldState of the
individual regime — the "all the variables acting on this person" the thesis demands. It is:

- PROVENANCE-TRACKED: every value knows whether it is `data` (observed), `user` (provided), `llm`
  (inferred), `heuristic`, or `prior` — so we never present an inference as a fact.
- UNCERTAINTY-AWARE: every value carries a confidence in [0,1]; the readout weights by it and cold
  variables (prior-only) contribute little.
- MERGEABLE: user-provided context overrides inferred values (higher provenance rank wins).
- BACKTESTABLE: `to_features()` yields a confidence-weighted vector for the calibrated readout, so
  the mapped variables must earn their place on held-out data.

The simulation reads this map; the map does not contain or peek at the outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.variables.schema import NAMES, PROVENANCE_RANK, SPECS, spec


@dataclass
class Variable:
    value: float
    provenance: str = "prior"          # data | user | llm | heuristic | prior
    confidence: float = 0.15           # [0,1]
    evidence: str = ""                 # one-line justification (for audit / explanations)

    @property
    def rank(self) -> int:
        return PROVENANCE_RANK.get(self.provenance, 0)


@dataclass
class VariableMap:
    entity_id: str = ""
    vars: dict[str, Variable] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def set(self, name: str, value: float, *, provenance: str = "llm", confidence: float = 0.5,
            evidence: str = "", override_lower: bool = True) -> None:
        """Set a variable, keeping the higher-provenance value on conflict (user/data > llm > heuristic)."""
        if name not in SPECS:
            return
        s = spec(name)
        lo, hi = (-1.0, 1.0) if s.signed else (0.0, 1.0)
        value = max(lo, min(hi, float(value)))
        existing = self.vars.get(name)
        new = Variable(value, provenance, max(0.0, min(1.0, confidence)), evidence)
        # keep the higher-provenance value; ties resolved by override_lower (last writer wins)
        if existing is None or new.rank > existing.rank or (new.rank == existing.rank and override_lower):
            self.vars[name] = new

    def get(self, name: str, default: float | None = None) -> float:
        v = self.vars.get(name)
        if v is not None:
            return v.value
        return default if default is not None else (spec(name).default if name in SPECS else 0.0)

    def confidence(self, name: str) -> float:
        v = self.vars.get(name)
        return v.confidence if v is not None else (spec(name).prior_confidence if name in SPECS else 0.0)

    def fill_priors(self) -> "VariableMap":
        """Ensure every schema variable is present (population prior where unset), so the state is complete."""
        for name in NAMES:
            if name not in self.vars:
                s = spec(name)
                self.vars[name] = Variable(s.default, "prior", s.prior_confidence, "population prior")
        return self

    def merge_user_context(self, user_vars: dict) -> "VariableMap":
        """User-provided known variables (highest trust). Accepts {name: value} or {name: (value, conf)}."""
        for name, v in (user_vars or {}).items():
            if isinstance(v, (tuple, list)):
                self.set(name, v[0], provenance="user", confidence=v[1] if len(v) > 1 else 0.9,
                         evidence="user-provided")
            else:
                self.set(name, v, provenance="user", confidence=0.9, evidence="user-provided")
        return self

    # ---- for the backtestable readout ----
    def to_features(self, *, confidence_weighted: bool = True) -> list[float]:
        """Ordered feature vector over all schema variables. Each value is shrunk toward its neutral
        point by (1 − confidence) — so a low-confidence inference (or an unset prior variable) barely
        moves the prediction, while a high-confidence data/user value moves it fully. A confidence
        channel per variable is appended so the readout can further modulate trust. Pre-shrinking (vs
        handing raw values to the readout) empirically reduces overfitting from the ~26 mostly-prior
        variables — uninformed variables contribute ~0 rather than noise."""
        self.fill_priors()
        feats = []
        for name in NAMES:
            s = spec(name)
            v = self.vars[name]
            neutral = 0.0 if s.signed else s.default
            feats.append(neutral + (v.value - neutral) * v.confidence if confidence_weighted else v.value)
        for name in NAMES:                       # confidence channel
            feats.append(self.vars[name].confidence)
        return feats

    @staticmethod
    def feature_names() -> list[str]:
        return [f"var:{n}" for n in NAMES] + [f"conf:{n}" for n in NAMES]

    def provenance_report(self) -> dict:
        counts = {}
        for v in self.vars.values():
            counts[v.provenance] = counts.get(v.provenance, 0) + 1
        return {"n_vars": len(self.vars), "by_provenance": counts,
                "mean_confidence": round(sum(v.confidence for v in self.vars.values())
                                         / max(1, len(self.vars)), 3)}

    def explain(self, top: int = 8) -> list[dict]:
        """Human-readable: the most confident non-prior variables + their evidence."""
        rows = [{"variable": n, "value": round(v.value, 2), "provenance": v.provenance,
                 "confidence": round(v.confidence, 2), "evidence": v.evidence}
                for n, v in self.vars.items() if v.provenance != "prior"]
        rows.sort(key=lambda r: r["confidence"], reverse=True)
        return rows[:top]
