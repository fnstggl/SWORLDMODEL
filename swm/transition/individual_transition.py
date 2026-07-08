"""Individual response function: EntityState + Action + Context -> response distribution.

The individual half of the world model: `this entity + this message/action + this context ->
p(response)`, with hierarchical partial pooling so the estimator degrades gracefully with evidence:

  no individual evidence  -> the segment prior, wide uncertainty
  some evidence           -> shrink toward the individual
  strong evidence         -> trust the individual posterior

Mechanics (all real, all dependency-free, none an LLM):
- per-entity responsiveness is a `BetaHierarchical` centered on the segment rate; as-of updates only.
- a calibrated logistic head over [message/context features + segment_logit + person_logit]
  produces the probability (the same shape as the proven email L0-L4 ladder), so the *content* of
  the action adds signal beyond the person's base rate.
- `sources` selects which evidence is on — {"segment","person","message"} — which is exactly the
  evidence-source ablation the eval needs (segment-only vs +person vs +message).

`transition()` advances the entity posterior after an observed outcome, so a repeated actor's next
prediction is conditioned on a changed state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.state.latent import BetaHierarchical
from swm.transition.readout import EnsembleReadout, LogisticReadout

_LOGIT = lambda p: math.log(max(1e-6, p) / max(1e-6, 1 - p))  # noqa: E731


@dataclass
class IndividualTransition:
    """Reusable individual response model with partial pooling + a message head.

    `message_feature_names` fixes the feature order for the readout. `sources` is the evidence-source
    ablation switch. `prior_strength` is the pooling knob (how many pseudo-obs the segment prior is
    worth).
    """
    message_feature_names: list[str] = field(default_factory=list)
    segment_rate: float = 0.3
    prior_strength: float = 4.0
    sources: frozenset = frozenset({"segment", "person", "message"})
    ensemble: bool = True
    head: object | None = None
    entities: dict[str, BetaHierarchical] = field(default_factory=dict)
    _fitted: bool = False

    # ---- per-entity posterior (as-of) ----
    def person(self, entity_id: str) -> BetaHierarchical:
        p = self.entities.get(entity_id)
        if p is None:
            p = BetaHierarchical(segment_rate=self.segment_rate, prior_strength=self.prior_strength)
            self.entities[entity_id] = p
        return p

    def _feature_names(self) -> list[str]:
        names = []
        if "message" in self.sources:
            names += list(self.message_feature_names)
        names += ["segment_logit", "person_logit"]
        return names

    def _vector(self, entity_id: str, message_features: dict[str, float]) -> list[float]:
        p = self.person(entity_id)
        person_logit = _LOGIT(p.mean) if "person" in self.sources else _LOGIT(self.segment_rate)
        row = []
        if "message" in self.sources:
            row += [float(message_features.get(n, 0.0)) for n in self.message_feature_names]
        row += [_LOGIT(self.segment_rate), person_logit]
        return row

    # ---- streaming fit (as-of correct) ----
    def fit_stream(self, samples: list[tuple[str, dict, int]], *,
                   segment_rate: float | None = None) -> "IndividualTransition":
        """samples: time-ordered (entity_id, message_features, outcome). Person posteriors are built
        online from PAST outcomes only, so every training row is as-of correct."""
        if segment_rate is not None:
            self.segment_rate = segment_rate
        self.entities.clear()
        X, y = [], []
        for entity_id, mf, outcome in samples:
            X.append(self._vector(entity_id, mf))     # features BEFORE seeing this outcome
            y.append(int(outcome))
            self.person(entity_id).observe(outcome)    # then transition
        if len(set(y)) < 2:
            self.head = None                           # degenerate; predict falls back to pooled rate
        elif self.ensemble:
            self.head = EnsembleReadout(n_members=15).fit(X, y)
        else:
            self.head = LogisticReadout().fit(X, y)
        self._fitted = True
        return self

    # ---- prediction ----
    def predict(self, entity_id: str, message_features: dict[str, float]) -> dict:
        p = self.person(entity_id)
        if self.head is None:
            mean = p.mean if "person" in self.sources else self.segment_rate
            lo, hi = p.interval()
            return {"p_mean": mean, "p_interval80": [lo, hi], "n_effective": p.n_effective,
                    "shrinkage": p.shrinkage, "source": "pooled_rate"}
        x = self._vector(entity_id, message_features)
        if isinstance(self.head, EnsembleReadout):
            mean, (lo, hi) = self.head.predict(x)
        else:
            mean = self.head.predict_proba(x)
            lo, hi = p.interval()
        return {"p_mean": mean, "p_interval80": [lo, hi], "n_effective": p.n_effective,
                "shrinkage": p.shrinkage, "source": "head"}

    def transition(self, entity_id: str, outcome: float) -> None:
        self.person(entity_id).observe(outcome)
