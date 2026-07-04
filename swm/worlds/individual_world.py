"""IndividualWorld — a fitted individual response world (spec Phase 4).

Channel-agnostic wrapper around `IndividualTransition`: given a labeled, time-ordered stream of
(entity, message-features, outcome), it fits the hierarchical response model and grades it on a
temporal holdout. Exposes the evidence-source ablation (segment / +person / +message) directly, so
the individual-response eval can compare regimes on identical data.

Contract: `WorldState_t(entity) + Action -> Outcome + WorldState_{t+1}(entity)`. The per-entity
posterior is the state; `predict` reads it, `observe` advances it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.eval.metrics import (brier_score, expected_calibration_error, log_loss, uplift_at_k)
from swm.transition.individual_transition import IndividualTransition


@dataclass
class IndividualWorld:
    message_feature_names: list[str] = field(default_factory=list)
    segment_rate: float = 0.3
    prior_strength: float = 4.0
    sources: frozenset = frozenset({"segment", "person", "message"})
    model: IndividualTransition = field(default=None)  # type: ignore
    grade: dict = field(default_factory=lambda: {"grade": "ungraded", "ece": None})

    def __post_init__(self):
        if self.model is None:
            self.model = IndividualTransition(
                message_feature_names=self.message_feature_names, segment_rate=self.segment_rate,
                prior_strength=self.prior_strength, sources=self.sources)

    def fit_stream(self, samples: list[tuple[str, dict, int]], *,
                   segment_rate: float | None = None) -> "IndividualWorld":
        self.model.fit_stream(samples, segment_rate=segment_rate)
        return self

    def predict(self, entity_id: str, message_features: dict) -> dict:
        out = self.model.predict(entity_id, message_features)
        out["report_type"] = "prediction"
        out["calibration"] = self.grade
        return out

    def backtest(self, samples: list[tuple[str, dict, int]], *, split: float = 0.7,
                 segment_rate: float | None = None) -> dict:
        """Temporal split; fit on train, evaluate as-of on test (person posteriors carried forward)."""
        n = len(samples)
        if n < 40:
            return {"error": f"only {n} samples; need >= 40"}
        cut = int(split * n)
        train, test = samples[:cut], samples[cut:]
        seg = segment_rate
        if seg is None:
            seg = (sum(o for _, _, o in train) + 1) / (len(train) + 2)
        m = IndividualTransition(message_feature_names=self.message_feature_names,
                                 segment_rate=seg, prior_strength=self.prior_strength,
                                 sources=self.sources)
        m.fit_stream(train, segment_rate=seg)
        preds, y = [], []
        for entity_id, mf, outcome in test:
            preds.append(min(1 - 1e-6, max(1e-6, m.predict(entity_id, mf)["p_mean"])))
            y.append(int(outcome))
            m.transition(entity_id, outcome)      # as-of: learn from test outcome AFTER predicting
        ece = expected_calibration_error(y, preds)
        self.grade = {"grade": "A" if ece < 0.05 else "B" if ece < 0.10 else "C" if ece < 0.15
                      else "F", "ece": round(ece, 4), "n": len(y)}
        seen = {e for e, _, _ in train}
        return {
            "log_loss": round(log_loss(y, preds), 4), "brier": round(brier_score(y, preds), 4),
            "ece": round(ece, 4), "uplift@20": round(uplift_at_k(y, preds, 0.2), 4),
            "n_test": len(y), "test_base_rate": round(sum(y) / len(y), 4),
            "seen_entity_fraction": round(sum(1 for e, _, _ in test if e in seen) / len(test), 3),
            "sources": sorted(self.sources), "grade": self.grade,
        }
