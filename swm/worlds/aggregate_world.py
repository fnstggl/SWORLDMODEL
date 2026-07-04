"""AggregateWorld — a fitted aggregate/community world (spec Phase 3).

Owns: a `PopulationState`, an `AggregateTransition` (state-sensitive calibrated head + drift), and a
calibration grade from a temporal backtest. Implements the world-model contract at the population
level:

    WorldState_t + Action -> Outcome + WorldState_{t+1}

- `fit_stream(samples)` builds the population state ONLINE (as-of correct), fits the head on the
  as-of feature vectors, then computes an honest grade on a temporal holdout.
- `predict(action)` conditions on the CURRENT population state (state genuinely enters — unlike the
  old PriorHead rollout).
- `backtest(samples)` is the go/no-go: state-transition vs a content-only ablation, on a temporal
  split, with proper scoring + calibration.

One domain per world; HN is the first backtested instance (see experiments/aggregate_harness.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.eval.metrics import (brier_score, expected_calibration_error, log_loss, uplift_at_k)
from swm.state.population import PopulationState
from swm.state.state import Action
from swm.transition.aggregate_transition import AggregateTransition
from swm.transition.nonstationarity import DriftTracker
from swm.transition.transition_head import OutcomeHead


@dataclass
class AggregateWorld:
    domain: str = "generic"
    thresholds: tuple[int, ...] = (10, 40, 100, 300)
    target_threshold: int = 40
    transition: AggregateTransition = field(default=None)  # type: ignore
    pop: PopulationState = field(default=None)             # type: ignore
    grade: dict = field(default_factory=lambda: {"grade": "ungraded", "ece": None})
    use_incentives: bool = True

    def __post_init__(self):
        if self.transition is None:
            self.transition = AggregateTransition(thresholds=self.thresholds,
                                                  use_incentives=self.use_incentives)
        if self.pop is None:
            self.pop = PopulationState(timestamp=0.0)

    # ---- fit ----
    def fit_stream(self, samples: list[tuple[Action, float]]) -> "AggregateWorld":
        """samples: time-ordered (Action, magnitude). Streams state, fits the head as-of."""
        self.pop = PopulationState(timestamp=samples[0][0].timing.get("ts", 0.0) if samples else 0.0)
        self.transition.drift = DriftTracker()
        X, scores = [], []
        for action, mag in samples:
            X.append(self.transition.feature_vector(self.pop, action))
            scores.append(mag)
            self.transition.transition(self.pop, action, mag)
        self.transition.head = OutcomeHead(thresholds=self.thresholds).fit(X, scores)
        return self

    def predict(self, action: Action) -> dict:
        pred = self.transition.predict(self.pop, action)
        pred["report_type"] = "prediction"
        pred["p_ge_target"] = pred["thresholds"].get(self.target_threshold)
        pred["calibration"] = self.grade
        pred["population_uncertainty"] = self.pop.uncertainty_summary()
        return pred

    # ---- backtest: state model vs content-only, temporal split ----
    def backtest(self, samples: list[tuple[Action, float]], *, split: float = 0.7) -> dict:
        n = len(samples)
        if n < 50:
            return {"error": f"only {n} samples; need >= 50 for a temporal split"}
        cut = int(split * n)
        thr = self.target_threshold

        def run(exclude: tuple[str, ...]) -> dict:
            tr = AggregateTransition(thresholds=self.thresholds, use_incentives=self.use_incentives,
                                     exclude=exclude)
            pop = PopulationState(timestamp=samples[0][0].timing.get("ts", 0.0))
            Xtr, ytr = [], []
            for action, mag in samples[:cut]:
                Xtr.append(tr.feature_vector(pop, action)); ytr.append(mag)
                tr.transition(pop, action, mag)
            tr.head = OutcomeHead(thresholds=self.thresholds).fit(Xtr, ytr)
            Xte, yte = [], []
            for action, mag in samples[cut:]:
                Xte.append(tr.feature_vector(pop, action)); yte.append(1 if mag >= thr else 0)
                tr.transition(pop, action, mag)
            preds = [tr.head.predict(x)["thresholds"].get(thr, 0.0) for x in Xte]
            preds = [min(1 - 1e-6, max(1e-6, p)) for p in preds]
            return {"log_loss": round(log_loss(yte, preds), 4),
                    "brier": round(brier_score(yte, preds), 4),
                    "ece": round(expected_calibration_error(yte, preds), 4),
                    "uplift@20": round(uplift_at_k(yte, preds, 0.2), 4),
                    "n_test": len(yte), "test_base_rate": round(sum(yte) / len(yte), 4)}

        # state features to strip for the content-only ablation
        state_feats = ("agg_base_logit", "agg_subgroup_logit", "agg_salience", "agg_reputation",
                       "agg_competition", "agg_drift")
        full = run(())
        content_only = run(state_feats)
        # base rate
        ytr_base = sum(1 for _, m in samples[:cut] if m >= thr) / cut
        yte = [1 if m >= thr else 0 for _, m in samples[cut:]]
        base = {"log_loss": round(log_loss(yte, [ytr_base] * len(yte)), 4)}

        ece = full["ece"]
        self.grade = {"grade": "A" if ece < 0.05 else "B" if ece < 0.10 else "C" if ece < 0.15
                      else "F", "ece": ece, "n": full["n_test"],
                      "note": "graded on temporal holdout, state-transition model"}
        delta_ll = round(content_only["log_loss"] - full["log_loss"], 4)
        return {
            "domain": self.domain, "target": f"P(score>={thr})",
            "base_rate": base, "content_only": content_only, "state_transition": full,
            "state_helps_logloss": delta_ll,           # >0 => state improves held-out log loss
            "grade": self.grade,
            "verdict": ("state-transition beats content-only" if delta_ll > 0
                        else "state-transition does NOT beat content-only"),
        }
