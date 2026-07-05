"""Aggregate state transition: PopulationState_t + Action -> Outcome + PopulationState_{t+1}.

This is the aggregate half of the world model, made reusable (the logic previously lived inline in
experiments/state_transition_harness.py). Two guarantees the audit demanded:

1. STATE ACTUALLY ENTERS THE PREDICTION. The feature vector fed to the calibrated head includes
   state-derived features — the pooled base rate, the most specific subgroup rate, topic salience,
   domain reputation, attention/competition, incentives, and the drift indicator — all read from the
   *current* PopulationState. Change the state and the prediction changes. (Contrast the old
   /v1/rollout, which used a state-ignoring PriorHead.)

2. THE STATE TRANSITIONS. After each outcome, `transition()` advances base rate, subgroup rates,
   salience, reputation, attention/competition, and the drift tracker — so the next prediction is
   conditioned on a changed world. That recurrence is the world model.

The head is a calibrated `OutcomeHead` (one monotone logistic per score-band threshold) — the same
honest, dependency-free readout used everywhere else; nothing here is an LLM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.state.incentives import IncentiveState, incentives_from_title
from swm.state.population import PopulationState
from swm.state.state import Action
from swm.transition.nonstationarity import DriftTracker
from swm.transition.transition_head import OutcomeHead, band_of

_LOGIT = lambda p: math.log(max(1e-6, p) / max(1e-6, 1 - p))  # noqa: E731


def _content_features(action: Action) -> dict[str, float]:
    cf = action.content_features
    return {
        "title_len": float(cf.get("title_len", 0.5)),
        "is_show": float(cf.get("is_show", 0.0)),
        "is_ask": float(cf.get("is_ask", 0.0)),
        "is_text": float(cf.get("is_text", 0.0)),
        "hour_sin": math.sin(2 * math.pi * action.timing.get("hour", 12) / 24),
        "hour_cos": math.cos(2 * math.pi * action.timing.get("hour", 12) / 24),
        "is_weekend": 1.0 if action.timing.get("weekday", 0) >= 5 else 0.0,
    }


# state-derived + incentive + content feature names, fixed order
_STATE_FEATURES = ["agg_base_logit", "agg_subgroup_logit", "agg_salience", "agg_reputation",
                   "agg_competition", "agg_drift"]
_INCENTIVE_FEATURES = ["inc_stakes", "inc_controversy", "inc_novelty", "inc_reward_gradient",
                       "inc_effort_cost"]
_CONTENT_FEATURES = ["title_len", "is_show", "is_ask", "is_text", "hour_sin", "hour_cos",
                     "is_weekend"]


@dataclass
class AggregateTransition:
    """Reusable aggregate transition over a PopulationState.

    `thresholds` are the score-band edges for the outcome head (HN uses [10,40,100,300]); for a
    plain binary aggregate outcome pass a single threshold. `subgroup_of` maps an Action to the
    subgroup keys whose conditional rate should feed the prediction (most-specific first).
    """
    thresholds: tuple[int, ...] = (10, 40, 100, 300)
    head: OutcomeHead = field(default_factory=lambda: OutcomeHead())
    drift: DriftTracker = field(default_factory=DriftTracker)
    use_incentives: bool = True
    exclude: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.head.thresholds:
            self.head.thresholds = self.thresholds

    @property
    def feature_names(self) -> list[str]:
        names = list(_STATE_FEATURES)
        if self.use_incentives:
            names += _INCENTIVE_FEATURES
        names += _CONTENT_FEATURES
        return [n for n in names if n not in self.exclude]

    def _subgroup_keys(self, action: Action) -> tuple[str, ...]:
        cf = action.content_features
        topic = cf.get("topic", "other")
        kind = ("show" if cf.get("is_show") else "ask" if cf.get("is_ask")
                else "text" if cf.get("is_text") else "link")
        dom = action.meta.get("domain", "")
        # most-specific first
        return (f"topic={topic}|kind={kind}", f"topic={topic}", f"kind={kind}",
                f"domain={dom}" if dom else f"kind={kind}")

    def incentives(self, action: Action) -> IncentiveState:
        title = action.meta.get("title", "")
        return incentives_from_title(
            title, domain=action.meta.get("domain", ""),
            is_text=bool(action.content_features.get("is_text")),
            title_len_chars=len(title))

    def feature_vector(self, pop: PopulationState, action: Action) -> list[float]:
        base = pop.base_rate.mean
        # most-specific subgroup that has evidence
        sub = base
        for k in self._subgroup_keys(action):
            sg = pop.subgroups.get(k)
            if sg is not None and sg.n >= 1:
                sub = sg.rate.mean
                break
        topic = action.content_features.get("topic", "other")
        dom = action.meta.get("domain", "")
        feats = {
            "agg_base_logit": _LOGIT(base),
            "agg_subgroup_logit": _LOGIT(sub),
            "agg_salience": pop.salience(topic) if isinstance(topic, str) else base,
            "agg_reputation": pop.reputation(dom, default=0.0),
            "agg_competition": math.log1p(pop.competition.mean),
            "agg_drift": self.drift.indicator(),
        }
        if self.use_incentives:
            feats.update(self.incentives(action).as_features())
        feats.update(_content_features(action))
        return [feats.get(n, 0.0) for n in self.feature_names]

    def predict(self, pop: PopulationState, action: Action) -> dict:
        pred = self.head.predict(self.feature_vector(pop, action))
        pred["uncertainty_inflation"] = self.drift.inflation()
        return pred

    def transition(self, pop: PopulationState, action: Action, magnitude: float) -> PopulationState:
        """Advance the population state with an observed outcome magnitude (in-place)."""
        thr = self.thresholds[0]
        hit = 1.0 if magnitude >= thr else 0.0
        topic = action.content_features.get("topic", "other")
        dom = action.meta.get("domain", "")
        pop.observe_outcome(
            hit, subgroup_keys=self._subgroup_keys(action),
            topic=topic if isinstance(topic, str) else None, domain=dom or None,
            domain_value=math.log1p(magnitude))
        # attention/competition: a live item consumes attention; decay slowly
        pop.competition.observe(1.0, 0.1)
        pop.timestamp = action.timing.get("ts", pop.timestamp)
        self.drift.observe(hit)
        pop.drift = self.drift.summary()
        return pop
