"""The unified prediction entry point — `Simulator.simulate(...)`.

This is the launch-readiness layer: it turns the validated VariableMap machinery into one call you can
point at any individual-regime question and get a number you can *trust or knowingly distrust*. It does
four things the raw `VariableWorld` does not:

1. REGIME ROUTING — every query is classified into the regime that actually carries its signal:
     entity_state    : the entity has enough as-of history that WHO dominates (GitHub/Enron reply).
     inference_driven: no useful entity state, but LLM-inferred latent variables are present
                       (persuasion / cold outreach) — the regime EXP-021 validated.
     message_only    : only cheap message/platform heuristics fire (weak but non-empty signal).
     cold_start      : nothing entity-specific and no inference — only population priors.
   Routing does not change the readout (one calibrated map-readout serves all regimes); it sets the
   HONEST CONFIDENCE and decides ABSTENTION, so a launched prediction never overstates what it knows.

2. HONEST CONFIDENCE — a [0,1] score from the map's provenance (history depth, inference presence,
   mean variable confidence) and whether the regime is one we have VALIDATED no-cheat.

3. ABSTENTION — when a query is a cold_start (outside the validated envelope) the call still returns
   the calibrated base rate but sets `abstain=True` with a reason, rather than presenting a prior as a
   confident prediction.

4. CALIBRATION BADGE — `fit()` grades calibration on a held-out temporal tail; the grade rides on every
   `Prediction`, per the project rule that a prediction without a grade is not allowed out the door.

The probability itself still flows through the one true pipeline: (entity, action, context) ->
VariableInferenceEngine -> VariableMap -> calibrated readout. Nothing here bypasses the variable map.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from swm.uncertainty.calibration import calibration_grade
from swm.variables.schema import NAMES
from swm.worlds.variable_world import VariableWorld

# regimes we have validated no-cheat on real outcomes (EXP-014/016 response, EXP-021 persuasion)
_VALIDATED_REGIMES = {"entity_state", "inference_driven"}
_MIN_HISTORY_FOR_STATE = 3          # prior interactions before WHO is a trustworthy signal
_ABSTAIN_BELOW = 0.30               # confidence floor; below this the query is outside the envelope


@dataclass
class Prediction:
    """One prediction, with everything a caller needs to trust or discount it."""
    p: float                        # calibrated probability of the outcome
    confidence: float               # [0,1] honest confidence in this specific prediction
    regime: str                     # entity_state | inference_driven | message_only | cold_start
    abstain: bool                   # True => outside the validated envelope; p is the prior, treat as weak
    reason: str                     # why this regime / confidence / abstention
    calibration: dict               # the model's calibration badge from fit() (grade + ECE + n)
    provenance: dict                # VariableMap.provenance_report()
    drivers: list = field(default_factory=list)   # top variables that moved this prediction

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Simulator:
    """A fitted, launch-ready individual-regime predictor. One `simulate()` for any (entity, action)."""
    platform: str = "generic"
    world: VariableWorld = None                     # type: ignore
    calibration: dict = field(default_factory=lambda: {"grade": "ungraded", "ece": None})
    base_rate: float = 0.3
    train_support: dict = field(default_factory=dict)   # regime -> fraction of training stream in it

    def __post_init__(self):
        if self.world is None:
            self.world = VariableWorld(platform=self.platform)

    # ---- fit: train the readout, then grade calibration on a held-out temporal tail ----
    def fit(self, instances, *, grade_split: float = 0.7) -> "Simulator":
        """instances: time-ordered (entity_id, action, context, outcome[, extras{user_context,llm_inference}]).
        Grades calibration on the last (1-grade_split) as a no-leakage tail, then refits on everything."""
        insts = list(instances)
        if len(insts) >= 40:
            grade_world = VariableWorld(platform=self.platform)
            badge, preds, y = grade_world.backtest(insts, split=grade_split)
            self.calibration = calibration_grade(y, preds)
            self.calibration.update({"log_loss": badge["log_loss"], "brier": badge["brier"],
                                     "uplift@20": badge["uplift@20"], "n_test": badge["n_test"]})
        # deploy model: fit on the full stream so entity history is maximally current
        self.world = VariableWorld(platform=self.platform).fit_stream(insts)
        self.base_rate = self.world.global_rate
        self.train_support = self._measure_support(insts)
        return self

    def _measure_support(self, insts) -> dict:
        """Which regimes did the training stream actually cover? A query whose regime the model barely
        saw is out-of-distribution — confidence must be discounted and abstention triggered. We replay
        the as-of maps (leakage-free: same history the fit saw) and route each to its regime."""
        counts = {"entity_state": 0, "inference_driven": 0, "message_only": 0, "cold_start": 0}
        w = VariableWorld(platform=self.platform)
        for entity_id, action, context, outcome, *extra in insts:
            ex = extra[0] if extra else {}
            now = getattr(action, "timing", {}).get("ts")
            vm = w.infer(entity_id, action, context, now=now, user_context=ex.get("user_context"),
                         llm_inference=ex.get("llm_inference"))
            counts[self._route(vm)[0]] += 1
            w.observe(entity_id, action, outcome)
        n = max(1, len(insts))
        return {k: round(v / n, 4) for k, v in counts.items()}

    # ---- the one entry point ----
    def simulate(self, entity_id, action, context=None, *, now=None, user_context=None,
                 llm_inference=None) -> Prediction:
        vm = self.world.infer(entity_id, action, context, now=now, user_context=user_context,
                              llm_inference=llm_inference)
        raw_p = self.world.readout.predict_proba(vm.to_features()) if self.world.readout else self.base_rate
        raw_p = min(1 - 1e-6, max(1e-6, raw_p))
        regime, ceiling, reason = self._route(vm)
        # within-regime confidence = prediction EXTREMITY: how far the model moves from the base rate,
        # normalized by the max reachable distance. EXP-024 shows extremity tracks accuracy monotonically
        # (selective log loss drops with it) while the LLM's self-reported confidence does not — so the
        # regime sets the ceiling (cross-regime trust) and extremity sets the position under it.
        extremity = abs(raw_p - self.base_rate) / max(self.base_rate, 1 - self.base_rate, 1e-6)
        conf = ceiling * (0.45 + 0.55 * min(1.0, extremity))
        # discount confidence by how much of this regime the fitted model actually saw (OOD guard)
        support = self.train_support.get(regime, 1.0) if self.train_support else 1.0
        if self.train_support and support < 0.05:
            conf *= 0.4
            reason += f"; regime unsupported by this model (train support {support:.0%}) — out of distribution"
        conf = round(min(0.98, conf), 3)
        abstain = regime not in _VALIDATED_REGIMES or conf < _ABSTAIN_BELOW or support < 0.05
        # trust the validated readout in-envelope (preserve the full no-cheat signal); shrink toward the
        # base rate ONLY when abstaining, so an out-of-distribution or low-reliability query can never
        # emit an overconfident 0.00/1.00. Confidence (extremity-based) is the reported reliability and
        # is kept separate from this safety shrink so it does not wash out in-regime signal.
        if abstain:
            p = 0.25 * raw_p + 0.75 * self.base_rate
            reason += f"; abstaining — shrunk toward base rate {self.base_rate:.3f} (treat as weak)"
        else:
            p = raw_p
        return Prediction(p=round(min(1 - 1e-6, max(1e-6, p)), 6), confidence=conf, regime=regime,
                          abstain=abstain, reason=reason, calibration=dict(self.calibration),
                          provenance=vm.provenance_report(), drivers=vm.explain(top=6))

    def observe(self, entity_id, action, outcome) -> None:
        """Feed a realized outcome back so the entity's as-of state stays current (online use)."""
        self.world.observe(entity_id, action, outcome)

    # ---- regime routing + honest confidence (the launch-readiness core) ----
    def _route(self, vm):
        # entity-specific data signal: how much as-of history informs WHO
        state_vars = [vm.vars.get(k) for k in ("base_responsiveness", "relationship_strength",
                                               "recency_of_contact")]
        state_conf = max([v.confidence for v in state_vars if v and v.provenance == "data"] + [0.0])
        has_state = state_conf > 0.0
        # inferred latent signal: any LLM-provenance variable beyond priors
        llm_vars = [v for v in vm.vars.values() if v.provenance == "llm"]
        llm_conf = (sum(v.confidence for v in llm_vars) / len(llm_vars)) if llm_vars else 0.0
        # message/platform heuristic signal
        heur = [v for v in vm.vars.values() if v.provenance in ("heuristic",)]
        heur_conf = (sum(v.confidence for v in heur) / len(heur)) if heur else 0.0

        if has_state:
            regime = "entity_state"
            # confidence scales with how much history (state_conf already saturates with n_prior)
            conf = 0.45 + 0.5 * state_conf
            reason = f"entity has as-of history (data-confidence {state_conf:.2f}); WHO dominates"
            if llm_vars:
                conf = min(0.98, conf + 0.05); reason += " + inferred variables present"
        elif llm_vars:
            regime = "inference_driven"
            conf = 0.35 + 0.5 * llm_conf
            reason = (f"no entity history; {len(llm_vars)} LLM-inferred latent variables "
                      f"(mean confidence {llm_conf:.2f}) carry the signal")
        elif heur_conf > 0.0:
            regime = "message_only"
            conf = 0.20 + 0.4 * heur_conf
            reason = "no entity history and no inference; only message/platform heuristics fire (weak)"
        else:
            regime = "cold_start"
            conf = 0.10
            reason = "no entity history, no inference, no message signal — outside the validated envelope"
        return regime, min(0.98, conf), reason
