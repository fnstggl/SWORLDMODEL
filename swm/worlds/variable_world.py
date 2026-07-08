"""VariableWorld — the world model built on the mapped-variable state (the core architecture).

Every prediction/simulation flows through the same pipeline the thesis demands:

    (entity, action, context)  ->  VariableInferenceEngine.infer  ->  VariableMap
                               ->  simulate / calibrated readout   ->  P(response) + next state

The `VariableMap` (all behavioral variables acting on the person, known + inferred, with provenance
and confidence) IS the state. The readout only reads the map's confidence-weighted features; it does
not bypass the variable mapping. Entity history is tracked online (as-of) so the DATA-provenance
variables (base responsiveness, relationship, recency) are always current and leakage-free.

Backtestable: `fit_stream` + `backtest` score the mapped-variable predictor on real outcomes, so the
variables — especially the LLM-inferred ones — must earn their place. `predict` also returns the map
(`explain`) so every prediction is auditable: which variables drove it, and from what provenance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.transition.readout import LogisticReadout
from swm.variables.inference import VariableInferenceEngine
from swm.variables.variable_map import VariableMap


@dataclass
class VariableWorld:
    platform: str = "generic"
    engine: VariableInferenceEngine = None            # type: ignore
    readout: object = None
    global_rate: float = 0.3
    _hist: dict = field(default_factory=dict)         # entity -> [(ts, outcome), ...] (as-of)

    def __post_init__(self):
        if self.engine is None:
            self.engine = VariableInferenceEngine(platform=self.platform)

    # ---- as-of history -> the data-provenance variables ----
    recency_halflife: int = 15                       # observations; recency-weighting of responsiveness

    def _history(self, entity_id, now=None):
        ev = self._hist.get(entity_id, [])
        if not ev:
            return {"n_prior": 0}
        outs = [o for _, o in ev]
        last = max(ts for ts, _ in ev)
        rec = ((now - last) / 86400.0) if (now is not None) else None
        # recency-weighted responsiveness (EWMA) — recent behavior predicts better than the lifetime
        # average (nonstationarity); this is the winning signal for time-sensitive channels (email).
        alpha = 1.0 - math.exp(-math.log(2) / self.recency_halflife)
        ewma = outs[0]
        for o in outs[1:]:
            ewma = alpha * o + (1 - alpha) * ewma
        return {"n_prior": len(outs), "response_rate": ewma,
                "response_rate_lifetime": sum(outs) / len(outs), "recency_days": rec}

    def infer(self, entity_id, action, context=None, *, now=None, user_context=None,
              llm_inference=None) -> VariableMap:
        return self.engine.infer(entity_id, action, context, history=self._history(entity_id, now),
                                 user_context=user_context, llm_inference=llm_inference)

    def _observe(self, entity_id, ts, outcome):
        self._hist.setdefault(entity_id, []).append((ts, outcome))

    # ---- fit ----
    def fit_stream(self, instances, *, global_rate=None):
        """instances: time-ordered (entity_id, action, context, outcome[, extras]).
        extras (optional dict) may carry user_context and llm_inference per instance."""
        insts = list(instances)
        ys = [i[3] for i in insts]
        self.global_rate = global_rate if global_rate is not None else (sum(ys) + 1) / (len(ys) + 2)
        self.engine.platform = self.platform
        self._hist.clear()
        X, y = [], []
        for entity_id, action, context, outcome, *extra in insts:
            ex = extra[0] if extra else {}
            now = getattr(action, "timing", {}).get("ts")
            vm = self.infer(entity_id, action, context, now=now,
                            user_context=ex.get("user_context"), llm_inference=ex.get("llm_inference"))
            X.append(vm.to_features()); y.append(int(outcome))
            self._observe(entity_id, now if now is not None else len(X), outcome)
        self.readout = LogisticReadout(epochs=250).fit(X, y) if len(set(y)) == 2 else None
        return self

    def predict(self, entity_id, action, context=None, *, now=None, user_context=None,
                llm_inference=None, explain=False):
        vm = self.infer(entity_id, action, context, now=now, user_context=user_context,
                        llm_inference=llm_inference)
        p = self.readout.predict_proba(vm.to_features()) if self.readout else self.global_rate
        out = {"p": p, "provenance": vm.provenance_report()}
        if explain:
            out["variables"] = vm.explain()
        return out

    def observe(self, entity_id, action, outcome):
        self._observe(entity_id, getattr(action, "timing", {}).get("ts", 0), outcome)

    # ---- backtest ----
    def backtest(self, instances, *, split=0.7):
        insts = list(instances)
        n = len(insts); cut = int(split * n)
        self.fit_stream(insts[:cut])
        preds, y = [], []
        for entity_id, action, context, outcome, *extra in insts[cut:]:
            ex = extra[0] if extra else {}
            now = getattr(action, "timing", {}).get("ts")
            p = self.predict(entity_id, action, context, now=now,
                             user_context=ex.get("user_context"),
                             llm_inference=ex.get("llm_inference"))["p"]
            preds.append(min(1 - 1e-6, max(1e-6, p))); y.append(int(outcome))
            self._observe(entity_id, now if now is not None else 0, outcome)
        return {"n_test": len(y), "base_rate": round(sum(y) / len(y), 4),
                "log_loss": round(log_loss(y, preds), 4), "brier": round(brier_score(y, preds), 4),
                "ece": round(expected_calibration_error(y, preds), 4),
                "uplift@20": round(uplift_at_k(y, preds, 0.2), 4)}, preds, y
