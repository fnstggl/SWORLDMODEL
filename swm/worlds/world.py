"""The per-population "world" object (audit C.11, J): fitted model + personas + calibration.

One World per customer/dataset. It owns:
- the event store handle
- the fitted ensemble readout + the training-window segment prior
- persona posteriors (built as-of "now" for live prediction; the harness builds its own as-of)
- the calibration grade from the most recent temporal backtest (echoed on every prediction)

Contract rules (audit J): /predict-, /compare-style outputs are PREDICTION (calibrated, graded);
draft generation and persona summaries are INSIGHT. The tag is on every payload.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from swm.actions.encoder import FEATURE_NAMES, encode_message, feature_vector
from swm.elicitation import ANSWER_VALUES, choose_voi_question
from swm.entities.persona import Persona, apply_correction, build_persona, segment_priors
from swm.eval.harness import _LOGIT, run_ladder
from swm.ingestion.store import EventStore
from swm.transition.readout import EnsembleReadout

ALL_FEATURE_NAMES = FEATURE_NAMES + ["segment_rate_logit", "person_rate_logit"]


@dataclass
class World:
    store: EventStore
    version: str = "v0"
    readout: EnsembleReadout | None = None
    seg_rate: float = 0.3
    grade: dict = field(default_factory=lambda: {"grade": "ungraded", "ece": None})
    backtest: dict = field(default_factory=dict)
    _personas: dict[str, Persona] = field(default_factory=dict)

    # ---------- fitting ----------

    def fit(self) -> dict:
        """Fit the readout on all labeled sends; run the temporal backtest for the grade."""
        sends = self.store.labeled_sends()
        if len(sends) < 30:
            return {"error": f"only {len(sends)} labeled sends; need >= 30"}
        per: dict[str, list[int]] = {}
        for s in sends:
            per.setdefault(s.recipient_id, []).append(1 if s.replied else 0)
        self.seg_rate = segment_priors([(sum(v), len(v)) for v in per.values()])

        X, y = [], []
        for s in sends:
            p = self._persona_asof(s.recipient_id, s.timestamp)
            f = encode_message(s.content, send_ts=s.timestamp, channel=s.channel, persona=p)
            X.append(feature_vector(f) + [_LOGIT(self.seg_rate), _LOGIT(p.responsiveness.mean)])
            y.append(1 if s.replied else 0)
        self.readout = EnsembleReadout(n_members=15).fit(X, y)

        # the honest grade comes from the temporal ladder, not training fit
        self.backtest = run_ladder(self.store)
        if "error" not in self.backtest:
            l4 = next(r for r in self.backtest["rungs"] if r["name"] == "L4")
            self.grade = {"grade": "A" if l4["ece"] < 0.05 else "B" if l4["ece"] < 0.10
                          else "C" if l4["ece"] < 0.15 else "F",
                          "ece": l4["ece"], "n": l4["n_test"],
                          "note": "graded on temporal holdout (L4 rung)"}
        self.version = f"v{int(len(sends))}"
        self._personas.clear()
        return {"fitted": True, "n_sends": len(sends), "grade": self.grade,
                "verdict": self.backtest.get("verdict")}

    def _persona_asof(self, contact_id: str, ts: float) -> Persona:
        hist = self.store.history_asof(contact_id, ts)
        return build_persona(contact_id, hist, segment_reply_rate=self.seg_rate)

    def persona(self, contact_id: str) -> Persona:
        """Live persona (as-of now), cached until corrected or refit."""
        if contact_id not in self._personas:
            self._personas[contact_id] = self._persona_asof(contact_id, time.time())
        return self._personas[contact_id]

    # ---------- prediction (calibrated) ----------

    def predict(self, contact_id: str, text: str, *, channel: str = "email",
                send_ts: float | None = None) -> dict:
        if self.readout is None:
            return {"error": "world not fitted; POST /fit first"}
        p = self.persona(contact_id)
        f = encode_message(text, send_ts=send_ts or time.time(), channel=channel, persona=p)
        x = feature_vector(f) + [_LOGIT(self.seg_rate), _LOGIT(p.responsiveness.mean)]
        mean, (lo, hi) = self.readout.predict(x)
        return {
            "report_type": "prediction",
            "outcome": "reply",
            "p_mean": round(mean, 4),
            "p_interval80": [round(lo, 4), round(hi, 4)],
            "calibration": self.grade,
            "drivers": [{"feature": n, "contribution": round(c, 4)}
                        for n, c in self.readout.drivers(x, ALL_FEATURE_NAMES)],
            "model_version": self.version,
            "as_of": send_ts or time.time(),
        }

    def compare(self, contact_id: str, texts: list[str], *, channel: str = "email") -> dict:
        preds = [self.predict(contact_id, t, channel=channel) for t in texts]
        if any("error" in p for p in preds):
            return {"error": "world not fitted; POST /fit first"}
        order = sorted(range(len(texts)), key=lambda i: preds[i]["p_mean"], reverse=True)
        return {
            "report_type": "prediction",
            "ranked": [{"index": i, "text": texts[i],
                        **{k: preds[i][k] for k in ("p_mean", "p_interval80", "drivers")}}
                       for i in order],
            "calibration": self.grade,
            "model_version": self.version,
        }

    # ---------- elicitation (insight) ----------

    def voi(self, contact_id: str, draft_text: str, *, channel: str = "email",
            who: str = "this person") -> dict | None:
        if self.readout is None:
            return None
        p = self.persona(contact_id)
        extra = [_LOGIT(self.seg_rate), _LOGIT(p.responsiveness.mean)]
        q = choose_voi_question(
            p, draft_text, lambda row: self.readout.predict(row)[0],
            extra_features=extra, channel=channel, who=who,
        )
        if q is None:
            return None
        return {"report_type": "insight", "factor": q.factor, "question": q.question,
                "answers": list(ANSWER_VALUES.get(q.factor, {})),
                "expected_value": q.value, "prediction_spread": q.prediction_spread}

    def correct(self, contact_id: str, factor: str, answer: str) -> dict:
        p = self.persona(contact_id)
        value = ANSWER_VALUES.get(factor, {}).get(answer)
        if value is None:
            try:
                value = float(answer)
            except ValueError:
                return {"error": f"unknown answer '{answer}' for factor '{factor}'"}
        apply_correction(p, factor, value)
        return {"report_type": "insight", "persona": p.summary()}


IMPLEMENTED = True
