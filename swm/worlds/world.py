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
from swm.entities.persona import PRIOR_STRENGTH, Persona, apply_correction, build_persona, segment_priors
from swm.entities.public_figure import PublicFigureProfile, PublicFigureResolver
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
    resolver: PublicFigureResolver | None = None      # public-figure lookup (bias-to-infer for strangers)
    _personas: dict[str, Persona] = field(default_factory=dict)
    _profiles: dict[str, PublicFigureProfile] = field(default_factory=dict)

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

    def _persona_asof(self, contact_id: str, ts: float, *, name: str | None = None,
                      domain: str = "", ask: str = "") -> Persona:
        hist = self.store.history_asof(contact_id, ts)
        p = build_persona(contact_id, hist, segment_reply_rate=self.seg_rate)
        # Bias to infer: when we have little/no private history on this contact, don't refuse — look
        # them up. If a resolver is configured and history is thin, fold public-figure web evidence
        # into the persona as confidence-weighted pseudo-observations (provenance: web).
        thin = p.n_sends < 3
        if self.resolver is not None and thin and (name or contact_id):
            profile = self.resolver.resolve(name or contact_id, domain=domain, ask=ask,
                                             channel="email")
            self._profiles[contact_id] = profile
            _fold_web_responsiveness(p, profile)
        return p

    def persona(self, contact_id: str, *, name: str | None = None, domain: str = "",
                ask: str = "") -> Persona:
        """Live persona (as-of now), cached until corrected or refit."""
        if contact_id not in self._personas:
            self._personas[contact_id] = self._persona_asof(contact_id, time.time(), name=name,
                                                            domain=domain, ask=ask)
        return self._personas[contact_id]

    def profile(self, contact_id: str) -> dict | None:
        """The public-figure evidence behind a resolved persona (audit), if any."""
        prof = self._profiles.get(contact_id)
        return prof.summary() if prof else None

    # ---------- prediction (calibrated) ----------

    def predict(self, contact_id: str, text: str, *, channel: str = "email",
                send_ts: float | None = None, name: str | None = None,
                domain: str = "", ask: str = "") -> dict:
        p = self.persona(contact_id, name=name, domain=domain, ask=ask)
        # No fitted readout is NOT a refusal any more. Fall back to an inference-only prediction from
        # the inferred persona (segment prior <- web/public-figure evidence <- any private history) and
        # label it UNVALIDATED — honest about provenance, but never a hard block. Bias to infer.
        if self.readout is None:
            return self._inference_prediction(contact_id, p, text, channel, send_ts)
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

    def _inference_prediction(self, contact_id: str, p: Persona, text: str, channel: str,
                              send_ts: float | None) -> dict:
        """Unfitted fallback: predict the recipient's inferred base responsiveness, nudged by a few
        transparent message-fit heuristics, with a wide interval from the persona posterior. Clearly
        graded 'unvalidated' — it is an inference, not a backtested readout, and we say so."""
        base = p.responsiveness.mean
        lo, hi = p.responsiveness.interval()
        # light, bounded message-fit adjustment (heuristic, NOT a calibrated readout)
        adj, drivers = _message_fit_adjustment(text)
        mean = min(0.97, max(0.005, base + adj))
        prof = self._profiles.get(contact_id)
        return {
            "report_type": "prediction",
            "outcome": "reply",
            "p_mean": round(mean, 4),
            "p_interval80": [round(max(0.0, lo + adj), 4), round(min(1.0, hi + adj), 4)],
            "calibration": {"grade": "unvalidated",
                            "note": "inference-only: no backtested fit for this world. p is the "
                                    "inferred base responsiveness + heuristic message-fit, not a "
                                    "graded readout. Import labeled sends and /fit to earn a grade."},
            "drivers": [{"feature": "inferred_base_responsiveness", "contribution": round(base, 4)},
                        *drivers],
            "provenance": {"base_responsiveness_n_effective": round(p.responsiveness.n_effective, 1),
                           "public_figure": prof.summary() if prof else None},
            "model_version": self.version + "-inferred",
            "as_of": send_ts or time.time(),
        }

    def compare(self, contact_id: str, texts: list[str], *, channel: str = "email",
                name: str | None = None) -> dict:
        preds = [self.predict(contact_id, t, channel=channel, name=name) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: preds[i]["p_mean"], reverse=True)
        return {
            "report_type": "prediction",
            "ranked": [{"index": i, "text": texts[i],
                        **{k: preds[i][k] for k in ("p_mean", "p_interval80", "drivers")}}
                       for i in order],
            "calibration": self.grade if self.readout is not None else preds[0]["calibration"],
            "model_version": self.version if self.readout is not None else self.version + "-inferred",
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


# ---------- inference-path helpers (used when no fitted readout exists) ---------------------------

def _fold_web_responsiveness(p: Persona, profile: PublicFigureProfile) -> None:
    """Fold a public-figure profile's inferred responsiveness into the persona's Beta posterior as
    confidence-weighted pseudo-observations. Weight = PRIOR_STRENGTH * confidence, so a thin lexical
    read barely moves it and a confident LLM read moves it toward the observed public rate — the same
    partial-pooling logic operator corrections use, but provenance is web."""
    resp = profile.responsiveness or {}
    mean = resp.get("mean")
    conf = float(resp.get("confidence", 0.0))
    if mean is None or conf <= 0.0:
        return
    weight = PRIOR_STRENGTH * conf
    p.responsiveness.update(mean * weight, (1.0 - mean) * weight)
    p.corrections["responsiveness(web)"] = round(float(mean), 3)


# small, transparent message-fit nudges for the unfitted fallback. NOT a calibrated readout — each is
# a bounded heuristic with a named driver, so the fallback prediction stays auditable and honest.
def _message_fit_adjustment(text: str):
    """Return (delta, drivers) — a small bounded adjustment to the base rate from cheap message-fit
    signals. Deliberately modest (|delta| <= ~0.12): with no fit data we must not fake precision."""
    import re
    t = text or ""
    words = max(1, len(t.split()))
    drivers, delta = [], 0.0

    pushy = len(re.findall(r"\b(asap|urgent|act now|circling back|just following up|per my last)\b", t, re.I))
    if pushy:
        d = -0.04 * min(2, pushy); delta += d
        drivers.append({"feature": "pushiness", "contribution": round(d, 4)})
    if "?" in t:
        delta += 0.03
        drivers.append({"feature": "explicit_ask", "contribution": 0.03})
    # length fit: ~15-90 words is the sweet spot for a cold ask; very long/very short suppress
    if words > 220:
        delta -= 0.06; drivers.append({"feature": "too_long", "contribution": -0.06})
    elif words < 8:
        delta -= 0.03; drivers.append({"feature": "too_short", "contribution": -0.03})
    personal = len(re.findall(r"\b(i saw your|i read your|congrat|your essay|your work on)\b", t, re.I))
    if personal:
        d = 0.03 * min(2, personal); delta += d
        drivers.append({"feature": "personalization", "contribution": round(d, 4)})
    return max(-0.12, min(0.12, delta)), drivers


IMPLEMENTED = True
