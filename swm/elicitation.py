"""Engine A: elicitation — correct-a-guess + the one value-of-information question.

The machine adapts to the human (design note §3A). We never show a form. We show the inferred
persona as a falsifiable guess the operator can correct (apply_correction in entities/persona),
and we compute which SINGLE unknown, if resolved, would most change the prediction — and ask
only that.

VOI here is sensitivity-based (v1): for each uncertain factor, sweep it across its posterior
interval, re-predict, and measure the spread of P(reply). The factor with the largest
prediction spread x posterior uncertainty is the one worth asking about. This is a greedy
myopic VOI — standard, cheap, and good enough until the readout is nonlinear.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from swm.actions.encoder import encode_message, feature_vector
from swm.entities.persona import Persona

# Human-readable question templates per factor. Forced-choice, not open-ended (§3A.4).
_QUESTIONS = {
    "responsiveness": "Does {who} usually reply to messages like this — even if briefly?",
    "verbosity": "Does {who} prefer short punchy messages, or fuller detailed ones?",
    "formality": "Is {who} more casual (texting register) or more formal/professional?",
    "latency": "Does {who} typically respond within hours, or after days?",
}

_SWEEPS = {
    # factor -> (low, high) values to sweep when uncertain
    "responsiveness": (0.1, 0.7),
    "verbosity": (2.0, 4.5),     # log-words: ~7 words .. ~90 words
    "formality": (0.1, 0.9),
    "latency": (7.0, 12.0),      # log-sec: ~20min .. ~1.8 days
}


@dataclass
class VOIQuestion:
    factor: str
    question: str
    prediction_spread: float   # how much P(reply) moves across the factor's plausible range
    posterior_uncertainty: float
    value: float               # spread x uncertainty — the ranking key


def choose_voi_question(
    persona: Persona,
    draft_text: str,
    predict_fn,           # (feature_vector: list[float]) -> float
    *,
    extra_features: list[float] | None = None,
    channel: str = "email",
    who: str = "this person",
) -> VOIQuestion | None:
    """Return the single most valuable question, or None if nothing moves the needle (< 2pts)."""
    candidates: list[VOIQuestion] = []
    for factor, (lo, hi) in _SWEEPS.items():
        unc = (
            1.0 / persona.responsiveness.n_effective ** 0.5
            if factor == "responsiveness"
            else getattr(persona, factor).uncertainty
        )
        preds = []
        for v in (lo, hi):
            p = copy.deepcopy(persona)
            if factor == "responsiveness":
                p.responsiveness.alpha = v * 10
                p.responsiveness.beta = (1 - v) * 10
            else:
                getattr(p, factor).mean = v
            f = encode_message(draft_text, channel=channel, persona=p)
            row = feature_vector(f) + (extra_features or [])
            preds.append(predict_fn(row))
        spread = abs(preds[1] - preds[0])
        candidates.append(
            VOIQuestion(
                factor=factor,
                question=_QUESTIONS[factor].format(who=who),
                prediction_spread=round(spread, 4),
                posterior_uncertainty=round(unc, 3),
                value=round(spread * unc, 5),
            )
        )
    best = max(candidates, key=lambda q: q.value)
    return best if best.prediction_spread >= 0.02 else None


# Mapping from forced-choice answers to correction values (feeds apply_correction).
ANSWER_VALUES = {
    "responsiveness": {"usually replies": 0.65, "rarely replies": 0.12, "depends": 0.35},
    "verbosity": {"short & punchy": 2.3, "full & detailed": 4.2, "depends": 3.3},
    "formality": {"casual": 0.15, "formal": 0.85, "in between": 0.5},
    "latency": {"within hours": 8.0, "after days": 11.8, "varies": 10.0},
}
