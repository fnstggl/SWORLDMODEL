"""Factored persona posteriors with hierarchical partial pooling (design note §2–3).

Every factor is a POSTERIOR (mean + evidence weight), never a point value. The pooling chain is
person <- segment <- population: with zero data you get the segment prior (wide); each observation
shrinks you toward the individual. This is the bridge between the aggregate and individual regimes
and the fix for both cold-start and variance-flattening.

v1 keeps the math deliberately simple and inspectable:
- responsiveness: Beta-Binomial (reply outcomes)
- verbosity / latency / formality: Normal with known-variance conjugate updates
Operator corrections (correct-a-guess) enter as pseudo-observations with an evidence weight,
so human tacit knowledge and behavioral data live in the same posterior.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field

from swm.ingestion.schema import Event

# Prior strength: how many observations the segment prior is "worth".
PRIOR_STRENGTH = 4.0


@dataclass
class BetaPosterior:
    """Posterior over a rate (e.g., responsiveness = P(they reply))."""
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n_effective(self) -> float:
        return self.alpha + self.beta

    def interval(self, z: float = 1.28) -> tuple[float, float]:
        """~80% interval via normal approx to the Beta (fine for UI; not for inference)."""
        m = self.mean
        var = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        s = math.sqrt(max(var, 1e-12))
        return (max(0.0, m - z * s), min(1.0, m + z * s))

    def update(self, successes: float, failures: float) -> None:
        self.alpha += successes
        self.beta += failures


@dataclass
class NormalPosterior:
    """Posterior over a continuous style factor (verbosity, formality, log-latency)."""
    mean: float
    n: float  # evidence weight (pseudo-count)

    def update(self, value: float, weight: float = 1.0) -> None:
        self.mean = (self.mean * self.n + value * weight) / (self.n + weight)
        self.n += weight

    @property
    def uncertainty(self) -> float:
        """Shrinks with evidence; 1.0 = prior-only. Used by the VOI chooser."""
        return 1.0 / math.sqrt(self.n)


@dataclass
class Persona:
    """The factored latent state for one contact. All fields are posteriors."""
    contact_id: str
    responsiveness: BetaPosterior
    verbosity: NormalPosterior      # their typical reply length, log-words
    formality: NormalPosterior      # 0 casual .. 1 formal (lexical heuristic or LLM)
    latency: NormalPosterior        # their typical reply latency, log-seconds
    n_sends: int = 0
    n_replies: int = 0
    corrections: dict[str, float] = field(default_factory=dict)  # operator overrides applied

    def summary(self) -> dict:
        lo, hi = self.responsiveness.interval()
        return {
            "contact_id": self.contact_id,
            "responsiveness": {"mean": round(self.responsiveness.mean, 3),
                               "interval80": [round(lo, 3), round(hi, 3)],
                               "n_effective": round(self.responsiveness.n_effective, 1)},
            "verbosity_logwords": {"mean": round(self.verbosity.mean, 2),
                                   "uncertainty": round(self.verbosity.uncertainty, 2)},
            "formality": {"mean": round(self.formality.mean, 2),
                          "uncertainty": round(self.formality.uncertainty, 2)},
            "latency_logsec": {"mean": round(self.latency.mean, 2),
                               "uncertainty": round(self.latency.uncertainty, 2)},
            "evidence": {"sends": self.n_sends, "replies": self.n_replies},
            "corrections": dict(self.corrections),
        }


_FORMAL_MARKERS = re.compile(r"\b(dear|sincerely|regards|best regards|hereby|pursuant|kindly)\b", re.I)
_CASUAL_MARKERS = re.compile(r"\b(lol|haha|yeah|yep|nah|gonna|wanna|hey|omg|btw|u|ur)\b|!{2,}|\bok\b", re.I)


def formality_score(text: str) -> float:
    """Cheap lexical formality in [0,1]. Replaceable by the LLM extractor (swm/llm.py)."""
    if not text.strip():
        return 0.5
    f = len(_FORMAL_MARKERS.findall(text))
    c = len(_CASUAL_MARKERS.findall(text))
    if f == c == 0:
        return 0.5
    return f / (f + c)


def build_persona(
    contact_id: str,
    history: list[Event],
    *,
    segment_reply_rate: float = 0.3,
    segment_verbosity: float = math.log(30),
    segment_formality: float = 0.5,
    segment_latency: float = math.log(4 * 3600),
) -> Persona:
    """Infer the persona posterior from an as-of history slice (Engine B).

    The segment_* arguments ARE the hierarchical prior — pass pooled values computed from the
    training window (never from the future). Their PRIOR_STRENGTH controls shrinkage.
    """
    p = Persona(
        contact_id=contact_id,
        responsiveness=BetaPosterior(alpha=segment_reply_rate * PRIOR_STRENGTH,
                                     beta=(1 - segment_reply_rate) * PRIOR_STRENGTH),
        verbosity=NormalPosterior(mean=segment_verbosity, n=PRIOR_STRENGTH),
        formality=NormalPosterior(mean=segment_formality, n=PRIOR_STRENGTH),
        latency=NormalPosterior(mean=segment_latency, n=PRIOR_STRENGTH),
    )
    # Walk the history: outbound sends to them, and their inbound messages.
    events = sorted(history, key=lambda e: e.timestamp)
    for i, e in enumerate(events):
        direction = e.features.get("direction")
        text = e.features.get("content", "") or ""
        if direction == "in" and e.actor_id == contact_id:
            words = max(1, len(text.split()))
            p.verbosity.update(math.log(words))
            p.formality.update(formality_score(text))
        if direction == "out" and contact_id in e.features.get("targets", []):
            p.n_sends += 1
            # replied if inbound from them arrives within the window AND before our next
            # outbound to them (same anti-inflation rule as store.labeled_sends)
            window = 7 * 86400.0 if e.channel == "email" else 86400.0
            window_end = e.timestamp + window
            nxt_out = next(
                (f.timestamp for f in events[i + 1:]
                 if f.features.get("direction") == "out"
                 and contact_id in f.features.get("targets", [])),
                None,
            )
            if nxt_out is not None:
                window_end = min(window_end, nxt_out)
            reply = next(
                (f for f in events[i + 1:]
                 if f.actor_id == contact_id and f.features.get("direction") == "in"
                 and e.timestamp < f.timestamp <= window_end),
                None,
            )
            if reply is not None and (reply.timestamp - e.timestamp) >= 60.0:
                p.n_replies += 1
                p.responsiveness.update(1, 0)
                p.latency.update(math.log(max(60.0, reply.timestamp - e.timestamp)))
            else:
                p.responsiveness.update(0, 1)
    return p


def apply_correction(p: Persona, factor: str, value: float, confidence: float = 1.0) -> Persona:
    """Correct-a-guess (Engine A): an operator judgment enters as pseudo-observations.

    confidence 1.0 ~ worth PRIOR_STRENGTH observations; the operator can be outvoted by
    enough contrary behavioral evidence — by design.
    """
    weight = PRIOR_STRENGTH * confidence
    if factor == "responsiveness":
        p.responsiveness.update(value * weight, (1 - value) * weight)
    elif factor in ("verbosity", "formality", "latency"):
        getattr(p, factor).update(value, weight)
    else:
        raise ValueError(f"unknown persona factor: {factor}")
    p.corrections[factor] = value
    return p


def segment_priors(personas_history: list[tuple[int, int]]) -> float:
    """Pooled segment reply rate from (n_replies, n_sends) pairs — training window only."""
    replies = sum(r for r, _ in personas_history)
    sends = sum(s for _, s in personas_history)
    return (replies + 1.0) / (sends + 2.0)  # Laplace


def robust_mean(values: list[float], default: float) -> float:
    return statistics.median(values) if values else default
