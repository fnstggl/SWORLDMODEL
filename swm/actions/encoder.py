"""Action encoder: a candidate message -> features the readout conditions on (audit C.7).

Two feature families:
1. Intrinsic message features (length, questions, reading ease, CTA, links, greeting, timing).
2. INTERACTION features against the recipient's persona (style match) — the design note's
   central claim that latents only matter in interaction with message features. "Gets mad easily"
   lives here as e.g. pushiness x reactivity; v1 ships the style-match subset.

Pure python, deterministic, no model calls — so the backtest is cheap and reproducible.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from swm.entities.persona import Persona, formality_score

_CTA_RE = re.compile(
    r"\b(call|book|schedule|sign up|register|buy|order|click|reply|let me know|confirm|"
    r"are you free|can you|could you|would you|when works|thoughts\?)\b", re.I)
_GREETING_RE = re.compile(r"^\s*(hi|hey|hello|dear|good (morning|afternoon|evening))\b", re.I)
_URL_RE = re.compile(r"https?://\S+")
_PUSHY_RE = re.compile(
    r"\b(urgent|asap|immediately|last chance|final|act now|don'?t miss|limited time|"
    r"just following up|bumping|circling back|per my last)\b", re.I)

FEATURE_NAMES = [
    "log_words", "n_questions", "has_greeting", "n_links", "n_cta",
    "reading_ease", "formality", "pushiness", "hour_sin", "hour_cos", "is_weekend",
    "is_email",
    # interaction features (persona-dependent)
    "len_match", "formality_match",
]


def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def reading_ease(text: str) -> float:
    """Flesch reading ease, squashed to [0,1] (1 = easiest)."""
    words = re.findall(r"[A-Za-z']+", text)
    sentences = max(1, len(re.findall(r"[.!?]+", text)) or 1)
    if not words:
        return 0.5
    wps = len(words) / sentences
    spw = sum(_syllables(w) for w in words) / len(words)
    flesch = 206.835 - 1.015 * wps - 84.6 * spw
    return min(1.0, max(0.0, flesch / 100.0))


def encode_message(
    text: str,
    *,
    send_ts: float | None = None,
    channel: str = "email",
    persona: Persona | None = None,
) -> dict[str, float]:
    """Feature dict (stable key order via FEATURE_NAMES)."""
    words = text.split()
    n_words = max(1, len(words))
    hour, weekend = 12.0, 0.0
    if send_ts is not None:
        dt = datetime.fromtimestamp(send_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        weekend = 1.0 if dt.weekday() >= 5 else 0.0
    form = formality_score(text)
    f = {
        "log_words": math.log(n_words),
        "n_questions": float(text.count("?")),
        "has_greeting": 1.0 if _GREETING_RE.search(text) else 0.0,
        "n_links": float(len(_URL_RE.findall(text))),
        "n_cta": float(len(_CTA_RE.findall(text))),
        "reading_ease": reading_ease(text),
        "formality": form,
        "pushiness": float(len(_PUSHY_RE.findall(text))),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "is_weekend": weekend,
        "is_email": 1.0 if channel == "email" else 0.0,
        "len_match": 0.0,
        "formality_match": 0.0,
    }
    if persona is not None:
        # style match: negative |difference| between message style and their inferred preference,
        # so 0 = perfect match and more negative = bigger mismatch.
        f["len_match"] = -abs(math.log(n_words) - persona.verbosity.mean)
        f["formality_match"] = -abs(form - persona.formality.mean)
    return f


def feature_vector(f: dict[str, float]) -> list[float]:
    return [f[k] for k in FEATURE_NAMES]
