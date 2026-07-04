"""LLM services: structured trait extraction + goal-conditioned draft generation.

Two jobs, both deliberately bounded (audit C.8, design note §3):
1. extract_traits(their past messages) -> structured persona evidence WITH uncertainty.
   The LLM turns tacit prose into features; it never sees the outcome label, so it cannot
   leak the answer into the backtest.
2. generate_drafts(persona, goal) -> K candidate messages. Tagged INSIGHT; the discriminative
   readout scores/ranks them (that ranking is the PREDICTION).

Degrades gracefully: if the anthropic SDK or credentials are unavailable, heuristic
fallbacks keep the pipeline runnable (clearly marked lower-quality).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from swm.entities.persona import Persona, formality_score

MODEL = "claude-opus-4-8"

_TRAIT_SCHEMA = {
    "type": "object",
    "properties": {
        "verbosity_preference": {
            "type": "string", "enum": ["short", "medium", "detailed"],
            "description": "How long this person's own messages tend to be / what they likely prefer receiving",
        },
        "formality": {"type": "string", "enum": ["casual", "neutral", "formal"]},
        "warmth": {"type": "string", "enum": ["warm", "neutral", "reserved"]},
        "reactivity": {
            "type": "string", "enum": ["easygoing", "neutral", "easily_irritated"],
            "description": "Evidence of irritation at pushiness/fluff in their replies",
        },
        "confidence": {
            "type": "string", "enum": ["low", "medium", "high"],
            "description": "How much evidence the messages actually contain for these judgments",
        },
        "notes": {"type": "string", "description": "One sentence of concrete evidence, quoting them if possible"},
    },
    "required": ["verbosity_preference", "formality", "warmth", "reactivity", "confidence", "notes"],
    "additionalProperties": False,
}


def _client():
    try:
        import anthropic  # noqa: PLC0415

        c = anthropic.Anthropic()
        return c
    except Exception:
        return None


def extract_traits(their_messages: list[str]) -> dict:
    """Structured persona evidence from the person's own text. Returns the schema above
    plus {"source": "llm" | "heuristic"}. Never sees outcomes; safe for backtest features."""
    text = "\n---\n".join(m.strip() for m in their_messages if m.strip())[:8000]
    if not text:
        return {"source": "none", "confidence": "low"}
    client = _client()
    if client is None:
        return _heuristic_traits(text)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=(
                "You infer communication-style traits from a person's own past messages. "
                "Be conservative: mark confidence 'low' unless the evidence is clear and repeated. "
                "Judge only style, never demographics or protected attributes."
            ),
            messages=[{"role": "user", "content": f"Messages this person wrote:\n\n{text}"}],
            output_config={"format": {"type": "json_schema", "schema": _TRAIT_SCHEMA}},
        )
        block = next(b for b in response.content if b.type == "text")
        traits = json.loads(block.text)
        traits["source"] = "llm"
        return traits
    except Exception:
        return _heuristic_traits(text)


def _heuristic_traits(text: str) -> dict:
    words = text.split()
    avg_msg_words = len(words) / max(1, text.count("---") + 1)
    form = formality_score(text)
    return {
        "verbosity_preference": "short" if avg_msg_words < 20 else "detailed" if avg_msg_words > 80 else "medium",
        "formality": "casual" if form < 0.35 else "formal" if form > 0.65 else "neutral",
        "warmth": "neutral",
        "reactivity": "neutral",
        "confidence": "low",
        "notes": "heuristic fallback (no LLM available): lexical features only",
        "source": "heuristic",
    }


TRAIT_TO_FACTOR = {
    # trait value -> (persona factor, correction value) for apply_correction
    ("verbosity_preference", "short"): ("verbosity", 2.3),
    ("verbosity_preference", "medium"): ("verbosity", 3.3),
    ("verbosity_preference", "detailed"): ("verbosity", 4.2),
    ("formality", "casual"): ("formality", 0.15),
    ("formality", "neutral"): ("formality", 0.5),
    ("formality", "formal"): ("formality", 0.85),
}

_CONFIDENCE_WEIGHT = {"low": 0.3, "medium": 0.6, "high": 1.0}


def traits_to_corrections(traits: dict) -> list[tuple[str, float, float]]:
    """(factor, value, confidence-weight) triples for persona.apply_correction."""
    w = _CONFIDENCE_WEIGHT.get(traits.get("confidence", "low"), 0.3)
    out = []
    for key in ("verbosity_preference", "formality"):
        mapping = TRAIT_TO_FACTOR.get((key, traits.get(key)))
        if mapping:
            out.append((mapping[0], mapping[1], w))
    return out


@dataclass
class Draft:
    text: str
    rationale: str
    source: str  # "llm" | "template"


def generate_drafts(persona: Persona, goal: str, *, channel: str = "email", k: int = 3,
                    context: str = "") -> list[Draft]:
    """K candidate messages tailored to the persona and goal. INSIGHT-tagged output;
    the readout ranks them and that ranking is the prediction."""
    client = _client()
    if client is None:
        return _template_drafts(persona, goal, channel, k)
    style = persona.summary()
    schema = {
        "type": "object",
        "properties": {
            "drafts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "rationale": {"type": "string"}},
                    "required": ["text", "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["drafts"],
        "additionalProperties": False,
    }
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=(
                "You draft candidate outbound messages. Write like a competent human, never salesy-slop. "
                "Match the recipient's inferred style. Vary the drafts meaningfully (length, angle, directness) "
                "so a downstream ranker has real choices. No placeholder brackets unless unavoidable."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Channel: {channel}\nGoal: {goal}\nContext: {context or 'n/a'}\n"
                    f"Recipient style posterior (inferred, uncertain): {json.dumps(style)}\n"
                    f"Write {k} distinct candidate messages."
                ),
            }],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        block = next(b for b in response.content if b.type == "text")
        data = json.loads(block.text)
        return [Draft(d["text"], d["rationale"], "llm") for d in data["drafts"][:k]]
    except Exception:
        return _template_drafts(persona, goal, channel, k)


def _template_drafts(persona: Persona, goal: str, channel: str, k: int) -> list[Draft]:
    short = persona.verbosity.mean < math.log(25)
    casual = persona.formality.mean < 0.4
    greet = "Hey" if casual else "Hi"
    variants = [
        Draft(f"{greet} — quick one: {goal}. Worth a chat?",
              "short + direct (matches low-verbosity preference)" if short else "short + direct", "template"),
        Draft(f"{greet}, hope you're well. I wanted to reach out about {goal.lower()}. "
              f"Would you have 15 minutes this week to talk it through?",
              "medium length, soft ask", "template"),
        Draft(f"{greet} — {goal}. If now's a bad time, just say so and I'll leave it here.",
              "pressure-release framing for reactive recipients", "template"),
    ]
    return variants[:k]
