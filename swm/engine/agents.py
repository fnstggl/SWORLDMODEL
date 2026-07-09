"""Reasoning agents — Stage 2: each actor becomes personas whose 'variables' live INSIDE the agent.

The per-agent state is a grounded DOSSIER (identity, current position, information they'd plausibly have,
incentives, relationships) plus a sampled VARIANT — and the LLM applies all of it contextually when the
agent decides. No global coefficients; no scalar `position` nudged by ODE constants. The elasticity of an
endorsement on a 34-year-old progressive renter is computed by the model reading the situation as that
person, which is the thing LLMs are actually good at.

Diversity (the anti-monoculture defenses, deliberate and layered — silicon populations agree too much and
understate tails unless forced apart):
  - VARIANTS: a segment is instantiated as several DISTINCT concrete personas (drawn by the LLM: different
    ages, jobs, information diets, engagement); a named individual as several latent states (busy/attentive,
    skeptical/curious — the states we cannot observe from outside).
  - PRIVATE INFORMATION: each variant gets a rotated, partial slice of the evidence — real people have not
    read everything.
  - TEMPERATURE: decision calls sample at temperature > 0, so the same persona is not deterministic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.engine.grounding import parse_json


@dataclass
class Persona:
    actor_name: str                    # the cast actor this persona instantiates
    kind: str                          # named | segment
    weight: float                      # this persona's share of its actor's outcome weight
    dossier: str                       # grounded identity/incentives/info — WHO this is
    variant: str = ""                  # the sampled concrete draw / latent state
    private_facts: list = field(default_factory=list)   # the evidence slice this persona has seen


_VARIANTS_PROMPT = """We are casting "{name}" ({role}) in a grounded social simulation of:
QUESTION: {q}
GROUNDED SCENE:
{scene}

{mode_instructions}

Return ONLY JSON: {{"variants": [{{"sketch": "<2-3 sentences: concrete identity, incentives, information
diet, disposition — specific enough that different variants would plausibly act differently>"}}]}}"""

_SEGMENT_MODE = """"{name}" is a population SEGMENT. Draw {k} DISTINCT concrete individuals from it —
different ages/occupations/attention-to-politics/media diets consistent with the segment. Real populations
are more dispersed than a stereotype: include at least one low-attention or cross-pressured draw."""

_NAMED_MODE = """"{name}" is a real person. Draw {k} plausible LATENT STATES for them right now — the
unobservables: attention/busyness, current priorities, mood toward this kind of situation, who has their
ear. States must be consistent with the grounded facts but genuinely different from each other."""


def draw_variants(llm, actor, question, scene_brief) -> list:
    """Instantiate an actor as k diverse personas. Falls back to one neutral persona on parse failure."""
    k = actor.n_variants
    mode = (_NAMED_MODE if actor.kind == "named" else _SEGMENT_MODE).format(name=actor.name, k=k)
    raw = parse_json(llm(_VARIANTS_PROMPT.format(name=actor.name, role=actor.role, q=question,
                                                 scene=scene_brief, mode_instructions=mode))) or {}
    sketches = [str(v.get("sketch", ""))[:500] for v in raw.get("variants", []) if isinstance(v, dict)]
    sketches = [s for s in sketches if s] or [f"{actor.name}: {actor.role}"]
    w = actor.weight / len(sketches)
    return [Persona(actor_name=actor.name, kind=actor.kind, weight=w,
                    dossier=f"{actor.name} — {actor.role}", variant=s) for s in sketches[:k]]


_DECIDE_PROMPT = """You are simulating ONE person inside a grounded social world model. Reason AS this
person — from who they are, what they know, and what it costs or gains them. Do not be an analyst; be them.

DATE: {date}
WHO YOU ARE: {dossier}
YOUR CURRENT STATE/DRAW: {variant}
WHAT YOU HAVE SEEN (your information — you have NOT necessarily seen everything):
{private}
WHAT IS PUBLICLY HAPPENING RIGHT NOW: {public}

THE SITUATION: {q}
{time_note}

As this person, on this date, where do you land? Give your probability of taking each option — YOUR
behavior, not a pundit's forecast of the aggregate. If you're the kind of person who wouldn't engage/vote/
reply, say so via the probabilities. Then one sentence you might say out loud about it (or "" if you'd
say nothing publicly).

OPTIONS: {options}
Return ONLY JSON: {{"probs": {{"<option>": <0..1>, ...}}, "statement": "<one sentence or ''>",
"why": "<one short sentence of this person's actual reason>"}}"""


def decide(llm, persona: Persona, question, options, *, date="", public="(nothing new)",
           time_note="") -> dict:
    """One persona's decision distribution at one dated round. Returns normalized probs + statement + why.
    None on parse failure (the round drops this persona's sample rather than inventing one)."""
    private = "\n".join(f"- {f}" for f in persona.private_facts[:8]) or "- (general awareness only)"
    raw = parse_json(llm(_DECIDE_PROMPT.format(
        date=date, dossier=persona.dossier, variant=persona.variant, private=private,
        public=public, q=question, time_note=time_note, options=json.dumps(list(options)))))
    if not raw or not isinstance(raw.get("probs"), dict):
        return None
    probs = {}
    for o in options:
        try:
            probs[o] = max(0.0, float(raw["probs"].get(o, 0.0)))
        except (TypeError, ValueError):
            probs[o] = 0.0
    z = sum(probs.values())
    if z <= 0:
        return None
    return {"probs": {o: p / z for o, p in probs.items()},
            "statement": str(raw.get("statement", ""))[:200], "why": str(raw.get("why", ""))[:200]}


def slice_private_facts(facts, idx, keep=0.7):
    """Rotated partial evidence slice for persona #idx — private information, not omniscience."""
    if not facts:
        return []
    n = max(1, int(len(facts) * keep))
    start = (idx * 3) % len(facts)
    rotated = facts[start:] + facts[:start]
    return [f"{f['fact']}: {f.get('detail', '')}" for f in rotated[:n]]
