"""Dossier assembly — Pillar 1 made real: gather every scrap of evidence about an entity, from whatever
source exists, so the world model infers that entity's variables from a POSTERIOR (evidence-conditioned),
never a cold prior.

EXP-086 proved the whole game: reference-class-only inference lands at ~23% of the way to a measurement, but
give the model real EVIDENCE about the individual and it jumps to ~87%. For a PUBLIC figure that evidence is
the LLM's own knowledge / a web search; for a PRIVATE individual (the person you're messaging, a lead, a
customer) the model can't look them up — so the evidence has to come from (a) the message history / their
posts, and (b) THE USER, who knows things no dataset does: the relationship, how they met, what the person
seems to care about, their read on them. This module assembles all of that into one dossier and hands it to
the grounded-inference engine as the evidence bundle.

  - `DossierAssembler.assemble(...)` — merge user-supplied context + message history + web footprint into a
    priority-ordered `Dossier` (user context first — it is the highest-signal thing for a private person).
  - `needs_user_context` / `context_questions` — when the public footprint is thin, tell the caller to ASK
    the user (drives an interactive prompt), with the specific questions that most improve the inference.
  - `infer_variables` — run each variable through the AnchoredExtractor with the dossier as evidence: the
    full three-pillar estimate (evidence → shrunk toward the reference-class base rate by calibrated
    uncertainty), so a rich dossier gives a sharp estimate and a thin one degrades honestly to the outside view.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Dossier:
    """The assembled evidence about an entity — the Pillar-1 substrate for grounded inference."""
    entity: str
    passages: list = field(default_factory=list)      # evidence snippets, highest-signal first
    tags: list = field(default_factory=list)          # provenance per passage: 'user' | 'history' | 'web'

    @property
    def strength(self) -> float:
        """A rough evidence-strength score in [0,1]: user context and history count more than web snippets.
        Feeds the honest 'do we need to ask the user?' gate and the estimate's confidence."""
        w = {"user": 1.0, "history": 0.7, "web": 0.4}
        return min(1.0, sum(w.get(t, 0.3) for t in self.tags) / 3.0)


@dataclass
class DossierAssembler:
    """Assemble a dossier for an entity from all available sources, in priority order. `search_fn(query,
    as_of) -> [passage,...]` is the same web backend the grounder uses (optional; None = no public lookup)."""
    search_fn: object = None
    k_web: int = 6

    def assemble(self, entity, *, user_context=None, message_history=None, question=None, as_of=None) -> Dossier:
        passages, tags = [], []
        if user_context:                                  # highest signal for a private individual
            for chunk in (user_context if isinstance(user_context, list) else [user_context]):
                if str(chunk).strip():
                    passages.append(str(chunk).strip())
                    tags.append("user")
        for m in (message_history or []):                 # their own words / prior exchanges
            if str(m).strip():
                passages.append(str(m).strip())
                tags.append("history")
        if self.search_fn is not None:                    # public footprint (may be empty for a private person)
            try:
                for p in (self.search_fn(f"{entity} {question or ''}".strip(), as_of) or [])[: self.k_web]:
                    passages.append(str(p))
                    tags.append("web")
            except Exception:
                pass
        return Dossier(entity, passages, tags)


def needs_user_context(dossier: Dossier, *, threshold: float = 0.35) -> bool:
    """True when the assembled evidence is too thin to infer sharply — the signal to ASK the user rather than
    guess. (The honest alternative to a confident inference from nothing.)"""
    return dossier.strength < threshold


def context_questions(entity: str, question: str = "") -> list:
    """The specific things to ask the user about a private individual — chosen to maximize inference quality:
    each answer is Pillar-1 evidence about a variable the outcome turns on (their disposition, the
    relationship, their state, and topic-specific stance)."""
    return [
        f"How do you know {entity}, and how would you describe your relationship (how you met, how close)?",
        f"What is {entity} like — their personality, what they seem to care about, how they typically react?",
        f"What's the recent history between you (last few exchanges, their current mood or situation)?",
        f"Anything specific about how {entity} feels about the subject of: {question or 'this'}?",
    ]


def infer_variables(dossier: Dossier, variables, extractor, *, question=None) -> dict:
    """Infer each variable acting on the entity FROM its dossier, through the full pillar stack. `extractor`
    is an `AnchoredExtractor` (or any object with `.extract(variable, question, evidence) -> {value, sd}`).
    Returns {variable: {value, sd, evidence_strength}} — a calibrated estimate per variable, wide when the
    dossier is thin (so the simulation discounts it) and sharp when it is rich."""
    out = {}
    for v in variables:
        r = extractor.extract(v, question, dossier.passages)
        if r is not None:
            out[v] = {"value": r["value"], "sd": r["sd"], "evidence_strength": round(dossier.strength, 3)}
    return out
