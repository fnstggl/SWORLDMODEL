"""Recipient conditioning from HISTORY — the product rule: ingest all the recipient data we can get.

A prediction held at the population mean is nobody's real number (the 3%-Harvard problem). To move off the
base rate we condition on the *specific* recipient — and the richest, most available signal for almost
anyone we'd model is their WRITING HISTORY. This module is the standing rule wired in:

    gather the recipient's corpus (posts, emails, comments, public writing)
      -> deep_inference builds their persona (the automated interview, leakage-safe as-of)
      -> map that persona to the scorer's recipient VariableMap + a per-recipient BASE RATE
      -> the objective is now conditioned on THIS person, not the population.

This is what de-compresses predictions: with a constant recipient the only variance is the message and the
model collapses toward the base rate; with a real per-recipient persona, different recipients get different
baselines AND different elasticities, so the model can say 0.9 for a great fit and 0.1 for a bad one.

`HistoryStore` is the durable primitive: `ingest(entity, text, ts)` any document you can get, then
`recipient(entity, now)` returns leakage-safe `(recipient_vars, base_rate)` for the objective. Degrades:
with a `deep_infer_fn` (LLM) it reads each document richly; offline it uses deep_inference's lexical
fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.variables.deep_inference import DeepInferenceEngine, DeepPersonaStore, persona_to_vars

# Map the deep-inference PERSONA traits to the message scorer's recipient variables. Each recipient var is
# a confidence-blended read of the persona (thin history stays near the population prior, deep history moves
# it). These are the variables the elasticity interactions condition on (proof×skepticism, ask×openness…).
_POP_PRIOR = {"trait_openness": 0.5, "epistemic_rigor": 0.5, "intellectual_humility": 0.5,
              "certainty_disposition": 0.5, "status_orientation": 0.5, "combativeness": 0.5,
              "moral_absolutism": 0.5}


def persona_to_recipient(persona: dict) -> tuple:
    """(recipient_vars, base_rate) from a synthesized persona. base_rate rises with how open/updatable the
    person is — a per-recipient baseline, which is what breaks the base-rate compression."""
    v = persona_to_vars(persona, prior=_POP_PRIOR)             # confidence-blended trait values
    humility = v.get("intellectual_humility", 0.5)
    rigor = v.get("epistemic_rigor", 0.5)
    certainty = v.get("certainty_disposition", 0.5)
    openness = v.get("trait_openness", 0.5)
    recipient = {
        "openness_to_outreach": round(0.5 * humility + 0.5 * openness, 4),
        "skepticism": round(0.4 + 0.6 * rigor, 4),
        "status_orientation": v.get("status_orientation", 0.5),
        "status": 0.4,
        "attention_availability": 0.6,
        "relationship_strength": 0.0,
        # kept for interplay/interaction use and audit
        "op_openness": round(humility, 4),
        "op_entrenchment": round(0.5 * certainty + 0.5 * v.get("moral_absolutism", 0.5), 4),
    }
    # per-recipient base rate: an open, humble recipient responds/updates more than an entrenched one.
    base = round(min(0.9, max(0.05, 0.25 + 0.5 * humility - 0.25 * certainty)), 4)
    return recipient, base


def recipient_from_history(corpus_texts: list, *, timestamps: list | None = None, now=None,
                           deep_infer_fn=None) -> tuple:
    """One-shot: a recipient's document corpus -> (recipient_vars, base_rate). corpus_texts as-of ordered."""
    engine = DeepInferenceEngine(deep_infer_fn=deep_infer_fn)
    persona = engine.infer_persona(corpus_texts, timestamps=timestamps, now=now)
    return persona_to_recipient(persona)


@dataclass
class HistoryStore:
    """Ingest all the recipient data you can get; serve leakage-safe, as-of recipient conditioning.
    Wraps deep_inference's per-document store so personas are computed from any prefix of the history."""
    engine: DeepInferenceEngine = field(default_factory=DeepInferenceEngine)
    _store: DeepPersonaStore = None

    def __post_init__(self):
        if self._store is None:
            self._store = DeepPersonaStore(engine=self.engine)

    def ingest(self, entity_id: str, text: str, ts=None) -> None:
        """Add one document authored by/about the recipient. Call for every document you can obtain."""
        self._store.add_doc(entity_id, ts, self.engine.per_doc(text))

    def depth(self, entity_id: str, now=None) -> int:
        return self._store.depth_asof(entity_id, now)

    def recipient(self, entity_id: str, now=None, *, max_docs: int | None = None) -> tuple:
        """Leakage-safe `(recipient_vars, base_rate)` from this entity's documents strictly before `now`."""
        persona = self._store.persona_asof(entity_id, now, max_docs=max_docs)
        return persona_to_recipient(persona)
