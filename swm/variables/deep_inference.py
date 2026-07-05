"""Deep per-person inference — our scalable analog of the 2-hour interview.

The measured driver of individual-simulation accuracy in the SOTA (Generative Agent Simulations of
1,000 People, Park et al.) is RICH PER-PERSON DATA: they interview each person for two hours and
condition an LLM agent on the transcript. We can't interview everyone — but almost everyone we want to
model has left a WRITING HISTORY. This engine is the automated interview: it reads a person's as-of
corpus and infers a deep, structured PERSONA — the stable traits across personality, epistemic style,
communication, values, and domain footprint (the PERSONA category of the schema) — the "everything we
would model about the person."

It is genuinely MULTI-PASS, and depth-honest:

  Pass A (per-document extraction)  : each document the person authored is read and scored on the
                                      persona facets it reveals, with a per-facet SALIENCE (how much
                                      this document informs that trait). Done once per document.
  Pass B (cross-document synthesis) : for any as-of PREFIX of the corpus, each trait's value is the
                                      salience-weighted mean of its per-document signals, and its
                                      CONFIDENCE grows with corpus depth (more documents => more sure).
  Pass C (consistency reflection)   : traits whose per-document evidence disagrees are down-weighted
                                      (low consistency => low confidence); stable, repeatedly-evidenced
                                      traits are trusted. This is why "the deeper and more inferences,
                                      the better" — depth AND agreement both raise a trait's weight, and
                                      the confidence-weighted readout then lets high-confidence persona
                                      traits move the prediction while thin/noisy ones stay near neutral.

Because it aggregates prefixes, an as-of persona exists at every point in the person's timeline with no
leakage: predicting the k-th action uses only documents strictly before it.

The per-document signals come from an LLM (an agent swarm, precomputed and passed in) or, for tests and
offline use, a transparent lexical fallback. The engine itself owns the depth/consistency mathematics —
that is the part that must earn its place on held-out backtests.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from swm.variables.schema import BY_CATEGORY, PERSONA, spec

PERSONA_VARS = list(BY_CATEGORY.get(PERSONA, []))
_DEPTH_TAU = 5.0          # documents; confidence saturation scale (n=5 -> ~0.63, n=8 -> ~0.80)
_MAX_PERSONA_CONF = 0.9   # a corpus-inferred trait never claims certainty


def depth_factor(n: int, tau: float = _DEPTH_TAU) -> float:
    """Confidence multiplier from corpus depth: 0 docs -> 0, saturating toward 1. Monotone in n."""
    return 1.0 - math.exp(-max(0, n) / tau) if n > 0 else 0.0


@dataclass
class DeepInferenceEngine:
    """Aggregates per-document persona signals into an as-of DeepPersona for any prefix of a corpus."""
    deep_infer_fn: object = None            # optional callable(doc_text) -> per-doc signal dict
    tau: float = _DEPTH_TAU

    def per_doc(self, text: str) -> dict:
        """Pass A for one document. Returns {trait: {"value":0..1|-1..1, "salience":0..1}}.
        Uses the injected LLM fn if present, else the lexical fallback."""
        if self.deep_infer_fn is not None:
            try:
                return self.deep_infer_fn(text) or {}
            except Exception:
                return {}
        return _lexical_persona(text)

    def synthesize(self, doc_signals: list[dict]) -> dict:
        """Passes B+C. doc_signals: list (as-of order) of per-doc signal dicts. Returns a persona dict
        {trait: {"value":.., "confidence":.., "evidence":..}} routable through VariableMap llm_inference."""
        persona = {}
        n = len(doc_signals)
        if n == 0:
            return persona
        for trait in PERSONA_VARS:
            vals, sals = [], []
            for sig in doc_signals:
                if trait in sig:
                    payload = sig[trait]
                    if isinstance(payload, dict):
                        v = payload.get("value"); s = float(payload.get("salience", 0.5))
                    else:
                        v = payload; s = 0.5
                    if v is not None:
                        vals.append(float(v)); sals.append(max(1e-3, s))
            if not vals:
                continue
            sw = sum(sals)
            mean = sum(v * s for v, s in zip(vals, sals)) / sw
            # Pass C: consistency = 1 - normalized spread of the (salient) evidence
            var = sum(s * (v - mean) ** 2 for v, s in zip(vals, sals)) / sw
            signed = spec(trait).signed
            spread_scale = 1.0 if signed else 0.5        # max meaningful sd on the trait's range
            consistency = max(0.0, 1.0 - math.sqrt(var) / spread_scale)
            eff_n = len(vals)                            # documents that actually spoke to this trait
            conf = _MAX_PERSONA_CONF * depth_factor(eff_n, self.tau) * (0.4 + 0.6 * consistency)
            persona[trait] = {"value": round(mean, 4), "confidence": round(conf, 4),
                              "evidence": f"{eff_n} docs, consistency {consistency:.2f}"}
        return persona

    def infer_persona(self, corpus_texts: list[str]) -> dict:
        """Convenience: run Pass A over each text then synthesize. corpus_texts must be as-of ordered."""
        return self.synthesize([self.per_doc(t) for t in corpus_texts])


@dataclass
class DeepPersonaStore:
    """Precomputed per-document signals per entity (e.g. from an agent swarm). Serves as-of personas."""
    engine: DeepInferenceEngine = field(default_factory=DeepInferenceEngine)
    _docs: dict = field(default_factory=dict)          # entity -> [(ts, signal_dict), ...]

    def add_doc(self, entity_id: str, ts, signal: dict) -> None:
        self._docs.setdefault(entity_id, []).append((ts, signal))

    def persona_asof(self, entity_id: str, now, *, max_docs: int | None = None) -> dict:
        """Synthesize the persona from this entity's documents strictly before `now` (no leakage)."""
        docs = sorted(self._docs.get(entity_id, []), key=lambda d: (d[0] is None, d[0]))
        prior = [s for ts, s in docs if now is None or ts is None or ts < now]
        if max_docs is not None:
            prior = prior[-max_docs:]
        return self.engine.synthesize(prior)

    def depth_asof(self, entity_id: str, now) -> int:
        docs = self._docs.get(entity_id, [])
        return sum(1 for ts, _ in docs if now is None or ts is None or ts < now)


# ---- transparent lexical fallback (no API key): coarse per-document persona signals ----------------
_EVID = re.compile(r"\b(evidence|study|data|source|research|statistic|cite|according to|"
                   r"for example|e\.g\.|because|therefore|study shows)\b", re.I)
_HEDGE = re.compile(r"\b(i think|maybe|perhaps|might|could be|i could be wrong|arguably|"
                    r"in my view|it seems|possibly|i'm not sure)\b", re.I)
_ABSOL = re.compile(r"\b(always|never|everyone|no one|obviously|clearly|definitely|"
                    r"without a doubt|the fact is|undeniably)\b", re.I)
_HOSTILE = re.compile(r"\b(stupid|idiot|nonsense|ridiculous|wrong|dumb|absurd|garbage)\b", re.I)
_POLITE = re.compile(r"\b(thanks|thank you|fair point|good point|i appreciate|i see your|"
                     r"you're right|respect|understand your)\b", re.I)
_EMOTE = re.compile(r"[!?]{1,}|\b(love|hate|angry|excited|afraid|frustrated|happy|sad)\b", re.I)
_HUMOR = re.compile(r"\b(lol|haha|;\)|:\)|joking|kidding|ironic)\b", re.I)
_CONCEDE = re.compile(r"\b(you'?re right|good point|fair enough|i concede|i was wrong|"
                      r"changed my mind|that's true|i'll grant)\b", re.I)


def _sal(hits, cap=3):
    return min(1.0, hits / cap)


def _lexical_persona(text: str) -> dict:
    """Coarse per-document persona signals from surface cues — a stand-in for the LLM per-doc pass."""
    t = text or ""
    words = t.split()
    n = max(1, len(words))
    evid = len(_EVID.findall(t)); hedge = len(_HEDGE.findall(t)); absol = len(_ABSOL.findall(t))
    hostile = len(_HOSTILE.findall(t)); polite = len(_POLITE.findall(t))
    emote = len(_EMOTE.findall(t)); humor = len(_HUMOR.findall(t)); concede = len(_CONCEDE.findall(t))
    sig = {
        "epistemic_rigor": {"value": min(1.0, evid / 3.0), "salience": _sal(evid)},
        "intellectual_humility": {"value": min(1.0, (hedge + concede) / 3.0), "salience": _sal(hedge + concede)},
        "certainty_disposition": {"value": min(1.0, absol / 2.0), "salience": _sal(absol)},
        "politeness_disposition": {"value": max(0.0, min(1.0, 0.5 + (polite - hostile) / 4.0)),
                                   "salience": _sal(polite + hostile)},
        "combativeness": {"value": min(1.0, hostile / 2.0), "salience": _sal(hostile + 1)},
        "empathy_display": {"value": min(1.0, polite / 3.0), "salience": _sal(polite)},
        "emotional_expressiveness": {"value": min(1.0, emote / 4.0), "salience": _sal(emote)},
        "humor_disposition": {"value": min(1.0, humor / 2.0), "salience": _sal(humor)},
        "verbosity": {"value": min(1.0, math.log1p(n) / math.log(400)), "salience": 0.7},
        "analytical_style": {"value": min(1.0, (evid + (1 if ";" in t or "however" in t.lower() else 0)) / 3.0),
                             "salience": _sal(evid + 1)},
        "trait_agreeableness": {"value": max(0.0, min(1.0, 0.5 + (polite + concede - hostile) / 5.0)),
                                "salience": _sal(polite + hostile + concede)},
        "trait_conscientiousness": {"value": min(1.0, (evid + (n > 80)) / 3.0), "salience": 0.4},
    }
    return sig
