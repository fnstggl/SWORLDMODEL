"""Public-figure resolution — infer an entity we've NEVER messaged from what's observable online.

The old failure mode was a hard block: "individual prediction needs private thread history + a fitted
world, so we can't score a stranger." That is the wrong default. The thesis is *bias to inferring the
variables when we're not told them* (VariableInferenceEngine) — and for a PUBLIC FIGURE the variables
are unusually inferable, because their disposition, status, and even their responsiveness leave a public
trail: interviews, essays, what they publicly fund/ignore, stories of who got a reply and who didn't.

So this resolver does the automated version of "look them up before you email them":

  1. SEARCH   — query a pluggable web backend for evidence about the figure's communication behavior,
                accessibility, stated preferences, and disposition (`search_fn`, same contract as
                swm/api/retrieval.web_search_retriever).
  2. INFER    — turn that evidence into behavioral variables (base_responsiveness, openness_to_outreach,
                status, skepticism, ...), each with a confidence and a one-line cited evidence string
                (`infer_fn`, an LLM/agent-swarm; a precomputed dict can be passed instead).
  3. FALL BACK— with no backend, a transparent lexical scorer reads whatever snippets are supplied (or
                the population prior for a high-status busy person), so the pipeline still runs offline —
                clearly at LOW confidence, never presented as fact.

Everything it emits carries provenance `web` (schema rank: below your own logs, above a bare llm prior).
The output plugs straight into `VariableInferenceEngine.infer(web_inference=...)` and into a `Persona`
as evidence-weighted pseudo-observations. It infers variables — never the outcome.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from swm.api.retrieval import RetrievedContext, web_search_retriever

# ---- lexical evidence lexicon (offline fallback + query building) --------------------------------
# Each pattern nudges one variable up (+) or down (-); weight scales the nudge and the confidence.
_RESPONSIVE_POS = re.compile(
    r"\b(responds?|replies|replied|answers?|accessible|reachable|gets back|known for replying|"
    r"reads every|reply guy|hands[- ]on|mentors?|backs young|emails? back|open to cold)\b", re.I)
_RESPONSIVE_NEG = re.compile(
    r"\b(declined|declines|ignores?|unreachable|reclusive|no comment|did not respond|does not respond|"
    r"rarely responds?|never replies|hard to reach|screens|gatekept|inbox zero is a myth)\b", re.I)
_OPENNESS_POS = re.compile(
    r"\b(fellowship|drop ?out|young founders?|first check|angel|cold email|took a meeting|"
    r"discovered|plucked|backed a teenager|responds to founders|scout)\b", re.I)
_CONTRARIAN = re.compile(
    r"\b(contrarian|provocative|heterodox|iconoclast|against the grain|unconventional|taboo|"
    r"skeptic|skeptical|challenges? consensus)\b", re.I)
_STATUS_HI = re.compile(
    r"\b(billionaire|founder|ceo|partner|investor|chairman|professor|senator|director|"
    r"famous|renowned|influential|prominent)\b", re.I)
_CREDENTIAL_AVERSE = re.compile(
    r"\b(credential|elite university|ivy|prestige|status game|higher ed bubble|degree is)\b", re.I)


@dataclass
class PublicFigureProfile:
    """The resolved, auditable view of a public figure — inferred, never asserted as fact."""
    name: str
    web_variables: dict = field(default_factory=dict)   # {var: {value, confidence, evidence}} for _apply_web
    responsiveness: dict = field(default_factory=dict)   # {mean, confidence, evidence}
    evidence: RetrievedContext | None = None
    source: str = "prior"                                # "web+llm" | "web+lexical" | "prior"

    def summary(self) -> dict:
        return {
            "name": self.name, "source": self.source,
            "n_evidence": len(self.evidence) if self.evidence else 0,
            "responsiveness": self.responsiveness,
            "inferred_variables": {k: {"value": round(v["value"], 2),
                                       "confidence": round(v.get("confidence", 0.0), 2),
                                       "evidence": v.get("evidence", "")}
                                   for k, v in self.web_variables.items()},
        }


@dataclass
class PublicFigureResolver:
    """Resolve a public figure to inferred variables by searching online and reasoning over evidence.

    - `search_fn(query) -> [{title, snippet, url?, date?}]` : the web backend (optional; offline => prior).
    - `infer_fn(name, evidence_text, domain, ask) -> {var: {value, confidence, evidence}}` : LLM/agent
      inference over the gathered evidence (optional; absent => transparent lexical fallback).
    """
    search_fn: object = None
    infer_fn: object = None
    max_results: int = 12

    # ---- queries we run to characterize an unknown recipient --------------------------------------
    def _queries(self, name: str, domain: str, ask: str) -> list[str]:
        base = [
            f"{name} responds to cold emails",
            f"{name} how to contact reach out",
            f"{name} communication style interviews",
            f"{name} what he looks for / values",
        ]
        if domain:
            base.append(f"{name} {domain}")
        if ask:
            base.append(f"{name} {ask}")
        return base

    def _gather(self, name: str, domain: str, ask: str) -> RetrievedContext:
        if self.search_fn is None:
            return RetrievedContext(question=name, snippets=[])
        retriever = web_search_retriever(self.search_fn, results=self.max_results)
        snippets: list[dict] = []
        seen = set()
        for q in self._queries(name, domain, ask):
            for s in retriever.retrieve(q).snippets:
                key = (s.get("title", ""), s.get("snippet", s.get("description", "")))
                if key not in seen:
                    seen.add(key)
                    snippets.append(s)
        return RetrievedContext(question=name, snippets=snippets[: self.max_results])

    def resolve(self, name: str, *, domain: str = "", ask: str = "",
                channel: str = "email") -> PublicFigureProfile:
        ctx = self._gather(name, domain, ask)
        text = ctx.to_prompt(max_items=self.max_results)

        if self.infer_fn is not None:
            try:
                web_vars = self.infer_fn(name, text, domain, ask) or {}
                source = "web+llm"
            except Exception:
                web_vars, source = self._lexical_infer(name, text), "web+lexical"
        else:
            web_vars = self._lexical_infer(name, text)
            source = "web+lexical" if text else "prior"

        resp = web_vars.pop("base_responsiveness", None)
        if isinstance(resp, dict):
            responsiveness = {"mean": resp.get("mean", resp.get("value", 0.28)),
                              "confidence": resp.get("confidence", 0.15),
                              "evidence": resp.get("evidence", "web evidence")}
        else:
            responsiveness = {"mean": 0.28, "confidence": 0.15,
                              "evidence": "high-status busy recipient, population prior"}
        # keep base_responsiveness in the variable map too (the readout reads it); _apply_web wants a
        # `value` key, while the persona-folding path wants `mean` — carry both.
        web_vars["base_responsiveness"] = {"value": responsiveness["mean"],
                                           "confidence": responsiveness["confidence"],
                                           "evidence": responsiveness["evidence"]}
        return PublicFigureProfile(name=name, web_variables=web_vars, responsiveness=responsiveness,
                                   evidence=ctx, source=source)

    # ---- transparent offline fallback -------------------------------------------------------------
    def _lexical_infer(self, name: str, text: str) -> dict:
        """Keyword-grounded variable estimates from whatever snippets we have. Deterministic and
        auditable: every value cites the signal it came from. Confidence stays LOW — this is a
        lexical proxy for an LLM read, not a claim of fact."""
        out: dict = {}

        def _count(pat):
            return len(pat.findall(text)) if text else 0

        pos, neg = _count(_RESPONSIVE_POS), _count(_RESPONSIVE_NEG)
        if pos or neg:
            total = pos + neg
            mean = max(0.03, min(0.85, 0.28 + 0.12 * pos - 0.16 * neg))
            conf = min(0.6, 0.2 + 0.08 * total)
            out["base_responsiveness"] = {
                "mean": mean, "confidence": conf,
                "evidence": f"{pos} positive / {neg} negative responsiveness signals in web evidence"}

        openness = _count(_OPENNESS_POS)
        if openness:
            out["openness_to_outreach"] = {
                "value": min(0.9, 0.5 + 0.12 * openness), "confidence": min(0.6, 0.25 + 0.08 * openness),
                "evidence": f"{openness} signals of backing young/unsolicited founders"}

        contra = _count(_CONTRARIAN)
        if contra:
            out["skepticism"] = {"value": min(0.9, 0.55 + 0.1 * contra), "confidence": min(0.55, 0.25 + 0.07 * contra),
                                 "evidence": f"{contra} contrarian/skeptic signals"}
            out["certainty_disposition"] = {"value": min(0.85, 0.55 + 0.08 * contra), "confidence": 0.35,
                                             "evidence": "asserts strong/definite positions publicly"}

        status = _count(_STATUS_HI)
        # a named public figure is high-status by default even with thin evidence (bias to infer)
        out["status"] = {"value": min(0.95, 0.7 + 0.05 * status), "confidence": 0.35 if status else 0.2,
                         "evidence": f"{status} status markers" if status else "named public figure"}

        cred = _count(_CREDENTIAL_AVERSE)
        if cred:
            out["status_orientation"] = {"value": min(0.85, 0.55 + 0.1 * cred), "confidence": 0.3,
                                         "evidence": f"{cred} signals of skepticism toward credential/prestige signaling"}
        return out


def default_resolver(search_fn=None, infer_fn=None) -> PublicFigureResolver:
    """Factory: a resolver that uses a web backend + LLM if given, else the offline lexical fallback."""
    return PublicFigureResolver(search_fn=search_fn, infer_fn=infer_fn)
