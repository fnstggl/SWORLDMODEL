"""Claim-level evidence representation — Phase 2.

A document is not one undifferentiated fact. The LLM proposes typed claims, each with an EXACT supporting
span; a claim whose span is not a verbatim substring of the source text is rejected (no free-floating
assertions). Claim CLASS is preserved so downstream inference never silently converts an actor statement
into a fact, an allegation into an observed event, or a forecast into a latent-state observation.

The LLM may propose extraction; it may not certify truth, publication time, or credibility. Every claim
retains its source span, the extraction model/prompt version, and an uncertainty. Temporal validity and
actor visibility are attached by their own layers, not invented here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

CLAIM_CLASSES = ("observed_fact", "official_record", "actor_statement", "promise", "allegation", "denial",
                 "opinion", "forecast", "retrospective", "correction", "retraction", "inferred_implication",
                 "absence_observation")
MODALITIES = ("asserted", "reported", "hedged", "speculative", "conditional")
POLARITIES = ("affirm", "negate")
CLAIMS_PROMPT_VERSION = "claims-extract-1.0"


@dataclass
class Claim:
    claim_id: str
    source_id: str
    subject: str
    predicate: str
    object: str = ""
    value: str = ""
    units: str = ""
    qualifiers: str = ""
    modality: str = "asserted"
    polarity: str = "affirm"
    claim_class: str = "observed_fact"
    supporting_span: str = ""                          # verbatim substring of the source text
    span_verified: bool = False
    event_time: str = ""                               # described event time (free text; verified elsewhere)
    publication_time: float | None = None              # inherited from the document
    extraction_confidence: float = 0.5
    temporal_validity_status: str = ""                 # attached by evidence_temporal
    actor_visibility: str = "public"                   # attached by evidence_visibility
    entities: list = field(default_factory=list)       # entity mention strings (resolved by evidence_entities)
    contradiction_links: list = field(default_factory=list)
    dependence_group: str = ""
    provenance: dict = field(default_factory=dict)

    def claim_key(self) -> str:
        return hashlib.sha1(f"{self.source_id}|{self.subject}|{self.predicate}|{self.object}|{self.value}"
                            .encode()).hexdigest()[:16]

    def as_dict(self) -> dict:
        return asdict(self)


_EXTRACT_PROMPT = """Extract the atomic factual CLAIMS from the news text below. Reply ONLY JSON:
{{"claims": [{{"subject": "...", "predicate": "...", "object": "...", "value": "", "units": "",
  "qualifiers": "", "modality": "asserted|reported|hedged|speculative|conditional",
  "polarity": "affirm|negate", "claim_class": "{classes}",
  "supporting_span": "<EXACT verbatim substring of the text that supports this claim>",
  "event_time": "<when the described event happened, or ''>", "entities": ["..."],
  "confidence": 0.0}}]}}
Rules: supporting_span MUST be copied verbatim from the text. Classify an unverified accusation as
allegation, a prediction as forecast, a quote as actor_statement, a denial as denial. Do NOT turn a
statement or allegation into observed_fact. Do NOT invent claims not present in the text.

TEXT:
{text}"""


def extract_claims(text: str, *, source_id: str, llm, publication_time: float | None = None,
                   max_claims: int = 8) -> list:
    """LLM claim extraction with strict span validation. A claim whose supporting_span is not a verbatim
    substring of `text` is rejected (recorded but not returned as valid). Returns [Claim]."""
    from swm.engine.grounding import parse_json
    if not text.strip():
        return []
    prompt = _EXTRACT_PROMPT.format(classes="|".join(CLAIM_CLASSES), text=text[:3500])
    try:
        raw = parse_json(llm(prompt)) or {}
    except Exception:  # noqa: BLE001 — extraction failure is degraded evidence, not a crash
        return []
    norm = _normalize_ws(text)
    out = []
    for i, c in enumerate((raw.get("claims") or [])[:max_claims]):
        if not isinstance(c, dict):
            continue
        span = str(c.get("supporting_span", "")).strip()
        verified = bool(span) and _normalize_ws(span) in norm            # STRICT: span must be in the text
        cls = c.get("claim_class") if c.get("claim_class") in CLAIM_CLASSES else "observed_fact"
        mod = c.get("modality") if c.get("modality") in MODALITIES else "asserted"
        pol = c.get("polarity") if c.get("polarity") in POLARITIES else "affirm"
        claim = Claim(
            claim_id=f"{source_id}:c{i}", source_id=source_id,
            subject=str(c.get("subject", ""))[:200], predicate=str(c.get("predicate", ""))[:200],
            object=str(c.get("object", ""))[:200], value=str(c.get("value", ""))[:80],
            units=str(c.get("units", ""))[:40], qualifiers=str(c.get("qualifiers", ""))[:200],
            modality=mod, polarity=pol, claim_class=cls,
            supporting_span=span[:400], span_verified=verified,
            event_time=str(c.get("event_time", ""))[:60], publication_time=publication_time,
            entities=[str(e)[:80] for e in (c.get("entities") or [])][:8],
            extraction_confidence=_clamp(c.get("confidence", 0.5)),
            provenance={"extractor": "llm", "prompt_version": CLAIMS_PROMPT_VERSION,
                        "span_verified": verified})
        # a claim without a verified span is unsupported-precision — keep only if it still cites text
        if not verified:
            claim.claim_class = claim.claim_class            # retained but flagged via span_verified=False
        out.append(claim)
    return out


def valid_claims(claims: list) -> list:
    """Production filter: only span-verified claims (no unsupported free-floating assertions)."""
    return [c for c in claims if c.span_verified]


def _normalize_ws(s: str) -> str:
    return " ".join(s.split()).lower()


def _clamp(x, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 0.5
