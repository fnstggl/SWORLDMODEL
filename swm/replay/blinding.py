"""Causal blinding — transform a historical event into a pseudonymized causal twin (Approach B).

A current LLM cannot be proven to forget a known outcome, so the blinded arm removes the trivially
identifying surface (real names, places, calendar anchors) while PRESERVING causal structure: roles,
institutional relationships, prior actions, trajectories, thresholds. The pseudonym MAPPING is written only
into the SEALED store; the forecaster works entirely on opaque names.

Blinding is two-stage:
  1. entity extraction (LLM, once per event): salient proper nouns + their roles;
  2. deterministic substitution: stable pseudonyms (Person A/B…, Organization K/L…, Federation X-7…) applied
     to the question AND to every evidence text field, whole-word, longest-first (so "Donald Trump" maps
     before "Trump").

Blinding is never assumed secure: every blinded packet gets an event-recognition probe (probes.py), and an
identified row is classified contamination_susceptible regardless of how good the blinding looked.
"""
from __future__ import annotations

import json
import re

_EXTRACT_PROMPT = """Extract the proper nouns from this forecasting question that would identify the real-world
event: people, organizations, institutions, places, branded products, named agreements. For each, give its
causal ROLE in one or two words (e.g. "candidate", "central bank", "country", "company", "product").
QUESTION: {q}
Return ONLY JSON: {{"entities": [{{"name": "...", "role": "..."}}, ...]}}"""

_KIND_PREFIX = {
    "person": "Person", "candidate": "Candidate", "leader": "Leader", "politician": "Person",
    "company": "Company", "organization": "Organization", "institution": "Institution",
    "central bank": "Central Bank", "country": "Federation", "state": "Region", "city": "City",
    "place": "Region", "product": "Product", "party": "Party", "agreement": "Accord",
    "court": "Tribunal", "legislature": "Assembly", "agency": "Agency", "team": "Team",
    "competition": "Tournament", "currency": "Asset", "asset": "Asset", "model": "Product",
    "election": "Election", "event": "Event", "war": "Conflict", "conflict": "Conflict",
    "meeting": "Session", "index": "Index",
}
_SUFFIX = ("A", "B", "C", "D", "E", "F", "G", "H", "J", "K")


def _prefix_for(role: str) -> str:
    r = str(role or "").lower()
    for k, v in _KIND_PREFIX.items():
        if k in r:
            return v
    return "Entity"


def build_mapping(question: str, llm, *, extra_names=()) -> dict:
    """{real_name: pseudonym}. Deterministic given the extraction (stable suffix order by first appearance)."""
    from swm.engine.grounding import parse_json
    raw = parse_json(llm(_EXTRACT_PROMPT.format(q=question))) or {}
    ents = [e for e in (raw.get("entities") or []) if isinstance(e, dict) and e.get("name")]
    for n in extra_names:
        if n and not any(e["name"] == n for e in ents):
            ents.append({"name": n, "role": "entity"})
    mapping, counters = {}, {}
    for e in ents:
        name = str(e["name"]).strip()
        if not name or name in mapping or len(name) < 3:
            continue
        prefix = _prefix_for(e.get("role"))
        idx = counters.get(prefix, 0)
        counters[prefix] = idx + 1
        mapping[name] = f"{prefix} {_SUFFIX[idx % len(_SUFFIX)]}"
        # surname/last-token alias (whole-word): "Trump" after "Donald Trump"
        toks = name.split()
        if len(toks) > 1 and len(toks[-1]) >= 4 and toks[-1] not in mapping:
            mapping[toks[-1]] = mapping[name]
    return mapping


def apply_mapping(text: str, mapping: dict) -> str:
    """Whole-word, longest-name-first substitution (case-preserving on exact match)."""
    out = str(text or "")
    for name in sorted(mapping, key=len, reverse=True):
        out = re.sub(rf"(?<![\w]){re.escape(name)}(?![\w])", mapping[name], out)
    return out


def blind_question(question: str, mapping: dict) -> str:
    q = apply_mapping(question, mapping)
    # strip trivially identifying calendar years (keep durations/relative time intact)
    return re.sub(r"\b(19|20)\d{2}\b", "a specified year", q)


def blind_bundle(bundle, mapping: dict):
    """Pseudonymize every LLM-visible text field of a frozen EvidenceBundleV2 IN PLACE (the bundle is a
    per-run copy). Timestamps/hashes/ids are untouched — the temporal audit trail survives blinding."""
    for c in (bundle.claims or []):
        for k in ("text", "title", "claim_text", "quote", "about", "summary"):
            if isinstance(c.get(k), str):
                c[k] = apply_mapping(c[k], mapping)
    for d in (bundle.documents or []):
        for k in ("title", "text", "snippet", "summary"):
            if isinstance(d.get(k), str):
                d[k] = apply_mapping(d[k], mapping)
    bundle.question = blind_question(bundle.question, mapping)
    return bundle


def serialize_mapping(mapping: dict) -> str:
    return json.dumps(mapping, sort_keys=True)
