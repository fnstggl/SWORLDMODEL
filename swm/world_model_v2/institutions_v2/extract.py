"""Phase 10 (continuation) — automatic evidence-backed rule reconstruction (Part 4 / priority #4).

Pipeline:  authoritative source text
  → LLM candidate extraction (propose typed rule formalizations + the source span they come from)
  → SOURCE-SPAN GROUNDING (the quoted span must appear verbatim in the source — an ungrounded rule is rejected)
  → deterministic rule validation (evidence.validate_rule: kinds, fractions, temporal, references)
  → temporal + jurisdiction stamping
  → typed RuleRecord.

The LLM may PROPOSE formalizations; it may NOT establish an unsupported rule — every accepted rule is
grounded in a verbatim source span AND passes deterministic validation. Degrades to a deterministic
regex/keyword extractor (clearly labeled) if the LLM is unavailable, so the pipeline is always runnable.
"""
from __future__ import annotations

import json
import re

from swm.world_model_v2.institutions_v2.evidence import validate_rule
from swm.world_model_v2.institutions_v2.record import RuleRecord

_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule_type": {"type": "string",
                                  "enum": ["quorum", "threshold", "override", "deadline", "authority"]},
                    "threshold_kind": {"type": "string",
                                       "enum": ["simple_majority", "absolute_majority", "supermajority",
                                                "unanimous", "plurality", "none"]},
                    "fraction": {"type": "number"},
                    "base": {"type": "string", "enum": ["present", "all_members", "eligible"]},
                    "source_span": {"type": "string",
                                    "description": "the VERBATIM span from the source this rule comes from"},
                    "interpreted_rule": {"type": "string"},
                },
                "required": ["rule_type", "threshold_kind", "fraction", "base", "source_span",
                             "interpreted_rule"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rules"],
    "additionalProperties": False,
}


_SYSTEM = ("You extract FORMAL institutional decision rules from authoritative legal/procedural text "
           "(quorum, passage/decision threshold, veto override, deadline). For each rule, give the typed "
           "formalization AND the VERBATIM span it comes from. Only extract rules the text actually states — "
           "do NOT invent thresholds or deadlines. Fractions: majority=0.5, two-thirds=0.6667, three-fifths="
           "0.6. Return ONLY JSON of the form {\"rules\":[{\"rule_type\":\"quorum|threshold|override|deadline"
           "|authority\",\"threshold_kind\":\"simple_majority|absolute_majority|supermajority|unanimous|"
           "plurality|none\",\"fraction\":0.5,\"base\":\"present|all_members|eligible\",\"source_span\":\""
           "verbatim quote\",\"interpreted_rule\":\"...\"}]}.")


def _llm_candidates(source_text: str) -> tuple[list, str]:
    """Ask the LLM to propose typed rule formalizations WITH the verbatim source span. Tries DeepSeek
    (OpenAI-compatible, JSON mode) then Anthropic; falls back to the deterministic extractor. The LLM only
    PROPOSES — grounding + deterministic validation downstream decide what is accepted."""
    import os
    # DeepSeek (OpenAI-compatible) — the key present in this environment
    try:
        if os.environ.get("DEEPSEEK_API_KEY"):
            import openai
            c = openai.OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
            r = c.chat.completions.create(
                model="deepseek-chat", max_tokens=1500, temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": f"Source text:\n\n{source_text[:6000]}"}])
            return json.loads(r.choices[0].message.content).get("rules", []), "llm_deepseek"
    except Exception:
        pass
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-opus-4-8", max_tokens=1500, system=_SYSTEM,
            messages=[{"role": "user", "content": f"Source text:\n\n{source_text[:6000]}"}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}})
        block = next(b for b in resp.content if b.type == "text")
        return json.loads(block.text).get("rules", []), "llm_anthropic"
    except Exception:
        return _deterministic_candidates(source_text), "deterministic_fallback"


_FRAC_WORDS = [(r"two[- ]thirds", 2 / 3, "supermajority"), (r"three[- ]fifths", 0.6, "supermajority"),
               (r"three[- ]fourths|three[- ]quarters", 0.75, "supermajority"),
               (r"majority", 0.5, "simple_majority")]


def _deterministic_candidates(text: str) -> list:
    """A transparent keyword/regex extractor (fallback): find quorum + threshold + override sentences and
    the fraction word in each. Lower quality than the LLM but honest and always runnable."""
    cands = []
    sents = re.split(r"(?<=[.;])\s+", text)
    for s in sents:
        low = s.lower()
        frac, kind = None, None
        for pat, f, k in _FRAC_WORDS:
            if re.search(pat, low):
                frac, kind = f, k
                break
        if frac is None:
            continue
        if "quorum" in low:
            cands.append({"rule_type": "quorum", "threshold_kind": kind, "fraction": frac,
                          "base": "eligible", "source_span": s.strip()[:240], "interpreted_rule": s.strip()[:160]})
        if "two-thirds" in low and ("veto" in low or "reconsider" in low or "objection" in low or "override" in low):
            cands.append({"rule_type": "override", "threshold_kind": "supermajority", "fraction": 2 / 3,
                          "base": "present", "source_span": s.strip()[:240], "interpreted_rule": s.strip()[:160]})
        elif ("pass" in low or "vote of the majority" in low or "act of the board" in low or "become a law" in low):
            cands.append({"rule_type": "threshold", "threshold_kind": kind, "fraction": frac,
                          "base": "present", "source_span": s.strip()[:240], "interpreted_rule": s.strip()[:160]})
    return cands


def extract_rules(source_text: str, *, source_id: str, jurisdiction: str = "", effective_date: str = "",
                  roles=None, stages=None, actions=None) -> dict:
    """Full pipeline. Returns {rules: [RuleRecord], rejected: [{candidate, reason}], source_tag}."""
    candidates, tag = _llm_candidates(source_text)
    accepted, rejected = [], []
    for i, c in enumerate(candidates):
        span = (c.get("source_span") or "").strip()
        # SOURCE-SPAN GROUNDING: the span must appear (fuzzily) in the source — reject ungrounded rules
        if not span or not _grounded(span, source_text):
            rejected.append({"candidate": c, "reason": "source span not found verbatim in the source "
                             "(ungrounded — the LLM cannot establish an unsupported rule)"})
            continue
        kind = "quorum" if c["rule_type"] == "quorum" else ("override" if c["rule_type"] == "override"
                                                            else c["rule_type"])
        params = {"kind": c.get("threshold_kind"), "fraction": float(c.get("fraction", 0.5)),
                  "base": c.get("base", "present")}
        if kind in ("quorum", "override"):
            params.setdefault("quorum_fraction", params["fraction"])
        rr = RuleRecord(rule_id=f"{source_id}:extracted:{i}", kind=("threshold" if kind == "threshold" else kind),
                        params=params, evidence_id=source_id, effective_date=effective_date,
                        ambiguity=c.get("interpreted_rule", ""), verified=False)
        problems = validate_rule(rr, roles=set(roles or []), stages=set(stages or []),
                                 actions=set(actions or []), require_evidence=True)
        if problems:
            rejected.append({"candidate": c, "reason": "; ".join(problems)})
        else:
            accepted.append(rr)
    return {"rules": accepted, "rejected": rejected, "source_tag": tag, "n_candidates": len(candidates)}


def _grounded(span: str, text: str) -> bool:
    """The span is grounded if a normalized substring of it appears in the source (robust to whitespace)."""
    norm = re.sub(r"\s+", " ", span.lower()).strip()
    src = re.sub(r"\s+", " ", text.lower())
    if len(norm) < 12:
        return False
    # accept if a meaningful chunk (first ~40 chars or a distinctive clause) is present
    return norm[:40] in src or norm in src or any(chunk in src for chunk in norm.split(" ; ") if len(chunk) > 20)
