"""Semantic stance judge — read the real news for THIS question's outcome with an LLM (the EXP-044 fix).

EXP-044 proved the ceiling of lexical reading: entity-linked term-matching recovers only ~13% of the
signal the market extracts from the same news, because outcome polarity is question-specific ("unemployment
rises" means NO for "will unemployment be BELOW 4%") — which term counts cannot represent. Closing that
gap needs semantic reading: an LLM judging, from the article text alone, whether the news points toward
the question resolving YES or NO.

This module wires that judge for PRODUCTION and keeps it testable/leakage-safe:

- `build_stance_prompt` frames a strict reading task: given the question, its resolution criterion, and the
  as-of news, output a signed stance toward YES in [-1,1] with confidence — based ONLY on the provided
  articles, explicitly NOT on any outside knowledge of how it turned out. (The anti-contamination
  instruction; the real guard is the evaluation — see EXP-047 — which scores stance against the as-of
  PRICE the judge never sees, a quantity independent of the future outcome.)
- `SemanticStanceJudge(judge_fn)` is model-agnostic: `judge_fn(prompt) -> {stance, confidence, relevant}`.
  Swap the backend without touching callers.
    * `anthropic_judge_fn(api_key, model)` — the PRODUCTION backend (real Anthropic API call).
    * `cached_judge_fn(cache)` — replay committed judgments (reproducible experiments, and the dev path
      where the judge is run once and cached, exactly like the cmv_infer / exp037 driver caches).

The same `SemanticStanceJudge` + prompt is used in this simulator and in production; only the `judge_fn`
backend differs, so a validation here calibrates the production system rather than a throwaway.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


def build_stance_prompt(question: str, news: list, resolution_hint: str = "", max_items: int = 8) -> str:
    lines = [f"QUESTION (resolves YES or NO): {question}"]
    if resolution_hint:
        lines.append(f"RESOLUTION CRITERION: {resolution_hint}")
    lines.append("\nAS-OF NEWS (all published before the question resolves; this is your ONLY evidence):")
    for i, nw in enumerate(news[:max_items], 1):
        title = (nw.get("title") or "").strip()
        desc = (nw.get("description") or "").strip()
        when = (nw.get("published_at") or "")[:10]
        lines.append(f"  [{i}] ({when}) {title}" + (f" — {desc}" if desc else ""))
    lines.append(
        "\nTASK: Reading ONLY the articles above, judge which way they point for THIS question's YES "
        "outcome. Account for the question's specific direction — e.g. an article that a metric ROSE is "
        "evidence toward YES for 'will it be above X' but toward NO for 'will it be below X'. Do NOT use "
        "any memory of how this actually resolved; if the articles are not informative, say so with low "
        "confidence.\n"
        'Return ONLY compact JSON: {"stance": <float -1..1, + = toward YES>, "confidence": <float 0..1>, '
        '"relevant": <int, how many articles bore on the outcome>, "reason": "<=15 words"}')
    return "\n".join(lines)


def _coerce(obj) -> dict:
    """Tolerant parse of a judge response into the stance record."""
    if isinstance(obj, str):
        s = obj.strip()
        a, b = s.find("{"), s.rfind("}")
        obj = json.loads(s[a:b + 1]) if a >= 0 and b > a else {}
    stance = float(obj.get("stance", 0.0)) if obj else 0.0
    conf = float(obj.get("confidence", 0.0)) if obj else 0.0
    rel = int(obj.get("relevant", 0)) if obj else 0
    return {"stance": max(-1.0, min(1.0, stance)), "confidence": max(0.0, min(1.0, conf)),
            "relevant": rel, "reason": (obj.get("reason", "") if obj else "")[:120]}


@dataclass
class SemanticStanceJudge:
    """Turns (question, as-of news) into a signed, confidence-weighted stance toward YES via an LLM judge."""
    judge_fn: object                       # callable(prompt:str) -> dict|str  (a stance record or raw JSON)
    max_items: int = 8

    def stance(self, question: str, news: list, resolution_hint: str = "") -> dict:
        if not news:
            return {"stance": 0.0, "confidence": 0.0, "relevant": 0, "reason": "no news"}
        prompt = build_stance_prompt(question, news, resolution_hint, self.max_items)
        return _coerce(self.judge_fn(prompt))

    def feature_vector(self, question: str, news: list, resolution_hint: str = "") -> list:
        s = self.stance(question, news, resolution_hint)
        # signed stance, confidence-weighted stance, confidence, relevance density
        conf = s["confidence"]
        rel = min(1.0, s["relevant"] / 5.0)
        return [s["stance"], s["stance"] * conf, conf, rel]


FEATURE_NAMES = ["semantic_stance", "confident_stance", "stance_confidence", "relevance_density"]


# ---- backends ---------------------------------------------------------------------------------------
def cached_judge_fn(cache: dict):
    """judge_fn that replays committed judgments keyed by question id. Raises if a key is missing so a
    run can never silently fall back to a neutral stance."""
    def fn(key):
        if key not in cache:
            raise KeyError(f"no cached stance for {key!r}")
        return cache[key]
    return fn


def anthropic_judge_fn(api_key: str, model: str = "claude-sonnet-5", max_tokens: int = 200):
    """PRODUCTION backend: a real Anthropic API call per question. Pure-stdlib (urllib) so it adds no
    dependency; returns the parsed stance record. Used when an API key is configured."""
    import urllib.request

    def fn(prompt):
        body = json.dumps({
            "model": model, "max_tokens": max_tokens,
            "system": "You are a careful news-reading analyst. Judge only from the provided articles. "
                      "Never use prior knowledge of outcomes. Respond with ONLY the requested JSON.",
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return _coerce(text)
    return fn
