"""The HUMAN-LANGUAGE judge — one of three separated judges (truth / language / outcome).

Job: would a sharp, busy person actually TYPE each sentence to another busy person? It checks
naturalness, effortlessness, clarity, and non-annoyingness of the language itself — nothing about
whether the recipient would reply (the outcome judge's job) and nothing about factual support (the
truth judge's job). Splitting the jobs matters because one blended judge lets "sounds impressive to
an imagined recipient" leak into register decisions; this judge reads the text as WRITING, not as a
pitch.

What it is harsh about (rubric, not a phrase blacklist):
  * formal-bot register — grammatical-stiff constructions no real person types in a quick note
    ("May I send you the one-page technical memo?" vs "Want the one-pager?");
  * jargon compounds that make the reader do the work ("constraint-aware orchestration",
    "SLA-safe goodput per dollar" unglossed);
  * more than one big number competing for attention;
  * performed politeness / assistant-speak / marketing cadence;
  * any sentence a busy human peer would not plausibly have typed.

PREFERENCE LEARNING HOOK: the caller can record real human A-vs-B choices
(`record_preference(chosen, rejected)`); the judge folds the most recent stored pairs into its
prompt as calibration examples, so the automated critic gradually inherits the human editor's
taste. This is the seam that eventually replaces LLM opinion with the user's demonstrated
judgment — until enough pairs exist, outputs stay labeled as uncalibrated judgment.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from swm.decision.llm_moves import _call

DEFAULT_PREFS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                                  "language_preferences.jsonl")


def record_preference(chosen: str, rejected: str, *, path: str = DEFAULT_PREFS_PATH,
                      note: str = ""):
    """Append one real human A-vs-B choice (the calibration asset)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"chosen": chosen, "rejected": rejected, "note": note}) + "\n")


def load_preferences(path: str = DEFAULT_PREFS_PATH, k: int = 4) -> list:
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[-k:]


@dataclass
class LanguageVerdict:
    ok: bool                                   # passes the human-language gate
    score: float                               # 0..1 overall "a human typed this"
    flags: list = field(default_factory=list)  # [{"sentence":..., "problem":...}]
    source: str = "llm"

    def as_dict(self):
        return {"ok": self.ok, "score": round(self.score, 3), "flags": self.flags,
                "source": self.source,
                "label": "uncalibrated language judgment (rubric + stored human preferences)"}


def llm_language_judge(chat_fn, *, prefs_path: str = DEFAULT_PREFS_PATH, gate: float = 0.6):
    """Build judge(text) -> LanguageVerdict. Offline (chat_fn None): the lexical critic's
    naturalness axis serves as a deterministic fallback."""
    def judge(text: str) -> LanguageVerdict:
        if chat_fn is None:
            from swm.decision.semantic_critic import SemanticCritic
            c = SemanticCritic().critique(text)
            flags = [{"sentence": f["sentence"][:90], "problem": f["issue"]} for f in c.flags()]
            score = min(c.naturalness, c.coherence)
            return LanguageVerdict(ok=score >= gate, score=score, flags=flags, source="lexical")
        prefs = load_preferences(prefs_path)
        pref_block = ""
        if prefs:
            pref_block = ("\nThe human editor whose taste you must match previously chose these "
                          "(CHOSEN over REJECTED):\n" +
                          "\n".join(f"- CHOSEN: \"{p['chosen'][:140]}\"\n  REJECTED: "
                                    f"\"{p['rejected'][:140]}\"" for p in prefs) + "\n")
        prompt = (
            "You judge LANGUAGE ONLY: would a sharp, busy person actually type each sentence of "
            "this email to another busy person? You do not care whether the idea is good or "
            "whether the recipient would reply — only whether the words read like a real human's "
            "quick, confident writing.\n"
            "Flag: formal-bot register (grammatical-stiff phrasing no one types in a quick note, "
            "e.g. ceremonious permission constructions); unglossed jargon compounds that make the "
            "reader do the work; more than ONE big number competing for attention; performed "
            "politeness or assistant-speak; marketing cadence; anything a busy peer would not "
            "plausibly have typed.\n" + pref_block +
            f"\n--- EMAIL ---\n{text}\n--- END ---\n"
            'Return ONLY JSON: {"score": 0.0-1.0, "flags": [{"sentence": "verbatim", '
            '"problem": "short reason"}]} where score is how much the WHOLE email reads as '
            "written by a real busy human (1.0 = fully).")
        try:
            raw = _call(chat_fn, prompt, max_tokens=420, temperature=0.0)
            m = re.search(r"\{.*\}", raw or "", re.S)
            obj = json.loads(m.group(0)) if m else {}
            score = max(0.0, min(1.0, float(obj.get("score", 0.0))))
            flags = [{"sentence": str(f.get("sentence", ""))[:120],
                      "problem": str(f.get("problem", ""))[:140]}
                     for f in obj.get("flags", []) if isinstance(f, dict)]
            return LanguageVerdict(ok=score >= gate and not flags, score=score, flags=flags)
        except Exception:  # noqa: BLE001 — judge outage falls back to the deterministic lexical gate
            from swm.decision.semantic_critic import SemanticCritic
            c = SemanticCritic().critique(text)
            score = min(c.naturalness, c.coherence)
            return LanguageVerdict(ok=score >= gate, score=score,
                                   flags=[{"sentence": f["sentence"][:90], "problem": f["issue"]}
                                          for f in c.flags()],
                                   source="lexical_fallback(llm_judge_failed)")
    return judge
