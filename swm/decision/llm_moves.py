"""Live-LLM seams for the message optimizer: a MOVE PROPOSER and a SENTENCE JUDGE.

These fill the two pluggable holes the architecture was built around, WITHOUT changing who is in charge.
The world model + beam search still select and assemble; the LLM only (a) writes candidate sentences for
ONE move at a time and (b) judges coherence/annoyingness. It never authors the whole email or picks the
winner — so the LLM's style bias can't dominate; it just raises the quality of the moves and the sharpness
of the critic.

  spec_to_instructions(strategy)  — translate the Layer-1 optimal strategy VECTOR into plain writing rules
                                    ("do not mention credentials", "lead contrarian", "one short question").
  llm_proposer(chat_fn, ...)      — returns propose_fn(slot, spec, ctx, k): the LLM writes k candidate
                                    sentences for that slot, honoring the rules + the recipient evidence +
                                    the sender's REAL facts (it may not invent). Local moves only.
  llm_sentence_judge(chat_fn)     — returns judge_fn(sentences) -> [{coherent, annoying, reason}] for the
                                    SemanticCritic's final gate.

`chat_fn(prompt) -> text` is any backend with that contract (swm/api/deepseek_backend.default_chat_fn is
the production one). Everything degrades: if the LLM errors or returns junk, callers fall back to the
offline bank / lexical critic, so the pipeline never breaks.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from swm.decision.strategy_scorer import MESSAGE_VARS
from swm.variables.schema import spec

# slot -> what that communicative move is. Roles are DISJOINT so the moves compose into one coherent
# email instead of four paraphrases of the same point.
_SLOT_ROLE = {
    "opener": "the FIRST line: greet the recipient by first name and state the single most specific, "
              "contrarian hook — one sentence. This is the ONLY place the core claim is stated.",
    "hook": "one short clause on who the sender is or why now — must add NEW information (e.g. the "
            "sender's age/situation), never restate the opener's claim. Return an empty string to skip.",
    "thesis": "the CONCRETE thing the sender is building or proposing, with a specific detail (a number, "
              "a mechanism) — NEW information the opener did not already state. Do not re-argue the hook.",
    "ask": "the SINGLE closing question — short, low-effort, answerable in a line. Do NOT restate the "
           "thesis; just ask for the reaction.",
    "close": "a sign-off — usually just the sender's first name.",
}


@dataclass
class SenderBrief:
    """The real substance the proposer may use (and must not exceed / invent beyond)."""
    sender: str = ""
    facts: list = field(default_factory=list)      # concrete, true facts about the sender/idea
    thesis: str = ""                               # the one-line real idea being pitched
    ask: str = ""                                  # what the sender actually wants back

    def to_prompt(self) -> str:
        lines = [f"Sender: {self.sender}"] if self.sender else []
        if self.thesis:
            lines.append(f"The real idea: {self.thesis}")
        if self.ask:
            lines.append(f"What the sender wants: {self.ask}")
        if self.facts:
            lines.append("Facts you may use (do NOT invent anything beyond these):")
            lines += [f"  - {f}" for f in self.facts]
        return "\n".join(lines)


def spec_to_instructions(strategy: dict) -> list[str]:
    """Turn the optimal strategy vector into natural-language writing constraints for the proposer."""
    g = lambda n: strategy.get(n, spec(n).default)
    rules = []
    if g("personalization") > 0.6:
        rules.append("Reference something specific and real about the recipient (from the notes). No generic flattery.")
    if g("credential_signaling") < 0.25:
        rules.append("Do NOT mention schools, degrees, GPA, awards, press features, or titles — they hurt here.")
    elif g("credential_signaling") > 0.7:
        rules.append("Briefly establish credibility with ONE concrete credential.")
    if g("contrarian_pitch") > 0.6:
        rules.append("Lead with a claim most people would disagree with; be specific about the disagreement.")
    if g("secret_density") > 0.6:
        rules.append("Include ONE specific, non-obvious insight — a real claim, not a vague teaser.")
    if g("pushiness") < 0.25:
        rules.append("No urgency and no pressure. Never say 'ASAP', 'circling back', 'following up', or 'quick call'.")
    if g("ask_directness") > 0.6:
        rules.append("If this is the ask, make it ONE clear, specific, low-effort question.")
    if g("length_fit") > 0.6:
        rules.append("Keep every sentence short and plain. Cut every unnecessary word.")
    if g("clarity") > 0.6:
        rules.append("Every sentence must make a concrete, literally-true claim a skeptic could act on.")
    return rules


_ANTI_SLOP = ("Write like a sharp, busy human who respects the reader's time. Each option is ONE short "
              "sentence. Being contrarian is about the CLAIM being non-consensus — NOT the tone: do not try "
              "to sound clever or provocative, do not write zingers, do not open with 'most people are "
              "wrong'. Write plainly, with understatement, like explaining to a smart friend. Never restate "
              "a point or pad with repetition. Forbidden: buzzwords (leverage, unlock, disrupt, paradigm, "
              "ecosystem, stack, margin, moat, flywheel), marketing adjectives (exciting, innovative, "
              "revolutionary, game-changing), fake humility ('I'll be brief', 'I'll leave you alone', "
              "'quick question', 'I know you're busy'), name-dropping the reader's own quotes back at them, "
              "and vague metaphors that don't literally parse. Avoid em dashes and en dashes (— –) in the "
              "body: a comma or a period almost always reads better and overusing dashes reads as AI "
              "writing (a dash in a sign-off like '— Beckett' is fine). Plain, concrete, specific.")


def _extract_list(text: str) -> list:
    """Tolerant parse of the LLM's reply into a list of candidate strings."""
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # fallback: one option per line, stripped of bullets/numbering/quotes/trailing commas
    out = []
    for ln in text.splitlines():
        ln = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', "", ln).strip()
        ln = ln.strip('",').strip().strip('"').strip()
        if ln and not ln.startswith("["):
            out.append(ln)
    return out


def llm_proposer(chat_fn, *, recipient_notes: str = "", sender: SenderBrief | None = None):
    """Build propose_fn(slot, spec_strategy, context, k) that asks the LLM for k candidate sentences."""
    sender = sender or SenderBrief()

    def propose(slot: str, spec_strategy: dict, context: dict, k: int = 6) -> list:
        role = _SLOT_ROLE.get(slot, "one line of the email")
        rules = spec_to_instructions(spec_strategy)
        prefix = (context or {}).get("prefix", "").strip()
        prompt = "\n".join([
            f"Write {k} DISTINCT candidate options for {role}",
            _ANTI_SLOP,
            "Writing rules for this message:",
            *[f"  - {r}" for r in rules],
            "",
            "Recipient notes:", recipient_notes or "  (none)",
            "",
            sender.to_prompt(),
            "",
            "The email so far reads: " + (f'"{prefix}"' if prefix else "(nothing yet — this is the start)"),
            "Continue it. Do NOT repeat any point already made above; only add new information.",
            "",
            f"Return ONLY a JSON array of {k} strings, each a single option for the {slot}. No prose.",
        ])
        try:
            options = _extract_list(chat_fn(prompt))
        except Exception:
            options = []
        # allow an empty option for optional slots (brevity), as the offline bank does
        if slot in ("hook", "close"):
            options = options + [""]
        return options[:k + 1] if options else ([""] if slot in ("hook", "close") else [])
    return propose


def llm_rewriter(chat_fn, *, recipient_notes: str = "", sender: SenderBrief | None = None):
    """Build rewrite_fn(sentence, reasons, spec_strategy) -> a plainer rewrite of a flagged line. This is
    the generator-level repair: instead of resampling more (equally tryhard) candidates, we tell the LLM
    exactly what a critic flagged and ask it to fix THAT line — the feedback loop from critic to writer."""
    sender = sender or SenderBrief()

    def rewrite(sentence: str, reasons: list, spec_strategy: dict) -> str:
        rules = spec_to_instructions(spec_strategy)
        prompt = "\n".join([
            "Rewrite this ONE line of a cold email so it is plainer and better. Keep its role and meaning.",
            f"Line: \"{sentence}\"",
            "A critic flagged it for: " + ("; ".join(reasons) if reasons else "sounding like AI slop"),
            _ANTI_SLOP,
            "Writing rules:", *[f"  - {r}" for r in rules],
            sender.to_prompt(),
            "Return ONLY the rewritten line as a plain string, no quotes, no prose.",
        ])
        try:
            out = chat_fn(prompt).strip().strip('"').strip()
            return out.split("\n")[0].strip() if out else sentence
        except Exception:
            return sentence
    return rewrite


def llm_sentence_judge(chat_fn):
    """Build judge_fn(sentences) -> [{coherent, annoying, reason}] for the SemanticCritic (LLM gate)."""
    SYSTEM = ("You are a ruthless editor of cold outreach emails. For each numbered sentence decide two "
              "booleans: COHERENT (it makes a concrete, literally-parseable claim a smart skeptic could "
              "act on — not vague metaphor or buzzwords) and ANNOYING (tryhard, embellishing, fake-humble, "
              "or a manipulative tic like 'I'll leave you alone'). Be harsh; when unsure, flag it.")

    def judge(sentences: list) -> list:
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
        prompt = (SYSTEM + "\n\nSentences:\n" + numbered +
                  '\n\nReturn ONLY a JSON array; element i = {"coherent": bool, "annoying": bool, '
                  '"reason": "short"} for sentence i, in order.')
        try:
            arr = _extract_list_json(chat_fn(prompt))
        except Exception:
            arr = []
        # pad/truncate to len(sentences) so the critic can zip safely
        out = []
        for i in range(len(sentences)):
            r = arr[i] if i < len(arr) and isinstance(arr[i], dict) else {}
            out.append({"coherent": bool(r.get("coherent", True)),
                        "annoying": bool(r.get("annoying", False)),
                        "reason": r.get("reason", "")})
        return out
    return judge


def _extract_list_json(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []
