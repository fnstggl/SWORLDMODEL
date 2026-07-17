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


def _call(chat_fn, prompt: str, *, max_tokens: int = None, temperature: float = None) -> str:
    """Budget-aware invocation: backends that accept per-call overrides (deepseek_chat_fn) get them;
    plain fn(prompt) callables still work. This exists because a judge/proposer sharing one small
    default completion budget TRUNCATES its JSON, and a truncated judge reply used to fail OPEN —
    the single worst failure mode this stack has had."""
    kw = {}
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    if temperature is not None:
        kw["temperature"] = temperature
    if kw:
        try:
            return chat_fn(prompt, **kw)
        except TypeError:
            pass                                       # fixed-budget callable — use as given
    return chat_fn(prompt)


# ---------------------------------------------------------------- deterministic numeric fact guard
_NUM_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SMALL_WHITELIST = {str(i) for i in range(0, 13)}      # "one line", "3 sentences" — natural language


def _canon_number(tok: str) -> str:
    t = tok.replace(",", "")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return t


def allowed_numbers(*texts: str) -> set:
    """The set of canonical numeric tokens the writer may use: everything appearing in the sender's
    real facts (and recipient notes). Anything else with >=2 significant digits is treated as a
    fabricated specific — rejected BEFORE scoring, not just flagged after."""
    out = set()
    for t in texts:
        for m in _NUM_TOKEN.findall(t or ""):
            out.add(_canon_number(m))
    return out


def number_violations(line: str, allowed: set) -> list:
    """Numeric tokens in `line` that are neither whitelisted small integers nor in the allowed set."""
    bad = []
    for m in _NUM_TOKEN.findall(line or ""):
        c = _canon_number(m)
        if c in _SMALL_WHITELIST or c in allowed:
            continue
        bad.append(m)
    return bad


def numbers_in(line: str) -> set:
    """Distinctive numbers in a line (for no-reuse checks): canonical, small integers excluded."""
    return {_canon_number(m) for m in _NUM_TOKEN.findall(line or "")} - _SMALL_WHITELIST

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


def spec_to_instructions(strategy: dict, levers: list | None = None) -> list[str]:
    """Turn the optimal strategy vector (general levers + situational levers) into natural-language
    writing constraints for the proposer."""
    from swm.variables.schema import SPECS
    g = lambda n: strategy.get(n, SPECS[n].default if n in SPECS else 0.0)
    rules = []
    if g("personalization") > 0.6:
        rules.append("Reference something specific and real about the recipient (from the notes). No generic flattery.")
    if g("relevance_fit") > 0.6:
        rules.append("Make clear, in a few words, why this is relevant to THEM specifically.")
    if g("credibility_proof") > 0.6:
        rules.append("Include ONE concrete piece of proof or traction (a real number, metric, or result).")
    if g("responder_incentive") > 0.6:
        rules.append("State plainly what the recipient personally gets out of engaging.")
    if g("credential_signaling") < 0.25:
        rules.append("Do NOT mention schools, degrees, GPA, awards, press features, or titles; they hurt here.")
    elif g("credential_signaling") > 0.7:
        rules.append("Briefly establish credibility with ONE concrete credential.")
    if g("pushiness") < 0.25:
        rules.append("No urgency and no pressure. Never say 'ASAP', 'circling back', 'following up', or 'quick call'.")
    if g("ask_directness") > 0.6 or g("low_effort_ask") > 0.6:
        rules.append("End with ONE clear, specific, low-effort ask they can answer in a line.")
    if g("length_fit") > 0.6:
        rules.append("Keep every sentence short and plain. Cut every unnecessary word.")
    if g("clarity") > 0.6:
        rules.append("Every sentence must make a concrete, literally-true claim a skeptic could act on.")
    if g("warmth") > 0.6:
        rules.append("Keep a warm, respectful, human tone.")
    # situational levers the optimizer turned up for THIS recipient
    for lv in (levers or []):
        if strategy.get(lv.name, 0.0) > 0.6 and lv.description:
            rules.append(f"For this recipient specifically: {lv.description}")
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
              "writing (a dash in a sign-off like '— Beckett' is fine). Plain, concrete, specific.\n"
              "NEVER SELL CONVENIENCE: do not tell the recipient what they will get, how little effort a "
              "reply takes, that there is 'no follow-up' or 'no obligation', how quickly they could verify "
              "something, or what they 'could test' — pre-chewing the reader's next step is presumptuous "
              "salesmanship dressed as politeness, and it is the single fastest way to sound like AI "
              "outreach. A real busy person states the thing once and asks one plain question. "
              "STAY INSIDE THE FACTS: every number, dataset, client, or result you mention must come "
              "verbatim from the sender facts below; if a detail is not in the facts, you may not use it.")


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


def llm_proposer(chat_fn, *, recipient_notes: str = "", sender: SenderBrief | None = None, levers=None):
    """Build propose_fn(slot, spec_strategy, context, k) that asks the LLM for k candidate sentences.

    Every candidate passes the DETERMINISTIC numeric fact guard before it may be scored: a line
    asserting a number that is in neither the sender facts nor the recipient notes is rejected at
    the door (the LLM judge is the semantic layer; this closes the '31% on a 256-GPU run' class
    mechanically). Candidates that REUSE a distinctive number already present in the email so far
    are also rejected — a statistic lands once."""
    sender = sender or SenderBrief()
    allowed = allowed_numbers(sender.to_prompt(), recipient_notes)

    def propose(slot: str, spec_strategy: dict, context: dict, k: int = 6) -> list:
        role = _SLOT_ROLE.get(slot, "one line of the email")
        rules = spec_to_instructions(spec_strategy, levers=levers)
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
            "Continue it. Do NOT repeat any point already made above; only add new information. "
            "If a number or statistic already appears in the email so far, you may NOT use it again.",
            "",
            f"Return ONLY a JSON array of {k} strings, each a single option for the {slot}. No prose.",
        ])
        try:
            options = _extract_list(_call(chat_fn, prompt, max_tokens=700))
        except Exception:
            options = []
        used = numbers_in(prefix)
        kept, rejected = [], []
        for o in options:
            bad = number_violations(o, allowed)
            if bad:
                rejected.append({"line": o[:90], "reason": f"fabricated number(s): {bad}"})
                continue
            reuse = numbers_in(o) & used
            if reuse:
                rejected.append({"line": o[:90], "reason": f"repeats number(s) already used: {sorted(reuse)}"})
                continue
            kept.append(o)
        propose.last_rejected = rejected                # observability: what the guard refused and why
        # allow an empty option for optional slots (brevity), as the offline bank does
        if slot in ("hook", "close"):
            kept = kept + [""]
        return kept[:k + 1] if kept else ([""] if slot in ("hook", "close") else [])
    propose.last_rejected = []
    return propose


def llm_rewriter(chat_fn, *, recipient_notes: str = "", sender: SenderBrief | None = None):
    """Build rewrite_fn(sentence, reasons, spec_strategy) -> a plainer rewrite of a flagged line. This is
    the generator-level repair: instead of resampling more (equally tryhard) candidates, we tell the LLM
    exactly what a critic flagged and ask it to fix THAT line — the feedback loop from critic to writer.
    A rewrite may DELETE the line (return "") when the flagged content shouldn't exist at all (redundant
    restatement, convenience-selling with nothing underneath). A rewrite that INTRODUCES a number outside
    the sender facts is discarded (repair must never inject fabrication)."""
    sender = sender or SenderBrief()
    allowed = allowed_numbers(sender.to_prompt(), recipient_notes)

    def rewrite(sentence: str, reasons: list, spec_strategy: dict) -> str:
        rules = spec_to_instructions(spec_strategy)
        prompt = "\n".join([
            "Rewrite this ONE line of a cold email so it is plainer and better. Keep its role; keep only "
            "the meaning that a busy human peer would actually say. If the line sells convenience "
            "(promises what the reader gets, how easy replying is, or how they could verify something), "
            "DELETE that framing entirely rather than rephrasing it. If it asserts any factual detail "
            "not in the sender facts, replace it with a listed fact or drop the claim. If the line only "
            "repeats a point or statistic the email already made, return an empty string to delete it.",
            f"Line: \"{sentence}\"",
            "A critic flagged it for: " + ("; ".join(reasons) if reasons else "sounding like AI slop"),
            _ANTI_SLOP,
            "Writing rules:", *[f"  - {r}" for r in rules],
            sender.to_prompt(),
            "Return ONLY the rewritten line as a plain string (or an empty string to delete the line), "
            "no quotes, no prose.",
        ])
        try:
            out = _call(chat_fn, prompt, max_tokens=300).strip().strip('"').strip()
            out = out.split("\n")[0].strip()
            if out.lower() in ("(empty)", "(deleted)", "empty string", "delete"):
                out = ""
            if number_violations(out, allowed):
                return sentence                        # repair injected a fabricated number — refuse it
            return out
        except Exception:
            return sentence
    return rewrite


def llm_message_encoder(chat_fn, *, levers: list | None = None):
    """The LLM MESSAGE ENCODER — replaces the hardcoded lexical encoder. A tightly system-prompted LLM
    reads a message and scores each general lever (and any situational levers) 0..1 with a one-line
    justification. Returns encode(text) -> {var: value}. Falls back to the lexical encoder on any error,
    so it degrades gracefully. This is the fix for the 'Wharton'/'reply yes' class of misses: meaning,
    not keywords."""
    from swm.decision.compositional_search import encode_text_to_strategy
    from swm.decision.strategy_scorer import MESSAGE_VARS
    from swm.variables.schema import spec

    defs = [f"- {v}: {spec(v).description}" for v in MESSAGE_VARS]
    lever_names = []
    for lv in (levers or []):
        defs.append(f"- {lv.name}: {lv.description}")
        lever_names.append(lv.name)
    names = list(MESSAGE_VARS) + lever_names

    SYSTEM = ("You score a cold message on how strongly it exhibits each quality below, each 0.0 to 1.0 "
              "(0 = absent, 1 = strongly present), reading for MEANING not keywords. Be accurate and "
              "calibrated: a clear imperative ask like \"just reply yes\" is high ask_directness AND high "
              "low_effort_ask even with no question mark; naming a school/press/accelerator is "
              "credential_signaling; concrete metrics (users, revenue, growth) are credibility_proof; "
              "stating what the recipient personally gains is responder_incentive.")

    def encode(text: str) -> dict:
        prompt = (SYSTEM + "\n\nQualities:\n" + "\n".join(defs) +
                  f"\n\nMESSAGE:\n\"\"\"\n{text}\n\"\"\"\n\n"
                  "Return ONLY a JSON object mapping each quality name to its 0..1 score. "
                  "Names must be exactly: " + ", ".join(names) + ".")
        try:
            raw = _call(chat_fn, prompt, max_tokens=500, temperature=0.0)
            m = re.search(r"\{.*\}", raw, re.S)
            obj = json.loads(m.group(0)) if m else {}
            out = {}
            for k in names:
                v = obj.get(k)
                out[k] = min(1.0, max(0.0, float(v))) if isinstance(v, (int, float)) else None
            # fill any missing/invalid from the lexical fallback so the vector is always complete
            if any(v is None for v in out.values()):
                lex = encode_text_to_strategy(text, levers=levers)
                out = {k: (out[k] if out[k] is not None else lex.get(k, 0.3)) for k in names}
            return out
        except Exception:
            return encode_text_to_strategy(text, levers=levers)
    return encode


def llm_sentence_judge(chat_fn, *, facts_text: str = ""):
    """Build judge_fn(sentences) -> [{coherent, annoying, ai_sounding, fabricated, reason}] for the
    SemanticCritic's gate. Four axes, judged by MEANING (never a phrase list):

      COHERENT    — a concrete, literally-parseable claim a smart skeptic could act on.
      ANNOYING    — reads as a turn-off to a busy, high-status recipient. The big class here is
                    CONVENIENCE-SELLING: telling the reader what they'll get, how little effort a reply
                    takes ('no follow-up required'), pre-chewing a verification step they never asked
                    for ('you could test this yourself by…'), or any benefit-assurance. It performs
                    'frictionless' and lands as pushy and presumptuous. Also: tryhard, fake-humble,
                    manipulative tics.
      AI_SOUNDING — would a real busy founder ever text this to a peer? Templated benefit-framing,
                    assistant-register politeness, symmetrical setup-payoff clauses, generic demo
                    instructions, and over-explained asks all read as AI outreach even when polite.
      FABRICATED  — (only when sender facts are supplied) the sentence asserts a specific factual
                    detail (a number, dataset, client, event, result) that the facts do not contain.
                    Rephrasing a fact is fine; inventing or altering one is a flag."""
    SYSTEM = ("You are a ruthless editor reviewing a cold email to a busy, skeptical, high-status "
              "recipient. Judge each numbered sentence on the axes defined below, by meaning, not by "
              "keyword. Be harsh; when unsure, flag it.\n"
              "COHERENT: concrete, literally parseable, actionable by a skeptic.\n"
              "ANNOYING: would make this recipient like the sender LESS — especially convenience-"
              "selling (promising what they'll get, 'no follow-up required', unprompted 'you could "
              "test/verify this yourself by…' instructions, benefit-assurances, presumptuous "
              "helpfulness), plus tryhard cleverness, fake humility, or manipulative tics. A natural "
              "human peer states the thing once and asks one plain question; anything performing "
              "easiness for the reader is annoying.\n"
              "AI_SOUNDING: reads like AI-generated outreach rather than something a busy human would "
              "actually type: templated benefit-framing, assistant-register politeness, symmetrical "
              "setup-and-payoff phrasing, over-structured explanation of the ask.\n"
              "FABRICATED: asserts a specific number/dataset/client/result NOT present in the sender "
              "facts (if facts are given). Rewording a listed fact is NOT fabrication.")

    def judge(sentences: list) -> list:
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
        facts = f"\n\nSENDER FACTS (ground truth; anything factual beyond these is fabricated):\n{facts_text}" \
            if facts_text else ""
        prompt = (SYSTEM + facts + "\n\nSentences:\n" + numbered +
                  '\n\nReturn ONLY a JSON array; element i = {"coherent": bool, "annoying": bool, '
                  '"ai_sounding": bool, "fabricated": bool, "reason": "short"} for sentence i, in order.')
        # STRICT, FAIL-CLOSED: the completion budget scales with the sentence count (a truncated JSON
        # array used to parse as [] and default every sentence to clean — the judge silently failing
        # OPEN). Now: one retry, then a parse/length failure RAISES, and SemanticCritic falls back to
        # the lexical critic — which flags slop — rather than to "everything passes".
        budget = 260 + 150 * len(sentences)
        arr = _extract_list_json(_call(chat_fn, prompt, max_tokens=budget, temperature=0.0))
        if len(arr) < len(sentences):
            arr = _extract_list_json(_call(chat_fn, prompt + "\nReturn the COMPLETE array — one element "
                                           "per sentence.", max_tokens=budget * 2, temperature=0.0))
        if len(arr) < len(sentences) or not all(isinstance(x, dict) for x in arr[:len(sentences)]):
            raise RuntimeError(f"sentence judge returned {len(arr)}/{len(sentences)} verdicts — "
                               "refusing to default-pass")
        out = []
        for i in range(len(sentences)):
            r = arr[i]
            out.append({"coherent": bool(r.get("coherent", True)),
                        "annoying": bool(r.get("annoying", False)),
                        "ai_sounding": bool(r.get("ai_sounding", False)),
                        "fabricated": bool(r.get("fabricated", False)) if facts_text else False,
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
