"""The COLD-OUTREACH CONTENT CONTRACT — what any cold message to a stranger must contain.

The failed Thiel output ("Peter, treating data center power as a static budget ignores that … cut
GPU-hours by 84% … Which assumption in that claim is wrong?") passed every register/style gate and
still was a bad cold email, because nothing in the architecture REQUIRED the things a stranger needs:
who is writing, what they built, why the recipient is being contacted, a believable claim, and a
tiny explicit next step. Style gates cannot supply missing content.

This module makes the content contract a first-class, deterministic gate:

    1. IDENTITY        — the sender introduces themself (who / what they're building). Identity is
                         NOT credential signaling: "I'm Beckett, 17, building Aurelius" is identity;
                         "a Princeton admit featured in the NYT" is credentials.
    2. THESIS          — one clear declarative claim (not only a question).
    3. EVIDENCE + PROVENANCE — any extraordinary number must sit WITH its provenance (what was
                         measured, against what). A bare "+724%" is implausible before interesting.
    4. RELEVANCE       — an explicit bridge to the recipient (name + why-them).
    5. TINY NEXT STEP  — an explicit, trivially answerable ask with an obvious payoff
                         ("may I send you the one-page memo?"), not an invitation to perform
                         unpaid diligence ("which assumption is wrong?").
    6. LENGTH          — a cold email a busy stranger will actually read (<= ~130 words).

`validate(text, sender)` returns typed missing-elements/flags; drafts failing hard elements are
rejected BEFORE scoring (same discipline as the numeric fact guard). `plain_baseline_draft` builds
the honest human-register baseline every optimized candidate must beat under the system's own
evaluator — if the machinery cannot beat the plain draft, the plain draft ships.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_IDENTITY = re.compile(
    r"\b(i'?m [A-Z][a-z]+|my name is|i am [A-Z][a-z]+|i'?m (a |an )?\d{2}[ -]year[ -]old|"
    r"i built|i'?m building|i am building|we built|i run|i started|i'?ve been building)", re.I)
_WHY_YOU = re.compile(r"\b(you|your)\b", re.I)
_NEXT_STEP = re.compile(
    r"(may i send|could i send|can i send|want me to send|should i send|happy to send|"
    r"would you (want|like|read)|can i share|could i share|interested in seeing|"
    r"reply ['\"]?(yes|no)|worth a look|one[- ]pag(e|er)|memo|write[- ]?up)\b", re.I)
_QUESTION = re.compile(r"\?")
_BIG_NUMBER = re.compile(r"\b\d{2,}(?:[,.]\d+)?\s?(%|percent|x\b)|\b\d+(?:\.\d+)?x\b", re.I)
_PROVENANCE = re.compile(
    r"\b(replay|replays|trace|traces|backtest|benchmark|baseline|vs\.?|versus|against|compared|"
    r"production[- ]style|production scheduler|pilot|deployment|method|public)\b", re.I)
_DILIGENCE_ASK = re.compile(
    r"(which (assumption|part|claim)|what('| i)s wrong|prove (me|it) wrong|poke holes|"
    r"tear (it|this) apart|find the flaw|is (that|this) (thesis )?wrong)", re.I)


@dataclass
class ContractVerdict:
    ok: bool
    missing: list = field(default_factory=list)      # hard failures (draft rejected)
    flags: list = field(default_factory=list)        # soft warnings (surfaced, scored down)

    def as_dict(self):
        return {"ok": self.ok, "missing": self.missing, "flags": self.flags}


def validate(text: str, sender=None, *, max_words: int = 130) -> ContractVerdict:
    """Deterministic contract check. Hard elements: identity, thesis, next step, length.
    Soft flags: unanchored big numbers, diligence-bait asks, missing recipient bridge."""
    t = (text or "").strip()
    v = ContractVerdict(ok=True)
    if not t:
        return ContractVerdict(ok=False, missing=["empty"])
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    words = len(t.split())

    # 1. identity — must appear in the first two sentences (an introduction, not a footnote)
    head = " ".join(sentences[:2])
    sender_name = getattr(sender, "sender", "") or ""
    if not (_IDENTITY.search(head) or (sender_name and re.search(
            rf"\b(i'?m|i am) {re.escape(sender_name)}\b", head, re.I))):
        v.missing.append("identity: a stranger cannot tell who is writing from the first two "
                         "sentences ('I'm <name>, building <thing>')")

    # 2. thesis — at least one declarative (non-question) sentence beyond the identity line
    declaratives = [s for s in sentences if not s.rstrip().endswith("?")]
    if len(declaratives) < 2:
        v.missing.append("thesis: no declarative claim beyond the introduction")

    # 5. next step — an explicit tiny ask; a bare challenge question is NOT a next step
    if not _NEXT_STEP.search(t):
        if _QUESTION.search(t) and not _DILIGENCE_ASK.search(t):
            v.flags.append("next_step: the closing question is not an explicit tiny ask with an "
                           "obvious payoff (e.g. 'may I send the one-page memo?')")
        else:
            v.missing.append("next_step: no explicit, trivially answerable ask — the recipient "
                             "cannot tell what replying accomplishes")
    if _DILIGENCE_ASK.search(t):
        v.flags.append("diligence_bait: the ask requests unpaid technical diligence / debate "
                       "('which assumption is wrong?') — high effort, adversarial from a stranger")

    # 3. evidence provenance — big numbers need an anchor in the SAME sentence
    for s in sentences:
        if _BIG_NUMBER.search(s) and not _PROVENANCE.search(s):
            v.flags.append(f"unanchored_claim: extraordinary number without provenance in: "
                           f"\"{s[:70]}\"")

    # 4. relevance bridge
    if not _WHY_YOU.search(t):
        v.flags.append("relevance: no explicit bridge to the recipient (why THEM)")

    # 6. length
    if words > max_words:
        v.missing.append(f"length: {words} words > {max_words} — a busy stranger stops reading")

    v.ok = not v.missing
    return v


def plain_baseline_draft(sender, recipient_label: str = "") -> str:
    """The honest human-register baseline: identity → thesis → evidence-with-provenance →
    tiny permission ask. Built deterministically from the REAL brief (no LLM, no cleverness).
    Every optimized candidate must beat this under the system's own evaluator, or this ships."""
    first = (recipient_label or "there").split()[0]
    name = getattr(sender, "sender", "") or "a founder"
    thesis = (getattr(sender, "thesis", "") or "").rstrip(".")
    facts = [f.rstrip(".") for f in (getattr(sender, "facts", []) or [])]
    ident_bits = [f for f in facts if re.search(r"\b(\d{2}[ -]year|years old|building|starting)\b",
                                                f, re.I)][:2]
    ident = "; ".join(ident_bits) if ident_bits else f"I'm {name}"
    evidence = next((f for f in facts if _BIG_NUMBER.search(f) and _PROVENANCE.search(f)),
                    next((f for f in facts if _PROVENANCE.search(f)), ""))
    lines = [f"{first}, I'm {name}: {ident}."]   # template prose spends no dash budget
    if thesis:
        lines.append(f"{thesis[0].upper()}{thesis[1:]}.")
    if evidence:
        lines.append(f"So far: {evidence}.")
    lines.append("May I send you the one-page memo?")
    lines.append(name)
    return " ".join(lines)
