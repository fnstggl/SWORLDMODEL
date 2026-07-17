"""The semantic critic — the quality axis the variable model is structurally blind to.

The strategy optimizer (L1–L3) scores a message on lexical VARIABLES (personalization, pushiness,
contrarian_pitch, secret_density…). Those are blind to two things that make an email read as AI slop:

  1. INCOHERENCE / EMBELLISHMENT — a sentence can be dense with the right markers and still be literal
     nonsense: "most of the AI stack rents margin it should own, and inference is where that flips" scores
     high on secret_density + contrarian_pitch, but when you actually READ it, it doesn't parse — vague
     referents ("that flips"), a mixed metaphor on an abstract noun ("rents margin it should own"), a
     tryhard opener ("the secret I'm betting on:").
  2. ANNOYINGNESS — "One line back and I'll leave you alone" scores as a low-friction ask but reads as a
     manipulative tic.

Meaning and tone are exactly what an LLM judge is good at and the variable readout cannot see. So the
critic is a separate, ADVERSARIAL evaluator over the actual text: it tries to FIND the slop. It runs in
two modes:

  - `judge_fn` (LLM): per-sentence {coherent?, annoying?, reason} — the production critic.
  - offline fallback: a transparent lexical detector over a curated slop/annoyance lexicon + structural
    incoherence heuristics (vague referents, buzz-metaphor density, no concrete anchor). Deterministic and
    auditable; catches the canonical tells; clearly lower-fidelity than the LLM.

It is used two ways (see compositional_search / message_pipeline): the CHEAP lexical critic penalizes slop
INSIDE the beam search so the email is never built toward it, and the FULL critic runs as a final GATE +
REPAIR loop — the "critical evaluator at the end" — flagging or rewriting any line that survived.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# --- annoyance / tryhard / embellishment lexicon (naturalness penalty) ----------------------------
_ANNOYING = [
    r"leave you alone", r"i'?ll be brief", r"i'?ll keep this short", r"keep it short",
    r"won'?t take (up )?much", r"won'?t take (up )?too much", r"one line( back)?", r"a single line",
    r"just one line", r"quick question", r"quick favou?r", r"pick your brain", r"picking your brain",
    r"circling back", r"just following up", r"reaching out", r"hope this (email )?finds you well",
    r"i know you'?re busy", r"no worries if not", r"feel free to", r"at your earliest convenience",
    r"would love to", r"i'?d love to", r"excited to", r"thrilled to", r"game[- ]changer",
    r"revolutionary", r"paradigm", r"synergy", r"touch base", r"hop on a (quick )?call",
    r"let me be direct", r"here'?s the thing", r"if i'?m being honest", r"real quick",
    r"the secret i'?m betting on", r"obviously wrong", r"needless to say", r"suffice it to say",
    r"look forward to hearing", r"awaiting your", r"per my last", r"as per", r"kindly",
    r"i'?ll cut to the chase", r"long story short", r"trust me", r"believe me",
]
_ANNOYING_RE = [re.compile(p, re.I) for p in _ANNOYING]

# --- convenience-selling / benefit-assurance (the pushy-frictionless register) ---------------------
# STRUCTURAL classes, not memorized lines: promising the reader what they'll get, assuring them a reply
# costs nothing, or pre-chewing an unrequested verification/demo step. A busy peer never performs
# easiness for the reader; performing it reads as salesy AI outreach. The production catch for this
# register is the LLM judge (meaning-level); these patterns are the transparent offline fallback.
_CONVENIENCE = [
    r"\byou('| wi|'l)?ll (get|receive|have|walk away with)\b",      # promising the reader a payoff
    r"\byou get (a|an|the)\b",
    r"\bno (follow[- ]?up|obligation|commitment|strings|pressure|need to (respond|reply))\b",
    r"\b(all|only) (it|this) takes is\b",
    r"\btakes? (just |only |under |less than )?(a |an |\d+ )?(second|seconds|minute|minutes|moment)\b",
    r"\bin under (a |an |\d+ )?(minute|minutes|hour|hours)\b",
    r"\byou could (test|verify|check|try|confirm) (this|it|that) (yourself|for yourself)\b",
    r"\bsee for yourself\b", r"\bjudge for yourself\b",
    r"\bif (i'?m|i am) wrong,? (just )?(delete|ignore|archive)\b",
    r"\brequired?: (none|nothing)\b", r"\bzero (effort|commitment|risk)\b",
]
_CONVENIENCE_RE = [re.compile(p, re.I) for p in _CONVENIENCE]

# --- incoherence signals (coherence penalty) ------------------------------------------------------
# vague referent + vacuous verb: "that flips", "this changes everything", "where it unlocks", ...
_VAGUE_REF = re.compile(r"\b(that|this|it|which|where that|where it)\s+"
                        r"(flips|changes everything|is the key|unlocks?|shifts?|matters|wins?|"
                        r"is where|comes in|breaks|clicks|compounds?)\b", re.I)
# buzzword/metaphor tokens; >=2 in a sentence with an abstract verb reads as vacuous
_BUZZ = re.compile(r"\b(margin|leverage|unlock|disrupt|paradigm|ecosystem|stack|moat|flywheel|"
                   r"primitive|rails|layer|curve|surface area|step[- ]change|inflection|tailwind|"
                   r"asymmetric|10x|first principles|second[- ]order|alpha|edge)\b", re.I)
# abstract-noun-as-physical-object metaphor: "rents margin", "owns the layer", "eats the stack"
_ABSTRACT_METAPHOR = re.compile(r"\b(rent|rents|own|owns|eat|eats|capture|captures|print|prints|"
                                r"harvest|harvests)\s+(the\s+)?(margin|stack|layer|moat|value|alpha|"
                                r"upside|surplus)\b", re.I)
# a concrete anchor makes a sentence legible: a number, a proper noun, or a concrete domain word
_CONCRETE = re.compile(r"\b(\d[\d,\.]*\s?(x|%|k|m|bn|billion|million|ms|gb|hours?|days?|weeks?|dollars?|cents?)?"
                       r"|inference|training|compute|latency|gpu|api|customers?|revenue|users?|model|chip|"
                       r"[A-Z][a-z]{2,})\b")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


_EM_DASH = re.compile(r"\s*[—–]\s*")          # em (U+2014) and en (U+2013) dash
_SIGNOFF_DASH = re.compile(r"[—–]\s*[A-Z][\w.'-]*(\s+[A-Z][\w.'-]*)?[.!?]?\s*$")   # "— Beckett" at line end
_EM_DASH_WEIGHT = 0.7                          # a body dash costs less than a full annoyance hit (soft bias)


def strip_em_dashes(text: str) -> str:
    """OPT-IN deterministic removal of em/en dashes (NOT applied by the pipeline). By default the system
    only DISCOURAGES body dashes (writer instruction + critic naturalness penalty) so a dash can still
    appear when it's genuinely the best option; call this if a caller wants a hard guarantee instead. A
    dash between clauses becomes a comma; a leading sign-off dash ('— Beckett') is dropped."""
    if not text or ("—" not in text and "–" not in text):
        return text
    t = re.sub(r"^\s*[—–]\s*", "", text.strip())      # leading sign-off dash
    t = _EM_DASH.sub(", ", t)                          # clause dash -> comma
    t = re.sub(r"\s*,\s*,", ",", t)                    # collapse doubled commas
    t = re.sub(r"\s+,", ",", t)                        # no space before comma
    t = re.sub(r",\s*([.!?])", r"\1", t)               # ", ." -> "."
    t = re.sub(r",\s+([A-Z][a-z]+\s*)$", r", \1", t)   # keep ", Name" sign-off tidy
    return t.strip()


def _sat(n: float, k: float = 1.2) -> float:
    return 1.0 - math.exp(-n / k)


_STOP = set("a an the and or but if to of in on for with is are was were be been being this that it "
            "i you he she we they my your our their me him her them as at by from about into so not "
            "no do does did would could should will can i'm you're it's that's what who how".split())


def _content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in _STOP and len(w) > 2}


_DISTINCT_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _distinct_numbers(s: str) -> set:
    """Distinctive numeric tokens (canonical; small counting numbers excluded) — a statistic like
    '724%' or '1.5M' is allowed to land ONCE in an email."""
    out = set()
    for m in _DISTINCT_NUM.findall(s or ""):
        c = m.replace(",", "")
        if "." in c:
            c = c.rstrip("0").rstrip(".")
        if c and c not in {str(i) for i in range(0, 13)}:
            out.add(c)
    return out


def _redundant_pairs(sents: list, thresh: float = 0.5) -> set:
    """Indices of sentences that substantially repeat an earlier one — by content-word Jaccard OR by
    REUSING a distinctive number an earlier sentence already delivered. The numeric check catches the
    paraphrased-statistic repeat ('a 724% gain…' / 'the 724% result…') that word overlap misses."""
    red = set()
    words = [_content_words(s) for s in sents]
    nums = [_distinct_numbers(s) for s in sents]
    for i in range(len(sents)):
        for j in range(i):
            a, b = words[i], words[j]
            if a and b and len(a & b) / len(a | b) >= thresh:
                red.add(i)
                break
            if nums[i] & nums[j]:
                red.add(i)
                break
    return red


@dataclass
class SentenceVerdict:
    sentence: str
    coherent: bool
    annoying: bool
    ai_sounding: bool = False   # reads as AI-generated outreach (register, not content)
    fabricated: bool = False    # asserts a factual detail the sender facts do not contain
    reasons: list = field(default_factory=list)


@dataclass
class Critique:
    coherence: float           # 0..1, higher = clearer
    naturalness: float         # 0..1, higher = less annoying
    humanness: float = 1.0     # 0..1, higher = less AI-sounding (register axis)
    factuality: float = 1.0    # 0..1, 1 = every claim entailed by the sender facts
    verdicts: list = field(default_factory=list)   # SentenceVerdict per sentence
    source: str = "lexical"

    @property
    def quality(self) -> float:
        """Single 0..1 quality gate for the search: the min across ALL axes — a line that is annoying,
        AI-sounding, or fabricated fails the gate even if it is perfectly coherent."""
        return min(self.coherence, self.naturalness, self.humanness, self.factuality)

    def flags(self) -> list:
        out = []
        for v in self.verdicts:
            if not v.coherent or v.annoying or v.ai_sounding or v.fabricated:
                issues = []
                if not v.coherent:
                    issues.append("incoherent/embellished")
                if v.annoying:
                    issues.append("annoying")
                if v.ai_sounding:
                    issues.append("AI-sounding")
                if v.fabricated:
                    issues.append("fabricated (not in sender facts)")
                out.append({"sentence": v.sentence, "issue": " + ".join(issues), "reasons": v.reasons})
        return out

    def summary(self) -> dict:
        return {"coherence": round(self.coherence, 3), "naturalness": round(self.naturalness, 3),
                "humanness": round(self.humanness, 3), "factuality": round(self.factuality, 3),
                "quality": round(self.quality, 3), "source": self.source, "flags": self.flags()}


@dataclass
class SemanticCritic:
    """Adversarial text-quality critic. `judge_fn(sentences) -> [{coherent, annoying, reason}]` uses an
    LLM; without it, the transparent lexical fallback runs. `critique(text)` returns per-sentence verdicts
    and the aggregate axes. `allowed_numbers` (optional, canonical numeric tokens from the sender's real
    facts) arms a DETERMINISTIC factuality overlay: a sentence asserting a distinctive number outside the
    facts is marked fabricated even if the LLM judge was lenient about it."""
    judge_fn: object = None
    allowed_numbers: object = None                      # set[str] | None

    def critique(self, text: str) -> Critique:
        sents = _sentences(text)
        if not sents:
            return Critique(coherence=0.0, naturalness=1.0, verdicts=[], source="empty")
        if self.judge_fn is not None:
            try:
                crit = self._llm_critique(text, sents)
            except Exception:  # noqa: BLE001 — judge outage falls back to the LEXICAL critic (which
                # flags slop), never to all-pass, and the source label says so honestly
                crit = self._lexical_critique(sents)
                crit.source = "lexical_fallback(llm_judge_failed)"
        else:
            crit = self._lexical_critique(sents)
        crit = self._apply_number_facts(crit, sents)
        return self._apply_redundancy(crit, sents)

    def _apply_number_facts(self, crit: Critique, sents: list) -> Critique:
        """Deterministic numeric-factuality overlay (both judge paths): distinctive numbers not present
        in the sender facts mark the sentence fabricated. Belt to the propose-time guard's suspenders."""
        if self.allowed_numbers is None:
            return crit
        allowed = set(self.allowed_numbers)
        any_bad = False
        for i, s in enumerate(sents):
            bad = sorted(_distinct_numbers(s) - allowed)
            if bad and i < len(crit.verdicts):
                crit.verdicts[i].fabricated = True
                crit.verdicts[i].reasons.append(f"number(s) not in sender facts: {bad}")
                any_bad = True
        if any_bad:
            crit.factuality = 0.0
        return crit

    def _apply_redundancy(self, crit: Critique, sents: list) -> Critique:
        """Structural cross-slot check (applies to BOTH the LLM and lexical paths): a sentence that
        substantially repeats an earlier one is marked incoherent (redundant), which drops coherence and
        lets the gate/repair remove the repetition an independent per-sentence judge can't see."""
        red = _redundant_pairs(sents)
        if not red:
            return crit
        for i in red:
            if i < len(crit.verdicts):
                crit.verdicts[i].coherent = False
                crit.verdicts[i].reasons.append("redundant: repeats an earlier sentence")
        crit.coherence = min(crit.coherence, 1.0 - _sat(len(red)))
        return crit

    # ---- offline, transparent ----
    def _lexical_critique(self, sents: list) -> Critique:
        verdicts, coh_scores, nat_scores = [], [], []
        n_sents = len(sents)
        for idx, s in enumerate(sents):
            reasons = []
            # annoyance
            ann_hits = [p.pattern for p in _ANNOYING_RE if p.search(s)]
            conv_hits = [p.pattern for p in _CONVENIENCE_RE if p.search(s)]
            if conv_hits:
                ann_hits = ann_hits + conv_hits
                reasons.append("convenience-selling: promises the reader a payoff / assures zero "
                               "effort / pre-chews a verification step — pushy-frictionless register")
            # em/en dashes in the BODY are discouraged (LLMs overuse them). A sign-off dash ("— Beckett"
            # / "..., — Beckett" as the final short line) is fine and exempt. This is a soft penalty that
            # biases the search toward commas/periods, NOT a hard ban — a dash survives when it's the best
            # option, just far less often than an LLM alone would use it.
            em_total = len(re.findall(r"[—–]", s))
            is_signoff = (idx == n_sents - 1 and len(s.split()) <= 6
                          and bool(_SIGNOFF_DASH.search(s)))
            em_body = max(0, em_total - (1 if is_signoff else 0))
            annoying = len(ann_hits) > 0
            if ann_hits:
                reasons.append("annoying/embellishing phrase: " + ", ".join(ann_hits[:3]))
            if em_body:
                reasons.append(f"{em_body} em/en dash(es) in the body: prefer a comma or period "
                               "(a sign-off dash is fine)")
            nat = 1.0 - _sat(len(ann_hits) + _EM_DASH_WEIGHT * em_body)
            # coherence
            vague = len(_VAGUE_REF.findall(s))
            buzz = len(_BUZZ.findall(s))
            metaphor = len(_ABSTRACT_METAPHOR.findall(s))
            has_anchor = bool(_CONCRETE.search(s))
            n_words = len(s.split())
            is_question = s.rstrip().endswith("?")
            # a short line or a question is often legitimately anaphoric ("Do you think that's wrong?") —
            # only a longer DECLARATIVE sentence with no concrete anchor reads as vacuous.
            anchor_pen = 0.6 if (not has_anchor and n_words >= 8 and not is_question) else 0.0
            incoh = vague + metaphor + max(0, buzz - 1) * 0.7 + anchor_pen
            if vague:
                reasons.append("vague referent (unclear what 'that/this/it' refers to)")
            if metaphor:
                reasons.append("abstract-noun metaphor that doesn't literally parse")
            if buzz >= 2:
                reasons.append("buzzword density without a concrete claim")
            if anchor_pen:
                reasons.append("no concrete anchor (number, name, or specific noun)")
            coh = 1.0 - _sat(incoh)
            coherent = coh >= 0.5
            # register axis (offline approximation): convenience-selling is the strongest lexical
            # signal of AI-outreach register; the LLM judge is the real detector
            ai_sounding = len(conv_hits) >= 2
            coh_scores.append(coh)
            nat_scores.append(nat)
            verdicts.append(SentenceVerdict(sentence=s, coherent=coherent, annoying=annoying,
                                            ai_sounding=ai_sounding, reasons=reasons))
        return Critique(coherence=min(coh_scores), naturalness=min(nat_scores),
                        humanness=min((0.0 if v.ai_sounding else 1.0) for v in verdicts),
                        verdicts=verdicts, source="lexical")

    # ---- production LLM critic ----
    def _llm_critique(self, text: str, sents: list) -> Critique:
        results = self.judge_fn(sents) or []
        verdicts, coh, nat, hum, fac = [], [], [], [], []
        for s, r in zip(sents, results):
            coherent = bool(r.get("coherent", True))
            annoying = bool(r.get("annoying", False))
            ai_sounding = bool(r.get("ai_sounding", False))
            fabricated = bool(r.get("fabricated", False))
            reasons = [r["reason"]] if r.get("reason") else []
            verdicts.append(SentenceVerdict(s, coherent, annoying, ai_sounding, fabricated, reasons))
            coh.append(1.0 if coherent else 0.0)
            nat.append(0.0 if annoying else 1.0)
            hum.append(0.0 if ai_sounding else 1.0)
            fac.append(0.0 if fabricated else 1.0)
        return Critique(coherence=min(coh) if coh else 0.0, naturalness=min(nat) if nat else 1.0,
                        humanness=min(hum) if hum else 1.0, factuality=min(fac) if fac else 1.0,
                        verdicts=verdicts, source="llm")


# The production sentence judge lives in swm/decision/llm_moves.llm_sentence_judge (four axes:
# coherent / annoying / ai_sounding / fabricated-vs-facts, strict fail-closed parsing). An older
# two-axis judge that lived here was removed: it silently defaulted the register and factuality
# axes to "pass", which is exactly the failure mode this critic exists to prevent.
