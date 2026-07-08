"""LAYER 2 — compositional construction: assemble the email move-by-move, the world model selects.

This is "build it sentence by sentence inside the simulation." An email is a sequence of communicative
MOVES (slots): opener → hook → thesis/secret → ask → close. At each slot a proposer offers a few short
candidate sentences; the world model scores every PARTIAL assembly; beam search keeps the top few and
extends them. The LLM (or the offline sentence bank) only ever proposes a LOCAL move — it never authors a
whole email — which is exactly why the chatty-cover-letter failure mode cannot occur: no global "write me
an email" call exists.

The objective at every partial step is the same Layer-1 objective — the `StrategyScorer`'s pessimistic
lower bound — computed on the assembly's ENCODED strategy (text → message variables), minus a small penalty
for drifting off the Layer-1 optimal spec. So the search realizes the optimal strategy in words and is
selected by P(reply), not by fluency.

`encode_text_to_strategy` is the bridge text→variables. It mirrors the lexical signals the
VariableInferenceEngine uses (pushiness, personalization, ask-directness, length-fit) plus content-stance
detectors (credential-signaling, contrarian-pitch, secret-density), and every signal SATURATES — repeating
a trick ("?????") stops paying — which, together with the lower-bound objective, closes the Goodhart hole.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from swm.decision.strategy_scorer import MESSAGE_VARS, StrategyScorer
from swm.variables.schema import spec

# --- lexical detectors (saturating) ---------------------------------------------------------------
_PUSHY = re.compile(r"\b(urgent|asap|immediately|act now|last chance|final notice|don'?t miss|limited "
                    r"time|just following up|circling back|per my last|quick favou?r|please respond|"
                    r"look forward to hearing|awaiting your|at your earliest)\b", re.I)
_PERSONAL = re.compile(r"\b(i saw your|i read your|your essay|your work on|your talk|your book|your "
                       r"post|congrat|loved your|admired your|since you)\b", re.I)
_SECOND_PERSON = re.compile(r"\b(you|your|you're|you've)\b", re.I)
_CREDENTIAL = re.compile(r"\b(princeton|harvard|yale|stanford|mit|ivy|nyt|new york times|featured|"
                         r"admit|admitted|forbes|valedictorian|gpa|ranked|award|prestigious|honou?rs?|"
                         r"scholarship|dean's list)\b", re.I)
_CONTRARIAN = re.compile(r"\b(wrong|most people|everyone (thinks|believes)|disagree|against the grain|"
                         r"contrarian|consensus|nobody|few people|unpopular|counterintuitive|"
                         r"conventional wisdom|the mistake)\b", re.I)
_SECRET = re.compile(r"\b(secret|betting|the real|actually|non[- ]obvious|should own|overlooked|hidden|"
                     r"the truth|the insight|what no one|underpriced|mispriced)\b", re.I)
_NUMBER = re.compile(r"\b\d+(\.\d+)?\s?(x|%|k|m|bn|billion|million|percent|cents?|ms| x faster)?\b", re.I)
_QMARK = re.compile(r"\?")


def _sat(count: int, k: float = 1.5) -> float:
    """Saturating 0..1 response to a marker count (diminishing returns — anti-Goodhart)."""
    return 1.0 - math.exp(-count / k)


def encode_text_to_strategy(text: str) -> dict:
    """Map an email's text to the message-controllable variable vector the scorer reads."""
    t = text or ""
    words = t.split()
    n = max(1, len(words))
    sentences = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    avg_sent = n / max(1, len(sentences))

    pushy = _sat(len(_PUSHY.findall(t)))
    personal = _sat(len(_PERSONAL.findall(t)) * 1.3 + 0.15 * len(_SECOND_PERSON.findall(t)))
    asks = len(_QMARK.findall(t))
    # one clear question is ideal; zero is vague; many is scattershot (saturate then penalize excess)
    ask_directness = min(1.0, _sat(asks, k=0.8)) * (1.0 if asks <= 2 else 0.7)
    # length fit: bell centered ~42 words (a crisp cold ask); very long / empty is worse
    length_fit = math.exp(-((math.log1p(n) - math.log(42)) ** 2) / 1.6)
    # clarity: short sentences + a concrete number read as clear/actionable
    clarity = min(1.0, 0.35 + 0.4 * (1.0 if _NUMBER.search(t) else 0.0)
                  + 0.25 * (1.0 if avg_sent <= 16 else 0.0))
    return {
        "personalization": personal,
        "pushiness": pushy,
        "ask_directness": ask_directness,
        "length_fit": length_fit,
        "clarity": clarity,
        "credential_signaling": _sat(len(_CREDENTIAL.findall(t))),
        "contrarian_pitch": _sat(len(_CONTRARIAN.findall(t))),
        "secret_density": _sat(len(_SECRET.findall(t))),
    }


# --- the offline proposer: a sentence bank per slot spanning the strategy space --------------------
# Deliberately includes GOOD and BAD candidates (pushy, credential-heavy, vague) so the winning email is
# EARNED by the scorer, not rigged by the bank. A real LLM proposer (propose_fn) drops in here.
SLOT_BANK: dict = {
    "opener": [
        "Peter — I read your essay on secrets and definite optimism.",
        "Peter — loved your take on how the best companies are built on a contrarian truth.",
        "Hi Mr. Thiel, I hope this email finds you well and that you are having a great week.",
        "Dear Mr. Thiel, I am a Princeton admit recently featured in the New York Times.",
        "Peter —",
        "Hi Peter, quick note.",
    ],
    "hook": [
        "I'm 17 and I think going to college is the wrong trade right now.",
        "I'm 17, building in AI infrastructure instead of taking the safe path.",
        "I've been building since before it was resume material.",
        "I was valedictorian and have a 4.0, and I've won several prestigious awards.",
        "",  # allow skipping the hook (brevity)
    ],
    "thesis": [
        "The secret I'm betting on: most of the AI stack rents margin it should own, and inference is where that flips.",
        "Everyone thinks the moat is the model; I think the conventional wisdom is wrong and it's the inference layer.",
        "I'm building AI infrastructure that makes inference dramatically cheaper.",
        "My startup is an exciting, innovative, next-generation AI platform with huge potential.",
        "The non-obvious truth: the winners will own the layer everyone currently rents.",
    ],
    "ask": [
        "Is that thesis obviously wrong to you? One line back and I'll leave you alone.",
        "Would you tell me the fastest way this is wrong?",
        "Could we set up a 30-minute call at your earliest convenience?",
        "Please respond ASAP — I'd love to get on your calendar this week.",
        "I'd love any thoughts you might have whenever you get a chance.",
    ],
    "close": [
        "— Beckett",
        "Thanks for your time, Beckett.",
        "Best regards and looking forward to hearing back from you soon, Beckett.",
        "",  # allow no close
    ],
}
SLOTS = ["opener", "hook", "thesis", "ask", "close"]


def default_proposer(slot: str, spec_strategy: dict, context: dict, k: int = 8) -> list[str]:
    return list(SLOT_BANK.get(slot, [""]))[:k]


def _spec_distance(strat: dict, spec_strategy: dict) -> float:
    return sum(abs(strat.get(v, 0.0) - spec_strategy.get(v, spec(v).default)) for v in MESSAGE_VARS) / len(MESSAGE_VARS)


@dataclass
class ConstructedEmail:
    text: str
    strategy: dict
    score: float                # objective value (lower bound − spec penalty) at selection
    mean: float                 # scorer mean P(reply) for the assembled text
    lower_bound: float
    slots: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"text": self.text,
                "encoded_strategy": {k: round(v, 3) for k, v in self.strategy.items()},
                "reply_mean": round(self.mean, 4), "reply_lower_bound": round(self.lower_bound, 4)}


def construct_email(scorer: StrategyScorer, spec_strategy: dict, *, proposer=default_proposer,
                    beam: int = 6, q: float = 0.2, spec_penalty: float = 0.15,
                    context: dict | None = None) -> ConstructedEmail:
    """Beam search over communicative moves. Returns the assembled email the world model scores highest
    while adhering to the Layer-1 optimal strategy. The email is CONSTRUCTED by the search, not written."""
    context = context or {}

    def objective(text):
        strat = encode_text_to_strategy(text)
        return scorer.lower_bound(strat, q=q) - spec_penalty * _spec_distance(strat, spec_strategy)

    # beams: list of (chosen_slots dict, text, score)
    beams = [({}, "", -1.0)]
    for slot in SLOTS:
        cands = proposer(slot, spec_strategy, context)
        scored = []
        for chosen, text, _ in beams:
            for c in cands:
                nt = (text + " " + c).strip() if c else text
                if not nt:
                    continue
                nchosen = {**chosen, slot: c}
                scored.append((nchosen, nt, objective(nt)))
        scored.sort(key=lambda x: x[2], reverse=True)
        beams = scored[:beam] if scored else beams

    chosen, text, score = beams[0]
    dist = scorer.score_dist(encode_text_to_strategy(text))
    return ConstructedEmail(text=text, strategy=encode_text_to_strategy(text), score=score,
                            mean=dist.mean, lower_bound=dist.lower_bound(q), slots=chosen)
