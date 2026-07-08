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
        "Peter, I read your essay on secrets.",
        "Peter, you've written that the best startups are built on a truth few people agree with.",
        "Hi Mr. Thiel, I hope this email finds you well.",                              # annoying (contrast)
        "Dear Mr. Thiel, I'm a Princeton admit recently featured in the New York Times.",  # credential (contrast)
        "Peter,",
    ],
    "hook": [
        "I'm 17. I got into Princeton and I don't think I should go.",
        "I'm 17 and building in AI instead of going to college.",
        "I was valedictorian with a 4.0 and several awards.",                          # credential (contrast)
        "",  # allow skipping the hook (brevity)
    ],
    "thesis": [
        "I build software that cuts the cost of running large AI models by about 40%.",
        "The expensive part of AI is shifting from training models to running them, and almost nobody is building for that.",
        "Most companies rent their AI compute; I think the ones that own it will win, and I'm building that.",
        "The secret I'm betting on: most of the AI stack rents margin it should own, and inference is where that flips.",  # slop (contrast)
        "My startup is an exciting next-generation AI platform with huge potential.",  # vague slop (contrast)
    ],
    "ask": [
        "Do you think that's wrong?",
        "Would you tell me the fastest way this falls apart?",
        "Is this a bad idea?",
        "Could we set up a 30-minute call at your earliest convenience?",             # pushy (contrast)
        "Is that thesis obviously wrong to you? One line back and I'll leave you alone.",  # annoying (contrast)
    ],
    "close": [
        "Beckett",
        "— Beckett",                                                                 # sign-off dash is fine
        "Thanks, Beckett.",
        "Best regards and looking forward to hearing back from you soon, Beckett.",   # annoying (contrast)
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
    score: float                # objective value (lower bound − spec penalty − slop) at selection
    mean: float                 # scorer mean P(reply) for the assembled text
    lower_bound: float
    slots: dict = field(default_factory=dict)
    critique: object = None     # SemanticCritic verdict on the final text

    def summary(self) -> dict:
        out = {"text": self.text,
               "encoded_strategy": {k: round(v, 3) for k, v in self.strategy.items()},
               "reply_mean": round(self.mean, 4), "reply_lower_bound": round(self.lower_bound, 4)}
        if self.critique is not None:
            out["critique"] = self.critique.summary()
        return out


def construct_email(scorer: StrategyScorer, spec_strategy: dict, *, proposer=default_proposer,
                    beam: int = 6, q: float = 0.2, spec_penalty: float = 0.15,
                    critic=None, critic_weight: float = 0.6, context: dict | None = None) -> ConstructedEmail:
    """Beam search over communicative moves. Returns the assembled email the world model scores highest
    while adhering to the Layer-1 optimal strategy AND passing the semantic critic (coherent, not annoying).
    The email is CONSTRUCTED by the search, not written. `critic` should be the CHEAP lexical
    SemanticCritic (no judge_fn) — it prunes slop as the email is built; the LLM critic runs later as a gate."""
    context = context or {}
    if critic is None:
        from swm.decision.semantic_critic import SemanticCritic
        critic = SemanticCritic()          # lexical, cheap — safe to call on every partial assembly

    def objective(text):
        strat = encode_text_to_strategy(text)
        base = scorer.lower_bound(strat, q=q) - spec_penalty * _spec_distance(strat, spec_strategy)
        # subtract a slop penalty: a partial assembly with an incoherent/annoying line is pruned even if
        # its lexical strategy scores well. This is the quality axis the variable readout is blind to.
        return base - critic_weight * (1.0 - critic.critique(text).quality)

    # beams: list of (chosen_slots dict, text, score). Candidates are proposed PER BEAM with the beam's
    # prefix as context, so a context-aware (LLM) proposer continues each draft coherently and does not
    # repeat what earlier slots already said. An offline bank ignores the prefix (cheap, unchanged result).
    beams = [({}, "", -1.0)]
    for slot in SLOTS:
        scored = []
        for chosen, text, _ in beams:
            cands = proposer(slot, spec_strategy, {**context, "prefix": text, "slot": slot})
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
                            mean=dist.mean, lower_bound=dist.lower_bound(q), slots=chosen,
                            critique=critic.critique(text))


def _assemble(slots: dict) -> str:
    return " ".join(slots[s] for s in SLOTS if slots.get(s)).strip()


def _prefix_before(slots: dict, target: str) -> str:
    """The assembled text of all slots that come before `target` — the prefix a context-aware proposer
    should continue from during repair."""
    out = []
    for s in SLOTS:
        if s == target:
            break
        if slots.get(s):
            out.append(slots[s])
    return " ".join(out).strip()


def _sent_overlap(a: str, b: str) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    return a in b or b in a


def polish_email(email: ConstructedEmail, scorer: StrategyScorer, spec_strategy: dict, *,
                 proposer=default_proposer, critic, q: float = 0.2, rounds: int = 3,
                 spec_penalty: float = 0.15, critic_weight: float = 0.8,
                 rank_critic=None, rewrite_fn=None) -> ConstructedEmail:
    """The critical evaluator AT THE END. Run the (LLM, if available) critic on the finalist; for every
    slot whose sentence the critic flags as incoherent/embellished or annoying, swap in the best
    alternative move that PASSES the critic and keeps the strategy — then re-critique. Repeat up to
    `rounds`. Bounds expensive-critic calls to (flagged slots × candidates), not the whole beam tree.
    If no clean realization exists in the proposer's candidates, the least-slop version is returned with
    its flags surfaced — honest, not hidden."""
    if rank_critic is None:
        from swm.decision.semantic_critic import SemanticCritic
        rank_critic = SemanticCritic()          # cheap lexical — ranks replacements without LLM calls
    slots = dict(email.slots)
    text = _assemble(slots)

    def strat_value(t):
        strat = encode_text_to_strategy(t)
        return scorer.lower_bound(strat, q=q) - spec_penalty * _spec_distance(strat, spec_strategy)

    for _ in range(rounds):
        crit = critic.critique(text)                    # the (possibly LLM) gate finds WHAT'S wrong
        flagged = {f["sentence"] for f in crit.flags()}
        if not flagged:
            break
        changed = False
        reason_by_sent = {f["sentence"]: f["reasons"] for f in crit.flags()}
        for slot, chosen in list(slots.items()):
            if not chosen or not any(_sent_overlap(fs, chosen) for fs in flagged):
                continue
            # Candidate replacements. With a rewrite_fn (live LLM), the focused fix is a TARGETED REWRITE
            # of the flagged line fed the critic's reason — resampling fresh proposer options just returns
            # the same slop register. Without it, fall back to fresh proposer candidates.
            if rewrite_fn is not None:
                reasons = next((r for fs, r in reason_by_sent.items() if _sent_overlap(fs, chosen)), [])
                candidates = [rewrite_fn(chosen, reasons, spec_strategy)]
            else:
                candidates = list(proposer(slot, spec_strategy, {"prefix": _prefix_before(slots, slot)}))
            # Rank lexicographically: (candidate cleanliness — fixes independent slop unmasked; then
            # whole-trial quality — fixes redundancy; then strategy value). Cheap lexical critic ranks.
            def rank(c):
                trial = _assemble({**slots, slot: c})
                cand_clean = rank_critic.critique(c).quality if c else 1.0
                trial_q = rank_critic.critique(trial).quality if trial else 0.0
                return (round(cand_clean, 2), round(trial_q, 2), strat_value(trial) if trial else -1.0)
            best_c = max(candidates + [chosen], key=rank)
            if best_c != chosen and rank(best_c) > rank(chosen):
                slots[slot] = best_c
                text = _assemble(slots)
                changed = True
        if not changed:
            break

    strat = encode_text_to_strategy(text)
    dist = scorer.score_dist(strat)
    return ConstructedEmail(text=text, strategy=strat, score=dist.lower_bound(q), mean=dist.mean,
                            lower_bound=dist.lower_bound(q), slots=slots, critique=critic.critique(text))
