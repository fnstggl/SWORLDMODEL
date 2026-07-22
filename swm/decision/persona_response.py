"""QUALITATIVE PERSONA RESPONSE — simulate what an individual actually DOES with an inbound message.

The circularity this replaces: the old path had an LLM invent numeric personality traits, an LLM
write to those traits, an LLM encode messages back into those traits, and a formula over those
traits declare a winner — the system optimized against a mathematical caricature it created itself.

This module applies the repo's qualitative-actor discipline (swm/world_model_v2/qualitative_actor.py,
merged as the primary decision mechanism for consequential humans) to the RECIPIENT of outreach:

  * QUALITATIVE COGNITION ONLY — the persona prompt carries the dossier as TEXT (beliefs,
    incentives, predispositions, evidence quotes), never invented numeric trait values.
  * COMPETING INBOX-REALITY HYPOTHESES — there is no single "Peter model". Several plausible
    hidden realities (reads own inbound in bursts / assistant screens everything / responds only
    through trusted introductions / evidence-first / ignores all cold contact) each get their own
    simulations; a candidate that wins under only one hypothesis is fragile and must be reported
    as such, not selected confidently.
  * THE LLM CHOOSES ONE OUTCOME PER DRAW — first person, categorical menu + the actual reply text
    if any. Probabilities come from COUNTING outcomes across draws x hypotheses, never from asking
    a model for a probability.
  * VALENCED OUTCOME VECTOR — {no_response, dismissive_reply, curious_reply, requests_material,
    refers_to_other, meeting_offer} plus a reputational-harm flag. "Any reply" is not the goal;
    an irritated correction is a cost.
  * HONESTY — the aggregate is labeled model_based_judgment (LLM role-play, uncalibrated). It is a
    structured judgment with explicit hypothesis sensitivity — NOT a calibrated response
    probability. Calibration requires real outreach outcomes (the ledger's job, over time).

UNIVERSAL: `PersonaDossier.for_public_figure` builds the dossier from the resolver's web evidence
(qualitative snippets, not the inferred numeric variables); `PersonaDossier.from_user_context`
serves private individuals from caller-supplied context. Same engine for any recipient.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- outcome vector (valenced)
OUTCOMES = ("no_response", "dismissive_reply", "curious_reply", "requests_material",
            "refers_to_other", "meeting_offer")

#: default utilities per outcome — the DECISION-MAKER's values, overridable per call. A dismissive
#: reply is worse than silence (reputational cost with this recipient); material-request is the
#: natural success of a permission-ask; meeting dominates.
DEFAULT_OUTCOME_UTILITIES = {"no_response": 0.0, "dismissive_reply": -0.3, "curious_reply": 0.5,
                             "requests_material": 1.0, "refers_to_other": 0.8,
                             "meeting_offer": 1.5}


# ---------------------------------------------------------------- the dossier (qualitative, universal)
@dataclass
class PersonaDossier:
    """Everything qualitatively known about the recipient — text, quotes, provenance. NO invented
    numbers. `evidence` rows are (source_label, text)."""
    name: str
    role: str = ""
    evidence: list = field(default_factory=list)         # [(source, text)]
    user_context: str = ""                               # caller-supplied qualitative context
    as_of: str = ""

    def render(self, max_chars: int = 2600) -> str:
        rows = [f"You are {self.name}" + (f" ({self.role})." if self.role else ".")]
        if self.user_context:
            rows.append(f"Context supplied about you: {self.user_context}")
        for src, text in self.evidence:
            rows.append(f"- [{src}] {text}")
        out = "\n".join(rows)
        return out[:max_chars]

    @classmethod
    def for_public_figure(cls, world, contact_id: str, *, name: str, role: str = "",
                          extra_context: str = "") -> "PersonaDossier":
        """Build from the World's resolver evidence (the qualitative snippets the web search
        returned — NOT the numeric inferred_variables)."""
        ev = []
        try:
            prof = world.profile(contact_id) or {}
            for e in (prof.get("evidence") or [])[:10]:
                if isinstance(e, dict):
                    ev.append((str(e.get("title", "web"))[:60], str(e.get("snippet", ""))[:300]))
        except Exception:  # noqa: BLE001
            pass
        return cls(name=name, role=role, evidence=ev, user_context=extra_context)

    @classmethod
    def from_user_context(cls, name: str, context: str, *, role: str = "") -> "PersonaDossier":
        """Private individuals: the user supplies what they know; no web research required."""
        return cls(name=name, role=role, user_context=context)


# ---------------------------------------------------------------- inbox-reality hypotheses
#: The generic competing hypotheses about how inbound actually reaches a busy, high-status person.
#: Priors are honest base rates for very-high-status recipients: most probability mass on screening
#: and ignoring, NOT on 'reads and engages'. Specialize per recipient via `specialize_hypotheses`.
GENERIC_INBOX_HYPOTHESES = [
    {"id": "reads_own_bursts", "prior": 0.15,
     "reality": "You personally skim your own inbound in short bursts between meetings. You open "
                "maybe a third of cold messages, read a few lines, and reply to almost none — but "
                "a genuinely interesting, immediately-legible note from a real person sometimes "
                "gets a one-line answer."},
    {"id": "assistant_screens", "prior": 0.35,
     "reality": "An assistant screens everything. Cold messages reach you only if the assistant "
                "flags them as clearly relevant and credible; the assistant forwards perhaps one "
                "in fifty, with a one-line summary. You see nothing else."},
    {"id": "intros_only", "prior": 0.25,
     "reality": "In practice you only engage inbound that arrives through people you trust — a "
                "portfolio founder, a partner, a friend forwarding with a vouch. Direct cold "
                "contact, however good, effectively never gets a response from you."},
    {"id": "evidence_first", "prior": 0.15,
     "reality": "You ignore claims and read artifacts. A message pointing at something concrete "
                "and checkable (a memo, a result, a running system) occasionally earns a short "
                "reply asking for the artifact; rhetoric alone never does."},
    {"id": "ignores_all_cold", "prior": 0.10,
     "reality": "You simply do not respond to unsolicited contact from strangers, ever. The only "
                "path to your attention is through your existing network."},
]


def specialize_hypotheses(chat_fn, dossier: PersonaDossier, *, k: int = 5) -> list:
    """Optionally specialize the generic hypothesis set to the recipient from their evidence
    (still qualitative; still competing; priors must stay a distribution). Falls back to the
    generic set on any failure — the generic set is honest, not a degenerate default."""
    if chat_fn is None:
        return [dict(h) for h in GENERIC_INBOX_HYPOTHESES]
    prompt = (
        f"{dossier.render()}\n\n"
        "Task: propose the competing HYPOTHESES about how this person actually handles UNSOLICITED "
        "inbound from strangers (their real inbox behavior — screening, delegation, what if "
        "anything they personally read and answer). This is about private behavior, which public "
        "evidence only weakly constrains — so the hypotheses must genuinely disagree, and priors "
        "must reflect honest base rates for someone this busy (most cold messages get no reply). "
        f"Return ONLY a JSON array of exactly {k} objects: "
        '{"id": "snake_case", "prior": 0..1, "reality": "second-person paragraph: You ..."} '
        "with priors summing to 1.")
    try:
        raw = chat_fn(prompt, max_tokens=1200) if _accepts_kwargs(chat_fn) else chat_fn(prompt)
        m = re.search(r"\[.*\]", raw, re.S)
        rows = json.loads(m.group(0)) if m else []
        rows = [r for r in rows if isinstance(r, dict) and r.get("id") and r.get("reality")]
        z = sum(max(0.0, float(r.get("prior", 0))) for r in rows)
        if len(rows) >= 3 and z > 0:
            for r in rows:
                r["prior"] = max(0.0, float(r.get("prior", 0))) / z
            return rows
    except Exception:  # noqa: BLE001
        pass
    return [dict(h) for h in GENERIC_INBOX_HYPOTHESES]


def _accepts_kwargs(fn) -> bool:
    try:
        import inspect
        sig = inspect.signature(fn)
        return any(p.kind == p.VAR_KEYWORD or p.name == "max_tokens"
                   for p in sig.parameters.values())
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- the persona simulation call
@dataclass
class PersonaResponse:
    outcome: str
    reply_text: str = ""
    reasoning: str = ""
    hypothesis_id: str = ""
    raw_available: bool = True

    def as_dict(self):
        return {"outcome": self.outcome, "reply_text": self.reply_text,
                "reasoning": self.reasoning, "hypothesis_id": self.hypothesis_id}


def simulate_response(chat_fn, dossier: PersonaDossier, hypothesis: dict, message: str, *,
                      channel: str = "email", arrival_context: str = "",
                      temperature: float = 0.7, seed_note: str = "") -> PersonaResponse:
    """ONE first-person draw: the recipient, under one inbox-reality hypothesis, receives the
    message and does something. Qualitative in, categorical-outcome out. The prompt carries the
    dossier as text — beliefs, incentives, predispositions — never numeric trait values."""
    arrival = arrival_context or f"A cold {channel} from a stranger arrives in your inbox."
    prompt = (
        f"{dossier.render()}\n\n"
        f"The reality of how you handle inbound today:\n{hypothesis.get('reality', '')}\n\n"
        f"{arrival}\n"
        f"--- MESSAGE ---\n{message}\n--- END ---\n\n"
        "Inhabit this person completely: their beliefs, incentives, schedule, and how they "
        "actually treat strangers' messages — not their public persona's style. Decide what you "
        "ACTUALLY do with this message, as the real person would on an ordinary busy day"
        + (f" ({seed_note})" if seed_note else "") + ".\n"
        'Return ONLY JSON: {"outcome": one of ["no_response","dismissive_reply","curious_reply",'
        '"requests_material","refers_to_other","meeting_offer"], '
        '"reply_text": "the exact reply you send, empty if none", '
        '"reasoning": "one private sentence on why"}')
    try:
        raw = (chat_fn(prompt, max_tokens=320, temperature=temperature)
               if _accepts_kwargs(chat_fn) else chat_fn(prompt))
        m = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(m.group(0)) if m else {}
        outcome = str(obj.get("outcome", "")).strip()
        if outcome not in OUTCOMES:
            return PersonaResponse(outcome="no_response", reasoning="unparseable draw -> counted "
                                   "as no_response (fail-closed)", hypothesis_id=hypothesis.get("id", ""),
                                   raw_available=False)
        return PersonaResponse(outcome=outcome, reply_text=str(obj.get("reply_text", ""))[:400],
                               reasoning=str(obj.get("reasoning", ""))[:200],
                               hypothesis_id=hypothesis.get("id", ""))
    except Exception:  # noqa: BLE001
        return PersonaResponse(outcome="no_response", reasoning="persona call failed -> counted as "
                               "no_response (fail-closed)", hypothesis_id=hypothesis.get("id", ""),
                               raw_available=False)


# ---------------------------------------------------------------- ensemble evaluation
@dataclass
class PersonaEnsembleResult:
    """Counted outcomes across hypotheses x draws, with the decision-relevant aggregates. The label
    is honest: model_based_judgment (uncalibrated LLM role-play), reported with per-hypothesis
    breakdown so fragility is visible, never averaged away."""
    counts: dict = field(default_factory=dict)           # hypothesis_id -> {outcome: n}
    priors: dict = field(default_factory=dict)           # hypothesis_id -> prior
    n_draws: int = 0
    sample_replies: list = field(default_factory=list)   # a few verbatim in-character replies
    failures: int = 0

    def outcome_dist(self, hypothesis_id: str = None) -> dict:
        if hypothesis_id is not None:
            c = self.counts.get(hypothesis_id, {})
            n = sum(c.values()) or 1
            return {o: c.get(o, 0) / n for o in OUTCOMES}
        agg = {o: 0.0 for o in OUTCOMES}
        for hid, c in self.counts.items():
            n = sum(c.values()) or 1
            w = self.priors.get(hid, 1.0 / max(1, len(self.counts)))
            for o in OUTCOMES:
                agg[o] += w * c.get(o, 0) / n
        return agg

    def expected_utility(self, utilities: dict = None, hypothesis_id: str = None) -> float:
        u = utilities or DEFAULT_OUTCOME_UTILITIES
        return sum(p * u.get(o, 0.0) for o, p in self.outcome_dist(hypothesis_id).items())

    def by_hypothesis_utility(self, utilities: dict = None) -> dict:
        return {hid: round(self.expected_utility(utilities, hid), 4) for hid in self.counts}

    def summary(self, utilities: dict = None) -> dict:
        return {"label": "model_based_judgment (LLM persona role-play; UNCALIBRATED — not a "
                         "measured response probability)",
                "outcome_dist": {o: round(p, 4) for o, p in self.outcome_dist().items()},
                "expected_utility": round(self.expected_utility(utilities), 4),
                "by_hypothesis_utility": self.by_hypothesis_utility(utilities),
                "n_draws": self.n_draws, "n_failed_draws": self.failures,
                "sample_replies": self.sample_replies[:3]}


def ensemble_evaluate(chat_fn, dossier: PersonaDossier, hypotheses: list, message: str, *,
                      channel: str = "email", arrival_context: str = "", draws_per_hypothesis: int = 3,
                      temperature: float = 0.7) -> PersonaEnsembleResult:
    """Run draws_per_hypothesis first-person simulations under EVERY hypothesis; count outcomes.
    The distribution is empirical over draws — no model call ever returns a probability."""
    res = PersonaEnsembleResult(priors={h["id"]: float(h.get("prior", 0.2)) for h in hypotheses})
    for h in hypotheses:
        c = res.counts.setdefault(h["id"], {})
        for j in range(draws_per_hypothesis):
            r = simulate_response(chat_fn, dossier, h, message, channel=channel,
                                  arrival_context=arrival_context, temperature=temperature,
                                  seed_note=f"draw {j + 1}")
            c[r.outcome] = c.get(r.outcome, 0) + 1
            res.n_draws += 1
            if not r.raw_available:
                res.failures += 1
            elif r.reply_text and len(res.sample_replies) < 6:
                res.sample_replies.append({"hypothesis": h["id"], "outcome": r.outcome,
                                           "reply": r.reply_text})
    return res


def fragility_report(results_by_arm: dict, utilities: dict = None) -> dict:
    """Across evaluated arms: who wins overall, under WHICH hypotheses, and whether the overall
    winner depends on a single hypothesis (the confidence-killer the critique demanded). Also says
    when top arms are within counting noise of each other."""
    u = utilities or DEFAULT_OUTCOME_UTILITIES
    overall = {a: r.expected_utility(u) for a, r in results_by_arm.items()}
    if not overall:
        return {}
    winner = max(overall, key=overall.get)
    hyp_ids = set()
    for r in results_by_arm.values():
        hyp_ids |= set(r.counts)
    per_h_winners = {}
    for hid in sorted(hyp_ids):
        per_h_winners[hid] = max(results_by_arm,
                                 key=lambda a: results_by_arm[a].expected_utility(u, hid))
    wins_under = [hid for hid, w in per_h_winners.items() if w == winner]
    n = next(iter(results_by_arm.values())).n_draws or 1
    noise = 1.5 / (n ** 0.5)                              # rough counting-noise scale on EU
    within = [a for a, v in overall.items()
              if a != winner and overall[winner] - v <= noise]
    return {"overall_utility": {a: round(v, 4) for a, v in sorted(overall.items(),
                                                                  key=lambda kv: -kv[1])},
            "winner": winner, "winner_wins_under_hypotheses": wins_under,
            "per_hypothesis_winner": per_h_winners,
            "fragile": len(wins_under) <= 1 and len(hyp_ids) > 1,
            "within_noise_of_winner": within,
            "note": "a winner that leads under only one inbox-reality hypothesis is FRAGILE; "
                    "within_noise arms are indistinguishable at this draw count — the system "
                    "reports 'best-supported among tested', never 'best possible'."}
