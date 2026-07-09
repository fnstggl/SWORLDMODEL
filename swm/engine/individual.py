"""Individual mode — N sampled runs of ONE grounded person reacting to ONE exact stimulus.

"If I send this cold email to Peter Thiel, does he reply?" is answered the only honest way a world model
can: ground WHO the person is (real retrieval — bio, current focus, recent news, known dispositions), draw
K plausible LATENT STATES for them right now (busy/attentive, skeptical/curious, who has their ear — the
unobservables that actually decide cold-email fate), then run the moment N times: in each run the person —
as the LLM reasoning from the dossier + one latent state, at temperature — READS THE EXACT MESSAGE and
decides. P(response) is the weighted fraction of runs that respond, with a real sampling interval, and the
audit says WHY (the reasons that recurred).

Two hard rules (the anti-regression clauses):
  - NEVER the base rate. There is no fallback to a global reply probability. Every number this module
    emits is conditioned on THIS person and THIS message. If the person cannot be grounded, it ABSTAINS
    loudly instead.
  - The stimulus is the ACTUAL artifact. The engine reads the real email text — no "personalization=0.7"
    feature vector between the message and the model.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from swm.engine.grounding import parse_json
from swm.engine.retrieval import multi_search


@dataclass
class IndividualForecast:
    person: str
    p_response: float = None
    interval_80: list = None                    # sampling interval on p across the N runs
    n_runs: int = 0
    per_state: list = field(default_factory=list)   # [{state, p, n}] — where the probability comes from
    reasons: list = field(default_factory=list)     # recurring reasons (the WHY, quoted from runs)
    grounding: dict = field(default_factory=dict)
    abstain: bool = False
    abstain_reason: str = ""

    def as_dict(self):
        return {"person": self.person, "p_response": self.p_response, "interval_80": self.interval_80,
                "n_runs": self.n_runs, "per_state": self.per_state, "reasons": self.reasons[:6],
                "abstain": self.abstain, "abstain_reason": self.abstain_reason,
                "grounding": self.grounding}


_STATES_PROMPT = """We are simulating how a real person reacts to one specific inbound message.
PERSON: {name}
GROUNDED DOSSIER (from live retrieval — the only facts you may rely on):
{dossier}

Draw {k} plausible LATENT STATES for this person right now — the unobservables that decide whether a cold
message gets a reply: attention/busyness, current priorities, disposition toward unsolicited outreach of
this kind, gatekeeping (who reads their inbox). States must fit the grounded facts and DIFFER meaningfully.
Weight them by plausibility (sum to 1).
Return ONLY JSON: {{"states": [{{"state": "<2 sentences>", "weight": <0..1>}}]}}"""

_READ_PROMPT = """You are {name}. Here is who you are (grounded from live evidence):
{dossier}
Your state right now: {state}

This message just reached you ({channel}):
---
{message}
---

Be this person, in this state, with their inbox and their incentives. Do you actually respond to it —
not "would it be reasonable to", but do YOU, today? Decide.
Return ONLY JSON: {{"decision": "respond" | "no_response", "why": "<this person's actual reason, one
sentence>"}}"""


def ground_person(llm, name, context="") -> tuple:
    """Retrieve who this person is right now. Returns (dossier_text, report). Empty dossier => abstain."""
    queries = [f"{name}", f"{name} recent news", f"{name} {context}".strip()]
    passages = multi_search([q for q in queries if q.strip()], 6)
    if len(passages) < 2:
        return "", {"n_passages": len(passages), "grounded": False}
    ptxt = "\n".join(p.cite() for p in passages[:24])
    raw = parse_json(llm(
        f"From ONLY these passages, write a grounded dossier of {name} for a behavioral simulation: who "
        f"they are, current role/focus, public disposition (how they engage with strangers/pitches), and "
        f"anything recent that would affect their attention. Mark anything not in the passages as unknown."
        f"\n\nPASSAGES:\n{ptxt}\n\nReturn ONLY JSON: {{\"dossier\": \"<the dossier, <=10 lines>\", "
        f"\"identified\": <true if the passages clearly identify this specific person, else false>}}")) or {}
    if not raw.get("identified") or not raw.get("dossier"):
        return "", {"n_passages": len(passages), "grounded": False}
    return str(raw["dossier"]), {"n_passages": len(passages), "grounded": True,
                                 "sources": sorted({p.source.split(":")[0] for p in passages})}


@dataclass
class IndividualSimulator:
    llm_hot: object                       # decision backend, temperature > 0 (the N runs must differ)
    llm: object = None                    # cold backend for grounding + latent states
    k_states: int = 4
    reps_per_state: int = 6
    max_workers: int = 8

    def simulate(self, person: str, message: str, *, channel: str = "email",
                 context: str = "") -> IndividualForecast:
        llm_cold = self.llm or self.llm_hot
        dossier, greport = ground_person(llm_cold, person, context)
        if not dossier:
            return IndividualForecast(
                person=person, abstain=True, grounding=greport,
                abstain_reason=(f"CANNOT GROUND WHO '{person}' IS from live retrieval "
                                f"({greport.get('n_passages', 0)} passages). Refusing to emit a number: "
                                f"an individual-response prediction without the individual is just a base "
                                f"rate, and base rates are banned here. Provide context or a fuller name."))

        raw = parse_json(llm_cold(_STATES_PROMPT.format(name=person, dossier=dossier,
                                                        k=self.k_states))) or {}
        states = [(str(s.get("state", ""))[:400], max(0.0, float(s.get("weight", 0) or 0)))
                  for s in raw.get("states", []) if isinstance(s, dict) and s.get("state")]
        states = states[: self.k_states] or [("ordinary day, normal inbox load", 1.0)]
        z = sum(w for _, w in states) or 1.0
        states = [(s, w / z) for s, w in states]

        def one_run(state):
            r = parse_json(self.llm_hot(_READ_PROMPT.format(
                name=person, dossier=dossier, state=state, channel=channel, message=message[:2500])))
            if not r or r.get("decision") not in ("respond", "no_response"):
                return None
            return (1 if r["decision"] == "respond" else 0, str(r.get("why", ""))[:200])

        jobs = [(si, states[si][0]) for si in range(len(states)) for _ in range(self.reps_per_state)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            outcomes = list(ex.map(lambda j: one_run(j[1]), jobs))

        per_state, reasons, p_mix, n_total = [], [], 0.0, 0
        for si, (state, w) in enumerate(states):
            runs = [o for j, o in zip(jobs, outcomes) if j[0] == si and o is not None]
            if not runs:
                continue
            yes = sum(y for y, _ in runs)
            p_s = (yes + 0.5) / (len(runs) + 1.0)            # Laplace-smoothed per-state rate
            per_state.append({"state": state[:120], "weight": round(w, 3),
                              "p": round(p_s, 3), "n": len(runs)})
            p_mix += w * p_s
            n_total += len(runs)
            reasons += [why for _, why in runs if why]
        if n_total == 0:
            return IndividualForecast(person=person, abstain=True, grounding=greport,
                                      abstain_reason="all decision runs failed to parse — no forecast")

        se = math.sqrt(max(p_mix * (1 - p_mix), 1e-4) / n_total)
        seen, top_reasons = set(), []
        for r in reasons:                                     # keep distinct recurring reasons
            k = r.lower()[:40]
            if k not in seen:
                seen.add(k)
                top_reasons.append(r)
        return IndividualForecast(
            person=person, p_response=round(p_mix, 4), n_runs=n_total,
            interval_80=[round(max(0.0, p_mix - 1.28 * se), 4), round(min(1.0, p_mix + 1.28 * se), 4)],
            per_state=per_state, reasons=top_reasons, grounding=greport)
