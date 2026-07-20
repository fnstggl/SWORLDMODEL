"""Phase 1.5 — the SCHEDULED-REALITY layer: dated public facts are FACTS, not distributions.

The Powell forensic trace exposed the defect class: a question dominated by a public calendar fact (a term
that ends on a known date) was simulated as a broad-prior draw. This layer extracts DATED FACTS from the
question + evidence + the model's world knowledge (allowed and REQUIRED under the identity-preserving
fidelity directive; every fact carries a provenance label and, where possible, an evidence citation), and
executes them as DETERMINISTIC events in the shared world:

  term expiries · scheduled votes/meetings/hearings · filing/response deadlines · launch/release dates ·
  election dates · contract/mandate end dates

Each fact becomes a `scheduled_fact` event; `ScheduledFactOperator` applies it at its date as a typed
StateDelta (entity field or quantity). A fact whose occurrence DIRECTLY entails the outcome (e.g. "the term
ends on May 15" for an out-by-May-15 question) is marked `outcome_entailing` with a direction, and the
aggregate/institutional outcome mechanism consumes it at high weight — the remaining uncertainty is only
whether the entailment holds (renewal, early exit), which stays with the strategic layer.

Provenance honesty: facts sourced from evidence carry the claim id; facts from model knowledge are labeled
`model_knowledge` (measurable by the name-only leakage probes — fidelity is chosen over blinding by
explicit direction, and the leakage risk is MEASURED, not hidden).
"""
from __future__ import annotations

import time

from swm.world_model_v2.state import F, parse_time
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)

_EXTRACT_PROMPT = """You are the SCHEDULED-REALITY extractor for a world simulation. List every DATED PUBLIC
FACT relevant to this question: term/mandate end dates, scheduled votes, meetings, hearings, deadlines,
elections, expirations, launches — AND recurring institutional patterns (annual conferences, regular
meetings, election cycles, earnings/filing calendars, expected annual product/software cycles). Use the
evidence AND your world knowledge of the real named people and institutions. Only include facts you are
confident are real, with dates as YYYY-MM-DD (approximate day is acceptable if the month is certain).

CRUCIAL — judge each fact's INFLUENCE ON THE OUTCOME, not just "direct entailment". A recurring pattern
counts: if a question asks whether an event that reliably happens on a schedule will happen (e.g. a company
that ships a new OS version at its annual conference every year, an incumbent body that meets and decides on
a fixed calendar), the pattern RAISES the probability strongly even though the specific occurrence is not
logically guaranteed. Weigh confirmed schedules and strong recurrences heavily; weigh one-off speculation
lightly. A strong recurrence heavily informs the forecast but is NOT certainty — genuine disrupting evidence
(cancellation, abandonment, a broken streak) can lower it.

RECURRING EVENTS — CHECK THIS EXPLICITLY. The world runs on calendars: central banks meet on fixed
schedules, companies hold annual conferences and ship annual OS/product releases, elections follow fixed
cycles, courts have terms, leagues have seasons. If any institution, company or body named in the question
has a RECURRING pattern relevant to the outcome (e.g. "Apple announces a new visionOS at WWDC every June",
"the BoJ policy board meets 8 times a year on published dates"), you MUST list the NEXT expected instance
inside the window as a dated fact with kind "recurring_event" and fill "recurrence" with the cadence and
its PAST instances (all strictly before {as_of}). A long unbroken pattern (5+ consecutive instances)
justifies confidence 0.9+; a question asking whether the next instance of an unbroken annual pattern will
happen is usually outcome-entailing. Base-rate structure like this must never be lost to the simulation.

QUESTION: {q}
AS-OF: {as_of}   HORIZON: {horizon}
EVIDENCE:
{ev}

For EACH fact give:
- pattern_strength: "confirmed_scheduled" (a specific dated event is officially set) |
  "strong_recurrence" (happens almost every cycle — e.g. annual OS at the annual conference) |
  "base_rate" (ordinary historical frequency) | "speculative" (a guess/rumor).
- outcome_influence: "raises" | "lowers" | "neutral" — does this fact make the question's YES more/less likely?
- influence_strength: 0..1 — how strongly it moves P(YES). confirmed_scheduled decisive ~0.9;
  strong_recurrence ~0.7-0.85; base_rate ~0.3-0.6; speculative ~0.1-0.3. Never 1.0 unless YES is logically entailed.

Return ONLY JSON:
{{"facts": [{{"fact": "<one sentence>", "date": "YYYY-MM-DD", "entity": "<who/what it concerns>",
  "kind": "term_expiry|scheduled_vote|scheduled_meeting|deadline|election|expiration|recurring_event|other",
  "recurrence": "<cadence + past instances all before {as_of}, e.g. 'annual at WWDC each June: 2017..2025 unbroken', or null>",
  "source": "evidence|model_knowledge", "evidence_quote": "<short quote or null>",
  "confidence": <0..1>,
  "pattern_strength": "confirmed_scheduled|strong_recurrence|base_rate|speculative",
  "outcome_influence": "raises|lowers|neutral",
  "influence_strength": <0..1>,
  "reason": "<why it influences the outcome that way, or null>"}}]}}"""


def extract_scheduled_facts(question, *, as_of, horizon, evidence_text="", llm=None) -> list:
    """Extract dated facts (LLM proposes; dates are validated/parsed; junk dropped loudly)."""
    if llm is None:
        return []
    from swm.engine.grounding import parse_json
    raw = parse_json(llm(_EXTRACT_PROMPT.format(q=question, as_of=as_of, horizon=horizon,
                                                ev=evidence_text[:2400] or "(none)"))) or {}
    out = []
    for f in (raw.get("facts") or []):
        if not isinstance(f, dict) or not f.get("date"):
            continue
        try:
            ts = parse_time(str(f["date"])[:10])
        except (ValueError, TypeError):
            continue
        influence = str(f.get("outcome_influence", "")).lower()
        strength = max(0.0, min(1.0, float(f.get("influence_strength", 0.0) or 0.0)))
        pattern = str(f.get("pattern_strength", "base_rate"))[:24]
        # back-compat: legacy callers/tests read outcome_entailing + entailed_direction. Accept the new
        # influence schema OR the old entailing schema; derive one from the other so both always agree.
        # STRICT entailment is a separate, narrower judgment: only a strictly entailing fact may become
        # a deterministic absorbing event on the event-time path (§NAP) — a weak influence nudge is
        # calendar CONTEXT (prior/actor knowledge), never an absorber.
        if influence in ("raises", "lowers"):
            entailing = strength > 0.0
            direction = "yes" if influence == "raises" else "no"
            strictly = bool(f.get("outcome_entailing")) or pattern == "confirmed_scheduled" \
                or strength >= 0.9
        else:
            entailing = bool(f.get("outcome_entailing"))
            strictly = entailing                              # old schema: the LLM judged DIRECT entailment
            direction = (str(f.get("entailed_direction")).lower()
                         if f.get("entailed_direction") in ("yes", "no") else None)
            if entailing and direction:                       # old schema → synthesize an influence
                influence = "raises" if direction == "yes" else "lowers"
                strength = strength or 0.6
        out.append({"fact": str(f.get("fact", ""))[:200], "ts": ts, "date": str(f["date"])[:10],
                    "entity": str(f.get("entity", ""))[:60], "kind": str(f.get("kind", "other"))[:24],
                    "recurrence": (str(f.get("recurrence"))[:200] if f.get("recurrence") else None),
                    "source": str(f.get("source", "model_knowledge")),
                    "evidence_quote": (str(f.get("evidence_quote"))[:200]
                                       if f.get("evidence_quote") else None),
                    "confidence": max(0.0, min(1.0, float(f.get("confidence", 0.6) or 0.6))),
                    "pattern_strength": pattern,
                    "outcome_influence": influence if influence in ("raises", "lowers", "neutral") else "neutral",
                    "influence_strength": strength,
                    "outcome_entailing": entailing,
                    "strictly_entailing": strictly and direction in ("yes", "no"),
                    "entailed_direction": direction,
                    "reason": (str(f.get("reason"))[:160] if f.get("reason") else None)})
    return out


def entailment_nudge(influence: str, strength: float, confidence: float) -> float:
    """Signed log-odds nudge one scheduled fact contributes to the shared `fact_entailment` accumulator.
    raises→+, lowers→−, magnitude = strength·confidence·2.4, per-fact capped at ±1.8 so no single fact
    dictates certainty. Pure + testable."""
    influence = str(influence).lower()
    if influence not in ("raises", "lowers"):
        return 0.0
    sign = 1.0 if influence == "raises" else -1.0
    s = max(0.0, min(1.0, float(strength or 0.0)))
    c = max(0.0, min(1.0, float(confidence or 0.6)))
    return max(-1.8, min(1.8, sign * s * c * 2.4))


class ScheduledFactOperator(TransitionOperator):
    """Execute one dated public fact deterministically at its date. Writes a typed quantity
    `fact:<kind>:<entity>` = 1.0 and, when outcome-entailing, `fact_entailment` in [0,1] (confidence-scaled
    direction) that the outcome mechanism consumes at high weight. No randomness: a calendar fact is not a
    distribution — remaining uncertainty lives in the strategic layer's ability to break the entailment."""
    name = "scheduled_fact"

    def applicable(self, world, event):
        return event.etype == "scheduled_fact" and bool(event.payload.get("fact"))

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action=dict(event.payload),
                                  reason_codes=[f"fact:{event.payload.get('kind', 'other')}",
                                                f"src:{event.payload.get('source', '?')}"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        var = f"fact:{a.get('kind', 'other')}:{str(a.get('entity', ''))[:32]}"
        register_quantity_type(var, units="fact")
        world.quantities[var] = Quantity(name=var, qtype=var, value=1.0, timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="scheduled_fact", operator=self.name,
                       reason_codes=proposal.reason_codes,
                       uncertainty={"fact": a.get("fact"), "source": a.get("source"),
                                    "evidence_quote": a.get("evidence_quote"),
                                    "confidence": a.get("confidence")})
        d.change(f"quantities[{var}]", None, 1.0)
        influence = str(a.get("outcome_influence", "")).lower()
        if influence in ("raises", "lowers"):
            # ACCUMULATE net influence across ALL scheduled facts (not last-writer-wins): each fact nudges a
            # shared log-odds by sign*strength*confidence, so a strong recurrence that RAISES YES and a
            # disruption that LOWERS it compose into one honest fact_entailment (the visionOS case: WWDC-ships-
            # a-new-visionOS-yearly raises; 'Vision Pro abandoned' lowers). No single fact saturates.
            import math
            register_quantity_type("fact_entailment", units="share")
            register_quantity_type("fact_entailment_logodds", units="logodds")
            nudge = entailment_nudge(influence, a.get("influence_strength", 0.0),
                                     a.get("confidence", 0.6))            # per-fact cap: no lone certainty
            acc0 = getattr(world.quantities.get("fact_entailment_logodds"), "value", 0.0) or 0.0
            acc = max(-4.0, min(4.0, acc0 + nudge))
            val = 1.0 / (1.0 + math.exp(-acc))
            before = world.quantities.get("fact_entailment")
            world.quantities["fact_entailment_logodds"] = Quantity(
                name="fact_entailment_logodds", qtype="fact_entailment_logodds", value=round(acc, 4),
                timestamp=world.clock.now)
            world.quantities["fact_entailment"] = Quantity(
                name="fact_entailment", qtype="fact_entailment", value=round(val, 4),
                timestamp=world.clock.now)
            d.change("quantities[fact_entailment]", getattr(before, "value", None), round(val, 4))
        return d


register_operator("scheduled_fact", ScheduledFactOperator(), requires=("quantities",),
                  modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="dated public facts (evidence-cited or model-knowledge-labeled); "
                                   "deterministic — a calendar fact is not a distribution", validated=True)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
if not event_type_registered("scheduled_fact"):
    register_event_type("scheduled_fact", scheduling="scheduled", reads=("quantities",),
                        deltas=("quantities",), parameter_source="scheduled-reality layer", validated=True)


def public_facts_lines(facts: list, *, limit: int = 8) -> list:
    """Render scheduled/recurring public facts as short lines for ACTOR COGNITION — the real Tim Cook knows
    Apple's own calendar; the simulated one must too. Recurrence is included so the pattern (not just the
    next date) enters the actor's knowledge."""
    lines = []
    for f in (facts or [])[:limit]:
        rec = f" [{f['recurrence']}]" if f.get("recurrence") else ""
        lines.append(f"- ({f.get('date', '?')}) {f.get('fact', '')}{rec}")
    return lines


def attach_scheduled_facts(plan, facts: list) -> dict:
    """Schedule in-window facts as deterministic events; wire the entailment quantity into the outcome
    mechanism's consumption list at HIGH weight (a dated fact dominates broad priors). ALL extracted facts
    (in-window or not) are kept on `plan._scheduled_facts` so actor cognition can know the public calendar
    its real counterpart knows. Returns a report."""
    try:
        plan._scheduled_facts = list(facts or [])
    except Exception:  # noqa: BLE001
        pass
    n_sched, entailing = 0, 0
    for f in facts:
        if not (plan.as_of <= f["ts"] <= plan.horizon_ts):
            continue
        plan.scheduled_events.append({"etype": "scheduled_fact", "ts": f["ts"], "participants": [],
                                      "payload": f})
        n_sched += 1
        entailing += 1 if f.get("outcome_entailing") else 0
    if n_sched and not any(m.get("operator") == "scheduled_fact"
                           for m in plan.accepted_mechanisms if isinstance(m, dict)):
        plan.accepted_mechanisms.append({
            "mech_id": "scheduled_reality", "ontology_type": "deterministic_fact",
            "causal_role": "execute dated public facts deterministically (calendar layer)",
            "parameter_source": "evidence-cited or model-knowledge-labeled dated facts",
            "temporal_scale": "scheduled", "calibration_status": "deterministic",
            "operator": "scheduled_fact", "sensitivity": 1.0})
    if entailing:
        if not hasattr(plan, "_consumed_state"):
            plan._consumed_state = []
        if not any(m.get("var") == "fact_entailment" for m in plan._consumed_state):
            # a confident dated entailment should dominate: weight 0.4 of the capped 0.45 channel
            plan._consumed_state.insert(0, {"var": "fact_entailment", "weight": 0.4})
    return {"n_facts_extracted": len(facts), "n_scheduled_in_window": n_sched,
            "n_outcome_entailing": entailing}
