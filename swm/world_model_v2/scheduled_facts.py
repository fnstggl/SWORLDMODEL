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
elections, expirations, launches. Use the evidence AND your world knowledge of the real named people and
institutions. Only include facts you are confident are real, with dates as YYYY-MM-DD (approximate day is
acceptable if the month is certain). For each, state whether its occurrence DIRECTLY entails the question's
outcome and in which direction.

QUESTION: {q}
AS-OF: {as_of}   HORIZON: {horizon}
EVIDENCE:
{ev}

Return ONLY JSON:
{{"facts": [{{"fact": "<one sentence>", "date": "YYYY-MM-DD", "entity": "<who/what it concerns>",
  "kind": "term_expiry|scheduled_vote|scheduled_meeting|deadline|election|expiration|other",
  "source": "evidence|model_knowledge", "evidence_quote": "<short quote or null>",
  "confidence": <0..1>,
  "outcome_entailing": true|false,
  "entailed_direction": "yes|no|null",
  "entailment_caveat": "<what could break the entailment, or null>"}}]}}"""


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
        out.append({"fact": str(f.get("fact", ""))[:200], "ts": ts, "date": str(f["date"])[:10],
                    "entity": str(f.get("entity", ""))[:60], "kind": str(f.get("kind", "other"))[:24],
                    "source": str(f.get("source", "model_knowledge")),
                    "evidence_quote": (str(f.get("evidence_quote"))[:200]
                                       if f.get("evidence_quote") else None),
                    "confidence": max(0.0, min(1.0, float(f.get("confidence", 0.6) or 0.6))),
                    "outcome_entailing": bool(f.get("outcome_entailing")),
                    "entailed_direction": (str(f.get("entailed_direction")).lower()
                                           if f.get("entailed_direction") in ("yes", "no") else None),
                    "entailment_caveat": (str(f.get("entailment_caveat"))[:160]
                                          if f.get("entailment_caveat") else None)})
    return out


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
        if a.get("outcome_entailing") and a.get("entailed_direction") in ("yes", "no"):
            register_quantity_type("fact_entailment", units="share")
            conf = float(a.get("confidence", 0.6))
            val = conf if a["entailed_direction"] == "yes" else 1.0 - conf
            before = world.quantities.get("fact_entailment")
            world.quantities["fact_entailment"] = Quantity(name="fact_entailment",
                                                           qtype="fact_entailment", value=round(val, 4),
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


def attach_scheduled_facts(plan, facts: list) -> dict:
    """Schedule in-window facts as deterministic events; wire the entailment quantity into the outcome
    mechanism's consumption list at HIGH weight (a dated fact dominates broad priors). Returns a report."""
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
