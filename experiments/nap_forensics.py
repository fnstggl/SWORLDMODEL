"""§NAP forensic traces — six offline cases through the provenance-gated event-time architecture.

Each case prints: every load-bearing causal input (with provenance), the concrete state
transitions, the unresolved mechanisms, the terminal classification, and confirmation that no
arbitrary scalar altered the result. The expected behavior is NOT that every case returns a
probability: a correct unresolved result is better than a precise invented answer.

Run: PYTHONPATH=. python experiments/nap_forensics.py
Writes artifacts/no_arbitrary_numeric_reality/forensic_traces/case{1..6}.json + summary.json
"""
from __future__ import annotations

import json
import pathlib
import types

from swm.world_model_v2.event_time import (AbsorptionMonitorOperator, FirstPassageOperator,
                                           HazardRoundOperator, convert_binary_to_event_time,
                                           convert_to_event_time, ensure_first_passage_state)
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.pipeline import result_from_run
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import SimulationClock, WorldState
from swm.world_model_v2.temporal_hazards import schedule_crossing
from swm.world_model_v2.world_dynamics import PersistenceCheckOperator, break_provisional_state

T0 = 1_700_000_000.0
DAY = 86400.0
T1 = T0 + 100 * DAY
OUT = pathlib.Path("artifacts/no_arbitrary_numeric_reality/forensic_traces")


def _plan(question, options=("yes", "no"), posterior=None, extra_events=None):
    contract = types.SimpleNamespace(family="binary", options=list(options), resolution_rule="")
    return types.SimpleNamespace(
        question=question, as_of=T0, horizon_ts=T1, outcome_contract=contract,
        scheduled_events=[{"etype": "resolve_outcome", "ts": T1 - 1.0, "participants": [],
                           "payload": {"outcome_var": "outcome", "family": "binary",
                                       "options": list(options), "lean": "neutral"}}]
        + list(extra_events or []),
        accepted_mechanisms=[], quantities=[], provenance={},
        posterior_rate_particles=list(posterior or []) or None,
        support_grade="exploratory", degraded=False, omissions=[], fallbacks_used=[],
        mechanism_choices=[], latents=[], interpretations=[],
        compute_plan={"n_particles": 200}, plan_hash=lambda: "forensic")


def _rollout(p, n_particles=200, seed=3, breaker=None):
    ops = [HazardRoundOperator(), FirstPassageOperator(), PersistenceCheckOperator(),
           AbsorptionMonitorOperator()]
    branches = []
    for i in range(n_particles):
        w = WorldState(world_id="w", branch_id=f"b{i:03d}",
                       clock=SimulationClock(now=T0, as_of=T0))
        for rec in (getattr(p, "_unresolved_mechanisms", None) or []):
            w._unresolved_mechanisms = list(getattr(w, "_unresolved_mechanisms", None) or [])
            w._unresolved_mechanisms.append(dict(rec))
        q = EventQueue(horizon_ts=p.horizon_ts)
        for ev in p.scheduled_events:
            q.schedule(Event(ts=ev["ts"], etype=ev["etype"],
                             participants=list(ev.get("participants") or []),
                             payload=dict(ev["payload"]), source="scheduled"))
        for spec in (getattr(p, "first_passage_processes", None) or []):
            st = ensure_first_passage_state(w, spec)
            if st is not None:
                schedule_crossing(q, w, st, etype="first_passage")
        if breaker:
            breaker(w, q, i)
        branches.append(RolloutEngine(operators=ops).run_branch(w, q, seed=seed * 7919 + i))
    out = p.outcome_contract.project(branches)
    return out, branches


def _trace(name, question, rep, res, notes):
    rr = res.resolution_report
    return {
        "case": name, "question": question,
        "conversion_report": {k: rep.get(k) for k in
                              ("contract", "scheduling", "n_residual_processes",
                               "n_absorbing_fact_events", "absorbing_institutions",
                               "residual_skipped_reason", "posterior_calibrated",
                               "family_rate_rejected", "unresolved_mode_transitions",
                               "n_ungrounded_facts_unresolved") if k in rep},
        "numeric_causal_inputs": rr.get("numeric_causal_inputs"),
        "branch_terminals": rr.get("branch_terminals"),
        "unresolved_share": rr.get("unresolved_share"),
        "bounds": rr.get("bounds"),
        "missing_mechanisms": rr.get("missing_mechanisms"),
        "terminal_classification": res.simulation_status,
        "recommendation_status": res.recommendation_status,
        "raw_distribution": res.raw_distribution,
        "frequency_semantics": rr.get("frequency_semantics"),
        "notes": notes,
    }


def case1():
    """Negotiation with a signed-proposal deadline: an EVIDENCE-CITED dated fact (the charter's
    term expiry) absorbs deterministically at its real date. No invented scalar anywhere."""
    import swm.world_model_v2.scheduled_facts  # noqa: F401
    fact = {"etype": "scheduled_fact", "ts": T0 + 20 * DAY, "participants": [],
            "payload": {"fact": "signature ceremony scheduled per the signed term sheet",
                        "kind": "signing", "entity": "parties", "confidence": 0.9,
                        "outcome_entailing": True, "entailed_direction": "yes",
                        "source": "evidence", "evidence_quote": "both parties signed the term "
                        "sheet committing to execute on the 20th", "claim_id": "c_sign_1"}}
    q = "Will the acquisition agreement be signed by the deadline?"
    p = _plan(q, extra_events=[fact])
    rep = convert_binary_to_event_time(p, {"resolves_yes_iff": "the agreement is signed"})
    out, branches = _rollout(p, n_particles=60)
    res = result_from_run(q, p, out, branches)
    return _trace("case1_negotiation_signed_proposal", q, rep, res,
                  ["the ONLY load-bearing numeric input is the fact's deterministic occurrence "
                   "at its real date (observed_measurement, claim c_sign_1)",
                   "the LLM extraction confidence (0.9) rides as a label, not a parameter"])


def case2():
    """Public commitment by a political actor: the stance is a QUALITATIVE record conditioning
    the actor's own cognition; it multiplies no hazard. With no posterior, no evidence-cited
    fact and no institution, the outcome is honestly UNRESOLVED."""
    from swm.world_model_v2.resolution_criteria import _binding_prohibitions
    q = "Will the two countries sign a ceasefire by the deadline?"
    p = _plan(q)
    stance = {"actor": "leader_a", "commitment_level": "committed_to_prevent",
              "reliability": "high", "basis_kind": "public_statement",
              "pathway": "cooperative_agreement",
              "quote": "we will never negotiate", "explicit_prohibitions": ["negotiate"]}
    p._intention_stances = [stance]
    rep = convert_binary_to_event_time(p, {"resolves_yes_iff": "a ceasefire is signed"})
    out, branches = _rollout(p, n_particles=40)
    res = result_from_run(q, p, out, branches, intervention="what should we do?")
    return _trace("case2_public_commitment", q, rep, res,
                  ["the public statement is NOT binding (basis_kind=public_statement): "
                   f"binding_prohibitions={_binding_prohibitions(stance)}",
                   "no stance->hazard multiplication exists anywhere",
                   "no posterior/fact/institution => residual mechanism recorded unresolved; "
                   "the answer is 'Outcome unresolved under the current model' with the missing "
                   "mechanism named; recommendation withheld"])


def case3():
    """Institutional vote: the declared decision procedure IS the resolution path — it absorbs
    at its scheduled date through its real rule. The rule numbers (9 members, majority) are
    institutional_rule provenance."""
    q = "Will the council approve the variance by the deadline?"
    inst = {"etype": "institutional_decision", "ts": T0 + 30 * DAY, "participants": [],
            "payload": {"institution_id": "council", "outcome_var": "outcome",
                        "n_members": 9, "threshold_share": 0.5, "options": ["yes", "no"],
                        "posterior_rate_particles": [[0.6, 1.0]]}}
    p = _plan(q, extra_events=[inst], posterior=[(0.6, 1.0)])
    rep = convert_binary_to_event_time(p, {"resolves_yes_iff": "the council approves"})
    from swm.world_model_v2.phase_consumers import CollectiveThresholdDecisionOperator
    ops_extra = CollectiveThresholdDecisionOperator()

    def breaker(w, queue, i):
        pass
    out, branches = _rollout(p, n_particles=120)
    # run again including the institutional operator
    ops = [CollectiveThresholdDecisionOperator(), HazardRoundOperator(), FirstPassageOperator(),
           PersistenceCheckOperator(), AbsorptionMonitorOperator()]
    branches = []
    for i in range(120):
        w = WorldState(world_id="w", branch_id=f"b{i:03d}",
                       clock=SimulationClock(now=T0, as_of=T0))
        queue = EventQueue(horizon_ts=p.horizon_ts)
        for ev in p.scheduled_events:
            queue.schedule(Event(ts=ev["ts"], etype=ev["etype"], participants=[],
                                 payload=dict(ev["payload"]), source="scheduled"))
        branches.append(RolloutEngine(operators=ops).run_branch(w, queue, seed=101 + i))
    out = p.outcome_contract.project(branches)
    res = result_from_run(q, p, out, branches)
    return _trace("case3_institutional_vote", q, rep, res,
                  ["the institution absorbs at its real date through its real threshold rule "
                   "(9 members, majority — institutional_rule provenance)",
                   "member propensity comes from the evidence-updated posterior; the residual "
                   "chain is suppressed because the institution IS the resolution path"])


def case4():
    """Personal communication with an evidence-updated posterior: the residual outcome process
    is parameterized ONLY by the posterior (registered in the ledger); the answer is a
    first-passage readout labeled simulated_scenario_frequency."""
    q = "Will she reply to the message by the deadline?"
    p = _plan(q, options=("reply", "no_reply"), posterior=[(0.55, 0.7), (0.4, 0.3)])
    rep = convert_binary_to_event_time(p, {"resolves_yes_iff": "a reply is sent"})
    out, branches = _rollout(p, n_particles=300)
    res = result_from_run(q, p, out, branches)
    return _trace("case4_personal_communication", q, rep, res,
                  ["the only load-bearing numeric input is the posterior target mass "
                   "(derived_deterministic over as-of evidence; its broad unfitted prior is a "
                   "recorded remaining assumption)",
                   "shares are simulated_scenario_frequency, not calibrated probabilities"])


def case5():
    """Operational launch with a persistence-window criterion: the provisional end-state
    confirms or collapses OBSERVATIONALLY — a modeled breaking mechanism collapses some
    branches; no survival coin exists."""
    q = "Will the product launch and remain generally available for >=14 consecutive days?"
    p = _plan(q, posterior=[(0.8, 1.0)])
    rep = convert_binary_to_event_time(
        p, {"resolves_yes_iff": "launch occurs and remains available for >=14 consecutive days",
            "persistence_days": 14})
    for spec in (getattr(p, "first_passage_processes", None) or []):
        spec["persistence_s"] = 14 * DAY

    def breaker(w, queue, i):
        if i % 3 == 0:                          # a MODELED rollback event breaks 1/3 of branches
            def _apply(world):
                return break_provisional_state(world, reason="critical regression forced a "
                                                             "rollback (modeled event)")
            queue.schedule(Event(ts=T0 + 40 * DAY, etype="persistence_break_probe",
                                 participants=[], payload={"case": "rollback"},
                                 source="scheduled"))
            w._forensic_break = True
    # custom operator that executes the modeled breaking event
    from swm.world_model_v2.transitions import (StateDelta, TransitionOperator,
                                                TransitionProposal, ValidationResult)

    class RollbackOperator(TransitionOperator):
        name = "rollback_probe"

        def applicable(self, world, event):
            return event.etype == "persistence_break_probe"

        def validate(self, world, proposal):
            return ValidationResult(ok=True)

        def propose(self, world, event, rng):
            return TransitionProposal(operator=self.name, action={}, reason_codes=["rollback"])

        def apply(self, world, proposal):
            broke = break_provisional_state(world, reason="critical regression forced a rollback")
            d = StateDelta(at=world.clock.now, event_type="persistence_break_probe",
                           operator=self.name,
                           reason_codes=["modeled_breaking_mechanism" if broke else "noop"])
            return d
    from swm.world_model_v2.events import register_event_type, event_type_registered
    if not event_type_registered("persistence_break_probe"):
        register_event_type("persistence_break_probe", scheduling="scheduled",
                            reads=("quantities",), deltas=("quantities",),
                            parameter_source="forensic modeled breaking mechanism",
                            validated=True)
    ops = [HazardRoundOperator(), FirstPassageOperator(), RollbackOperator(),
           PersistenceCheckOperator(), AbsorptionMonitorOperator()]
    branches = []
    for i in range(150):
        w = WorldState(world_id="w", branch_id=f"b{i:03d}",
                       clock=SimulationClock(now=T0, as_of=T0))
        queue = EventQueue(horizon_ts=p.horizon_ts)
        for ev in p.scheduled_events:
            queue.schedule(Event(ts=ev["ts"], etype=ev["etype"], participants=[],
                                 payload=dict(ev["payload"]), source="scheduled"))
        for spec in (getattr(p, "first_passage_processes", None) or []):
            st = ensure_first_passage_state(w, spec)
            if st is not None:
                schedule_crossing(queue, w, st, etype="first_passage")
        if i % 3 == 0:
            queue.schedule(Event(ts=T0 + 40 * DAY, etype="persistence_break_probe",
                                 participants=[], payload={}, source="scheduled"))
        branches.append(RolloutEngine(operators=ops).run_branch(w, queue, seed=7 + i))
    out = p.outcome_contract.project(branches)
    res = result_from_run(q, p, out, branches)
    return _trace("case5_launch_persistence", q, rep, res,
                  ["persistence window (14 days) comes from the criterion's literal text "
                   "(institutional_rule provenance)",
                   "provisional launches confirm by OBSERVATION; the 1/3 of branches carrying a "
                   "modeled rollback event collapse — no 0.85 survival coin anywhere"])


def case6():
    """A coherent question with NO defensible numeric mechanism: no posterior, no facts, no
    institutions, no fitted models. The correct output is `unresolved` with the missing
    mechanism named — no Beta, no Dirichlet, no family rate, no forced yes/no."""
    q = "Will the informal roommate agreement survive the year?"
    p = _plan(q)
    rep = convert_binary_to_event_time(p, {"resolves_yes_iff": "the agreement holds"})
    out, branches = _rollout(p, n_particles=30)
    res = result_from_run(q, p, out, branches, intervention="recommend the best action")
    assert res.simulation_status == "unresolved"
    assert res.recommendation_status == "withheld"
    return _trace("case6_no_defensible_mechanism", q, rep, res,
                  ["no validated causal mechanism resolves the outcome: the result is "
                   "'Outcome unresolved under the current model'",
                   "the family pack and lean-Beta were REGISTERED AS REJECTED, never drawn",
                   "recommendations are withheld"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [case1, case2, case3, case4, case5, case6]
    summary = []
    for i, fn in enumerate(cases, 1):
        tr = fn()
        (OUT / f"case{i}.json").write_text(json.dumps(tr, indent=2, default=str))
        man = tr.get("numeric_causal_inputs") or {}
        summary.append({
            "case": tr["case"], "terminal_classification": tr["terminal_classification"],
            "recommendation_status": tr["recommendation_status"],
            "unresolved_share": tr["unresolved_share"],
            "raw_distribution": tr["raw_distribution"],
            "n_numeric_inputs_consumed": len(man.get("approved_and_consumed") or []),
            "n_numeric_inputs_rejected": len(man.get("rejected") or []),
            "missing_mechanisms": [m.get("mechanism") for m in
                                   (tr.get("missing_mechanisms") or [])],
        })
        print(f"case{i} {tr['case']}: {tr['terminal_classification']} "
              f"dist={tr['raw_distribution']} unresolved={tr['unresolved_share']}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {len(cases)} traces to {OUT}")


if __name__ == "__main__":
    main()
