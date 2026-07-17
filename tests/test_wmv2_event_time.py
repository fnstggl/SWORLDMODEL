"""Event-time contract architecture — unit tests (first-passage timing, universal).

Includes the binary unification: the answer to a binary deadline question is a READOUT of the same
first-passage trajectories (P(yes) = F(deadline), polarity-mapped) — never a resolver's draw."""
import types

import pytest

from swm.world_model_v2.event_time import (AbsorptionMonitorOperator, EventTimeContract,
                                           HazardRoundOperator, _lexical_event_polarity,
                                           _mass_weights_from_curve, convert_binary_to_event_time,
                                           convert_to_event_time, fit_survival_pack, is_when_question)
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.state import SimulationClock, WorldBranch, WorldState

T0 = 1_700_000_000.0
T1 = T0 + 100 * 86400.0


def _world(now=T0, **quants):
    w = WorldState(world_id="w", branch_id="b1:x", clock=SimulationClock(now=now, as_of=T0))
    for k, v in quants.items():
        register_quantity_type(k, units="unit")
        w.quantities[k] = Quantity(name=k, qtype=k, value=v, timestamp=now)
    return w


def _branch(absorbed_at=None, absorbed_by=None):
    q = {}
    if absorbed_at is not None:
        q["absorbed_at"] = absorbed_at
        q["absorbed_by"] = absorbed_by or "modeA"
    w = _world(**q)
    return WorldBranch(branch_id=w.branch_id, world=w)


def test_projection_distribution_cdf_monotone_and_binary_view():
    c = EventTimeContract(as_of=T0, horizon_ts=T1, modes=["modeA"]).validate()
    branches = [_branch(T0 + f * (T1 - T0)) for f in (0.1, 0.3, 0.5)] + [_branch(), _branch()]
    out = c.project(branches)
    d = out["distribution"]
    assert d["absorbed_by_horizon"] == pytest.approx(0.6)
    assert d["censored_beyond_horizon"] == pytest.approx(0.4)
    et = out["event_time"]
    assert et["cdf"] == sorted(et["cdf"])                     # monotone by construction
    assert et["cdf"][-1] == pytest.approx(0.6)
    assert et["survival"][0] == pytest.approx(1 - et["cdf"][0])
    assert et["mode_distribution"] == {"modeA": pytest.approx(0.6)}
    # binary unification: P(yes by D) = F(D)
    assert c.cdf_at(T0 + 0.4 * (T1 - T0), branches) == pytest.approx(0.4)
    assert c.cdf_at(T1, branches) == pytest.approx(0.6)


def test_absorption_monitor_is_first_passage_and_immutable():
    op = AbsorptionMonitorOperator()
    w = _world(now=T0 + 5 * 86400, absorbing_state_reached=True, absorbing_mode="ceasefire")
    ev = types.SimpleNamespace(etype="anything", payload={})
    assert op.applicable(w, ev)
    d = op.apply(w, op.propose(w, ev, None))
    assert w.quantities["absorbed_at"].value == pytest.approx(w.clock.now)
    assert w.quantities["absorbed_by"].value == "ceasefire"
    assert d.event_type == "absorption"
    # first passage only: once stamped, never again applicable (immutability)
    w.clock.now += 86400
    assert not op.applicable(w, ev)
    assert w.quantities["absorbed_at"].value == pytest.approx(T0 + 5 * 86400)


def test_hazard_round_intention_factor_crushes_hazard():
    op = HazardRoundOperator()

    def run(intention_factor, seed_salt):
        hits = 0
        for i in range(400):
            w = _world(now=T0 + 30 * 86400)
            w.branch_id = f"b{i}:{seed_salt}"
            payload = {"mode": "deal", "base_hazard": 0.3, "intention_factor": intention_factor,
                       "as_of": T0, "span_s": T1 - T0, "consume": []}
            prop = op.propose(w, types.SimpleNamespace(etype="hazard_round", payload=payload), None)
            d = op.apply(w, prop)
            assert d.uncertainty["hazard"] <= 0.95
            if getattr(w.quantities.get("absorbing_state_reached"), "value", None):
                hits += 1
        return hits
    assert run(0.1, "a") < run(1.6, "a")                      # refusal crushes; commitment boosts


def test_hazard_round_not_applicable_after_absorption():
    op = HazardRoundOperator()
    w = _world(absorbed_at=T0 + 1)
    assert not op.applicable(w, types.SimpleNamespace(etype="hazard_round", payload={}))


def test_is_when_question_routing():
    assert is_when_question("When will the Russia-Ukraine conflict end?")
    assert is_when_question("How long until the Fed cuts rates?")
    assert not is_when_question("Will Powell be out as Fed Chair by 2025-12-31?")


def _plan():
    p = types.SimpleNamespace(
        question="When will the conflict end?", as_of=T0, horizon_ts=T1,
        structural_hypotheses=[{"id": "negotiated_settlement", "prior": 0.5},
                               {"id": "military_collapse", "prior": 0.25},
                               {"id": "frozen_conflict", "prior": 0.25}],
        outcome_contract=types.SimpleNamespace(options=["a", "b", "c"]),
        scheduled_events=[{"etype": "resolve_outcome", "ts": T1 - 1, "participants": [], "payload": {}}],
        accepted_mechanisms=[], quantities=[{"name": "actor_intentions", "qtype": "actor_intentions",
                                             "value": 0.2, "sd": None}],
        _consumed_state=[{"var": "actor_intentions", "weight": 0.2}])
    return p


def test_convert_to_event_time_full_rewire():
    p = _plan()
    lin = {}
    rep = convert_to_event_time(p, {"resolves_yes_iff": "an agreed end of hostilities holds"}, lineage=lin)
    assert isinstance(p.outcome_contract, EventTimeContract)
    assert p.outcome_contract.resolution_rule.startswith("an agreed end")
    assert not any(e["etype"] == "resolve_outcome" for e in p.scheduled_events)
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert rounds and rep["n_hazard_rounds"] == len(rounds)
    assert {e["payload"]["mode"] for e in rounds} == {"negotiated_settlement", "military_collapse",
                                                      "frozen_conflict"}
    assert all(e["payload"]["consume"] == [{"var": "actor_intentions", "weight": 0.2}] for e in rounds)
    assert all(T0 < e["ts"] < T1 for e in rounds)
    # total scheduled hazard mass equals the base envelope (0.5) — no len(modes) inflation
    assert sum(e["payload"]["base_hazard"] for e in rounds) == pytest.approx(0.5)
    # grounded intentions modulate ONLY agreement modes: refusal crushes deal hazards, never a
    # unilateral end-state's hazard
    by_mode = {e["payload"]["mode"]: e["payload"]["intention_factor"] for e in rounds}
    assert by_mode["negotiated_settlement"] == pytest.approx(0.2 + 1.4 * 0.2)
    assert by_mode["military_collapse"] == 1.0
    assert by_mode["frozen_conflict"] == 1.0
    ops = {m["operator"] for m in p.accepted_mechanisms}
    assert {"absorption_monitor", "hazard_round"} <= ops
    declared = {q["name"] for q in p.quantities}
    assert {"absorbed_at", "absorbing_state_reached"} <= declared   # readout binds at materialization
    assert 0.1 <= rep["intention_factor"] <= 1.6
    assert rep["intention_factor"] == pytest.approx(0.2 + 1.4 * 0.2)  # grounded intentions shape hazards
    assert lin["event_time"] is rep


def test_mode_entailment_filter_drops_non_absorbing_modes():
    p = _plan()

    def fake_llm(prompt):
        return ('{"absorbing_mode_ids": ["negotiated_settlement", "military_collapse"], '
                '"rejected": [{"id": "frozen_conflict", "why": "not an end state"}]}')
    rep = convert_to_event_time(p, {"resolves_yes_iff": "hostilities formally end"}, llm=fake_llm)
    assert set(rep["modes"]) == {"negotiated_settlement", "military_collapse"}
    assert rep["rejected_non_absorbing_modes"] == ["frozen_conflict"]
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert not any(e["payload"]["mode"] == "frozen_conflict" for e in rounds)
    assert sum(e["payload"]["base_hazard"] for e in rounds) == pytest.approx(0.5)  # renormalized


def test_mode_entailment_filter_fails_open():
    p = _plan()

    def broken_llm(prompt):
        raise RuntimeError("llm down")
    rep = convert_to_event_time(p, {"resolves_yes_iff": "hostilities formally end"}, llm=broken_llm)
    assert len(rep["modes"]) == 3                              # filter must never block the forecast


def test_fit_survival_pack_calibration_only_shape():
    rows = ([{"question": "Will X and Y sign a deal?", "lifetime_fraction_resolved": f}
             for f in (0.1, 0.3, None, None)] +
            [{"question": "Will team A win the match?", "lifetime_fraction_resolved": 0.9}])
    pack = fit_survival_pack(rows)
    assert pack["fit_on"] == "calibration split only"
    assert len(pack["global_hazards"]) == 5
    assert all(0.0 <= h <= 1.0 for h in pack["global_hazards"])
    assert set(pack["families"]) == {"meeting_or_deal_by_date", "sports_match"}


# ================================================================ binary unification (readout, no resolver)
def _binary_plan(question="Will X be out as CEO by the deadline?", options=("yes", "no"),
                 posterior=None, extra_events=None, consumed=None):
    contract = types.SimpleNamespace(family="binary", options=list(options), resolution_rule="")
    return types.SimpleNamespace(
        question=question, as_of=T0, horizon_ts=T1, outcome_contract=contract,
        scheduled_events=[{"etype": "resolve_outcome", "ts": T1 - 1.0, "participants": [],
                           "payload": {"outcome_var": "outcome", "family": "binary",
                                       "options": list(options), "lean": "neutral"}}]
        + list(extra_events or []),
        accepted_mechanisms=[], quantities=[], provenance={"outcome_lean": "neutral"},
        posterior_rate_particles=list(posterior or []) or None,
        _consumed_state=list(consumed or []))


def _rollout(p, n_particles=300, seed=3, extra_ops=None, meta=None):
    from swm.world_model_v2.events import Event, EventQueue
    from swm.world_model_v2.rollout import RolloutEngine
    ops = [HazardRoundOperator(), AbsorptionMonitorOperator()] + list(extra_ops or [])
    branches = []
    for i in range(n_particles):
        w = _world()
        w.branch_id = f"b{i:03d}"
        if meta:
            w.uncertainty_meta = dict(meta)
        q = EventQueue(horizon_ts=p.horizon_ts)
        for ev in p.scheduled_events:
            q.schedule(Event(ts=ev["ts"], etype=ev["etype"],
                             participants=list(ev.get("participants") or []),
                             payload=dict(ev["payload"]), source="scheduled"))
        branches.append(RolloutEngine(operators=ops).run_branch(w, q, seed=seed * 7919 + i))
    return p.outcome_contract.project(branches), branches


def test_binary_conversion_removes_every_resolver_and_rewires():
    p = _binary_plan(posterior=[(0.3, 1.0)], consumed=[{"var": "actor_intentions", "weight": 0.2}],
                     extra_events=[{"etype": "aggregate_outcome_resolution", "ts": T1 - 0.5,
                                    "participants": [], "payload": {"outcome_var": "outcome",
                                                                    "options": ["yes", "no"],
                                                                    "consume": []}}])
    lin = {}
    rep = convert_binary_to_event_time(p, {"resolves_yes_iff": "X departs the CEO role"}, lineage=lin)
    assert rep["contract"] == "binary_first_passage" and rep["n_resolver_events_removed"] == 2
    assert not any(e["etype"] in ("resolve_outcome", "aggregate_outcome_resolution")
                   for e in p.scheduled_events)                # NO component declares the answer
    assert isinstance(p.outcome_contract, EventTimeContract)
    assert p.outcome_contract.options == ["yes", "no"]         # the question's own options project
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert rounds and all("calibration" in e["payload"] for e in rounds)
    assert sum(e["payload"]["calibration"]["exponent"] for e in rounds) == pytest.approx(1.0)
    assert all(e["payload"]["consume"] == [{"var": "actor_intentions", "weight": 0.2}] for e in rounds)
    assert all(T0 < e["ts"] <= p.outcome_contract.deadline_ts for e in rounds)
    order = [m["operator"] for m in p.accepted_mechanisms]
    # absorbing writers run BEFORE the monitor so first passage stamps at the write's own clock time
    assert order.index("hazard_round") < order.index("absorption_monitor")
    assert {q["name"] for q in p.quantities} >= {"absorbed_at", "absorbing_state_reached"}
    assert lin["event_time"] is rep and rep["posterior_calibrated"] is True


def test_binary_answer_is_first_passage_readout_matching_posterior_target():
    p = _binary_plan(posterior=[(0.3, 1.0)])
    convert_binary_to_event_time(p, {})
    out, _ = _rollout(p, n_particles=400)
    d = out["distribution"]
    assert set(d) == {"yes", "no"}
    # the calibrated chain's total absorbed mass reproduces the posterior target — but as OBSERVED events
    assert d["yes"] == pytest.approx(0.3, abs=0.07)
    et = out["event_time"]
    assert et["cdf"] == sorted(et["cdf"])                      # multi-cutoff coherence by construction
    assert et["p_event_by_deadline"] == d["yes"]               # the binary answer IS the timing readout
    assert et["mode_distribution"].get("resolution", 0.0) == pytest.approx(d["yes"], abs=1e-9)
    assert out["readout"] == "terminal_states"


def test_binary_survival_polarity_maps_absorption_to_no():
    p = _binary_plan(question="Will X remain CEO through the deadline?", posterior=[(0.7, 1.0)])
    rep = convert_binary_to_event_time(p, {})
    assert rep["event_polarity"] == "occurrence_resolves_no" and rep["polarity_source"] == "lexical"
    assert p.outcome_contract.occurrence_resolves == "no"
    out, _ = _rollout(p, n_particles=400)
    # P(remain)=0.7 ⇒ the state-breaking event carries mass 0.3; absorption maps to the NO option
    assert out["distribution"]["yes"] == pytest.approx(0.7, abs=0.07)


def test_entailed_fact_absorbs_at_its_real_date_and_floors_the_answer():
    import swm.world_model_v2.scheduled_facts  # noqa: F401 — registers the scheduled_fact event type
    fact_ts = T0 + 10 * 86400.0
    fact_ev = {"etype": "scheduled_fact", "ts": fact_ts, "participants": [],
               "payload": {"fact": "the term ends", "kind": "term_expiry", "entity": "X",
                           "confidence": 0.9, "outcome_entailing": True, "entailed_direction": "yes",
                           "source": "model_knowledge"}}
    p = _binary_plan(posterior=[(0.3, 1.0)], extra_events=[fact_ev],
                     consumed=[{"var": "fact_entailment", "weight": 0.4}])
    rep = convert_binary_to_event_time(p, {})
    assert rep["n_absorbing_fact_events"] == 1 and rep["fact_floor"] == pytest.approx(0.9)
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"
              and "calibration" in e["payload"]]
    # the fact acts through ABSORPTION at its date — the rate-blend channel is removed (no double count)
    assert all(m["var"] != "fact_entailment" for e in rounds for m in e["payload"]["consume"])
    out, branches = _rollout(p, n_particles=400)
    # total mass = max(fact confidence, posterior target) = 0.9 — and the event happens AT ITS DATE
    assert out["distribution"]["yes"] == pytest.approx(0.9, abs=0.05)
    assert out["event_time"]["mode_distribution"].get("entailed_fact:term_expiry", 0.0) >= 0.8
    c = p.outcome_contract
    fact_times = [c.readout(b.world) for b in branches
                  if c.readout(b.world) and c._mode_of(b.world).startswith("entailed_fact")]
    assert fact_times and all(t == pytest.approx(fact_ts) for t in fact_times)


def test_absorbing_institution_is_the_resolution_path():
    inst_ev = {"etype": "institutional_decision", "ts": T1 - 2.0, "participants": [],
               "payload": {"institution_id": "senate", "outcome_var": "outcome",
                           "options": ["yes", "no"], "n_members": 9, "needed": 5,
                           "posterior_rate_particles": [[0.95, 1.0]]}}
    p = _binary_plan(extra_events=[inst_ev])
    rep = convert_binary_to_event_time(p, {})
    assert rep["absorbing_institutions"] == ["senate"]
    assert rep["n_residual_rounds"] == 0 and rep["residual_skipped_reason"]
    ie = next(e for e in p.scheduled_events if e["etype"] == "institutional_decision")
    assert ie["payload"]["absorbing"] is True
    from swm.world_model_v2.phase_consumers import CollectiveThresholdDecisionOperator
    out, _ = _rollout(p, n_particles=200, extra_ops=[CollectiveThresholdDecisionOperator()])
    # 0.95 member propensity under a declared 5-of-9 rule → passes; YES is READ from the absorption
    assert out["distribution"]["yes"] >= 0.9
    assert out["event_time"]["mode_distribution"].get("institutional:senate", 0.0) >= 0.9


def test_conversion_skips_non_binary_contracts_without_mutation():
    p = _binary_plan(options=("a", "b", "c"))
    p.outcome_contract.family = "categorical"
    before = [dict(e) for e in p.scheduled_events]
    rep = convert_binary_to_event_time(p, {})
    assert "skipped" in rep
    assert p.scheduled_events == before and p.accepted_mechanisms == []


def test_lexical_polarity_and_criterion_override():
    assert _lexical_event_polarity("Will Powell be out as Fed Chair by Aug 31?") == \
        "occurrence_resolves_yes"
    assert _lexical_event_polarity("Will Powell remain Fed Chair through Aug 31?") == \
        "occurrence_resolves_no"
    assert _lexical_event_polarity("Will X still be CEO on June 30?") == "occurrence_resolves_no"
    p = _binary_plan(question="Will the ceasefire hold?")
    rep = convert_binary_to_event_time(p, {"event_polarity": "occurrence_resolves_no"})
    assert rep["event_polarity"] == "occurrence_resolves_no"
    assert rep["polarity_source"] == "criterion_parser"


def test_hypothesis_lean_shifts_first_passage_mass():
    p = _binary_plan(posterior=[(0.4, 1.0)])
    convert_binary_to_event_time(p, {})

    def absorbed_share(lean):
        out, branches = _rollout(p, n_particles=300, seed=11, meta={"hypothesis_lean": lean})
        c = p.outcome_contract
        return sum(1 for b in branches if c.readout(b.world)) / 300.0
    # competing structures shift the per-particle target rate — genuinely different terminals
    assert absorbed_share("strong_no") < absorbed_share("neutral") < absorbed_share("strong_yes")


def test_consume_invert_channel():
    from swm.world_model_v2.phase_consumers import consume_state_rate
    w = _world(actor_intentions=0.9)
    p1, used1 = consume_state_rate(w, 0.5, [{"var": "actor_intentions", "weight": 0.4}])
    p2, used2 = consume_state_rate(w, 0.5, [{"var": "actor_intentions", "weight": 0.4, "invert": True}])
    assert used1 == used2 == ["actor_intentions"]
    assert p1 == pytest.approx(0.5 * 0.6 + 0.4 * 0.9)
    assert p2 == pytest.approx(0.5 * 0.6 + 0.4 * 0.1)


def test_readout_falls_back_to_absorbing_write_time():
    """A writer that fires on the branch's LAST event leaves the monitor no later event to stamp on —
    the readout recovers first passage from the predicate quantity's own write time."""
    c = EventTimeContract(as_of=T0, horizon_ts=T1, binary_options=["yes", "no"]).validate()
    w = _world()
    register_quantity_type("absorbing_state_reached", units="bool")
    w.quantities["absorbing_state_reached"] = Quantity(
        name="absorbing_state_reached", qtype="absorbing_state_reached", value=True,
        timestamp=T0 + 5 * 86400.0)
    assert c.readout(w) == pytest.approx(T0 + 5 * 86400.0)


def test_criterion_deadline_bounds_the_chain_and_the_binary_view():
    p = _binary_plan(posterior=[(0.5, 1.0)])
    rep = convert_binary_to_event_time(p, {"deadline": "2024-01-01"})   # inside [T0, T1]
    assert T0 < rep["deadline_ts"] < T1
    rounds = [e for e in p.scheduled_events if e["etype"] == "hazard_round"]
    assert rounds and all(e["ts"] <= rep["deadline_ts"] for e in rounds)
    assert p.outcome_contract.deadline_ts == pytest.approx(rep["deadline_ts"])


def test_mass_weights_from_curve_shape():
    assert _mass_weights_from_curve(None) == [0.2] * 5
    w = _mass_weights_from_curve([0.5, 0.0, 0.0, 0.0, 0.0])
    assert w[0] == pytest.approx(1.0) and sum(w) == pytest.approx(1.0)
    w2 = _mass_weights_from_curve([0.1, 0.1, 0.1, 0.1, 0.1])
    assert sum(w2) == pytest.approx(1.0) and w2[0] > w2[-1]    # early buckets carry more surviving mass


def test_unified_runtime_routes_binary_through_event_time():
    import inspect
    from swm.world_model_v2 import unified_runtime as U
    src = inspect.getsource(U.simulate_world)
    assert "convert_binary_to_event_time" in src and "convert_to_event_time" in src
