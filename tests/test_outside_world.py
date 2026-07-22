"""Contract tests for swm/world_model_v2/outside_world.py (residual outside-world process)."""
import json
import math
import random

import pytest

from swm.world_model_v2.outside_world import (
    ARRIVAL_KINDS,
    ENTRY_MECHANISMS,
    FORBIDDEN_WRITES,
    OUTSIDE_SCHEMA,
    ArrivalModel,
    ExternalEventFamily,
    OutsideWorldProcess,
    entry_event_payload,
    generate_outside_world,
    sample_arrivals,
    validate_entry,
)
from swm.world_model_v2.world_boundary import BoundaryComponent, WorldBoundary

T0 = 1_700_000_000.0
DAY = 86400.0


def fam(**kw):
    base = dict(family_id="f1", description="a plain external event",
                impact_mechanism="observation_delivery",
                impact_description="delivers a news item to the sales channel")
    base.update(kw)
    return ExternalEventFamily(**base)


# ------------------------------------------------------------------ validate_entry (§5.1)
def test_validate_rejects_untyped_impact_mechanism():
    f = validate_entry(fam(impact_mechanism="direct_belief_write"))
    assert f.validation_error and "not a typed" in f.validation_error
    assert "direct_belief_write" in f.validation_error


def test_validate_rejects_success_write_in_affected_components():
    f = validate_entry(fam(affected_boundary_components=["campaign success metric"]))
    assert f.validation_error and "'success'" in f.validation_error
    assert "causal mechanism" in f.validation_error


def test_validate_rejects_terminal_outcome_write_in_impact_description():
    f = validate_entry(fam(impact_description="writes the terminal outcome of the run"))
    assert f.validation_error and "'terminal_outcome'" in f.validation_error


def test_validate_rejects_actor_reaction_write():
    f = validate_entry(fam(affected_boundary_components=["actor reaction of the ceo"]))
    assert f.validation_error and "'actor_reaction'" in f.validation_error


def test_validate_coerces_provenance_less_rate_to_unresolved_with_honesty_note():
    f = validate_entry(fam(arrival=ArrivalModel(kind="observed_base_rate", rate_per_day=0.4,
                                                provenance="   ")))
    assert f.validation_error == ""                    # coerced, not rejected
    assert f.arrival.kind == "unresolved" and f.arrival.rate_per_day == 0.0
    assert f.uncertainty == "unresolved"
    assert "no provenance" in f.evidence[-1]
    assert "never mints a precise event probability" in f.evidence[-1]
    # with a named source the rate is admissible
    g = validate_entry(fam(arrival=ArrivalModel(kind="observed_base_rate", rate_per_day=0.4,
                                                provenance="2019-2024 outage log")))
    assert g.arrival.kind == "observed_base_rate" and g.arrival.rate_per_day == 0.4
    assert g.validation_error == ""


def test_validate_scheduled_exact_without_times_is_unresolved():
    f = validate_entry(fam(arrival=ArrivalModel(kind="scheduled_exact")))
    assert f.arrival.kind == "unresolved"
    g = validate_entry(fam(arrival=ArrivalModel(kind="scheduled_exact",
                                                scheduled_times=[T0 + DAY])))
    assert g.arrival.kind == "scheduled_exact"


def test_validate_unknown_arrival_kind_coerced_to_unresolved():
    f = validate_entry(fam(arrival=ArrivalModel(kind="vibes_based", rate_per_day=3.0)))
    assert f.arrival.kind == "unresolved"


# ------------------------------------------------------------------ sampling (§5.2)
def _expected_poisson(seed, rate_per_day, t0, t1):
    """Replicates the documented thinning-free exponential-gap sampler."""
    rng = random.Random(seed)
    out, t = [], float(t0)
    while True:
        u = rng.random()
        if u <= 0.0:
            u = 1e-12
        t += -math.log(u) / (rate_per_day / DAY)
        if t >= t1:
            return out
        out.append(t)


def test_sample_arrivals_raises_on_unresolved_and_rejected_families():
    f = fam(arrival=ArrivalModel(kind="unresolved"))
    with pytest.raises(ValueError, match="never sampled"):
        sample_arrivals(f, t0=T0, t1=T0 + DAY, rng=random.Random(0))
    g = validate_entry(fam(impact_mechanism="direct_belief_write",
                           arrival=ArrivalModel(kind="observed_base_rate", rate_per_day=1.0,
                                                provenance="src")))
    with pytest.raises(ValueError):
        sample_arrivals(g, t0=T0, t1=T0 + DAY, rng=random.Random(0))


def test_poisson_sampling_deterministic_under_seeded_rng():
    f = fam(arrival=ArrivalModel(kind="observed_base_rate", rate_per_day=2.0,
                                 provenance="observed log"))
    a = sample_arrivals(f, t0=T0, t1=T0 + 3 * DAY, rng=random.Random(42))
    b = sample_arrivals(f, t0=T0, t1=T0 + 3 * DAY, rng=random.Random(42))
    assert a == b and a == _expected_poisson(42, 2.0, T0, T0 + 3 * DAY)
    assert a and all(T0 <= t < T0 + 3 * DAY for t in a)
    assert a == sorted(a)
    c = sample_arrivals(f, t0=T0, t1=T0 + 3 * DAY, rng=random.Random(7))
    assert c == _expected_poisson(7, 2.0, T0, T0 + 3 * DAY) and c != a


def test_zero_rate_yields_no_arrivals():
    f = fam(arrival=ArrivalModel(kind="grounded_scenario_data", rate_per_day=0.0,
                                 provenance="src"))
    assert sample_arrivals(f, t0=T0, t1=T0 + 30 * DAY, rng=random.Random(1)) == []


def test_scheduled_exact_filters_to_window():
    times = [T0 - 5.0, T0, T0 + 10.0, T0 + DAY - 1.0, T0 + DAY, T0 + 50 * DAY]
    f = fam(arrival=ArrivalModel(kind="scheduled_exact", scheduled_times=times))
    got = sample_arrivals(f, t0=T0, t1=T0 + DAY, rng=random.Random(0))
    assert got == [T0, T0 + 10.0, T0 + DAY - 1.0]          # [t0, t1) window


def test_uncertainty_band_draws_within_band_and_varies_across_branch_rngs():
    # wide band (ratio >= 10) → log-uniform intensity draw, then exponential gaps
    lo, hi = 0.05, 50.0
    f = fam(arrival=ArrivalModel(kind="documented_broad_prior", rate_per_day=1.0,
                                 provenance="documented prior", uncertainty_band=[lo, hi]))

    def replicate(seed):
        rng = random.Random(seed)
        rate = math.exp(rng.uniform(math.log(max(lo, 1e-9)), math.log(hi)))
        assert lo <= rate <= hi                             # the draw stays inside the band
        out, t = [], T0
        while True:
            u = rng.random()
            if u <= 0.0:
                u = 1e-12
            t += -math.log(u) / (rate / DAY)
            if t >= T0 + 10 * DAY:
                return rate, out
            out.append(t)

    rate5, exp5 = replicate(5)
    rate6, exp6 = replicate(6)
    assert sample_arrivals(f, t0=T0, t1=T0 + 10 * DAY, rng=random.Random(5)) == exp5
    assert sample_arrivals(f, t0=T0, t1=T0 + 10 * DAY, rng=random.Random(6)) == exp6
    assert rate5 != rate6 and exp5 != exp6                  # branch-to-branch spread
    # narrow band (ratio < 10) → plain uniform draw
    g = fam(arrival=ArrivalModel(kind="documented_broad_prior", rate_per_day=1.5,
                                 provenance="documented prior", uncertainty_band=[1.0, 2.0]))
    rng = random.Random(9)
    rate = rng.uniform(1.0, 2.0)
    expected, t = [], T0
    while True:
        u = rng.random()
        if u <= 0.0:
            u = 1e-12
        t += -math.log(u) / (rate / DAY)
        if t >= T0 + 5 * DAY:
            break
        expected.append(t)
    assert sample_arrivals(g, t0=T0, t1=T0 + 5 * DAY, rng=random.Random(9)) == expected


# ------------------------------------------------------------------ entry payload
def test_entry_event_payload_is_typed_and_never_terminal():
    f = fam(family_id="press_leak", impact_mechanism="observation_delivery",
            impact_description="a journalist emails the CFO",
            marks=["m0", "m1", "m2"],
            affected_boundary_components=["cfo"], observability_paths=["email"],
            arrival=ArrivalModel(kind="observed_base_rate", rate_per_day=0.1,
                                 provenance="press archive"))
    p = entry_event_payload(f, at=T0 + 5.0, branch_id="bZ", arrival_index=2,
                            rng=random.Random(7))
    assert p["outside_world_family"] == "press_leak"
    assert p["entry_mechanism"] == "observation_delivery"
    assert p["entry_mechanism"] in ENTRY_MECHANISMS
    assert p["mark"] == ["m0", "m1", "m2"][random.Random(7).randrange(3)]
    assert p["at"] == T0 + 5.0 and p["branch_id"] == "bZ" and p["arrival_index"] == 2
    assert p["arrival_kind"] == "observed_base_rate"
    assert p["arrival_provenance"] == "press archive"
    assert p["schema"] == OUTSIDE_SCHEMA
    # no terminal/forbidden field may ever ride on an entry payload
    assert not set(p) & set(FORBIDDEN_WRITES)
    for bad in FORBIDDEN_WRITES:
        assert bad not in p
    # deterministic mark fallbacks
    assert entry_event_payload(f, at=T0)["mark"] == "m0"                  # no rng → first mark
    bare = fam(family_id="bare", description="a bare family")
    assert entry_event_payload(bare, at=T0)["mark"] == "a bare family"    # no marks → description


# ------------------------------------------------------------------ generation (LLM)
OUTSIDE_JSON = json.dumps({
    "families": [
        {"family_id": "competitor_price_move", "description": "rival cuts subscription price",
         "marks": ["rival announces a 20 percent cut"],
         "affected_boundary_components": ["subscription_market"],
         "observability_paths": ["trade press"], "impact_mechanism": "price_change",
         "impact_description": "changes the market price input the sales team observes",
         "arrival": {"kind": "observed_base_rate", "rate_per_day": "0.05",
                     "provenance": "3 price moves in the 2022-2024 trade-press archive"},
         "evidence": ["trade-press archive"], "uncertainty": "broad_but_bounded",
         "promotion_trigger": "rival opens direct talks"},
        {"family_id": "minted_rate", "description": "regulator inquiry",
         "impact_mechanism": "institutional_rule_change",
         "impact_description": "adds a compliance constraint",
         "arrival": {"kind": "documented_broad_prior", "rate_per_day": 0.2, "provenance": ""},
         "uncertainty": "well_characterized"},
        {"family_id": "terminal_writer", "description": "news that decides everything",
         "affected_boundary_components": ["forecast answer"],
         "impact_mechanism": "observation_delivery",
         "impact_description": "directly sets the answer",
         "arrival": {"kind": "observed_base_rate", "rate_per_day": 1.0, "provenance": "src"}},
        {"family_id": "honest_unknown", "description": "platform ban",
         "impact_mechanism": "capacity_change",
         "impact_description": "removes a distribution channel",
         "arrival": {"kind": "unresolved"}, "uncertainty": "wild"},
        {"family_id": "bad_mechanism", "description": "mind control ray",
         "impact_mechanism": "direct_belief_write", "impact_description": "changes minds",
         "arrival": {"kind": "scheduled_exact", "scheduled_times": [T0 + DAY]}},
        "not-a-dict",
    ],
    "external_state_processes": [
        {"name": "market_sentiment", "description": "drifting demand mood",
         "provenance": "industry survey"}],
})


def _boundary():
    b = WorldBoundary(boundary_id="wbX", structural_model_id="smX",
                      question="will churn spike", as_of="2026-01-01", horizon="90d")
    b.components = [
        BoundaryComponent(component_id="c1", kind="individual_actor", name="sales_lead",
                          representation="individual", reason="r"),
        BoundaryComponent(component_id="c2", kind="external_event_family",
                          name="competitor_moves", representation="external_process",
                          reason="r"),
    ]
    b.rederive_views()
    return b


def test_generate_outside_world_parses_validates_and_records_unresolved_risks():
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        assert "RESIDUAL OUTSIDE-WORLD PROCESS" in prompt
        return OUTSIDE_JSON

    proc = generate_outside_world(_boundary(), llm=llm)
    assert "sales_lead" in prompts[0] and "competitor_moves" in prompts[0]
    assert len(proc.families) == 5                                   # non-dict entry skipped
    by_id = {f.family_id: f for f in proc.families}
    good = by_id["competitor_price_move"]
    assert good.arrival.kind == "observed_base_rate"
    assert good.arrival.rate_per_day == 0.05                         # string rate coerced
    assert good.validation_error == "" and good.uncertainty == "broad_but_bounded"
    assert [f.family_id for f in proc.samplable()] == ["competitor_price_move"]
    minted = by_id["minted_rate"]
    assert minted.arrival.kind == "unresolved" and minted.arrival.rate_per_day == 0.0
    assert "no provenance" in minted.evidence[-1]                    # honesty note survived
    assert by_id["terminal_writer"].validation_error                 # forbidden write rejected
    assert "'forecast_answer'" in by_id["terminal_writer"].validation_error
    assert by_id["honest_unknown"].uncertainty == "unresolved"       # invalid label coerced
    assert by_id["bad_mechanism"].validation_error
    risk_ids = [r["family_id"] for r in proc.unresolved_external_risks]
    assert sorted(risk_ids) == ["bad_mechanism", "honest_unknown", "minted_rate",
                                "terminal_writer"]
    assert len(risk_ids) == len(set(risk_ids))                       # no double counting
    for r in proc.unresolved_external_risks:
        assert r["why"]
    assert proc.external_state_processes == [
        {"name": "market_sentiment", "description": "drifting demand mood",
         "provenance": "industry survey"}]
    tr = proc.generation_trace[0]
    assert tr["ok"] is True and tr["prompt_hash"] and tr["response_hash"]


def test_generate_without_llm_records_trace_error_and_stays_unjustified():
    proc = generate_outside_world(_boundary(), llm=None)
    tr = proc.generation_trace[0]
    assert tr["ok"] is False and tr["error"] == "no_llm_backend" and tr["prompt_hash"]
    assert proc.families == [] and proc.unresolved_external_risks == []
    assert proc.empty_residual_justification == ""       # an unexplained empty residual stays unjustified


def test_generate_llm_exception_recorded():
    def broken(prompt):
        raise ValueError("api down")
    proc = generate_outside_world(_boundary(), llm=broken)
    tr = proc.generation_trace[0]
    assert tr["ok"] is False and "ValueError" in tr["error"]
    assert proc.families == [] and proc.empty_residual_justification == ""


def test_generate_empty_families_with_explicit_justification_is_kept():
    def llm(prompt):
        return json.dumps({"families": [],
                           "empty_residual_justification":
                               "the question is fully endogenous to the modeled team"})
    proc = generate_outside_world(_boundary(), llm=llm)
    assert proc.families == []
    assert proc.empty_residual_justification == \
        "the question is fully endogenous to the modeled team"


def test_unresolved_view_counts_a_doubly_bad_family_once():
    proc = OutsideWorldProcess(boundary_id="b1")
    both = validate_entry(fam(family_id="doubly_bad",
                              affected_boundary_components=["campaign success"],
                              arrival=ArrivalModel(kind="unresolved")))
    assert both.validation_error and both.arrival.kind == "unresolved"
    proc.families.append(both)
    assert [f.family_id for f in proc.unresolved()] == ["doubly_bad"]
    assert proc.samplable() == []


def test_process_hash_and_as_dict_round_trip():
    proc = OutsideWorldProcess(boundary_id="b1", structural_model_id="sm1")
    proc.families.append(validate_entry(fam(
        family_id="fA", arrival=ArrivalModel(kind="scheduled_exact",
                                             scheduled_times=[T0]))))
    d = proc.as_dict()
    assert d["process_hash"] == proc.process_hash() and len(d["process_hash"]) == 16
    assert d["families"][0]["family_id"] == "fA"
    assert d["families"][0]["arrival"]["kind"] == "scheduled_exact"
    assert d["schema_version"] == OUTSIDE_SCHEMA
    assert set(ARRIVAL_KINDS) >= {"observed_base_rate", "scheduled_exact", "unresolved"}
