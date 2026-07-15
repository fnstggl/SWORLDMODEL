"""Post-snapshot pool invariants; no network or outcome store access."""
import hashlib
from types import SimpleNamespace

from experiments.post_snapshot_benchmark.build_capsules import _canonical_bytes, reduce_fingerprint
from experiments.post_snapshot_benchmark.forecast import PHASES, _atomic_write_attempt, qualify
from experiments.post_snapshot_benchmark.build_pool import (
    collapse_independence_families,
    eligible_candidates,
    family_key,
    select_representative,
)
from swm.world_model_v2.fallback import _terminal_value_compatible
from experiments.activation200 import _trajectory_signature
from swm.replay.probes2 import classify_row_v2


def _source(event_id, title, *, yes_won=True, domain_hint="", opened_day=1):
    prices = '["1", "0"]' if yes_won else '["0", "1"]'
    return {
        "id": str(event_id), "title": title, "description": domain_hint,
        "createdAt": f"2026-05-{opened_day:02d}T00:00:00Z",
        "closedTime": f"2026-06-{opened_day:02d}T00:00:00Z",
        "markets": [{
            "id": f"m{event_id}", "question": title, "description": domain_hint,
            "outcomes": '["Yes", "No"]', "outcomePrices": prices,
            "volume": "1000", "umaResolutionStatus": "resolved",
            "clobTokenIds": '["yes-token", "no-token"]',
        }],
    }


def test_intraday_threshold_variants_share_an_independence_family():
    first = family_key("Bitcoin above $100,000 on July 12, 3PM ET?", "crypto")
    second = family_key("Bitcoin above $105,000 on July 12, 4PM ET?", "crypto")
    assert first == second
    assert first != family_key("Bitcoin above $100,000 on July 13, 3PM ET?", "crypto")


def test_eligibility_does_not_depend_on_which_binary_side_won():
    yes, yes_resolution = eligible_candidates([_source(1, "Will Alpha happen?", yes_won=True)])
    no, no_resolution = eligible_candidates([_source(1, "Will Alpha happen?", yes_won=False)])
    assert yes == no
    assert yes_resolution["pm_1"]["outcome"] == 1
    assert no_resolution["pm_1"]["outcome"] == 0


def test_family_collapse_keeps_one_world_and_preserves_membership():
    candidates, _ = eligible_candidates([
        _source(1, "Bitcoin above $100,000 on July 12, 3PM ET?", opened_day=1),
        _source(2, "Bitcoin above $105,000 on July 12, 4PM ET?", opened_day=2),
    ])
    independent, mapping = collapse_independence_families(candidates)
    assert len(independent) == 1
    family = next(iter(mapping.values()))
    assert family["n_source_events"] == 2
    assert set(family["source_event_ids"]) == {"pm_1", "pm_2"}


def test_exact_selection_and_chronological_40_20_40_split():
    candidates = []
    domains = ("politics", "sports", "crypto", "technology", "business", "culture")
    for index in range(180):
        day = index + 1
        candidates.append({
            "event_id": f"event_{index:03d}", "domain": domains[index % len(domains)],
            "question_open_time": f"2026-05-{1 + index // 24:02d}T{index % 24:02d}:00:00Z",
            "resolution_time": f"2026-06-{1 + index // 24:02d}T{index % 24:02d}:00:00Z",
        })
    selected = select_representative(candidates)
    assert len(selected) == 100
    assert len({row["event_id"] for row in selected}) == 100
    assert [row["split"] for row in selected].count("calibration") == 40
    assert [row["split"] for row in selected].count("validation") == 20
    assert [row["split"] for row in selected].count("locked_test") == 40
    assert max([row["domain"] for row in selected].count(domain) for domain in domains) <= 30
    assert all(len(row["forecast_cutoffs"]) == 4 for row in selected)
    assert [row["question_open_time"] for row in selected] == sorted(
        row["question_open_time"] for row in selected)


def test_tier_c_transform_blinds_identity_date_time_and_long_quote():
    text = ('Alice Example meets Example Corp on July 12, 2026 at 3PM ET. '
            'The source says "This identifying quotation is intentionally quite long."')
    blinded, steps = reduce_fingerprint(
        text, {"Alice Example": "Person A", "Example Corp": "Organization A"})
    assert "Alice Example" not in blinded
    assert "Example Corp" not in blinded
    assert "July 12" not in blinded and "3PM" not in blinded and "2026" not in blinded
    assert "identifying quotation removed" in blinded
    assert "stable_entity_pseudonymization" in steps


def test_canonical_source_bytes_match_pool_hash_contract():
    world = {
        "source_event_id": "12", "source_market_id": "34", "source_condition_id": "condition",
        "question_open_time": "2026-05-01T00:00:00Z", "resolution_time": "2026-06-01T00:00:00Z",
        "question": "Will it happen?", "description": "Description", "resolution_rule": "Official",
    }
    expected = {"event_id": "12", "market_id": "34", "condition_id": "condition",
                "created_at": "2026-05-01T00:00:00Z", "resolution_time": "2026-06-01T00:00:00Z",
                "question": "Will it happen?", "description": "Description",
                "resolution_rule": "Official"}
    import json
    assert hashlib.sha256(_canonical_bytes(world)).hexdigest() == hashlib.sha256(
        json.dumps(expected, sort_keys=True).encode()).hexdigest()


def test_binary_safety_net_overwrites_noncontract_numeric_domain_write():
    assert _terminal_value_compatible("yes", "binary", ["yes", "no"])
    assert _terminal_value_compatible(1.0, "binary", ["yes", "no"])
    assert not _terminal_value_compatible(0.42, "binary", ["yes", "no"])


def test_full_system_qualification_requires_all_eleven_typed_records():
    records = {phase: {"relevant": False, "execution_status": "no_op_causally_irrelevant"}
               for phase in PHASES}
    result = SimpleNamespace(
        has_forecast=lambda: True, raw_probability=0.5, raw_distribution={"yes": 0.5, "no": 0.5},
        provenance={"phase_execution_records": records, "fully_integrated": True})
    assert qualify(result) == (True, [])
    records["phase9_networks"] = {"relevant": True, "execution_status": "blocked_missing_state"}
    passed, failures = qualify(result)
    assert not passed
    assert any("phase9_networks:relevant_not_active" in failure for failure in failures)


def test_attempt_shards_and_aggregate_are_atomic_and_preserve_retries(tmp_path):
    output = tmp_path / "rows.jsonl"
    row = {"event_id": "event", "forecast_cutoff": "2026-05-01T00:00:00Z", "value": 1}
    _atomic_write_attempt(output, row)
    _atomic_write_attempt(output, {**row, "value": 2})
    import json
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [entry["value"] for entry in rows] == [1, 2]
    assert len(list((tmp_path / "rows_rows").glob("*.json"))) == 2


def test_activation_trajectory_signature_detects_state_path_change():
    delta = SimpleNamespace(at=1.0, operator="network_diffusion", event_type="network_diffusion",
                            changes=[{"path": "quantities[reach]"}])
    full = [SimpleNamespace(log=[delta])]
    ablated = [SimpleNamespace(log=[])]
    assert _trajectory_signature(full)["sha256"] != _trajectory_signature(ablated)["sha256"]


def test_leakage_classifier_checks_nested_permutation_and_parse_failures():
    base = {"name_only": {"output": {}}, "recognition": {"output": {"identified": False}},
            "identity_permutation": {
                "base": {"output": {"p_yes": 0.1}},
                "permuted": {"output": {"p_yes": 0.8}},
            }}
    assert classify_row_v2(base, arm="causally_blinded_historical",
                           name_only_correct=None) == "contamination_susceptible"
    nested_error = {
        "name_only": {"output": {}}, "recognition": {"output": {"identified": False}},
        "identity_permutation": {"base": {"output": {"parse_failed": True}}},
    }
    assert classify_row_v2(nested_error, arm="causally_blinded_historical",
                           name_only_correct=None) == "uncertain_leakage"
