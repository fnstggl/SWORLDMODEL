"""Real-world fidelity focused tests (Phase B milestone 1: direct correctness bugs D1,D3,D4,D5,D6).

Universal machinery only — no question-specific logic. These reproduce the concrete EXP-113
defects and lock their fixes."""
from __future__ import annotations

import pytest

from swm.world_model_v2.lean_v2.blueprint import ConsumerWorldBlueprint
from swm.world_model_v2.lean_v2.canonical_options import (build_option_set, normalize_option,
                                                          normalize_to_label, strip_menu_prefix)
from swm.world_model_v2.lean_v2.readiness import (CANONICAL_TERMINAL_KEY,
                                                  canonicalize_terminal_writers,
                                                  pure_terminal_outcome, terminal_round_trip)
from swm.world_model_v2.lean_v2.resolution_spec import (BOOLEAN_EVENT, INSTITUTION_VOTE,
                                                        NUMERIC_THRESHOLD, parse_resolution,
                                                        spec_matches_blueprint)

OPTS = ["Maintain at 0.75%", "Raise to 1.0%"]


# ============================================================ D1 — canonical option identity
# 1 — `vote:Raise to 1.0%` maps to the correct canonical option (the BoJ Ueda drop)
def test_1_vote_prefix_maps_correctly():
    assert normalize_to_label("vote:Raise to 1.0%", OPTS) == "Raise to 1.0%"
    assert normalize_to_label("Vote: Raise to 1.0%", OPTS) == "Raise to 1.0%"
    assert normalize_to_label("  RAISE TO 1.0% ", OPTS) == "Raise to 1.0%"
    assert normalize_to_label("cast_vote: Maintain at 0.75%", OPTS) == "Maintain at 0.75%"
    assert strip_menu_prefix("vote:Raise to 1.0%") == "Raise to 1.0%"


# 2 — an unknown option does NOT silently map to another option
def test_2_unknown_option_never_silently_maps():
    assert normalize_to_label("Cut rates by 50bp", OPTS) is None
    assert normalize_to_label("abstain entirely", OPTS) is None
    # a genuinely ambiguous string that could match two options returns None, never a guess
    amb = build_option_set(["approve plan A", "approve plan B"])
    assert normalize_option("approve", amb) is None


# ============================================================ D3 — no alphabetical fallback
# 5 — no first/lexicographic/fixed-default action fallback survives in the engine source
def test_5_no_alphabetical_or_first_option_fallback():
    import inspect
    import swm.world_model_v2.lean_v2.engine as E
    src = inspect.getsource(E)
    assert "sorted(allowed)[0]" not in src
    assert "sorted(allowed_opts)[0]" not in src
    # the deadline path must not fabricate the lowest option
    assert "default lowest" not in src


# a required participant whose state action does not map to an option is a labeled failure,
# never a fabricated lowest-option vote
def test_5b_force_vote_labels_failure_not_fabricates():
    from swm.world_model_v2.lean_v2.canonical_options import normalize_to_label as n
    # the precommitment fallback only fires on a mappable action; otherwise None
    assert n("keep waiting and see", OPTS) is None
    assert n("raise the policy rate to 1.0 percent", OPTS) == "Raise to 1.0%"


# ============================================================ D4 — type-safe canonicalization
def _numeric_bp():
    return ConsumerWorldBlueprint(
        resolution={"interpretation": "Resolves YES if the daily transit count reaches 50 or "
                                      "more on any single day", "options": ["Yes", "No"]},
        actors=[{"id": "op", "private_state_variants": []}],
        action_templates=[{"action_id": "declare", "actor_ids": ["op"], "writes_terminal": True,
                           "effects": [{"kind": "set_state",
                                        "params": {"key": "daily_transit_count", "value": "7"}}]}],
        terminal={"kind": "state_predicate", "yes_when": "daily transit count >= 50",
                  "threshold": 50, "no_when": "fewer than 50", "evaluation_day": "2026-06-01"})


def _boolean_bp():
    return ConsumerWorldBlueprint(
        resolution={"interpretation": "Apple announces a new major version of visionOS at WWDC "
                                      "2026", "options": ["Yes", "No"]},
        actors=[{"id": "cook", "private_state_variants": []}],
        action_templates=[{"action_id": "announce", "actor_ids": ["cook"], "writes_terminal": True,
                           "effects": [{"kind": "set_state",
                                        "params": {"key": "keynote_visionos", "value": "true"}}]}],
        terminal={"kind": "event_occurs", "yes_when": "visionOS announced",
                  "evaluation_day": "2026-06-12"})


# 7 — numeric terminal canonicalization is SKIPPED (never collapses the numeric variable)
def test_7_numeric_canonicalization_is_type_safe():
    bp = _numeric_bp()
    rec = canonicalize_terminal_writers(bp)
    assert rec.get("type_safe_skip") is True and rec["terminal_kind"] == NUMERIC_THRESHOLD
    # the numeric writer's key is untouched — NOT rewritten to the boolean canonical key
    key = bp.action_templates[0]["effects"][0]["params"]["key"]
    assert key == "daily_transit_count" and key != CANONICAL_TERMINAL_KEY


# 8 — boolean-event canonicalization still works (visionOS keeper preserved)
def test_8_boolean_canonicalization_still_works():
    bp = _boolean_bp()
    rec = canonicalize_terminal_writers(bp)
    assert rec.get("needed") is True
    assert bp.action_templates[0]["effects"][0]["params"]["key"] == CANONICAL_TERMINAL_KEY
    out = pure_terminal_outcome(bp, world_state={CANONICAL_TERMINAL_KEY: True})
    assert out["resolved"] and out["outcome"] == "YES"


# ============================================================ D5 — numeric threshold parsing
# 6 — numeric threshold 50 is extracted correctly across phrasings
def test_6_numeric_threshold_parsed():
    for phrasing, comp, thr in [
        ("at least 50 tankers per day", ">=", 50.0),
        ("50 or more on any single day", ">=", 50.0),
        ("more than 50", ">", 50.0),
        ("fewer than 50", "<", 50.0),
        ("greater than or equal to 50 transits", ">=", 50.0)]:
        s = parse_resolution(phrasing)
        assert s.terminal_kind == NUMERIC_THRESHOLD, phrasing
        assert s.comparator == comp and s.threshold == thr, phrasing
    assert parse_resolution("reaches 50 or more on any single day").aggregation_window == "any_day"
    # vote phrasings
    assert parse_resolution("Wale receives at least 26 votes").terminal_kind == INSTITUTION_VOTE
    assert parse_resolution("at least 26 votes").vote_threshold == 26.0
    assert parse_resolution("majority of 9 members").vote_of_total == 9
    assert parse_resolution("unanimous 5-of-5").vote_rule == "unanimity"


# a numeric resolution against a boolean-only blueprint terminal is flagged (parser vs blueprint)
def test_6b_spec_blueprint_mismatch_flagged():
    bp = ConsumerWorldBlueprint(
        resolution={"interpretation": "Resolves YES if the daily transit count reaches 50 or "
                                      "more on any single day", "options": ["Yes", "No"]},
        terminal={"kind": "event_occurs", "yes_when": "disruption occurred"})
    spec = parse_resolution(bp.resolution["interpretation"])
    rep = spec_matches_blueprint(spec, bp)
    assert rep["ok"] is False and "numeric threshold" in rep["mismatches"][0]


# ============================================================ D6 — event-absence writer
# 9 — a boolean deadline event that did not occur resolves NO (not unresolved)
def test_9_event_absence_resolves_no():
    bp = _boolean_bp()
    # nothing written → the event did not occur by the deadline → NO (event_absent)
    out = pure_terminal_outcome(bp, world_state={})
    assert out["resolved"] is True and out["outcome"] == "NO"
    assert "event_absent" in out["detail"]["note"]


# a NUMERIC terminal with no mechanism stays honestly unresolved (not a false NO)
def test_9b_numeric_absence_stays_unresolved():
    bp = _numeric_bp()
    out = pure_terminal_outcome(bp, world_state={}, terminal_kind=NUMERIC_THRESHOLD)
    assert out["resolved"] is False and "numeric" in out["cause"]


# 10 — the synthetic round-trip still proves YES->1 / NO->0 through the live path
def test_10_round_trip_still_holds_for_boolean():
    bp = _boolean_bp()
    canonicalize_terminal_writers(bp)
    rt = terminal_round_trip(bp)
    assert rt["ok"] is True
