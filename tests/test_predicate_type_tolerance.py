"""Regression (EXP-107 Hormuz crash): evaluate_predicate over LLM-compiled predicates and
record fields must never raise on ill-typed comparisons — an ill-typed comparison is simply
not satisfied (success never assumed), never a run-killing TypeError."""
from swm.world_model_v2.scenario_schema import evaluate_predicate


def _rec(fields):
    return {"record_type": "tanker_log", "fields": fields}


def test_list_field_probed_with_in_against_string_is_unsatisfied_not_a_crash():
    records = [_rec({"transits": ["t1", "t2"]})]
    p = {"record_type": "tanker_log", "field": "transits", "op": "in", "value": "confirmed"}
    assert evaluate_predicate(p, records) is False        # was: TypeError after 4h of rollout


def test_non_numeric_gte_is_unsatisfied_not_a_crash():
    records = [_rec({"count": "many"})]
    p = {"record_type": "tanker_log", "field": "count", "op": "gte", "value": "not_a_number"}
    assert evaluate_predicate(p, records) is False


def test_well_typed_predicates_still_evaluate():
    records = [_rec({"state": "open", "count": 51})]
    assert evaluate_predicate({"record_type": "tanker_log", "field": "state", "op": "eq",
                               "value": "open"}, records)
    assert evaluate_predicate({"record_type": "tanker_log", "field": "count", "op": "gte",
                               "value": 50}, records)
    assert evaluate_predicate({"record_type": "tanker_log", "field": "state", "op": "in",
                               "value": ["open", "closed"]}, records)
