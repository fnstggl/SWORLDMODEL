"""Canonical builder + schema validation."""
import pytest

from machine_learning.canonical import make_record
from machine_learning.tasks import TASK_TYPES
from machine_learning.validation.schema_validation import payload_schema, validate_record


def _msg_record(**over):
    kw = dict(dataset_id="demo", task_type="PREDICT_NEXT_MESSAGE", converter="d.C",
              converter_version="1", license_class="cc_by", episode_id="e1", sequence_index=1,
              actor_id="a", context={"known_history": [{"index": 0, "text": "hi"}]},
              payload={"input": {"dialogue_history": [{"text": "hi"}]},
                       "target": {"message_text": "hello there", "dialogue_act": None, "strategy": None}},
              raw_locator={"files": ["f"], "indices": [0], "ids": ["e1"]})
    kw.update(over)
    return make_record(**kw)


def test_valid_record_passes():
    assert validate_record(_msg_record()).ok


def test_every_task_has_payload_schema():
    for t in TASK_TYPES:
        assert payload_schema(t) is not None, t


def test_empty_target_rejected():
    r = _msg_record(payload={"input": {}, "target": {"message_text": ""}})
    assert not validate_record(r).ok


def test_determinism_independent_of_timestamp():
    a = _msg_record(normalization_timestamp="2020-01-01T00:00:00Z")
    b = _msg_record(normalization_timestamp="2026-07-22T00:00:00Z")
    assert a["record_id"] == b["record_id"]
    assert a["provenance"]["content_hash"] == b["provenance"]["content_hash"]


def test_unknown_task_type_raises():
    with pytest.raises(ValueError):
        _msg_record(task_type="NOT_A_TASK")


def test_future_hidden_true_by_default():
    assert _msg_record()["cutoff"]["future_hidden"] is True


def test_missing_payload_shape_raises():
    with pytest.raises(ValueError):
        _msg_record(payload={"target": {"message_text": "x"}})
