"""Integration: fixture -> canonical -> split -> leakage-free -> formatted (offline)."""
import pytest

from machine_learning.config import FIXTURES_DIR
from machine_learning.normalization.base import load_converter
from machine_learning.examples.formatters.sft import format_record
from machine_learning.splitting.policies import SplitPolicy, assign_split
from machine_learning.validation.schema_validation import validate_record

DATASETS = ["casino", "dealornodeal", "abcd", "open_bandit"]


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_pipeline_offline(dataset_id):
    conv = load_converter(dataset_id, timestamp="2026-07-22T00:00:00Z")
    recs = list(conv.iter_records(FIXTURES_DIR / dataset_id))
    assert recs

    # all valid + unique record ids
    ids = set()
    for r in recs:
        assert validate_record(r).ok
        ids.add(r["record_id"])
    assert len(ids) == len(recs)

    # deterministic content hashes
    for r in recs:
        assert r["provenance"]["content_hash"] == r["split_metadata"]["dedup_hash"]

    # split assignment leakage-free at the episode level
    pol = SplitPolicy.from_registry(dataset_id)
    ep_split = {}
    for r in recs:
        sp, _ = assign_split(r, pol)
        ep_split.setdefault(r["episode"]["episode_id"], set()).add(sp)
    assert all(len(s) == 1 for s in ep_split.values())

    # formatting: the target text lives strictly after the prompt (loss boundary)
    fx = format_record(recs[0])
    assert fx.text[fx.target_char_start:] == fx.completion


def test_target_not_in_prompt_for_messages():
    conv = load_converter("casino", timestamp="2026-07-22T00:00:00Z")
    for r in conv.iter_records(FIXTURES_DIR / "casino"):
        if r["task_type"] != "PREDICT_NEXT_MESSAGE":
            continue
        fx = format_record(r)
        msg = r["payload"]["target"]["message_text"]
        if len(msg) > 12:
            # the exact target message must not already be in the visible prompt
            assert msg not in fx.prompt
