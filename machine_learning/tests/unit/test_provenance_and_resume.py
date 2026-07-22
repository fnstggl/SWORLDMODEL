"""Provenance completeness + checkpoint-resume bookkeeping (torch-free parts)."""
import json

from machine_learning.canonical import make_record
from machine_learning.validation.provenance import _render
from machine_learning.training.resume import latest_checkpoint, resume_step


def _rec():
    return make_record(
        dataset_id="demo", task_type="PREDICT_NEXT_CHOICE", converter="m.C",
        converter_version="1.0.0", license_class="cc_by", episode_id="e", sequence_index=0,
        payload={"input": {"observation": {}}, "target": {"choice": "A", "acted": True}},
        raw_locator={"files": ["raw/x.csv"], "indices": [3], "ids": ["row3"]},
        transformation_steps=["read row", "map choice"])


def test_provenance_render_is_complete():
    r = _rec()
    lineage = _render(r)
    assert lineage["dataset_id"] == "demo"
    assert lineage["converter"] == "m.C"
    assert lineage["raw_source"]["files"] == ["raw/x.csv"]
    assert lineage["raw_source"]["record_ids"] == ["row3"]
    assert lineage["content_hash"]


def test_resume_bookkeeping(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoint-0000005").mkdir(parents=True)
    (run / "checkpoint-0000005" / "trainer_state.json").write_text(json.dumps({"step": 5}))
    (run / "latest").write_text("checkpoint-0000005")
    assert latest_checkpoint(run).name == "checkpoint-0000005"
    assert resume_step(run) == 5


def test_resume_empty(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert resume_step(tmp_path) == 0
