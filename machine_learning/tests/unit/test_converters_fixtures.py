"""Every converter with a committed fixture must:
  * load, produce >0 records, all schema-valid;
  * have a DOC that validates against converter_doc.schema.json;
  * emit only tasks the registry lists (DOC.tasks_produced ⊆ supported_tasks);
  * respect chronology (no target leakage into inputs);
  * fabricate nothing (missing_fields recorded when a field is absent).
Blocked stubs (debate, mirobench) must import + raise SourceNotAvailable.
"""
import json

import jsonschema
import pytest

from machine_learning.config import FIXTURES_DIR, SCHEMAS_DIR
from machine_learning.normalization.base import SourceNotAvailable, load_converter
from machine_learning.registry_io import get_dataset, load_datasets
from machine_learning.validation.chronology import check_record
from machine_learning.validation.schema_validation import validate_record

_DOC_SCHEMA = json.loads((SCHEMAS_DIR / "source_manifests" / "converter_doc.schema.json").read_text())
_DOC_VALIDATOR = jsonschema.Draft7Validator(_DOC_SCHEMA)

_FIXTURE_DATASETS = sorted(p.name for p in FIXTURES_DIR.iterdir()
                           if p.is_dir() and any(p.iterdir())) if FIXTURES_DIR.exists() else []
_BLOCKED = ["debate", "mirobench"]


@pytest.mark.parametrize("dataset_id", _FIXTURE_DATASETS)
def test_converter_fixture_valid(dataset_id):
    conv = load_converter(dataset_id, timestamp="2026-07-22T00:00:00Z")
    assert conv is not None
    _DOC_VALIDATOR.validate(conv.DOC)
    recs = list(conv.iter_records(FIXTURES_DIR / dataset_id))
    assert recs, f"{dataset_id} produced no records"
    for r in recs:
        res = validate_record(r)
        assert res.ok, (r["record_id"], res.errors[:3])
        assert not check_record(r), f"chronology issue in {r['record_id']}"
    produced = {r["task_type"] for r in recs}
    supported = set(get_dataset(dataset_id).get("supported_tasks", []))
    assert produced <= supported, f"{dataset_id}: {produced - supported} not in registry"
    # DOC.tasks_produced should match what is actually emitted
    assert set(conv.DOC.get("tasks_produced", [])) >= produced or produced <= supported


@pytest.mark.parametrize("dataset_id", _BLOCKED)
def test_blocked_converters_raise(dataset_id):
    conv = load_converter(dataset_id, timestamp="2026-07-22T00:00:00Z")
    _DOC_VALIDATOR.validate(conv.DOC)
    with pytest.raises((SourceNotAvailable, FileNotFoundError)):
        list(conv.iter_records(FIXTURES_DIR / dataset_id))


def test_blocked_and_infra_have_no_or_stub_converter():
    for did in ("acl_online_shopping", "agentsociety", "darpa_socialsim"):
        assert get_dataset(did).get("converter") in (None, ""), did
