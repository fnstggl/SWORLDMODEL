"""Registry parsing + consistency."""
import yaml

from machine_learning import registry_io as R
from machine_learning.config import REGISTRY_DIR
from machine_learning.tasks import TASK_TYPES


def test_registry_loads_all():
    ds = R.load_datasets()
    assert len(ds) == 23
    for did, e in ds.items():
        assert e["dataset_id"] == did


def test_registry_has_no_errors():
    issues = R.verify_registry()
    errors = [i for i in issues if i.severity == "error"]
    assert not errors, errors


def test_taxonomy_yaml_matches_code():
    tax = yaml.safe_load((REGISTRY_DIR / "task_taxonomy.yaml").read_text())
    assert set(tax["tasks"].keys()) == set(TASK_TYPES)


def test_every_license_class_defined():
    licenses = R.load_licenses()
    for e in R.load_datasets().values():
        assert e["license_class"] in licenses, e["license_class"]


def test_supported_tasks_are_valid():
    for e in R.load_datasets().values():
        for t in e.get("supported_tasks", []):
            assert t in TASK_TYPES


def test_eval_only_never_training_eligible():
    for did, e in R.load_datasets().items():
        if e["dataset_role"] in ("CROSS_DATASET_EVAL_ONLY", "LICENSE_RESTRICTED_EVAL_ONLY",
                                 "ACCESS_BLOCKED", "INFRASTRUCTURE_ONLY"):
            elig, _ = R.training_eligibility(did, require_approval=False)
            assert not elig


def test_nd_license_blocks_training():
    # BehaviorBench is CC-BY-NC-ND -> No-Derivatives forbids training
    elig, reason = R.training_eligibility("behaviorbench", require_approval=False)
    assert not elig
