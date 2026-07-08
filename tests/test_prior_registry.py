"""Tests for the learned-prior registry (the calibration flywheel) + calibrated-compiler hooks."""
import os

from swm.api.model_spec import parse_spec
from swm.api.calibrated_compiler import apply_registry, calibrate_from_data
from swm.variables.prior_registry import PriorRegistry, semantic_key


def test_semantic_key_canonicalizes_but_preserves_levels():
    assert semantic_key("Inflation Rate!", "Rate Hike") == semantic_key("inflation rate", "rate hike")
    assert semantic_key("party=democrat", "x") != semantic_key("party=republican", "x")   # levels distinct


def test_precision_weighted_combination_pulls_toward_tighter_evidence():
    reg = PriorRegistry()
    reg.update("inflation", "hike", mean=1.0, sd=1.0, n=50)          # loose
    reg.update("inflation", "hike", mean=2.0, sd=0.2, n=500)         # tight -> should dominate
    p = reg.get("inflation", "hike")
    assert 1.8 < p.mean < 2.0 and p.sd < 0.2                          # pulled toward the tight estimate, tighter
    assert reg.records[semantic_key("inflation", "hike")].n == 550    # evidence accumulates


def test_registry_override_tightens_compiler_weight():
    spec = parse_spec({"mechanism": "calibrated_readout", "extra": {"intercept": 0.0},
                       "variables": [{"name": "inflation", "value": 0.8, "weight": 1.0, "weight_sd": 0.8}],
                       "outcome": {"event": {"op": ">", "value": 0.5}}})
    reg = PriorRegistry()
    reg.update("inflation", "rate hike", mean=2.4, sd=0.15, n=800)
    s2 = apply_registry(spec, "rate hike", reg)
    v = s2.variables[0]
    assert v.weight > 1.8 and v.weight_sd < 0.8 and "registry" in v.weight_source


def test_calibrate_from_data_fits_weights_and_feeds_registry():
    import random
    rng = random.Random(0)
    rows = []
    for _ in range(500):
        x = rng.random()
        rows.append({"x": x, "y": 1 if rng.random() < x else 0})    # y driven by x
    spec = parse_spec({"mechanism": "calibrated_readout", "extra": {"intercept": 0.0},
                       "variables": [{"name": "driver", "value": 0.5, "weight": 0.0, "weight_sd": 3.0}],
                       "outcome": {"event": {"op": ">", "value": 0.5}}})
    reg = PriorRegistry()
    s2, model = calibrate_from_data(spec, rows, lambda r: [r["x"]], "adoption", registry=reg)
    assert s2.variables[0].weight > 0 and s2.variables[0].weight_source == "fit"   # positive elasticity learned
    assert reg.get("driver", "adoption") is not None      # fed the flywheel


def test_save_load_roundtrip(tmp_path):
    reg = PriorRegistry()
    reg.update("a", "b", 0.5, 0.3, n=10)
    path = str(tmp_path / "p.json")
    reg.save(path)
    reg2 = PriorRegistry.load(path)
    assert reg2.get("a", "b").mean == reg.get("a", "b").mean
