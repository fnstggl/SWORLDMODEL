"""Tests for the navigable outcome object (distribution + reducible/irreducible + pivotal-branch discovery)."""
import random

from swm.api.model_spec import parse_spec
from swm.report.navigable import navigable_from_samples, navigable_from_spec


def test_pivotal_branch_is_discovered():
    """The factor that actually drives the outcome should be surfaced as the top pivot; a noise factor not."""
    rng = random.Random(0)
    samples = []
    for _ in range(2000):
        driver = rng.gauss(0, 1)          # the real cause
        noise = rng.gauss(0, 1)           # irrelevant
        outcome = 1.0 if driver > 0 else 0.0
        samples.append((outcome, {"driver": driver, "noise": noise}))
    nav = navigable_from_samples(samples)
    assert nav.pivots[0].factor == "driver"
    assert nav.pivots[0].outcome_high > 0.8 and nav.pivots[0].outcome_low < 0.2   # clean fork
    # the driver explains far more than the noise factor
    explained = {p.factor: p.explained for p in nav.pivots}
    assert explained["driver"] > 5 * explained.get("noise", 0.0)


def test_numeric_distribution_and_quantiles():
    rng = random.Random(1)
    samples = [(rng.gauss(0.5, 0.1), {}) for _ in range(3000)]
    nav = navigable_from_samples(samples)
    assert nav.kind == "numeric" and abs(nav.mean - 0.5) < 0.02
    assert nav.quantiles["p05"] < nav.quantiles["p50"] < nav.quantiles["p95"]


def test_categorical_defaults_to_modal_target():
    samples = [(("A" if i % 3 else "B"), {}) for i in range(300)]   # A ~ 2/3, B ~ 1/3
    nav = navigable_from_samples(samples)
    assert nav.kind == "categorical" and nav.point == "A"
    assert nav.target_desc == "P(A)" and abs(nav.target_value - 2 / 3) < 0.05


def test_decomp_fields_passthrough():
    samples = [(0.4, {}) for _ in range(50)]
    nav = navigable_from_samples(samples, decomp={"reducible_sd": 0.01, "irreducible_sd": 0.05,
                                                  "irreducible_frac": 0.9, "forecastable": False})
    assert nav.irreducible_frac == 0.9 and nav.forecastable is False


def test_navigable_from_generic_scm_spec_has_split():
    spec = parse_spec({"mechanism": "generic_scm",
                       "variables": [{"name": "x", "value": 0.5, "est_sd": 0.05, "volatility": 0.03}],
                       "equations": {"x": "0.0"}, "outcome": {"variable": "x"}, "horizon": 6})
    nav = navigable_from_spec(spec, n=3000, seed=0)
    assert nav.kind == "numeric" and nav.irreducible_frac is not None
    assert nav.pivots and any("x" in p.factor for p in nav.pivots)
    assert "irreducible" in nav.summary()
