"""Tests for EXP-023 population-opinion prediction (GlobalOpinionQA via inferred country values)."""
import math
from pathlib import Path

from experiments.exp023_global_opinion import (
    VALUES_COMMITTED, _ce, _cos, _entropy_floor, _evaluate, _norm, _standardize, _tv,
)


def test_scoring_primitives():
    # cross-entropy is minimized when pred == actual (equals the distribution's entropy)
    a = [0.5, 0.3, 0.2]
    ent = -sum(x * math.log(x) for x in a)
    assert abs(_ce(a, a) - ent) < 1e-9
    assert _ce(a, [0.1, 0.1, 0.8]) > _ce(a, a)          # wrong pred costs more
    # total variation: 0 for identical, 1 for disjoint
    assert _tv(a, a) == 0.0
    assert abs(_tv([1, 0], [0, 1]) - 1.0) < 1e-9
    # cosine: identical dir = 1, opposite = -1
    assert abs(_cos([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(_cos([1, 0], [-1, 0]) + 1.0) < 1e-9
    # normalize
    assert abs(sum(_norm([2, 2, 4])) - 1.0) < 1e-9


def test_standardize_zero_mean_unit_var():
    vals = {"a": {"religiosity": 0.9}, "b": {"religiosity": 0.1}, "c": {"religiosity": 0.5}}
    sv = _standardize(vals)
    col0 = [sv[k][0] for k in ("a", "b", "c")]
    assert abs(sum(col0)) < 1e-6                          # mean ~0 after standardizing
    assert sv["a"][0] > sv["c"][0] > sv["b"][0]           # order preserved


def test_committed_country_values_present_and_wellformed():
    import json
    vals = json.loads(Path(VALUES_COMMITTED).read_text())
    assert len(vals) >= 50
    from experiments.exp023_global_opinion import VDIMS
    for c, v in vals.items():
        assert set(VDIMS).issubset(v)                    # every country has all value dims
        assert all(0.0 <= float(v[d]) <= 1.0 for d in VDIMS)


def test_inferred_values_beat_global_mean_when_data_present():
    """No-cheat integration: predict unseen test countries from train countries by value-similarity.
    Skips gracefully if the (gitignored) survey CSV isn't downloaded in this environment."""
    import json
    from experiments.datasets_globalopinion import CSV, load
    if not Path(CSV).exists():
        return  # data-access dependent; the loader docstring documents the fetch
    recs = load()
    vals = json.loads(Path(VALUES_COMMITTED).read_text())
    sv = _standardize(vals)
    countries = sorted(set(vals))
    cut = int(0.7 * len(countries))
    train_c, test_c = set(countries[:cut]), set(countries[cut:])
    m = _evaluate(recs, sv, train_c, test_c, beta=4, hybrid_w=1.0)
    base_ce = m["base"]["ce"] / m["base"]["n"]
    val_ce = m["values"]["ce"] / m["values"]["n"]
    floor = _entropy_floor(recs, set(vals))
    assert val_ce < base_ce                              # inferred values help vs the global mean
    assert floor < val_ce < base_ce                      # and stay above the irreducible entropy floor
