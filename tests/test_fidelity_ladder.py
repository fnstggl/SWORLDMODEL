"""Test the fidelity ladder: adding noise variables hurts the naive weighting far more than shrinkage."""
import random

from swm.eval.fidelity_ladder import run_ladder


def _data(n, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = rng.gauss(0, 1)
        p = 1.0 / (1.0 + 2.718 ** (-1.6 * s))
        rows.append({"s": s, "y": 1 if rng.random() < p else 0,
                     "noise": [rng.gauss(0, 1) for _ in range(8)]})
    return rows


def test_shrinkage_absorbs_added_noise_variables_better_than_naive():
    train, test = _data(400, 0), _data(400, 99)
    specs = [("signal", lambda r: r["s"])] + [(f"noise{j}", (lambda r, j=j: r["noise"][j])) for j in range(8)]
    arms = {"naive": {"l2": 0.02, "integrate": False}, "strong": {"l2": 5.0, "integrate": False}}
    out = run_ladder(train, test, specs, arms=arms, extras=False)
    v = out["verdict"]
    # both start from the same 1-variable signal model; piling on 8 noise vars should hurt naive MORE
    assert v["naive"]["full_minus_best"] >= v["strong"]["full_minus_best"]
    assert len(out["curves"]["naive"]) == 9                 # a rung per added variable
