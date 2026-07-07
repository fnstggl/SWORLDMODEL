"""Guard: the single-source `build_sampler` must reproduce the `_run_*` aggregates (no divergent code path),
and each mechanism's traced sampler must record exogenous factors for the navigable layer."""
from swm.api.compiler import CompiledModel, build_sampler
from swm.api.model_spec import parse_spec
from swm.simulation.structural import montecarlo


def _mc_mean(spec, seed, n=3000):
    return montecarlo(build_sampler(spec).once, n=n, seed=seed)["mean"]


def test_generic_scm_sampler_matches_run():
    spec = parse_spec({"mechanism": "generic_scm",
                       "variables": [{"name": "v", "value": 0.49, "est_sd": 0.01, "volatility": 0.02}],
                       "equations": {"v": "0.2*(0.55 - v)"}, "outcome": {"variable": "v"}, "horizon": 10})
    run = CompiledModel(spec).run(n=3000)
    assert round(_mc_mean(spec, seed=0), 4) == run["mean"]       # identical closure & seed (run rounds to 4dp)
    # traced records the epistemic initial draw and the aleatoric shock as exogenous factors
    _, factors = build_sampler(spec).traced(__import__("random").Random(0))
    assert "v@0" in factors and "v~shock" in factors


def test_bracket_sampler_matches_run_favorite():
    spec = parse_spec({"mechanism": "bracket", "outcome": {"target": "A"},
                       "extra": {"competitors": [{"name": "A", "strength": 1660, "est_sd": 40},
                                                 {"name": "B", "strength": 1600, "est_sd": 40},
                                                 {"name": "C", "strength": 1560, "est_sd": 40},
                                                 {"name": "D", "strength": 1520, "est_sd": 40}]}})
    run = CompiledModel(spec).run(n=4000)
    dist = montecarlo(build_sampler(spec).once, n=4000, seed=run and 1)["distribution"]  # run used seed+1
    assert max(dist, key=dist.get) == run["favorite"] == "A"
    _, factors = build_sampler(spec).traced(__import__("random").Random(0))
    assert "A~strength" in factors


def test_committee_electorate_single_agent_traced_factors():
    comm = parse_spec({"mechanism": "committee", "outcome": {"event": {"op": ">", "value": 0.5}},
                       "extra": {"agents": [{"id": f"a{i}", "position": p} for i, p in
                                            enumerate([0.8, 0.3, 0.55])], "rounds": 3}})
    _, cf = build_sampler(comm).traced(__import__("random").Random(0))
    assert any(k.startswith("a0") for k in cf)

    elec = parse_spec({"mechanism": "electorate", "outcome": {"event": {"op": ">", "value": 0.5}},
                       "extra": {"cells": [{"id": "x", "stance": 0.6, "weight": 3, "est_sd": 0.03},
                                           {"id": "y", "stance": 0.4, "weight": 2, "est_sd": 0.03}]}})
    _, ef = build_sampler(elec).traced(__import__("random").Random(0))
    assert "x~stance" in ef

    sa = parse_spec({"mechanism": "single_agent",
                     "extra": {"person": {"trait_openness": 0.7}, "est_sd": {"trait_openness": 0.05},
                               "message": {"clarity": 0.8}}})
    _, sf = build_sampler(sa).traced(__import__("random").Random(0))
    assert "trait_openness~est" in sf
