"""D12 — shared condition graph + tail preservation. Universal machinery only.

Locks: actors under a common cause are correlated through it (never independently multiplied);
conditional dependencies are enumerated jointly (not an independent product); mutually-exclusive
worlds are never created; low-probability tail worlds are preserved (bounded, never renormalized
to zero); unidentified dependence is reported as sensitivity, not invented."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.shared_conditions import (
    ConditionNode, DependencyEdge, SharedConditionGraph, TAIL_FLOOR)


def _g(*nodes):
    g = SharedConditionGraph()
    for n in nodes:
        g.add_node(n)
    return g


# ============================================================ common cause → correlated actors
def test_common_cause_correlates_affected_actors():
    g = _g(ConditionNode("inflation", ["high", "low"], {"high": 0.7, "low": 0.3},
                         affects_actors=["m1", "m2", "m3"]))
    corr = g.correlated_actors()
    assert corr["inflation"] == ["m1", "m2", "m3"]           # all three share the cause
    # sampling the condition first yields ONE state per world that all three respond to
    worlds = g.joint_worlds()["worlds"]
    assert all(set(c.keys()) == {"inflation"} for c, _w in worlds)
    assert abs(dict((tuple(c.items())[0], w) for c, w in worlds)[("inflation", "high")] - 0.7) < 1e-6


# ============================================================ 34 — no independent multiplication
def test_34_conditional_dependency_is_not_an_independent_product():
    g = _g(ConditionNode("inflation", ["high", "low"], {"high": 0.5, "low": 0.5}),
           ConditionNode("growth", ["strong", "weak"], {"strong": 0.5, "weak": 0.5}))
    # high inflation makes weak growth much more likely (anti-correlation), not independent 0.25
    g.add_dependency(DependencyEdge("inflation", "growth",
                                    {"high": {"strong": 0.2, "weak": 0.8},
                                     "low": {"strong": 0.8, "weak": 0.2}}))
    w = {tuple(sorted(c.items())): x for c, x in g.joint_worlds()["worlds"]}
    high_weak = w[(("growth", "weak"), ("inflation", "high"))]
    high_strong = w[(("growth", "strong"), ("inflation", "high"))]
    assert high_weak > 0.35 and high_strong < 0.15           # correlated, not 0.25/0.25
    assert abs(high_weak - 0.4) < 1e-6


# ============================================================ 35 — tail preserved, not pruned
def test_35_low_probability_tail_is_preserved_not_renormalized_away():
    g = _g(ConditionNode("shock", ["none", "crisis"], {"none": 0.995, "crisis": 0.005}))
    jw = g.joint_worlds()
    kept = {tuple(c.items())[0][1] for c, _w in jw["worlds"]}
    assert "crisis" not in kept                              # sub-floor → not a kept world...
    assert jw["tail_mass"] == 0.005                          # ...but its mass is PRESERVED as a bound
    # and the crisis world is exposed for a reversal check, never silently dropped
    assert any(c.get("shock") == "crisis" for c, _w in jw["tail_worlds"])


def test_35b_above_floor_tail_world_is_always_kept():
    g = _g(ConditionNode("shock", ["none", "crisis"], {"none": 0.97, "crisis": 0.03}))
    kept = {tuple(c.items())[0][1] for c, _w in g.joint_worlds()["worlds"]}
    assert "crisis" in kept                                  # 0.03 >= floor 0.01 → kept (reversal-safe)


# ============================================================ mutual exclusivity
def test_mutually_exclusive_worlds_are_not_created():
    g = _g(ConditionNode("boom", ["holds", "not"], {"holds": 0.5, "not": 0.5}),
           ConditionNode("recession", ["holds", "not"], {"holds": 0.5, "not": 0.5}))
    g.exclusive_groups.append([("boom", "holds"), ("recession", "holds")])
    worlds = [c for c, _w in g.joint_worlds()["worlds"]]
    # a world where both boom and recession hold is impossible and never enumerated
    assert not any(c.get("boom") == "holds" and c.get("recession") == "holds" for c in worlds)


# ============================================================ unidentified dependence → sensitivity
def test_unidentified_dependence_reports_sensitivity_not_invented_correlation():
    g = _g(ConditionNode("a", ["x", "y"], {"x": 0.5, "y": 0.5}),
           ConditionNode("b", ["x", "y"], {"x": 0.5, "y": 0.5}))
    g.add_dependency(DependencyEdge("a", "b", conditional={}, identified=False))
    ds = g.dependence_structures()
    assert ds and ds[0]["structures"] == ["independent", "comonotonic"]
    assert "sensitivity" in ds[0]["note"]


# ============================================================ manifest
def test_manifest_reports_structure_and_preserved_tail():
    g = _g(ConditionNode("infl", ["high", "low"], {"high": 0.6, "low": 0.4},
                         affects_actors=["m1", "m2"]))
    m = g.manifest()
    assert m["correlated_actors"]["infl"] == ["m1", "m2"]
    assert "preserved_tail_mass" in m and m["kept_worlds"] >= 1
