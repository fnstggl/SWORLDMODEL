"""D8 — action-grounded weighting. Universal machinery only.

Locks the EXP-113 defect: the number of generated prose stories must NEVER determine
probability. Probability over what an actor does is a COUNTED, hierarchically partial-pooled
rate over ACTION CLASSES; states live beneath their tendency and only split a tendency's mass
(for trajectory/sensitivity), never create it."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.action_baseline import (
    ActionCase, ActorActionBaseline, POOLING_TAU, build_action_baseline,
    cases_from_counted_class, partial_pool_categorical)


def _cases(spec):
    """spec: list of (action_class, level, count) → flat ActionCase list."""
    out = []
    for cls, lvl, k in spec:
        out += [ActionCase(action_class=cls, hierarchy_level=lvl) for _ in range(k)]
    return out


# ============================================================ 28 — story count never sets prob
def test_28_action_mass_is_invariant_to_number_of_states():
    # The baseline is over ACTION CLASSES and COUNTED CASES. There is no per-story input at all:
    # two action classes with the same counted evidence give the same mass no matter how many
    # prose stories a downstream layer attaches to each tendency.
    cases = _cases([("hold", "same_institution", 6), ("raise", "same_institution", 2)])
    b = build_action_baseline("m", "rate_decision", ["hold", "raise"], cases)
    # counted 6:2 → hold clearly favored, from COUNTS, not story count
    assert b.mass("hold") > b.mass("raise")
    assert abs(sum(b.class_mass.values()) - 1.0) < 1e-9
    # the same counts with the classes listed in the other order → same masses (order-free)
    b2 = build_action_baseline("m", "rate_decision", ["raise", "hold"], cases)
    assert abs(b2.mass("hold") - b.mass("hold")) < 1e-9


def test_28b_uniform_fallback_depends_on_class_count_not_story_count():
    # No counted evidence → disclosed uniform over the ACTION CLASSES. Two classes → 0.5 each,
    # three classes → 1/3 each. It can NEVER be 1/(number of stories).
    b2 = build_action_baseline("m", "d", ["hold", "raise"], [])
    assert b2.class_mass == {"hold": 0.5, "raise": 0.5}
    assert b2.disclosed_uniform
    b3 = build_action_baseline("m", "d", ["cut", "hold", "raise"], [])
    assert all(abs(v - 1 / 3) < 1e-5 for v in b3.class_mass.values())   # stored mass is 6-dp
    assert b3.disclosed_uniform


# ============================================================ 29 — partial pooling, dense specific
def test_29_dense_specific_level_dominates():
    # Eight same-individual "hold" cases override a weak prior.
    b = build_action_baseline("m", "d", ["hold", "raise"],
                              _cases([("hold", "same_individual", 8)]))
    assert b.mass("hold") > 0.8
    assert b.levels_used == ["same_individual"]
    assert not b.disclosed_uniform


# ============================================================ 30 — partial pooling, sparse specific
def test_30_sparse_specific_shrinks_toward_parent():
    # One same-individual "hold" case against twenty broad "raise" cases: the single specific
    # case NUDGES but does not override the broad parent — the mass stays raise-heavy.
    b = build_action_baseline("m", "d", ["hold", "raise"],
                              _cases([("hold", "same_individual", 1),
                                      ("raise", "broad_human_decision_class", 20)]))
    assert b.mass("raise") > b.mass("hold")
    assert 0.15 < b.mass("hold") < 0.45     # shrunk toward the broad parent, not 0.5, not 1.0
    assert set(b.levels_used) == {"same_individual", "broad_human_decision_class"}


# ============================================================ 31 — more specificity beats more data
def test_31_specific_evidence_can_outweigh_broad_when_dense():
    # A dense same-individual signal (this person almost always holds) overrides a broad base
    # rate that leans the other way — specificity with enough count wins.
    cases = _cases([("hold", "same_individual", 10),
                    ("raise", "broad_human_decision_class", 10)])
    b = build_action_baseline("m", "d", ["hold", "raise"], cases)
    assert b.mass("hold") > 0.6


# ============================================================ 32 — conditional on the shared world
def test_32_baseline_is_conditional_on_shared_world():
    # Typed state↔condition alignment: cases carry a world context; conditioning on a world
    # counts only the consistent cases. Under "inflation high" the raise-cases dominate; under
    # "inflation low" the hold-cases do.
    cases = [
        ActionCase("raise", "same_institution", {"inflation": "high"}),
        ActionCase("raise", "same_institution", {"inflation": "high"}),
        ActionCase("raise", "same_institution", {"inflation": "high"}),
        ActionCase("hold", "same_institution", {"inflation": "low"}),
        ActionCase("hold", "same_institution", {"inflation": "low"}),
        ActionCase("hold", "same_institution", {"inflation": "low"}),
    ]
    hi = build_action_baseline("m", "d", ["hold", "raise"], cases,
                               condition_state={"inflation": "high"})
    lo = build_action_baseline("m", "d", ["hold", "raise"], cases,
                               condition_state={"inflation": "low"})
    assert hi.mass("raise") > hi.mass("hold")
    assert lo.mass("hold") > lo.mass("raise")
    # a context-free case counts under every world
    mixed = cases + [ActionCase("hold", "same_institution", {})]
    both = build_action_baseline("m", "d", ["hold", "raise"], mixed,
                                 condition_state={"inflation": "high"})
    assert both.provenance["cases_kept"] == 4   # 3 high-raise + 1 context-free hold


# ============================================================ 33 — intervals widen when sparse
def test_33_credible_interval_widens_when_sparse():
    sparse = build_action_baseline("m", "d", ["hold", "raise"],
                                   _cases([("hold", "same_institution", 1),
                                           ("raise", "same_institution", 1)]))
    dense = build_action_baseline("m", "d", ["hold", "raise"],
                                  _cases([("hold", "same_institution", 50),
                                          ("raise", "same_institution", 50)]))
    sp = sparse.interval("hold")
    dn = dense.interval("hold")
    assert (sp[1] - sp[0]) > (dn[1] - dn[0])     # sparser ⇒ wider band


# ============================================================ 34 — counts, not rates, are counted
def test_34_derive_action_cases_from_counted_binary_class():
    # A counted binary class (rate 0.75 over 8 cases that this member dissents-for-a-hike) becomes
    # 6 "raise" + 2 "hold" ActionCases at the class's hierarchy level.
    tbl = {"quantity": "this member dissents for a hike",
           "provenance": {"rate_mean": 0.75, "denominator": 8,
                          "hierarchy_level": "same_role_same_institution"}}
    derived = cases_from_counted_class(tbl, "raise", complement_class="hold")
    weights = {c.action_class: c.weight for c in derived}
    assert weights == {"raise": 6.0, "hold": 2.0}
    assert all(c.hierarchy_level == "same_role_same_institution" for c in derived)
    b = build_action_baseline("m", "d", ["hold", "raise"], derived)
    assert b.mass("raise") > b.mass("hold")


# ============================================================ 35 — normalization + monotonicity
def test_35_mass_normalizes_and_is_monotone_in_counts():
    base = build_action_baseline("m", "d", ["hold", "raise"],
                                 _cases([("hold", "same_institution", 3),
                                         ("raise", "same_institution", 3)]))
    more_hold = build_action_baseline("m", "d", ["hold", "raise"],
                                      _cases([("hold", "same_institution", 9),
                                              ("raise", "same_institution", 3)]))
    assert abs(sum(base.class_mass.values()) - 1.0) < 1e-9
    assert abs(sum(more_hold.class_mass.values()) - 1.0) < 1e-9
    assert more_hold.mass("hold") > base.mass("hold")     # more hold cases ⇒ more hold mass


# ============================================================ core pooling math
def test_partial_pool_core_no_data_is_uniform():
    mean, conc, n, levels = partial_pool_categorical(["a", "b", "c"], {})
    assert all(abs(v - 1 / 3) < 1e-9 for v in mean.values())
    assert n == 0.0 and levels == [] and conc == POOLING_TAU


def test_partial_pool_core_empty_specific_inherits_parent():
    # broad has data, specific levels empty → specific inherits the broad-informed parent
    counts = {"broad_human_decision_class": {"a": 12, "b": 0}}
    mean, conc, n, levels = partial_pool_categorical(["a", "b"], counts)
    assert mean["a"] > mean["b"]
    assert levels == ["broad_human_decision_class"]
    assert n == 12


def test_baseline_top_and_entropy():
    b = build_action_baseline("m", "d", ["hold", "raise"],
                              _cases([("hold", "same_institution", 9),
                                      ("raise", "same_institution", 1)]))
    assert b.top() == "hold"
    # a near-deterministic baseline has lower entropy than a uniform one
    u = build_action_baseline("m", "d", ["hold", "raise"], [])
    assert b.entropy() < u.entropy()


def test_no_action_classes_is_empty_baseline():
    b = build_action_baseline("m", "d", [], _cases([("hold", "same_institution", 3)]))
    assert b.class_mass == {}
    assert b.top() == ""
