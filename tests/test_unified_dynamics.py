"""Tests for the unified belief-dynamics operator (one transition for market and person)."""
from swm.transition.unified_dynamics import UnifiedBeliefDynamics, responsiveness_from_map
from swm.variables.variable_map import VariableMap


def _vm(openness, skepticism, prior):
    vm = VariableMap("op")
    vm.set("openness_to_outreach", openness, provenance="llm")
    vm.set("skepticism", skepticism, provenance="llm")
    vm.set("prior_stance", prior, provenance="llm")
    return vm


def test_responsiveness_monotone_in_traits():
    open_mind = responsiveness_from_map(_vm(0.9, 0.1, 0.0))
    closed_mind = responsiveness_from_map(_vm(0.2, 0.9, 0.9))
    assert open_mind > closed_mind
    assert 0.0 <= closed_mind <= open_mind <= 1.0


def test_entrenchment_resists_change_either_sign():
    neutral = responsiveness_from_map(_vm(0.6, 0.4, 0.0))
    entrenched_pos = responsiveness_from_map(_vm(0.6, 0.4, 0.9))
    entrenched_neg = responsiveness_from_map(_vm(0.6, 0.4, -0.9))
    assert entrenched_pos < neutral and entrenched_neg < neutral   # strong prior (either sign) resists


def test_predict_update_is_responsiveness_times_impact():
    uni = UnifiedBeliefDynamics(scale=1.0)
    assert abs(uni.predict_update(0.4, 0.5) - 0.2) < 1e-9
    # same operator, two scales
    assert uni.update_market(0.4, 1.0) == uni.predict_update(0.4, 1.0)
    vm = _vm(1.0, 0.0, 0.0)                                          # maximally responsive person
    assert abs(uni.update_person(0.4, vm) - 0.4) < 1e-9


def test_open_person_moves_more_than_entrenched_for_same_event():
    uni = UnifiedBeliefDynamics()
    event = 0.6
    open_move = uni.update_person(event, _vm(0.9, 0.1, 0.0))
    entrenched_move = uni.update_person(event, _vm(0.3, 0.8, 0.9))
    assert open_move > entrenched_move                              # heterogeneity from the VariableMap
