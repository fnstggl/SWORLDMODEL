"""Bargaining / coalition / participation / platform / rewiring family transitions."""
from swm.world_model_v2.registry.families.interaction import (
    banzhaf_power, click_probability, coalition_forms, concession_offer, donation_amount,
    position_examination, rank_by_score, rewire_probability, rubinstein_split, turnout_probability)


def test_rubinstein_split_favors_patient_player():
    patient_a = rubinstein_split(0.95, 0.5)
    patient_b = rubinstein_split(0.5, 0.95)
    assert patient_a > patient_b                              # more patient proposer keeps more
    assert 0 < patient_a < 1 and 0 < patient_b < 1


def test_concession_moves_toward_reservation_near_deadline():
    early = concession_offer(100, 40, t=1, deadline=10, beta=1.0)
    late = concession_offer(100, 40, t=9, deadline=10, beta=1.0)
    assert early > late > 40                                  # concedes toward reservation over time
    # tough (boulware) concedes less early than conceder
    tough = concession_offer(100, 40, t=5, deadline=10, beta=0.3)
    conceder = concession_offer(100, 40, t=5, deadline=10, beta=3.0)
    assert tough > conceder


def test_banzhaf_power_reflects_pivotality():
    # a dominant member (weight 3 vs two 1s, quota 3) is always pivotal
    power = banzhaf_power({"big": 3, "s1": 1, "s2": 1}, quota=3)
    assert power["big"] > power["s1"]
    assert abs(sum(power.values()) - 1.0) < 1e-9


def test_coalition_forms_minimal_winning():
    members, tot = coalition_forms({"a": 0.4, "b": 0.35, "c": 0.25}, quota=0.5)
    assert tot >= 0.5 and set(members) <= {"a", "b", "c"}
    assert len(members) <= 2                                  # minimal


def test_turnout_and_mobilization_monotone():
    base = turnout_probability(0.5)
    mobilized = turnout_probability(0.5, mobilized=1.0)
    high_cost = turnout_probability(0.5, cost=2.0)
    assert mobilized > base > high_cost


def test_donation_saturates_at_capacity():
    d = donation_amount(100, ask=1000, affinity=1.0, prior_gifts=100)
    assert d <= 100


def test_position_examination_decays_with_rank():
    assert position_examination(0) == 1.0
    assert position_examination(5) < position_examination(1)
    assert click_probability(2, 0.9) < click_probability(0, 0.9)


def test_rank_by_score_orders_desc():
    ranked = rank_by_score({"lo": 0.1, "hi": 0.9, "mid": 0.5})
    assert ranked[0][0] == "hi" and ranked[0][1] == 0


def test_rewire_probability_higher_for_homophilous_recent_ties():
    fresh_similar = rewire_probability(0.9, 1.0)
    old_dissimilar = rewire_probability(0.1, 300.0)
    assert fresh_similar > old_dissimilar
