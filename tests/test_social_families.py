"""Trust / reciprocity / opinion-dynamics family transitions (candidate structural families)."""
from swm.world_model_v2.registry.families.social import (
    bounded_confidence_step, degroot_step, expressed_opinion, reciprocity_update, threshold_adopt,
    trust_update)


def test_trust_asymmetry_loss_steeper_than_gain():
    t0 = 0.6
    after_coop = trust_update(t0, "cooperated")
    after_defect = trust_update(t0, "defected")
    assert after_coop > t0 > after_defect
    assert (t0 - after_defect) > (after_coop - t0)           # asymmetry: loss > gain
    repaired = trust_update(after_defect, "repaired")
    assert after_defect < repaired < t0                       # partial repair


def test_reciprocity_tracks_kindness():
    up = reciprocity_update(0.5, 0.8, rate=0.3)
    down = reciprocity_update(0.5, -0.8, rate=0.3)
    assert up > 0.5 > down


def test_degroot_converges_to_consensus():
    opinions = {"a": 0.0, "b": 1.0, "c": 0.5}
    w = {"a": {"a": 0.5, "b": 0.5}, "b": {"b": 0.5, "c": 0.5}, "c": {"a": 0.5, "c": 0.5}}
    for _ in range(50):
        opinions = degroot_step(opinions, w)
    vals = list(opinions.values())
    assert max(vals) - min(vals) < 0.05                       # consensus


def test_bounded_confidence_forms_clusters():
    opinions = {f"a{i}": i / 10 for i in range(11)}           # 0.0 .. 1.0
    for _ in range(30):
        opinions = bounded_confidence_step(opinions, eps=0.15)
    # extremes stay apart (do not merge into one consensus with small eps)
    assert max(opinions.values()) - min(opinions.values()) > 0.3


def test_threshold_adoption_cascades():
    assert threshold_adopt(False, 0.6, 0.5) is True
    assert threshold_adopt(False, 0.3, 0.5) is False
    assert threshold_adopt(True, 0.0, 0.5) is True            # stays adopted


def test_expressed_opinion_falsifies_under_pressure():
    # low pressure → express latent; high pressure > conviction → pulled toward norm
    assert expressed_opinion(0.8, social_pressure=0.1, conviction=0.5) == 0.8
    pulled = expressed_opinion(0.8, social_pressure=0.9, conviction=0.3)
    assert pulled < 0.8
