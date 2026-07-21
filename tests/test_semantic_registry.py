"""Phase 5: semantic features are gated by measured evidence; the harmful/null interpretation channel is
quarantined from production; reliability + promotion gates enforce; a genuinely useful feature promotes."""
from swm.world_model_v2.semantic_registry import (SemanticFeature, SemanticRegistry, inter_run_agreement,
                                                  seed_registry)


def test_seeded_interpretation_channel_is_quarantined_not_production():
    reg = seed_registry()
    summ = reg.summary()
    assert summ["by_status"].get("quarantined", 0) >= 11        # all FEATURE_DIMS quarantined
    assert reg.production_features("upworthy_headline") == []   # NONE reach production (all harmful/null)
    assert reg.production_features("enron_messaging") == []


def test_quarantine_preserves_the_failure_records():
    reg = seed_registry()
    f = reg.features["interp.urgency"]
    assert f.failures                                            # negative results preserved, not deleted
    assert any(rec["domain"] == "upworthy_headline" for rec in f.failures)


def test_promotion_blocked_without_reliability_and_incremental_value():
    reg = SemanticRegistry()
    reg.register(SemanticFeature(feature_id="x", construct="a construct", observable_input="text",
                                 support="[0,1]", causal_role="utility", status="operationally_defined"))
    blockers = reg.features["x"].promotion_blockers("production_eligible")
    assert any("agreement" in b for b in blockers)
    assert any("incremental" in b for b in blockers)


def test_a_genuinely_useful_feature_can_promote():
    reg = SemanticRegistry()
    f = SemanticFeature(feature_id="good", construct="c", observable_input="text", support="[0,1]",
                        causal_role="utility", status="operationally_defined",
                        inter_prompt_agreement=0.8, inter_model_agreement=0.75)
    reg.register(f)
    reg.set_status("good", "reliability_validated", reason="agreement 0.8")
    reg.record_incremental("good", "some_domain", delta=-0.02, ci95=[-0.03, -0.01],
                           controls=["metadata"], beneficial=True)   # improves, CI excludes 0
    reg.set_status("good", "incrementally_predictive", reason="beneficial in some_domain")
    f.transport = {"passed": True}
    reg.set_status("good", "production_eligible", reason="transport ok")
    assert "good" in reg.production_features("some_domain")


def test_inter_run_agreement_high_for_stable_low_for_noisy():
    # ICC-style agreement needs multiple ITEMS (between-item variance must exist). 3 items, 1 dim, 3 runs.
    # stable: each item's runs agree tightly; items differ → high agreement.
    stable = [[[0.1], [0.5], [0.9]], [[0.11], [0.51], [0.89]], [[0.09], [0.49], [0.91]]]
    # noisy: runs for the same item disagree wildly → low agreement.
    noisy = [[[0.1], [0.5], [0.9]], [[0.9], [0.1], [0.5]], [[0.5], [0.9], [0.1]]]
    assert inter_run_agreement(stable) > 0.8
    assert inter_run_agreement(noisy) < 0.5
