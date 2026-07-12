"""Phase 12: calibrators improve ECE on synthetic miscalibrated data; abstention grades signal-drive;
critic never overwrites; uncertainty decomposition separates structural from state uncertainty."""
import random

from swm.world_model_v2.calibration import (AbstentionDecision, decide_abstention, decompose_uncertainty,
                                            ece, fit_conditioned, fit_isotonic, fit_platt, reliability_table,
                                            run_critic, build_result)


def _miscalibrated(n=800, seed=1):
    """Overconfident forecasts: true p = 0.5 + 0.4*(pred-0.5) → predictions too extreme."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pred = rng.random()
        true_p = 0.5 + 0.4 * (pred - 0.5)
        out.append((pred, 1 if rng.random() < true_p else 0))
    return out


def test_platt_reduces_ece():
    data = _miscalibrated()
    tr, te = data[:600], data[600:]
    cal = fit_platt(tr)
    raw_ece = ece(te)
    cal_ece = ece([(cal.apply(p), y) for p, y in te])
    assert cal_ece < raw_ece


def test_isotonic_reduces_ece():
    data = _miscalibrated(seed=2)
    tr, te = data[:600], data[600:]
    cal = fit_isotonic(tr)
    assert ece([(cal.apply(p), y) for p, y in te]) < ece(te)


def test_conditioned_falls_back_to_global_for_small_cells():
    data = [(p, y, "big") for p, y in _miscalibrated(400)] + [(0.9, 1, "tiny"), (0.9, 0, "tiny")]
    cal = fit_conditioned(data, min_cell=30)
    # 'tiny' cell (n=2) must blend toward global, not overfit
    prov = cal.provenance("tiny")
    assert prov["used"] == "pooled_or_global"
    p = cal.apply(0.9, "tiny")
    assert 0.0 <= p <= 1.0


def test_abstention_is_signal_driven():
    ok = decide_abstention(has_applicable_validated_mechanism=True, evidence_grade="A", n_evidence_items=5)
    assert ok.grade == "supported"
    hard = decide_abstention(has_applicable_validated_mechanism=False, out_of_distribution=True,
                             structural_model_uncertainty=0.8)
    assert hard.grade == "abstain"
    leaked = decide_abstention(evidence_grade="F (leaked)")
    assert leaked.grade == "abstain" and "leakage" in leaked.reasons[0]
    nocomp = decide_abstention(compiler_abstained=True)
    assert nocomp.grade == "unresolvable"


def test_critic_flags_disagreement_but_does_not_overwrite():
    rep = run_critic(0.9, direct_p=0.3, ensemble_p=0.35)
    assert rep.v2_p == 0.9                                     # NOT overwritten
    assert rep.disagreement is not None and rep.flags


def test_uncertainty_decomposition_separates_structural():
    class W:
        def __init__(self, hyp, sampled):
            self.uncertainty_meta = {"model": {"hypothesis": hyp}, "sampled": sampled}

    class B:
        def __init__(self, w, weight):
            self.world = w
            self.weight = weight

    branches = [B(W("A", {"x": 0.2}), 0.5), B(W("B", {"x": 0.8}), 0.5)]
    dec = decompose_uncertainty(branches, structural_posterior={"A": 0.5, "B": 0.5})
    assert dec["structural_model_uncertainty"] > 0.9          # maximal disagreement (50/50)
    assert dec["state_parameter_uncertainty"] > 0.0


def test_build_result_contract_is_complete():
    res = build_result(0.7, abstention=AbstentionDecision("supported"),
                       critic=run_critic(0.7, direct_p=0.65))
    for k in ("raw_probability", "calibrated_probability", "confidence_grade", "abstention",
              "direct_model_disagreement", "calibration_provenance"):
        assert k in res
