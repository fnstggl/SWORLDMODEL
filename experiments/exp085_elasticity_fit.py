"""EXP-085 — the elasticity-fitting harness: turn the optimizer's PRIORS into a GRADED objective.

The message optimizer's objective starts from coarse world-knowledge elasticities (directions trustworthy,
magnitudes `unvalidated`). This is how the magnitudes earn a grade: fit them to labeled (recipient,
strategy, replied) data, regularized toward the priors, and grade on a held-out split.

We don't have real reply logs, so this validates the ESTIMATOR on synthetic data with a KNOWN ground-truth
elasticity model — it recovers the weights and is calibrated on held-out data. That grades the harness, not
the real world; a real grade needs real reply logs (same stance as IndividualWorld).

Run:  PYTHONPATH=. python experiments/exp085_elasticity_fit.py
"""
from __future__ import annotations

import statistics

from swm.decision.elasticity_fit import fit_elasticities, grade_fit, synthetic_reply_dataset, _predict
from swm.decision.message_pipeline import RecipientState, optimize_message
from swm.eval.metrics import log_loss


def main():
    print("=" * 78)
    print("EXP-085  elasticity-fitting harness — earn a calibration grade")
    print("=" * 78)

    data, truth = synthetic_reply_dataset(2500, seed=1)
    fit = grade_fit(data, split=0.7, temporal=True)
    g = fit.grade
    print(f"\n[FIT] n_train={fit.n_train}  ->  GRADE {g['grade']}")
    print(f"      ECE={g['ece']}  Brier={g['brier']}  log_loss={g['log_loss']}  "
          f"uplift@20={g['uplift@20']}  base_rate={g['test_base_rate']}")

    # weight recovery vs the known ground truth
    names = list(truth)
    fw = [fit.weights[n][0] for n in names]
    tw = [truth[n] for n in names]
    mfw, mtw = statistics.mean(fw), statistics.mean(tw)
    cov = sum((a - mfw) * (b - mtw) for a, b in zip(fw, tw))
    den = (sum((a - mfw) ** 2 for a in fw) * sum((b - mtw) ** 2 for b in tw)) ** 0.5
    print(f"\n[RECOVERY] corr(fitted weights, ground truth) = {cov / den:.3f}")

    # the prior earns its keep on THIN data (world-knowledge prior + likelihood beats ridge-to-zero)
    thin, test = data[:80], data[1800:2100]
    y = [o for *_, o in test]
    ll_prior = log_loss(y, [_predict(fit_elasticities(thin, use_prior=True), r, s, b) for r, s, b, _ in test])
    ll_ridge = log_loss(y, [_predict(fit_elasticities(thin, use_prior=False), r, s, b) for r, s, b, _ in test])
    print(f"[THIN DATA] log_loss with prior={ll_prior:.3f}  vs ridge-to-zero={ll_ridge:.3f}  "
          f"(prior should win)")

    # use the FITTED objective in the optimizer -> the result now carries a real grade
    rs = RecipientState(vars={"status_orientation": 0.85, "skepticism": 0.9, "status": 0.9,
                              "openness_to_outreach": 0.9, "attention_availability": 0.4,
                              "platform_response_norm": 0.3, "relationship_strength": 0.0},
                        base_mean=0.2, base_n_effective=6.0, label="Peter Thiel")
    res = optimize_message(rs, fit=fit, n_mc=1500, seed=0)
    print(f"\n[OPTIMIZE with fitted objective] calibration_grade = "
          f"{res.summary()['calibration_grade']}  (was 'unvalidated' with priors)")
    print("  email:", res.email.text[:90], "...")
    print("  honesty:", res.summary()["honesty"])


if __name__ == "__main__":
    main()
