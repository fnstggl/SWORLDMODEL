"""Elasticity-fitting harness — turn the scorer's world-knowledge PRIORS into DATA-CALIBRATED weights
with a real grade.

The message optimizer's objective (`StrategyScorer`) starts from coarse, signed world-knowledge
elasticities and is therefore `unvalidated`: the directions are trustworthy, the magnitudes and the
absolute P(reply) are not. This harness is how those magnitudes earn their keep — the exact "world
knowledge as the prior, data as the update" pattern the repo uses elsewhere (`llm_prior`,
`calibrated_weights`):

  FIT    — a logistic over the SAME main + interaction features the scorer uses, with a per-recipient
           OFFSET (their base reply rate) and coefficients regularized TOWARD the priors (β→prior, not
           β→0). Data-poor terms stay near their world-knowledge prior; data-rich terms move to fit.
  GRADE  — evaluate on a held-out (temporal or random) split: ECE (calibration), Brier, log-loss, and
           uplift@k versus the base-rate baseline. ECE thresholds → A/B/C/F, like the rest of the repo.
  USE    — `FittedElasticities.scorer_for(recipient, base)` returns a `StrategyScorer` whose weights are
           the fitted ones (provenance: fit) and which reports the GRADE instead of `unvalidated`.

Honesty (the load-bearing part): a labeled dataset of (recipient, message-strategy, replied) is the
scarce content→outcome asset. We don't have Peter Thiel's inbox, so `synthetic_reply_dataset` provides a
dataset with a KNOWN ground-truth elasticity model to validate the ESTIMATOR — it recovers the weights and
earns a calibrated grade on held-out synthetic data. That validates the harness, NOT the real-world
numbers; a real grade needs real reply logs (the same stance as `IndividualWorld`: "validated as an
estimator on synthetic data; the real-behavior claim is blocked on private data").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.decision.strategy_scorer import (ELASTICITY_SCALE, TERM_NAMES, TERMS, StrategyScorer,
                                          base_logit, term_features, _sigmoid)
from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k

# a labeled sample: (recipient_vars, message_strategy, base_responsiveness, replied 0/1)
Sample = tuple


def _features(recipient: dict, strategy: dict) -> list:
    """Ordered feature vector over TERM_NAMES for one sample (same features the scorer uses)."""
    return [f for _name, f, _m, _sd in term_features(strategy, recipient)]


@dataclass
class FittedElasticities:
    weights: dict                       # term -> (mean, sd)
    grade: dict                         # {grade, ece, brier, ...}
    n_train: int = 0
    prior_blend: float = 1.0

    def scorer_for(self, recipient: dict, base_responsiveness: float, **kw) -> StrategyScorer:
        """A StrategyScorer that uses the FITTED weights (calibrated) and reports the grade."""
        return StrategyScorer(recipient=dict(recipient), base_responsiveness=base_responsiveness,
                              weights=self.weights, grade=self.grade, **kw)

    def summary(self) -> dict:
        return {"grade": self.grade, "n_train": self.n_train,
                "weights": {k: [round(v[0], 3), round(v[1], 3)] for k, v in self.weights.items()}}


def fit_elasticities(samples: list, *, use_prior: bool = True, prior_strength: float = 1.0,
                     epochs: int = 400, lr: float = 0.2) -> dict:
    """Fit the elasticity weights by gradient descent on a logistic with a per-sample OFFSET (the base
    reply rate) and an L2 pull toward the world-knowledge priors. Returns {term: (weight, sd)}.

    `use_prior=False` shrinks toward 0 (ordinary ridge) — the ablation that shows the prior helps when
    data is thin. `prior_strength` scales the pull toward the prior (its precision)."""
    names = TERM_NAMES
    # prior means are the scorer's priors × the same scale the scorer applies, so fitted ≈ prior at n=0.
    w0 = {name: (ELASTICITY_SCALE * mean if use_prior else 0.0) for name, _mv, _rv, mean, _sd in TERMS}
    prec = {name: (prior_strength / max(1e-3, ELASTICITY_SCALE * sd) ** 2 if use_prior else prior_strength)
            for name, _mv, _rv, _mean, sd in TERMS}
    w = dict(w0)

    X, offs, y = [], [], []
    for recipient, strategy, base, replied in samples:
        X.append(dict(zip(names, _features(recipient, strategy))))
        offs.append(base_logit(base, recipient))
        y.append(int(replied))
    n = max(1, len(y))

    for _ in range(epochs):
        grad = {name: 0.0 for name in names}
        for xi, off, yi in zip(X, offs, y):
            z = off + sum(w[name] * xi[name] for name in names)
            err = _sigmoid(z) - yi
            for name in names:
                grad[name] += err * xi[name]
        for name in names:
            g = grad[name] / n + prec[name] / n * (w[name] - w0[name])   # data grad + pull to prior
            w[name] -= lr * g

    # Laplace-ish sd per weight: (data curvature + prior precision)^-1/2 (honest "how sure of this weight")
    sd = {}
    for name in names:
        info = prec[name]
        for xi, off in zip(X, offs):
            z = off + sum(w[m] * xi[m] for m in names)
            p = _sigmoid(z)
            info += p * (1 - p) * xi[name] ** 2
        sd[name] = 1.0 / math.sqrt(max(1e-6, info))
    return {name: (w[name], sd[name]) for name in names}


def _predict(weights: dict, recipient: dict, strategy: dict, base: float) -> float:
    z = base_logit(base, recipient) + sum(weights[n][0] * f for n, f, _m, _sd in term_features(strategy, recipient))
    return min(1 - 1e-6, max(1e-6, _sigmoid(z)))


def grade_fit(samples: list, *, split: float = 0.7, temporal: bool = True, **fit_kw) -> FittedElasticities:
    """Fit on the first `split` of the (time-ordered) data, grade on the held-out remainder."""
    n = len(samples)
    if n < 40:
        return FittedElasticities(weights={name: (0.0, 3.0) for name in TERM_NAMES},
                                  grade={"grade": "F", "error": f"only {n} samples; need >= 40"}, n_train=0)
    order = list(range(n)) if temporal else sorted(range(n), key=lambda i: (i * 2654435761) % n)
    cut = int(split * n)
    train = [samples[i] for i in order[:cut]]
    test = [samples[i] for i in order[cut:]]
    weights = fit_elasticities(train, **fit_kw)
    preds = [_predict(weights, r, s, b) for r, s, b, _ in test]
    y = [int(o) for _, _, _, o in test]
    ece = expected_calibration_error(y, preds)
    grade = {"grade": "A" if ece < 0.05 else "B" if ece < 0.10 else "C" if ece < 0.15 else "F",
             "ece": round(ece, 4), "brier": round(brier_score(y, preds), 4),
             "log_loss": round(log_loss(y, preds), 4), "uplift@20": round(uplift_at_k(y, preds, 0.2), 4),
             "n_test": len(y), "test_base_rate": round(sum(y) / len(y), 4),
             "note": "graded on a held-out split of the supplied reply outcomes"}
    return FittedElasticities(weights=weights, grade=grade, n_train=len(train))


# ---- synthetic validation (known ground truth) --------------------------------------------------

def synthetic_reply_dataset(n: int = 1500, *, seed: int = 0, noise: float = 0.0) -> list:
    """Generate (recipient, strategy, base, replied) from a KNOWN elasticity model, so the harness can be
    shown to recover the weights and be calibrated on held-out data. Ground-truth weights are the scorer's
    priors × scale with a mild per-term perturbation, plus label noise. Deterministic given `seed`."""
    import random
    rng = random.Random(seed)
    truth = {name: ELASTICITY_SCALE * mean * (0.6 + 0.8 * rng.random())
             for name, _mv, _rv, mean, _sd in TERMS}
    from swm.decision.strategy_scorer import MESSAGE_VARS
    rvars = ["status_orientation", "skepticism", "status", "openness_to_outreach",
             "attention_availability", "relationship_strength"]
    data = []
    for _ in range(n):
        recipient = {v: rng.random() for v in rvars}
        recipient["platform_response_norm"] = 0.3
        strategy = {v: rng.random() for v in MESSAGE_VARS}
        base = 0.05 + 0.35 * rng.random()
        z = base_logit(base, recipient) + sum(truth[nm] * f for nm, f, _m, _sd in term_features(strategy, recipient))
        p = _sigmoid(z + rng.gauss(0.0, noise))
        data.append((recipient, strategy, base, 1 if rng.random() < p else 0))
    return data, truth
