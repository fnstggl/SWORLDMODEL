"""Calibrated weights — the full weighting engine: four sources of a weight, three principles, one object.

A variable's weight is a causal ELASTICITY you never know exactly. This assembles it honestly:

  FOUR SOURCES (each becomes a `WeightPrior` = a mean + a CI + a provenance tag):
    1. FIT from outcomes        — the data does it (a WeightPrior with a wide/uninformative prior).
    2. PARTIAL POOLING          — a population-average elasticity as the per-unit prior mean.
    3. LITERATURE / EXPERIMENT  — a measured effect size with its confidence interval (`effect_size_prior`).
    4. LLM PRIOR with a CI       — the compressed literature as a regularizer, never a point (`llm_elasticity_prior`).

  THREE PRINCIPLES:
    A. INTEGRATE OVER WEIGHT UNCERTAINTY — the prior's CI sets a per-weight precision (tight CI ⇒ shrink hard;
       wide/absent ⇒ free). The Laplace posterior folds prior + data uncertainty together, and `predict_dist`
       samples weights from it, so an unknown weight WIDENS the forecast instead of biasing it. `active_
       learning_targets` names the weight worth measuring next (high leverage × high remaining uncertainty).
    B. VARIANCE TRIAGE — `triage` ranks variables by weight²·Var(feature); only the few high-leverage weights
       need precise calibration, so "model everything" is tractable.
    C. WEIGHTS AS HYPERPARAMETERS — `fit(tune=True)` picks the global shrinkage temper by empirical Bayes
       (internal held-out log-loss), so the model neither under- nor over-trusts its priors given the data
       (the n-adaptive fix). Held-out accuracy is the weight-oracle.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from swm.eval.metrics import log_loss
from swm.variables.bayes_logistic import BayesianLogistic, variance_contribution


@dataclass
class WeightPrior:
    """A weight's elasticity prior: `mean` (signed per-unit effect), `sd` (how unsure the prior is — the CI),
    and `source` (provenance: fit / pooling / literature / llm / none). Wide sd ⇒ let the data decide."""
    name: str
    mean: float = 0.0
    sd: float = 3.0
    source: str = "none"

    def precision(self) -> float:
        return 1.0 / max(1e-6, self.sd) ** 2


def effect_size_prior(name, effect, ci95, source="literature") -> WeightPrior:
    """Source 3: a measured effect size ± its 95% CI half-width."""
    return WeightPrior(name, float(effect), max(1e-3, float(ci95) / 1.96), source)


def llm_elasticity_prior(name, mean, ci95, source="llm") -> WeightPrior:
    """Source 4: an LLM elasticity estimate WITH a confidence interval (never a bare point)."""
    return WeightPrior(name, float(mean), max(1e-3, float(ci95) / 1.96), source)


def uninformative_prior(name, sd=3.0) -> WeightPrior:
    """Source 1: no prior knowledge — a wide prior, so the data fits the weight freely."""
    return WeightPrior(name, 0.0, sd, "fit")


def empirical_bayes_temper(X, y, w0, base_prec, grid=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0), *, seed=0, epochs=200):
    """Principle C: pick the global precision multiplier that minimizes INTERNAL held-out log-loss — the
    n-adaptive shrinkage the fidelity ladder showed you need (thin data ⇒ trust priors more)."""
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    cut = max(2, int(0.75 * len(idx)))
    tr, va = idx[:cut], idx[cut:]
    if not va:
        return 1.0
    Xtr, ytr = [X[i] for i in tr], [y[i] for i in tr]
    Xva, yva = [X[i] for i in va], [y[i] for i in va]
    best_t, best_ll = 1.0, float("inf")
    for t in grid:
        m = BayesianLogistic(l2=[p * t for p in base_prec], w0=w0, epochs=epochs).fit(Xtr, ytr)
        preds = [min(1 - 1e-6, max(1e-6, m.predict_proba(x))) for x in Xva]
        ll = log_loss(yva, preds)
        if ll < best_ll:
            best_t, best_ll = t, ll
    return best_t


@dataclass
class CalibratedWeights:
    """Fit weights with per-weight priors (from the four sources) + empirical-Bayes shrinkage + integrated
    uncertainty. `priors` has one WeightPrior per feature column."""
    priors: list
    temper_grid: tuple = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    epochs: int = 300
    eb_epochs: int = 200                 # epochs for the empirical-Bayes inner tuning fits (cheaper than final)
    model: BayesianLogistic = None
    temper: float = 1.0

    def fit(self, X, y, *, tune=True, seed=0):
        w0 = [p.mean for p in self.priors]
        base_prec = [p.precision() for p in self.priors]
        self.temper = (empirical_bayes_temper(X, y, w0, base_prec, self.temper_grid, seed=seed,
                                              epochs=self.eb_epochs) if tune else 1.0)
        prec = [p * self.temper for p in base_prec]
        self.model = BayesianLogistic(l2=prec, w0=w0, epochs=self.epochs).fit(X, y)
        return self

    def predict(self, x) -> float:
        return self.model.predict_proba(x)

    def predict_dist(self, x, **kw) -> dict:
        """Integrate over the (prior + data) weight posterior — unknown weights widen the prediction."""
        return self.model.predict_dist(x, **kw)

    def weight_report(self):
        sd = self.model.weight_sd()
        out = []
        for j, p in enumerate(self.priors):
            snr = abs(self.model.w[j]) / sd[j] if sd[j] > 0 else float("inf")
            out.append({"name": p.name, "weight": round(self.model.w[j], 4), "sd": round(sd[j], 4),
                        "snr": round(snr, 2), "prior_mean": round(p.mean, 3), "source": p.source})
        return out

    def triage(self, X):
        """Principle B: variables ranked by their share of outcome variance (weight²·Var)."""
        cols = [p.name for p in self.priors]
        return [{"name": cols[j], "share": round(s, 4)} for j, s in variance_contribution(X, self.model.w)]

    def active_learning_targets(self, X):
        """Principle A (bonus): which weight to MEASURE next = high leverage × high remaining uncertainty.
        Ranks variables by variance-share × posterior weight-SD — where nailing the weight would most sharpen
        the forecast."""
        share = {j: s for j, s in variance_contribution(X, self.model.w)}
        sd = self.model.weight_sd()
        scored = [(self.priors[j].name, round(share.get(j, 0.0) * sd[j], 5)) for j in range(len(self.priors))]
        return sorted(scored, key=lambda t: -t[1])
