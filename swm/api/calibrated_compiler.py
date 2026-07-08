"""Calibrated compilation — make every emitted variable arrive with a calibrated weight-with-uncertainty.

Two hooks that sit between the compiler and the runtime:

  - `apply_registry(spec, outcome_class, registry)` — for each readout variable, COMBINE the LLM/literature
    weight prior with the data-learned elasticity from the `PriorRegistry` (precision-weighted). Where we have
    accumulated evidence the weight tightens toward the data; where we don't it stays at the LLM prior. This
    is the flywheel consumed: more datasets calibrated ⇒ better default weights for every new question.
  - `calibrate_from_data(spec, rows, feature_of, outcome_class, registry)` — FIT the readout weights on a
    labeled dataset (the ground-truth source of an elasticity), set each variable's weight + posterior SD +
    provenance='fit', and feed the fitted elasticities back into the registry so they transfer.

Together with `swm/api/compiler.py::calibrated_readout` (which integrates the weight uncertainty in the
Monte-Carlo), this realizes: model all relevant variables, each with a calibrated weight-with-uncertainty,
for any question — prediction or best-action.
"""
from __future__ import annotations

import copy

from swm.variables.calibrated_weights import CalibratedWeights, WeightPrior


def apply_registry(spec, outcome_class, registry):
    """Tighten each readout variable's weight by combining its LLM prior with the learned-registry elasticity
    (precision-weighted). Returns a copy; leaves non-readout variables (weight=None) untouched."""
    s = copy.deepcopy(spec)
    for v in s.variables:
        if v.weight is None:
            continue
        rec = registry.get(v.name, outcome_class)
        if rec is None:
            continue
        llm_sd = v.weight_sd if v.weight_sd else 1.0
        p1, p2 = 1.0 / llm_sd ** 2, rec.precision() if hasattr(rec, "precision") else 1.0 / rec.sd ** 2
        v.weight = (v.weight * p1 + rec.mean * p2) / (p1 + p2)
        v.weight_sd = (1.0 / (p1 + p2)) ** 0.5
        v.weight_source = rec.source
    return s


def calibrate_from_data(spec, rows, feature_of, outcome_class, *, registry=None, tune=True, seed=0):
    """Fit the readout weights on labeled `rows`. `feature_of(row) -> [x per spec.variable, in order]`; each
    row needs a `y` in {0,1}. Sets each variable's weight = fitted elasticity, weight_sd = posterior SD,
    center = 0, source = 'fit', and the spec intercept. Optionally feeds the registry. Returns (spec', model)."""
    s = copy.deepcopy(spec)
    X = [feature_of(r) for r in rows]
    y = [int(r["y"]) for r in rows]
    priors = [WeightPrior(v.name, v.weight if v.weight is not None else 0.0,
                          v.weight_sd if v.weight_sd else 3.0,
                          source=v.weight_source) for v in s.variables]
    cw = CalibratedWeights(priors).fit(X, y, tune=tune, seed=seed)
    rep = cw.weight_report()
    for v, r in zip(s.variables, rep):
        v.weight, v.weight_sd, v.center, v.weight_source = r["weight"], r["sd"], 0.0, "fit"
    s.extra = dict(s.extra)
    s.extra["intercept"] = cw.model.b
    if registry is not None:
        registry.register_from_fit(cw, outcome_class)
    return s, cw
