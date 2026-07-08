"""Corpus harvest — fit variable→outcome elasticities across EVERY dataset and persist them.

The flywheel at scale: for each dataset, fit a `CalibratedWeights` model (demographics→opinion,
person×message→persuasion, macro→rate-move, headline→engagement, …) and register each variable's learned
elasticity (mean + posterior SD) into the `PriorRegistry` under a semantic outcome-class. Estimates from
different datasets combine precision-weighted, so the more data we harvest the tighter and more transferable
the default weights become — and the compiler consults them so every emitted variable arrives pre-calibrated.

This module is the generic per-source fitter; the dataset-specific featurizers live in the experiment.
"""
from __future__ import annotations

import random

from swm.variables.calibrated_weights import CalibratedWeights, uninformative_prior


def onehot(rows, attrs, demo_of):
    """One-hot encode categorical attributes. Returns (X, feature_names). `demo_of(row) -> {attr: level}`."""
    vocab = {}
    for r in rows:
        d = demo_of(r)
        for a in attrs:
            vocab.setdefault((a, d.get(a, "unknown")), len(vocab))
    names = [f"{a}={lv}" for (a, lv) in vocab]

    def enc(r):
        d = demo_of(r)
        x = [0.0] * len(vocab)
        for a in attrs:
            j = vocab.get((a, d.get(a, "unknown")))
            if j is not None:
                x[j] = 1.0
        return x
    return [enc(r) for r in rows], names


def harvest_source(registry, X, y, names, outcome_class, *, source, cap=4000, seed=0, epochs=60):
    """Fit calibrated weights on one dataset and register the learned elasticities. Returns the fitted model
    (or None if too little/degenerate data). Deterministic subsampling to `cap` for tractability."""
    n = len(X)
    if n < 30 or len(set(y)) < 2:
        return None
    if n > cap:
        idx = random.Random(seed).sample(range(n), cap)
        X, y = [X[i] for i in idx], [y[i] for i in idx]
    cw = CalibratedWeights([uninformative_prior(nm) for nm in names], temper_grid=(1.0, 4.0),
                           epochs=epochs).fit(X, y, tune=True, seed=seed)
    registry.register_from_fit(cw, outcome_class, source=source)
    return cw
