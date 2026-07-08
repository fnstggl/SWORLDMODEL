"""Adaptive fidelity — spend calibration only where the outcome is SENSITIVE (variance triage in the loop).

The tractable form of "model every relevant variable": emit them all, but invest precise, data-calibrated
weights only in the few the outcome actually turns on, and leave the rest at a rough prior. A variable's
leverage = its share of the outcome-logit variance = Var(weight·(value−center)) integrating BOTH the value
and the weight uncertainty. This ranks the variables so the compile→run loop knows where to look up / fit a
calibrated weight (`calibrated_compiler`) and where a coarse LLM prior is good enough — 100 variables in,
calibration compute on the ~10 that move the answer.
"""
from __future__ import annotations

import random


def variable_leverage(spec, *, n=3000, seed=0):
    """Per-variable share of the outcome-logit variance for a calibrated_readout spec. Returns [(name,
    share)] descending — integrates value uncertainty (est_sd) AND weight uncertainty (weight_sd)."""
    rng = random.Random(seed)
    vs = [v for v in spec.variables if v.weight is not None]
    terms = {v.name: [] for v in vs}
    for _ in range(n):
        for v in vs:
            xv = min(v.hi, max(v.lo, v.value + (rng.gauss(0, v.est_sd) if v.est_sd else 0.0)))
            w = v.weight + (rng.gauss(0, v.weight_sd) if v.weight_sd else 0.0)
            terms[v.name].append(w * (xv - v.center))

    def var(a):
        m = sum(a) / len(a)
        return sum((x - m) ** 2 for x in a) / len(a)

    contrib = [(v.name, var(terms[v.name])) for v in vs]
    tot = sum(c for _, c in contrib) or 1.0
    return sorted([(nm, c / tot) for nm, c in contrib], key=lambda t: -t[1])


def triage(spec, *, keep_frac=0.9, n=3000, seed=0) -> dict:
    """Split the variables into the high-leverage set worth precise calibration (cumulatively covering
    `keep_frac` of outcome variance) and the low-leverage set a rough prior handles."""
    lev = variable_leverage(spec, n=n, seed=seed)
    total = sum(s for _, s in lev)
    cum, invest, rough = 0.0, [], []
    for name, s in lev:
        (invest if cum < keep_frac * total else rough).append(name)
        cum += s
    return {"leverage": [{"name": nm, "share": round(s, 4)} for nm, s in lev],
            "invest_in": invest, "rough_prior_ok": rough,
            "note": f"{len(invest)}/{len(lev)} variables carry {round(keep_frac * 100)}% of the outcome variance"}
