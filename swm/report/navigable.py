"""NavigableOutcome — the decomposed, navigable object that replaces a scalar probability.

A market gives you a number; a world model gives you the WORLDS. For any Monte-Carlo ensemble this builds
the object the product is really about — "37%, here's why, here's the fork, here's what to watch":

  - the DISTRIBUTION (mode/mean/spread/quantiles for a numeric outcome; a category distribution for a label);
  - the REDUCIBLE vs IRREDUCIBLE split (epistemic spread that better estimates would shrink vs the aleatoric
    forecastability ceiling no model can beat) — from `variance_decomposition` when a StructuralModel is
    available;
  - the PIVOTAL BRANCHES: the exogenous factors whose realized value most determines the outcome, each with
    the conditional outcome on either side of its split and the probability of each branch. Discovery is a
    decision-stump / ANOVA attribution over the recorded factors (the exogenous draws each trajectory took),
    ranked by the between-branch variance they explain — cheap, because the ensemble already exists.

The input is a list of `(outcome, factors)` samples (as produced by a compiler `Sampler.traced`) or bare
outcomes (then no pivots). `target` selects the scalar the pivots explain — identity for a numeric outcome,
an indicator for a label, or an explicit `(fn, desc)` — so the branches explain *the thing you care about*.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Pivot:
    """One exogenous factor that forks the outcome: the outcome conditional on each side of its split, the
    probability of the high side, and the between-branch variance it explains (the ranking score)."""
    factor: str
    p_high: float
    outcome_high: float
    outcome_low: float
    explained: float

    @property
    def gap(self) -> float:
        return self.outcome_high - self.outcome_low

    def as_dict(self):
        return {"factor": self.factor, "p_high": round(self.p_high, 3),
                "outcome_high": round(self.outcome_high, 4), "outcome_low": round(self.outcome_low, 4),
                "gap": round(self.gap, 4), "explained": round(self.explained, 6)}


@dataclass
class NavigableOutcome:
    kind: str                                     # 'numeric' | 'categorical'
    point: object                                 # mean (numeric) or modal label (categorical)
    target_desc: str = ""
    target_value: float = None                    # E[target scalar] (== P(target) for an indicator)
    mean: float = None
    sd: float = None
    quantiles: dict = None                        # {'p05','p50','p95'}
    distribution: dict = None                     # categorical label -> prob
    reducible_sd: float = None
    irreducible_sd: float = None
    irreducible_frac: float = None
    forecastable: bool = None
    pivots: list = field(default_factory=list)
    n: int = 0

    def as_dict(self):
        d = {"kind": self.kind, "point": self.point, "target_desc": self.target_desc,
             "target_value": round(self.target_value, 4) if self.target_value is not None else None,
             "n": self.n, "pivots": [p.as_dict() for p in self.pivots]}
        if self.kind == "numeric":
            d.update({"mean": round(self.mean, 4), "sd": round(self.sd, 4), "quantiles": self.quantiles})
        else:
            d["distribution"] = {k: round(v, 4) for k, v in (self.distribution or {}).items()}
        if self.irreducible_frac is not None:
            d.update({"reducible_sd": self.reducible_sd, "irreducible_sd": self.irreducible_sd,
                      "irreducible_frac": self.irreducible_frac, "forecastable": self.forecastable})
        return d

    def summary(self) -> str:
        """One honest human line: the point, the irreducible share, and the top pivot's fork."""
        if self.kind == "categorical":
            head = f"{self.point} most likely ({self.target_value:.0%})"
        else:
            q = self.quantiles or {}
            head = f"{self.target_desc or 'outcome'} ≈ {self.target_value:.3f}"
            if q:
                head += f" (80% {q.get('p05'):.3f}–{q.get('p95'):.3f})"
        if self.irreducible_frac is not None:
            head += f"; {self.irreducible_frac:.0%} irreducible"
        if self.pivots:
            p = self.pivots[0]
            head += (f"; pivot: {p.factor} → {p.outcome_high:.2f} if high vs {p.outcome_low:.2f} if low "
                     f"(P(high)={p.p_high:.0%})")
        return head


def _target_fn(outcomes, target):
    """Resolve `target` into (fn: outcome->float, desc, is_indicator). Defaults: numeric identity; for a
    categorical outcome, the indicator of the modal label."""
    numeric = outcomes and all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in outcomes)
    if callable(target):
        return target, getattr(target, "desc", "utility"), False
    if isinstance(target, tuple) and len(target) == 2 and callable(target[0]):
        return target[0], target[1], False
    if isinstance(target, str):                                    # a label indicator
        return (lambda o: 1.0 if o == target else 0.0), f"P({target})", True
    if numeric:
        return (lambda o: float(o)), "outcome", False
    mode = Counter(outcomes).most_common(1)[0][0]                  # categorical, no target -> modal label
    return (lambda o: 1.0 if o == mode else 0.0), f"P({mode})", True


def _pivots(factor_rows, t, top_pivots):
    """Decision-stump attribution: for each exogenous factor, split at its median and score the between-
    branch variance it explains of the target scalar `t`. Returns the top pivots by explained variance."""
    if not factor_rows or not any(factor_rows):
        return []
    n = len(t)
    m_all = sum(t) / n
    keys = set()
    for f in factor_rows:
        keys.update(f.keys())
    out = []
    for key in keys:
        pairs = [(f[key], t[i]) for i, f in enumerate(factor_rows) if key in f]
        if len(pairs) < max(10, n // 5):                          # need enough coverage to trust the split
            continue
        vals = sorted(p[0] for p in pairs)
        med = vals[len(vals) // 2]
        hi = [tv for fv, tv in pairs if fv > med]
        lo = [tv for fv, tv in pairs if fv <= med]
        if len(hi) < 3 or len(lo) < 3:
            continue
        p_high = len(hi) / len(pairs)
        m_hi, m_lo = sum(hi) / len(hi), sum(lo) / len(lo)
        explained = p_high * (m_hi - m_all) ** 2 + (1 - p_high) * (m_lo - m_all) ** 2
        out.append(Pivot(key, p_high, m_hi, m_lo, explained))
    out.sort(key=lambda p: -p.explained)
    return out[:top_pivots]


def navigable_from_samples(samples, *, target=None, top_pivots=3, decomp=None) -> NavigableOutcome:
    """Build a NavigableOutcome from an ensemble. `samples` is a list of `(outcome, factors)` tuples (factors
    a dict of the exogenous draws that defined that world) or bare outcomes. `target` selects the scalar the
    pivots explain (see module docstring). `decomp` is an optional `variance_decomposition` dict."""
    traced = bool(samples) and isinstance(samples[0], tuple) and len(samples[0]) == 2 and isinstance(samples[0][1], dict)
    outcomes = [s[0] for s in samples] if traced else list(samples)
    factor_rows = [s[1] for s in samples] if traced else []
    n = len(outcomes)
    numeric = outcomes and all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in outcomes)
    fn, desc, _ = _target_fn(outcomes, target)
    t = [fn(o) for o in outcomes]
    tval = sum(t) / n if n else None

    nav = NavigableOutcome(kind=("numeric" if numeric else "categorical"),
                           point=None, target_desc=desc, target_value=tval, n=n,
                           pivots=_pivots(factor_rows, t, top_pivots))
    if numeric:
        mean = sum(outcomes) / n
        var = sum((o - mean) ** 2 for o in outcomes) / n
        s = sorted(outcomes)
        def q(p): return s[min(n - 1, int(p * n))]
        nav.point = round(mean, 4)
        nav.mean, nav.sd = mean, math.sqrt(var)
        nav.quantiles = {"p05": round(q(.05), 4), "p50": round(q(.5), 4), "p95": round(q(.95), 4)}
    else:
        c = Counter(outcomes)
        nav.distribution = {k: v / n for k, v in c.most_common()}
        nav.point = c.most_common(1)[0][0]
    if decomp:
        nav.reducible_sd = decomp.get("reducible_sd")
        nav.irreducible_sd = decomp.get("irreducible_sd")
        nav.irreducible_frac = decomp.get("irreducible_frac")
        nav.forecastable = decomp.get("forecastable")
    return nav


def navigable_from_spec(spec, *, n=6000, seed=0, target=None, top_pivots=3, decompose=True) -> NavigableOutcome:
    """Convenience: compile a `ModelSpec`'s sampler, run a traced ensemble, and (for generic_scm) attach the
    reducible/irreducible split. Keeps the pure `navigable_from_samples` mechanism-agnostic."""
    from random import Random
    from swm.api.compiler import build_sampler
    sampler = build_sampler(spec)
    rng = Random(seed)
    samples = [sampler.traced(rng) for _ in range(n)]
    decomp = None
    if decompose and "model" in sampler.aux:
        from swm.simulation.structural import variance_decomposition
        decomp = variance_decomposition(sampler.aux["model"], spec.horizon, spec.dt, n=min(n, 4000))
    if target is None and sampler.kind == "categorical" and spec.outcome.get("target"):
        target = spec.outcome["target"]
    return navigable_from_samples(samples, target=target, top_pivots=top_pivots, decomp=decomp)
