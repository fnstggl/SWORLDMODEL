"""Structural simulation engine — the runtime for a world-model that actually mirrors the world.

This is the architectural pivot. The earlier levels apply ONE generic mechanism (an agent society, a
mean-field pool) to every question. That is why the NBA forecast was wrong: a championship is not a
deliberation, and no amount of extra variables fixes a wrong MECHANISM. The thesis — "model the same
variables and dynamics as the world, roll the state forward, read the outcome" — is right, but with three
first-principles constraints that dictate the architecture:

  1. STRUCTURE over variable-count. Accuracy comes from matching the real GENERATIVE PROCESS (the causal
     structure), not from piling on variables. Every inferred variable is an estimate with error, and over
     a rollout those errors COMPOUND. Blind overbuilding raises variance and hurts (we saw it: adding
     bandwagon coupling made GSS worse). So: model the structure RICHLY and correctly, estimate each
     variable HUMBLY (with uncertainty), and integrate the uncertainty out — do not fake precision.

  2. CALIBRATED TIME. Rolling forward is only meaningful if each variable changes by the amount it really
     changes in that elapsed time. This engine makes dynamics a proper DIFFUSION: drift scales with dt,
     stochastic change scales with sqrt(dt) (Wiener scaling). Volatility is a per-unit-time quantity you
     CALIBRATE against real data, so "roll forward 1 day" moves variables by one day's worth — not an
     arbitrary "6 rounds".

  3. IRREDUCIBLE UNCERTAINTY is the point, not a nuisance. Social systems are chaotic and partly random;
     past the predictability horizon, MORE fidelity cannot help (this is why perfect physics still caps
     weather at ~2 weeks). The honest output is therefore a DISTRIBUTION from a Monte-Carlo ensemble, and
     a decomposition of its spread into the REDUCIBLE part (epistemic — better estimates would shrink it)
     and the IRREDUCIBLE part (aleatoric — the forecastability ceiling no model can beat). A model that
     says "the favorite wins 30%" is not weak if 70% is genuinely irreducible; a model that says "52%" is
     simply overconfident.

The engine is deliberately mechanism-agnostic: `montecarlo` runs ANY stochastic `simulate_once(rng)`, so
the RIGHT structure per question (a playoff bracket, an electorate, a negotiation, a diffusion) plugs in.
`StructuralModel` is the general time-calibrated diffusion SCM for continuous coupled variables.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field


def montecarlo(simulate_once, n: int = 10000, seed: int = 0) -> dict:
    """Run an ensemble of a stochastic simulation. `simulate_once(rng) -> outcome`.
    Numeric outcomes -> mean/sd/quantiles; hashable (e.g. a winner label) -> a category distribution."""
    rng = random.Random(seed)
    outs = [simulate_once(rng) for _ in range(n)]
    numeric = outs and all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in outs)
    if numeric:
        s = sorted(outs)
        mean = sum(outs) / n
        var = sum((o - mean) ** 2 for o in outs) / n
        def q(p): return s[min(n - 1, int(p * n))]
        return {"kind": "numeric", "mean": mean, "sd": var ** 0.5, "var": var,
                "p05": q(.05), "p50": q(.5), "p95": q(.95), "n": n}
    c = Counter(outs)
    return {"kind": "categorical", "distribution": {k: v / n for k, v in c.most_common()},
            "mode": c.most_common(1)[0][0], "n": n}


def prob_of(simulate_once, predicate, n: int = 10000, seed: int = 0) -> float:
    """P(predicate(outcome)) over the ensemble — the calibrated probability of any event."""
    rng = random.Random(seed)
    return sum(1 for _ in range(n) if predicate(simulate_once(rng))) / n


@dataclass
class SVar:
    """A world variable: our estimate of its current value, our UNCERTAINTY in that estimate (epistemic),
    and its real per-unit-time VOLATILITY (aleatoric) — calibrated to the timescale it actually moves on."""
    name: str
    value: float
    est_sd: float = 0.0          # epistemic: how unsure we are of the current value (shrinks with better data)
    vol: float = 0.0             # aleatoric: true std of change per unit time (calibrated; sqrt(dt) scaling)
    lo: float = 0.0
    hi: float = 1.0


@dataclass
class StructuralModel:
    """A time-calibrated stochastic SCM over coupled continuous variables. `drift_fn(state, dt) -> {name:
    rate}` encodes the causal coupling (how variables push each other); dynamics are a proper diffusion."""
    variables: dict                          # name -> SVar
    drift_fn: object = None                  # causal structure: (state, dt) -> {name: d/dt}
    outcome_fn: object = None                # state -> outcome (numeric or a label)

    def simulate_once(self, horizon: float, dt: float = 1.0, *, epistemic: bool = True,
                      aleatoric: bool = True):
        """Return a `simulate_once(rng)` closure. epistemic/aleatoric toggle the two uncertainty sources
        so the ensemble variance can be DECOMPOSED (see variance_decomposition)."""
        def f(rng):
            state = {n: v.value + (rng.gauss(0, v.est_sd) if epistemic else 0.0)
                     for n, v in self.variables.items()}
            for n, v in self.variables.items():
                state[n] = min(v.hi, max(v.lo, state[n]))
            t = 0.0
            while t < horizon - 1e-9:
                step = min(dt, horizon - t)
                rates = self.drift_fn(state, step) if self.drift_fn else {}
                for n, v in self.variables.items():
                    drift = rates.get(n, 0.0) * step
                    noise = rng.gauss(0, 1) * v.vol * math.sqrt(step) if aleatoric else 0.0
                    state[n] = min(v.hi, max(v.lo, state[n] + drift + noise))
                t += step
            return self.outcome_fn(state) if self.outcome_fn else state
        return f

    def simulate_once_traced(self, horizon: float, dt: float = 1.0, *, epistemic: bool = True,
                             aleatoric: bool = True, interventions=None):
        """Like `simulate_once`, but also returns the EXOGENOUS FACTORS that defined this trajectory's
        'world' — each variable's initial draw (`name@0`, the epistemic world) and its cumulative aleatoric
        shock over the path (`name~shock`, the aleatoric world). These are independent by construction, so
        they are the honest knobs to attribute the outcome to (pivotal-branch analysis in swm.report.
        navigable). Distribution is identical to `simulate_once`; only the extra trace differs.

        `interventions` is an optional list of `(time, mutator)` scheduled do-operators applied DURING the
        rollout (a temporal/sequential intervention — an event injected at `time`, or step k of a policy):
        `mutator(state, rng)` mutates the live state once, the first time the clock reaches `time`. The
        Wiener diffusion runs between shocks, so state carries forward across steps — this is the substrate
        for temporal event-injection (Component 1) and sequential policies (Component 6)."""
        sched = sorted(interventions or [], key=lambda x: x[0])

        def _apply(state, rng, t, cursor):
            while cursor[0] < len(sched) and sched[cursor[0]][0] <= t + 1e-9:
                sched[cursor[0]][1](state, rng)
                cursor[0] += 1

        def f(rng):
            factors = {}
            state = {}
            for n, v in self.variables.items():
                draw = v.value + (rng.gauss(0, v.est_sd) if epistemic else 0.0)
                state[n] = min(v.hi, max(v.lo, draw))
                if epistemic and v.est_sd > 0:
                    factors[f"{n}@0"] = state[n]
            noise_sum = {n: 0.0 for n in self.variables}
            cursor = [0]
            _apply(state, rng, 0.0, cursor)                 # interventions scheduled at t=0
            t = 0.0
            while t < horizon - 1e-9:
                step = min(dt, horizon - t)
                rates = self.drift_fn(state, step) if self.drift_fn else {}
                for n, v in self.variables.items():
                    drift = rates.get(n, 0.0) * step
                    noise = (rng.gauss(0, 1) * v.vol * math.sqrt(step)) if aleatoric else 0.0
                    noise_sum[n] += noise
                    state[n] = min(v.hi, max(v.lo, state[n] + drift + noise))
                t += step
                _apply(state, rng, t, cursor)               # scheduled shocks that fall in (t-step, t]
            for n, v in self.variables.items():
                if aleatoric and v.vol > 0:
                    factors[f"{n}~shock"] = noise_sum[n]
            outcome = self.outcome_fn(state) if self.outcome_fn else state
            return outcome, factors
        return f


def variance_decomposition(model: "StructuralModel", horizon: float, dt: float = 1.0,
                           n: int = 4000) -> dict:
    """Split predictive spread into REDUCIBLE (epistemic — better estimates shrink it) and IRREDUCIBLE
    (aleatoric — the forecastability ceiling). Only meaningful for a numeric outcome."""
    total = montecarlo(model.simulate_once(horizon, dt, epistemic=True, aleatoric=True), n=n)
    aleatoric = montecarlo(model.simulate_once(horizon, dt, epistemic=False, aleatoric=True), n=n, seed=1)
    epistemic = montecarlo(model.simulate_once(horizon, dt, epistemic=True, aleatoric=False), n=n, seed=2)
    tv = total.get("var", 0.0)
    return {"total_sd": round(total["sd"], 4),
            "irreducible_sd": round(aleatoric["sd"], 4),      # noise-only: cannot be beaten by any model
            "reducible_sd": round(epistemic["sd"], 4),        # estimate-only: shrinks with better inference
            "irreducible_frac": round((aleatoric.get("var", 0) / tv) if tv > 1e-12 else 1.0, 3),
            "forecastable": (aleatoric.get("var", 0) / tv if tv > 1e-12 else 1.0) < 0.7}
