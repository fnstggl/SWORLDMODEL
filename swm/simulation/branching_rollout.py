"""Branching-realities rollout — Monte-Carlo over sampled future-event trajectories.

The engine that turns a forward question into a calibrated distribution WITHOUT enumerating 2ⁿ realities and
WITHOUT picking one branch (the ROADMAP's "hard core: future events and branching realities"). Each of K
trajectories walks time forward: between events the belief evolves under a pluggable CONTINUOUS step (a
diffusion by default — or the calibrated transition operator, dropped in here), and at each dated event it
SAMPLES the discrete outcome, applies that outcome's jump (or resolves and stops), and RECORDS which branch
it took. The K terminal outcomes are the forecast; scaling is linear in K, not 2ⁿ.

The object this produces that a smooth diffusion structurally cannot: the **pivotal-branch decomposition**.
When one event makes the future genuinely bimodal ("25% if the Fed holds, 85% if it cuts"), the marginal
mean (~55%) is a number no one should act on. `pivotal_branches` reads the per-trajectory branch record and
surfaces the fork explicitly — conditional outcome per branch, the branch's probability, and how much of the
spread that event resolves (its share of variance = what resolving it would buy you = *what to watch*). The
remaining within-branch spread is the irreducible floor (surprise shocks + the resolving draw) — the honest
"model past the horizon as a branching distribution, never a false point."

Composition contract (so this does NOT collide with the transition operator): this engine owns the DISCRETE
jumps + branching; the `continuous_step(belief, dt, rng) -> belief` you pass owns the between-event drift/
volatility. Pass a calibrated transition operator there and the two become one jump-diffusion.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.transition.future_events import EventCalendar


def martingale_step(vol: float = 0.0):
    """Default continuous step: a driftless random walk (EXP-033 — an efficient belief is a martingale, so
    NO endogenous drift) with optional per-√dt volatility. Replace with the calibrated transition operator
    for structural/population quantities where drift is real (EXP-045)."""
    def step(belief, dt, rng):
        return belief + (vol * math.sqrt(max(0.0, dt)) * rng.gauss(0, 1) if vol > 0 else 0.0)
    return step


@dataclass
class BranchingRollout:
    """Monte-Carlo over event trajectories. `continuous_step(belief, dt, rng)->belief` is the between-event
    dynamics (pluggable — the transition operator slots in here); the calendar supplies the discrete forks."""
    calendar: EventCalendar
    continuous_step: object = None
    dt: float = 1.0
    lo: float = 0.0
    hi: float = 1.0

    def _clip(self, b):
        return min(self.hi, max(self.lo, b))

    def simulate_once(self, b0: float, horizon: float, rng):
        """One trajectory → (terminal_outcome, {event_name: branch_label}). If a resolving event fires, the
        terminal outcome is its resolved value; otherwise it is the belief at the horizon."""
        step_fn = self.continuous_step or martingale_step()
        b = self._clip(b0)
        branches = {}
        t = 0.0
        while t < horizon - 1e-9:
            step = min(self.dt, horizon - t)
            b = self._clip(step_fn(b, step, rng))                        # continuous between-event drift/vol
            for imp in self.calendar.hazard.sample_impacts(step, rng):   # unscheduled surprises
                b = self._clip(b + imp)
            for ev in self.calendar.scheduled_in(t, t + step):           # dated events → BRANCH
                label, (kind, val) = ev.sample(rng, b)
                branches[ev.name] = label
                if kind == "resolve":
                    return val, branches                                 # question settled; trajectory ends
                b = self._clip(b + val)
            t += step
        return b, branches

    def run(self, b0: float, horizon: float, *, n: int = 4000, seed: int = 0) -> dict:
        """Ensemble → terminal distribution + the per-trajectory branch records (for the decomposition)."""
        rng = random.Random(seed)
        outs, recs = [], []
        for _ in range(n):
            o, br = self.simulate_once(b0, horizon, rng)
            outs.append(o)
            recs.append(br)
        s = sorted(outs)
        mean = sum(outs) / n
        var = sum((o - mean) ** 2 for o in outs) / n

        def q(p):
            return s[min(n - 1, int(p * n))]
        return {"p_event": round(mean, 4), "mean": round(mean, 4), "sd": round(var ** 0.5, 4), "var": var,
                "interval_80": [round(q(.1), 4), round(q(.9), 4)],
                "p05": round(q(.05), 4), "p50": round(q(.5), 4), "p95": round(q(.95), 4),
                "n": n, "_outcomes": outs, "_branches": recs}


def pivotal_branches(result: dict, calendar: EventCalendar, *, top: int = 3, min_frac: float = 0.02) -> list:
    """The decomposition. For each event, group trajectories by the branch they took and compute the
    conditional outcome per branch + the event's share of total variance (η², the between-branch variance /
    total — how much resolving THIS event would sharpen the forecast). Ranked most-pivotal first.

    This is the 'never fake a point' object: instead of the marginal mean, return "conditional on E: X if
    branch A (p=a), Y if branch B (p=b)" for the events that actually fork the outcome."""
    outs = result["_outcomes"]
    recs = result["_branches"]
    n = len(outs)
    if n == 0:
        return []
    grand = sum(outs) / n
    tot_var = sum((o - grand) ** 2 for o in outs) / n
    rows = []
    for ev in calendar.events:
        if ev.from_belief:                          # the belief's own coin-flip resolution is the aleatoric
            continue                                # FLOOR, not a watchable fork — it explains itself, trivially
        groups = {}
        for o, br in zip(outs, recs):
            if ev.name in br:
                groups.setdefault(br[ev.name], []).append(o)
        m = sum(len(v) for v in groups.values())
        if m == 0:
            continue
        between = 0.0
        conds = []
        for label, vals in groups.items():
            f = len(vals) / m
            cm = sum(vals) / len(vals)
            between += f * (cm - grand) ** 2
            conds.append({"label": label, "prob": round(f, 4), "p_event": round(cm, 4)})
        conds.sort(key=lambda c: -c["p_event"])
        eta2 = between / tot_var if tot_var > 1e-12 else 0.0             # variance share this event resolves
        rows.append({"event": ev.name, "time": ev.time, "pivotality": round(eta2, 4),
                     "spread": round(max(c["p_event"] for c in conds) - min(c["p_event"] for c in conds), 4),
                     "branches": conds})
    rows = [r for r in rows if r["pivotality"] >= min_frac]
    rows.sort(key=lambda r: -r["pivotality"])
    return rows[:top]


def forward_forecast(b0: float, horizon: float, calendar: EventCalendar, *, continuous_step=None,
                     dt: float = 1.0, n: int = 4000, seed: int = 0, top_pivots: int = 3) -> dict:
    """The forward-question front object: run the branching ensemble and return the calibrated distribution,
    the pivotal-branch decomposition, and the reducible/irreducible split — never a bare point.

    reducible_frac = the share of variance the KNOWN pivotal events resolve (what watching them would buy);
    irreducible_frac = the within-branch remainder (surprise hazard + the resolving Bernoulli draw) — the
    forecastability floor. This REPLACES abstention: past the horizon we still return the full branching
    distribution and say which forks are reducible vs irreducible, rather than declining to answer."""
    roll = BranchingRollout(calendar=calendar, continuous_step=continuous_step, dt=dt)
    res = roll.run(b0, horizon, n=n, seed=seed)
    pivots = pivotal_branches(res, calendar, top=top_pivots)
    reducible = sum(p["pivotality"] for p in pivots)                     # variance share known events resolve
    reducible = max(0.0, min(1.0, reducible))
    out = {"p_event": res["p_event"], "interval_80": res["interval_80"], "sd": res["sd"],
           "p50": res["p50"], "distribution": {"p05": res["p05"], "p50": res["p50"], "p95": res["p95"]},
           "pivotal_branches": pivots,
           "reducible_frac": round(reducible, 3), "irreducible_frac": round(1.0 - reducible, 3),
           "n": res["n"],
           "headline": _forward_headline(res, pivots)}
    return out


def _forward_headline(res: dict, pivots: list) -> str:
    base = f"P(event) = {res['p_event']}  (80% interval {res['interval_80']})"
    if pivots:
        p = pivots[0]
        forks = "; ".join(f"{b['p_event']} if {p['event']}={b['label']} (p={b['prob']})" for b in p["branches"])
        return f"{base} — PIVOT {p['event']}: {forks}"
    return base
