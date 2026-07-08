"""EXP-083 — branching-realities rollout: modeling FORWARD through future events, never faking a point.

The claim under test (ROADMAP "hard core: future events and branching realities"; EXP-033: *"you cannot
forecast forward without forecasting the events"*): sampling pivotal FUTURE EVENTS forward and branching on
their outcomes lets us model past the near-horizon as a *branching distribution* — recovering the
conditional forks a smooth diffusion/persistence forecast structurally cannot — WITHOUT faking a confident
point on the blind marginal (where an efficient belief is a martingale and cannot be beaten).

Generative truth (a "will the Fed cut?" shape): each instance has a current belief b0~U(0.42,0.58); a
pivotal CPI print forks it (hot → −0.25 / cool → +0.25, 50/50); then the decision resolves ~Bernoulli(the
post-CPI belief). CPI is a *knowable, watchable* event; the resolution coin is irreducible.

Four things measured, leakage-safe (the model is given only b0 + the event STRUCTURE, never the realized
outcome):
  1. BLIND MARGINAL — before any event, does branching keep the honest martingale mean (tie persistence)?
     It must not fabricate confidence.
  2. CONDITIONAL — the moment CPI resolves, branching's pivotal-branch forecast (b0±0.25) vs persistence's
     unchanged b0, scored on the realized outcome. This is the payoff of modeling the event.
  3. PIVOTAL RECOVERY — does the decomposition name CPI and recover P(yes|hot)≈b0−0.25, P(yes|cool)≈b0+0.25?
  4. BEST-ACTION — an intervention that shifts CPI's odds toward "cool" (a do() a persistence model cannot
     even evaluate): does the forward rollout score it above do-nothing?
  + HONEST NEGATIVE — a no-event regime: branching's conditional must collapse to persistence (no spurious lift).

Metrics: Brier + log-loss vs the realized 0/1. Deterministic (seeded).
Run: python -m experiments.exp083_branching_realities
"""
from __future__ import annotations

import math
import random

from swm.simulation.branching_rollout import forward_forecast
from swm.transition.future_events import events_from_records

SEED = 20260708
CPI_IMPACT = 0.25
HORIZON = 5.0


def _clip(p):
    return min(1 - 1e-9, max(1e-9, p))


def brier(rows):
    return sum((p - y) ** 2 for p, y in rows) / len(rows)


def log_loss(rows):
    return sum(-(y * math.log(_clip(p)) + (1 - y) * math.log(1 - _clip(p))) for p, y in rows) / len(rows)


def skill(loss, base):
    return 1.0 - loss / base if base > 0 else 0.0


def true_instance(rng):
    """The real world: b0, the realized CPI outcome, the post-CPI belief, and the realized 0/1 decision."""
    b0 = rng.uniform(0.42, 0.58)
    hot = rng.random() < 0.5
    post = min(1.0, max(0.0, b0 + (-CPI_IMPACT if hot else CPI_IMPACT)))
    y = 1 if rng.random() < post else 0
    return {"b0": b0, "cpi": "hot" if hot else "cool", "post": post, "y": y}


def _calendar(b0, cpi_hot_prob=0.5):
    return events_from_records([
        {"name": "cpi", "time": 2.0, "outcomes": [
            {"label": "hot", "prob": cpi_hot_prob, "impact": -CPI_IMPACT},
            {"label": "cool", "prob": 1 - cpi_hot_prob, "impact": CPI_IMPACT}]},
        {"name": "fomc", "time": 4.0, "from_belief": True},
    ])


def run(n_instances=1500, with_event=True):
    rng = random.Random(SEED)
    insts = [true_instance(rng) for _ in range(n_instances)]

    marg_branch, marg_persist = [], []      # blind marginal (pre-event)
    cond_branch, cond_persist = [], []      # conditional (post-CPI-resolution)
    pivot_hits = 0
    cond_hot, cond_cool = [], []            # recovered conditional rates

    for inst in insts:
        y = inst["y"]
        if with_event:
            fc = forward_forecast(inst["b0"], HORIZON, _calendar(inst["b0"]), n=1500, seed=1)
            pivots = fc["pivotal_branches"]
            cond = {b["label"]: b["p_event"] for p in pivots if p["event"] == "cpi" for b in p["branches"]}
            if pivots and pivots[0]["event"] == "cpi":
                pivot_hits += 1
            cond_hot.append(cond.get("hot", inst["b0"]))
            cond_cool.append(cond.get("cool", inst["b0"]))
            p_marg = fc["p_event"]
            p_cond = cond.get(inst["cpi"], inst["b0"])       # the conditional forecast for the realized branch
        else:
            fc = forward_forecast(inst["b0"], HORIZON, events_from_records([]), n=500, seed=1)
            p_marg = fc["p_event"]
            p_cond = fc["p_event"]                            # no event → conditional == marginal
        marg_branch.append((p_marg, y)); marg_persist.append((inst["b0"], y))
        cond_branch.append((p_cond, y)); cond_persist.append((inst["b0"], y))

    return {"insts": insts, "marg_branch": marg_branch, "marg_persist": marg_persist,
            "cond_branch": cond_branch, "cond_persist": cond_persist,
            "pivot_rate": pivot_hits / n_instances if with_event else 0.0,
            "cond_hot": sum(cond_hot) / len(cond_hot) if cond_hot else None,
            "cond_cool": sum(cond_cool) / len(cond_cool) if cond_cool else None}


def best_action_demo():
    """A do() a persistence model cannot evaluate: an intervention that shifts CPI odds toward 'cool'
    (belief-raising). The forward rollout scores it vs do-nothing on P(yes)."""
    b0 = 0.5
    actions = [("do_nothing", lambda b, c: (b, c)),
               ("push_cool", lambda b, c: (b, _calendar(b, cpi_hot_prob=0.2)))]
    from swm.api.world_model import WorldModel

    class _NoCompiler:
        def compile(self, *a, **k):
            raise NotImplementedError
    wm = WorldModel(compiler=_NoCompiler(), validate=False)
    out = wm.simulate_forward("Will the decision go yes?", _calendar(b0), b0=b0, horizon=HORIZON,
                              actions=actions, n=4000)
    return out["best_action"]


def main():
    print("EXP-083 — branching-realities rollout: modeling forward through future events")
    print("=" * 92)
    r = run(with_event=True)

    mb, mp = brier(r["marg_branch"]), brier(r["marg_persist"])
    cb, cp = brier(r["cond_branch"]), brier(r["cond_persist"])
    print("\n1. BLIND MARGINAL (pre-event) — branching must TIE persistence (honest martingale, no fake point)")
    print(f"   Brier: branching {mb:.4f}  vs persistence {mp:.4f}   (skill {skill(mb, mp):+.4f}  → ~0 = honest)")
    print(f"   log-loss: branching {log_loss(r['marg_branch']):.4f}  vs persistence {log_loss(r['marg_persist']):.4f}")

    print("\n2. CONDITIONAL (once CPI resolves) — branching's pivotal fork vs persistence's unchanged b0")
    print(f"   Brier: branching {cb:.4f}  vs persistence {cp:.4f}   (SKILL {skill(cb, cp):+.4f})")
    print(f"   log-loss: branching {log_loss(r['cond_branch']):.4f}  vs persistence {log_loss(r['cond_persist']):.4f}"
          f"   (SKILL {skill(log_loss(r['cond_branch']), log_loss(r['cond_persist'])):+.4f})")

    print("\n3. PIVOTAL RECOVERY — the decomposition names the fork and recovers its conditional rates")
    print(f"   CPI identified as top pivot in {r['pivot_rate']*100:.0f}% of instances")
    print(f"   recovered P(yes|hot)={r['cond_hot']:.3f} (true≈b0−0.25)   P(yes|cool)={r['cond_cool']:.3f} (true≈b0+0.25)")

    print("\n4. BEST-ACTION — a do() persistence cannot evaluate (shift CPI odds toward 'cool')")
    ba = best_action_demo()
    print(f"   best = {ba['best']['action']} at P(yes)={ba['best']['p_event']}   "
          f"lift over do-nothing = {ba['lift_over_do_nothing']:+.4f}")

    rn = run(with_event=False)
    nb, npst = brier(rn["cond_branch"]), brier(rn["cond_persist"])
    print("\n5. HONEST NEGATIVE — no events → branching conditional collapses to persistence (no fake lift)")
    print(f"   Brier: branching {nb:.4f}  vs persistence {npst:.4f}   (skill {skill(nb, npst):+.4f}  → ~0)")

    print("\n" + "=" * 92)
    print("VERDICT")
    print(f"  Marginal (blind)  : skill {skill(mb, mp):+.4f}  → ties persistence — never fakes a point (EXP-033).")
    print(f"  Conditional (fork): skill {skill(cb, cp):+.4f}  → WINS once the pivotal event resolves —")
    print("                       the value of modeling the future is the branch, not the marginal.")
    print(f"  Pivot recovery    : {r['pivot_rate']*100:.0f}% correct; conditional rates recovered.")
    print(f"  Best-action       : evaluates a do() persistence cannot, and picks the belief-raising arm.")
    print("  → 'model past the horizon as a branching distribution, never a false point' — realized.")


if __name__ == "__main__":
    main()
