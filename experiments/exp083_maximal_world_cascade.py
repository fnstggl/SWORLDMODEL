"""EXP-083 — The Maximal World, Part A: does modeling every person as an AGENT beat the mean-field shortcut
on a real tipping-point cascade?

The user's hypothesis: to predict what actually happens we must "rebuild the world" — model every person as
an agent with their own predisposition/state, on a shared social plane, and roll it forward. This is the
falsifiable test of that hypothesis in the ONE regime where richer world-modeling should pay (endogenous
cascades; EXP-072 showed the coupled dynamic wins here).

Three models forecast a real baby-name's popularity share H=10yr ahead from its as-of trajectory only
(identical leakage-free samples as EXP-072, so this is an apples-to-apples add-on):

  PERSISTENCE / TREND      — the dumb baselines (share stays / linear extrapolation).
  MEAN-FIELD (EXP-072)     — the coupled bandwagon+fatigue ODE: g = rho*g - lam*p. TWO parameters, the whole
                             population collapsed to one number p. This is the "calibrated shortcut."
  AGENT WORLD (this file)  — the MAXIMAL world: the population is a distribution of HETEROGENEOUS agents,
                             each with their own adoption THRESHOLD (Granovetter: some jump early, some need
                             everyone else first) and their own fashion-FATIGUE hazard (over-exposure makes
                             them abandon). Every agent sees the shared prevalence p (the social plane / the
                             coupling) and independently decides. The aggregate S-curve-then-crash EMERGES
                             from the threshold distribution rather than being imposed by an ODE.

The scientific question: heterogeneous-agent structure is strictly richer than the mean-field number. Does
that richness BUY turning-point accuracy, or does it just add variance (EXP-072's warning: calibration beats
completeness)? Both models fit their parameters on the SAME train names and are scored on the SAME held-out
test names — a fair fight. If the agent world wins at the turning points, "model every person" is a real
lever; if it ties/loses, the mean-field number already captured all the recoverable signal.

Run: python -m experiments.exp083_maximal_world_cascade
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.exp072_contagion import _samples, _score, _contagion_roll

DATA = "experiments/results/exp072/baby_names.json"
RESULT = "experiments/results/exp083_maximal_world_cascade.json"
H = 10

# ---- the AGENT WORLD: a binned heterogeneous-threshold population on a shared prevalence plane ----------
# Agents are binned by adoption threshold (fast: B bins instead of N individuals, exact for the aggregate).
# Bin b holds population mass m_b at threshold tau_b; a_b in [0,1] is the adopted fraction WITHIN the bin.
# Shared plane: everyone sees the same current prevalence p = sum_b m_b a_b.
B_THRESHOLDS = 15


def _threshold_bins(mu, sd):
    """A discretized Normal(mu, sd) threshold distribution over B bins — the population's heterogeneity."""
    xs = [mu + sd * (-2.5 + 5.0 * (i + 0.5) / B_THRESHOLDS) for i in range(B_THRESHOLDS)]
    w = [math.exp(-0.5 * ((x - mu) / sd) ** 2) for x in xs]
    s = sum(w)
    return xs, [wi / s for wi in w]


def _agent_roll(p0, g0, params, steps):
    """Roll the heterogeneous-agent cascade forward `steps` years from as-of (prevalence p0, momentum g0),
    and return the SHAPE anchored to the observed share: p0 * level_H / level_0. Anchoring keeps the forecast
    bounded and, for a name at rest (g0~0, fat=0), pins it at p0 (a proper fixed point) — while the internal
    agent dynamics still supply the rise/peak/CRASH shape.

    Each agent (threshold bin b) RELAXES toward its response to the shared social proof s = beta*level at
    speed alpha:  a_b += alpha*(logistic(k*(s - tau_eff_b)) - a_b).  Fashion FATIGUE drifts every agent's
    effective threshold UP with cumulative exposure (tau_eff = tau_b + fat*exposure), so a long-popular name
    eventually falls out of fashion and CRASHES. The S-curve-then-decline EMERGES from the distribution.
    """
    beta, k, mu, sd, fat, alpha = (params["beta"], params["k"], params["mu"], params["sd"],
                                   params["fat"], params["alpha"])
    tau, mass = _threshold_bins(mu, sd)
    p0f = max(1e-9, p0 / 100.0)
    a = [1.0 / (1.0 + math.exp(-k * (beta * p0f - tau[b]))) for b in range(B_THRESHOLDS)]  # consistent init
    level0 = sum(mass[b] * a[b] for b in range(B_THRESHOLDS))
    if level0 < 1e-9:
        return p0                                            # degenerate -> persistence
    exposure = 0.0
    boost = max(0.0, g0 / 100.0) * beta                      # recent momentum -> a decaying extra proof push
    level = level0
    for _ in range(steps):
        exposure += level
        s = beta * level + boost
        for b in range(B_THRESHOLDS):
            eq = 1.0 / (1.0 + math.exp(-k * (s - (tau[b] + fat * exposure))))       # fatigue lifts threshold
            a[b] = min(1.0, max(0.0, a[b] + alpha * (eq - a[b])))
        level = sum(mass[b] * a[b] for b in range(B_THRESHOLDS))
        boost *= 0.6                                         # momentum decays
    return max(0.0, p0 * level / level0)                     # anchored shape forecast


def _fit_agent(train_samples):
    """Fit the agent-world parameters on TRAIN names (same fit-on-train protocol as the mean-field ODE).
    Five params (beta, k, mu, sd, fatigue, alpha) don't need all 18k points — a fixed 1-in-6 subsample fits
    them and keeps the grid search tractable; the winner is re-scored on the FULL test set."""
    sub = train_samples[::6]
    best = None
    for beta in (2.0, 5.0, 9.0, 15.0):
        for k in (3.0, 8.0):
            for mu in (0.3, 0.6, 1.0):
                for sd in (0.2, 0.45):
                    for fat in (0.0, 0.03, 0.08):
                        for alpha in (0.2, 0.5, 0.9):
                            params = {"beta": beta, "k": k, "mu": mu, "sd": sd, "fat": fat, "alpha": alpha}
                            mae = _score(sub, lambda s, p=params: _agent_roll(s["p"], s["g"], p, H))
                            if best is None or mae < best[0]:
                                best = (mae, params)
    return best[1]


def run():
    names = json.loads(Path(DATA).read_text())
    keys = sorted(names)
    cut = int(0.6 * len(keys))
    train = {k: names[k] for k in keys[:cut]}
    test = {k: names[k] for k in keys[cut:]}
    tr, te = _samples(train), _samples(test)

    persistence = lambda s: s["p"]
    trend = lambda s: max(0.0, s["p"] + s["g"] * H)

    # mean-field (re-fit here for a clean same-run comparison)
    best_mf = None
    for rho in (0.2, 0.4, 0.6, 0.8, 0.95):
        for lam in (0.02, 0.05, 0.1, 0.2, 0.35):
            mae = _score(tr, lambda s, r=rho, l=lam: _contagion_roll(s["p"], s["g"], r, l, H))
            if best_mf is None or mae < best_mf[0]:
                best_mf = (mae, rho, lam)
    _, rho, lam = best_mf
    meanfield = lambda s: _contagion_roll(s["p"], s["g"], rho, lam, H)

    agent_params = _fit_agent(tr)
    agent = lambda s: _agent_roll(s["p"], s["g"], agent_params, H)

    def block(samples, label):
        return {"label": label, "n": len(samples),
                "persistence": round(_score(samples, persistence), 4),
                "trend": round(_score(samples, trend), 4),
                "mean_field": round(_score(samples, meanfield), 4),
                "agent_world": round(_score(samples, agent), 4)}

    overall = block(te, "ALL test points")
    turning = block([s for s in te if s["turning"]], "TURNING POINTS (near peak)")
    rising = block([s for s in te if s["rising"]], "RISING")
    stable = block([s for s in te if not s["turning"] and not s["rising"]], "STABLE")

    def cmp(b):
        base = min(b["persistence"], b["trend"])
        return {"agent_vs_best_simple": round((base - b["agent_world"]) / base, 4) if base else 0.0,
                "agent_vs_mean_field": round((b["mean_field"] - b["agent_world"]) / b["mean_field"], 4)
                if b["mean_field"] else 0.0}

    out = {"experiment": "Maximal World Part A — every-person-as-agent cascade vs the mean-field shortcut",
           "data": "SSA baby-name shares, 481 names 1880-2008, H=10yr, leakage-free (same samples as EXP-072)",
           "agent_params": agent_params, "mean_field_params": {"rho": rho, "lambda": lam},
           "ALL": overall, "TURNING_POINTS": turning, "RISING": rising, "STABLE": stable,
           "comparisons": {b["label"]: cmp(b) for b in (overall, turning, rising, stable)},
           "verdict_turning": ("AGENT WORLD wins turning points" if turning["agent_world"] <
                               min(turning["mean_field"], turning["persistence"], turning["trend"])
                               else "agent world does NOT beat the mean-field shortcut at turning points")}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-083  MAXIMAL WORLD, Part A: every-person-as-agent cascade vs the mean-field shortcut")
    print(f"  agent-world params: {agent_params}")
    print(f"  mean-field params:  rho={rho}, lambda={lam}    (H={H}yr, same leakage-free samples as EXP-072)")
    print(f"  {'regime':30s} {'n':>5s}  {'persist':>8s} {'trend':>8s} {'meanfld':>8s} {'AGENT':>8s}  winner")
    for b in (overall, rising, turning, stable):
        cand = {"persist": b["persistence"], "trend": b["trend"], "meanfld": b["mean_field"], "AGENT": b["agent_world"]}
        win = min(cand, key=cand.get)
        print(f"  {b['label']:30s} {b['n']:5d}  {b['persistence']:8.3f} {b['trend']:8.3f} "
              f"{b['mean_field']:8.3f} {b['agent_world']:8.3f}  -> {win}")
    tc = out["comparisons"]["TURNING POINTS (near peak)"]
    print(f"  TURNING POINTS: agent vs best-simple {tc['agent_vs_best_simple']:+}, "
          f"agent vs mean-field {tc['agent_vs_mean_field']:+}")
    print(f"  VERDICT: {out['verdict_turning']}")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
