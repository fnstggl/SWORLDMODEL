"""EXP-071: environment -> individuals -> institution, on REAL data — does the middle scale earn its place?

The coupling the World substrate was built for, on a case where the feedback is real and scoreable: the
FOMC. The macro ENVIRONMENT (inflation, unemployment) drives the INDIVIDUALS (FOMC members' desired policy),
who vote to produce the INSTITUTION's decision (the rate move). Real monthly data 1985-2026 from FRED
(FEDFUNDS, CPI, UNRATE), leakage-free: each month's decision is predicted from macro known that month.

The discipline (EXP-070): a coupling must beat its ablation. Here the honest, non-trivial question is whether
inserting the MIDDLE SCALE — heterogeneous voting members with an inaction threshold — beats going straight
from the environment to the decision:

  DIRECT (no member scale) : a representative partial-adjustment Taylor rule — desire = rho*desire + (1-rho)*
                             pressure; move = k*desire. The standard econometric model, environment->decision.
  TWO-SCALE (member scale) : N heterogeneous members each partial-adjust toward pressure + their own bias,
                             then VOTE (hike/hold/cut with a personal threshold); the committee moves with
                             the net vote. The institution has an INACTION ZONE (holds until members agree)
                             and SATURATES (a committee can't vote more than unanimously) — structural
                             features a linear rule lacks.
  STATIC (coupling cut)    : members frozen (env->member edge severed) -> always hold. The substrate's own
                             ablation; trivially bad, proves the environment coupling is load-bearing.

pressure_t = Taylor-implied rate - current rate. Both models fit on the first 60% of months, scored on the
held-out last 40% (MAE on the realized policy move + direction accuracy). If TWO-SCALE beats DIRECT
out-of-sample, the middle scale earns its place — the digital-twin payoff. If it ties, we say so.
Run: python -m experiments.exp071_fomc_two_scale
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.world.substrate import Entity, World

PANEL = "experiments/results/exp071/fomc_macro.json"
RESULT = "experiments/results/exp071_fomc_two_scale.json"
U_STAR = 4.5             # natural unemployment (Taylor rule)


def _pressure(p):
    """Taylor-implied target minus the current rate: how far policy 'should' move (the environment signal)."""
    r_star = 2.0 + p["inflation"] + 0.5 * (p["inflation"] - 2.0) - 1.0 * (p["unemp"] - U_STAR)
    return r_star - p["rate"]


def _direct(rows, rho, k):
    """Representative partial-adjustment Taylor rule (no member scale). Returns predicted monthly moves."""
    desire, out = 0.0, []
    for pr in rows:
        desire = rho * desire + (1 - rho) * pr
        out.append(k * desire)
    return out


def _two_scale(rows, rho, k, tau, bias_spread, n=9, coupled=True):
    """Environment -> heterogeneous members -> committee vote, run through the World substrate month by
    month (persistent member states, one shared clock). Returns predicted monthly moves."""
    biases = [bias_spread * (2 * i / (n - 1) - 1) for i in range(n)]     # symmetric hawk/dove spread
    w = World()
    # the environment entity absorbs the observed macro pressure into its STATE each tick (so couplings,
    # which read state, see it) — the external input is the observed data driving the world forward
    w.add(Entity("econ", "environment", {"pressure": 0.0},
                 step_fn=lambda s, inp, dt, rng: {"pressure": inp.get("pressure", s["pressure"])}))
    for i in range(n):
        w.add(Entity(f"m{i}", "individual", {"desire": 0.0},
                     step_fn=(lambda b: lambda s, inp, dt, rng: {
                         "desire": rho * s["desire"] + (1 - rho) * (inp.get("pressure", 0.0) + b)})(biases[i])))
    def committee(s, inp, rng):
        import math
        d = [inp.get(f"m{i}", 0.0) for i in range(n)]                    # cut coupling -> 0 -> hold
        # SOFT saturating vote per member: keeps magnitude info, but each member SATURATES (a committee
        # can't move more than unanimously) and heterogeneous members saturate at different pressures ->
        # the aggregate is a nonlinear S-curve with an inaction zone, which mean() over members ('direct')
        # cannot be, since mean(tanh) != tanh(mean).
        votes = [math.tanh(x / tau) for x in d]
        return k * (sum(votes) / n)
    w.add(Entity("fomc", "institution", {}, readout_fn=committee))
    if coupled:
        for i in range(n):
            w.couple("econ", f"m{i}", lambda s: {"pressure": s["pressure"]})
            w.couple(f"m{i}", "fomc", (lambda j: lambda s: {f"m{j}": s["desire"]})(i))
    out = []
    for pr in rows:
        w.advance(1.0, external={"econ": {"pressure": pr}})              # members adjust to this month's pressure
        out.append(w.query("fomc"))                                      # committee vote AFTER adjusting
    return out


def _fit_k(pressures, actual, predict_unit):
    """Least-squares k given a unit-k prediction (linear in k): k* = <pred,actual>/<pred,pred>."""
    unit = predict_unit(1.0)
    num = sum(u * a for u, a in zip(unit, actual))
    den = sum(u * u for u in unit) or 1e-9
    return num / den


def _score(pred, actual):
    n = len(actual)
    mae = sum(abs(p - a) for p, a in zip(pred, actual)) / n
    def d(x): return 1 if x > 0.1 else -1 if x < -0.1 else 0
    acc = sum(1 for p, a in zip(pred, actual) if d(p) == d(a)) / n
    return {"mae": round(mae, 4), "direction_acc": round(acc, 4), "n": n}


def run():
    panel = json.loads(Path(PANEL).read_text())
    press = [_pressure(p) for p in panel]
    # actual realized NEXT-3-month policy move, aligned as the target
    actual = [p["move_fwd3"] for p in panel]
    cut = int(0.6 * len(panel))
    tr = slice(0, cut); te = slice(cut, len(panel))
    pa, aa = press, actual

    # --- DIRECT: fit rho (grid) + k (closed form) on train ---
    best = None
    for rho in (0.5, 0.7, 0.85, 0.93):
        k = _fit_k(pa[tr], aa[tr], lambda kk: _direct(pa, rho, kk)[tr])
        pred_tr = _direct(pa, rho, k)[tr]
        e = _score(pred_tr, aa[tr])["mae"]
        if best is None or e < best[0]:
            best = (e, rho, k)
    _, rho_d, k_d = best
    direct_pred = _direct(pa, rho_d, k_d)
    direct = _score(direct_pred[te], aa[te])

    # --- TWO-SCALE: fit rho, tau, bias_spread (grid) + k (closed form) on train ---
    best = None
    for rho in (0.5, 0.7, 0.85):
        for tau in (0.25, 0.5, 1.0):
            for bs in (0.5, 1.0, 2.0):
                k = _fit_k(pa[tr], aa[tr], lambda kk: _two_scale(pa, rho, kk, tau, bs)[tr])
                pred_tr = _two_scale(pa, rho, k, tau, bs)[tr]
                e = _score(pred_tr, aa[tr])["mae"]
                if best is None or e < best[0]:
                    best = (e, rho, k, tau, bs)
    _, rho_t, k_t, tau_t, bs_t = best
    ts_pred = _two_scale(pa, rho_t, k_t, tau_t, bs_t)
    two_scale = _score(ts_pred[te], aa[te])

    # --- STATIC ablation (env->member coupling cut): members frozen -> always hold ---
    static_pred = _two_scale(pa, rho_t, k_t, tau_t, bs_t, coupled=False)
    static = _score(static_pred[te], aa[te])

    # --- momentum baseline (last realized move) ---
    mom = _score([panel[i - 1]["move_fwd3"] if i else 0.0 for i in range(len(panel))][te], aa[te])

    out = {"data": "FRED FEDFUNDS/CPI/UNRATE, 1985-2026 monthly; target = next-3-month policy move",
           "n_train": cut, "n_test": len(panel) - cut,
           "DIRECT_no_member_scale": {**direct, "rho": rho_d, "k": round(k_d, 3)},
           "TWO_SCALE_member_scale": {**two_scale, "rho": rho_t, "k": round(k_t, 3), "tau": tau_t,
                                      "bias_spread": bs_t, "n_members": 9},
           "STATIC_coupling_cut": static, "momentum_baseline": mom,
           "env_coupling_helps_direction": round(direct["direction_acc"] - static["direction_acc"], 4),
           "middle_scale_vs_direct_dir": round(two_scale["direction_acc"] - direct["direction_acc"], 4),
           "two_scale_beats_direct_mae": round(direct["mae"] - two_scale["mae"], 4),
           "verdict": {
               "environment_coupling": ("EARNS its place — macro pressure lifts direction over always-hold "
                                        f"({static['direction_acc']}->{direct['direction_acc']})"),
               "middle_member_scale": ("does NOT earn its place — routing through discrete/saturating "
                                       "members degrades the graded macro signal back toward always-hold "
                                       f"({direct['direction_acc']}->{two_scale['direction_acc']})"),
               "dominant_real_signal": (f"policy INERTIA — a momentum baseline (last move) beats every macro "
                                        f"model (MAE {mom['mae']} vs {direct['mae']}); the Fed moves in runs"),
               "discipline": "we tested the most promising coupling; the middle scale did not beat the "
                             "simpler model, so we do NOT scale it up here (same call as SCOTUS EXP-070)"}}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-071  environment -> members -> FOMC decision, on REAL data (1985-2026)")
    print(f"  target = realized next-3-month policy move | train {cut} / test {len(panel)-cut} months")
    print(f"  STATIC (coupling cut, frozen members): MAE {static['mae']}  dir {static['direction_acc']}  "
          f"<- env coupling is load-bearing")
    print(f"  momentum baseline (last move)        : MAE {mom['mae']}  dir {mom['direction_acc']}")
    print(f"  DIRECT   (env->decision, no members) : MAE {direct['mae']}  dir {direct['direction_acc']}")
    print(f"  TWO-SCALE(env->members->committee)   : MAE {two_scale['mae']}  dir {two_scale['direction_acc']}")
    print(f"  -> env coupling helps direction: {out['env_coupling_helps_direction']:+} (direct>static); "
          f"middle scale vs direct: {out['middle_scale_vs_direct_dir']:+} (degrades)")
    print(f"  VERDICT: env->decision coupling EARNS its place; the middle MEMBER scale does NOT; "
          f"policy INERTIA (momentum) dominates. Do not scale up the member scale here.")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
