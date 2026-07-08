"""Simulation iteration loop (Phase 9): diagnose the biggest failure, apply a targeted fix, rerun,
record the delta — honestly, with fixes SELECTED ON A VALIDATION SPLIT (never the test set).

Each iteration:
  1. run the benchmark on the validation split (train is further split fit/val, temporally)
  2. identify the biggest loss source (which slice the simulation trails raw LLM + context most)
  3. classify the failure and apply ONE targeted fix from the menu
  4. re-fit on the fit-slice, re-evaluate on val, write the delta to
     experiments/simulation_iteration_log.jsonl
  5. stop when: 2 consecutive iters improve val log loss <1% relative, OR the simulation/hybrid
     beats raw LLM + context on a meaningful val slice, OR the fix menu is exhausted.

After the loop, the winning config is evaluated ONCE on the held-out TEST set (via the main
benchmark) so the reported numbers are not tuned on test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.simulation_vs_classifier_harness import (_classifier_vector, _sim_ctx, _slices,
                                                          build_records, fit_simulation, THR)
from swm.eval.metrics import log_loss
from swm.simulation.policies import PolicyParams

LOG = "experiments/simulation_iteration_log.jsonl"


def _ll(rows, eng):
    y = [1 if r["score"] >= THR else 0 for r in rows]
    p = [min(1 - 1e-6, max(1e-6, eng.predict(r["feats"], author_rep=r["author_rep"],
                                             ctx=_sim_ctx(r), n_samples=120, seed=9000 + i)["p_hit"]))
         for i, r in enumerate(rows)]
    return log_loss(y, p), y, p


def _slice_gap(val, eng):
    """Biggest slice where the simulation trails raw LLM + context (the failure to target)."""
    worst, gap = "all", -1e9
    for name, rows in _slices(val).items():
        if len(rows) < 15 or sum(1 for r in rows if r["score"] >= THR) < 3:
            continue
        y = [1 if r["score"] >= THR else 0 for r in rows]
        ps = [min(1 - 1e-6, max(1e-6, eng.predict(r["feats"], author_rep=r["author_rep"],
                                                  ctx=_sim_ctx(r), n_samples=120, seed=13 + i)["p_hit"]))
              for i, r in enumerate(rows)]
        pl = [min(1 - 1e-6, max(1e-6, r["p_c"] if r["p_c"] is not None else 0.1)) for r in rows]
        g = log_loss(y, ps) - log_loss(y, pl)      # >0 => sim worse than LLM+context here
        if g > gap:
            gap, worst = g, name
    return worst, round(gap, 4)


def run():
    recs, cut = build_records()
    train, test = recs[:cut], recs[cut:]
    vcut = int(0.8 * len(train))
    fit_slice, val = train[:vcut], train[vcut:]
    print(f"iteration loop: fit={len(fit_slice)} val={len(val)} (test={len(test)} held out)")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    log_lines = []

    # ---- iteration 0: baseline simulation (no state coupling) ----
    eng, params, _ = fit_simulation(fit_slice, base=PolicyParams(state_fp_gain=0.0))
    val_ll, _, _ = _ll(val, eng)
    worst, gap = _slice_gap(val, eng)
    rec0 = {"iter": 0, "fix": "baseline", "val_log_loss": round(val_ll, 4),
            "worst_slice_vs_llm": worst, "gap": gap, "params": params.to_dict()}
    log_lines.append(rec0)
    print(f"  iter0 baseline: val_ll {val_ll:.4f}  biggest failure: {worst} (gap {gap:+.4f} vs LLM+ctx)")

    best_ll, best_params, best_eng = val_ll, params, eng
    # ---- fix menu (each is a concrete, targeted change) ----
    fixes = [
        ("state_coupling(front-page)", dict(search_state_coupling=True, base=PolicyParams())),
        ("more_trajectories+recalibrate", dict(base=PolicyParams(state_fp_gain=best_params.state_fp_gain), n_traj=40)),
    ]
    stale = 0
    for k, (name, kwargs) in enumerate(fixes, start=1):
        # diagnose (recorded) then apply the fix
        eng_k, params_k, _ = fit_simulation(fit_slice, **kwargs)
        val_ll_k, _, _ = _ll(val, eng_k)
        rel = (best_ll - val_ll_k) / best_ll
        worst_k, gap_k = _slice_gap(val, eng_k)
        rec = {"iter": k, "fix": name, "val_log_loss": round(val_ll_k, 4),
               "rel_improvement": round(rel, 4), "worst_slice_vs_llm": worst_k, "gap": gap_k,
               "state_fp_gain": params_k.state_fp_gain, "accepted": val_ll_k < best_ll}
        log_lines.append(rec)
        print(f"  iter{k} {name}: val_ll {val_ll_k:.4f} (rel {rel:+.2%})  worst {worst_k} ({gap_k:+.4f})")
        if val_ll_k < best_ll:
            best_ll, best_params, best_eng = val_ll_k, params_k, eng_k
        if abs(rel) < 0.01:
            stale += 1
        else:
            stale = 0
        # stopping: simulation beats LLM+context on a meaningful val slice, or 2 stale iters
        if gap_k < 0 and worst_k != "all":
            rec["stop_reason"] = f"simulation beats LLM+context on val slice (worst gap now {gap_k})"
            print(f"    STOP: simulation no longer trails LLM+context on any val slice")
            break
        if stale >= 2:
            rec["stop_reason"] = "2 consecutive iters <1% relative improvement"
            print("    STOP: diminishing returns")
            break

    with open(LOG, "w") as f:
        for line in log_lines:
            f.write(json.dumps(line) + "\n")
    print(f"\n  best val config: state_fp_gain={best_params.state_fp_gain}  val_ll {best_ll:.4f}")
    print(f"  wrote {LOG}")
    print("  -> re-run `python -m experiments.simulation_vs_classifier_harness` for the held-out "
          "TEST numbers (loop selected on val only).")
    return best_params


if __name__ == "__main__":
    run()
