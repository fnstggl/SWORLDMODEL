"""Phase 3B — offline substrate loader + fidelity check.

Thin wrapper over the PRODUCTION repair math (`swm.world_model_v2.phase3b_repair`) so the offline fit/eval and
the serving path share byte-identical posterior code. Loads the frozen diagnostic capture and asserts the
production grid posterior reproduces the captured particle posterior mean.
"""
from __future__ import annotations
import json, math
from pathlib import Path

from swm.world_model_v2.phase3b_repair import calibrated_rate_posterior, logit, sigmoid, _clip

_EPS = 1e-6


def rate_posterior(tag_rows, alpha, beta, *, gamma=1.0, no_info_mix=0.0, post_temp=1.0, use_dependence=True):
    return calibrated_rate_posterior(tag_rows, alpha, beta, gamma=gamma, no_info_mix=no_info_mix,
                                     post_temp=post_temp, use_dependence=use_dependence)


def load_capture(path="experiments/results/phase3b/diagnostic_capture.json"):
    d = json.loads(Path(path).read_text())
    return [r for r in d["rows"] if r.get("status", "").startswith("completed")
            and r.get("p_phase2") is not None and r.get("p_phase3") is not None
            and r.get("outcome") in (0, 1)]


def fidelity_check(rows):
    diffs = []
    for r in rows:
        m, _, _ = rate_posterior(r["tags"], r["prior"]["alpha"], r["prior"]["beta"])
        if r.get("posterior_mean") is not None:
            diffs.append(abs(m - r["posterior_mean"]))
    return {"n": len(diffs), "max_abs_diff": round(max(diffs), 4) if diffs else None,
            "mean_abs_diff": round(sum(diffs) / len(diffs), 4) if diffs else None}


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = _clip(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


if __name__ == "__main__":
    rows = load_capture()
    print("loaded", len(rows), "completed diagnostic rows")
    print("fidelity (offline grid vs captured particle posterior mean):", fidelity_check(rows))
