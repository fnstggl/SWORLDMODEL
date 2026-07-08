"""Uniform scored-validation harness — one scoreboard across every compiler mechanism.

The compiler routes each question to a different mechanism, so a fair validation scores each with the
metric that fits its outcome, then reports them side by side against the honest baseline that mechanism
must beat. Two record kinds:

  - BINARY  {mechanism, p, y, base}         -> Brier + accuracy + skill over the base rate (committee,
                                               single_agent ranking, any yes/no event);
  - SHARE   {mechanism, pred, truth, marginal, lo, hi} -> share-RMSE + interval coverage + coupling skill
                                               over the marginal composite (electorate, generic_scm).

The point is not a single number — it is whether the COMPILED model, produced through the one front door,
beats the naive baseline on REAL resolved outcomes, mechanism by mechanism.
"""
from __future__ import annotations

from collections import defaultdict

from swm.eval.metrics import brier_score, log_loss
from swm.eval.population_metrics import coupling_skill, interval_coverage, share_rmse


def score_binary(records: list) -> dict:
    """records: [{mechanism, p, y, base}]. Per-mechanism Brier/acc + skill over the base-rate baseline."""
    out = {}
    by_m = defaultdict(list)
    for r in records:
        by_m[r["mechanism"]].append(r)
    for m, rs in by_m.items():
        y = [r["y"] for r in rs]
        p = [min(1 - 1e-6, max(1e-6, r["p"])) for r in rs]
        base = [min(1 - 1e-6, max(1e-6, r.get("base", sum(y) / len(y)))) for r in rs]
        acc = sum(1 for r in rs if (r["p"] > 0.5) == (r["y"] > 0.5)) / len(rs)
        out[m] = {"n": len(rs), "brier": round(brier_score(y, p), 4),
                  "brier_baseline": round(brier_score(y, base), 4),
                  "accuracy": round(acc, 4), "base_rate": round(sum(y) / len(y), 4),
                  "brier_skill_vs_base": round(1 - brier_score(y, p) / max(1e-9, brier_score(y, base)), 4)}
    return out


def score_share(records: list) -> dict:
    """records: [{mechanism, pred, truth, marginal, lo, hi}]. RMSE + coverage + coupling skill per mech."""
    out = {}
    by_m = defaultdict(list)
    for r in records:
        by_m[r["mechanism"]].append(r)
    for m, rs in by_m.items():
        truth = [r["truth"] for r in rs]
        pred = [r["pred"] for r in rs]
        marg = [r["marginal"] for r in rs]
        card = {"n": len(rs), "rmse": round(share_rmse(truth, pred), 4),
                "rmse_marginal": round(share_rmse(truth, marg), 4),
                "coupling_skill": coupling_skill(truth, marg, pred)["skill"]}
        if all("lo" in r for r in rs):
            card["interval_coverage"] = interval_coverage(truth, [r["lo"] for r in rs],
                                                          [r["hi"] for r in rs], nominal=0.8)
        out[m] = card
    return out
