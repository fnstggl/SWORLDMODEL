"""Fair, no-cheat market comparison (spec Phase 6; formalizes manifold_harness scoring).

The ONLY legitimate way to compare a forecaster to a market:
- snapshot the market probability at a FIXED lead T (creation + Δ), reconstructed from bet history
  strictly before T — never the closing/near-resolution price;
- the forecaster predicts at T using only as-of context, BLIND to the market price;
- both are scored against the eventual resolution;
- segment by liquidity, market uncertainty, and information availability — because an edge, if it
  exists, lives in the thin/early/information-symmetric slice.

This module scores predictions against a truth file with market@T prices, and reports whether an
as-of-retrieval arm closes the information gap vs a no-retrieval arm.
"""
from __future__ import annotations

from swm.eval.metrics import brier_score, log_loss


def _clip(p, lo=0.01, hi=0.99):
    return min(hi, max(lo, p))


def compare(truth: list[dict], predictions: dict[str, float], *,
            market_key: str = "market_at_T") -> dict:
    """truth: list of {id, resolution(0/1), market_at_T, bettors, ...}. predictions: id -> p_yes.
    Returns overall + segmented Brier/log-loss for model vs market vs coin, plus head-to-head."""
    truth_by_id = {t["id"]: t for t in truth}
    ids = [i for i in predictions if i in truth_by_id]

    def report(label: str, idx: list[str]) -> dict | None:
        if not idx:
            return None
        y = [truth_by_id[i]["resolution"] for i in idx]
        me = [_clip(predictions[i]) for i in idx]
        mk = [_clip(truth_by_id[i][market_key]) for i in idx]
        coin = [0.5] * len(idx)
        wins = sum(1 for i in idx
                   if abs(predictions[i] - truth_by_id[i]["resolution"])
                   < abs(truth_by_id[i][market_key] - truth_by_id[i]["resolution"]))
        return {
            "segment": label, "n": len(idx), "yes_rate": round(sum(y) / len(y), 3),
            "model_brier": round(brier_score(y, me), 4),
            "market_brier": round(brier_score(y, mk), 4),
            "coin_brier": round(brier_score(y, coin), 4),
            "model_logloss": round(log_loss(y, me), 4),
            "market_logloss": round(log_loss(y, mk), 4),
            "model_beats_market_frac": round(wins / len(idx), 3),
        }

    segs = {
        "ALL": ids,
        "thin(<25 bettors)": [i for i in ids if truth_by_id[i].get("bettors", 0) < 25],
        "deep(>=25 bettors)": [i for i in ids if truth_by_id[i].get("bettors", 0) >= 25],
        "market_uncertain(0.25-0.75)": [i for i in ids
                                        if 0.25 <= truth_by_id[i][market_key] <= 0.75],
    }
    return {"segments": [r for r in (report(k, v) for k, v in segs.items()) if r]}


def retrieval_gap(truth: list[dict], no_retrieval: dict[str, float],
                  with_retrieval: dict[str, float], *, market_key: str = "market_at_T") -> dict:
    """Does as-of retrieval close the gap to the market? Compare the two prediction arms' Brier to
    the market's, overall and on the market-uncertain (information-symmetric) subset."""
    a = compare(truth, no_retrieval, market_key=market_key)["segments"]
    b = compare(truth, with_retrieval, market_key=market_key)["segments"]
    out = {}
    for ra, rb in zip(a, b):
        seg = ra["segment"]
        out[seg] = {
            "n": ra["n"],
            "no_retrieval_brier": ra["model_brier"],
            "with_retrieval_brier": rb["model_brier"],
            "market_brier": ra["market_brier"],
            "gap_closed": round((ra["model_brier"] - rb["model_brier"]), 4),  # >0 => retrieval helped
            "still_behind_market": round(rb["model_brier"] - ra["market_brier"], 4),
        }
    return out
