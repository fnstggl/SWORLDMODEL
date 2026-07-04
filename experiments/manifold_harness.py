"""EXP-006: beat the prediction market at a FAIR information horizon (no cheating).

The earlier "we lose to Manifold" was against the market's CLOSING price -- near-hindsight,
unwinnable by construction. The legitimate ForecastBench-style KPI:

  Snapshot the market price at horizon T (fixed lead after creation). The predictor forecasts at T
  using only question text + as-of context, BLIND to the market price. Both are scored against the
  eventual resolution. KPI: predictor Brier < market@T Brier, aggregated -- and segmented by
  liquidity (thin vs deep) and topic, because a thin/early crowd is where an edge is plausible.

Honest prior (ForecastBench/Metaculus AIB): frontier models usually LOSE to liquid markets. The
goal is to measure exactly where we stand and iterate toward the segments we can win.

Contamination control: markets resolve AFTER the model's Jan-2026 cutoff; predictors get question
text only (never the price/resolution) and must NOT web-search (that would surface the outcome).

Usage:
  python -m experiments.manifold_harness fetch --target 140 --lead-hours 48
  python -m experiments.manifold_harness split --k 8
  python -m experiments.manifold_harness score --preds data/mf_pred_*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swm.eval.metrics import brier_score, log_loss

API = "https://api.manifold.markets/v0"
CUTOFF_MS = 1740787200000   # 2025-03-01: require resolution well after model training cutoff
_POOL = ThreadPoolExecutor(8)
_JUNK = re.compile(r"coinflip|managram|bronze league|will i |blink|daily|\bme\b|my \w+ league|"
                   r"whale|test market|permanent|resolves? (yes|no)\b", re.I)


def _get(path, retries=5):
    for a in range(retries):
        try:
            with urllib.request.urlopen(f"{API}/{path}", timeout=25) as r:
                return json.loads(r.read())
        except Exception:
            if a == retries - 1:
                return None
            time.sleep(0.4 * 2 ** a + random.random())


def _price_at(contract_id: str, t_ms: float, created_prob: float = 0.5) -> float | None:
    """Reconstruct the market probability at time t_ms from bet history (probAfter of the last
    bet strictly before t_ms). None if no bets before t (market had no price yet)."""
    bets = _get(f"bets?contractId={contract_id}&limit=1000")
    if not bets:
        return None
    bets = sorted(bets, key=lambda b: b["createdTime"])
    price = None
    for b in bets:
        if b["createdTime"] <= t_ms and b.get("probAfter") is not None:
            price = b["probAfter"]
        elif b["createdTime"] > t_ms:
            break
    return price


def fetch(target: int, lead_hours: float) -> None:
    lead_ms = lead_hours * 3600 * 1000
    packets, truth = [], []
    before = None
    scanned = 0
    while len(packets) < target and scanned < 60000:
        ms = _get(f"markets?limit=1000" + (f"&before={before}" if before else ""))
        if not ms:
            break
        scanned += len(ms)
        cand = [m for m in ms if m.get("outcomeType") == "BINARY" and m.get("isResolved")
                and m.get("resolution") in ("YES", "NO")
                and (m.get("resolutionTime") or 0) > CUTOFF_MS
                and m.get("uniqueBettorCount", 0) >= 8
                and (m.get("resolutionTime", 0) - m.get("createdTime", 0)) > (lead_ms + 2 * 86400000)
                and not _JUNK.search(m.get("question", ""))]
        def build(m):
            t = m["createdTime"] + lead_ms
            price = _price_at(m["id"], t)
            if price is None:
                return None
            return (m, price)
        for res in _POOL.map(build, cand):
            if res is None:
                continue
            m, price = res
            packets.append({"id": m["id"], "question": m["question"],
                            "created_iso": _iso(m["createdTime"]),
                            "closes_iso": _iso(m.get("closeTime") or m["resolutionTime"])})
            truth.append({"id": m["id"], "resolution": 1 if m["resolution"] == "YES" else 0,
                          "market_at_T": round(price, 4), "bettors": m.get("uniqueBettorCount"),
                          "lead_hours": lead_hours})
            if len(packets) >= target:
                break
        before = ms[-1]["id"]
        print(f"  scanned {scanned}, kept {len(packets)}")
    Path("data/mf_packets.json").write_text(json.dumps(packets, indent=1))
    Path("data/mf_truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {len(packets)} markets (question-only packets + hidden truth)")


def _iso(ms):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def split(k: int) -> None:
    packets = json.loads(Path("data/mf_packets.json").read_text())
    for j in range(k):
        Path(f"data/mf_batch_{j}.json").write_text(
            json.dumps(packets[j::k], indent=1))   # stride split
    print(f"wrote {k} question-only batch files (data/mf_batch_*.json)")


def score(pred_globs: list[str]) -> None:
    preds = {}
    for g in pred_globs:
        for f in glob.glob(g):
            for p in json.loads(Path(f).read_text()):
                preds[p["id"]] = min(0.99, max(0.01, p["p_yes"]))
    truth = {t["id"]: t for t in json.loads(Path("data/mf_truth.json").read_text())}
    ids = [i for i in preds if i in truth]
    def rep(label, idx):
        y = [truth[i]["resolution"] for i in idx]
        if not y:
            print(f"{label:<22} (no markets)"); return
        me = [preds[i] for i in idx]
        mk = [min(0.99, max(0.01, truth[i]["market_at_T"])) for i in idx]
        base = [0.5] * len(idx)
        wins = sum(1 for i in idx if abs(preds[i] - truth[i]["resolution"])
                   < abs(truth[i]["market_at_T"] - truth[i]["resolution"]))
        print(f"{label:<22} n={len(idx):<4} yes={sum(y)/len(y):.2f} | "
              f"model {brier_score(y,me):.3f}  market@T {brier_score(y,mk):.3f}  "
              f"coin {brier_score(y,base):.3f} | model beats market on {wins}/{len(idx)} ({wins/len(idx):.0%})")
    print("Brier (lower=better); 'beats market' = closer to the realized 0/1 than the market price\n")
    rep("ALL", ids)
    thin = [i for i in ids if truth[i]["bettors"] < 25]
    deep = [i for i in ids if truth[i]["bettors"] >= 25]
    rep("thin (<25 bettors)", thin)
    rep("deep (>=25 bettors)", deep)
    # markets where the market@T was uncertain (0.25-0.75) -- the hard, informative subset
    unc = [i for i in ids if 0.25 <= truth[i]["market_at_T"] <= 0.75]
    rep("market uncertain", unc)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--target", type=int, default=140)
    f.add_argument("--lead-hours", type=float, default=48)
    s = sub.add_parser("split"); s.add_argument("--k", type=int, default=8)
    sc = sub.add_parser("score"); sc.add_argument("--preds", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.target, a.lead_hours)
    elif a.cmd == "split":
        split(a.k)
    else:
        score(a.preds)


if __name__ == "__main__":
    main()
