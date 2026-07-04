"""No-cheat Hacker News backtest harness (EXP-002).

Contamination-free by construction: all items are AFTER the predictor's training cutoff
(Jan 2026), so outcomes cannot have been memorized. Inputs and outcomes are written to
SEPARATE files; the predictor (human, LLM, or code) sees only the inputs file, commits
predictions, and only then is the outcomes file opened by the scorer.

Usage:
  python -m experiments.hn_harness fetch --start 2026-03-01 --end 2026-03-08 \
      --n 150 --out data/hn_train           # training window: outcomes visible, for priors
  python -m experiments.hn_harness fetch --start 2026-05-01 --end 2026-05-08 \
      --n 120 --out data/hn_test            # test window: DO NOT OPEN *_outcomes.json
  python -m experiments.hn_harness score --pred data/round1_predictions.json \
      --inputs data/hn_test_inputs.json --outcomes data/hn_test_outcomes.json \
      --train data/hn_train
"""
from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss

API = "https://hacker-news.firebaseio.com/v0"
THRESHOLDS = [10, 40, 100]


def _get(path: str, retries: int = 5):
    import time as _time

    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "swm-exp002"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == retries - 1:
                return None
            _time.sleep(0.5 * (2 ** attempt) + random.random())


def _ts(datestr: str) -> float:
    return datetime.fromisoformat(datestr).replace(tzinfo=timezone.utc).timestamp()


def _find_id_at(ts: float, lo: int = 1, hi: int | None = None) -> int:
    """Binary search: smallest item id with time >= ts."""
    hi = hi or _get("maxitem.json")
    while lo < hi:
        mid = (lo + hi) // 2
        it = _get(f"item/{mid}.json")
        t = (it or {}).get("time")
        # walk forward past deleted/timeless items
        step = mid
        while t is None and step < hi:
            step += 1
            it = _get(f"item/{step}.json")
            t = (it or {}).get("time")
        if t is None or t >= ts:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fetch_window(start: str, end: str, n_stories: int, out_prefix: str, seed: int = 0) -> None:
    t0, t1 = _ts(start), _ts(end)
    print(f"locating id range for [{start}, {end}) ...")
    id0, id1 = _find_id_at(t0), _find_id_at(t1)
    print(f"ids [{id0}, {id1})  (~{id1-id0} items)")
    rng = random.Random(seed)
    stories, tried = [], 0
    pool = ThreadPoolExecutor(10)
    while len(stories) < n_stories and tried < 60 * n_stories:
        batch = [rng.randrange(id0, id1) for _ in range(120)]
        tried += len(batch)
        for it in pool.map(lambda i: _get(f"item/{i}.json"), batch):
            if (it and it.get("type") == "story" and not it.get("deleted")
                    and not it.get("dead") and it.get("title") and "score" in it):
                stories.append(it)
                if len(stories) >= n_stories:
                    break
        print(f"  {len(stories)}/{n_stories} stories after {tried} items")
    # de-dup & author profiles
    seen, uniq = set(), []
    for s in stories:
        if s["id"] not in seen:
            seen.add(s["id"])
            uniq.append(s)
    authors = sorted({s["by"] for s in uniq if s.get("by")})
    profiles = dict(zip(authors, pool.map(lambda a: _get(f"user/{a}.json"), authors)))

    inputs, outcomes = [], []
    for s in uniq:
        u = profiles.get(s.get("by")) or {}
        dt = datetime.fromtimestamp(s["time"], tz=timezone.utc)
        url = s.get("url", "")
        domain = url.split("/")[2].removeprefix("www.") if "://" in url else ""
        inputs.append({
            "id": s["id"], "title": s["title"], "author": s.get("by"),
            "author_karma": u.get("karma"),          # CAVEAT: current karma (small leak, noted)
            "author_created": u.get("created"),
            "domain": domain, "is_text_post": not url,
            "hour_utc": dt.hour, "weekday": dt.weekday(),
            "date": dt.date().isoformat(),
        })
        outcomes.append({"id": s["id"], "score": s["score"],
                         "descendants": s.get("descendants", 0)})
    Path(f"{out_prefix}_inputs.json").write_text(json.dumps(inputs, indent=1))
    Path(f"{out_prefix}_outcomes.json").write_text(json.dumps(outcomes, indent=1))
    print(f"wrote {len(inputs)} stories -> {out_prefix}_inputs.json / _outcomes.json")


# ---------- baselines & scoring ----------

def _features(x: dict) -> list[float]:
    karma = x.get("author_karma") or 1
    return [
        1.0 if x["is_text_post"] else 0.0,
        math.log(1 + karma),
        1.0 if x["title"].lower().startswith(("show hn", "ask hn")) else 0.0,
        min(1.0, len(x["title"]) / 80.0),
        1.0 if x["weekday"] >= 5 else 0.0,
        1.0 if 14 <= x["hour_utc"] <= 20 else 0.0,   # US daytime
    ]


def score(pred_path: str, inputs_path: str, outcomes_path: str, train_prefix: str) -> None:
    preds = {p["id"]: p for p in json.loads(Path(pred_path).read_text())}
    inputs = json.loads(Path(inputs_path).read_text())
    outcomes = {o["id"]: o for o in json.loads(Path(outcomes_path).read_text())}
    tr_in = json.loads(Path(f"{train_prefix}_inputs.json").read_text())
    tr_out = {o["id"]: o for o in json.loads(Path(f"{train_prefix}_outcomes.json").read_text())}

    from swm.transition.readout import LogisticReadout
    ids = [x["id"] for x in inputs if x["id"] in preds]
    print(f"scoring {len(ids)} predictions (of {len(inputs)} stories)\n")
    print(f"{'threshold':<10} {'model':<22} {'logloss':>8} {'brier':>8} {'ece':>7}")
    for thr in THRESHOLDS:
        y = [1 if outcomes[i]["score"] >= thr else 0 for i in ids]
        base = sum(1 for x in tr_in if tr_out[x["id"]]["score"] >= thr) / len(tr_in)
        rows = {
            "claude": [max(1e-4, min(1 - 1e-4, preds[i][f"p_ge_{thr}"])) for i in ids],
            "base_rate(train)": [base] * len(ids),
        }
        Xtr = [_features(x) for x in tr_in]
        ytr = [1 if tr_out[x["id"]]["score"] >= thr else 0 for x in tr_in]
        if len(set(ytr)) == 2:
            m = LogisticReadout(seed=0).fit(Xtr, ytr)
            by_id = {x["id"]: x for x in inputs}
            rows["logistic(features)"] = [m.predict_proba(_features(by_id[i])) for i in ids]
        for name, p in rows.items():
            print(f">= {thr:<7} {name:<22} {log_loss(y, p):>8.4f} "
                  f"{brier_score(y, p):>8.4f} {expected_calibration_error(y, p):>7.4f}")
        print(f"{'':10} test base rate: {sum(y)/len(y):.3f}  (train: {base:.3f})")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--n", type=int, default=120)
    f.add_argument("--out", required=True)
    f.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--inputs", required=True)
    s.add_argument("--outcomes", required=True)
    s.add_argument("--train", required=True)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch_window(a.start, a.end, a.n, a.out, a.seed)
    else:
        score(a.pred, a.inputs, a.outcomes, a.train)


if __name__ == "__main__":
    main()
