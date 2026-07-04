"""EXP-003 harness: individual-level author priors + distributional (band) scoring.

Upgrades over hn_harness.py, from first-principles bottleneck analysis:

1. MAGNITUDE = a mixture, not a scalar. Score with an ORDINAL band distribution
   (<10 / 10-39 / 40-99 / 100-299 / 300+) via proper multiclass log loss, so tail
   calibration is explicit. Predictions carry P(>=10,40,100,300); bands are derived.

2. UNCUED CONTEXT = missing the author's own track record. We fetch each author's
   PAST submissions (temporally filtered to time < target) and compute a clean prior
   (n_past, median/max past score, frac>=10). This is the hierarchical persona prior.

3. STRONGER BASELINE. Beating a global base rate is weak; we add a SEGMENT model that
   already knows the author prior + post type. The real question: does LLM language
   judgment beat a model that already knows how this author usually does?

4. POOLING. `pool` subcommand runs a paired bootstrap of Claude-vs-baseline across all
   rounds so effective n compounds instead of needing 1000-item rounds.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from experiments.hn_harness import _find_id_at, _get, _ts
from swm.eval.metrics import brier_score, expected_calibration_error, log_loss

THRESHOLDS = [10, 40, 100, 300]
BAND_EDGES = [10, 40, 100, 300]  # -> bands [<10, 10-39, 40-99, 100-299, 300+]
_POOL = ThreadPoolExecutor(10)


def _author_prior(username: str, before_ts: float, cap: int = 22) -> dict:
    """Temporally-clean track record: stats over the author's stories BEFORE before_ts."""
    u = _get(f"user/{username}.json") if username else None
    if not u:
        return {"n_past": 0, "median_past": None, "max_past": None, "frac_ge10_past": None,
                "account_age_days": None}
    subs = (u.get("submitted") or [])[:cap]
    items = [it for it in _POOL.map(lambda i: _get(f"item/{i}.json"), subs) if it]
    past = [it for it in items
            if it.get("type") == "story" and it.get("time", 1e18) < before_ts and "score" in it]
    scores = sorted(s["score"] for s in past)
    created = u.get("created")
    return {
        "n_past": len(scores),
        "median_past": scores[len(scores) // 2] if scores else None,
        "max_past": scores[-1] if scores else None,
        "frac_ge10_past": round(sum(s >= 10 for s in scores) / len(scores), 3) if scores else None,
        "account_age_days": round((before_ts - created) / 86400) if created else None,
    }


def fetch_window(start: str, end: str, n_stories: int, out_prefix: str, seed: int = 0) -> None:
    id0, id1 = _find_id_at(_ts(start)), _find_id_at(_ts(end))
    print(f"ids [{id0}, {id1})")
    rng = random.Random(seed)
    stories, tried = [], 0
    while len(stories) < n_stories and tried < 60 * n_stories:
        batch = [rng.randrange(id0, id1) for _ in range(120)]
        tried += len(batch)
        for it in _POOL.map(lambda i: _get(f"item/{i}.json"), batch):
            if (it and it.get("type") == "story" and not it.get("deleted")
                    and not it.get("dead") and it.get("title") and "score" in it):
                stories.append(it)
                if len(stories) >= n_stories:
                    break
        print(f"  {len(stories)}/{n_stories} after {tried}")
    seen, uniq = set(), []
    for s in stories:
        if s["id"] not in seen:
            seen.add(s["id"]); uniq.append(s)

    inputs, outcomes = [], []
    for n, s in enumerate(uniq):
        dt = datetime.fromtimestamp(s["time"], tz=timezone.utc)
        url = s.get("url", "")
        domain = url.split("/")[2].removeprefix("www.") if "://" in url else ""
        prior = _author_prior(s.get("by", ""), s["time"])
        inputs.append({
            "id": s["id"], "title": s["title"], "author": s.get("by"),
            "domain": domain, "is_text_post": not url,
            "hour_utc": dt.hour, "weekday": dt.weekday(), "date": dt.date().isoformat(),
            "author": s.get("by"), **{f"author_{k}": v for k, v in prior.items()},
        })
        outcomes.append({"id": s["id"], "score": s["score"], "descendants": s.get("descendants", 0)})
        if (n + 1) % 20 == 0:
            print(f"  enriched {n+1}/{len(uniq)}")
    Path(f"{out_prefix}_inputs.json").write_text(json.dumps(inputs, indent=1))
    Path(f"{out_prefix}_outcomes.json").write_text(json.dumps(outcomes, indent=1))
    print(f"wrote {len(inputs)} -> {out_prefix}_inputs.json / _outcomes.json")


# ---------------- baselines ----------------

def _seg_key(x: dict) -> tuple:
    mp = x.get("author_median_past")
    mbucket = "none" if mp is None else "0-3" if mp < 4 else "4-15" if mp < 16 else "16+"
    t = x["title"].lower()
    kind = ("ask" if t.startswith("ask hn") else "show" if t.startswith("show hn")
            else "text" if x["is_text_post"] else "link")
    return (mbucket, kind)


def _segment_rates(train_in, train_out, thr, smooth=6.0):
    glob = sum(1 for x in train_in if train_out[x["id"]]["score"] >= thr) / len(train_in)
    buckets: dict = {}
    for x in train_in:
        buckets.setdefault(_seg_key(x), []).append(1 if train_out[x["id"]]["score"] >= thr else 0)
    return glob, {k: (sum(v) + smooth * glob) / (len(v) + smooth) for k, v in buckets.items()}


# ---------------- scoring ----------------

def _band(score: int) -> int:
    b = 0
    for e in BAND_EDGES:
        if score >= e:
            b += 1
    return b


def _band_probs(pred: dict) -> list[float]:
    t = [pred[f"p_ge_{k}"] for k in THRESHOLDS]
    t = [min(0.999, max(0.001, p)) for p in t]
    for i in range(1, len(t)):        # enforce monotonicity
        t[i] = min(t[i], t[i - 1])
    p = [1 - t[0]]
    for i in range(len(t) - 1):
        p.append(max(1e-4, t[i] - t[i + 1]))
    p.append(t[-1])
    s = sum(p)
    return [x / s for x in p]


def score(pred_path, inputs_path, outcomes_path, train_prefix) -> None:
    preds = {p["id"]: p for p in json.loads(Path(pred_path).read_text())}
    inputs = json.loads(Path(inputs_path).read_text())
    by_id = {x["id"]: x for x in inputs}
    outcomes = {o["id"]: o for o in json.loads(Path(outcomes_path).read_text())}
    tr_in = json.loads(Path(f"{train_prefix}_inputs.json").read_text())
    tr_out = {o["id"]: o for o in json.loads(Path(f"{train_prefix}_outcomes.json").read_text())}
    ids = [x["id"] for x in inputs if x["id"] in preds]
    print(f"scoring {len(ids)} predictions\n")
    print(f"{'thr':<6}{'model':<24}{'logloss':>9}{'brier':>8}{'ece':>7}")
    for thr in THRESHOLDS:
        y = [1 if outcomes[i]["score"] >= thr else 0 for i in ids]
        glob, seg = _segment_rates(tr_in, tr_out, thr)
        rows = {
            "claude": [min(1 - 1e-4, max(1e-4, preds[i][f"p_ge_{thr}"])) for i in ids],
            "base_rate": [glob] * len(ids),
            "segment(author+type)": [seg.get(_seg_key(by_id[i]), glob) for i in ids],
        }
        for name, p in rows.items():
            tag = "  <-- strong" if name.startswith("segment") else ""
            print(f">={thr:<4}{name:<24}{log_loss(y, p):>9.4f}{brier_score(y, p):>8.4f}"
                  f"{expected_calibration_error(y, p):>7.4f}{tag}")
        print(f"      test base rate {sum(y)/len(y):.3f}\n")

    # ordinal band log loss (the distributional / magnitude metric)
    band_true = [_band(outcomes[i]["score"]) for i in ids]
    claude_band = [-math.log(_band_probs(preds[i])[b]) for i, b in zip(ids, band_true)]
    tr_bands = [_band(tr_out[x["id"]]["score"]) for x in tr_in]
    hist = [tr_bands.count(b) / len(tr_bands) for b in range(len(BAND_EDGES) + 1)]
    base_band = [-math.log(max(1e-4, hist[b])) for b in band_true]
    print(f"ORDINAL band log loss (5 bands): claude {sum(claude_band)/len(ids):.4f}  "
          f"base-hist {sum(base_band)/len(ids):.4f}")


def pool(triples: list[tuple[str, str, str]], thr: int = 10, seed: int = 0) -> None:
    """Paired bootstrap of per-item logloss(claude) - logloss(base_rate) across rounds."""
    diffs = []
    for pred_path, inputs_path, outcomes_path in triples:
        preds = {p["id"]: p for p in json.loads(Path(pred_path).read_text())}
        outc = {o["id"]: o for o in json.loads(Path(outcomes_path).read_text())}
        inp = json.loads(Path(inputs_path).read_text())
        ids = [x["id"] for x in inp if x["id"] in preds]
        base = sum(1 for i in ids if outc[i]["score"] >= thr) / len(ids)
        for i in ids:
            y = 1 if outc[i]["score"] >= thr else 0
            pc = min(1 - 1e-4, max(1e-4, preds[i][f"p_ge_{thr}"]))
            ll_c = -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
            ll_b = -(y * math.log(base) + (1 - y) * math.log(1 - base))
            diffs.append(ll_c - ll_b)
    n = len(diffs)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    worse = sum(1 for _ in range(2000)
                if sum(diffs[rng.randrange(n)] for _ in range(n)) / n >= 0)
    print(f"pooled n={n}  mean logloss diff (claude - base) = {mean:+.4f} "
          f"(negative = claude better)")
    print(f"paired bootstrap P(claude NOT better) = {worse/2000:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    for a in ("start", "end", "out"):
        f.add_argument(f"--{a}", required=True)
    f.add_argument("--n", type=int, default=110)
    f.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    for a in ("pred", "inputs", "outcomes", "train"):
        s.add_argument(f"--{a}", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--triples", nargs="+", required=True,
                   help="pred,inputs,outcomes  pred,inputs,outcomes ...")
    p.add_argument("--thr", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch_window(a.start, a.end, a.n, a.out, a.seed)
    elif a.cmd == "score":
        score(a.pred, a.inputs, a.outcomes, a.train)
    else:
        pool([tuple(t.split(",")) for t in a.triples], a.thr)


if __name__ == "__main__":
    main()
