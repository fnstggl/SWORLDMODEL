"""EXP-008/010 aggregate harness: backtest AggregateWorld on real HN, no-cheat, reproducible.

Fetches a time-ordered stream of real HN stories (author, domain, topic, timestamp, score),
builds (Action, magnitude) samples, and runs:
  (A) AggregateWorld.backtest        -> state-transition vs content-only vs base rate, temporal split
  (B) decision_lift                   -> hit-capture of ranking by the model vs base-rate ranking
  (C) calibration_by_horizon          -> free-running vs teacher-forced multi-step (the honest
                                         multi-step eval the prior repo never ran)

Writes a committed result artifact to experiments/results/ so the numbers in the report are
reproducible from the repo (raw HN data stays gitignored; the fetched sample is cached in data/).

Usage:
  python -m experiments.aggregate_harness fetch --n 2500 --start 2026-02-01 --end 2026-05-01
  python -m experiments.aggregate_harness run
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from experiments.hn_harness import _find_id_at, _get, _ts
from swm.eval.decision_lift import decision_lift
from swm.simulation.rollout import calibration_by_horizon
from swm.state.factors import tag_topic
from swm.state.state import Action
from swm.transition.aggregate_transition import AggregateTransition
from swm.worlds.aggregate_world import AggregateWorld

SAMPLE_PATH = "data/hn_agg_stream.json"
RESULT_PATH = "experiments/results/exp010_aggregate_hn.json"
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(12)
    return _POOL


def fetch(n: int, start: str, end: str, seed: int = 0) -> None:
    id0, id1 = _find_id_at(_ts(start)), _find_id_at(_ts(end))
    print(f"scanning id range [{id0}, {id1}) for {n} stories")
    rng = random.Random(seed)
    stories, tried = [], 0
    seen = set()
    while len(stories) < n and tried < 80 * n:
        batch = [rng.randrange(id0, id1) for _ in range(200)]
        tried += len(batch)
        for it in _pool().map(lambda i: _get(f"item/{i}.json"), batch):
            if (it and it.get("type") == "story" and not it.get("deleted") and not it.get("dead")
                    and it.get("title") and "score" in it and it.get("by")
                    and it["id"] not in seen):
                seen.add(it["id"])
                url = it.get("url", "")
                domain = url.split("/")[2].removeprefix("www.") if "://" in url else ""
                stories.append({"id": it["id"], "ts": it["time"], "score": it["score"],
                                "title": it["title"], "author": it["by"], "domain": domain,
                                "is_text": not url})
                if len(stories) >= n:
                    break
        if tried % 2000 < 200:
            print(f"  {len(stories)}/{n} kept after {tried} tried")
    stories.sort(key=lambda s: s["ts"])
    Path("data").mkdir(exist_ok=True)
    Path(SAMPLE_PATH).write_text(json.dumps(stories))
    print(f"wrote {len(stories)} time-ordered stories -> {SAMPLE_PATH}")


def _action(s: dict, i: int) -> Action:
    t = s["title"].lower()
    return Action(
        action_id=f"{s['author']}-{s['id']}", actor_id=s["author"],
        content_features={"title_len": min(1.0, len(s["title"]) / 80),
                          "is_show": 1.0 if t.startswith("show hn") else 0.0,
                          "is_ask": 1.0 if t.startswith("ask hn") else 0.0,
                          "is_text": 1.0 if s["is_text"] else 0.0,
                          "topic": tag_topic(s["title"])},
        timing={"hour": datetime.fromtimestamp(s["ts"], tz=timezone.utc).hour,
                "weekday": datetime.fromtimestamp(s["ts"], tz=timezone.utc).weekday(),
                "ts": s["ts"]},
        meta={"domain": s["domain"], "title": s["title"]})


def run() -> None:
    stories = json.loads(Path(SAMPLE_PATH).read_text())
    samples = [(_action(s, i), float(s["score"])) for i, s in enumerate(stories)]
    n = len(samples)
    thr = 40
    base = sum(1 for _, m in samples if m >= thr) / n
    print(f"\n{n} HN stories; P(score>={thr}) = {base:.3f}\n")

    # (A) state-transition vs content-only vs base rate
    aw = AggregateWorld(domain="hn", target_threshold=thr)
    bt = aw.backtest(samples)
    print("== (A) aggregate state-transition backtest (temporal split) ==")
    print(f"  base rate         log loss {bt['base_rate']['log_loss']}")
    print(f"  content-only      log loss {bt['content_only']['log_loss']}  ece {bt['content_only']['ece']}")
    print(f"  STATE-transition  log loss {bt['state_transition']['log_loss']}  ece {bt['state_transition']['ece']}"
          f"  uplift@20 {bt['state_transition']['uplift@20']}")
    print(f"  state_helps_logloss {bt['state_helps_logloss']}  grade {bt['grade']['grade']}  -> {bt['verdict']}")

    # (B) decision lift: rank test items by model P(hit) vs by base rate (=random within test)
    cut = int(0.7 * n)
    test = samples[cut:]
    fitted = AggregateWorld(domain="hn", target_threshold=thr).fit_stream(samples[:cut])
    # as-of test preds (carry state forward)
    tr = fitted.transition
    pop = fitted.pop
    model_scores, y = [], []
    for action, mag in test:
        model_scores.append(tr.predict(pop, action)["thresholds"].get(thr, 0.0))
        y.append(1 if mag >= thr else 0)
        tr.transition(pop, action, mag)
    dl = decision_lift(y, model_scores, [0.5] * len(y), target_k=0.2)
    print("\n== (B) decision lift (hit-capture, model ranking vs random) ==")
    for row in dl["curve"]:
        print(f"  top {int(row['k']*100):>2}%  model {row['model']:.3f}  random {row['random']:.3f}  oracle {row['oracle']:.3f}")
    print(f"  lift@20% over random {dl['lift_over_baseline_at_target_k']:+.3f}")

    # (C) multi-step: per-author held-out sequences, free-running vs teacher-forced
    from collections import defaultdict
    by_author = defaultdict(list)
    for action, mag in samples:
        by_author[action.actor_id].append((action, mag))
    seqs = [v for v in by_author.values() if len(v) >= 4][:120]
    def build_tr():
        w = AggregateWorld(domain="hn", target_threshold=thr).fit_stream(samples[:cut])
        return w.transition
    ch = calibration_by_horizon(build_tr, seqs, target_threshold=thr, n_samples=30)
    print("\n== (C) multi-step calibration by horizon (per-author) ==")
    print("  teacher-forced:", [(r["horizon"], r["log_loss"], r["ece"]) for r in ch["teacher_forced"]])
    print("  free-running:  ", [(r["horizon"], r["log_loss"], r["ece"]) for r in ch["free_running"]])

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT_PATH).write_text(json.dumps({
        "n": n, "base_rate": round(base, 4), "target": f"P(score>={thr})",
        "backtest": bt, "decision_lift": dl, "calibration_by_horizon": ch,
        "n_authors_multistep": len(seqs),
    }, indent=1))
    print(f"\nwrote {RESULT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--n", type=int, default=2500)
    f.add_argument("--start", default="2026-02-01")
    f.add_argument("--end", default="2026-05-01")
    f.add_argument("--seed", type=int, default=0)
    sub.add_parser("run")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.n, a.start, a.end, a.seed)
    else:
        run()


if __name__ == "__main__":
    main()
