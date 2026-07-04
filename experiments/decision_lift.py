"""EXP-004: decision lift, not just calibration (audit E.2, L.2).

Calibration answers "is the probability honest?" A buyer asks a different question: "if I act on
this model instead of my current method, do I capture more of the outcome?" This measures that,
on the same contamination-free HN predictions, framed as the real operator decision:

  LIMITED ATTENTION. You can only act on the top-K% of candidates (review, feature, send, spend).
  Ranking them by the model, what fraction of the actual WINNERS (score >= 40, the tail where our
  edge lives) do you capture -- vs random, vs the strong author-aware baseline, vs a perfect oracle?

This is uplift@k. It is the product's value proposition made falsifiable.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from experiments.hn_harness2 import _seg_key, _segment_rates

ROUNDS = [  # rounds with author priors + segment baseline available
    ("data/round3_predictions.json", "data/hn_r3_inputs.json", "data/hn_r3_outcomes.json"),
    ("data/round4_predictions.json", "data/hn_r4_inputs.json", "data/hn_r4_outcomes.json"),
]
TRAIN = "data/hn_train2"
HIT = 40           # "real hit" threshold -- the tail where the model's edge is significant
KS = [0.05, 0.10, 0.20, 0.30]


def _load():
    tr_in = json.loads(Path(f"{TRAIN}_inputs.json").read_text())
    tr_out = {o["id"]: o for o in json.loads(Path(f"{TRAIN}_outcomes.json").read_text())}
    _, seg = _segment_rates(tr_in, tr_out, HIT)
    glob = sum(1 for x in tr_in if tr_out[x["id"]]["score"] >= HIT) / len(tr_in)
    items = []
    for pp, ip, op in ROUNDS:
        P = {p["id"]: p for p in json.loads(Path(pp).read_text())}
        I = {x["id"]: x for x in json.loads(Path(ip).read_text())}
        O = {o["id"]: o for o in json.loads(Path(op).read_text())}
        for i in P:
            if i in O and i in I:
                items.append({
                    "id": i, "score": O[i]["score"], "hit": O[i]["score"] >= HIT,
                    "claude": P[i][f"p_ge_{HIT}"],
                    "segment": seg.get(_seg_key(I[i]), glob),
                })
    return items


def _capture(items, key, k, rng=None):
    """Fraction of actual hits (and of total score) captured in the top-k fraction by `key`."""
    n = max(1, round(k * len(items)))
    if key == "random":
        order = list(range(len(items))); rng.shuffle(order); top = [items[j] for j in order[:n]]
    else:
        top = sorted(items, key=lambda x: x[key], reverse=True)[:n]
    total_hits = sum(x["hit"] for x in items) or 1
    total_score = sum(x["score"] for x in items) or 1
    return sum(x["hit"] for x in top) / total_hits, sum(x["score"] for x in top) / total_score


def main() -> None:
    items = _load()
    n, hits = len(items), sum(x["hit"] for x in items)
    print(f"n={n} candidates, {hits} actual hits (>= {HIT} pts, {hits/n:.1%})\n")
    print(f"{'act on top':<12}{'model':<20}{'hit-capture':>12}{'engagement':>12}")
    rng = random.Random(0)
    rand_curve = {}
    for k in KS:
        # random baseline: average over 500 shuffles
        rs = [_capture(items, "random", k, rng) for _ in range(500)]
        rand_curve[k] = (sum(a for a, _ in rs) / len(rs), sum(b for _, b in rs) / len(rs))
    for k in KS:
        for model in ("claude", "segment"):
            hc, ec = _capture(items, model, k)
            print(f"{f'{int(k*100)}%':<12}{model:<20}{hc:>11.1%}{ec:>12.1%}")
        rhc, rec = rand_curve[k]
        print(f"{f'{int(k*100)}%':<12}{'random':<20}{rhc:>11.1%}{rec:>12.1%}")
        # oracle
        ohc, oec = _capture(items, "score", k)
        print(f"{f'{int(k*100)}%':<12}{'oracle(perfect)':<20}{ohc:>11.1%}{oec:>12.1%}")
        print()

    # headline lift + bootstrap CI at K=20%
    k = 0.20
    claude_hc, _ = _capture(items, "claude", k)
    seg_hc, _ = _capture(items, "segment", k)
    rand_hc = rand_curve[k][0]
    print(f"HEADLINE @ act-on-top-20%:")
    print(f"  model captures {claude_hc:.0%} of hits vs random {rand_hc:.0%} vs "
          f"author-aware baseline {seg_hc:.0%}")
    print(f"  lift over random: {claude_hc-rand_hc:+.0%}   over baseline: {claude_hc-seg_hc:+.0%}")
    # bootstrap CI on lift-over-baseline
    boot = []
    for _ in range(2000):
        samp = [items[rng.randrange(n)] for _ in range(n)]
        c, _ = _capture(samp, "claude", k); s, _ = _capture(samp, "segment", k)
        boot.append(c - s)
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    p_le0 = sum(1 for d in boot if d <= 0) / len(boot)
    print(f"  lift-over-baseline 95% CI [{lo:+.0%}, {hi:+.0%}]   P(no lift) = {p_le0:.3f}")


if __name__ == "__main__":
    main()
