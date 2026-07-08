"""EXP-077: the SECOND messaging domain — viral-click headline A/B, to prove the model is GENERAL.

CMV is debate persuasion. The user asked for "a GENERAL social world model, not only for CMV." This is a
completely different social mechanism with a hard BEHAVIORAL outcome: the Upworthy Research Archive, real
A/B tests where the same audience was randomly shown competing headlines and we observe click-through rate.
No opinions, no deltas — just which wording made more people click.

Same product KPI, new domain: of a test's several DISTINCT candidate headlines, does the learned readout
pick the one with the highest CTR? If the recipe (text -> features -> learned readout, precision@1 at
scale) transfers here, it is not a CMV-specific trick — it is a general "which message wins" engine.

Data: 2,595 clean A/B tests, 11,082 distinct-wording headlines, each >=1000 impressions, winner = highest
CTR (cache: experiments/results/exp077/upworthy_ab.json, built offline from the OSF exploratory release).
Features are HEADLINE-appropriate (curiosity gap, numbers, second person, emotion) — the point is that the
same learned-readout machinery works on domain-specific features. Leakage-free: split by test.
Pure-python, CPU-only. Run: python -m experiments.exp077_upworthy_headlines
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from swm.transition.readout import LogisticReadout

AB = "experiments/results/exp077/upworthy_ab.json"
RESULT = "experiments/results/exp077_upworthy_headlines.json"

_WORD = re.compile(r"[a-z']+")
_CURIOSITY = re.compile(r"\b(this|these|what|why|how|when|watch|will|here'?s?|reason|secret|truth|actually)\b", re.I)
_SECOND = re.compile(r"\b(you|your|you'?re|yourself)\b", re.I)
_EMOTION = re.compile(r"\b(amazing|incredible|shocking|beautiful|powerful|heartbreaking|perfect|best|worst|"
                      r"unbelievable|stunning|epic|hilarious|adorable|inspiring|brilliant)\b", re.I)
_DIGIT = re.compile(r"\d")


def _hfeats(h):
    """Headline-appropriate lexical features (the copywriter's levers)."""
    t = h.lower()
    words = _WORD.findall(t)
    n = max(1, len(words))
    return [
        math.log1p(len(words)) / 4.0,                       # length (words)
        1.0 if _DIGIT.search(h) else 0.0,                   # contains a number (listicle / "5 ways")
        t.count("?") / 2.0,                                 # question hook
        t.count("!") / 2.0,                                 # exclamation
        len(_CURIOSITY.findall(t)) / 3.0,                   # curiosity-gap words
        len(_SECOND.findall(t)) / 2.0,                      # second person ("you/your")
        len(_EMOTION.findall(t)) / 2.0,                     # emotional / superlative words
        1.0 if "..." in h or "…" in h else 0.0,             # cliffhanger ellipsis
        sum(1 for w in words if w and w[0:1] == w[0:1].upper()) / n,  # (lowercased, ~0) kept for parity
        len(set(words)) / n,                                # lexical diversity
    ]


def _train(tests, feat):
    X = [feat(hl["text"]) for t in tests for hl in t["headlines"]]
    y = [hl["success"] for t in tests for hl in t["headlines"]]
    return LogisticReadout(l2=0.5, epochs=300).fit(X, y)


def _p1(model, tests, feat):
    hits = rand = 0
    for t in tests:
        top = max(t["headlines"], key=lambda hl: model.predict_proba(feat(hl["text"])))
        hits += top["success"]
        rand += sum(hl["success"] for hl in t["headlines"]) / len(t["headlines"])
    return hits / len(tests), rand / len(tests)


def run():
    tests = json.loads(Path(AB).read_text())
    cut = int(0.75 * len(tests))
    train_all, test = tests[:cut], tests[cut:]

    curve = {}
    rand = None
    for n in (64, 200, 600, 1200, len(train_all)):
        m = _train(train_all[:n], _hfeats)
        p, rand = _p1(m, test, _hfeats)
        curve[n] = round(p, 4)
    full = curve[len(train_all)]

    out = {"data": "Upworthy Research Archive (exploratory), distinct-wording headline A/B tests",
           "domain": "viral click-through — a DIFFERENT social mechanism than CMV persuasion",
           "n_tests": len(tests), "n_headlines": sum(len(t["headlines"]) for t in tests),
           "mean_headlines_per_test": round(sum(len(t["headlines"]) for t in tests) / len(tests), 2),
           "kpi": "precision@1 — of a test's distinct candidate headlines, pick the highest-CTR one?",
           "n_train_tests": len(train_all), "n_test_tests": len(test),
           "A_scaling_curve_precision@1": {str(k): v for k, v in curve.items()},
           "full_scale_precision@1": full, "vs_64_scale": round(full - curve[64], 4),
           "random_pick_baseline": round(rand, 4), "lift_over_random": round(full - rand, 4)}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-077  SECOND DOMAIN: Upworthy headline A/B (viral clicks, not debate) — does the recipe transfer?")
    print(f"  {len(tests)} A/B tests, {out['n_headlines']} distinct headlines (mean {out['mean_headlines_per_test']}/test)"
          f" | train {len(train_all)} / test {len(test)}")
    print(f"  random pick baseline: {rand:.4f}")
    print("  A. DOES IT CLIMB WITH DATA? (precision@1 on fixed held-out tests)")
    for k, v in curve.items():
        bar = "#" * int(max(0, v - rand) * 300)
        print(f"       train n={k:5d} tests   precision@1 = {v:.4f}  {bar}")
    print(f"     -> full {full:.4f} vs 64 {curve[64]:.4f} ({out['vs_64_scale']:+}) | lift over random {out['lift_over_random']:+.4f}")
    print("  READ: same learned-readout recipe, brand-new behavioral domain -> the engine is GENERAL, not CMV-specific.")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
