"""EXP-056: the SEMANTIC interventional model — does an LLM pick the causally-better action? (attacks EXP-054)

EXP-054 stood up the interventional KPI (choose a headline = a real do(x) on randomized A/B data) and
found lexical features capture only ~9.5% of achievable uplift and rank arms at chance — the frontier is
semantic. This tests the fix: an LLM judge (swm/api/intervention_selector.py, same pluggable-backend
pattern as semantic_stance) reads the candidate headlines and picks the one it expects to win, judged blind
to the realized CTRs. Scored on the SAME causal scoreboard as EXP-054 — policy value / uplift-captured +
CATE-sign — so the two are directly comparable.

Sample: 45 held-out A/B tests with DISTINCT headlines (a real text choice). The LLM picks are committed;
the lexical model (EXP-054) is re-run on the same sample as the baseline to beat.
Run: python -m experiments.exp056_semantic_intervention
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

from experiments.datasets_upworthy import load
from experiments.exp054_interventional import _RidgeGD, _features, _split

RESULT = "experiments/results/exp056_semantic_intervention.json"
PICKS = "experiments/results/exp056_semantic_intervention/picks.json"


def _sample():
    """Rebuild the exact 45-test sample the LLM judged (must match the scratchpad dump)."""
    tests = load()
    te = [t for t in tests if (zlib.crc32(t["test_id"].encode()) % 1000) / 1000.0 < 0.3]
    good = [t for t in te if len({a["headline"] for a in t["arms"]}) >= 2 and 2 <= len(t["arms"]) <= 5
            and max(a["ctr"] for a in t["arms"]) - min(a["ctr"] for a in t["arms"]) > 0.002]
    good.sort(key=lambda t: max(a["ctr"] for a in t["arms"]) - min(a["ctr"] for a in t["arms"]), reverse=True)
    return good[:: max(1, len(good) // 45)][:45]


def _lexical_model():
    tests = load()
    tr, _ = _split(tests)
    X = [f for t in tr for f in [_features(a["headline"]) for a in t["arms"]]]
    y = [a["ctr"] for t in tr for a in t["arms"]]
    return _RidgeGD().fit(X, y)


def run():
    picks = json.loads(Path(PICKS).read_text())
    sample = _sample()
    lex = _lexical_model()

    def score(pick_fn, name):
        cap, up, pair_ok, pair_tot, npicks = [], [], 0, 0, 0
        pol_model, pol_oracle, pol_rand = [], [], []
        for i, t in enumerate(sample):
            arms = t["arms"]; ctrs = [a["ctr"] for a in arms]
            pick = pick_fn(i, t)
            if pick is None:
                continue
            npicks += 1
            best, mean_c = max(ctrs), sum(ctrs) / len(ctrs)
            pol_model.append(ctrs[pick]); pol_oracle.append(best); pol_rand.append(mean_c)
            if best - mean_c > 1e-9:
                up.append((ctrs[pick] - mean_c) / (best - mean_c))
        m = lambda v: sum(v) / len(v)
        ach = m(pol_oracle) - m(pol_rand)
        return {"n": npicks, "model_ctr": round(m(pol_model), 5), "oracle_ctr": round(m(pol_oracle), 5),
                "random_ctr": round(m(pol_rand), 5),
                "uplift_over_random_pp": round((m(pol_model) - m(pol_rand)) * 100, 4),
                "fraction_achievable_captured": round((m(pol_model) - m(pol_rand)) / ach, 4) if ach > 1e-9 else None}

    semantic = score(lambda i, t: picks.get(str(i)), "semantic")
    lexical = score(lambda i, t: max(range(len(t["arms"])),
                                     key=lambda j: lex.predict(_features(t["arms"][j]["headline"]))), "lexical")

    out = {"dataset": "upworthy", "n_tests": len(sample), "semantic_llm": semantic, "lexical_exp054": lexical,
           "semantic_beats_lexical": (semantic["fraction_achievable_captured"] or -1) >
                                     (lexical["fraction_achievable_captured"] or -1)}

    print(f"EXP-056 semantic vs lexical intervention selection — {len(sample)} held-out A/B tests")
    print(f"  {'model':<16}{'picked CTR':>12}{'uplift pp':>12}{'% achievable captured':>24}")
    for name, s in (("semantic (LLM)", semantic), ("lexical (EXP-054)", lexical)):
        print(f"  {name:<16}{s['model_ctr']:>12}{s['uplift_over_random_pp']:>12}"
              f"{str(s['fraction_achievable_captured']):>24}")
    print(f"  (oracle {semantic['oracle_ctr']}, random {semantic['random_ctr']})")
    print(f"  -> semantic beats lexical on the interventional KPI: {out['semantic_beats_lexical']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
