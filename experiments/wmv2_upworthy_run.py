"""Upworthy heterogeneous-population benchmark — Reference World E, randomized ground truth (item 4).

Task: pick the CTR winner among 2-4 randomized headline variants (precision@1 + pairwise accuracy) on
held-out tests. Splits: immutable shuffle(seed 13) of eligible tests → train (fit the CTR layer) / test.

Arms (identical test set):
  U0 random ranking          U1 fitted surface-features ranker (statistical baseline; no LLM)
  U2 grounded direct LLM (one-shot winner pick)     U3 TRUE 3-call ensemble (majority, temp 0.7)
  U4 V2 MAX-CAPACITY population world: universal interpretation of each headline (typed dims) + surface
     features → TRAIN-fitted CTR layer → heterogeneous audience particles (perturbed fitted weights,
     argmax-choosing members) → winner by simulated preference share
Ablations: U4_no_population (point fitted scalar over the same dims) | U4_no_interp (surface only ≡ U1
population variant) | U4_no_fit (unfitted equal weights).

Metrics: precision@1, pairwise accuracy, paired bootstrap CIs over tests; cost + latency.
Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_upworthy_run
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_upworthy_v2.json"
DS_IN, DS_OUT = 0.27e-6, 1.10e-6


def paired_p1_ci(hits_a, hits_b, n_boot=1000, seed=5):
    d = [a - b for a, b in zip(hits_a, hits_b)]
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n_boot))
    return {"mean": round(sum(d) / len(d), 4), "ci95": [round(bs[25], 4), round(bs[-26], 4)], "n": len(d)}


def run(n_train, n_test, particles):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.grounding import parse_json
    from swm.eval.response_datasets import download_upworthy, load_upworthy_tests
    from swm.world_model_v2.actor_cognition import interpret
    from swm.world_model_v2.reference.upworthy import (fit_ctr_layer, population_rank,
                                                       surface_features, zscores)
    t0 = time.time()
    meter = {"calls": 0, "tokens": 0}
    llm = default_chat_fn(system="You are a careful media analyst. Reply ONLY compact JSON.",
                          max_tokens=160, temperature=0.3)
    llm_hot = default_chat_fn(system="You are a careful media analyst. Reply ONLY compact JSON.",
                              max_tokens=160, temperature=0.7)
    if llm is None:
        raise SystemExit("needs DEEPSEEK_API_KEY")

    def call(fn, prompt):
        txt = fn(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
        return txt

    tests = load_upworthy_tests(download_upworthy(), min_impressions=4000)
    rng = random.Random(13)
    rng.shuffle(tests)
    train, test = tests[:n_train], tests[n_train:n_train + n_test]
    print(f"tests: eligible={len(tests)} train={len(train)} test={len(test)}", flush=True)

    _imemo = {}

    def interp_dims(headline):
        if headline not in _imemo:
            it = interpret(lambda p: call(llm, p),
                           actor="a person scrolling a social feed",
                           channel="social-media headline",
                           context="- you click only what genuinely pulls you in",
                           content=headline)
            _imemo[headline] = it.features() if it else [0.5] * 18
        return _imemo[headline]

    # ---- fit CTR layers on TRAIN (within-test z-scored CTR) ----
    S_full, S_surf = [], []
    for i, t in enumerate(train):
        zs = zscores([v["ctr"] for v in t["variants"]])
        for v, z in zip(t["variants"], zs):
            sf = surface_features(v["headline"])
            S_surf.append((sf, z))
            S_full.append((sf + interp_dims(v["headline"]), z))
        if i % 20 == 0:
            print(f"  [fit] {i}/{len(train)} calls={meter['calls']}", flush=True)
    pred_full, coef_full = fit_ctr_layer(S_full)
    pred_surf, coef_surf = fit_ctr_layer(S_surf)
    k_full = len(S_full[0][0])
    print(f"fitted layers: surface_w={coef_surf['w']}", flush=True)

    # ---- evaluate on identical held-out tests ----
    hits = {a: [] for a in ("U0", "U1", "U2", "U3", "U4", "U4_no_population", "U4_no_interp",
                            "U4_no_fit")}
    pairs_ok = {a: [0, 0] for a in hits}
    lat = {"U2_s": 0.0, "U3_s": 0.0, "U4_s": 0.0}
    rng_eval = random.Random(99)
    for i, t in enumerate(test):
        heads = [v["headline"] for v in t["variants"]]
        ctr = {v["headline"]: v["ctr"] for v in t["variants"]}
        winner = t["winner_headline"]

        def score_rank(order, arm):
            hits[arm].append(1 if order and order[0] == winner else 0)
            for x in range(len(order)):
                for y in range(x + 1, len(order)):
                    if ctr[order[x]] == ctr[order[y]]:
                        continue
                    pairs_ok[arm][1] += 1
                    pairs_ok[arm][0] += ctr[order[x]] > ctr[order[y]]

        sh = list(heads)
        rng_eval.shuffle(sh)
        score_rank(sh, "U0")
        score_rank(sorted(heads, key=lambda h: -pred_surf(surface_features(h))), "U1")
        # U2 direct one-shot
        tt = time.time()
        numbered = "\n".join(f"{j + 1}. {h}" for j, h in enumerate(heads))
        r2 = parse_json(call(llm, f"These headline variants ran in a randomized A/B test on the same "
                                  f"story to the same audience. Which got the highest click-through "
                                  f"rate?\n{numbered}\nReturn ONLY JSON: "
                                  f'{{"ranking": [<best index>, ...]}} using 1-based indices.')) or {}
        try:
            order2 = [heads[int(j) - 1] for j in r2.get("ranking", []) if 0 < int(j) <= len(heads)]
        except (TypeError, ValueError):
            order2 = []
        if len(set(order2)) != len(heads):
            order2 = sh
        score_rank(order2, "U2")
        lat["U2_s"] += time.time() - tt
        # U3 true 3-call ensemble (majority top pick, temp 0.7)
        tt = time.time()
        votes = {}
        for _ in range(3):
            r3 = parse_json(call(llm_hot, f"Randomized A/B test, same story, same audience. Which "
                                          f"headline got the highest CTR?\n{numbered}\n"
                                          f'Return ONLY JSON: {{"best": <1-based index>}}')) or {}
            try:
                votes[int(r3.get("best"))] = votes.get(int(r3.get("best")), 0) + 1
            except (TypeError, ValueError):
                continue
        if votes:
            best3 = heads[max(votes, key=votes.get) - 1] if 0 < max(votes, key=votes.get) <= len(heads) \
                else sh[0]
            order3 = [best3] + [h for h in sh if h != best3]
        else:
            order3 = sh
        score_rank(order3, "U3")
        lat["U3_s"] += time.time() - tt
        # U4 population world (+ ablations) over fitted dims
        tt = time.time()
        feats = [surface_features(h) + interp_dims(h) for h in heads]
        w_full = coef_full["w"]
        sc = population_rank(feats, w_full, n_particles=particles, seed=i, heterogeneity=True)
        score_rank([h for _, h in sorted(zip(sc, heads), key=lambda z: -z[0])], "U4")
        sc_np = population_rank(feats, w_full, heterogeneity=False)
        score_rank([h for _, h in sorted(zip(sc_np, heads), key=lambda z: -z[0])], "U4_no_population")
        feats_s = [surface_features(h) for h in heads]
        sc_ni = population_rank(feats_s, coef_surf["w"], n_particles=particles, seed=i,
                                heterogeneity=True)
        score_rank([h for _, h in sorted(zip(sc_ni, heads), key=lambda z: -z[0])], "U4_no_interp")
        sc_nf = population_rank(feats, [1.0] * k_full, n_particles=particles, seed=i,
                                heterogeneity=True)
        score_rank([h for _, h in sorted(zip(sc_nf, heads), key=lambda z: -z[0])], "U4_no_fit")
        lat["U4_s"] += time.time() - tt
        if i % 20 == 0:
            print(f"  [eval] {i}/{len(test)} calls={meter['calls']}", flush=True)

    detail = {a: {"precision_at_1": round(sum(h) / len(h), 3),
                  "pairwise_accuracy": round(pairs_ok[a][0] / max(1, pairs_ok[a][1]), 3),
                  "n": len(h)} for a, h in hits.items() if h}
    paired = {"U4_vs_U1": paired_p1_ci(hits["U4"], hits["U1"]),
              "U4_vs_U2": paired_p1_ci(hits["U4"], hits["U2"]),
              "U4_vs_U3": paired_p1_ci(hits["U4"], hits["U3"]),
              "U4_vs_no_population": paired_p1_ci(hits["U4"], hits["U4_no_population"]),
              "U4_vs_no_interp": paired_p1_ci(hits["U4"], hits["U4_no_interp"]),
              "U4_vs_no_fit": paired_p1_ci(hits["U4"], hits["U4_no_fit"]),
              "U2_vs_U1": paired_p1_ci(hits["U2"], hits["U1"])}
    out = {"n_train": len(train), "n_test": len(test), "detail": detail, "paired": paired,
           "fitted": {"surface_w": coef_surf["w"]},
           "random_p1": round(sum(1 / len(t["variants"]) for t in test) / len(test), 3),
           "_meta": {"llm_calls": meter["calls"], "llm_tokens_est": meter["tokens"],
                     "est_cost_usd": round(meter["tokens"] * (DS_IN + DS_OUT) / 2, 4),
                     "model_name": "deepseek-chat (DeepSeek V3)",
                     "runtime_s": round(time.time() - t0, 1),
                     "latency_s": {k: round(v, 1) for k, v in lat.items()},
                     "license_note": "Upworthy Research Archive CC-BY"}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    for a, m in detail.items():
        print(f"  {a:17s} p@1={m['precision_at_1']} pairwise={m['pairwise_accuracy']}")
    for k, v in paired.items():
        print(f"  {k}: Δ={v['mean']:+.4f} CI{v['ci95']}")
    print(f"wrote {RESULT} (calls={meter['calls']}, ~${out['_meta']['est_cost_usd']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--particles", type=int, default=300)
    a = ap.parse_args()
    run(a.n_train, a.n_test, a.particles)
