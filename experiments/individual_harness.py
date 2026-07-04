"""EXP-011 individual harness: validate the individual response ESTIMATOR (spec Phase 4).

There is no real labeled individual-response dataset in this repo (private email/CRM logs are the
scarce asset; `data/` is gitignored and empty of behavior). So this harness does the one honest
thing available: it validates the *estimator* on SYNTHETIC data whose generative structure is
known, answering "does hierarchical partial pooling actually recover the individual and beat the
alternatives — and where?" A positive result here proves the machinery is correct; it does NOT
prove a real-world individual claim, which stays BLOCKED-ON-PRIVATE-DATA (see report).

Generative model (known ground truth):
  - each contact i has a latent reply rate theta_i ~ Beta(2,6) (population mean ~0.25)
  - contacts are seen a variable number of times (Zipf-ish), so evidence is uneven — the regime
    where pooling matters most
  - each message has features; a true message effect nudges the odds (content adds signal)
  - outcome ~ Bernoulli(sigmoid(logit(theta_i) + message_effect))

Arms compared on a temporal holdout, same rows:
  segment            : population/segment rate only
  no_pooling         : per-person empirical rate, NO shrinkage (overfits low-evidence contacts)
  partial_pooling    : hierarchical shrinkage toward segment  (THE model)
  +message           : partial pooling + message features
  raw_llm / +context : BLOCKED (needs an LLM/agent predictor over the same rows)

Writes experiments/results/exp011_individual_synth.json.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss
from swm.transition.individual_transition import IndividualTransition

RESULT_PATH = "experiments/results/exp011_individual_synth.json"
_SIG = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
_LOGIT = lambda p: math.log(max(1e-6, p) / max(1e-6, 1 - p))  # noqa: E731


def gen(n_contacts: int = 300, n_events: int = 6000, seed: int = 0):
    rng = random.Random(seed)
    theta = {f"c{i}": min(0.97, max(0.03, rng.betavariate(2, 6))) for i in range(n_contacts)}
    # Zipf-ish contact frequency: a few frequent, a long tail of one-offs
    weights = [1.0 / (i + 1) for i in range(n_contacts)]
    ids = list(theta)
    samples = []
    for _ in range(n_events):
        cid = rng.choices(ids, weights=weights)[0]
        log_words = rng.uniform(2.0, 5.0)
        has_cta = float(rng.random() < 0.5)
        # true message effect: short + CTA helps
        eff = 0.5 * has_cta - 0.35 * (log_words - 3.5)
        p = _SIG(_LOGIT(theta[cid]) + eff)
        samples.append((cid, {"log_words": log_words, "has_cta": has_cta}, int(rng.random() < p)))
    return samples, theta


def _score(y, p):
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4)}


def _run_arm(train, test, sources, seg_rate, prior_strength, mfn):
    m = IndividualTransition(message_feature_names=mfn, segment_rate=seg_rate,
                             prior_strength=prior_strength, sources=sources)
    m.fit_stream(train, segment_rate=seg_rate)
    preds, y, ncat = [], [], []
    for cid, mf, o in test:
        p = m.person(cid)
        preds.append(m.predict(cid, mf)["p_mean"])
        y.append(o)
        ncat.append("cold(<3)" if p.n_obs < 3 else "warm(3-15)" if p.n_obs < 15 else "hot(15+)")
        m.transition(cid, o)
    return preds, y, ncat


def run():
    samples, theta = gen()
    n = len(samples)
    cut = int(0.7 * n)
    train, test = samples[:cut], samples[cut:]
    y = [o for _, _, o in test]
    seg_rate = (sum(o for _, _, o in train) + 1) / (len(train) + 2)
    mfn = ["log_words", "has_cta"]
    print(f"{n} synthetic events, {len(theta)} contacts; test base rate {sum(y)/len(y):.3f}, "
          f"segment rate {seg_rate:.3f}\n")

    arms = {
        "segment": (frozenset({"segment"}), 4.0),
        "no_pooling": (frozenset({"segment", "person"}), 0.01),
        "partial_pooling": (frozenset({"segment", "person"}), 4.0),
        "+message": (frozenset({"segment", "person", "message"}), 4.0),
    }
    results, preds_by_arm = {}, {}
    for name, (sources, ps) in arms.items():
        preds, yy, ncat = _run_arm(train, test, sources, seg_rate, ps, mfn)
        results[name] = _score(yy, preds)
        preds_by_arm[name] = (preds, ncat)
        print(f"  {name:<16} log loss {results[name]['log_loss']:.4f}  brier {results[name]['brier']:.4f}"
              f"  ece {results[name]['ece']:.4f}")

    # where does pooling win? break log loss down by evidence bucket (partial vs no-pooling vs segment)
    print("\n  log loss by contact-evidence bucket (partial_pooling vs no_pooling vs segment):")
    buckets = {}
    p_pp, cats = preds_by_arm["partial_pooling"]
    p_np, _ = preds_by_arm["no_pooling"]
    p_sg, _ = preds_by_arm["segment"]
    for cat in ("cold(<3)", "warm(3-15)", "hot(15+)"):
        idx = [i for i, c in enumerate(cats) if c == cat]
        if len(idx) < 10:
            continue
        yb = [y[i] for i in idx]
        ll = lambda ps: round(log_loss(yb, [min(1 - 1e-6, max(1e-6, ps[i])) for i in idx]), 4)
        buckets[cat] = {"n": len(idx), "partial_pooling": ll(p_pp), "no_pooling": ll(p_np),
                        "segment": ll(p_sg)}
        print(f"    {cat:<11} n={len(idx):<5} partial {buckets[cat]['partial_pooling']:.4f}"
              f"  no_pool {buckets[cat]['no_pooling']:.4f}  segment {buckets[cat]['segment']:.4f}")

    verdict = ("partial pooling beats BOTH segment and no-pooling"
               if results["partial_pooling"]["log_loss"] < results["segment"]["log_loss"]
               and results["partial_pooling"]["log_loss"] < results["no_pooling"]["log_loss"]
               else "partial pooling does not dominate")
    out = {
        "note": "SYNTHETIC estimator validation; real individual claim is BLOCKED-ON-PRIVATE-DATA",
        "n": n, "n_contacts": len(theta), "test_base_rate": round(sum(y) / len(y), 4),
        "segment_rate": round(seg_rate, 4), "arms": results, "by_evidence_bucket": buckets,
        "raw_llm": "BLOCKED: needs an LLM/agent predictor over the same rows",
        "raw_llm_context": "BLOCKED: needs an LLM/agent predictor + as-of history",
        "verdict": verdict,
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT_PATH).write_text(json.dumps(out, indent=1))
    print(f"\n  verdict: {verdict}\n  wrote {RESULT_PATH}")


if __name__ == "__main__":
    run()
