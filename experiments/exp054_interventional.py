"""EXP-054: the interventional KPI — can the model pick the causally-better headline? (SIMULATION_AUDIT KPI-A)

The audit's verdict: every current KPI scores predictive RECONSTRUCTION, none scores INTERVENTION — "what
happens if I do X." The Upworthy Research Archive is randomized headline A/B tests, so the observed CTR
difference between arms IS the causal effect of the headline. Choosing a headline is a real `do(x)`, and
its realized CTR is an unbiased outcome (uniform randomization → off-policy value is exact).

We train a headline→CTR model on TRAIN experiments and, on held-out experiments, have it PICK the arm it
predicts best. Scored the way a what-would-happen engine must be:
  - POLICY VALUE / REGRET: realized CTR of the model-chosen arm vs the oracle (best), the random/observed
    policy (mean), and worst — and the fraction of achievable uplift (best − mean) captured;
  - CATE-SIGN accuracy: over arm pairs, does the model rank the causally-better headline higher?

This is the first KPI in the project that tests the thesis directly. Writes JSON.
Run: python -m experiments.exp054_interventional
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from experiments.datasets_upworthy import load

RESULT = "experiments/results/exp054_interventional.json"
_NUM = re.compile(r"\d")
_CURIO = re.compile(r"\b(why|how|what|secret|reason|reasons|will|this|these|when|watch|proves?|"
                    r"actually|really|truth|need|should)\b", re.I)
_EMOT = re.compile(r"\b(amazing|incredible|shocking|nightmare|perfect|beautiful|powerful|heartbreaking|"
                   r"stunning|worst|best|hilarious|epic|genius|brilliant|horrifying|inspiring|wrong)\b", re.I)


def _features(h):
    words = h.split()
    nw = max(1, len(words))
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    return [
        math.log1p(len(words)),
        1.0 if "?" in h else 0.0,
        1.0 if "!" in h else 0.0,
        1.0 if _NUM.search(h) else 0.0,
        caps / nw,
        1.0 if re.search(r"\byou\b|\byour\b", h, re.I) else 0.0,
        min(1.0, len(_CURIO.findall(h)) / 2.0),
        min(1.0, len(_EMOT.findall(h)) / 2.0),
        1.0 if h.endswith("...") or h.endswith(".") else 0.0,
        min(1.0, h.count(",") / 2.0),
    ]


class _RidgeGD:
    """Tiny standardized ridge regressor (pure Python) — predicts CTR from headline features."""
    def __init__(self, l2=1.0, lr=0.1, epochs=400):
        self.l2, self.lr, self.epochs = l2, lr, epochs

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.mu = [sum(r[j] for r in X) / n for j in range(d)]
        self.sd = [max(1e-9, (sum((r[j] - self.mu[j]) ** 2 for r in X) / n) ** 0.5) for j in range(d)]
        Xs = [[(r[j] - self.mu[j]) / self.sd[j] for j in range(d)] for r in X]
        self.w = [0.0] * d; self.b = sum(y) / n
        for _ in range(self.epochs):
            gw = [0.0] * d; gb = 0.0
            for xi, yi in zip(Xs, y):
                e = (self.b + sum(self.w[j] * xi[j] for j in range(d))) - yi
                for j in range(d):
                    gw[j] += e * xi[j]
                gb += e
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j] / n)
            self.b -= self.lr * gb / n
        return self

    def predict(self, x):
        xs = [(x[j] - self.mu[j]) / self.sd[j] for j in range(len(x))]
        return self.b + sum(self.w[j] * xs[j] for j in range(len(x)))


def _split(tests, frac=0.3):
    import zlib
    tr, te = [], []
    for t in tests:
        (te if (zlib.crc32(t["test_id"].encode()) % 1000) / 1000.0 < frac else tr).append(t)
    return tr, te


def run():
    tests = load()
    tr, te = _split(tests)
    Xtr = [f for t in tr for f in [_features(a["headline"]) for a in t["arms"]]]
    ytr = [a["ctr"] for t in tr for a in t["arms"]]
    model = _RidgeGD().fit(Xtr, ytr)

    picked, oracle, mean_pol, worst = [], [], [], []
    pair_correct, pair_tot = 0, 0
    uplift_captured = []
    for t in te:
        arms = t["arms"]
        preds = [model.predict(_features(a["headline"])) for a in arms]
        ctrs = [a["ctr"] for a in arms]
        pick = max(range(len(arms)), key=lambda i: preds[i])
        best, mean_c, worst_c = max(ctrs), sum(ctrs) / len(ctrs), min(ctrs)
        picked.append(ctrs[pick]); oracle.append(best); mean_pol.append(mean_c); worst.append(worst_c)
        if best - mean_c > 1e-9:
            uplift_captured.append((ctrs[pick] - mean_c) / (best - mean_c))
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                if abs(ctrs[i] - ctrs[j]) < 1e-6:
                    continue
                pair_tot += 1
                pair_correct += int((preds[i] > preds[j]) == (ctrs[i] > ctrs[j]))

    n = len(picked)
    mp = lambda v: sum(v) / len(v)
    policy = {"model_policy_ctr": round(mp(picked), 5), "oracle_ctr": round(mp(oracle), 5),
              "random_policy_ctr": round(mp(mean_pol), 5), "worst_ctr": round(mp(worst), 5)}
    achievable = policy["oracle_ctr"] - policy["random_policy_ctr"]
    captured = (policy["model_policy_ctr"] - policy["random_policy_ctr"])
    out = {"dataset": "upworthy", "n_train_arms": len(ytr), "n_test_experiments": n,
           "policy_ctr": policy,
           "uplift_over_random_pp": round(captured * 100, 4),
           "fraction_of_achievable_uplift_captured": round(captured / achievable, 4) if achievable > 1e-9 else None,
           "mean_uplift_captured_within_test": round(mp(uplift_captured), 4) if uplift_captured else None,
           "cate_sign_accuracy": round(pair_correct / pair_tot, 4) if pair_tot else None,
           "beats_random_policy": captured > 0,
           "feature_weights": [round(w, 5) for w in model.w]}

    print(f"EXP-054 interventional KPI (Upworthy A/B) — {n} held-out experiments, {len(ytr)} train arms")
    print(f"  POLICY VALUE (realized CTR of the chosen headline; randomized => causal):")
    print(f"     oracle (best arm)   {policy['oracle_ctr']}")
    print(f"     MODEL pick          {policy['model_policy_ctr']}")
    print(f"     random/observed     {policy['random_policy_ctr']}")
    print(f"     worst arm           {policy['worst_ctr']}")
    print(f"  -> uplift over random: {out['uplift_over_random_pp']} pp; "
          f"fraction of achievable uplift captured: {out['fraction_of_achievable_uplift_captured']}")
    print(f"  -> CATE-sign (pairwise causal ranking) accuracy: {out['cate_sign_accuracy']} "
          f"(chance 0.5); beats random policy: {out['beats_random_policy']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
