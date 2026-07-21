"""Phase 3 accuracy — FITTED hierarchical observation models (replacing the single global shrinkage).

The Phase-3B repair used one global likelihood temperature (gamma). This module instead LEARNS, from a
historical training corpus of (question outcome, its tagged claims), how discriminating a directional claim of
a given CLASS actually is — with PARTIAL POOLING so sparse groups shrink toward broader calibrated ones.

Model (per question q with tagged claims c):
    score(q)      = sum_c  sign(c) * strength(c) * reliability(c) * w[class(c)]
    logit P(yes)  = a + score(q)
  sign(c)   = +1 supports_yes, -1 supports_no, 0 neutral
  strength  = weak .5 / moderate .75 / strong 1.0
  class(c)  = the claim's evidence class (observed_fact, forecast, actor_statement, allegation, denial, ...)
  w[class]  ~ Normal(w_global, .)   -- partial pooling: penalty pulls each class weight toward the global one,
              and the global toward 0. Domain enters as an optional additive level when data supports it.

This is a genuine hierarchical fit on TRAINING data only. Its per-observation LR
    fitted_lr(tag) = exp(w[class] * strength * reliability)     (for a 'favors/supports_yes' observation)
feeds both the generic rate posterior and the causal-latent inference. The module is compared on held-out data
against the hand-set tables, global gamma, and no-info updates; the SIMPLER model wins ties (Occam gate).
"""
from __future__ import annotations
import json
import math
from pathlib import Path

_EPS = 1e-6
_STRENGTH = {"weak": 0.5, "moderate": 0.75, "strong": 1.0}


def _sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else (lambda z: z / (1 + z))(math.exp(x))


def _sign(direction):
    return 1.0 if direction == "supports_yes" else (-1.0 if direction == "supports_no" else 0.0)


def claim_features(tag):
    """Extract (class, sign, strength, reliability) from a captured tag row. class falls back to 'generic'."""
    cls = (tag.get("claim_class") or tag.get("source_type") or "generic")
    return {"class": str(cls), "sign": _sign(tag.get("outcome_direction", "neutral")),
            "strength": _STRENGTH.get(tag.get("strength", "moderate"), 0.75),
            "reliability": float(tag.get("reliability", 0.8))}


def _question_terms(row):
    """Per-question list of (class, contribution) where contribution = sign*strength*reliability."""
    out = []
    for t in row.get("tags", []):
        fe = claim_features(t)
        if fe["sign"] == 0.0:
            continue
        out.append((fe["class"], fe["sign"] * fe["strength"] * fe["reliability"]))
    return out


def fit(train_rows, *, l2_global=0.5, l2_class=2.0, iters=3000, lr=0.05):
    """Fit intercept a, global weight wg, and per-class deltas d[class] (w[class]=wg+d[class]) by penalized
    logistic regression on question outcomes. Partial pooling: l2_class pulls d->0 (toward global), l2_global
    pulls wg->0. Deterministic."""
    classes = sorted({c for r in train_rows for (c, _) in _question_terms(r)})
    a = 0.0
    wg = 0.5
    d = {c: 0.0 for c in classes}
    n = max(1, len(train_rows))
    terms = [( _question_terms(r), r["outcome"]) for r in train_rows if r.get("outcome") in (0, 1)]
    for _ in range(iters):
        ga = 0.0
        gwg = 0.0
        gd = {c: 0.0 for c in classes}
        for tlist, y in terms:
            score = sum((wg + d.get(c, 0.0)) * contrib for c, contrib in tlist)
            p = _sigmoid(a + score)
            e = p - y
            ga += e
            for c, contrib in tlist:
                gwg += e * contrib
                gd[c] += e * contrib
        a -= lr * ga / n
        wg -= lr * (gwg / n + l2_global * wg)
        for c in classes:
            d[c] -= lr * (gd[c] / n + l2_class * d[c])
    return {"a": round(a, 4), "w_global": round(wg, 4),
            "class_delta": {c: round(v, 4) for c, v in d.items()},
            "classes": classes, "l2_global": l2_global, "l2_class": l2_class}


def w_for_class(params, cls):
    return params["w_global"] + params.get("class_delta", {}).get(str(cls), 0.0)


def predict_rate(row, params):
    """P(yes) for a question from its fitted per-class weighted claim votes."""
    score = sum(w_for_class(params, c) * contrib for c, contrib in _question_terms(row))
    return max(_EPS, min(1 - _EPS, _sigmoid(params["a"] + score)))


def fitted_lr(tag, params):
    """Direction-SIGNED per-observation likelihood ratio for the YES outcome: a supports_yes claim gives >1,
    a supports_no claim gives <1, neutral gives 1. Used by the generic-rate path."""
    fe = claim_features(tag)
    w = w_for_class(params, fe["class"])
    return math.exp(w * fe["sign"] * fe["strength"] * fe["reliability"])


def fitted_lr_magnitude(tag, params):
    """Direction-AGNOSTIC discrimination MAGNITUDE (always >=1) of a claim of this class/strength. The consumer
    (e.g. the causal-latent inference) applies the link direction itself, so sign is excluded here."""
    fe = claim_features(tag)
    w = w_for_class(params, fe["class"])
    return math.exp(abs(w) * fe["strength"] * fe["reliability"])


def save(params, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(params, indent=2))


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None
