"""EXP-052: the unified readout on a real FULL VariableMap — does reliability weighting generalize?

EXP-050 validated reliability weighting with INJECTED noise (redundant noisy copies). The honest test is a
real mixed-provenance VariableMap: ChangeMyView persuasion, where the map mixes
  - LLM-INFERRED persuasion variables (provenance "llm", reliability 0.55): op_openness/skepticism/
    entrenchment, arg crux-fit/evidence/clarity/respect/expertise — EXP-021 showed these ARE the signal;
  - GROUNDED surface variables (provenance "data"/"heuristic"): message length, links, quotes, question
    marks — weak.

This exposes a subtlety the injected-noise proxy hid: RELIABILITY ≠ RELEVANCE. A reliably-measured
variable can be irrelevant; a noisily-inferred variable can carry the unique signal. So we compare, on a
no-cheat temporal split:
  1. uniform            — standard logistic (coefficients LEARN relevance; no reliability scaling)
  2. reliability-scaled — features pre-scaled by provenance reliability (down-weights the inferred signal)
  3. reliability-prior  — reliability as a soft prior on coefficient magnitude (shrink low-reliability vars
                          toward zero, but let data override) — the honest middle path
  4. decorrelated       — latent factors over the full map (does structure help on the real map?)

The point is to find out honestly whether provenance-reliability weighting helps, hurts, or must be made
relevance-aware on a real VariableMap. Writes JSON. Run: python -m experiments.exp052_full_variablemap
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.transition.readout import LogisticReadout
from swm.variables.latent_factor_readout import _cov, _top_factors

RESULT = "experiments/results/exp052_full_variablemap.json"

INFERRED = ["op_openness", "op_skepticism", "op_entrenchment", "arg_addresses_crux", "arg_evidence",
            "arg_clarity", "arg_respectfulness", "arg_expertise"]        # provenance llm, reliability 0.55
# grounded surface features (name, provenance)
GROUNDED = [("arg_logwords", "data"), ("arg_has_link", "data"), ("arg_quotes", "heuristic"),
            ("arg_q", "heuristic"), ("op_logwords", "data")]
RELIABILITY = {"data": 1.0, "heuristic": 0.3, "llm": 0.55}


def _f(v, d=0.5):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _surface(op, arg):
    aw = arg.split()
    return {"arg_logwords": math.log1p(len(aw)), "arg_has_link": 1.0 if "http" in arg else 0.0,
            "arg_quotes": min(1.0, arg.count("&gt;") / 3.0), "arg_q": min(1.0, arg.count("?") / 3.0),
            "op_logwords": math.log1p(len(op.split()))}


def _load():
    cases = json.loads(Path("data/cmv_common.json").read_text())
    inf = {}
    for fp in glob.glob("data/cmv_infer_*.json"):
        for r in json.loads(Path(fp).read_text()):
            inf[r["id"]] = r
    rows = []
    for s in cases:
        if s["id"] not in inf:
            continue
        i = inf[s["id"]]; surf = _surface(s["op_text"], s["arg_text"])
        feats = [_f(i.get(k)) for k in INFERRED] + [surf[k] for k, _ in GROUNDED]
        rows.append({"x": feats, "y": int(s["success"]), "ts": s["ts"]})
    rows.sort(key=lambda r: r["ts"])
    return rows


NAMES = INFERRED + [k for k, _ in GROUNDED]
PROV = ["llm"] * len(INFERRED) + [p for _, p in GROUNDED]
REL = [RELIABILITY[p] for p in PROV]


def _score(y, p):
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4), "uplift@20": round(uplift_at_k(y, p, 0.2), 4),
            "accuracy": round(sum(int((pi >= .5) == (yi == 1)) for pi, yi in zip(p, y)) / len(y), 4)}


def _fit_predict(Xtr, ytr, Xte, l2=1.0):
    m = LogisticReadout(epochs=400, l2=l2).fit(Xtr, ytr)
    return [m.predict_proba(x) for x in Xte], m


def run():
    rows = _load()
    n = len(rows); cut = int(0.7 * n)
    tr, te = rows[:cut], rows[cut:]
    ytr = [r["y"] for r in tr]; yte = [r["y"] for r in te]

    def X(rowset, scale=None):
        if scale is None:
            return [r["x"] for r in rowset]
        return [[v * s for v, s in zip(r["x"], scale)] for r in rowset]

    # 1. uniform (logistic learns relevance)
    p_uniform, m_u = _fit_predict(X(tr), ytr, X(te))
    # 2. reliability-scaled features
    p_relscale, _ = _fit_predict(X(tr, REL), ytr, X(te, REL))
    # 3. reliability as a soft prior: stronger L2 on low-reliability features (scale l2 per feature) —
    #    approximate by scaling features by sqrt(reliability) (a milder down-weight the data can override)
    soft = [r ** 0.5 for r in REL]
    p_relprior, _ = _fit_predict(X(tr, soft), ytr, X(te, soft))
    # 4. decorrelated: latent factors over the full map
    Xtr_raw = X(tr)
    mean = [sum(row[j] for row in Xtr_raw) / len(Xtr_raw) for j in range(len(NAMES))]
    factors = _top_factors(_cov(Xtr_raw, mean), 5)

    def proj(x):
        c = [x[j] - mean[j] for j in range(len(x))]
        return [sum(c[j] * f[j] for j in range(len(f))) for f in factors]
    p_decorr, _ = _fit_predict([proj(r["x"]) for r in tr], ytr, [proj(r["x"]) for r in te])

    arms = {"uniform_learns_relevance": _score(yte, p_uniform),
            "reliability_scaled": _score(yte, p_relscale),
            "reliability_soft_prior": _score(yte, p_relprior),
            "decorrelated_factors": _score(yte, p_decorr)}
    # which features the uniform model actually leaned on (learned relevance) — by provenance
    drivers = sorted(zip(NAMES, PROV, m_u.w), key=lambda t: -abs(t[2]))[:6]

    u = arms["uniform_learns_relevance"]["log_loss"]
    out = {"dataset": "ChangeMyView", "n": n, "n_test": len(yte),
           "base_rate": round(sum(yte) / len(yte), 4), "arms": arms,
           "reliability_scaling_helps": arms["reliability_scaled"]["log_loss"] < u,
           "top_learned_drivers": [[nm, pv, round(w, 3)] for nm, pv, w in drivers]}

    print(f"EXP-052 full VariableMap on CMV — n={n} test={len(yte)} base rate {out['base_rate']}, "
          f"{len(INFERRED)} inferred + {len(GROUNDED)} grounded vars")
    for k, v in arms.items():
        print(f"  {k:<26} log_loss {v['log_loss']}  brier {v['brier']}  acc {v['accuracy']}  up@20 {v['uplift@20']}")
    print(f"  -> reliability-scaling helps vs uniform: {out['reliability_scaling_helps']}")
    print("  top LEARNED drivers (uniform model) — note the provenance:")
    for nm, pv, w in drivers:
        print(f"     {nm:<22} [{pv:<9}] weight {round(w, 3)}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
