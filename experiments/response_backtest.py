"""General individual-response backtest + improvement ladder (Parts B & C).

Any dataset becomes a time-ordered list of samples: (entity_id, segment_id, message_features,
outcome). This harness runs a no-cheat temporal split and evaluates an IMPROVEMENT LADDER of the
`ResponseModel` — each rung adds one realism upgrade — so we can see which additions actually help,
across domains, until diminishing returns. Also reports depth slices (does entity state help more
with evidence?) and a raw-LLM comparison hook.

Datasets plug in via loaders in experiments/datasets_*.py. Run:
  python -m experiments.response_backtest github
  python -m experiments.response_backtest stackexchange
  python -m experiments.response_backtest convokit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.transition.response_model import ResponseConfig, ResponseModel

# the improvement ladder: each rung adds ONE upgrade on top of the previous
LADDER = [
    ("segment_only", ResponseConfig(use_pooled_rate=False, use_message=False, use_recency=False,
                                    readout="pooled")),
    ("pooled_rate", ResponseConfig(use_recency=False, readout="pooled")),
    ("+message(logistic)", ResponseConfig(use_recency=False, readout="logistic")),
    ("+recency", ResponseConfig(readout="logistic", use_recency=True)),
    ("+multilevel", ResponseConfig(readout="logistic", use_recency=True, use_multilevel=True)),
    ("+state_feats", ResponseConfig(readout="logistic", use_recency=True, use_multilevel=True,
                                    use_state_features=True)),
    ("+interactions", ResponseConfig(readout="logistic", use_recency=True, use_multilevel=True,
                                     use_state_features=True, use_interactions=True)),
    ("+gbdt_readout", ResponseConfig(readout="gbdt", use_recency=True, use_multilevel=True,
                                     use_state_features=True, use_interactions=True)),
    ("+calibration", ResponseConfig(readout="gbdt", use_recency=True, use_multilevel=True,
                                    use_state_features=True, use_interactions=True, calibrate=True)),
]


def _score(y, p):
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    return {"n": len(y), "pos": sum(y), "log_loss": round(log_loss(y, p), 4),
            "brier": round(brier_score(y, p), 4), "ece": round(expected_calibration_error(y, p), 4),
            "uplift@20": round(uplift_at_k(y, p, 0.2), 4)}


def _fit_predict(samples, mfn, cfg, cut):
    train, test = samples[:cut], samples[cut:]
    gr = (sum(o for *_, o in train) + 1) / (len(train) + 2)
    m = ResponseModel(message_feature_names=mfn, config=cfg).fit_stream(train, global_rate=gr)
    preds, y, depths = [], [], []
    for eid, seg, mf, o in test:
        e = m._ent.get(eid)
        depths.append(e.n if e else 0)
        preds.append(m.predict(eid, seg, mf)); y.append(int(o))
        m.observe(eid, seg, o)
    return preds, y, depths


def _stacked(samples, mfn, cut):
    """The learned evidence-aware combiner (stacking) — fit meta-learner on a held-out tail of TRAIN."""
    from swm.transition.stacked_response import StackedResponseModel
    train, test = samples[:cut], samples[cut:]
    gr = (sum(o for *_, o in train) + 1) / (len(train) + 2)
    m = StackedResponseModel(message_feature_names=mfn).fit_stream(train, global_rate=gr)
    preds, y, depths = [], [], []
    for eid, seg, mf, o in test:
        e = m._ent.get(eid)
        depths.append(e.n if e else 0)
        preds.append(m.predict(eid, seg, mf)); y.append(int(o))
        m.observe(eid, seg, o)
    return preds, y, depths


def run_ladder(samples, mfn, *, name="dataset", split=0.7):
    n = len(samples)
    cut = int(split * n)
    base = sum(o for *_, o in samples[:cut]) / cut
    print(f"\n=== {name}: n={n} train={cut} test={n-cut} base(train)={base:.3f} ===")
    print(f"  {'rung':<22}{'log_loss':>9}{'brier':>8}{'ece':>7}{'up@20':>8}{'Δll':>8}")
    rows = {}
    prev = None
    best_ll, best_name = 1e9, None
    depth_cache = None
    for rung, cfg in LADDER:
        preds, y, depths = _fit_predict(samples, mfn, cfg, cut)
        s = _score(y, preds)
        d = (prev - s["log_loss"]) if prev is not None else 0.0
        flag = " *" if s["log_loss"] < best_ll else ""
        print(f"  {rung:<22}{s['log_loss']:>9.4f}{s['brier']:>8.4f}{s['ece']:>7.4f}{s['uplift@20']:>8.3f}{d:>+8.4f}{flag}")
        rows[rung] = s
        prev = s["log_loss"]
        if s["log_loss"] < best_ll:
            best_ll, best_name = s["log_loss"], rung
            depth_cache = (preds, y, depths)          # slice on the BEST config (not the overfit GBDT)
    # the learned combiner (stacking) — the highest-leverage fusion of state-model + content prior
    sp, sy, sd = _stacked(samples, mfn, cut)
    ss = _score(sy, sp)
    print(f"  {'STACKED_combiner':<22}{ss['log_loss']:>9.4f}{ss['brier']:>8.4f}{ss['ece']:>7.4f}"
          f"{ss['uplift@20']:>8.3f}{best_ll - ss['log_loss']:>+8.4f}"
          f"{'  <-- vs best single' }")
    rows["STACKED_combiner"] = ss
    if ss["log_loss"] < best_ll:
        best_ll, best_name, depth_cache = ss["log_loss"], "STACKED_combiner", (sp, sy, sd)

    # depth slices on the BEST config
    if depth_cache:
        preds, y, depths = depth_cache
        print("  by entity-history depth (does state help more with evidence?):")
        for lo, hi, nm in [(0, 0, "cold(0)"), (1, 4, "repeat(1-4)"), (5, 10 ** 9, "deep(5+)")]:
            idx = [i for i, d in enumerate(depths) if lo <= d <= hi]
            if len(idx) < 20 or sum(y[i] for i in idx) < 5:
                continue
            ys = [y[i] for i in idx]
            ll = log_loss(ys, [min(1 - 1e-6, max(1e-6, preds[i])) for i in idx])
            print(f"    {nm:<12} n={len(idx):<5} gbdt_ll {ll:.4f}  base {sum(ys)/len(ys):.3f}")
    print(f"  BEST: {best_name} ({best_ll:.4f})   improvement vs pooled_rate: "
          f"{rows['pooled_rate']['log_loss'] - best_ll:+.4f}")
    return {"name": name, "n": n, "base_rate": round(base, 4), "ladder": rows,
            "best": best_name, "best_log_loss": best_ll}


# ------------------------------------------------------------------ dataset dispatch
def load(name):
    import importlib
    mod = importlib.import_module(f"experiments.datasets_{name}")
    return mod.load_samples()


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "github"
    samples, mfn = load(name)
    out = run_ladder(samples, mfn, name=name)
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(f"experiments/results/exp016_{name}_ladder.json").write_text(json.dumps(out, indent=1))
    print(f"  wrote experiments/results/exp016_{name}_ladder.json")


if __name__ == "__main__":
    main()
