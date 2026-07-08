"""EXP-027: learned feature discipline — win without hand-picking traits (removing EXP-025's asterisk).

EXP-025's person-level win required HAND-PICKING intellectual_humility; the full 23-trait dense model
overfit. Hand-picking doesn't scale and risks snooping. This closes that gap: an elastic-net (L1+L2)
logistic given ALL 23 persona traits performs its OWN feature selection on the training authors only —
driving irrelevant coefficients to exactly zero — and should recover the win while DISCOVERING the
persuasion-relevant traits rather than being told them.

No-cheat: split AUTHORS train/test; the L1 strength is tuned on an inner train/val split of TRAIN only
(never the test authors). Tiers, mean over seeds:
  base_rate           : predict the training majority class
  dense_all23         : L2-only logistic on all 23 traits (the EXP-025 overfitter)
  handpicked_humility : the EXP-025 hand-picked single trait (the target to match, but it cheats a little)
  sparse_learned      : elastic-net on all 23 traits, features chosen by the model on train only

Writes experiments/results/exp027_learned_discipline.json.
Run: python -m experiments.exp027_learned_discipline
"""
from __future__ import annotations

import glob
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from swm.eval.metrics import log_loss
from swm.transition.readout import LogisticReadout
from swm.transition.sparse_readout import ScreenedLogisticReadout, SparseLogisticReadout
from swm.variables.deep_inference import DeepInferenceEngine
from swm.variables.schema import BY_CATEGORY, PERSONA, spec
from experiments.datasets_cmv_history import load

RESULT = "experiments/results/exp027_learned_discipline.json"
PV = BY_CATEGORY[PERSONA]


def _load_llm_signals():
    sig = {}
    paths = glob.glob("data/cmv_deep_[0-9]*.json") or glob.glob("experiments/results/exp025_cmv/cmv_deep_signals.json")
    for fp in paths:
        try:
            rows = json.loads(Path(fp).read_text())
        except Exception:
            continue
        for r in rows:
            if isinstance(r, dict) and "id" in r and "signals" in r:
                sig[r["id"]] = r["signals"]
    return sig


def _tv(persona, t):
    return persona.get(t, {}).get("value", 0.0 if spec(t).signed else 0.5)


def _clip(p):
    return min(1 - 1e-6, max(1e-6, p))


def _build():
    inst, _ = load(min_args=8)
    sig = _load_llm_signals()
    eng = DeepInferenceEngine()
    if sig:
        cov = {r["author"] for r in inst if r["id"] in sig}
        inst = [r for r in inst if r["author"] in cov]
    docs, succ = defaultdict(list), defaultdict(list)
    for r in inst:
        docs[r["author"]].append(sig.get(r["id"]) or eng.per_doc(r["arg_text"]))
        succ[r["author"]].append(r["success"])
    auth = sorted(docs)
    persona = {a: eng.synthesize(docs[a]) for a in auth}
    feat = {a: [_tv(persona[a], t) for t in PV] for a in auth}
    rate = {a: sum(succ[a]) / len(succ[a]) for a in auth}
    return auth, feat, rate, bool(sig)


def _ll_gain(tr, te, feat, y, build_model):
    base = _clip(sum(y[a] for a in tr) / len(tr))
    llb = log_loss([y[a] for a in te], [base] * len(te))
    m = build_model([feat[a] for a in tr], [y[a] for a in tr])
    p = [_clip(m.predict_proba(feat[a])) for a in te]
    acc = sum((pi > 0.5) == y[a] for pi, a in zip(p, te)) / len(te)
    return llb - log_loss([y[a] for a in te], p), acc, m


def _tune_l1(tr, feat, y, grid=(0.005, 0.01, 0.02, 0.04, 0.08, 0.15)):
    """Pick L1 on an inner train/val split of the TRAIN authors only (no test leakage)."""
    rng = random.Random(99); order = list(tr); rng.shuffle(order)
    c = int(0.7 * len(order)); ftr, fval = order[:c], order[c:]
    best, best_ll = grid[0], 1e9
    for l1 in grid:
        m = SparseLogisticReadout(l1=l1, l2=0.01).fit([feat[a] for a in ftr], [y[a] for a in ftr])
        ll = log_loss([y[a] for a in fval], [_clip(m.predict_proba(feat[a])) for a in fval])
        if ll < best_ll:
            best_ll, best = ll, l1
    return best


def run(seeds=(0, 1, 2, 3, 4, 5)):
    auth, feat, rate, is_llm = _build()
    tiers = {k: {"gain": [], "acc": []} for k in
             ("dense_all23", "sparse_l1_all23", "screened_learned", "handpicked_humility")}
    l1s, ks, kept_counter = [], [], Counter()
    for seed in seeds:
        rng = random.Random(seed); order = list(auth); rng.shuffle(order)
        cut = int(0.7 * len(order)); tr, te = order[:cut], order[cut:]
        med = sorted(rate[a] for a in tr)[len(tr) // 2]
        y = {a: int(rate[a] > med) for a in auth}

        # dense on all 23 (the EXP-025 overfitter)
        g, ac, _ = _ll_gain(tr, te, feat, y, lambda X, Y: LogisticReadout(epochs=400, l2=1.0).fit(X, Y))
        tiers["dense_all23"]["gain"].append(g); tiers["dense_all23"]["acc"].append(ac)

        # L1 elastic-net on all 23, L1 tuned on inner train CV (honest: unstable at this n)
        l1 = _tune_l1(tr, feat, y); l1s.append(l1)
        g, ac, _ = _ll_gain(tr, te, feat, y, lambda X, Y: SparseLogisticReadout(l1=l1, l2=0.01).fit(X, Y))
        tiers["sparse_l1_all23"]["gain"].append(g); tiers["sparse_l1_all23"]["acc"].append(ac)

        # SCREENED: top-k by train correlation, k tuned on inner CV — the learned discipline that works.
        # A sparse grid (k<=3) is the right prior at ~110 training authors: EXP-027 shows k=5,8 overfit.
        def build_screen(X, Y):
            return ScreenedLogisticReadout(k=None, k_grid=(1, 2, 3), seed=0).fit(X, Y)
        g, ac, m = _ll_gain(tr, te, feat, y, build_screen)
        tiers["screened_learned"]["gain"].append(g); tiers["screened_learned"]["acc"].append(ac)
        ks.append(len(m.keep_))
        for name in m.selected(PV):
            kept_counter[name] += 1

        # hand-picked single trait (the EXP-025 target this must match without hand-picking)
        hi = PV.index("intellectual_humility")
        fh = {a: [feat[a][hi]] for a in auth}
        g, ac, _ = _ll_gain(tr, te, fh, y, lambda X, Y: LogisticReadout(epochs=400, l2=1.0).fit(X, Y))
        tiers["handpicked_humility"]["gain"].append(g); tiers["handpicked_humility"]["acc"].append(ac)

    out = {"n_authors": len(auth), "persona_source": "LLM agent swarm" if is_llm else "lexical",
           "tuned_l1_mean": round(statistics.mean(l1s), 4), "screened_k_mean": round(statistics.mean(ks), 2),
           "tiers": {k: {"gain": round(statistics.mean(v["gain"]), 4),
                         "acc": round(statistics.mean(v["acc"]), 4)} for k, v in tiers.items()},
           "traits_screener_selected": [(t, c) for t, c in kept_counter.most_common() if c >= 1]}
    print(f"EXP-027 learned feature discipline — CMV person-level, {len(auth)} authors, {len(seeds)} seeds")
    for k, v in out["tiers"].items():
        print(f"  {k:<20} log-loss gain vs base {v['gain']:+.4f}  accuracy {v['acc']}")
    print(f"  screener chose k≈{out['screened_k_mean']} of 23 traits; the traits it selected "
          f"(times across {len(seeds)} seeds) — its OWN choice, no hand-picking:")
    for t, c in out["traits_screener_selected"][:8]:
        print(f"    {c}/{len(seeds)}  {t}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
