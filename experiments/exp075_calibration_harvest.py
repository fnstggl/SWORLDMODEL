"""EXP-075: the calibration harvest — turn the flywheel ON.

The registry is only as good as the elasticities we've fit into it. This harvests demographic→opinion
elasticities from EVERY GSS attitude item (each signed so "1" = the conservative pole), and COMBINES them in
the learned-prior registry keyed by outcome-class "conservative_opinion". Because 15 items each estimate the
same underlying elasticities (e.g. party=republican → conservative), the precision-weighted combination
across items yields TIGHT, transferable priors — the data-scaling flywheel: more data ⇒ more calibrated
default weights for every future question. The committed `swm/variables/learned_priors.json` is then consulted
by the compiler (`calibrated_compiler.apply_registry`) so an emitted demographic variable arrives pre-
calibrated from real data.

Run: python -m experiments.exp075_calibration_harvest
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from experiments.datasets_gss import load
from experiments.exp073_event_backtest import ATTRS_FULL, _item_rows
from swm.variables.calibrated_weights import CalibratedWeights, uninformative_prior
from swm.variables.llm_prior import ITEM_POLE
from swm.variables.prior_registry import PriorRegistry, semantic_key

REGISTRY_PATH = "swm/variables/learned_priors.json"
RESULT = "experiments/results/exp075_calibration_harvest.json"
OUTCOME_CLASS = "conservative_opinion"


def _vocab(rows):
    v = {}
    for r in rows:
        for a in ATTRS_FULL:
            v.setdefault((a, r["demo"].get(a, "unknown")), len(v))
    return v


def _encode(demo, vocab):
    x = [0.0] * len(vocab)
    for a in ATTRS_FULL:
        j = vocab.get((a, demo.get(a, "unknown")))
        if j is not None:
            x[j] = 1.0
    return x


def run(cap=3000) -> dict:
    rows = load()
    reg = PriorRegistry.load(REGISTRY_PATH)          # accumulate onto any existing registry
    harvested, sd_trace = [], []
    for item, pole in ITEM_POLE.items():
        irows = _item_rows(rows, item)
        if len(irows) < 300:
            continue
        for r in irows:
            r["ys"] = r["y"] if pole > 0 else 1 - r["y"]     # sign so 1 = conservative pole
        sample = irows if len(irows) <= cap else random.Random(hash(item) & 255).sample(irows, cap)
        if len(set(r["ys"] for r in sample)) < 2:
            continue
        vocab = _vocab(sample)
        X = [_encode(r["demo"], vocab) for r in sample]
        y = [r["ys"] for r in sample]
        priors = [uninformative_prior(f"{a}={lv}") for (a, lv) in vocab]
        cw = CalibratedWeights(priors, temper_grid=(1.0, 4.0), epochs=60).fit(X, y, tune=True)
        reg.register_from_fit(cw, OUTCOME_CLASS, source=f"gss:{item}")
        harvested.append(item)
        # trace how the shared 'party=republican' elasticity tightens as items accumulate
        rec = reg.records.get(semantic_key("party=republican", OUTCOME_CLASS))
        if rec is not None:
            sd_trace.append(round(rec.sd, 4))
    reg.save(REGISTRY_PATH)

    # report a few flagship learned elasticities (signed toward the conservative pole)
    flagship = ["party=republican", "party=democrat", "ideology=conservative", "ideology=liberal",
                "relig=none", "attendance=high", "degree=graduate", "race=black"]
    learned = {}
    for name in flagship:
        rec = reg.records.get(semantic_key(name, OUTCOME_CLASS))
        if rec is not None:
            learned[name] = {"elasticity": round(rec.mean, 3), "sd": round(rec.sd, 3), "n": rec.n}
    res = {"outcome_class": OUTCOME_CLASS, "items_harvested": harvested, "n_items": len(harvested),
           "total_priors_in_registry": len(reg.records),
           "party_republican_sd_as_items_accumulate": sd_trace,
           "flagship_elasticities": learned, "registry_path": REGISTRY_PATH}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print(f"EXP-075  calibration harvest — {len(harvested)} GSS items -> {len(reg.records)} learned priors")
    print(f"  the shared 'party=republican -> conservative' elasticity tightens as items accumulate:")
    print(f"    sd trace: {sd_trace[0] if sd_trace else '-'} -> {sd_trace[-1] if sd_trace else '-'} "
          f"(more data => a tighter, transferable prior)")
    print("  flagship learned elasticities (signed toward conservative; +=conservative, −=liberal):")
    for name, v in learned.items():
        print(f"    {name:24s} {v['elasticity']:+.3f} ± {v['sd']:.3f}  (n={v['n']})")
    print(f"  committed {REGISTRY_PATH}  ·  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
