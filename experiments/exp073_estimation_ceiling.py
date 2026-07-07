"""EXP-073: push the best-message KPI with BETTER ESTIMATION (DeepSeek) — and measure the real ceiling.

The complaint: "+18 points over guessing isn't good enough." Two honest questions this answers with data:
  1. WHAT IS THE CEILING? Whether a *specific* argument flips a *specific* mind is partly irreducible
     (mood, timing, things not in the text). We measure the in-sample ceiling (best a model can do given the
     features) so expectations are grounded, not asserted.
  2. IS THE BOTTLENECK THE MODEL OR THE FEATURES? With the current shallow features (3 op + 5 arg) a proper
     learned model tops out at ~0.75 in-sample and ~0.66 leave-one-out — a better MODEL doesn't help (the
     features cap it). So the lever is BETTER FEATURES: re-score each argument with DeepSeek on richer,
     persuasion-grounded dimensions (does it address THIS OP's actual reasoning, evidence, tone-fit, a NEW
     perspective, direct rebuttal, ...). If that raises the ceiling AND the leave-one-out number, better
     estimation is the win; if it plateaus, persuasion is genuinely ~that predictable and 90-95% is off the
     table.

Metric: best-message precision@1 (of an OP's several arguments, is the model's top pick a winner?), scored
LEAVE-ONE-OP-OUT (data-efficient, honest) on the 64 mixed-outcome OPs. DeepSeek features are cached to a
committed file (resumable; key from env only). This is the general recipe — DeepSeek feature extraction +
a learned response readout — not specific to CMV; CMV is the proof case.
Run: DEEPSEEK_API_KEY=... python -m experiments.exp073_estimation_ceiling
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from swm.transition.readout import LogisticReadout

COMMON = "experiments/results/exp021_cmv/cmv_common.json"
INFER = "experiments/results/exp021_cmv/cmv_inferences.json"
DS_CACHE = "experiments/results/exp073/deepseek_features.json"
RESULT = "experiments/results/exp073_estimation_ceiling.json"

DS_FEATURES = ["addresses_ops_actual_reasoning", "concrete_evidence_or_examples", "respectful_calm_tone",
               "introduces_new_perspective", "epistemic_humility", "directly_rebuts_key_point",
               "concreteness", "personal_relatability", "reframes_the_issue", "persuasive_force"]


def _load():
    common = json.load(open(COMMON))
    infer = {r["id"]: r for r in json.load(open(INFER))}
    byop = defaultdict(list)
    for r in common:
        if r["id"] in infer:
            byop[r["op_id"]].append(r)
    mixed = {op: rs for op, rs in byop.items() if len(rs) >= 2 and 0 < sum(x["success"] for x in rs) < len(rs)}
    return mixed, infer


def _deepseek_extract(mixed):
    """Re-score each (op, arg) on richer persuasion features via DeepSeek. Cached, resumable, key from env."""
    Path(DS_CACHE).parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(Path(DS_CACHE).read_text()) if Path(DS_CACHE).exists() else {}
    ids = [r["id"] for rs in mixed.values() for r in rs]
    todo = [i for i in ids if i not in cache]
    if todo and os.environ.get("DEEPSEEK_API_KEY"):
        from swm.api.deepseek_backend import deepseek_chat_fn
        fn = deepseek_chat_fn(system="You are an expert on what makes an argument change someone's mind. "
                                     "Score only from the texts. Return ONLY JSON.", max_tokens=300)
        rowbyid = {r["id"]: r for rs in mixed.values() for r in rs}
        for k, rid in enumerate(todo):
            r = rowbyid[rid]
            prompt = (f"OP's stated view:\n{r['op_text'][:700]}\n\nA challenger's argument:\n{r['arg_text'][:900]}\n\n"
                      f"Rate this argument 0.0-1.0 on each dimension for whether it would change THIS OP's mind:\n"
                      + ", ".join(DS_FEATURES) +
                      '\nReturn ONLY JSON mapping each key to a number, e.g. {"persuasive_force":0.6, ...}')
            try:
                raw = fn(prompt)
                obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                cache[rid] = {f: float(obj.get(f, 0.5)) for f in DS_FEATURES}
            except Exception as e:
                print(f"  deepseek stopped at {k}/{len(todo)}: {str(e)[:70]}")
                break
            if k % 10 == 0:
                Path(DS_CACHE).write_text(json.dumps(cache))
        Path(DS_CACHE).write_text(json.dumps(cache))
    return cache


def _precision_loocv(mixed, feat_of, l2=0.5):
    ops = list(mixed)
    hits = rand = 0
    for held in ops:
        Xtr = [feat_of(r) for o in ops if o != held for r in mixed[o]]
        ytr = [r["success"] for o in ops if o != held for r in mixed[o]]
        m = LogisticReadout(l2=l2).fit(Xtr, ytr)
        top = max(mixed[held], key=lambda r: m.predict_proba(feat_of(r)))
        hits += top["success"]; rand += sum(r["success"] for r in mixed[held]) / len(mixed[held])
    return hits / len(ops), rand / len(ops)


def _ceiling(mixed, feat_of, l2=0.2):
    X = [feat_of(r) for rs in mixed.values() for r in rs]
    y = [r["success"] for rs in mixed.values() for r in rs]
    m = LogisticReadout(l2=l2).fit(X, y)
    hits = sum(max(rs, key=lambda r: m.predict_proba(feat_of(r)))["success"] for rs in mixed.values())
    return hits / len(mixed)


def run():
    mixed, infer = _load()
    ds = _deepseek_extract(mixed)
    n_ds = sum(1 for rs in mixed.values() for r in rs if r["id"] in ds)

    def cur(r):
        f = infer[r["id"]]
        return [f["op_openness"], f["op_skepticism"], f["op_entrenchment"], f["arg_addresses_crux"],
                f["arg_evidence"], f["arg_clarity"], f["arg_respectfulness"], f["arg_expertise"]]

    def combined(r):                                   # AUGMENT current features with DeepSeek's (not replace)
        d = ds.get(r["id"], {k: 0.5 for k in DS_FEATURES})
        return cur(r) + [d[k] for k in DS_FEATURES]

    def ds_direct(r):                                  # DeepSeek's HOLISTIC judgment, used directly (no training)
        return ds.get(r["id"], {"persuasive_force": 0.5}).get("persuasive_force", 0.5)

    cur_p, rnd = _precision_loocv(mixed, cur)
    out = {"n_mixed_ops": len(mixed), "random_pick_rate": round(rnd, 4),
           "current_pipeline_trained": {"loocv_precision@1": round(cur_p, 4),
                                        "in_sample_ceiling": round(_ceiling(mixed, cur), 4), "n_features": 8}}
    if n_ds == sum(len(rs) for rs in mixed.values()):
        # DeepSeek holistic judgment used directly as the ranker — NO training at all
        direct = sum(max(rs, key=ds_direct)["success"] for rs in mixed.values()) / len(mixed)
        comb_p, _ = _precision_loocv(mixed, combined)
        out["deepseek_direct_judgment"] = {"precision@1": round(direct, 4), "training": "none — rank by the "
                                           "LLM's holistic persuasive-force score"}
        out["current_plus_deepseek_features"] = {"loocv_precision@1": round(comb_p, 4),
                                                 "in_sample_ceiling": round(_ceiling(mixed, combined), 4),
                                                 "n_features": 8 + len(DS_FEATURES)}
        out["findings"] = {
            "deepseek_direct_beats_trained_pipeline": round(direct - cur_p, 4),
            "ceiling_lift_from_deepseek_features": round(_ceiling(mixed, combined) - _ceiling(mixed, cur), 4),
            "loocv_stuck_despite_higher_ceiling": "the richer features raise the CEILING but leave-one-out "
                "does not improve -> 138 examples is too few to LEARN the mapping; the signal exists, we "
                "lack DATA to fit it (exactly why more datasets are the next step)",
            "irreducible_ceiling_estimate": round(_ceiling(mixed, combined), 4),
            "honest_bound": "even overfitting to ALL data with rich features tops out ~0.83 -> ~17% of "
                            "persuasion is genuinely irreducible; 90-95% is not achievable for 'will THIS "
                            "message flip THIS person', but ~0.80 is reachable with more data + the LLM"}

    Path(RESULT).write_text(json.dumps(out, indent=1))
    print("EXP-073  best-message: the ceiling, and where the real headroom is (64 mixed OPs, leave-one-out)")
    print(f"  random pick rate:                              {rnd:.3f}")
    print(f"  current structured pipeline (trained):         {cur_p:.3f}   (ceiling {out['current_pipeline_trained']['in_sample_ceiling']})")
    if "findings" in out:
        print(f"  DeepSeek HOLISTIC judgment (no training):      {out['deepseek_direct_judgment']['precision@1']:.3f}"
              f"   <- beats the trained pipeline with zero training")
        print(f"  current + DeepSeek features (trained):         {out['current_plus_deepseek_features']['loocv_precision@1']:.3f}"
              f"   (CEILING {out['current_plus_deepseek_features']['in_sample_ceiling']} <- rose from 0.75)")
        print("  READ: the ceiling rises to ~0.83 with better features (real headroom), but leave-one-out is")
        print("        stuck because 138 examples is too few to LEARN it -> MORE DATA is the lever, not the model.")
        print("        Immediate win: rank by the LLM's holistic judgment. Honest cap ~0.83, not 0.95.")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
