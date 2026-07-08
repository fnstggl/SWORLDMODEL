"""EXP-086 — How close can an INFERRED variable get to a MEASURED one? (the three pillars, scored)

The core question of the project: in the real world we won't always have 7,000 senate votes to MEASURE a
variable. So can we INFER it well enough that the world model still wins? EXP-085 gives us a perfect
yardstick — it already established, for senator ideology fed into a committee-vote world model:
    party-only floor  ~0.902   |   MEASURED-ideology ceiling  ~0.916
This hides every senator's voting record and asks: inferring ideology from only what we'd know about a
stranger (name, state, party, era + the LLM's trained knowledge), how much of that 0.902->0.916 gap can we
close, and WHICH pillar closes it?

The arm-ladder (each adds one pillar; all scored through the IDENTICAL EXP-085 world model):
  measured (ceiling)   — the real prior-congress ideal point (EXP-085's 0.916).
  party_base (floor)   — Pillar 2 alone: every senator = their party's MEASURED mean ideology. No individuation.
  llm_classonly        — LLM given party+state+era but NOT the name: pure reference-class reasoning, no recall.
  llm_named_cold       — LLM given the name; raw ensemble mean. Individuation, but uncalibrated + unanchored.
  llm_named_anchored   — + Pillar 3 shrink toward the measured party base rate (kill over-individuation).
  llm_named_calibrated — + Pillar 3 calibration map (LLM->truth, learned on TRAIN senators). The full stack.

Leakage discipline: ideology is a CURRENT-STATE trait, so using the LLM's world knowledge / public info to
infer it is legitimate (the no-cheat rule is about the OUTCOME — the future votes — which stay hidden). The
calibration map is fit on TRAIN-era senators and applied to TEST-era senators (different people). Ensemble
K sampled at temperature. LLM cached & resumable (DEEPSEEK_API_KEY from env only).
Run: DEEPSEEK_API_KEY=... python -m experiments.exp086_inference_quality
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

from swm.transition.readout import LogisticReadout
from swm.variables.grounded_inference import (ensemble_infer, fit_calibration, apply_calibration,
                                              reference_prior, grounded_estimate)
from experiments.exp085_committee_vote_world import _split

BILLS = "experiments/results/exp085/senate_bills.json"
INFO = "experiments/results/exp086/senator_info.json"
LLM = "experiments/results/exp086/llm_ideology.json"
RESULT = "experiments/results/exp086_inference_quality.json"
BASE = "https://voteview.com/static/data/out"

TRAIN_CONG = set(range(106, 114))       # senators from these congresses calibrate the LLM->truth map
TEST_CONG = set(range(114, 119))        # ...and we score the world model on these congresses' bills
K = 3                                   # ensemble samples per senator
_NUM = re.compile(r"-?\d+\.?\d*")


def _fetch_info():
    """icpsr -> {name, state, party, congs:[...], measured: mean nominate_dim1}. From VoteView members CSVs."""
    if Path(INFO).exists():
        return json.loads(Path(INFO).read_text())
    Path(INFO).parent.mkdir(parents=True, exist_ok=True)
    info = {}
    for c in sorted(TRAIN_CONG | TEST_CONG):
        try:
            with urllib.request.urlopen(f"{BASE}/members/S{c}_members.csv", timeout=90) as r:
                rows = list(csv.DictReader(io.TextIOWrapper(r, encoding="latin-1")))
        except Exception as e:
            print(f"  members S{c} failed: {str(e)[:50]}")
            continue
        for m in rows:
            try:
                x = float(m["nominate_dim1"])
            except (ValueError, KeyError):
                continue
            d = info.setdefault(m["icpsr"], {"name": m["bioname"], "state": m["state_abbrev"],
                                             "party": m["party_code"], "congs": [], "xs": []})
            d["congs"].append(c)
            d["xs"].append(x)
    for d in info.values():
        d["measured"] = round(fmean(d["xs"]), 4)
        d["name"] = d["name"].title() if d["name"].isupper() else d["name"]
    Path(INFO).write_text(json.dumps(info))
    print(f"  senator info: {len(info)} senators")
    return info


def _party_name(code):
    return {"100": "Democrat", "200": "Republican"}.get(str(code), "Independent")


def _parse_ideo(raw):
    m = _NUM.findall(raw.replace(",", " "))
    if not m:
        return None
    v = float(m[0])
    return max(-1.5, min(1.5, v))


def _infer_llm(info):
    """Ensemble LLM ideology for each senator, WITH and WITHOUT the name. Cached & resumable."""
    Path(LLM).parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(Path(LLM).read_text()) if Path(LLM).exists() else {}
    todo = [i for i in info if i not in cache or "named" not in cache.get(i, {})]
    if todo and os.environ.get("DEEPSEEK_API_KEY"):
        from swm.api.deepseek_backend import deepseek_chat_fn
        fn = deepseek_chat_fn(system="You are a political scientist estimating ideology on the DW-NOMINATE "
                              "scale. Answer with ONLY a single number.", temperature=0.7, max_tokens=12)
        for j, icpsr in enumerate(todo):
            d = info[icpsr]
            party, state = _party_name(d["party"]), d["state"]
            era = f"{min(d['congs'])*2+1787}-{max(d['congs'])*2+1789}"
            named = (f"On the DW-NOMINATE first-dimension scale from -1.0 (very liberal) to +1.0 (very "
                     f"conservative), estimate the ideology of US Senator {d['name']} ({party}-{state}), "
                     f"serving around {era}. Reply with ONLY the number.")
            classonly = (f"On the DW-NOMINATE first-dimension scale from -1.0 (very liberal) to +1.0 (very "
                         f"conservative), estimate the ideology of a typical {party} US Senator from {state} "
                         f"serving around {era}. Reply with ONLY the number.")
            en = ensemble_infer(fn, named, _parse_ideo, k=K)
            ec = ensemble_infer(fn, classonly, _parse_ideo, k=1)
            if en is None:
                continue
            cache[icpsr] = {"named": en[0], "named_spread": en[1], "classonly": (ec[0] if ec else None)}
            if j % 20 == 0:
                Path(LLM).write_text(json.dumps(cache))
                print(f"  inferred {j}/{len(todo)}")
        Path(LLM).write_text(json.dumps(cache))
    return cache


def _score(bills, xof, rng_seed=0):
    """EXP-085 world model with member ideology supplied by xof(icpsr, fallback_x). Held-out 20% accuracy."""
    import random
    rng = random.Random(rng_seed)
    hit = n = 0
    for b in bills:
        ms = b["members"]
        if len(ms) < 40:
            continue
        obs, hold = _split(ms, rng)
        if len(set(m["vote"] for m in obs)) < 2:
            continue
        model = LogisticReadout(l2=0.05, epochs=200).fit(
            [[xof(m["icpsr"], m["x"])] for m in obs], [m["vote"] for m in obs])
        for m in hold:
            p = model.predict_proba([xof(m["icpsr"], m["x"])])
            hit += int((p >= 0.5) == (m["vote"] == 1))
            n += 1
    return hit / n if n else float("nan")


def run():
    info = _fetch_info()
    llm = _infer_llm(info)
    bills = json.loads(Path(BILLS).read_text())
    test_bills = [b for b in bills if b["congress"] in TEST_CONG]

    # reference-class base rates (Pillar 2), MEASURED from TRAIN senators only (no test leakage)
    byparty = defaultdict(list)
    for i, d in info.items():
        if set(d["congs"]) & TRAIN_CONG:
            byparty[d["party"]].append(d["measured"])
    party_prior = {p: (fmean(xs), max(0.15, pstdev(xs) if len(xs) > 1 else 0.3)) for p, xs in byparty.items()}
    global_prior = (fmean([d["measured"] for d in info.values()]), 0.4)

    # Pillar 3 calibration: fit LLM(named) -> measured on TRAIN senators; apply to everyone
    tr = [(llm[i]["named"], info[i]["measured"]) for i in info
          if i in llm and set(info[i]["congs"]) & TRAIN_CONG and llm[i].get("named") is not None]
    cal = fit_calibration([r for r, _ in tr], [t for _, t in tr])

    def prior_of(icpsr):
        return reference_prior(party_prior, info[icpsr]["party"], global_prior)

    # ---- build the per-senator ideology for each arm ----
    def measured(icpsr, fx):
        return fx                                          # the real prior-congress ideal point (ceiling)

    def party_base(icpsr, fx):
        return prior_of(icpsr)[0]

    def llm_classonly(icpsr, fx):
        v = llm.get(icpsr, {}).get("classonly")
        return v if v is not None else prior_of(icpsr)[0]

    def llm_named_cold(icpsr, fx):
        v = llm.get(icpsr, {}).get("named")
        return v if v is not None else prior_of(icpsr)[0]

    def llm_named_anchored(icpsr, fx):
        r = llm.get(icpsr, {})
        if r.get("named") is None:
            return prior_of(icpsr)[0]
        est = grounded_estimate(llm_mean=r["named"], llm_spread=r.get("named_spread"), cal=None,
                                class_prior=prior_of(icpsr))
        return est.value

    def llm_named_calibrated(icpsr, fx):
        r = llm.get(icpsr, {})
        if r.get("named") is None:
            return prior_of(icpsr)[0]
        est = grounded_estimate(llm_mean=r["named"], llm_spread=r.get("named_spread"), cal=cal,
                                class_prior=prior_of(icpsr))
        return est.value

    arms = [("measured (ceiling)", measured), ("party_base (floor)", party_base),
            ("llm_classonly", llm_classonly), ("llm_named_cold", llm_named_cold),
            ("llm_named_anchored", llm_named_anchored), ("llm_named_calibrated", llm_named_calibrated)]
    scores = {name: round(_score(test_bills, fn), 4) for name, fn in arms}

    floor, ceil = scores["party_base (floor)"], scores["measured (ceiling)"]
    gap = ceil - floor
    closed = {name: (round((s - floor) / gap, 3) if gap > 1e-9 else None)
              for name, s in scores.items()}

    # how well does each arm's inferred ideology correlate with truth? (the upstream quality signal)
    def corr(vals_fn):
        pairs = [(vals_fn(i, info[i]["measured"]), info[i]["measured"]) for i in info
                 if set(info[i]["congs"]) & TEST_CONG and i in llm]
        if len(pairs) < 3:
            return None
        xs, ys = [p for p, _ in pairs], [t for _, t in pairs]
        mx, my = fmean(xs), fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        return round(num / den, 3) if den > 1e-9 else None

    ideo_corr = {name: corr(fn) for name, fn in arms if name != "measured (ceiling)"}

    out = {"experiment": "Inference quality — how close can an inferred variable get to a measured one?",
           "yardstick": "EXP-085 committee-vote world model; held-out 20% of members per bill",
           "n_test_bills": len(test_bills), "n_senators": len(info),
           "calibration_map_named->truth": {"a": round(cal[0], 3), "b": round(cal[1], 3),
                                            "rmse": round(cal[2], 3) if cal[2] else None},
           "accuracy_by_arm": scores, "fraction_of_gap_closed": closed,
           "inferred_ideology_corr_with_truth": ideo_corr}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-086  how close can an INFERRED variable get to a MEASURED one? (senator ideology, EXP-085 yardstick)")
    print(f"  {len(test_bills)} test-congress bills | {len(info)} senators | calibration map named->truth: "
          f"a={cal[0]:.2f} b={cal[1]:.2f} rmse={cal[2]:.3f}" if cal[2] else "")
    print(f"  {'arm':24s} {'vote acc':>9s} {'gap closed':>11s} {'ideo r':>8s}")
    for name, _ in arms:
        gc = closed[name]
        r = ideo_corr.get(name)
        print(f"  {name:24s} {scores[name]:9.4f} {('' if gc is None else f'{gc*100:5.0f}%'):>11s} "
              f"{('' if r is None else f'{r:.2f}'):>8s}")
    print(f"  READ: floor(party) {floor:.4f} -> ceiling(measured) {ceil:.4f}; the pillar stack closes "
          f"{(closed['llm_named_calibrated'] or 0)*100:.0f}% of the gap with NO ground-truth votes for the target.")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
