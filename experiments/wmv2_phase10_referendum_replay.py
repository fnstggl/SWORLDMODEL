"""Phase 10 (continuation) — 3rd REAL institution category: Swiss federal DIRECT DEMOCRACY (referenda).

Genuinely different from a representative legislature (Congress) and a court (SCOTUS): the People (and, for
constitutional matters, the Cantons) decide directly. Data: cached real Swiss federal referendum records
(experiments/results/exp074/referenda.json — 704 votes, 1848–2026, each tagged with its institutional legal
FORM). All dimensions are leakage-safe: the legal form + date are known BEFORE the vote; nothing after the
vote is a model input.

  (A) INSTITUTIONAL-FORM REGULARITY (reconstruction): the legal form sets the bar. Art. 140 mandatory
      referendum and Art. 139 popular initiative require a DOUBLE majority (People AND Cantons); Art. 141
      optional referendum needs a single majority of the People. Reconstruct the outcome regularity by form —
      popular initiatives pass ~10% (the famous double-majority + establishment-opposition regularity),
      mandatory referenda ~75%. The institutional form is strongly outcome-structuring.

  (B) OUT-OF-SAMPLE PREDICTION: train the earlier era, predict the held-out later era from the legal form via
      the REAL Phase-3 Dirichlet posterior; score accuracy + Brier against the base rate. Leakage-safe.

  (C) NON-VOTING TIMING dimension: the fixed quarterly voting-Sunday cadence — the modern institution holds
      referenda on ~4 official dates per year. Reconstruct distinct voting dates/year — a real institutional
      SCHEDULING constraint (non-voting), the required non-vote dimension for this category.

LIMITATION (preserved, honest): this cached file has the legal form + outcome but NOT per-canton vote shares,
so the double-majority is reconstructed as an outcome REGULARITY by form — it is NOT executed on canton counts
like the Senate roll-call threshold engine. A full double-majority EXECUTION replay needs the canton-level
Swissvotes data (remaining work, recorded in the failures artifact).

Run: PYTHONPATH=. python -m experiments.wmv2_phase10_referendum_replay
Writes experiments/results/phase10/wmv2_phase10_referendum_replay.json
"""
from __future__ import annotations

import collections
import json

from swm.world_model_v2.phase3_posterior import infer_compositional_posterior

SRC = "experiments/results/exp074/referenda.json"
OUT = "experiments/results/phase10/wmv2_phase10_referendum_replay.json"

# Swissvotes legal-form (rechtsform) codes → the institutional rule (verified against the Swiss Federal
# Constitution: Art. 139/140 require the double majority of People AND Cantons; Art. 141 a single People's
# majority). Core-checked: the type-3 (initiative) ~10% pass rate is the well-documented double-majority
# regularity, which cross-validates the coding.
FORM = {
    "1": ("mandatory_referendum", "double_majority", "Art. 140 — constitutional amendments/treaties: People + Cantons"),
    "2": ("optional_referendum", "single_majority", "Art. 141 — federal laws challenged by petition: People"),
    "3": ("popular_initiative", "double_majority", "Art. 139 — citizen constitutional initiative: People + Cantons"),
    "4": ("counter_proposal", "double_majority", "direct counter-proposal to an initiative: People + Cantons"),
}
TRAIN_MAX_YEAR = 1990          # train ≤ 1990, predict > 1990 (out-of-sample, leakage-safe by construction)


def _load():
    return json.load(open(SRC))


def _regularity(rows):
    by = collections.defaultdict(lambda: [0, 0])          # form -> [fail, pass]
    for r in rows:
        f = str(r.get("type"))
        by[f][int(r.get("accepted", 0))] += 1
    out = {}
    for f, (no, yes) in sorted(by.items()):
        n = no + yes
        name, rule, cite = FORM.get(f, (f"form_{f}", "unknown", ""))
        out[name] = {"form_code": f, "majority_rule": rule, "constitutional_basis": cite,
                     "n": n, "accept_rate": round(yes / max(1, n), 4)}
    return out


def _predict_oos(rows):
    """Out-of-sample: fit per-form pass propensity via the REAL Phase-3 posterior on the TRAIN era, predict the
    held-out later era. Leakage-safe (form known before the vote; the target vote's outcome only scores)."""
    train = [r for r in rows if str(r.get("year", "")).isdigit() and int(r["year"]) <= TRAIN_MAX_YEAR]
    test = [r for r in rows if str(r.get("year", "")).isdigit() and int(r["year"]) > TRAIN_MAX_YEAR]
    tc = collections.defaultdict(lambda: [0, 0])
    for r in train:
        tc[str(r.get("type"))][int(r.get("accepted", 0))] += 1
    post = {}
    for f, (no, yes) in tc.items():
        res = infer_compositional_posterior(
            ["pass", "fail"], [1.0, 1.0],
            [{"counts": {"pass": yes, "fail": no}, "reliability": 1.0, "source": f"train_form_{f}",
              "method": "referendum_tally"}], n_particles=200, seed=23)
        post[f] = float(res.posterior_mean[0])
    train_pass = sum(v[1] for v in tc.values())
    train_tot = sum(v[0] + v[1] for v in tc.values())
    global_p = train_pass / max(1, train_tot)
    probs, actual, correct = [], [], 0
    for r in test:
        p = post.get(str(r.get("type")), global_p)
        y = int(r.get("accepted", 0))
        probs.append(p)
        actual.append(y)
        correct += int((p >= 0.5) == bool(y))
    n = len(test)
    base = sum(actual) / max(1, n)
    brier = sum((p - a) ** 2 for p, a in zip(probs, actual)) / max(1, n)
    base_brier = sum((global_p - a) ** 2 for a in actual) / max(1, n)
    return {"train_n": len(train), "test_n": n, "train_max_year": TRAIN_MAX_YEAR,
            "accuracy": round(correct / max(1, n), 4), "brier": round(brier, 4),
            "always_pass_baseline_accuracy": round(max(base, 1 - base), 4),
            "predict_base_rate_brier": round(base_brier, 4),
            "brier_beats_base_rate": brier < base_brier,
            "fitted_form_propensity_train": {FORM.get(f, (f,))[0]: round(p, 3) for f, p in post.items()}}


def _timing(rows):
    """NON-VOTING dimension: the institutional voting-date cadence. Count distinct voting dates per year — the
    modern institution clusters referenda on ~4 fixed Sundays/year (a real scheduling constraint)."""
    per_year = collections.defaultdict(set)
    for r in rows:
        y = str(r.get("year", ""))
        d = str(r.get("date", ""))
        if y.isdigit() and d:
            per_year[int(y)].add(d)
    modern = {y: len(ds) for y, ds in per_year.items() if y >= 1990}
    dates_per_year = sorted(modern.values())
    n = len(dates_per_year)
    med = dates_per_year[n // 2] if n else None
    # fraction of modern years with the characteristic ≤4 official voting dates
    within_4 = sum(1 for v in modern.values() if v <= 4) / max(1, len(modern))
    return {"modern_years": len(modern), "median_distinct_voting_dates_per_year": med,
            "max_distinct_dates_in_a_year": max(dates_per_year) if dates_per_year else None,
            "fraction_modern_years_within_4_official_dates": round(within_4, 4),
            "note": "the fixed quarterly voting-Sunday cadence — a non-voting institutional scheduling rule"}


def replay():
    rows = _load()
    reg = _regularity(rows)
    pred = _predict_oos(rows)
    tim = _timing(rows)
    # the load-bearing institutional signal: form predicts outcome far above the base rate (initiatives fail)
    init = reg.get("popular_initiative", {}).get("accept_rate")
    mand = reg.get("mandatory_referendum", {}).get("accept_rate")
    return {
        "n_referenda": len(rows), "source": "Swissvotes/BFS-derived (cached exp074/referenda.json)",
        "year_range": [min(int(r["year"]) for r in rows if str(r.get("year", "")).isdigit()),
                       max(int(r["year"]) for r in rows if str(r.get("year", "")).isdigit())],
        "form_regularity_reconstruction": reg,
        "institutional_signal": {
            "initiative_accept_rate": init, "mandatory_accept_rate": mand,
            "double_majority_regularity_holds": (init is not None and init < 0.2),
            "fact": "popular initiatives (double majority + establishment opposition) pass ~10%; mandatory "
                    "referenda ~75% — the legal FORM is strongly outcome-structuring"},
        "out_of_sample_prediction": pred,
        "timing_dimension_non_voting": tim,
        "limitation": "cached file lacks per-canton vote shares → the double majority is reconstructed as an "
                      "outcome REGULARITY by legal form, NOT executed on canton counts (full execution replay "
                      "needs canton-level Swissvotes data; preserved as remaining work).",
    }


def main():
    res = replay()
    doc = {"_meta": {"harness": "experiments/wmv2_phase10_referendum_replay.py",
                     "category": "direct_democracy", "source": res["source"],
                     "leakage": "legal form + date known before the vote; no post-vote inputs",
                     "note": "3rd real institution category (beyond Congress + court); includes a NON-VOTING "
                             "timing/cadence dimension and an out-of-sample predictive dimension"},
           "replay": res}
    json.dump(doc, open(OUT, "w"), indent=1, default=str)
    reg, pred, tim = res["form_regularity_reconstruction"], res["out_of_sample_prediction"], res["timing_dimension_non_voting"]
    print(f"=== Phase 10 referendum replay (Swiss direct democracy, n={res['n_referenda']}, "
          f"{res['year_range'][0]}–{res['year_range'][1]}) ===")
    for name, d in reg.items():
        print(f"  FORM {name} [{d['majority_rule']}]: n={d['n']} accept_rate={d['accept_rate']}")
    print(f"  SIGNAL: initiative {res['institutional_signal']['initiative_accept_rate']} vs mandatory "
          f"{res['institutional_signal']['mandatory_accept_rate']} — double-majority regularity holds: "
          f"{res['institutional_signal']['double_majority_regularity_holds']}")
    print(f"  OUT-OF-SAMPLE (train≤{pred['train_max_year']} → test>{pred['train_max_year']}, n={pred['test_n']}): "
          f"acc {pred['accuracy']} Brier {pred['brier']} (base-rate Brier {pred['predict_base_rate_brier']}, "
          f"beats: {pred['brier_beats_base_rate']})")
    print(f"  TIMING (non-voting): median {tim['median_distinct_voting_dates_per_year']} distinct voting dates/"
          f"year; {tim['fraction_modern_years_within_4_official_dates']} of modern years within 4 official dates")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
