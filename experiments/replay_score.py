"""Temporal Replay Laboratory — SCORER (the ONLY component that reads the sealed resolution store).

Run AFTER the forecaster froze its rows (REPLAY_SCORER=1 required). Verifies every row's freeze hash
(post-hoc edits are detectable), classifies each row's leakage (probes + arm), scores Brier / log-loss
against the sealed outcomes, and reports:

  * clean headline  — blinded arm, rows classified low_leakage_risk / uncertain excluded, CLUSTERED by
    event family (correlated contracts are one cluster; bootstrap resamples clusters, not rows);
  * diagnostic arm  — cutoff_prompted_unblinded, always contamination_not_excluded, never mixed in;
  * baselines       — base-rate 0.5 and the always-affirmative-lean; a market baseline is reported null
    (no defensible as-of market snapshot is stored — fabricating one would poison the comparison).

Usage:  REPLAY_SCORER=1 PYTHONPATH=. python experiments/replay_score.py
"""
from __future__ import annotations
import json
import math
import os
import random
from pathlib import Path

from swm.replay.vault import sealed_resolutions, freeze_hash
from swm.replay.probes import classify_row

ART = Path("experiments/results/replay/forecasts.json")
OUT = Path("experiments/results/replay/scores.json")


def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = min(1 - 1e-6, max(1e-6, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _cluster_bootstrap(rows, stat, n_boot=4000, seed=99):
    clusters = {}
    for r in rows:
        clusters.setdefault(r["cluster"], []).append(r)
    keys = sorted(clusters)
    if not keys:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = []
        for _ in keys:
            sample.extend(clusters[rng.choice(keys)])
        vals.append(stat(sample))
    vals.sort()
    return {"lo95": round(vals[int(0.025 * len(vals))], 4),
            "hi95": round(vals[int(0.975 * len(vals))], 4)}


def _name_only_correct(probe, outcome, note) -> bool | None:
    """Did the name-only probe state the ACTUAL resolution? Lexical check against the sealed note's
    direction: probe must claim knowledge AND its stated resolution must agree with the outcome side."""
    if not isinstance(probe, dict) or not probe.get("known"):
        return False
    stated = str(probe.get("resolution") or "").lower()
    if not stated:
        return None
    yes_words = ("won", "passed", "occurred", "happened", "released", "succeeded", "cut", "agreed",
                 "reached", "caught", "exceeded", "fell", "launched", "yes")
    no_words = ("did not", "didn't", "no ", "failed", "lost", "withdrew", "held steady", "not ", "never")
    said_yes = any(w in stated for w in yes_words) and not any(w in stated for w in no_words)
    said_no = any(w in stated for w in no_words)
    if said_yes and outcome == 1:
        return True
    if said_no and outcome == 0:
        return True
    if said_yes or said_no:
        return False
    return None


def main():
    if os.environ.get("REPLAY_SCORER") != "1":
        raise SystemExit("scorer requires REPLAY_SCORER=1 (structural separation from the forecaster)")
    payload = json.loads(ART.read_text())
    resolutions = sealed_resolutions()["resolutions"]
    scored, tampered = [], []
    for r in payload["rows"]:
        fh = r.get("freeze_hash")
        if fh != freeze_hash({k: v for k, v in r.items() if k != "freeze_hash"}):
            tampered.append(r.get("event_id"))
            continue
        if r.get("error") or r.get("p_yes") is None:
            continue
        seal = resolutions.get(r["event_id"])
        if seal is None:
            continue
        y = int(seal["outcome"])
        noc = _name_only_correct((r.get("probes") or {}).get("name_only"), y,
                                 seal.get("resolution_note", ""))
        leak = classify_row(r.get("probes") or {}, arm=r["arm"], name_only_correct=noc)
        p = float(r["p_yes"])
        scored.append({**{k: r[k] for k in ("event_id", "cluster", "cutoff", "arm")},
                       "p_yes": p, "outcome": y, "brier": round(_brier(p, y), 4),
                       "logloss": round(_logloss(p, y), 4), "leakage_class": leak,
                       "name_only_correct": noc})

    def _arm_stats(rows, label):
        if not rows:
            return {"label": label, "n": 0}
        mb = lambda rs: sum(x["brier"] for x in rs) / len(rs)                      # noqa: E731
        ml = lambda rs: sum(x["logloss"] for x in rs) / len(rs)                    # noqa: E731
        base = sum(_brier(0.5, x["outcome"]) for x in rows) / len(rows)
        acc = sum(1 for x in rows if (x["p_yes"] >= 0.5) == (x["outcome"] == 1)) / len(rows)
        return {"label": label, "n_rows": len(rows), "n_clusters": len({x["cluster"] for x in rows}),
                "brier": round(mb(rows), 4), "brier_ci_cluster": _cluster_bootstrap(rows, mb),
                "logloss": round(ml(rows), 4), "directional_accuracy": round(acc, 3),
                "baseline_brier_p05": round(base, 4),
                "beats_base_rate": mb(rows) < base}

    blinded = [x for x in scored if x["arm"] == "blinded_current_llm"]
    clean = [x for x in blinded if x["leakage_class"] == "low_leakage_risk"]
    susceptible = [x for x in blinded if x["leakage_class"] in
                   ("contamination_susceptible", "known_contaminated")]
    diag = [x for x in scored if x["arm"] == "cutoff_prompted_unblinded"]
    leak_census = {}
    for x in blinded:
        leak_census[x["leakage_class"]] = leak_census.get(x["leakage_class"], 0) + 1
    report = {
        "headline_clean_blinded": _arm_stats(clean, "blinded, low_leakage_risk only"),
        "blinded_all_rows_diagnostic": _arm_stats(blinded, "blinded, all rows (incl. susceptible)"),
        "contamination_susceptible_excluded": _arm_stats(susceptible, "blinded but recognized/known"),
        "cutoff_prompted_unblinded_DIAGNOSTIC_ONLY": {
            **_arm_stats(diag, "current product path"),
            "contamination": "NOT excluded by construction — never a clean historical accuracy claim"},
        "leakage_census_blinded": leak_census,
        "market_baseline": None,
        "market_baseline_note": "no defensible as-of market snapshot stored; null rather than fabricated",
        "tampered_rows": tampered,
        "standard": "contamination-safe rows only enter the headline; arm-1 pre-cutoff checkpoint "
                    "unavailable for this backend (recorded); process-level sealing (documented limitation)"}
    OUT.write_text(json.dumps({"rows": scored, "report": report}, indent=1))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
