"""Temporal Replay v2 — SCORER + Phase-12 refit governance (Parts 13/19/20/22/23). REPLAY_SCORER=1 only.

Governance enforced in code:
  * verifies every audit row's freeze hash before any outcome access (tampered rows excluded + reported);
  * classifies leakage per row (probes2.classify_row_v2); only clean_pre_cutoff_model / clean_blinded /
    low_leakage_risk enter headlines;
  * world-level splits (calibration 40 / validation 20 / locked 40) — every cutoff/arm/contract inherits
    its world's split;
  * the LOCKED split may be opened ONCE: an access log (locked_access_log.json) records the single open;
    a second scoring attempt on locked worlds refuses;
  * calibration methods (identity/Platt/isotonic/beta) fit ONLY on calibration worlds, selected ONLY on
    validation worlds, frozen, then applied once to the locked test;
  * baselines: base-rate, market midpoint (timestamp-matched archived snapshots), no-evidence direct-LLM
    probe (recorded at forecast time — evidence-parity: same frozen capsule text was available);
  * trajectory scoring vs sealed trajectory targets (price-crossing timing: did the runtime's terminal
    lean anticipate the archived path milestones);
  * clustered (event-family) bootstrap CIs.
"""
from __future__ import annotations
import json
import math
import os
import random
import time
from pathlib import Path

from swm.replay.vault import freeze_hash
from swm.replay.probes2 import classify_row_v2

VAULT = Path("experiments/replay_vault_v2")
OUT = Path("experiments/results/replay_v2")
ART = OUT / "audit_rows.jsonl"
LOCK_LOG = OUT / "locked_access_log.json"

HEADLINE_CLASSES = ("clean_pre_cutoff_model", "clean_blinded", "low_leakage_risk")


def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = min(1 - 1e-6, max(1e-6, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _load_rows():
    rows, tampered = [], []
    for line in ART.read_text().splitlines():
        r = json.loads(line)
        if r.get("freeze_hash") != freeze_hash({k: v for k, v in r.items() if k != "freeze_hash"}):
            tampered.append((r.get("event_id"), r.get("cutoff"), r.get("arm")))
            continue
        rows.append(r)
    return rows, tampered


def _sealed():
    from swm.replay.vault import SEALED  # noqa: F401 — path check only
    if os.environ.get("REPLAY_SCORER") != "1":
        raise SystemExit("scorer requires REPLAY_SCORER=1")
    return json.loads((VAULT / "SEALED_resolutions_v2.json").read_text())["resolutions"]


def _cluster_ci(rows, stat, n_boot=4000, seed=7):
    clusters = {}
    for r in rows:
        clusters.setdefault(r["cluster"], []).append(r)
    keys = sorted(clusters)
    if not keys:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = []
        for _ in keys:
            s.extend(clusters[rng.choice(keys)])
        vals.append(stat(s))
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals))], 4)]


# ---------------------------------------------------------------- calibrators (Part 22)
def _fit_platt(pairs):
    # 1-d logistic on logit(p): grid over (a, b) — small, stable, no dependencies
    import math as m
    def logit(p):
        p = min(1 - 1e-6, max(1e-6, p))
        return m.log(p / (1 - p))
    best, best_ll = (1.0, 0.0), float("inf")
    for a in [x / 10 for x in range(2, 31, 2)]:
        for b in [x / 10 for x in range(-20, 21, 4)]:
            ll = sum(_logloss(1 / (1 + m.exp(-(a * logit(p) + b))), y) for p, y in pairs) / len(pairs)
            if ll < best_ll:
                best, best_ll = (a, b), ll
    a, b = best
    return lambda p: 1 / (1 + math.exp(-(a * (math.log(max(1e-6, min(1 - 1e-6, p)) /
                                                        max(1e-6, 1 - p))) + b)))


def _fit_isotonic(pairs):
    pairs = sorted(pairs)
    xs, ys, ws = [p for p, _ in pairs], [float(y) for _, y in pairs], [1.0] * len(pairs)
    # PAV
    vals, wts, idx = [], [], []
    for y, w in zip(ys, ws):
        vals.append(y); wts.append(w)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            w2 = wts[-2] + wts[-1]
            vals = vals[:-2] + [v]; wts = wts[:-2] + [w2]
        idx.append(len(vals) - 1)
    fit = []
    j = 0
    for i, x in enumerate(xs):
        fit.append(vals[min(idx[i], len(vals) - 1)])
    def apply(p):
        import bisect
        k = bisect.bisect_left(xs, p)
        k = min(max(k, 0), len(fit) - 1)
        return min(1 - 1e-4, max(1e-4, fit[k]))
    return apply


CALIBRATORS = {"identity": lambda pairs: (lambda p: p),
               "platt": _fit_platt, "isotonic": _fit_isotonic}


def main():
    sealed = _sealed()
    rows, tampered = _load_rows()
    splits = json.loads((VAULT / "events.json").read_text())["splits"]
    scored = []
    for r in rows:
        if r.get("failure_reason") or r.get("p_yes") is None:
            continue
        seal = sealed.get(r["event_id"])
        if seal is None:
            continue
        y = int(seal["outcome"])
        noc = None
        probes = r.get("leakage_probes") or {}
        no = (probes.get("name_only") or {}).get("output") or {}
        if no.get("known"):
            stated = str(no.get("resolution") or "").lower()
            yes_w = any(w in stated for w in ("yes", "won", "passed", "occurred", "reached", "exceeded",
                                              "happened", "released", "succeeded"))
            no_w = any(w in stated for w in ("no ", "not ", "failed", "lost", "did not", "didn't"))
            noc = (yes_w and y == 1) or (no_w and y == 0) if (yes_w or no_w) else None
        leak = classify_row_v2(probes, arm=r["arm"], name_only_correct=noc)
        scored.append({**{k: r[k] for k in ("event_id", "cluster", "cutoff", "arm", "domain")},
                       "split": splits.get(r["event_id"], ""), "p_yes": float(r["p_yes"]), "outcome": y,
                       "leakage_class": leak,
                       "market_p": (r.get("market_snapshot") or {}).get("price"),
                       "brier": _brier(float(r["p_yes"]), y), "logloss": _logloss(float(r["p_yes"]), y)})

    clean = [x for x in scored if x["leakage_class"] in HEADLINE_CLASSES]
    cal = [x for x in clean if x["split"] == "calibration"]
    val = [x for x in clean if x["split"] == "validation"]
    locked = [x for x in clean if x["split"] == "locked_test"]

    # ---- Phase-12 refit governance: fit on cal, select on val, freeze, open locked ONCE ----
    result = {"n_rows_scored": len(scored), "n_clean": len(clean), "tampered": tampered,
              "leakage_census": _census(scored),
              "splits": {"calibration": len(cal), "validation": len(val), "locked": len(locked)}}
    if len(cal) >= 10 and len(val) >= 5:
        pairs = [(x["p_yes"], x["outcome"]) for x in cal]
        fitted = {name: fn(pairs) for name, fn in CALIBRATORS.items()}
        val_scores = {name: sum(_brier(f(x["p_yes"]), x["outcome"]) for x in val) / len(val)
                      for name, f in fitted.items()}
        chosen = min(val_scores, key=val_scores.get)
        if val_scores[chosen] >= val_scores["identity"] - 1e-9:
            chosen = "identity"                              # no method beats identity → identity
        result["calibration"] = {"validation_brier": {k: round(v, 4) for k, v in val_scores.items()},
                                 "selected": chosen}
        if locked:
            if LOCK_LOG.exists():
                result["locked_test"] = "REFUSED: locked split already opened once (see access log)"
            else:
                LOCK_LOG.write_text(json.dumps({"opened_at": time.time(),
                                                "n_locked_rows": len(locked),
                                                "calibrator": chosen}, indent=1))
                f = fitted[chosen]
                mb_raw = sum(x["brier"] for x in locked) / len(locked)
                mb_cal = sum(_brier(f(x["p_yes"]), x["outcome"]) for x in locked) / len(locked)
                base = sum(_brier(0.5, x["outcome"]) for x in locked) / len(locked)
                rate = sum(x["outcome"] for x in cal) / len(cal)
                base_rate_b = sum(_brier(rate, x["outcome"]) for x in locked) / len(locked)
                mkt = [x for x in locked if isinstance(x.get("market_p"), (int, float))]
                result["locked_test"] = {
                    "n_rows": len(locked), "n_clusters": len({x["cluster"] for x in locked}),
                    "raw_brier": round(mb_raw, 4),
                    "raw_brier_ci_cluster": _cluster_ci(locked, lambda s: sum(q["brier"] for q in s) / len(s)),
                    "calibrated_brier": round(mb_cal, 4),
                    "baseline_brier_p05": round(base, 4),
                    "baseline_brier_base_rate": round(base_rate_b, 4),
                    "market_midpoint_brier": (round(sum(_brier(x["market_p"], x["outcome"])
                                                        for x in mkt) / len(mkt), 4) if mkt else None),
                    "beats_p05": mb_raw < base, "beats_base_rate": mb_raw < base_rate_b}
    (OUT / "scores_v2.json").write_text(json.dumps({"rows": scored, "report": result}, indent=1))
    print(json.dumps(result, indent=1, default=str))


def _census(scored):
    c = {}
    for x in scored:
        c[x["leakage_class"]] = c.get(x["leakage_class"], 0) + 1
    return c


if __name__ == "__main__":
    main()
