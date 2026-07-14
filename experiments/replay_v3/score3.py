"""Post-snapshot benchmark SCORER (v3). REPLAY_SCORER=1 only; locked split opens ONCE.

Governance: verify freeze hashes → classify leakage → fit calibrators on CALIBRATION worlds only →
select on VALIDATION worlds only (identity wins ties) → freeze → open LOCKED once (access log; second
open refuses) → score V2 raw/calibrated + all baselines with world-clustered bootstrap CIs, Brier,
log loss, AUROC, directional accuracy, ECE.
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

VAULT = Path("experiments/replay_vault_v3")
OUT = Path("experiments/results/replay_v3")
LOCK_LOG = OUT / "locked_access_log.json"
from experiments.replay_v2.score import CALIBRATORS, _brier, _logloss, _cluster_ci  # noqa: E402


def _rows(split):
    art = OUT / f"forecasts_{split}.jsonl"
    if not art.exists():
        return [], []
    rows, tampered = [], []
    for line in art.read_text().splitlines():
        r = json.loads(line)
        if r.get("freeze_hash") != freeze_hash({k: v for k, v in r.items() if k != "freeze_hash"}):
            tampered.append((r["event_id"], r["cutoff"]))
            continue
        rows.append(r)
    return rows, tampered


def _auroc(pairs):
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if pp > pn else 0.5 if pp == pn else 0.0) for pp in pos for pn in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def _ece(pairs, bins=10):
    if not pairs:
        return None
    tot, e = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        grp = [(p, y) for p, y in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if grp:
            conf = sum(p for p, _ in grp) / len(grp)
            acc = sum(y for _, y in grp) / len(grp)
            e += len(grp) / tot * abs(conf - acc)
    return round(e, 4)


def _stats(rows, key, label):
    pairs = [(r[key], r["outcome"]) for r in rows if isinstance(r.get(key), (int, float))]
    if not pairs:
        return {"label": label, "n": 0}
    subset = [dict(r, _p=r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    mb = lambda rs: sum(_brier(x["_p"], x["outcome"]) for x in rs) / len(rs)          # noqa: E731
    return {"label": label, "n_rows": len(pairs), "n_worlds": len({r["event_id"] for r in subset}),
            "brier": round(mb(subset), 4), "brier_ci_world_cluster": _cluster_ci(subset, mb),
            "logloss": round(sum(_logloss(p, y) for p, y in pairs) / len(pairs), 4),
            "auroc": _auroc(pairs), "ece": _ece(pairs),
            "directional_accuracy": round(sum((p >= 0.5) == (y == 1) for p, y in pairs) / len(pairs), 3)}


def _prep(rows, sealed):
    out = []
    for r in rows:
        if r.get("failure_reason") or r.get("p_yes") is None:
            continue
        seal = sealed.get(r["event_id"])
        if seal is None:
            continue
        y = int(seal["outcome"])
        probes = r.get("leakage_probes") or {}
        no = (probes.get("name_only") or {}).get("output") or {}
        noc = None
        if no.get("known"):
            stated = str(no.get("resolution") or "").lower()
            yw = any(w in stated for w in ("yes", "won", "passed", "occurred", "reached", "happened"))
            nw = any(w in stated for w in ("no ", "not ", "failed", "lost", "did not"))
            noc = (yw and y == 1) or (nw and y == 0) if (yw or nw) else None
        b = r.get("baselines") or {}
        ens = [x for x in (b.get("ensemble") or []) if isinstance(x, (int, float))]
        pan = [x for x in (b.get("panel") or []) if isinstance(x, (int, float))]
        out.append({"event_id": r["event_id"], "cluster": r["cluster"], "cutoff": r["cutoff"],
                    "outcome": y, "p_yes": float(r["p_yes"]),
                    "leakage_class": classify_row_v2(probes, arm="blinded", name_only_correct=noc),
                    "direct": b.get("direct"),
                    "ensemble": (sorted(ens)[len(ens) // 2] if ens else None),
                    "panel": (sum(pan) / len(pan) if pan else None),
                    "analogical": b.get("analogical"),
                    "market": (r.get("market_snapshot") or {}).get("price")})
    return out


def main(open_locked=False):
    if os.environ.get("REPLAY_SCORER") != "1":
        raise SystemExit("REPLAY_SCORER=1 required")
    sealed = json.loads((VAULT / "SEALED_resolutions_v3.json").read_text())["resolutions"]
    report = {"tier": "B_provider_attested_post_cutoff_blinded"}
    cal_rows, t1 = _rows("calibration")
    val_rows, t2 = _rows("validation")
    cal = _prep(cal_rows, sealed)
    val = _prep(val_rows, sealed)
    report["tampered"] = t1 + t2
    report["leakage_census_cal_val"] = {}
    for x in cal + val:
        report["leakage_census_cal_val"][x["leakage_class"]] = \
            report["leakage_census_cal_val"].get(x["leakage_class"], 0) + 1
    clean = [x for x in cal if x["leakage_class"] in ("clean_blinded", "low_leakage_risk")]
    vclean = [x for x in val if x["leakage_class"] in ("clean_blinded", "low_leakage_risk")]
    fitted, chosen = {}, "identity"
    if len(clean) >= 20 and len(vclean) >= 10:
        pairs = [(x["p_yes"], x["outcome"]) for x in clean]
        fitted = {n: f(pairs) for n, f in CALIBRATORS.items()}
        vs = {n: sum(_brier(f(x["p_yes"]), x["outcome"]) for x in vclean) / len(vclean)
              for n, f in fitted.items()}
        chosen = min(vs, key=vs.get)
        if vs[chosen] >= vs["identity"] - 1e-9:
            chosen = "identity"
        report["calibration"] = {"validation_brier": {k: round(v, 4) for k, v in vs.items()},
                                 "selected": chosen, "fit_n": len(clean), "val_n": len(vclean)}
    else:
        report["calibration"] = {"selected": "identity",
                                 "reason": f"insufficient clean rows (cal={len(clean)}, val={len(vclean)})"}
        fitted = {"identity": lambda p: p}
    for name, rows, in (("calibration", cal), ("validation", val)):
        report[f"{name}_v2_raw"] = _stats([dict(x, _k=x["p_yes"]) for x in rows], "p_yes", "V2 raw")
    if open_locked:
        if LOCK_LOG.exists():
            report["locked_test"] = "REFUSED: already opened once"
        else:
            lrows, t3 = _rows("locked_test")
            locked = _prep(lrows, sealed)
            LOCK_LOG.write_text(json.dumps({"opened_at": time.time(), "n_rows": len(locked),
                                            "calibrator": chosen}, indent=1))
            lc = [x for x in locked if x["leakage_class"] in ("clean_blinded", "low_leakage_risk")]
            f = fitted.get(chosen, lambda p: p)
            for x in lc:
                x["p_cal"] = f(x["p_yes"])
            rate = sum(x["outcome"] for x in clean) / len(clean) if clean else 0.5
            for x in lc:
                x["p05"], x["base_rate"] = 0.5, rate
            rep = {"n_clean_rows": len(lc), "n_worlds": len({x["event_id"] for x in lc}),
                   "tampered_locked": t3,
                   "leakage_census_locked": {}}
            for x in locked:
                rep["leakage_census_locked"][x["leakage_class"]] = \
                    rep["leakage_census_locked"].get(x["leakage_class"], 0) + 1
            for key, label in (("p_yes", "V2 raw"), ("p_cal", f"V2 calibrated ({chosen})"),
                               ("direct", "direct single-call"), ("ensemble", "call-matched ensemble"),
                               ("panel", "observer panel"), ("analogical", "analogical retrieval"),
                               ("market", "market midpoint @cutoff"), ("p05", "constant 0.5"),
                               ("base_rate", "calibration base rate")):
                rep[key] = _stats(lc, key, label)
            report["locked_test"] = rep
    (OUT / "scores_v3.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str)[:4000])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-locked", action="store_true")
    main(open_locked=ap.parse_args().open_locked)
