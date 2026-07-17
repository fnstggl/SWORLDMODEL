"""Isolated scorer (REPLAY_SCORER=1 required — the resolution store enforces it at import).

Verification before any outcome is read: runtime-freeze manifest, question-vault seal,
resolution-vault seal, forecast-ledger seal, per-row question-hash + temporal ordering +
qualification. Rotating-locked outcomes are opened exactly once (locked_access_log). Metrics per
split / causal scale / domain / horizon: Brier, log loss, AUROC, calibration slope+intercept,
ECE, accuracy@0.5, censoring-aware event-time CRPS, interval coverage, sharpness; paired
differences vs every baseline with event-cluster bootstrap CIs; capability-normalized skill
1 - WMv2_Brier / Direct_Brier.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from historical_backtests.framework.resolution_store import (read_resolutions, verify_seal,
                                                             VAULT_DIR)

ROOT = Path(__file__).resolve().parents[1]


from historical_backtests.framework.metrics import (_auroc, _cal_line, _ece, _ll,
                                                    crps_event_time, interval_cover)


def _brier_rows(rows, key):
    return [( float(r[key]), r["_y"]) for r in rows if isinstance(r.get(key), (int, float))]


def _score_block(rows) -> dict:
    pairs = _brier_rows(rows, "p_yes")
    if not pairs:
        return {"n": 0}
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    slope, intercept = _cal_line(pairs)
    crps = [r["_crps"] for r in rows if isinstance(r.get("_crps"), (int, float))]
    cov80 = [r["_cov80"] for r in rows if r.get("_cov80") is not None]
    cov50 = [r["_cov50"] for r in rows if r.get("_cov50") is not None]
    return {"n": len(pairs), "brier": round(brier, 4),
            "log_loss": round(sum(_ll(p, y) for p, y in pairs) / len(pairs), 4),
            "auroc": _auroc(pairs), "ece": _ece(pairs),
            "calibration_slope": slope, "calibration_intercept": intercept,
            "accuracy_at_half": round(sum(1 for p, y in pairs
                                          if (p >= 0.5) == (y == 1)) / len(pairs), 4),
            "event_time_crps": round(sum(crps) / len(crps), 4) if crps else None,
            "interval_coverage_80": round(sum(cov80) / len(cov80), 3) if cov80 else None,
            "interval_coverage_50": round(sum(cov50) / len(cov50), 3) if cov50 else None,
            "base_rate": round(sum(y for _, y in pairs) / len(pairs), 3)}


def _paired_bootstrap(rows, arm_key, n_boot=2000, seed=7):
    """Cluster bootstrap (by event cluster) of Brier(WMv2) - Brier(arm)."""
    byc = {}
    for r in rows:
        if isinstance(r.get("p_yes"), (int, float)) and isinstance(r.get(arm_key), (int, float)):
            byc.setdefault(r["_cluster"], []).append(r)
    clusters = list(byc.values())
    if len(clusters) < 3:
        return None
    def _diff(sample):
        a = [ (r["p_yes"] - r["_y"]) ** 2 for c in sample for r in c]
        b = [ (r[arm_key] - r["_y"]) ** 2 for c in sample for r in c]
        return sum(a) / len(a) - sum(b) / len(b)
    rng = random.Random(seed)
    obs = _diff(clusters)
    boots = sorted(_diff([rng.choice(clusters) for _ in clusters]) for _ in range(n_boot))
    return {"mean_diff": round(obs, 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4),
                     round(boots[int(0.975 * n_boot)], 4)],
            "n_rows": sum(len(c) for c in clusters), "n_clusters": len(clusters)}


def score(benchmark_id: str, run_id: str, *, splits=("calibration", "validation"),
          open_locked: bool = False) -> dict:
    bdir = ROOT / "benchmark_versions" / benchmark_id
    rdir = ROOT / "results" / benchmark_id / run_id
    verify_seal(bdir / "question_vault.json")
    ledger = rdir / "forecast_ledger.jsonl"
    seal = json.loads(ledger.with_suffix(".jsonl.seal").read_text())["sha256"]
    if hashlib.sha256(ledger.read_bytes()).hexdigest() != seal:
        raise RuntimeError("forecast ledger tampered")
    vault = json.loads((bdir / "question_vault.json").read_text())
    cases = {c["case_id"]: c for c in vault["cases"]}
    want = set(splits) | ({"rotating_locked"} if open_locked else set())
    if open_locked:
        log = rdir / "locked_access_log.json"
        entries = json.loads(log.read_text()) if log.exists() else []
        if any(e.get("event") == "rotating_locked_opened" for e in entries):
            raise PermissionError("rotating locked outcomes already opened for this run — "
                                  "one-time scoring only")
        entries.append({"event": "rotating_locked_opened", "at": time.time(),
                        "ledger_sha": seal})
        log.write_text(json.dumps(entries, indent=1))
    reso = read_resolutions(benchmark_id, purpose=f"score:{run_id}:{sorted(want)}")
    rows = []
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        case = cases.get(r["case_id"]) or {}
        if case.get("split") not in want:
            continue
        if not r.get("qualified"):
            rows.append({**r, "_unscored": "not_qualified"})
            continue
        rz = reso.get(r["case_id"]) or {}
        y = int(rz.get("actual_outcome"))
        res_ts = float(rz.get("resolution_ts"))
        from historical_backtests.models.registry import _ts
        if not (_ts(r["cutoff"]) < res_ts):
            rows.append({**r, "_unscored": "cutoff_after_resolution"})
            continue
        evt = r.get("event_time") or {}
        row = {**r, "_y": y, "_cluster": case.get("cluster_id"),
               "_crps": crps_event_time(evt, outcome=y, resolution_ts=res_ts,
                                        deadline_ts=case.get("resolution_deadline_ts")),
               "_cov80": interval_cover(evt, outcome=y, resolution_ts=res_ts,
                                        q_lo="0.1", q_hi="0.9"),
               "_cov50": interval_cover(evt, outcome=y, resolution_ts=res_ts,
                                        q_lo="0.25", q_hi="0.75")}
        for b in (r.get("baselines") or []):
            if isinstance(b.get("p"), (int, float)):
                row[f"p_{b['arm']}"] = float(b["p"])
        rows.append(row)
    scored = [r for r in rows if "_y" in r]
    out = {"benchmark_id": benchmark_id, "run_id": run_id, "scored_at": time.time(),
           "splits_scored": sorted(want),
           "label": ("ROTATING_SEALED_HOLDOUT" if open_locked
                     else "REUSABLE_DEVELOPMENT_BACKTEST"),
           "n_ledger_rows": len(rows), "n_scored": len(scored),
           "n_unscored": {k: sum(1 for r in rows if r.get("_unscored") == k)
                          for k in {r.get("_unscored") for r in rows if r.get("_unscored")}},
           "overall": _score_block(scored)}
    for split in sorted(want):
        out[f"split_{split}"] = _score_block([r for r in scored
                                              if cases[r["case_id"]]["split"] == split])
    for key, label in (("causal_scale", "scale"), ("domain", "domain")):
        out[f"by_{label}"] = {v: _score_block([r for r in scored if r.get(key) == v])
                              for v in sorted({r.get(key) for r in scored})}
    arms = sorted({b["arm"] for r in scored for b in (r.get("baselines") or [])})
    out["baseline_blocks"] = {}
    out["paired_vs_baselines"] = {}
    for arm in arms:
        key = f"p_{arm}"
        out["baseline_blocks"][arm] = _score_block(
            [dict(r, p_yes=r[key]) for r in scored if key in r])
        out["paired_vs_baselines"][arm] = _paired_bootstrap(scored, key)
    d = out["baseline_blocks"].get("direct_same_model") or {}
    if d.get("brier") and out["overall"].get("brier") is not None:
        out["capability_normalized_skill"] = round(1 - out["overall"]["brier"] / d["brier"], 4)
    outp = rdir / f"scores_{'locked' if open_locked else 'dev'}.json"
    outp.write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--open-locked", action="store_true")
    a = ap.parse_args()
    r = score(a.benchmark, a.run, open_locked=a.open_locked)
    print(json.dumps({k: r[k] for k in ("overall", "capability_normalized_skill",
                                        "paired_vs_baselines", "label") if k in r}, indent=1))
