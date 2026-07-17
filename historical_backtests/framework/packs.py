"""Walk-forward fitted parameters — every number a historical forecast consumes obeys the clock.

Survival packs: monthly snapshots fit ONLY on markets whose realized resolution predates the
snapshot month, from one cached deterministic corpus (fetched once; per-market resolution
timestamps recorded). `load_pack(as_of=T)` selects the latest snapshot strictly before T and
points the production runtime's pack path at it. Fail-closed: a snapshot whose
`latest_included_resolution_ts >= T` refuses to load.

Intention-HR and coupling packs: no labeled pre-cutoff corpus exists for these yet, so the
DOCUMENTED PRIORS serve (they are time-invariant constants shipped in code, not fitted from any
outcome) and every row records `fallback_reason: insufficient_pre_cutoff_fit_data`. The runtime's
`hr_pack_info()/coupling_pack_info()` provenance already reports `documented_priors_unfitted`.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "fitted_packs" / "survival_corpus_cache.json"
SNAP_DIR = ROOT / "fitted_packs" / "survival_snapshots"
CORPUS_TARGET = 500


def build_corpus_cache(max_pages: int = 12) -> dict:
    """One deterministic volume-descending fetch of closed binary markets with per-market
    (question, resolution_ts, lifetime_fraction) rows. Cached; refreshed only explicitly."""
    import experiments.replay_v2.build_vault as V2B
    from experiments.replay_v3.fit_survival_pack import (_iso_ts, _yes_token,
                                                         effective_resolution_fraction)
    rows, offset = [], 0
    while len(rows) < CORPUS_TARGET and offset < max_pages * 100:
        page = V2B._get(f"{V2B.GAMMA}/markets?closed=true&limit=100&offset={offset}"
                        f"&order=volumeNum&ascending=false&end_date_min=2024-01-01") or []
        if not page:
            break
        offset += 100
        for m in page:
            q = str(m.get("question") or "").strip()
            outs = sorted(o.lower() for o in json.loads(m.get("outcomes") or "[]"))
            if outs != ["no", "yes"] or not q:
                continue
            end_ts = _iso_ts(m.get("endDate"))
            if not end_ts:
                continue
            tok = _yes_token(m)
            hist = V2B._history(tok) if tok else []
            frac, ok, _proxy = effective_resolution_fraction(
                hist, end_ts=end_ts, closed_ts=_iso_ts(m.get("closedTime")))
            if not ok:
                continue
            t0 = hist[0]["t"]
            res_ts = (t0 + frac * (end_ts - t0)) if frac is not None else end_ts
            rows.append({"question": q, "lifetime_fraction_resolved": frac,
                         "resolution_ts": res_ts, "condition_id": m.get("conditionId")})
            if len(rows) >= CORPUS_TARGET:
                break
            time.sleep(0.05)
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    out = {"built_at": time.time(), "n": len(rows), "rows": rows,
           "rule": "volume-descending closed binary markets; sticky/early-close resolution proxy"}
    CORPUS.write_text(json.dumps(out, indent=1))
    return out


def build_snapshots(months: list = None) -> list:
    """Fit one survival pack per month boundary using ONLY rows resolved before that boundary."""
    from swm.world_model_v2.event_time import fit_survival_pack
    corpus = json.loads(CORPUS.read_text())["rows"]
    if months is None:
        months = []
        y, mth = 2024, 9
        now = time.gmtime()
        while (y, mth) <= (now.tm_year, now.tm_mon):
            months.append(f"{y}-{mth:02d}-01")
            mth += 1
            if mth == 13:
                y, mth = y + 1, 1
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for m in months:
        boundary = time.mktime(time.strptime(m, "%Y-%m-%d"))
        rows = [r for r in corpus if r["resolution_ts"] < boundary]
        if len(rows) < 25:
            continue
        pack = fit_survival_pack([{"question": r["question"],
                                   "lifetime_fraction_resolved": r["lifetime_fraction_resolved"]}
                                  for r in rows])
        pack["walk_forward"] = {
            "snapshot_boundary": m, "n_training_rows": len(rows),
            "latest_included_resolution_ts": max(r["resolution_ts"] for r in rows),
            "min_source_ts": min(r["resolution_ts"] for r in rows),
            "fitted_at": time.time(),
            "source_hash": hashlib.sha256(json.dumps(
                sorted(r["condition_id"] or "" for r in rows)).encode()).hexdigest()[:16],
            "training_code": "swm.world_model_v2.event_time.fit_survival_pack"}
        p = SNAP_DIR / f"survival_pack_asof_{m}.json"
        p.write_text(json.dumps(pack, indent=1))
        written.append(str(p))
    return written


def load_pack(as_of_ts: float) -> dict:
    """Select the latest monthly snapshot strictly before as_of; POINT the production runtime at
    it (event_time.SURV_PACK). Fail-closed on any future-resolved training row."""
    import swm.world_model_v2.event_time as ET
    best, best_boundary = None, None
    for p in sorted(SNAP_DIR.glob("survival_pack_asof_*.json")):
        m = p.stem.replace("survival_pack_asof_", "")
        boundary = time.mktime(time.strptime(m, "%Y-%m-%d"))
        if boundary <= as_of_ts and (best_boundary is None or boundary > best_boundary):
            best, best_boundary = p, boundary
    if best is None:
        ET.SURV_PACK = SNAP_DIR / "_none_"                   # no pack → runtime uses base envelope
        return {"survival_pack": None,
                "fallback_reason": "insufficient_pre_cutoff_fit_data",
                "intention_hr_pack": {"fallback_reason": "insufficient_pre_cutoff_fit_data",
                                      "source": "documented_priors_unfitted"},
                "coupling_pack": {"fallback_reason": "insufficient_pre_cutoff_fit_data",
                                  "source": "documented_priors_sampled_per_branch"}}
    wf = json.loads(best.read_text())["walk_forward"]
    if wf["latest_included_resolution_ts"] >= as_of_ts:
        raise RuntimeError(f"walk-forward violation: snapshot {best.name} contains a row resolved "
                           f"at/after the forecast cutoff")
    ET.SURV_PACK = best
    ET.INTENTION_HR_PACK = SNAP_DIR / "_none_intention_"     # priors, never a future-fitted pack
    try:
        import swm.world_model_v2.world_dynamics as WD
        WD.COUPLING_PACK = SNAP_DIR / "_none_coupling_"
    except Exception:  # noqa: BLE001
        pass
    return {"survival_pack": {"path": best.name, **{k: wf[k] for k in
                                                    ("snapshot_boundary", "n_training_rows",
                                                     "latest_included_resolution_ts",
                                                     "fitted_at", "source_hash")}},
            "intention_hr_pack": {"fallback_reason": "insufficient_pre_cutoff_fit_data",
                                  "source": "documented_priors_unfitted"},
            "coupling_pack": {"fallback_reason": "insufficient_pre_cutoff_fit_data",
                              "source": "documented_priors_sampled_per_branch"}}


if __name__ == "__main__":
    if not CORPUS.exists():
        print("building corpus cache …", flush=True)
        c = build_corpus_cache()
        print("corpus rows:", c["n"])
    w = build_snapshots()
    print(f"wrote {len(w)} monthly survival snapshots")
