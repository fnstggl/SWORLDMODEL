"""Fit the family SURVIVAL pack (event-time architecture, component 4) — governance-safe corpus.

Two data sources, both outcome-legal:
 1. The 40 CALIBRATION-split worlds of the frozen v3 vault (never validation/locked).
 2. A WIDE CORPUS of additional closed Polymarket binary markets fetched deterministically
    (gamma /markets, volume-descending), EXCLUDING every validation/locked benchmark world by
    condition_id AND by exact question string — those splits stay sealed.

RESOLUTION-TIME PROXY (v2 — the naive first-0.9-crossing overstates early resolution):
 * STICKY crossing: the first time the YES price crosses 0.9 and never falls back below 0.5 —
   a transient spike that later collapses was not the event happening;
 * TRUE resolution timestamp preferred: a market that CLOSED well before its scheduled end with a
   decisive terminal price resolved early — its closure time is the event time, sharper than any
   price crossing;
 * the lifetime DENOMINATOR is the scheduled window (start → endDate), not the last trade — an
   early-resolving market's fraction must not be biased toward 1 by its own early closure.

`fit_survival_pack` turns the fractions into discrete per-bucket hazards with partial pooling
toward the global curve. Residual-risk note: corpus markets can be topically correlated with sealed
worlds; the pack carries only FAMILY-level timing shapes (5 pooled hazard numbers per family), no
outcome or per-question information.
"""
import datetime as _dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.replay_v2.build_vault as V2B                      # verified archive fetch helpers
from swm.world_model_v2.event_time import SURV_PACK, fit_survival_pack

VAULT = Path("experiments/replay_vault_v3")
THRESHOLD = 0.9
STICKY_FLOOR = 0.5                                                   # a crossing that later falls below
                                                                     # this was a spike, not resolution
CORPUS_TARGET = 400                                                  # additional markets beyond calibration
MIN_VOLUME = 5000.0
MIN_HIST_POINTS = 8
EARLY_CLOSE_MARGIN_S = 86400.0                                       # closed >1d before scheduled end


def _yes_token(m):
    outs = json.loads(m.get("outcomes") or "[]")
    toks = json.loads(m.get("clobTokenIds") or "[]")
    low = [str(o).lower() for o in outs]
    if "yes" in low and len(toks) == len(outs):
        return toks[low.index("yes")]
    return None


def _iso_ts(s):
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def effective_resolution_fraction(hist, *, end_ts=None, closed_ts=None,
                                  threshold=THRESHOLD, floor=STICKY_FLOOR,
                                  min_points=MIN_HIST_POINTS):
    """(lifetime_fraction | None, usable, proxy_kind) — PURE function (unit-tested offline).

    fraction None with usable=True means CENSORED (the event never effectively happened inside the
    window). Denominator = the scheduled window (start → end_ts) so early resolution reads as an
    early fraction instead of being renormalized away."""
    hist = list(hist or [])
    if len(hist) < min_points:
        return None, False, "insufficient_history"
    t0, t_last = hist[0]["t"], hist[-1]["t"]
    t_end = max(float(end_ts) if end_ts else t_last, t_last)
    if t_end <= t0:
        return None, False, "degenerate_window"
    # sticky crossing: first threshold crossing never followed by a fall below the floor
    cross = None
    for i, p in enumerate(hist):
        if p["p"] >= threshold and all(q["p"] >= floor for q in hist[i:]):
            cross = p["t"]
            break
    proxy = "sticky_price_crossing"
    # true resolution: closed clearly before the scheduled end with a decisive terminal price —
    # the closure timestamp is the event time (sharper than the crossing)
    if closed_ts and closed_ts < t_end - EARLY_CLOSE_MARGIN_S and hist[-1]["p"] >= threshold:
        cross = min(cross, float(closed_ts)) if cross is not None else float(closed_ts)
        proxy = "early_close_resolution_time"
    if cross is None:
        return None, True, "censored"
    return max(0.0, min(1.0, (cross - t0) / (t_end - t0))), True, proxy


def _frac_first_cross(tok, market=None):
    hist = V2B._history(tok) if tok else []
    m = market or {}
    frac, ok, _proxy = effective_resolution_fraction(
        hist, end_ts=_iso_ts(m.get("endDate")), closed_ts=_iso_ts(m.get("closedTime")))
    return frac, ok


def _market_by_condition(cid):
    ms = V2B._get(f"{V2B.GAMMA}/markets?condition_ids={cid}&closed=true") or []
    return ms[0] if ms else None


def main():
    ev = json.loads((VAULT / "events.json").read_text())
    cal = [w for w in ev["worlds"] if ev["splits"].get(w["event_id"]) == "calibration"]
    sealed_worlds = [w for w in ev["worlds"] if ev["splits"].get(w["event_id"]) != "calibration"]
    excluded_cids = {(w.get("source") or {}).get("condition_id") for w in sealed_worlds}
    excluded_qs = {str(w.get("question", "")).strip().lower() for w in sealed_worlds}
    print(f"calibration worlds: {len(cal)}; sealed (excluded): {len(sealed_worlds)}")

    rows, seen_cids, skipped = [], set(), 0
    for w in cal:                                            # source 1: calibration split
        cid = (w.get("source") or {}).get("condition_id")
        m = _market_by_condition(cid) if cid else None
        frac, ok = _frac_first_cross(_yes_token(m) if m else None, m)
        if not ok:
            skipped += 1
            continue
        seen_cids.add(cid)
        rows.append({"question": w["question"], "lifetime_fraction_resolved": frac})
        time.sleep(0.1)
    print(f"calibration rows: {len(rows)} (skipped {skipped})")

    offset, n_corpus = 0, 0                                   # source 2: wide corpus, deterministic order
    # volume-descending over the recent era: high-volume closed markets are overwhelmingly the
    # binary political/macro/geopolitical questions this pack needs (id-descending is an esports flood)
    while n_corpus < CORPUS_TARGET and offset < 4000:
        page = V2B._get(f"{V2B.GAMMA}/markets?closed=true&limit=100&offset={offset}"
                        f"&order=volumeNum&ascending=false"
                        f"&end_date_min=2025-01-01&end_date_max=2026-06-30") or []
        if not page:
            break
        offset += 100
        for m in page:
            if n_corpus >= CORPUS_TARGET:
                break
            cid = m.get("conditionId")
            q = str(m.get("question") or "").strip()
            if (not cid or cid in seen_cids or cid in excluded_cids
                    or q.lower() in excluded_qs or not q):
                continue
            outs = sorted(o.lower() for o in json.loads(m.get("outcomes") or "[]"))
            if outs != ["no", "yes"]:
                continue
            frac, ok = _frac_first_cross(_yes_token(m), m)
            if not ok:
                continue
            seen_cids.add(cid)
            rows.append({"question": q, "lifetime_fraction_resolved": frac})
            n_corpus += 1
            if n_corpus % 50 == 0:
                print(f"  corpus rows: {n_corpus}")
            time.sleep(0.08)
    print(f"corpus rows: {n_corpus}; total rows: {len(rows)}")

    pack = fit_survival_pack(rows)
    pack["proxy"] = (f"early-close resolution time when decisive, else STICKY YES>={THRESHOLD} "
                     f"crossing (never re-descending below {STICKY_FLOOR}), over the SCHEDULED "
                     f"window; None = censored")
    pack["n_worlds_used"] = len(rows)
    pack["sources"] = {"calibration_split": len(rows) - n_corpus, "wide_corpus_excl_sealed": n_corpus}
    pack["governance"] = ("validation/locked benchmark worlds excluded by condition_id and question; "
                          "pack carries family-level timing shapes only")
    SURV_PACK.write_text(json.dumps(pack, indent=1))
    fams = ", ".join("{}(n={})".format(k, v["n"]) for k, v in sorted(pack["families"].items()))
    print("\nwrote {}: global={} families={}".format(SURV_PACK, pack["global_hazards"], fams))


if __name__ == "__main__":
    main()
