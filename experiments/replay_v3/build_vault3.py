"""Post-snapshot benchmark vaults (v3) — FROZEN selection rules, written before any model forecast exists.

Temporal ordering enforced per world (Tier B): model release S=2026-04-24 < question open Q <= every
forecast cutoff T < resolution R <= 2026-07-10. Q and R come from the market archive's server timestamps;
a world violating the ordering is ineligible.

Two vaults from ONE frozen candidate pool (>=300 eligible worlds recorded):
  A. REPRESENTATIVE (exactly 100 worlds): chronological order by question-open time; a world is taken in
     that order unless its domain already holds 15 (deterministic stratification, no phase quotas, no
     hand-selection). Splits are CHRONOLOGICAL: earliest 40 calibration, next 20 validation, latest 40
     locked test. Correlated contracts of one event = one world (primary = highest-volume resolved binary).
  B. CAUSAL COVERAGE (up to 60 worlds, overlap allowed): from the REMAINING pool, frozen keyword rules
     assign causal categories; fill each category to 10 in chronological order. Reported separately —
     never merged into the representative headline.

Cutoffs: 4 per world at 15/40/65/88% of the archived trading lifetime (early / mid-early / mid-late /
near-resolution). Market snapshots at the exact cutoff tick (at-or-before, staleness recorded).
Trajectory targets: >=3 objective archived price-path milestones (sealed).
Resolutions + trajectory targets go ONLY to the sealed store (REPLAY_SCORER=1).
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import experiments.replay_v2.build_vault as V2B                      # reuse verified fetch/history helpers

VAULT = Path("experiments/replay_vault_v3")
S_RELEASE = "2026-04-25"                                             # first eligible question-open day
R_MAX = "2026-07-12"
MIN_VOLUME, MIN_LIFETIME_D = 1_000.0, 12.0
CUTOFF_FRACS = (0.15, 0.40, 0.65, 0.88)
N_REPR = 100
QUOTA_PER_DOMAIN = 15
SPLITS = (("calibration", 0, 40), ("validation", 40, 60), ("locked_test", 60, 100))
COVERAGE_TARGET, COVERAGE_PER_CAT = 60, 10


def _ts(iso):
    return time.mktime(time.strptime(iso[:10], "%Y-%m-%d"))


def fetch_recent(max_pages=200):
    """Market-first fetch: gamma /markets supports the exact frozen filters server-side (closed, opened
    after the model release, resolved in-window, volume floor). Markets are then grouped into their
    underlying EVENT so correlated contracts become one world."""
    mkts, offset = [], 0
    while offset < max_pages * 100:
        page = V2B._get(f"{V2B.GAMMA}/markets?closed=true&limit=100&offset={offset}"
                        f"&order=volumeNum&ascending=false&volume_num_min={int(MIN_VOLUME)}"
                        f"&start_date_min={S_RELEASE}T00:00:00Z&end_date_max={R_MAX}T00:00:00Z")
        if not page:
            break
        mkts.extend(page)
        offset += 100
        if len(page) < 100:
            break
    groups = {}
    for m in mkts:
        evs = m.get("events") or [{}]
        eid = str((evs[0] or {}).get("id") or f"solo_{m.get('id')}")
        g = groups.setdefault(eid, {"id": eid, "title": (evs[0] or {}).get("title") or m.get("question"),
                                    "slug": (evs[0] or {}).get("slug") or m.get("slug"),
                                    "category": (evs[0] or {}).get("category"), "markets": []})
        g["markets"].append(m)
    return list(groups.values())


def build():
    VAULT.mkdir(parents=True, exist_ok=True)
    raw = fetch_recent()
    print(f"fetched {len(raw)} closed events in window")
    pool = []
    seen = set()
    for ev in raw:
        if ev.get("id") in seen:
            continue
        seen.add(ev.get("id"))
        mkts = ev.get("markets") or []
        bins = []
        for m in mkts:
            try:
                outs = json.loads(m.get("outcomes") or "[]")
                prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            except Exception:  # noqa: BLE001
                continue
            if sorted(o.lower() for o in outs) == ["no", "yes"] and sorted(prices) == [0.0, 1.0] \
                    and float(m.get("volume") or 0) >= MIN_VOLUME:
                bins.append((float(m.get("volume") or 0), m, outs, prices))
        if not bins:
            continue
        bins.sort(key=lambda x: -x[0])
        vol, m, outs, prices = bins[0]
        q_open = str(m.get("startDate") or ev.get("startDate") or "")[:10]
        r_end = str(m.get("endDate") or ev.get("endDate") or "")[:10]
        if not q_open or q_open < S_RELEASE or not r_end or r_end > R_MAX:
            continue                                        # Tier-B ordering: S < Q and R <= construction
        if (_ts(r_end) - _ts(q_open)) / 86400.0 < MIN_LIFETIME_D:
            continue
        pool.append({"event": ev, "market": m, "outcomes": outs, "prices": prices,
                     "volume": vol, "q_open": q_open, "r_end": r_end, "n_mkts": len(mkts)})
    pool.sort(key=lambda c: (c["q_open"], str(c["event"].get("id"))))   # chronological, deterministic
    print(f"{len(pool)} eligible candidates (frozen pool)")
    (VAULT / "candidate_pool.json").write_text(json.dumps(
        {"n": len(pool), "frozen_rules": "see build_vault3.py docstring", "s_release": S_RELEASE,
         "candidates": [{"event_id": f"pm_{c['event'].get('id')}", "q_open": c["q_open"],
                         "r_end": c["r_end"], "volume": c["volume"],
                         "question": str(c["market"].get("question"))[:140]} for c in pool]}, indent=1))

    def _materialize(c):
        """Fetch archived price history; build the world dict + sealed record; None if history too thin."""
        ev, m = c["event"], c["market"]
        token_ids = json.loads(m.get("clobTokenIds") or "[]")
        if not token_ids:
            return None
        yes_idx = [o.lower() for o in c["outcomes"]].index("yes")
        hist = V2B._history(token_ids[yes_idx])
        if len(hist) < 8:
            return None
        t0, t1 = hist[0]["t"], hist[-1]["t"]
        if (t1 - t0) / 86400.0 < MIN_LIFETIME_D * 0.7:
            return None
        cutoffs, snaps = [], {}
        for f in CUTOFF_FRACS:
            ts = int(t0 + f * (t1 - t0))
            s = V2B._snapshot_at(hist, ts)
            if s is None:
                return None
            iso = time.strftime("%Y-%m-%d", time.gmtime(ts))
            if iso <= c["q_open"]:
                iso = time.strftime("%Y-%m-%d", time.gmtime(ts + 86400))
            cutoffs.append(iso)
            snaps[iso] = s
        traj = V2B._trajectory_targets(hist)
        if len(traj) < 3:
            return None
        q = str(m.get("question") or ev.get("title") or "")
        wid = f"pm_{ev.get('id')}"
        world = {"event_id": wid, "question": q, "domain": V2B._domain(ev, q),
                 "event_family": str(ev.get("slug") or wid),
                 "causal_categories": V2B._causal_categories(q),
                 "n_correlated_contracts": c["n_mkts"], "question_open": c["q_open"],
                 "forecast_cutoffs": cutoffs,
                 "horizon": time.strftime("%Y-%m-%d", time.gmtime(t1 + 86400)),
                 "market_snapshots": snaps, "primary_volume_usdc": c["volume"],
                 "temporal_ordering": {"model_release": "2026-04-24", "question_open": c["q_open"],
                                       "resolution": c["r_end"], "ordering_ok": True},
                 "source": {"archive": "polymarket_gamma+clob", "event_id": ev.get("id"),
                            "condition_id": m.get("conditionId")}}
        sealed = {"outcome": int(c["prices"][yes_idx] == 1.0),
                  "resolution_source": "polymarket outcomePrices (UMA-resolved)",
                  "trajectory_targets": traj}
        return world, sealed

    repr_worlds, cov_worlds, sealed = [], [], {}
    dom_count, cov_cat = {}, {}
    used = set()
    for c in pool:                                          # representative fill (chronological, capped)
        if len(repr_worlds) >= N_REPR:
            break
        ev, m = c["event"], c["market"]
        dom = V2B._domain(ev, str(m.get("question") or ""))
        if dom_count.get(dom, 0) >= QUOTA_PER_DOMAIN:
            continue
        out = _materialize(c)
        if out is None:
            continue
        world, srec = out
        repr_worlds.append(world)
        sealed[world["event_id"]] = srec
        dom_count[dom] = dom_count.get(dom, 0) + 1
        used.add(c["event"].get("id"))
        print(f"[R{len(repr_worlds):3d}] {world['event_id']} {dom:12s} open={c['q_open']} "
              f"cuts={world['forecast_cutoffs']}", flush=True)
    for c in pool:                                          # coverage fill (frozen category rules)
        if len(cov_worlds) >= COVERAGE_TARGET:
            break
        if c["event"].get("id") in used:
            continue
        cats = V2B._causal_categories(str(c["market"].get("question") or ""))
        need = [k for k in cats if cov_cat.get(k, 0) < COVERAGE_PER_CAT]
        if not need:
            continue
        out = _materialize(c)
        if out is None:
            continue
        world, srec = out
        cov_worlds.append(world)
        sealed[world["event_id"]] = srec
        for k in cats:
            cov_cat[k] = cov_cat.get(k, 0) + 1
        used.add(c["event"].get("id"))
        print(f"[C{len(cov_worlds):3d}] {world['event_id']} cats={cats}", flush=True)

    order = [w["event_id"] for w in repr_worlds]            # already chronological
    split_map = {}
    for name, a, b in SPLITS:
        for wid in order[a:b]:
            split_map[wid] = name
    (VAULT / "events.json").write_text(json.dumps(
        {"note": "PUBLIC representative vault v3 (post-snapshot, chronological splits) — no outcomes.",
         "tier": "provider_attested_post_cutoff", "n_worlds": len(repr_worlds),
         "domain_counts": dom_count, "splits": split_map, "worlds": repr_worlds}, indent=1))
    (VAULT / "coverage_events.json").write_text(json.dumps(
        {"note": "PUBLIC causal-coverage vault v3 — reported separately, never in the accuracy headline.",
         "n_worlds": len(cov_worlds), "category_counts": cov_cat, "worlds": cov_worlds}, indent=1))
    (VAULT / "SEALED_resolutions_v3.json").write_text(json.dumps(
        {"note": "SEALED — scorer only (REPLAY_SCORER=1).", "resolutions": sealed}, indent=1))
    print(f"\nrepresentative={len(repr_worlds)} domains={dom_count}")
    print(f"coverage={len(cov_worlds)} cats={cov_cat}")
    print("splits:", {k: sum(1 for v in split_map.values() if v == k)
                      for k in ("calibration", "validation", "locked_test")})


if __name__ == "__main__":
    build()
