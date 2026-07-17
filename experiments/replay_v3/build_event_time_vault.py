"""Build the FROZEN EVENT-TIME BENCHMARK VAULT (item: no benchmark covers event-time contracts).

TWO TRANCHES, one seal:
  * NEAR (~1/3): OPEN markets scheduled to resolve TODAY or TOMORROW (calendar days at freeze)
    that have NOT effectively happened yet — the freeze price must be genuinely undecided
    (NEAR_P_MIN ≤ p ≤ NEAR_P_MAX); a price pinned at 0.01/0.99 means the world already knows.
    This tranche becomes scoreable within ~2 days — the fast first read on event-time skill.
  * FAR (~2/3): OPEN markets resolving in [21, 180] days — nothing about them is knowable at
    freeze; the full-strength future-window test.

Each tranche has its OWN time gate and single-open gate in the seal; both live under one SHA-256
over the canonical vault JSON. Scored by score_event_time_vault.py --tranche near|far|all:

  * every frozen question is forecast by the V2 system at the frozen as_of (event-time contract:
    first-passage CDF; binary questions are the F(deadline) view of the same object — "when"-type
    and deadline-type questions are scored as ONE object);
  * realized event times come from the resolution-time proxy (early-close resolution time, else
    sticky 0.9 crossing — fit_survival_pack.effective_resolution_fraction) with censoring;
  * scores: censoring-aware CRPS (event_time.crps_first_passage) against the MARKET-IMPLIED
    baseline CDF (constant hazard from the freeze price: λ = −ln(1−p)/(T−t0)), interval coverage,
    and Brier at the deadline.

Until a tranche is scored, every event-time performance statement is development-split evidence —
that limitation is stamped into the vault itself.

Requires network. Deterministic given the freeze moment (volume-descending order, filters recorded).
"""
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VAULT = Path("experiments/replay_vault_v3")
OUT = VAULT / "event_time_frozen_vault.json"
SEAL = VAULT / "event_time_vault_seal.json"
N_TARGET = 60
NEAR_SHARE = 1 / 3                                          # ~1/3 of questions resolve today/tomorrow
MIN_VOLUME = 10000.0
NEAR_MIN_LEAD_S = 1800.0                                    # ≥30min out — never freeze a market mid-gavel
NEAR_P_MIN, NEAR_P_MAX = 0.05, 0.95                         # "hasn't happened yet": price undecided
WINDOW_MIN_DAYS = 21                                        # FAR: resolution ≥3 weeks out (real window)
WINDOW_MAX_DAYS = 180                                       # ... and ≤6 months (scoreable soon)


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def end_of_tomorrow_utc(now: float) -> float:
    d = dt.datetime.fromtimestamp(now, dt.timezone.utc).date() + dt.timedelta(days=2)
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()


def _parse_market(m, _iso_ts, _yes_token):
    q = str(m.get("question") or "").strip()
    cid = m.get("conditionId")
    end_ts = _iso_ts(m.get("endDate"))
    outs = sorted(o.lower() for o in json.loads(m.get("outcomes") or "[]"))
    if not (q and cid and end_ts and outs == ["no", "yes"]):
        return None
    if float(m.get("volumeNum") or 0.0) < MIN_VOLUME:
        return None
    try:
        prices = json.loads(m.get("outcomePrices") or "[]")
        outs_raw = [str(o).lower() for o in json.loads(m.get("outcomes") or "[]")]
        p_yes = float(prices[outs_raw.index("yes")])
    except (ValueError, TypeError, IndexError):
        return None
    events = m.get("events") or []
    ev_slug = str((events[0] or {}).get("slug", "")) if isinstance(events, list) and events else ""
    return {"question": q, "condition_id": cid, "yes_token": _yes_token(m),
            "end_date": m.get("endDate"), "end_ts": end_ts,
            "market_p_yes_at_freeze": round(p_yes, 4),
            "market_slug": str(m.get("slug", "")),
            "event_cluster": ev_slug or str(m.get("slug", "")) or str(cid),
            "volume": float(m.get("volumeNum") or 0.0)}


def _collect(V2B, _iso_ts, _yes_token, *, n_want, accept, extra_query="", max_offset=6000,
             cap_per_cluster=3):
    """Volume-descending deterministic collection. At most `cap_per_cluster` markets share one
    underlying EVENT (a football match spawns winner/draw/exact-score contracts whose outcomes are
    one realization — cluster-capped at freeze, cluster-aware at scoring)."""
    rows, offset, seen, per_cluster = [], 0, set(), {}
    while len(rows) < n_want and offset < max_offset:
        page = V2B._get(f"{V2B.GAMMA}/markets?closed=false&active=true&limit=100&offset={offset}"
                        f"&order=volumeNum&ascending=false{extra_query}") or []
        if not page:
            break
        offset += 100
        for m in page:
            if len(rows) >= n_want:
                break
            row = _parse_market(m, _iso_ts, _yes_token)
            if row is None or row["condition_id"] in seen or not accept(row):
                continue
            cl = row["event_cluster"]
            if per_cluster.get(cl, 0) >= cap_per_cluster:
                continue
            per_cluster[cl] = per_cluster.get(cl, 0) + 1
            seen.add(row["condition_id"])
            rows.append(row)
        time.sleep(0.15)
    return rows


def main():
    import experiments.replay_v2.build_vault as V2B
    from experiments.replay_v3.fit_survival_pack import _iso_ts, _yes_token
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists — a frozen vault is never rebuilt in place. "
                         f"Score it, or build a new versioned vault file.")
    now = time.time()
    n_near = max(1, int(round(N_TARGET * NEAR_SHARE)))
    n_far = N_TARGET - n_near
    near_lo, near_hi = now + NEAR_MIN_LEAD_S, end_of_tomorrow_utc(now)
    far_lo, far_hi = now + WINDOW_MIN_DAYS * 86400.0, now + WINDOW_MAX_DAYS * 86400.0

    def _accept_near(r):
        # resolves today/tomorrow AND has not effectively happened: freeze price undecided
        return (near_lo <= r["end_ts"] <= near_hi
                and NEAR_P_MIN <= r["market_p_yes_at_freeze"] <= NEAR_P_MAX)

    def _accept_far(r):
        return far_lo <= r["end_ts"] <= far_hi
    near_max_iso = dt.datetime.fromtimestamp(near_hi, dt.timezone.utc).strftime("%Y-%m-%d")
    near = _collect(V2B, _iso_ts, _yes_token, n_want=n_near, accept=_accept_near,
                    extra_query=f"&end_date_max={near_max_iso}")
    far = _collect(V2B, _iso_ts, _yes_token, n_want=n_far, accept=_accept_far)
    for r in near:
        r["tranche"] = "near"
    for r in far:
        r["tranche"] = "far"
    rows = near + far
    if len(rows) < 10 or not near or not far:
        raise SystemExit(f"only {len(near)} near + {len(far)} far eligible markets — refusing to "
                         f"freeze a vault too small to score")
    vault = {"version": "event-time-vault-2.0",
             "frozen_at": dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
             "as_of_ts": now,
             "window": {"near_min_end": dt.datetime.fromtimestamp(near_lo, dt.timezone.utc).isoformat(),
                        "near_max_end": dt.datetime.fromtimestamp(near_hi, dt.timezone.utc).isoformat(),
                        "far_min_end": dt.datetime.fromtimestamp(far_lo, dt.timezone.utc).isoformat(),
                        "far_max_end": dt.datetime.fromtimestamp(far_hi, dt.timezone.utc).isoformat()},
             "scoring": {"crps": "event_time.crps_first_passage (censoring-aware, span-normalized)",
                         "baseline": "market-implied constant-hazard CDF from freeze price",
                         "coverage": "event_time.interval_coverage on [0.1, 0.9]",
                         "resolution_proxy": "fit_survival_pack.effective_resolution_fraction"},
             "governance": ("future-window per tranche: NEAR resolves today/tomorrow and is frozen "
                            f"only while genuinely undecided ({NEAR_P_MIN} <= p <= {NEAR_P_MAX} at "
                            "freeze — the recorded freeze price is also the baseline, a fair fight); "
                            "FAR resolves in 21-180d. Each tranche scored ONCE after ITS window by "
                            "score_event_time_vault.py --tranche; until then all event-time "
                            "performance claims are development-split evidence"),
             "n_questions": len(rows), "n_near": len(near), "n_far": len(far),
             "questions": rows}
    OUT.write_text(json.dumps(vault, indent=1))
    digest = hashlib.sha256(canonical_bytes(vault)).hexdigest()

    def _opens(tranche_rows):
        return dt.datetime.fromtimestamp(max(r["end_ts"] for r in tranche_rows) + 86400.0,
                                         dt.timezone.utc).isoformat()
    SEAL.write_text(json.dumps({
        "file": OUT.name, "sha256": digest, "sealed_at": vault["frozen_at"],
        "opens_after": _opens(rows),                          # full-vault gate (back-compat)
        "opened": False,
        "tranches": {"near": {"n": len(near), "opens_after": _opens(near), "opened": False},
                     "far": {"n": len(far), "opens_after": _opens(far), "opened": False}}},
        indent=1))
    print(f"froze {len(rows)} questions ({len(near)} near resolving by {near_max_iso}, "
          f"{len(far)} far) → {OUT}\nseal {digest[:16]}… → {SEAL}")


if __name__ == "__main__":
    main()
