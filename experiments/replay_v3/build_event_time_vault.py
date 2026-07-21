"""Build the FROZEN EVENT-TIME BENCHMARK VAULT (item: no benchmark covers event-time contracts).

A FUTURE-WINDOW vault: OPEN markets whose scheduled resolution falls inside [WINDOW_START,
WINDOW_END], frozen NOW with their freeze-time prices. Nothing in the vault has an outcome at
freeze time, so no tuning against outcomes is possible — the vault is sealed (SHA-256 over the
canonical JSON) and scored ONCE by score_event_time_vault.py after the window closes:

  * every frozen question is forecast by the V2 system at the frozen as_of (event-time contract:
    first-passage CDF; binary questions are the F(deadline) view of the same object — "when"-type
    and deadline-type questions are scored as ONE object);
  * realized event times come from the resolution-time proxy (early-close resolution time, else
    sticky 0.9 crossing — fit_survival_pack.effective_resolution_fraction) with censoring;
  * scores: censoring-aware CRPS (event_time.crps_first_passage) against the MARKET-IMPLIED
    baseline CDF (constant hazard from the freeze price: λ = −ln(1−p)/(T−t0)), interval coverage,
    and Brier at the deadline.

Until this vault is scored, every event-time performance statement is development-split evidence —
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
MIN_VOLUME = 10000.0
WINDOW_MIN_DAYS = 21                                        # resolution must be ≥3 weeks out (real window)
WINDOW_MAX_DAYS = 180                                       # ... and ≤6 months (scoreable soon)


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def main():
    import experiments.replay_v2.build_vault as V2B
    from experiments.replay_v3.fit_survival_pack import _iso_ts, _yes_token
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists — a frozen vault is never rebuilt in place. "
                         f"Score it, or build a new versioned vault file.")
    now = time.time()
    lo = now + WINDOW_MIN_DAYS * 86400.0
    hi = now + WINDOW_MAX_DAYS * 86400.0
    rows, offset = [], 0
    while len(rows) < N_TARGET and offset < 4000:
        page = V2B._get(f"{V2B.GAMMA}/markets?closed=false&active=true&limit=100&offset={offset}"
                        f"&order=volumeNum&ascending=false") or []
        if not page:
            break
        offset += 100
        for m in page:
            if len(rows) >= N_TARGET:
                break
            q = str(m.get("question") or "").strip()
            cid = m.get("conditionId")
            end_ts = _iso_ts(m.get("endDate"))
            outs = sorted(o.lower() for o in json.loads(m.get("outcomes") or "[]"))
            if not (q and cid and end_ts and outs == ["no", "yes"]):
                continue
            if not (lo <= end_ts <= hi) or float(m.get("volumeNum") or 0.0) < MIN_VOLUME:
                continue
            try:
                prices = json.loads(m.get("outcomePrices") or "[]")
                outs_raw = [str(o).lower() for o in json.loads(m.get("outcomes") or "[]")]
                p_yes = float(prices[outs_raw.index("yes")])
            except (ValueError, TypeError, IndexError):
                continue
            rows.append({"question": q, "condition_id": cid, "yes_token": _yes_token(m),
                         "end_date": m.get("endDate"), "end_ts": end_ts,
                         "market_p_yes_at_freeze": round(p_yes, 4),
                         "volume": float(m.get("volumeNum") or 0.0)})
        time.sleep(0.15)
    if len(rows) < 10:
        raise SystemExit(f"only {len(rows)} eligible future-window markets — refusing to freeze a "
                         f"vault too small to score")
    vault = {"version": "event-time-vault-1.0",
             "frozen_at": dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
             "as_of_ts": now,
             "window": {"min_end": dt.datetime.fromtimestamp(lo, dt.timezone.utc).isoformat(),
                        "max_end": dt.datetime.fromtimestamp(hi, dt.timezone.utc).isoformat()},
             "scoring": {"crps": "event_time.crps_first_passage (censoring-aware, span-normalized)",
                         "baseline": "market-implied constant-hazard CDF from freeze price",
                         "coverage": "event_time.interval_coverage on [0.1, 0.9]",
                         "resolution_proxy": "fit_survival_pack.effective_resolution_fraction"},
             "governance": ("future-window: no outcome exists at freeze; scored ONCE after "
                            "max end_ts by score_event_time_vault.py; until then all event-time "
                            "performance claims are development-split evidence"),
             "n_questions": len(rows), "questions": rows}
    OUT.write_text(json.dumps(vault, indent=1))
    digest = hashlib.sha256(canonical_bytes(vault)).hexdigest()
    SEAL.write_text(json.dumps({"file": OUT.name, "sha256": digest,
                                "sealed_at": vault["frozen_at"],
                                "opens_after": dt.datetime.fromtimestamp(
                                    max(r["end_ts"] for r in rows) + 86400.0,
                                    dt.timezone.utc).isoformat(),
                                "opened": False}, indent=1))
    print(f"froze {len(rows)} future-window questions → {OUT}\nseal {digest[:16]}… → {SEAL}")


if __name__ == "__main__":
    main()
