"""Score the FROZEN EVENT-TIME VAULT — once, after the window closes (never before).

Gates, in order:
 1. TIME GATE: refuses to run before the seal's `opens_after` (the latest frozen end date + 1d).
 2. SEAL GATE: recomputes the vault SHA-256; a mismatch aborts (the vault was touched).
 3. SINGLE-OPEN GATE: the seal records `opened`; a second scoring run refuses (the vault is
    consumed — build a fresh one for the next claim).

Per frozen question:
 * the V2 system runs at the FROZEN as_of (archived evidence only — the evidence layer's paired-date
   invariant enforces temporal hygiene; the run is TIER-labeled honestly since model weights may
   post-date the freeze) producing the first-passage CDF + quantiles;
 * the realized event time / censoring comes from the resolution-time proxy
   (effective_resolution_fraction) on the archived price path;
 * scores: censoring-aware CRPS (system vs the market-implied constant-hazard baseline CDF from the
   freeze price), interval coverage on [0.1, 0.9], Brier at the deadline (F(deadline) vs outcome) —
   "when"-type and deadline-type questions scored as the SAME object.

Requires network + DEEPSEEK_API_KEY at scoring time.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.replay_v3.build_event_time_vault import OUT, SEAL, canonical_bytes

RESULTS = Path("experiments/results/replay_v3/event_time_vault_scores.json")
MODEL = "deepseek-chat"


def market_baseline_cdf(p_yes: float, as_of: float, end_ts: float, grid: list) -> list:
    """The market-implied constant-hazard CDF from the freeze price: λ = −ln(1−p)/(T−t0);
    F(t) = 1 − exp(−λ(t−t0)). The comparable first-passage object for CRPS-vs-market."""
    p = max(1e-4, min(0.985, float(p_yes)))
    lam = -math.log(1.0 - p) / max(3600.0, end_ts - as_of)
    return [1.0 - math.exp(-lam * max(0.0, g - as_of)) for g in grid]


def main():
    import datetime as dt
    import experiments.replay_v2.build_vault as V2B
    from experiments.replay_v3.fit_survival_pack import (_iso_ts, _market_by_condition,
                                                         effective_resolution_fraction)
    from swm.world_model_v2.event_time import crps_first_passage, interval_coverage

    seal = json.loads(SEAL.read_text())
    now = time.time()
    opens = dt.datetime.fromisoformat(seal["opens_after"]).timestamp()
    if now < opens:                                          # 1. time gate
        raise SystemExit(f"vault opens after {seal['opens_after']} — refusing to score early "
                         f"({(opens - now) / 86400.0:.1f} days remain)")
    if seal.get("opened"):                                   # 3. single-open gate
        raise SystemExit("vault already opened and scored once — it is consumed; build a fresh one")
    vault = json.loads(OUT.read_text())
    digest = hashlib.sha256(canonical_bytes(vault)).hexdigest()
    if digest != seal["sha256"]:                             # 2. seal gate
        raise SystemExit(f"SEAL MISMATCH: vault sha256 {digest[:16]}… != sealed {seal['sha256'][:16]}…")
    seal["opened"] = True
    seal["opened_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    SEAL.write_text(json.dumps(seal, indent=1))

    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world
    llm = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=2400, temperature=0.2)
    as_of_iso = vault["frozen_at"][:10]
    rows = []
    for w in vault["questions"]:
        # ---- realized outcome (resolution-time proxy, censoring-aware) ----
        m = _market_by_condition(w["condition_id"]) or {}
        hist = V2B._history(w.get("yes_token")) if w.get("yes_token") else []
        frac, usable, proxy = effective_resolution_fraction(
            hist, end_ts=_iso_ts(m.get("endDate")) or w["end_ts"], closed_ts=_iso_ts(m.get("closedTime")))
        t0 = hist[0]["t"] if hist else vault["as_of_ts"]
        t_end = max(_iso_ts(m.get("endDate")) or w["end_ts"], hist[-1]["t"] if hist else w["end_ts"])
        event_ts = (t0 + frac * (t_end - t0)) if (usable and frac is not None) else None
        # only the post-freeze part of the window is forecastable — clamp the realized time
        if event_ts is not None and event_ts < vault["as_of_ts"]:
            event_ts = vault["as_of_ts"] + 1.0
        # ---- the V2 forecast at the frozen as_of ----
        row = {"question": w["question"], "condition_id": w["condition_id"],
               "event_ts": event_ts, "censored": event_ts is None, "resolution_proxy": proxy,
               "usable_outcome": usable}
        try:
            res = simulate_world(w["question"], as_of=as_of_iso, horizon=str(w["end_date"])[:10],
                                 llm=llm, seed=0)
            proj = getattr(res, "raw_distribution", None) or {}
            # the full first-passage readout travels on result.provenance["event_time"]
            # (pipeline.result_from_run carries contract.project()'s event_time block verbatim)
            evt = (getattr(res, "provenance", None) or {}).get("event_time") or {}
            grid, cdf = evt.get("cdf_grid_ts") or [], evt.get("cdf") or []
            if grid and cdf:
                row["crps_v2"] = crps_first_passage(grid, cdf, event_ts=event_ts,
                                                    as_of=vault["as_of_ts"], horizon_ts=w["end_ts"])
                base = market_baseline_cdf(w["market_p_yes_at_freeze"], vault["as_of_ts"],
                                           w["end_ts"], grid)
                row["crps_market"] = crps_first_passage(grid, base, event_ts=event_ts,
                                                        as_of=vault["as_of_ts"], horizon_ts=w["end_ts"])
                row["covered_80"] = interval_coverage(evt.get("first_passage_quantiles_ts") or {},
                                                      event_ts)
                p_yes = None
                for k, v in (proj or {}).items():
                    if str(k).lower() in ("yes", "true"):
                        p_yes = float(v)
                if p_yes is None and isinstance(evt.get("p_event_by_deadline"), (int, float)):
                    p_yes = float(evt["p_event_by_deadline"])
                if p_yes is not None:
                    y = 0.0 if event_ts is None else 1.0
                    row["brier_v2"] = (p_yes - y) ** 2
                    row["brier_market"] = (float(w["market_p_yes_at_freeze"]) - y) ** 2
            else:
                row["error"] = "no event_time CDF in result projection"
            row["status"] = getattr(res, "simulation_status", "?")
        except Exception as e:  # noqa: BLE001 — a failed row is a recorded failure, never dropped
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(row)
        time.sleep(0.5)

    scored = [r for r in rows if "crps_v2" in r]
    out = {"opened_at": seal["opened_at"], "n_questions": len(rows), "n_scored": len(scored),
           "n_censored": sum(1 for r in rows if r.get("censored")),
           "mean_crps_v2": (sum(r["crps_v2"] for r in scored) / len(scored)) if scored else None,
           "mean_crps_market": (sum(r["crps_market"] for r in scored) / len(scored)) if scored else None,
           "coverage_80": (sum(1 for r in scored if r.get("covered_80")) / len(scored)) if scored else None,
           "mean_brier_v2": (sum(r["brier_v2"] for r in scored if "brier_v2" in r)
                             / max(1, sum(1 for r in scored if "brier_v2" in r))) if scored else None,
           "mean_brier_market": (sum(r["brier_market"] for r in scored if "brier_market" in r)
                                 / max(1, sum(1 for r in scored if "brier_market" in r))) if scored else None,
           "rows": rows}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
