"""Score the FROZEN EVENT-TIME VAULT — once per tranche, after THAT tranche's window closes.

Gates, in order (per --tranche near|far|all):
 1. TIME GATE: refuses to run before the tranche's `opens_after` (its latest frozen end date + 1d).
 2. SEAL GATE: recomputes the vault SHA-256; a mismatch aborts (the vault was touched).
 3. SINGLE-OPEN GATE: the seal records `opened` per tranche; a second scoring run for the same
    tranche refuses (that tranche is consumed — build a fresh vault for the next claim).

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


def main(tranche: str = None):
    import argparse
    import datetime as dt
    import experiments.replay_v2.build_vault as V2B
    from experiments.replay_v3.fit_survival_pack import (_iso_ts, _market_by_condition,
                                                         effective_resolution_fraction)
    from swm.world_model_v2.event_time import crps_first_passage, interval_coverage

    if tranche is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--tranche", choices=("near", "far", "all"), default="all")
        tranche = ap.parse_args().tranche
    seal = json.loads(SEAL.read_text())
    now = time.time()
    tr = (seal.get("tranches") or {}).get(tranche) if tranche != "all" else None
    gate = tr if tr is not None else seal                    # v1 seals / --tranche all → full gate
    opens = dt.datetime.fromisoformat(gate["opens_after"]).timestamp()
    if now < opens:                                          # 1. time gate (per tranche)
        raise SystemExit(f"tranche '{tranche}' opens after {gate['opens_after']} — refusing to "
                         f"score early ({(opens - now) / 86400.0:.1f} days remain)")
    if gate.get("opened"):                                   # 3. single-open gate (per tranche)
        raise SystemExit(f"tranche '{tranche}' already opened and scored once — it is consumed; "
                         f"build a fresh vault")
    vault = json.loads(OUT.read_text())
    digest = hashlib.sha256(canonical_bytes(vault)).hexdigest()
    if digest != seal["sha256"]:                             # 2. seal gate
        raise SystemExit(f"SEAL MISMATCH: vault sha256 {digest[:16]}… != sealed {seal['sha256'][:16]}…")
    gate["opened"] = True
    gate["opened_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if tranche == "all":
        for t in (seal.get("tranches") or {}).values():      # opening all consumes every tranche
            t["opened"] = True
            t.setdefault("opened_at", gate["opened_at"])
    SEAL.write_text(json.dumps(seal, indent=1))

    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world
    llm = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=2400, temperature=0.2)
    as_of_iso = vault["frozen_at"][:10]
    rows = []
    targets = [w for w in vault["questions"]
               if tranche == "all" or str(w.get("tranche", "far")) == tranche]
    for w in targets:
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
    # cluster-aware headline: correlated same-event contracts (winner/draw/exact-score of one
    # match) are ONE realization — average within clusters first, then across clusters
    by_cluster = {}
    for r, w in zip(rows, targets):
        if "crps_v2" in r:
            by_cluster.setdefault(str(w.get("event_cluster") or w["condition_id"]), []).append(r)
    cl_means_v2 = [sum(x["crps_v2"] for x in g) / len(g) for g in by_cluster.values()]
    cl_means_mkt = [sum(x["crps_market"] for x in g) / len(g) for g in by_cluster.values()]
    out = {"tranche": tranche, "opened_at": gate["opened_at"],
           "n_questions": len(rows), "n_scored": len(scored),
           "n_clusters": len(by_cluster),
           "cluster_mean_crps_v2": (sum(cl_means_v2) / len(cl_means_v2)) if cl_means_v2 else None,
           "cluster_mean_crps_market": (sum(cl_means_mkt) / len(cl_means_mkt)) if cl_means_mkt else None,
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
    results_path = RESULTS.with_name(f"event_time_vault_scores_{tranche}.json")
    results_path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print(f"rows → {results_path}")


if __name__ == "__main__":
    main()
