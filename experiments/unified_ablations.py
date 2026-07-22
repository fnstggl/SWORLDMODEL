"""Unified runtime — cross-domain end-to-end run + per-phase causal ablations (Parts Q/R).

Runs the ONE canonical `simulate_world` on a stratified cross-domain sample, then re-runs it dropping one phase
at a time (via `execution_policy={'drop_phases':[...]}`) with identical question/as_of/seed. A phase is
CAUSALLY INTEGRATED only if its removal changes the terminal (or execution) where it should matter; this
harness supplies that evidence honestly — including when removal changes nothing (shallow integration).

Incremental + resumable. Machine-readable: experiments/results/unified/ablations.json.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

OUT = Path("experiments/results/unified")
ART = OUT / "ablations.json"

# stratified cross-domain sample (question, as_of, horizon, domain) — resolved historical, reused from corpora
SAMPLE = [
    ("Will the US Federal Reserve cut interest rates at its September 2024 meeting?", "2024-09-10", "2024-09-19", "econ"),
    ("Will there be a US federal government shutdown on October 1, 2024?", "2024-09-20", "2024-10-01", "politics"),
    ("Will the Kansas City Chiefs win Super Bowl LVIII in February 2024?", "2024-02-05", "2024-02-12", "sports"),
    ("Will Microsoft complete its acquisition of Activision Blizzard in 2023?", "2023-07-01", "2023-12-31", "tech"),
    ("Will Sweden formally join NATO in 2024?", "2024-01-15", "2024-12-31", "geopolitics"),
    ("Will the SAG-AFTRA actors' strike end with a tentative agreement in 2023?", "2023-10-01", "2023-12-31", "labor"),
    ("Will Reddit complete its initial public offering in 2024?", "2024-01-15", "2024-12-31", "finance"),
    ("Will India's Chandrayaan-3 successfully soft-land near the Moon's south pole in 2023?", "2023-08-01", "2023-08-31", "science"),
]

DROP_ARMS = ["phase2_evidence", "phase3_posterior", "phase8_persistence", "phase11_recompilation"]


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)


def _manifest_summary(res):
    m = (res.provenance or {}).get("active_component_manifest", {})
    return {k: {"executed": v["executed"], "omitted": v["omitted"]} for k, v in m.items()}


def run(limit=None):
    import functools
    from swm.world_model_v2.unified_runtime import simulate_world as _sw_default
    # archival full-fidelity harness: pinned since the §25 default switch
    simulate_world = functools.partial(_sw_default, execution_profile="full_fidelity")
    OUT.mkdir(parents=True, exist_ok=True)
    llm = _make_llm()
    if llm is None:
        print("no llm"); return
    existing = {}
    if ART.exists():
        for r in json.loads(ART.read_text()).get("rows", []):
            existing[r["question"]] = r
    rows = []
    qs = SAMPLE[:limit] if limit else SAMPLE
    for q, as_of, horizon, domain in qs:
        if q in existing:
            rows.append(existing[q]); continue
        rec = {"question": q, "domain": domain, "as_of": as_of}
        try:
            full = simulate_world(q, as_of=as_of, horizon=horizon, seed=0, llm=llm)
            rec["full_p"] = full.raw_probability
            rec["full_status"] = full.simulation_status
            rec["full_manifest"] = _manifest_summary(full)
            arms = {}
            for phase in DROP_ARMS:
                r = simulate_world(q, as_of=as_of, horizon=horizon, seed=0, llm=llm,
                                   execution_policy={"drop_phases": [phase]})
                dp = r.raw_probability
                arms[phase] = {"p": dp,
                               "delta_vs_full": (None if dp is None or full.raw_probability is None
                                                 else round(dp - full.raw_probability, 4))}
            rec["ablation_arms"] = arms
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:160]
        rows.append(rec)
        ART.write_text(json.dumps({"rows": rows, "drop_arms": DROP_ARMS,
                       "retrieval_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
        print(f"[{domain:11s}] full={rec.get('full_p')} " +
              " ".join(f"{p.split('_')[1][:4]}Δ={rec.get('ablation_arms',{}).get(p,{}).get('delta_vs_full')}"
                       for p in DROP_ARMS))
    _aggregate(rows)
    print("DONE ablations", len(rows))


def _aggregate(rows):
    ok = [r for r in rows if r.get("ablation_arms")]
    per_phase = {}
    # activation rate from full manifests
    act = {}
    for r in ok:
        for ph, v in (r.get("full_manifest") or {}).items():
            act.setdefault(ph, {"executed": 0, "n": 0})
            act[ph]["executed"] += 1 if v["executed"] else 0
            act[ph]["n"] += 1
    for phase in DROP_ARMS:
        deltas = [abs(r["ablation_arms"][phase]["delta_vs_full"]) for r in ok
                  if r["ablation_arms"].get(phase, {}).get("delta_vs_full") is not None]
        changed = sum(1 for d in deltas if d > 1e-4)
        per_phase[phase] = {
            "n": len(deltas), "n_terminal_changed": changed,
            "mean_abs_terminal_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "max_abs_terminal_delta": round(max(deltas), 4) if deltas else None,
            "causally_integrated_on_terminal": changed > 0}
    agg = {"n_questions": len(ok), "activation_rates": {
        k: round(v["executed"] / max(1, v["n"]), 3) for k, v in act.items()},
        "per_phase_ablation": per_phase}
    payload = json.loads(ART.read_text())
    payload["aggregate"] = agg
    ART.write_text(json.dumps(payload, indent=2))
    print("\nAGGREGATE:", json.dumps(agg, indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
