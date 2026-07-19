"""PRE-REGISTER full-simulation forecasts for every frozen vault question — before outcomes exist.

The freeze recorded questions + market prices; this runner records the SYSTEM'S OWN forecasts,
produced by the FULL production pipeline per question (compile → as-of evidence → posterior →
fidelity/stances/mode graph → event-time conversion → 200-particle rollout → first-passage
readout) with the DeepSeek LLM, and writes them to an append-only ledger BEFORE the markets
resolve — pre-registered predictions, sealed (SHA-256) and committed, so scoring later cannot
tune anything.

Order: NEAREST DEADLINE FIRST — near-tranche questions resolve today/tomorrow; every row records
`predicted_at`, and the scorer flags any row whose prediction landed after its market's end as
`post_deadline` (excluded from the pre-registered headline, honestly).

Resumable: rows are keyed by condition_id in event_time_vault_predictions.jsonl; reruns skip
completed rows and re-seal.

Run:  PYTHONPATH=. DEEPSEEK_API_KEY=… python experiments/replay_v3/predict_event_time_vault.py
"""
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.replay_v3.build_event_time_vault import OUT, VAULT, canonical_bytes

PRED = VAULT / "event_time_vault_predictions.jsonl"
PRED_SEAL = VAULT / "event_time_vault_predictions_seal.json"
MODEL = "deepseek-chat"
PER_QUESTION_TIMEOUT_S = 900.0


def _now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def seal_predictions():
    rows = [json.loads(l) for l in PRED.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["condition_id"])
    digest = hashlib.sha256(canonical_bytes(rows)).hexdigest()
    PRED_SEAL.write_text(json.dumps({
        "file": PRED.name, "sha256": digest, "n_predictions": len(rows),
        "sealed_at": _now_iso(),
        "note": ("pre-registered forecasts: each row's predicted_at is the moment the full "
                 "simulation finished; the scorer flags rows predicted after their market's end "
                 "as post_deadline and excludes them from the pre-registered headline")},
        indent=1))
    return digest, len(rows)


def main(retry_failed: bool = None):
    import argparse
    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world
    if retry_failed is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--retry-failed", action="store_true",
                        help="re-run rows whose latest attempt failed (new attempt rows appended; "
                             "the ledger keeps every attempt — scoring uses the latest pre-deadline "
                             "row per market)")
        retry_failed = ap.parse_args().retry_failed
    vault = json.loads(OUT.read_text())
    as_of_iso = vault["frozen_at"][:10]
    latest, attempts = {}, {}
    if PRED.exists():
        for line in PRED.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["condition_id"]] = r
                attempts[r["condition_id"]] = attempts.get(r["condition_id"], 0) + 1
    def _failed(r):
        return r.get("status") in ("execution_failed", "runner_exception", "clarification_required")
    done = {cid for cid, r in latest.items() if not (retry_failed and _failed(r))}
    todo = [w for w in vault["questions"] if w["condition_id"] not in done]
    todo.sort(key=lambda w: w["end_ts"])                     # nearest deadline first
    print(f"[{_now_iso()}] {len(done)} done, {len(todo)} to predict "
          f"(as_of={as_of_iso}, model={MODEL}, head={_git_head()})", flush=True)
    llm = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=2400, temperature=0.2)
    for i, w in enumerate(todo):
        t0 = time.time()
        # horizon = the market's exact scheduled end (intraday-safe); as_of = the freeze day
        horizon = str(w["end_date"]).replace("+00:00", "Z")[:20].rstrip("Z") + "Z" \
            if "T" in str(w["end_date"]) else str(w["end_date"])[:10]
        row = {"condition_id": w["condition_id"], "question": w["question"],
               "tranche": w.get("tranche", "far"), "end_ts": w["end_ts"],
               "as_of": as_of_iso, "model": MODEL, "seed": 0, "git_head": _git_head(),
               "attempt": attempts.get(w["condition_id"], 0) + 1}
        try:
            res = simulate_world(w["question"], as_of=as_of_iso, horizon=horizon, llm=llm, seed=0)
            prov = getattr(res, "provenance", None) or {}
            evt = prov.get("event_time") or {}
            proj = getattr(res, "raw_distribution", None) or {}
            p_yes = None
            for k, v in proj.items():
                if str(k).lower() in ("yes", "true"):
                    p_yes = float(v)
            if p_yes is None and isinstance(evt.get("p_event_by_deadline"), (int, float)):
                p_yes = float(evt["p_event_by_deadline"])
            lin = (prov.get("plan_lineage") or {}).get("event_time") or {}
            row.update({
                "status": getattr(res, "simulation_status", "?"),
                "p_yes": p_yes, "raw_distribution": proj,
                "event_time": {k: evt.get(k) for k in
                               ("cdf_grid_ts", "cdf", "survival", "first_passage_quantiles_ts",
                                "p_censored", "mode_distribution", "p_event_by_deadline",
                                "deadline_ts", "occurrence_resolves", "n_particles")},
                "support_grade": getattr(res, "support_grade", None),
                "plan_hash": getattr(res, "plan_hash", None),
                "hr_pack": lin.get("hr_pack"), "coupling_source": lin.get("coupling_source"),
                "n_evidence_claims": len((prov.get("plan_lineage") or {})
                                         .get("actor_intentions", {}).get("intentions", []) or []),
                "limitations": list(getattr(res, "limitations", []) or [])[:4],
            })
        except Exception as e:  # noqa: BLE001 — a failed row is a recorded failure, never dropped
            row.update({"status": "runner_exception", "error": f"{type(e).__name__}: {e}"[:300]})
        row["latency_s"] = round(time.time() - t0, 1)
        row["predicted_at"] = _now_iso()
        row["pre_deadline"] = time.time() < float(w["end_ts"])
        with PRED.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        left_h = (w["end_ts"] - time.time()) / 3600.0
        print(f"[{_now_iso()}] {i + 1}/{len(todo)} {w['condition_id'][:10]} "
              f"p_yes={row.get('p_yes')} status={row.get('status')} "
              f"({row['latency_s']}s; deadline in {left_h:+.1f}h; "
              f"pre_deadline={row['pre_deadline']}) {w['question'][:60]}", flush=True)
        digest, n = seal_predictions()                       # re-seal after every row (crash-safe)
    digest, n = seal_predictions()
    print(f"[{_now_iso()}] sealed {n} pre-registered predictions → {PRED_SEAL} sha={digest[:16]}…",
          flush=True)


if __name__ == "__main__":
    main()
