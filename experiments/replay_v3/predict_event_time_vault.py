"""PRE-REGISTER V2 forecasts for the frozen event-time vault — while the markets are still OPEN.

Why this exists: the scorer's original design ran the LLM at scoring time, after outcomes exist —
retrieval could leak the outcome into the forecast. Pre-registration closes that hole. Every
forecast here is generated while its market is still open and undecided, stamped with the live
market state at prediction time (proof of openness), checkpointed, then sealed as a per-tranche
predictions file. Scoring later uses ONLY the sealed predictions — no LLM runs at scoring time.

Gates, per question, at prediction time:
 * VAULT SEAL GATE: the vault seal must verify before any forecast is attempted.
 * OPEN-MARKET GATE: if the market's scheduled end has passed, or the live market says closed,
   or the live price is outside (0.02, 0.98) — the world has effectively decided — the question
   is recorded as `not_preregisterable` with the reason, and is NEVER forecast after the fact.
 * STRICT COMPLETION RULE: the scorer only accepts a row whose `completed_at` precedes the
   market's scheduled end (enforced again at scoring time from the sealed record).

Mechanics:
 * questions are processed in ascending end-time order (the soonest-resolving markets first);
 * one checkpoint JSON per question under predictions/<tranche>/ — reruns skip finished
   checkpoints, so a crash never forces re-forecasting an already-registered question;
 * failures are checkpointed as `failed` (with the error) and retried on rerun via --retry-failed;
 * --finalize assembles the tranche's checkpoints into event_time_vault_predictions_<tranche>.json
   plus a SHA-256 seal; an existing sealed predictions file is NEVER overwritten (append-only).

Requires network + DEEPSEEK_API_KEY.
"""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.replay_v3.build_event_time_vault import OUT, SEAL, VAULT, canonical_bytes

PRED_DIR = VAULT / "predictions"
LOG_DIR = Path("experiments/results/replay_v3/prereg_logs")
MODEL = "deepseek-chat"
DECIDED_LO, DECIDED_HI = 0.02, 0.98
#: an extreme price only means "the world already knows" when resolution is IMMINENT — a 1.5%
#: far-dated longshot is a genuinely open question, not a decided one
DECIDED_WINDOW_S = 72 * 3600.0


def predictions_path(tranche: str) -> Path:
    return VAULT / f"event_time_vault_predictions_{tranche}.json"


def predictions_seal_path(tranche: str) -> Path:
    return VAULT / f"event_time_vault_predictions_{tranche}_seal.json"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()[:12]
    except Exception:  # noqa: BLE001 — provenance stamp only, never blocks a forecast
        return "unknown"


def _seed_for(condition_id: str) -> int:
    """Deterministic per-question seed — documented, reproducible, independent across questions."""
    return int(hashlib.sha256(str(condition_id).encode()).hexdigest()[:8], 16)


def verify_vault_seal() -> dict:
    """The vault must be intact before any forecast is registered against it."""
    seal = json.loads(SEAL.read_text())
    vault = json.loads(OUT.read_text())
    digest = hashlib.sha256(canonical_bytes(vault)).hexdigest()
    if digest != seal["sha256"]:
        raise SystemExit(f"SEAL MISMATCH: vault sha256 {digest[:16]}… != sealed "
                         f"{seal['sha256'][:16]}… — refusing to pre-register against a touched vault")
    return vault


def live_market_state(condition_id: str) -> dict:
    """Live openness proof, recorded into the prediction: price + closed flag at prediction time.
    NOTE: fit_survival_pack._market_by_condition filters closed=true (built for resolved markets) —
    an open market needs the unfiltered query."""
    import experiments.replay_v2.build_vault as V2B
    ms = V2B._get(f"{V2B.GAMMA}/markets?condition_ids={condition_id}") or []
    m = ms[0] if ms else {}
    state = {"fetched_at": _now_iso(), "closed": bool(m.get("closed"))}
    try:
        prices = json.loads(m.get("outcomePrices") or "[]")
        outs = [str(o).lower() for o in json.loads(m.get("outcomes") or "[]")]
        state["p_yes"] = float(prices[outs.index("yes")])
    except (ValueError, TypeError, IndexError):
        state["p_yes"] = None
    return state


def openness_gate(w: dict, live: dict, now_ts: float) -> str:
    """'' when forecastable; otherwise the reason this question cannot be pre-registered.

    The pinned-price refusal is TIME-AWARE: within DECIDED_WINDOW_S of the scheduled end an
    extreme price means the outcome is effectively known (a 0.99 hours before resolution IS the
    answer); on a far-dated market the same price is just a longshot forecast — the question is
    open and refusing to pre-register it would silently bias the tranche toward mid-price rows."""
    if now_ts >= float(w["end_ts"]):
        return "market end passed before prediction started"
    if live.get("closed"):
        return "market reports closed at prediction time"
    p = live.get("p_yes")
    if p is not None and not (DECIDED_LO < p < DECIDED_HI) \
            and (float(w["end_ts"]) - now_ts) <= DECIDED_WINDOW_S:
        return f"market effectively decided at prediction time (p_yes={p:.3f})"
    return ""


def p_at_ts(evt: dict, ts: float):
    """P(question resolves YES by ts), interpolated from the stored first-passage CDF — the exact
    quantity scoring compares to the outcome. Polarity-mapped: survival questions invert F."""
    grid, cdf = list(evt.get("cdf_grid_ts") or []), list(evt.get("cdf") or [])
    if not grid or not cdf or len(grid) != len(cdf):
        return None
    if ts <= grid[0]:
        f = cdf[0] * max(0.0, min(1.0, ts / grid[0] if grid[0] else 0.0))
        f = cdf[0] if ts >= grid[0] else f
    elif ts >= grid[-1]:
        f = cdf[-1]
    else:
        f = cdf[-1]
        for i in range(1, len(grid)):
            if ts <= grid[i]:
                g0, g1, c0, c1 = grid[i - 1], grid[i], cdf[i - 1], cdf[i]
                f = c0 + (c1 - c0) * ((ts - g0) / (g1 - g0) if g1 > g0 else 0.0)
                break
    return round(1.0 - f, 4) if str(evt.get("occurrence_resolves", "yes")) == "no" else round(f, 4)


def _checkpoint(tranche: str, condition_id: str) -> Path:
    return PRED_DIR / tranche / f"{condition_id}.json"


def _write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def predict_one(w: dict, as_of_iso: str, tranche: str) -> dict:
    """Forecast ONE frozen question with the full production V2 system. Runs in a worker process;
    writes its own checkpoint atomically so parent death loses nothing."""
    import contextlib

    ck = _checkpoint(tranche, w["condition_id"])
    row = {"condition_id": w["condition_id"], "question": w["question"], "tranche": tranche,
           "end_date": w["end_date"], "end_ts": w["end_ts"],
           "event_cluster": w.get("event_cluster"),
           "market_p_yes_at_freeze": w["market_p_yes_at_freeze"],
           "as_of": as_of_iso, "model": MODEL, "seed": _seed_for(w["condition_id"]),
           "code_git_sha": _git_sha(), "started_at": _now_iso()}
    live = live_market_state(w["condition_id"])
    row["live_market_at_prediction"] = live
    reason = openness_gate(w, live, time.time())
    if reason:
        row.update(status="not_preregisterable", reason=reason, completed_at=_now_iso())
        _write_atomic(ck, row)
        return row

    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world
    llm = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=2400, temperature=0.2)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{tranche}_{w['condition_id'][:18]}.log"
    # horizon = the day AFTER the market's end TIMESTAMP. The date-floor of end_date is WRONG twice
    # over: a market ending 16:00 today floors to as_of (zero-length window, nothing can fire) and a
    # match played tomorrow evening floors to tomorrow MIDNIGHT (window ends before the event). The
    # scored quantity is F(end_ts) interpolated from the stored CDF, so the +1d ceiling costs nothing.
    horizon_iso = (dt.datetime.fromtimestamp(float(w["end_ts"]), dt.timezone.utc)
                   + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    row["horizon"] = horizon_iso
    t0 = time.time()
    try:
        with open(log_path, "w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            res = simulate_world(w["question"], as_of=as_of_iso,
                                 horizon=horizon_iso, llm=llm, seed=row["seed"])
        proj = getattr(res, "raw_distribution", None) or {}
        evt = (getattr(res, "provenance", None) or {}).get("event_time") or {}
        p_yes = None
        for k, v in proj.items():
            if str(k).lower() in ("yes", "true"):
                p_yes = float(v)
        if p_yes is None and isinstance(evt.get("p_event_by_deadline"), (int, float)):
            p_yes = float(evt["p_event_by_deadline"])
        sim_status = str(getattr(res, "simulation_status", "?"))
        usable = bool(evt.get("cdf_grid_ts") and evt.get("cdf")) or p_yes is not None
        p_end = p_at_ts(evt, float(w["end_ts"]))
        row.update(status="predicted" if usable else "failed",
                   simulation_status=sim_status,
                   p_yes=p_yes, p_at_market_end=p_end,
                   raw_distribution=proj, event_time=evt,
                   wall_seconds=round(time.time() - t0, 1), completed_at=_now_iso(),
                   trace_log=str(log_path))
        if not usable:
            row["error"] = (f"unusable forecast: status={sim_status} "
                            f"taxonomy={getattr(res, 'failure_taxonomy', '')} "
                            f"lim={str(getattr(res, 'limitations', ''))[:200]}")
    except Exception as e:  # noqa: BLE001 — a failed forecast is a recorded failure, never dropped
        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:400],
                   wall_seconds=round(time.time() - t0, 1), completed_at=_now_iso())
    _write_atomic(ck, row)
    return row


def finalize(tranche: str, vault: dict) -> None:
    """Assemble the tranche's checkpoints into the sealed predictions file — append-only."""
    out_path, seal_path = predictions_path(tranche), predictions_seal_path(tranche)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists — sealed predictions are never overwritten")
    targets = [w for w in vault["questions"] if str(w.get("tranche", "far")) == tranche]
    rows, missing = [], []
    for w in targets:
        ck = _checkpoint(tranche, w["condition_id"])
        if ck.exists():
            rows.append(json.loads(ck.read_text()))
        else:
            missing.append(w["condition_id"])
    if missing:
        raise SystemExit(f"{len(missing)} questions have no checkpoint yet — run predictions first "
                         f"(missing: {missing[:3]}…)")
    rows.sort(key=lambda r: str(r["condition_id"]))
    doc = {"version": "event-time-predictions-1.0", "tranche": tranche,
           "vault_sha256": json.loads(SEAL.read_text())["sha256"],
           "finalized_at": _now_iso(), "model": MODEL, "code_git_sha": _git_sha(),
           "governance": ("pre-registered forecasts: generated while every included market was "
                          "open and undecided; scoring uses ONLY this sealed file, no LLM at "
                          "scoring time; rows whose completed_at is not before the market end "
                          "are excluded at scoring"),
           "n_rows": len(rows),
           "n_predicted": sum(1 for r in rows if r.get("status") == "predicted"),
           "n_not_preregisterable": sum(1 for r in rows if r.get("status") == "not_preregisterable"),
           "n_failed": sum(1 for r in rows if r.get("status") == "failed"),
           "rows": rows}
    out_path.write_text(json.dumps(doc, indent=1, default=str))
    digest = hashlib.sha256(canonical_bytes(doc)).hexdigest()
    seal_path.write_text(json.dumps({"file": out_path.name, "sha256": digest,
                                     "sealed_at": doc["finalized_at"]}, indent=1))
    print(f"sealed {doc['n_predicted']} predictions ({doc['n_not_preregisterable']} not_preregisterable, "
          f"{doc['n_failed']} failed) → {out_path}\nseal {digest[:16]}… → {seal_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", choices=("near", "far"), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="forecast at most N pending questions")
    ap.add_argument("--only", default="", help="comma-separated condition_ids to run")
    ap.add_argument("--retry-failed", action="store_true", help="re-run checkpoints whose status=failed")
    ap.add_argument("--finalize", action="store_true", help="assemble + seal the tranche predictions file")
    args = ap.parse_args()

    vault = verify_vault_seal()
    if args.finalize:
        finalize(args.tranche, vault)
        return
    as_of_iso = vault["frozen_at"][:10]
    targets = sorted([w for w in vault["questions"] if str(w.get("tranche", "far")) == args.tranche],
                     key=lambda w: float(w["end_ts"]))
    if args.only:
        keep = {c.strip() for c in args.only.split(",") if c.strip()}
        targets = [w for w in targets if w["condition_id"] in keep]
    pending = []
    for w in targets:
        ck = _checkpoint(args.tranche, w["condition_id"])
        if ck.exists():
            st = json.loads(ck.read_text()).get("status")
            if st == "failed" and args.retry_failed:
                pending.append(w)
            continue
        pending.append(w)
    if args.limit:
        pending = pending[:args.limit]
    print(f"[{_now_iso()}] tranche={args.tranche} total={len(targets)} pending={len(pending)} "
          f"workers={args.workers}", flush=True)
    if not pending:
        print("nothing pending — use --finalize to seal", flush=True)
        return
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(predict_one, w, as_of_iso, args.tranche): w for w in pending}
        for fut in as_completed(futs):
            w = futs[fut]
            done += 1
            try:
                r = fut.result()
                p = r.get("p_yes")
                print(f"[{_now_iso()}] {done}/{len(pending)} {r['status']:20s} "
                      f"p_yes={p if p is None else round(p, 3)} "
                      f"({r.get('wall_seconds', 0)}s) {w['question'][:70]}", flush=True)
            except Exception as e:  # noqa: BLE001 — keep the batch alive; the checkpoint is the record
                print(f"[{_now_iso()}] {done}/{len(pending)} WORKER-ERROR {type(e).__name__}: "
                      f"{str(e)[:150]} {w['question'][:60]}", flush=True)
    print(f"[{_now_iso()}] batch complete", flush=True)


if __name__ == "__main__":
    main()
