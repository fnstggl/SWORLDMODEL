"""ForecastBench adapter — sealed, preregistered, future-scoreable external benchmark track.

Drives the EXACT production runtime (`swm.world_model_v2.unified_runtime.simulate_world`) against the
public ForecastBench question sets (https://github.com/forecastingresearch/forecastbench-datasets,
CC BY-SA 4.0) and produces a tamper-evident, self-scored evaluation:

  1. fetch_latest_question_set  — download + cache the newest public question set (sha256 of raw bytes).
  2. freeze_eligible_questions  — deterministic, source-stratified sample of questions whose outcomes are
     still in the FUTURE at freeze time; writes a frozen JSON + a seal (sha256 of canonical-JSON bytes).
  3. preregister_forecasts      — run the production runtime on each frozen question, append forecast rows
     to a JSONL with a per-row hash CHAIN (genesis = the frozen seal), re-sealing after every row.
  4. score_frozen               — refuses to run before `scoring_valid_from`; verifies both seals; joins
     preregistered forecasts against the OFFICIAL resolution file; computes Brier / log loss / calibration.
     Never calls any LLM and never reads evidence other than the official resolution rows, each of which is
     additionally required to postdate the prediction (`resolution_date` > `predicted_at`).

Verified schema (2026-07-18, round 2026-07-05):
  question set  = {forecast_due_date, question_set, questions:[{id, source, question, resolution_criteria,
                   background, market_info_open_datetime, market_info_close_datetime,
                   market_info_resolution_criteria, url, freeze_datetime, freeze_datetime_value,
                   freeze_datetime_value_explanation, source_intro, resolution_dates}]}
  resolution set= {forecast_due_date, question_set, resolutions:[{id, source, direction, resolution_date,
                   resolved_to, resolved}]}   (direction is null for single questions, a 2-list for combos)

Dataset-source questions (fred/acled/wikipedia/dbnomics/yfinance) carry explicit `resolution_dates`
horizons; market questions (manifold/metaculus/polymarket/infer) carry `resolution_dates == "N/A"` and a
`market_info_close_datetime`.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import re
import urllib.request

RAW_BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main"
API_QUESTION_SETS = ("https://api.github.com/repos/forecastingresearch/forecastbench-datasets"
                     "/contents/datasets/question_sets")
QUESTION_SET_URL = RAW_BASE + "/datasets/question_sets/{date}-llm.json"
RESOLUTION_SET_URL = RAW_BASE + "/datasets/resolution_sets/{date}_resolution_set.json"
LICENSE = "CC BY-SA 4.0 (forecastingresearch/forecastbench-datasets)"

MARKET_SOURCES = ("manifold", "metaculus", "polymarket", "infer")
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(_HERE, "cache")
DEFAULT_FROZEN_DIR = os.path.join(_HERE, "frozen")


# ------------------------------------------------------------------ canonical hashing / seals
def canonical_json_bytes(obj) -> bytes:
    """Deterministic serialization: the sealed representation. Any byte change breaks the seal."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _rfc3339(dt: _dt.datetime) -> str:
    return dt.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seal_path_for(path: str) -> str:
    return re.sub(r"\.(json|jsonl)$", "", path) + ".seal.json"


def write_frozen_seal(frozen_path: str) -> dict:
    """Seal = sha256 of the frozen file's canonical-JSON bytes (parse → re-canonicalize so that the seal
    binds CONTENT, not formatting)."""
    with open(frozen_path, "rb") as f:
        doc = json.loads(f.read().decode("utf-8"))
    seal = {"sealed_file": os.path.basename(frozen_path),
            "algorithm": "sha256(canonical_json)",
            "sha256": sha256_hex(canonical_json_bytes(doc)),
            "sealed_at": _rfc3339(_utc_now())}
    with open(_seal_path_for(frozen_path), "w", encoding="utf-8") as f:
        json.dump(seal, f, indent=1)
    return seal


def verify_frozen_seal(frozen_path: str) -> None:
    """Raises ValueError on any tamper (content hash mismatch) or a missing seal."""
    seal_path = _seal_path_for(frozen_path)
    if not os.path.exists(seal_path):
        raise ValueError(f"no seal file for {frozen_path} (expected {seal_path})")
    with open(seal_path, encoding="utf-8") as f:
        seal = json.load(f)
    with open(frozen_path, "rb") as f:
        doc = json.loads(f.read().decode("utf-8"))
    got = sha256_hex(canonical_json_bytes(doc))
    if got != seal.get("sha256"):
        raise ValueError(f"SEAL MISMATCH for {frozen_path}: content sha256 {got} != sealed {seal.get('sha256')} "
                         "— the frozen file was modified after sealing")


# predictions JSONL: hash chain. C_0 = sha256("genesis|" + frozen_seal_sha). For line i (raw bytes L_i,
# whose JSON already contains prev_chain=C_{i-1}): C_i = sha256(C_{i-1} + sha256(L_i)).
def _chain_genesis(frozen_seal_sha: str) -> str:
    return sha256_hex(f"genesis|{frozen_seal_sha}".encode("utf-8"))


def _chain_next(prev: str, line_bytes: bytes) -> str:
    return sha256_hex((prev + sha256_hex(line_bytes)).encode("utf-8"))


def compute_predictions_chain(predictions_path: str, frozen_seal_sha: str):
    """Recompute the chain over the JSONL file; raises ValueError if any line's embedded prev_chain does
    not match the recomputed chain (i.e. a line was edited, dropped, reordered, or inserted)."""
    chain = _chain_genesis(frozen_seal_sha)
    n = 0
    if os.path.exists(predictions_path):
        with open(predictions_path, "rb") as f:
            for raw in f:
                raw = raw.rstrip(b"\n")
                if not raw:
                    continue
                row = json.loads(raw.decode("utf-8"))
                if row.get("prev_chain") != chain:
                    raise ValueError(f"PREDICTIONS CHAIN BROKEN at line {n + 1}: embedded prev_chain "
                                     f"{row.get('prev_chain')!r} != recomputed {chain!r}")
                chain = _chain_next(chain, raw)
                n += 1
    return chain, n


def write_predictions_seal(predictions_path: str, frozen_seal_sha: str) -> dict:
    chain, n = compute_predictions_chain(predictions_path, frozen_seal_sha)
    seal = {"sealed_file": os.path.basename(predictions_path),
            "algorithm": "sha256 chain: C_i = sha256(C_{i-1} + sha256(line_i)); C_0 = sha256('genesis|'+frozen_seal)",
            "chain_head": chain, "n_rows": n,
            "genesis_frozen_seal_sha256": frozen_seal_sha,
            "sealed_at": _rfc3339(_utc_now())}
    with open(_seal_path_for(predictions_path), "w", encoding="utf-8") as f:
        json.dump(seal, f, indent=1)
    return seal


def verify_predictions_seal(predictions_path: str, frozen_seal_sha: str) -> None:
    seal_path = _seal_path_for(predictions_path)
    if not os.path.exists(seal_path):
        raise ValueError(f"no seal file for {predictions_path}")
    with open(seal_path, encoding="utf-8") as f:
        seal = json.load(f)
    if seal.get("genesis_frozen_seal_sha256") != frozen_seal_sha:
        raise ValueError("predictions seal is chained to a DIFFERENT frozen set "
                         f"({seal.get('genesis_frozen_seal_sha256')} != {frozen_seal_sha})")
    chain, n = compute_predictions_chain(predictions_path, frozen_seal_sha)
    if chain != seal.get("chain_head") or n != seal.get("n_rows"):
        raise ValueError(f"PREDICTIONS SEAL MISMATCH for {predictions_path}: recomputed head {chain} "
                         f"({n} rows) != sealed {seal.get('chain_head')} ({seal.get('n_rows')} rows)")


# ------------------------------------------------------------------ fetch
def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sworldmodel-forecastbench-adapter/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _discover_latest_date(max_lookback_days: int = 70) -> str:
    """Find the newest question-set date. Prefers the GitHub API listing; falls back to probing raw file
    URLs backwards from today (rounds are biweekly, so 70 days covers >=4 rounds)."""
    try:
        listing = json.loads(_http_get(API_QUESTION_SETS, timeout=60).decode("utf-8"))
        dates = sorted(m.group(1) for x in listing
                       if (m := re.match(r"^(\d{4}-\d{2}-\d{2})-llm\.json$", x.get("name", ""))))
        if dates:
            return dates[-1]
    except Exception:  # noqa: BLE001 — API may be unreachable (proxied env); fall through to probing
        pass
    today = _utc_now().date()
    for i in range(max_lookback_days + 1):
        d = (today - _dt.timedelta(days=i)).isoformat()
        try:
            req = urllib.request.Request(QUESTION_SET_URL.format(date=d), method="HEAD",
                                         headers={"User-Agent": "sworldmodel-forecastbench-adapter/1.0"})
            with urllib.request.urlopen(req, timeout=30):
                return d
        except Exception:  # noqa: BLE001 — 404/reset means "not this date"
            continue
    raise RuntimeError(f"no ForecastBench question set found in the last {max_lookback_days} days "
                       "(network blocked or naming changed)")


def fetch_latest_question_set(cache_dir: str = DEFAULT_CACHE_DIR) -> dict:
    """Download + cache the newest question set. Returns {source_url, question_set_date, sha256,
    fetched_at, forecast_due_date, questions:[...]}. sha256 is over the RAW downloaded bytes."""
    os.makedirs(cache_dir, exist_ok=True)
    date = _discover_latest_date()
    url = QUESTION_SET_URL.format(date=date)
    cache_path = os.path.join(cache_dir, f"{date}-llm.json")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            raw = f.read()
    else:
        raw = _http_get(url)
        with open(cache_path, "wb") as f:
            f.write(raw)
    doc = json.loads(raw.decode("utf-8"))
    return {"source_url": url, "question_set_date": date, "sha256": sha256_hex(raw),
            "fetched_at": _rfc3339(_utc_now()), "cache_path": cache_path, "license": LICENSE,
            "forecast_due_date": doc.get("forecast_due_date", date),
            "question_set": doc.get("question_set", f"{date}-llm.json"),
            "questions": doc.get("questions", [])}


def fetch_resolution_set(question_set_date: str, cache_dir: str = DEFAULT_CACHE_DIR,
                         refresh: bool = True) -> dict:
    """Fetch the OFFICIAL resolution set for a round. refresh=True re-downloads (resolution files are
    updated in place as questions resolve)."""
    os.makedirs(cache_dir, exist_ok=True)
    url = RESOLUTION_SET_URL.format(date=question_set_date)
    cache_path = os.path.join(cache_dir, f"{question_set_date}_resolution_set.json")
    if refresh or not os.path.exists(cache_path):
        raw = _http_get(url)
        with open(cache_path, "wb") as f:
            f.write(raw)
    else:
        with open(cache_path, "rb") as f:
            raw = f.read()
    doc = json.loads(raw.decode("utf-8"))
    doc["_source_url"] = url
    doc["_sha256"] = sha256_hex(raw)
    return doc


# ------------------------------------------------------------------ freeze
def _parse_date(s):
    if not s or s == "N/A":
        return None
    try:
        return _dt.date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _eligible_horizons(q: dict, freeze_date: _dt.date):
    """Future scoreable dates for a question, strictly after the freeze date. Dataset questions: their
    listed resolution_dates. Market questions: the market close date (the outcome may resolve earlier,
    but the scorer independently requires resolution_date > predicted_at, so close-date is only used
    for eligibility/scheduling, never for scoring)."""
    rd = q.get("resolution_dates")
    if isinstance(rd, list):
        dates = sorted(d for d in (_parse_date(x) for x in rd) if d is not None and d > freeze_date)
        return [d.isoformat() for d in dates]
    close = _parse_date(q.get("market_info_close_datetime"))
    if close is not None and close > freeze_date:
        return [close.isoformat()]
    return []


def freeze_eligible_questions(qset: dict, *, max_questions: int = 25, seed: int = 0,
                              freeze_ts: str = None, out_dir: str = DEFAULT_FROZEN_DIR,
                              exclude_resolved_ids=None) -> dict:
    """Deterministic, source-stratified freeze of ELIGIBLE questions (>=1 horizon strictly in the future
    at freeze time; not already resolved). Writes forecastbench_frozen_<date>.json + its seal.

    exclude_resolved_ids: ids already resolved per the official resolution file at freeze time (a freeze
    made AFTER the round's due date must not include already-decided questions)."""
    freeze_dt = (_dt.datetime.fromisoformat(freeze_ts.replace("Z", "+00:00")) if freeze_ts else _utc_now())
    freeze_ts = _rfc3339(freeze_dt)
    freeze_date = freeze_dt.date()
    exclude_resolved_ids = set(exclude_resolved_ids or ())

    excluded = {"already_resolved": 0, "no_future_horizon": 0}
    by_source = {}
    for q in qset["questions"]:
        if q["id"] in exclude_resolved_ids:
            excluded["already_resolved"] += 1
            continue
        horizons = _eligible_horizons(q, freeze_date)
        if not horizons:
            excluded["no_future_horizon"] += 1
            continue
        by_source.setdefault(q["source"], []).append((q, horizons))

    # deterministic stratified sample: per-source shuffle (seeded), then round-robin across sources
    rng = random.Random(seed)
    for src in sorted(by_source):
        group = sorted(by_source[src], key=lambda t: t[0]["id"])
        rng.shuffle(group)
        by_source[src] = group
    picked, idx = [], 0
    sources = sorted(by_source)
    while len(picked) < max_questions and any(idx < len(by_source[s]) for s in sources):
        for src in sources:
            if idx < len(by_source[src]) and len(picked) < max_questions:
                picked.append(by_source[src][idx])
        idx += 1

    rows = []
    for q, horizons in picked:
        rows.append({
            "id": q["id"], "source": q["source"], "question": q["question"],
            "resolution_criteria": q.get("resolution_criteria", ""),
            "market_info_resolution_criteria": q.get("market_info_resolution_criteria", "N/A"),
            "background": q.get("background", ""), "url": q.get("url", ""),
            "market_info_open_datetime": q.get("market_info_open_datetime", "N/A"),
            "market_info_close_datetime": q.get("market_info_close_datetime", "N/A"),
            "freeze_datetime": q.get("freeze_datetime", ""),
            "freeze_datetime_value": q.get("freeze_datetime_value", ""),
            "freeze_datetime_value_explanation": q.get("freeze_datetime_value_explanation", ""),
            "resolution_dates": q.get("resolution_dates", "N/A"),
            "eligible_horizons": horizons,            # strictly-future scoreable dates at freeze time
            "freeze_ts": freeze_ts,
        })

    earliest = min((r["eligible_horizons"][0] for r in rows), default=None)
    scoring_valid_from = ((_dt.date.fromisoformat(earliest) + _dt.timedelta(days=1)).isoformat()
                          if earliest else None)
    doc = {
        "benchmark": "ForecastBench",
        "track": "sealed_preregistered_self_scored",
        "license": LICENSE,
        "question_set_date": qset["question_set_date"],
        "question_set_file": qset.get("question_set", ""),
        "question_set_source_url": qset.get("source_url", ""),
        "question_set_sha256": qset.get("sha256", ""),
        "forecast_due_date": qset.get("forecast_due_date", ""),
        "synthetic_sample": bool(qset.get("synthetic_sample", False)),
        "freeze_ts": freeze_ts, "seed": seed, "max_questions": max_questions,
        "n_frozen": len(rows), "excluded_counts": excluded,
        "scoring_valid_from": scoring_valid_from,
        "scoring_rule": ("score only OFFICIAL resolution rows with resolved==true, direction==null, "
                         "resolution_date > predicted_at date; Brier + log loss + calibration bins"),
        "questions": rows,
    }
    os.makedirs(out_dir, exist_ok=True)
    frozen_path = os.path.join(out_dir, f"forecastbench_frozen_{qset['question_set_date']}.json")
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    seal = write_frozen_seal(frozen_path)
    return {"frozen_path": frozen_path, "seal_path": _seal_path_for(frozen_path),
            "seal_sha256": seal["sha256"], "n_frozen": len(rows),
            "scoring_valid_from": scoring_valid_from, "doc": doc}


# ------------------------------------------------------------------ preregister
def _default_runner(question: str, *, as_of: str, horizon: str, llm, seed: int):
    """The EXACT production entry — no wrapper logic beyond the import."""
    from swm.world_model_v2.unified_runtime import simulate_world
    return simulate_world(question, as_of=as_of, horizon=horizon, llm=llm, seed=seed)


def _run_with_timeout(fn, timeout_s: float):
    """Wall-clock guard: run fn() in a worker thread; on timeout the row is recorded as a timeout failure
    (the worker thread cannot be killed and is left to die with the process — acceptable for a harness)."""
    import concurrent.futures
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout_s), None
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False, cancel_futures=True)
        return None, f"per-question wall clock guard exceeded ({timeout_s:.0f}s)"
    except Exception as e:  # noqa: BLE001 — record, don't crash the preregistration loop
        return None, f"{type(e).__name__}: {e}"
    finally:
        if fut.done():
            ex.shutdown(wait=False)


def preregister_forecasts(frozen_path: str, *, llm=None, runner=None, limit: int = None,
                          per_question_timeout_s: float = 480.0, predictions_path: str = None,
                          skip_existing: str = "non_fallback") -> dict:
    """For each frozen question, call the production runtime and append a forecast row to the predictions
    JSONL, RE-SEALING the hash chain after every row. Probability = calibrated_probability if not None
    else raw_probability; 0.5 ONLY with prob_fallback=true + the recorded failure taxonomy.

    runner: injectable stub for tests (same signature as the production call); default = simulate_world.
    skip_existing: 'non_fallback' (default) re-forecasts questions that only have fallback rows;
                   'any' skips every already-predicted id; 'none' always appends. The scorer uses the
                   LAST row per question id, so a later real forecast supersedes an earlier fallback."""
    verify_frozen_seal(frozen_path)
    with open(frozen_path, encoding="utf-8") as f:
        frozen = json.load(f)
    with open(_seal_path_for(frozen_path), encoding="utf-8") as f:
        frozen_seal_sha = json.load(f)["sha256"]

    if predictions_path is None:
        predictions_path = os.path.join(os.path.dirname(frozen_path),
                                        f"predictions_{frozen['question_set_date']}.jsonl")
    # verify the existing chain before appending (refuse to extend a tampered file)
    chain, _n = compute_predictions_chain(predictions_path, frozen_seal_sha)

    done_any, done_real = set(), set()
    if os.path.exists(predictions_path):
        with open(predictions_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done_any.add(row["question_id"])
                    if not row.get("prob_fallback"):
                        done_real.add(row["question_id"])

    try:
        from swm.world_model_v2.runtime_fingerprint import runtime_fingerprint
        fp_hash = runtime_fingerprint()["fingerprint_hash"]
    except Exception:  # noqa: BLE001
        fp_hash = "unavailable"
    runner = runner or _default_runner
    llm_name = getattr(llm, "__name__", None) or (type(llm).__name__ if llm is not None else None)

    written, failures = [], []
    todo = []
    for q in frozen["questions"]:
        if skip_existing == "any" and q["id"] in done_any:
            continue
        if skip_existing == "non_fallback" and q["id"] in done_real:
            continue
        todo.append(q)
    if limit is not None:
        todo = todo[:limit]

    for q in todo:
        as_of = _rfc3339(_utc_now())
        horizon = (q.get("eligible_horizons") or [""])[0]
        res, err = _run_with_timeout(
            lambda q=q, as_of=as_of, horizon=horizon: runner(q["question"], as_of=as_of,
                                                             horizon=horizon, llm=llm, seed=0),
            per_question_timeout_s)
        if res is not None:
            prob = res.calibrated_probability if res.calibrated_probability is not None else res.raw_probability
            fallback = prob is None
            row = {
                "question_id": q["id"], "source": q["source"],
                "probability": 0.5 if fallback else float(prob),
                "prob_fallback": bool(fallback),
                "predicted_at": as_of, "as_of": as_of, "horizon_used": horizon, "seed": 0,
                "simulation_status": res.simulation_status,
                "support_grade": res.support_grade,
                "failure_taxonomy": res.failure_taxonomy,
                "plan_hash": res.plan_hash,
                "evidence_bundle_hash": (res.provenance or {}).get("evidence_bundle_hash", ""),
                "runtime_fingerprint_hash": fp_hash,
                "llm": llm_name, "latency_s": round(float(res.latency_s or 0.0), 3),
                "cost_usd": float(res.cost_usd or 0.0),
                "limitations": [str(x)[:200] for x in (res.limitations or [])[:3]],
            }
            if fallback:
                failures.append({"question_id": q["id"], "simulation_status": res.simulation_status,
                                 "failure_taxonomy": res.failure_taxonomy,
                                 "limitations": row["limitations"]})
        else:
            taxonomy = "timeout" if "wall clock" in (err or "") else "runtime_exception"
            row = {
                "question_id": q["id"], "source": q["source"], "probability": 0.5,
                "prob_fallback": True, "predicted_at": as_of, "as_of": as_of,
                "horizon_used": horizon, "seed": 0,
                "simulation_status": "execution_failed", "support_grade": "",
                "failure_taxonomy": taxonomy, "harness_error": err,
                "plan_hash": "", "evidence_bundle_hash": "",
                "runtime_fingerprint_hash": fp_hash, "llm": llm_name, "latency_s": None,
                "cost_usd": 0.0,
            }
            failures.append({"question_id": q["id"], "simulation_status": "execution_failed",
                             "failure_taxonomy": taxonomy, "harness_error": err})
        row["prev_chain"] = chain
        line_bytes = canonical_json_bytes(row)
        with open(predictions_path, "ab") as f:
            f.write(line_bytes + b"\n")
        chain = _chain_next(chain, line_bytes)
        write_predictions_seal(predictions_path, frozen_seal_sha)      # RE-SEAL after every row
        written.append(row)

    return {"predictions_path": predictions_path, "seal_path": _seal_path_for(predictions_path),
            "n_written": len(written), "n_failures": len(failures), "failures": failures,
            "chain_head": chain, "rows": written}


# ------------------------------------------------------------------ score
def score_frozen(frozen_path: str, predictions_path: str, *, resolutions_url: str = None,
                 now: str = None, report_path: str = None) -> dict:
    """The scorer. NEVER calls any LLM. Refuses to run before scoring_valid_from. Verifies both seals.
    Joins preregistered forecasts (LAST row per question id) against OFFICIAL resolution rows, keeping
    only rows resolved==true, direction==null, and resolution_date strictly AFTER the prediction's
    predicted_at date — the scorer structurally cannot use outcomes knowable at prediction time.

    resolutions_url: override for tests — an http(s) URL or a local file path to a resolution-set JSON.
    now: override 'today' for tests (YYYY-MM-DD)."""
    verify_frozen_seal(frozen_path)
    with open(frozen_path, encoding="utf-8") as f:
        frozen = json.load(f)
    with open(_seal_path_for(frozen_path), encoding="utf-8") as f:
        frozen_seal_sha = json.load(f)["sha256"]
    verify_predictions_seal(predictions_path, frozen_seal_sha)

    today = _dt.date.fromisoformat(now) if now else _utc_now().date()
    svf = frozen.get("scoring_valid_from")
    if not svf:
        raise RuntimeError("frozen set has no scoring_valid_from — nothing is scoreable")
    if today < _dt.date.fromisoformat(svf):
        raise RuntimeError(
            f"SCORING REFUSED: today ({today.isoformat()}) is before scoring_valid_from ({svf}). "
            "The first preregistered outcome has not occurred yet; scoring now would be meaningless "
            "or would require access to future evidence.")

    # official resolutions
    if resolutions_url is None:
        resolutions_url = RESOLUTION_SET_URL.format(date=frozen["question_set_date"])
    if re.match(r"^https?://", resolutions_url):
        raw = _http_get(resolutions_url)
    else:
        with open(resolutions_url, "rb") as f:
            raw = f.read()
    res_doc = json.loads(raw.decode("utf-8"))

    # LAST preregistered row per question id supersedes earlier (e.g. fallback) rows
    preds = {}
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                preds[row["question_id"]] = row

    frozen_ids = {q["id"] for q in frozen["questions"]}
    scored, skipped_pre_prediction = [], 0
    for r in res_doc.get("resolutions", []):
        if not r.get("resolved") or r.get("direction") is not None:
            continue
        qid = r.get("id")
        if qid not in frozen_ids or qid not in preds:
            continue
        p_row = preds[qid]
        pred_date = _dt.date.fromisoformat(p_row["predicted_at"][:10])
        res_date = _parse_date(r.get("resolution_date"))
        if res_date is None or res_date <= pred_date:
            skipped_pre_prediction += 1        # outcome was (potentially) knowable at prediction time
            continue
        outcome = float(r["resolved_to"])
        p = min(max(float(p_row["probability"]), 0.0), 1.0)
        pc = min(max(p, 1e-6), 1 - 1e-6)
        scored.append({"question_id": qid, "source": r.get("source"),
                       "resolution_date": r.get("resolution_date"), "outcome": outcome,
                       "probability": p, "prob_fallback": bool(p_row.get("prob_fallback")),
                       "brier": (p - outcome) ** 2,
                       "log_loss": -(outcome * _ln(pc) + (1 - outcome) * _ln(1 - pc))})

    n = len(scored)
    bins = [{"lo": i / 10, "hi": (i + 1) / 10, "n": 0, "mean_p": None, "mean_outcome": None}
            for i in range(10)]
    for s in scored:
        b = bins[min(int(s["probability"] * 10), 9)]
        b["n"] += 1
        b["mean_p"] = ((b["mean_p"] or 0.0) * (b["n"] - 1) + s["probability"]) / b["n"]
        b["mean_outcome"] = ((b["mean_outcome"] or 0.0) * (b["n"] - 1) + s["outcome"]) / b["n"]

    report = {
        "benchmark": "ForecastBench", "track": "sealed_preregistered_self_scored",
        "frozen_path": os.path.abspath(frozen_path),
        "predictions_path": os.path.abspath(predictions_path),
        "resolutions_source": resolutions_url,
        "resolutions_sha256": sha256_hex(raw),
        "scored_at": _rfc3339(_utc_now()), "scoring_valid_from": svf,
        "n_frozen": len(frozen_ids), "n_predicted": len(preds), "n_scored": n,
        "n_resolutions_skipped_pre_prediction": skipped_pre_prediction,
        "mean_brier": (sum(s["brier"] for s in scored) / n) if n else None,
        "mean_log_loss": (sum(s["log_loss"] for s in scored) / n) if n else None,
        "n_fallback_scored": sum(1 for s in scored if s["prob_fallback"]),
        "calibration_bins": bins, "rows": scored,
        "notes": ["scorer never calls an LLM and reads ONLY the official resolution rows",
                  "official ForecastBench leaderboard uses difficulty-adjusted Brier; this self-scored "
                  "track reports unadjusted Brier and is therefore not directly comparable to the "
                  "public leaderboard"],
    }
    if report_path is None:
        report_path = os.path.join(os.path.dirname(os.path.abspath(frozen_path)),
                                   f"score_report_{frozen['question_set_date']}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    report["report_path"] = report_path
    return report


def _ln(x: float) -> float:
    import math
    return math.log(x)


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    import argparse
    import sys
    _repo_root = os.path.dirname(os.path.dirname(_HERE))
    if _repo_root not in sys.path:      # script-mode fix: make `import swm` resolve to the repo
        sys.path.insert(0, _repo_root)
    ap = argparse.ArgumentParser(description="ForecastBench sealed self-scored track")
    ap.add_argument("cmd", choices=["fetch", "freeze", "preregister", "score"])
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-questions", type=int, default=25)
    ap.add_argument("--llm", action="store_true", help="use the production DeepSeek LLM")
    args = ap.parse_args()
    if args.cmd == "fetch":
        qs = fetch_latest_question_set()
        print(json.dumps({k: v for k, v in qs.items() if k != "questions"}, indent=1))
        print("n questions:", len(qs["questions"]))
    elif args.cmd == "freeze":
        qs = fetch_latest_question_set()
        try:
            res_doc = fetch_resolution_set(qs["question_set_date"])
            resolved = {r["id"] for r in res_doc["resolutions"] if r.get("resolved")}
        except Exception as e:  # noqa: BLE001
            print("warning: could not fetch resolution set for already-resolved filter:", e)
            resolved = set()
        out = freeze_eligible_questions(qs, max_questions=args.max_questions,
                                        exclude_resolved_ids=resolved)
        print(json.dumps({k: v for k, v in out.items() if k != "doc"}, indent=1))
    elif args.cmd == "preregister":
        llm = None
        if args.llm:
            from swm.api.deepseek_backend import deepseek_chat_fn
            llm = deepseek_chat_fn(max_tokens=1400)
        out = preregister_forecasts(args.frozen, llm=llm, limit=args.limit)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    elif args.cmd == "score":
        out = score_frozen(args.frozen, args.predictions)
        print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "calibration_bins")}, indent=1))
