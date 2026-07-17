"""The row executor: for every (question, cutoff), ONE complete production World Model V2 run
through the canonical facade `swm.world_model_v2.unified_runtime.simulate_world`, plus the
same-model same-evidence baseline arms. Append-only resumable ledger; every row carries the
machine-verifiable full-run proof; failures are preserved visibly.

Outcome isolation: this module never imports the resolution store (a sentinel test asserts the
whole forecast-time module tree holds no reference to it). The question vault it reads contains
no outcomes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from historical_backtests.framework import baselines, evidence_build, packs, qualify
from historical_backtests.models.registry import assert_temporal_ordering, get_model, _ts

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = "swm.world_model_v2.unified_runtime.simulate_world"
SEED = 0
MIN_PARTICLES = 200

_FORBIDDEN_IMPORT = ".".join(["historical_backtests", "framework", "resolution" + "_store"])
assert _FORBIDDEN_IMPORT not in sys.modules, "runner must not run in a scorer process"


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_question_vault(benchmark_id: str) -> dict:
    bdir = ROOT / "benchmark_versions" / benchmark_id
    path = bdir / "question_vault.json"
    seal = json.loads((bdir / "question_vault.json.seal").read_text())["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != seal:
        raise RuntimeError("question vault tampered")
    v = json.loads(path.read_text())
    for c in v["cases"]:
        assert "actual_outcome" not in json.dumps(c), "outcome leaked into question vault"
    return v


def _make_llm_strict(model: dict, audit_path: str):
    from swm.api.openrouter_backend import OpenRouterPinnedClient
    rc = model["request_configuration"]
    c = OpenRouterPinnedClient(model["openrouter_slug"],
                               provider=model["openrouter_provider"],
                               quantization=model["quantization"],
                               system="Reply ONLY JSON.",
                               max_tokens=int(rc["max_tokens"]),
                               temperature=float(rc["temperature"]),
                               audit_path=audit_path)
    c.provider_display = model["openrouter_provider_display"]
    return c


def run_row(case: dict, cutoff_iso: str, model: dict, *, results_dir: Path,
            capsule_dir: Path, run_baselines: bool = True) -> dict:
    """One complete production run for one (case, cutoff). Returns the ledger row."""
    t0 = time.time()
    row = {"case_id": case["case_id"], "cutoff": cutoff_iso,
           "raw_question": case["raw_question"], "question_sha256": case["question_sha256"],
           "split": case["split"], "causal_scale": case["causal_scale"],
           "domain": case["domain"], "seed": SEED, "git_head": _git_head(),
           "registry_model_id": model["registry_model_id"],
           "tier": model["temporal_safety_tier"]}
    try:
        # ---- exact-question integrity: byte-for-byte, asserted and recorded ----
        q = case["raw_question"]
        assert hashlib.sha256(q.encode()).hexdigest() == case["question_sha256"], \
            "question hash mismatch — the vault question must enter the compiler unchanged"
        row["question_hash_verified"] = True
        # ---- temporal gate (fail-closed; resolution checked scorer-side) ----
        row["temporal_proof"] = assert_temporal_ordering(
            model, question_open_ts=case["question_open_ts"], cutoff_ts=_ts(cutoff_iso))
        # ---- frozen evidence only (no live retrieval path exists in this process) ----
        bundle = evidence_build.load_bundle(case, cutoff_iso, out_dir=capsule_dir)
        row["evidence_capsule_sha"] = bundle.bundle_hash() or "unhashed"
        row["n_evidence_items"] = len(bundle.claims)
        # ---- walk-forward parameters ----
        row["fitted_packs"] = packs.load_pack(_ts(cutoff_iso))
        # ---- the ONE canonical production facade ----
        audit = str(results_dir / "provider_audit" / f"{case['case_id']}__{cutoff_iso[:10]}.jsonl")
        Path(audit).parent.mkdir(parents=True, exist_ok=True)
        llm = _make_llm_strict(model, audit)
        from swm.world_model_v2.unified_runtime import simulate_world
        horizon = case["resolution_deadline"][:19].rstrip("Z") + "Z" \
            if "T" in str(case["resolution_deadline"]) else str(case["resolution_deadline"])[:10]
        res = simulate_world(q, as_of=cutoff_iso[:10], horizon=horizon, seed=SEED, llm=llm,
                             prebuilt_bundle=bundle)
        prov = getattr(res, "provenance", None) or {}
        evt = prov.get("event_time") or {}
        lin = prov.get("plan_lineage") or {}
        # p(YES) under the unification: F(deadline), polarity-mapped by the runtime
        p_yes = None
        dist = getattr(res, "raw_distribution", None) or {}
        for k, v in dist.items():
            if str(k).lower() in ("yes", "true"):
                p_yes = float(v)
        if p_yes is None and isinstance(evt.get("p_event_by_deadline"), (int, float)):
            p_yes = float(evt["p_event_by_deadline"])
        if p_yes is None and dist:
            p_yes = float(dist.get("absorbed_by_horizon") or 0.0) or None
        row.update({
            "status": getattr(res, "simulation_status", "?"),
            "p_yes": p_yes, "raw_distribution": dist,
            "support_grade": getattr(res, "support_grade", None),
            "plan_hash": getattr(res, "plan_hash", None),
            "event_time": {k: evt.get(k) for k in
                           ("cdf_grid_ts", "cdf", "survival", "first_passage_quantiles_ts",
                            "p_censored", "mode_distribution", "p_event_by_deadline",
                            "deadline_ts", "occurrence_resolves", "n_particles")},
            "lineage_event_time": {k: (lin.get("event_time") or {}).get(k) for k in
                                   ("modes", "hazard_ratio_by_mode", "decision_structures",
                                    "n_stance_reviews", "hr_pack", "coupling_source",
                                    "agreement_hazard_ratio", "n_grounded_stances")},
            "actor_intentions": (lin.get("actor_intentions") or {}),
            "mode_graph_consensus": (lin.get("mode_graph") or {}),
            "limitations": list(getattr(res, "limitations", []) or [])[:6],
        })
        row["full_run_proof"] = qualify.extract_proof(res, entrypoint=ENTRYPOINT,
                                                      runtime_commit=row["git_head"])
        ok, reasons = qualify.qualify(row["full_run_proof"], min_particles=MIN_PARTICLES)
        row["qualified"], row["disqualify_reasons"] = ok, reasons
        row["llm_usage"] = {"n_calls": llm.n_calls, "total_tokens": llm.total_tokens,
                            "cost_usd": round(llm.total_cost, 5)}
        if run_baselines:
            ev_text = bundle.render(max_chars=6000)
            mkt = (case.get("market_snapshots") or {}).get(cutoff_iso, {}).get("market_price")
            b_audit = str(results_dir / "provider_audit" /
                          f"{case['case_id']}__{cutoff_iso[:10]}__baselines.jsonl")
            bllm = _make_llm_strict(model, b_audit)
            bllm.system = ""                                 # baselines answer in JSON via prompt
            row["baselines"] = baselines.run_all(bllm, q, cutoff_iso, ev_text, mkt)
            row["baseline_usage"] = {"n_calls": bllm.n_calls, "total_tokens": bllm.total_tokens,
                                     "cost_usd": round(bllm.total_cost, 5)}
    except Exception as e:  # noqa: BLE001 — a failed row is preserved, never dropped or retried away
        row.update({"status": "runner_exception", "qualified": False,
                    "disqualify_reasons": [f"runner_exception:{type(e).__name__}: {e}"[:300]]})
    row["latency_s"] = round(time.time() - t0, 1)
    row["predicted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return row


def seal_ledger(ledger: Path):
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    ledger.with_suffix(".jsonl.seal").write_text(json.dumps(
        {"sha256": digest, "sealed_at": time.time(), "n_lines": ledger.read_text().count("\n")}))
    return digest


def run_benchmark(benchmark_id: str, *, splits=("calibration", "validation", "rotating_locked"),
                  limit: int | None = None, with_baselines: bool = True,
                  only_cases: list | None = None):
    vault = load_question_vault(benchmark_id)
    model = get_model(vault["registry_model_id"])
    run_id = f"runtime_{_git_head()}"
    results_dir = ROOT / "results" / benchmark_id / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    capsule_dir = ROOT / "evidence_archives" / benchmark_id
    ledger = results_dir / "forecast_ledger.jsonl"
    done = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["case_id"], r["cutoff"]))
    todo = [(c, cut) for c in vault["cases"] if c["split"] in splits
            and (only_cases is None or c["case_id"] in only_cases)
            for cut in c["forecast_cutoffs"] if (c["case_id"], cut) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{time.strftime('%H:%M:%S')}] benchmark={benchmark_id} run={run_id} "
          f"done={len(done)} todo={len(todo)} splits={splits}", flush=True)
    for i, (case, cut) in enumerate(todo):
        row = run_row(case, cut, model, results_dir=results_dir, capsule_dir=capsule_dir,
                      run_baselines=with_baselines)
        with ledger.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        seal_ledger(ledger)                                  # crash-safe re-seal every row
        print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(todo)} {row['case_id']} "
              f"@{cut[:10]} p={row.get('p_yes')} q={row.get('qualified')} "
              f"status={row.get('status')} ({row['latency_s']}s "
              f"${(row.get('llm_usage') or {}).get('cost_usd', 0)}) "
              f"{case['raw_question'][:56]}", flush=True)
    return ledger
