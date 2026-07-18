"""Offline tests for the ForecastBench sealed self-scored track. NO network: a synthetic mini question
set (real schema, synthetic_sample=true) + a stub runner replace the live fetch and the production LLM."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.external import forecastbench_adapter as fba  # noqa: E402
from benchmarks.external import audit as fb_audit  # noqa: E402

FREEZE_TS = "2026-07-18T12:00:00Z"


def _mini_qset():
    """Synthetic mini question set in the VERIFIED real schema (see adapter docstring)."""
    questions = []
    # 4 market questions (manifold/metaculus) — one already closed (ineligible)
    for i, (src, close) in enumerate([("manifold", "2026-09-01T00:00:00+00:00"),
                                      ("manifold", "2026-08-15T00:00:00+00:00"),
                                      ("metaculus", "2026-10-01T00:00:00+00:00"),
                                      ("metaculus", "2026-07-01T00:00:00+00:00")]):  # past close
        questions.append({
            "id": f"mkt{i}", "source": src, "question": f"Will synthetic market event {i} happen?",
            "resolution_criteria": f"Resolves to the market at https://example.test/{i}.",
            "background": "synthetic", "market_info_open_datetime": "2026-01-01T00:00:00+00:00",
            "market_info_close_datetime": close,
            "market_info_resolution_criteria": "N/A", "url": f"https://example.test/{i}",
            "freeze_datetime": "2026-06-25T00:00:00+00:00", "freeze_datetime_value": "0.4",
            "freeze_datetime_value_explanation": "synthetic", "source_intro": "synthetic",
            "resolution_dates": "N/A"})
    # 3 dataset questions — one with only past horizons (ineligible)
    for i, dates in enumerate([["2026-07-12", "2026-08-04", "2026-10-03"],
                               ["2026-08-04", "2027-01-01"],
                               ["2026-07-01", "2026-07-12"]]):  # all past
        questions.append({
            "id": f"ds{i}", "source": "fred", "question": f"Will synthetic series {i} increase?",
            "resolution_criteria": "Resolves from the dataset.", "background": "synthetic",
            "market_info_open_datetime": "N/A", "market_info_close_datetime": "N/A",
            "market_info_resolution_criteria": "N/A", "url": "https://example.test/fred",
            "freeze_datetime": "2026-06-25T00:00:00+00:00", "freeze_datetime_value": "1.0",
            "freeze_datetime_value_explanation": "synthetic", "source_intro": "synthetic",
            "resolution_dates": dates})
    return {"source_url": "synthetic://mini", "question_set_date": "2026-07-05",
            "question_set": "2026-07-05-llm.json", "forecast_due_date": "2026-07-05",
            "sha256": "0" * 64, "synthetic_sample": True, "questions": questions}


class _StubResult:
    def __init__(self, p):
        self.calibrated_probability = p
        self.raw_probability = p if p is not None else None
        self.simulation_status = "completed" if p is not None else "execution_failed"
        self.support_grade = "exploratory"
        self.failure_taxonomy = "" if p is not None else "unavailable_service"
        self.plan_hash = "stubplan"
        self.provenance = {"evidence_bundle_hash": "stubbundle"}
        self.latency_s = 0.01
        self.cost_usd = 0.0


def _stub_runner(question, *, as_of, horizon, llm, seed):
    # deterministic prob from the question text
    if "market event 0" in question:
        return _StubResult(0.9)
    if "series" in question:
        return _StubResult(0.2)
    if "market event 2" in question:
        return _StubResult(None)  # forces prob_fallback path
    return _StubResult(0.6)


def _freeze(tmp_path, seed=0):
    return fba.freeze_eligible_questions(_mini_qset(), max_questions=5, seed=seed,
                                         freeze_ts=FREEZE_TS, out_dir=str(tmp_path))


# ------------------------------------------------------------------ freeze
def test_freeze_deterministic_and_eligibility(tmp_path):
    a = fba.freeze_eligible_questions(_mini_qset(), max_questions=5, seed=0, freeze_ts=FREEZE_TS,
                                      out_dir=str(tmp_path / "a"))
    b = fba.freeze_eligible_questions(_mini_qset(), max_questions=5, seed=0, freeze_ts=FREEZE_TS,
                                      out_dir=str(tmp_path / "b"))
    assert a["seal_sha256"] == b["seal_sha256"], "same seed+freeze_ts must produce identical seals"
    ids = {q["id"] for q in a["doc"]["questions"]}
    assert "mkt3" not in ids, "closed market must be ineligible"
    assert "ds2" not in ids, "dataset question with only past horizons must be ineligible"
    assert a["doc"]["excluded_counts"]["no_future_horizon"] == 2
    # earliest future horizon is 2026-08-04 (ds0/ds1) → scoring_valid_from is the day after
    assert a["scoring_valid_from"] == "2026-08-05"
    # eligible_horizons must all be strictly after the freeze date
    for q in a["doc"]["questions"]:
        assert all(h > "2026-07-18" for h in q["eligible_horizons"])


def test_freeze_excludes_already_resolved(tmp_path):
    out = fba.freeze_eligible_questions(_mini_qset(), max_questions=5, seed=0, freeze_ts=FREEZE_TS,
                                        out_dir=str(tmp_path), exclude_resolved_ids={"mkt0"})
    assert "mkt0" not in {q["id"] for q in out["doc"]["questions"]}
    assert out["doc"]["excluded_counts"]["already_resolved"] == 1


def test_seal_tamper_detection(tmp_path):
    out = _freeze(tmp_path)
    fba.verify_frozen_seal(out["frozen_path"])  # intact → ok
    with open(out["frozen_path"], encoding="utf-8") as f:
        doc = json.load(f)
    doc["questions"][0]["question"] += "?"      # tamper one byte of content
    with open(out["frozen_path"], "w", encoding="utf-8") as f:
        json.dump(doc, f)
    with pytest.raises(ValueError, match="SEAL MISMATCH"):
        fba.verify_frozen_seal(out["frozen_path"])


# ------------------------------------------------------------------ preregister
def test_preregister_writes_chained_rows_and_seal(tmp_path):
    out = _freeze(tmp_path)
    pre = fba.preregister_forecasts(out["frozen_path"], runner=_stub_runner)
    assert pre["n_written"] == out["n_frozen"]
    # seal verifies
    with open(fba._seal_path_for(out["frozen_path"]), encoding="utf-8") as f:
        frozen_seal_sha = json.load(f)["sha256"]
    fba.verify_predictions_seal(pre["predictions_path"], frozen_seal_sha)
    # fallback row flagged with taxonomy
    rows = [json.loads(l) for l in open(pre["predictions_path"], encoding="utf-8") if l.strip()]
    fb_rows = [r for r in rows if r["prob_fallback"]]
    assert fb_rows and all(r["probability"] == 0.5 for r in fb_rows)
    assert all(r["failure_taxonomy"] == "unavailable_service" for r in fb_rows)
    ok_rows = [r for r in rows if not r["prob_fallback"]]
    assert all(r["plan_hash"] == "stubplan" and r["evidence_bundle_hash"] == "stubbundle"
               for r in ok_rows)
    # re-run with skip_existing='non_fallback' only redoes the fallback question
    pre2 = fba.preregister_forecasts(out["frozen_path"],
                                     runner=lambda q, **kw: _StubResult(0.7))
    assert pre2["n_written"] == len(fb_rows)
    fba.verify_predictions_seal(pre["predictions_path"], frozen_seal_sha)


def test_predictions_tamper_detection(tmp_path):
    out = _freeze(tmp_path)
    pre = fba.preregister_forecasts(out["frozen_path"], runner=_stub_runner, limit=3)
    with open(fba._seal_path_for(out["frozen_path"]), encoding="utf-8") as f:
        frozen_seal_sha = json.load(f)["sha256"]
    lines = open(pre["predictions_path"], "rb").read().splitlines()
    row = json.loads(lines[1])
    row["probability"] = 0.99                    # tamper a preregistered forecast
    lines[1] = fba.canonical_json_bytes(row)
    with open(pre["predictions_path"], "wb") as f:
        f.write(b"\n".join(lines) + b"\n")
    with pytest.raises(ValueError, match="SEAL MISMATCH|CHAIN BROKEN"):
        fba.verify_predictions_seal(pre["predictions_path"], frozen_seal_sha)


# ------------------------------------------------------------------ score
def _mini_resolutions():
    return {"forecast_due_date": "2026-07-05", "question_set": "2026-07-05-llm.json",
            "resolutions": [
                # scoreable: after predicted_at
                {"id": "mkt0", "source": "manifold", "direction": None,
                 "resolution_date": "2026-08-20", "resolved_to": 1.0, "resolved": True},
                # SAME question at a later horizon — must NOT be double-counted
                {"id": "mkt0", "source": "manifold", "direction": None,
                 "resolution_date": "2026-09-20", "resolved_to": 1.0, "resolved": True},
                {"id": "ds0", "source": "fred", "direction": None,
                 "resolution_date": "2026-08-04", "resolved_to": 0.0, "resolved": True},
                # ds0 at a later horizon with a DIFFERENT outcome — horizon_used (2026-08-04) must win
                {"id": "ds0", "source": "fred", "direction": None,
                 "resolution_date": "2026-10-03", "resolved_to": 1.0, "resolved": True},
                # must be EXCLUDED: resolved on/before prediction date (pre-prediction evidence)
                {"id": "ds1", "source": "fred", "direction": None,
                 "resolution_date": "2026-07-12", "resolved_to": 1.0, "resolved": True},
                # must be ignored: unresolved / combo rows
                {"id": "mkt1", "source": "manifold", "direction": None,
                 "resolution_date": "2026-08-15", "resolved_to": 0.4, "resolved": False},
                {"id": "mkt0", "source": "manifold", "direction": [1, 1],
                 "resolution_date": "2026-08-20", "resolved_to": 1.0, "resolved": True},
            ]}


def test_scorer_refuses_before_scoring_valid_from(tmp_path):
    out = _freeze(tmp_path)
    pre = fba.preregister_forecasts(out["frozen_path"], runner=_stub_runner)
    res_path = tmp_path / "res.json"
    res_path.write_text(json.dumps(_mini_resolutions()))
    with pytest.raises(RuntimeError, match="SCORING REFUSED"):
        fba.score_frozen(out["frozen_path"], pre["predictions_path"],
                         resolutions_url=str(res_path), now="2026-07-20")


def test_scorer_joins_and_computes_brier(tmp_path):
    out = _freeze(tmp_path)
    pre = fba.preregister_forecasts(out["frozen_path"], runner=_stub_runner)
    res_path = tmp_path / "res.json"
    res_path.write_text(json.dumps(_mini_resolutions()))
    rep = fba.score_frozen(out["frozen_path"], pre["predictions_path"],
                           resolutions_url=str(res_path), now="2026-09-01",
                           report_path=str(tmp_path / "report.json"))
    # stub: mkt0 → p=0.9 outcome 1 → brier 0.01 ; ds0 → p=0.2 outcome 0 at its preregistered
    # horizon (2026-08-04) → brier 0.04. One row per question despite multiple horizon rows.
    assert rep["n_scored"] == 2
    assert rep["mean_brier"] == pytest.approx((0.01 + 0.04) / 2)
    assert rep["n_resolutions_skipped_pre_prediction"] == 1, "pre-prediction outcome must be excluded"
    assert os.path.exists(tmp_path / "report.json")
    by_id = {r["question_id"]: r for r in rep["rows"]}
    assert by_id["mkt0"]["brier"] == pytest.approx(0.01)
    assert by_id["mkt0"]["resolution_date"] == "2026-08-20", "earliest post-prediction row wins"
    assert by_id["ds0"]["brier"] == pytest.approx(0.04)
    assert by_id["ds0"]["horizon_match"] is True
    assert by_id["ds0"]["resolution_date"] == "2026-08-04", "preregistered horizon_used must win"
    assert rep["mean_log_loss"] > 0
    assert sum(b["n"] for b in rep["calibration_bins"]) == 2


def test_scorer_rejects_tampered_frozen(tmp_path):
    out = _freeze(tmp_path)
    pre = fba.preregister_forecasts(out["frozen_path"], runner=_stub_runner, limit=2)
    with open(out["frozen_path"], encoding="utf-8") as f:
        doc = json.load(f)
    doc["scoring_valid_from"] = "2020-01-01"     # attacker tries to unlock early scoring
    with open(out["frozen_path"], "w", encoding="utf-8") as f:
        json.dump(doc, f)
    with pytest.raises(ValueError, match="SEAL MISMATCH"):
        fba.score_frozen(out["frozen_path"], pre["predictions_path"],
                         resolutions_url="unused.json", now="2026-09-01")


def test_scorer_source_is_llm_free():
    """The scorer must be structurally unable to call an LLM: no identifier, argument, or import in
    score_frozen's code mentions an llm / the runtime / a chat backend (docstrings excluded)."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(fba.score_frozen))
    idents = set()
    for node in ast.walk(tree):
        for f in ("id", "attr", "arg", "name", "module"):
            v = getattr(node, f, None)
            if isinstance(v, str):
                idents.add(v.lower())
    banned = {"llm", "simulate_world", "deepseek_chat_fn", "chat_fn", "unified_runtime"}
    assert not (idents & banned), f"scorer references banned identifiers: {idents & banned}"


# ------------------------------------------------------------------ audit
def test_audit_file_schema(tmp_path):
    path = tmp_path / "benchmark_audit.json"
    doc = fb_audit.write_audit(str(path))
    assert path.exists()
    names = {b["benchmark"] for b in doc["benchmarks"]}
    assert names == {"ForecastBench", "ForecastBench-Sim",
                     "FutureSearch BTF (Bench to the Future, BTF-2/BTF-3)",
                     "Metaculus (API / bot tournaments)"}
    for b in doc["benchmarks"]:
        for field in fb_audit.AUDIT_FIELDS:
            assert field in b, f"{b['benchmark']} missing {field}"
    fb = next(b for b in doc["benchmarks"] if b["benchmark"] == "ForecastBench")
    assert fb["recommended_use"].startswith("primary: sealed preregistered self-scored track now")


def test_checked_in_audit_exists():
    assert os.path.exists(fb_audit.DEFAULT_PATH), "run benchmarks/external/audit.py to generate it"
    doc = json.load(open(fb_audit.DEFAULT_PATH, encoding="utf-8"))
    assert len(doc["benchmarks"]) == 4
