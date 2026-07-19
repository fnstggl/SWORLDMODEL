"""Historical backtest lab — enforcement gates (all mocked; no network, no outcomes)."""
import hashlib
import json
import os
import subprocess
import sys
import time
import types
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swm.api.openrouter_backend import (OpenRouterEnforcementError, OpenRouterPinnedClient)


def _resp(model="meta-llama/llama-3.1-70b-instruct", provider="WandB", gid="gen-1",
          usage=True, text='{"ok": 1}'):
    r = {"model": model, "provider": provider, "id": gid,
         "choices": [{"message": {"content": text}}]}
    if usage:
        r["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                      "cost": 1e-5}
    return r


def _client(transport, tmp_path=None):
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key-never-real")
    return OpenRouterPinnedClient("meta-llama/llama-3.1-70b-instruct", provider="wandb",
                                  quantization="bf16",
                                  audit_path=(str(tmp_path / "audit.jsonl") if tmp_path else None),
                                  _transport=transport)


def test_exact_model_enforced():
    c = _client(lambda u, b, t: _resp(model="meta-llama/llama-3.3-70b-instruct"))
    with pytest.raises(OpenRouterEnforcementError, match="pinned"):
        c("hi")


def test_provider_pin_no_fallback():
    c = _client(lambda u, b, t: _resp(provider="DeepInfra"))
    c.provider_display = "WandB"
    with pytest.raises(OpenRouterEnforcementError, match="provider"):
        c("hi")


def test_missing_generation_id_and_usage_fail():
    with pytest.raises(OpenRouterEnforcementError, match="generation id"):
        _client(lambda u, b, t: _resp(gid=""))("hi")
    with pytest.raises(OpenRouterEnforcementError, match="usage"):
        _client(lambda u, b, t: _resp(usage=False))("hi")


def test_request_body_pins_provider_and_denies_fallback(tmp_path):
    seen = {}

    def transport(url, body, timeout):
        seen.update(body)
        return _resp()
    c = _client(transport, tmp_path)
    c.provider_display = "WandB"
    out = c("hello")
    assert out == '{"ok": 1}'
    assert seen["provider"] == {"order": ["wandb"], "allow_fallbacks": False,
                                "require_parameters": True, "data_collection": "deny",
                                "quantizations": ["bf16"]}
    assert seen["model"] == "meta-llama/llama-3.1-70b-instruct"
    row = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert row["provider_returned"] == "WandB" and row["generation_id"] == "gen-1"
    assert row["response_sha256"] == hashlib.sha256(b'{"ok": 1}').hexdigest()
    assert "api_key" not in json.dumps(row).lower()


def test_retry_same_config_then_success():
    calls = {"n": 0}

    def transport(url, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(url, 503, "unavailable", None, None)
        return _resp()
    c = _client(transport)
    c.provider_display = "WandB"
    assert c("hi") == '{"ok": 1}'
    assert calls["n"] == 2                                   # bounded retry, identical endpoint


def test_temporal_ordering_gate():
    from historical_backtests.models.registry import assert_temporal_ordering, get_model
    m = get_model("llama31_70b_instruct_post_release")
    rel = 1721779199.0                                       # 2024-07-23T23:59:59Z
    with pytest.raises(ValueError, match="strictly after"):
        assert_temporal_ordering(m, question_open_ts=rel - 5, cutoff_ts=rel + 100)
    with pytest.raises(ValueError, match=">= question_open"):
        assert_temporal_ordering(m, question_open_ts=rel + 100, cutoff_ts=rel + 50)
    with pytest.raises(ValueError, match="precede resolution"):
        assert_temporal_ordering(m, question_open_ts=rel + 100, cutoff_ts=rel + 200,
                                 resolution_ts=rel + 200)
    proof = assert_temporal_ordering(m, question_open_ts=rel + 100, cutoff_ts=rel + 200,
                                     resolution_ts=rel + 300)
    assert "model_release < question_open <= cutoff < resolution" == proof["ordering"]


def test_scale_classifier_and_exclusions():
    from historical_backtests.framework.scales import classify_scale, excluded, proper_nouns
    assert classify_scale("Will the CEO of Company X resign before 2025?")[0] \
        == "single_decision_maker"
    assert classify_scale("Will the Federal Reserve cut rates in September?")[0] \
        == "small_group_decision"
    assert classify_scale("Will Congress pass the bill before April 30?")[0] \
        == "institutional_process"
    assert classify_scale("Will Russia and Ukraine sign a ceasefire before 2026?")[0] \
        == "multi_actor_strategic"
    assert classify_scale("Will EV sales exceed 10% of new vehicle sales?")[0] \
        == "broad_aggregate"
    assert classify_scale("Will Candidate X win the election?")[0] == "mixed_scale"
    assert excluded("Will Bitcoin hit $100k?")
    assert excluded("Will the Lakers win the NBA title?")
    assert not excluded("Will the ceasefire hold through March?")
    assert "SpaceX" in proper_nouns("Will SpaceX complete an initial public offering?")


def test_contamination_scrub():
    from historical_backtests.framework.evidence_build import contamination_scrub
    cutoff = time.mktime(time.strptime("2025-03-01", "%Y-%m-%d"))
    bad = {"text": "The deal was signed the following week. On 2025-06-11 the agreement "
                   "took effect, and by 2025-08-02 it was fully implemented."}
    assert contamination_scrub(bad, cutoff, ["Agreement"]) is not None
    ok = {"text": "Talks continued in Doha on 2025-02-20 with no resolution announced."}
    assert contamination_scrub(ok, cutoff, ["Doha"]) is None


def test_walk_forward_pack_isolation(tmp_path, monkeypatch):
    from historical_backtests.framework import packs
    monkeypatch.setattr(packs, "SNAP_DIR", tmp_path)
    good = {"families": {}, "global_hazards": [0.01] * 5,
            "walk_forward": {"snapshot_boundary": "2025-01-01", "n_training_rows": 30,
                             "latest_included_resolution_ts": 1735600000.0,
                             "min_source_ts": 1710000000.0, "fitted_at": 1735700000.0,
                             "source_hash": "x", "training_code": "t"}}
    (tmp_path / "survival_pack_asof_2025-01-01.json").write_text(json.dumps(good))
    rec = packs.load_pack(time.mktime(time.strptime("2025-03-01", "%Y-%m-%d")))
    assert rec["survival_pack"]["snapshot_boundary"] == "2025-01-01"
    assert rec["intention_hr_pack"]["fallback_reason"] == "insufficient_pre_cutoff_fit_data"
    # a future-resolved training row can NEVER affect an earlier forecast
    bad = json.loads(json.dumps(good))
    bad["walk_forward"]["latest_included_resolution_ts"] = 9e9
    (tmp_path / "survival_pack_asof_2025-01-01.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="walk-forward violation"):
        packs.load_pack(time.mktime(time.strptime("2025-03-01", "%Y-%m-%d")))
    # cutoff earlier than every snapshot → documented-prior fallback, recorded
    rec2 = packs.load_pack(time.mktime(time.strptime("2024-08-02", "%Y-%m-%d")))
    assert rec2["survival_pack"] is None
    assert rec2["fallback_reason"] == "insufficient_pre_cutoff_fit_data"


def _proof(**over):
    from historical_backtests.framework.qualify import current_phase_contract
    base = {"simulation_status": "completed",
            "phase_execution_records": {p: {"relevant": p in ("phase1_compiler",
                                                              "phase4_actor_policy"),
                                            "execution_status": "causally_active",
                                            "n_state_deltas": 5}
                                        for p in current_phase_contract()},
            "terminal_source": "simulated_world_states/first_passage_readout",
            "event_time_readout": {"p_event_by_deadline": 0.3, "n_particles": 200},
            "n_particles": 200, "n_actor_action_deltas": 40}
    base.update(over)
    return base


def test_qualification_gates():
    from historical_backtests.framework.qualify import qualify
    ok, why = qualify(_proof())
    assert ok, why
    # SENTINEL: remove one required production phase → row FAILS, never silently succeeds
    p = _proof()
    del p["phase_execution_records"]["phase4_actor_policy"]
    ok, why = qualify(p)
    assert not ok and any("missing_phase_records" in r for r in why)
    ok, why = qualify(_proof(n_particles=30))
    assert not ok and any("particle_floor" in r for r in why)
    ok, why = qualify(_proof(event_time_readout=None))
    assert not ok
    ok, why = qualify(_proof(n_actor_action_deltas=0))
    assert not ok and any("actor_decisions_expected" in r for r in why)
    p = _proof()
    p["phase_execution_records"]["phase4_actor_policy"]["execution_status"] = "blocked_no_mechanism"
    ok, why = qualify(p)
    assert not ok and any("relevant_phase_blocked" in r for r in why)


def test_runner_never_imports_resolution_store():
    """Forecast-time modules must hold no reference to the resolution store (vault_build and
    scorer are scorer-side by design and excluded)."""
    src_dir = Path(__file__).resolve().parents[1] / "framework"
    for f in ("runner.py", "evidence_build.py", "packs.py", "qualify.py", "baselines.py",
              "metrics.py"):
        text = (src_dir / f).read_text().replace(
            "never imports historical_backtests.framework.resolution_store", "")
        assert "resolution_store" not in text, f"{f} references the resolution store"


def test_resolution_store_import_guard():
    code = "import historical_backtests.framework.resolution_store"
    env = {**os.environ}
    env.pop("REPLAY_SCORER", None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, cwd=str(Path(__file__).resolve().parents[2]))
    assert r.returncode != 0 and "scorer-only" in (r.stderr + r.stdout)
    env["REPLAY_SCORER"] = "1"
    r2 = subprocess.run([sys.executable, "-c", code + "; print('ok')"], capture_output=True,
                        text=True, env=env, cwd=str(Path(__file__).resolve().parents[2]))
    assert r2.returncode == 0 and "ok" in r2.stdout


def test_sentinel_legacy_simulate_path_never_used(monkeypatch, tmp_path):
    """Monkeypatch the known simplified entrypoint (pipeline.simulate) to explode; stub the
    canonical facade; run_row must succeed through the facade only, byte-identical question."""
    import swm.world_model_v2.pipeline as P
    import swm.world_model_v2.unified_runtime as UR
    from historical_backtests.framework import runner, evidence_build, packs

    def boom(*a, **k):
        raise AssertionError("legacy pipeline.simulate must never be called by the benchmark")
    monkeypatch.setattr(P, "simulate", boom)
    seen = {}

    def fake_simulate_world(question, **kw):
        seen["question"] = question
        r = types.SimpleNamespace()
        r.simulation_status = "completed"
        r.raw_distribution = {"yes": 0.4, "no": 0.6}
        r.support_grade = "exploratory"
        r.plan_hash = lambda: "ph"
        r.limitations = []
        r.fallbacks_used = []
        from historical_backtests.framework.qualify import current_phase_contract
        r.provenance = {"runtime": "unified-1.0",
                        "event_time": {"p_event_by_deadline": 0.4, "n_particles": 200,
                                       "occurrence_resolves": "yes"},
                        "plan_lineage": {},
                        "operator_delta_census": {"production_actor_policy":
                                                  {"n_deltas": 12, "fields_written": ["a", "b"]}},
                        "phase_execution_records": {p: {"relevant": True,
                                                        "execution_status": "causally_active",
                                                        "n_state_deltas": 3}
                                                    for p in current_phase_contract()},
                        "fully_integrated": True, "phase_integration_failures": []}
        return r
    monkeypatch.setattr(UR, "simulate_world", fake_simulate_world)
    monkeypatch.setattr(evidence_build, "load_bundle",
                        lambda case, cutoff, out_dir: types.SimpleNamespace(
                            bundle_hash=lambda: "cap", claims=[{"c": 1}],
                            render=lambda max_chars=0: "evidence"))
    monkeypatch.setattr(packs, "load_pack", lambda ts: {"survival_pack": None,
                                                        "fallback_reason": "test"})
    q = "Will the Senate confirm Nominee X before the end of the session?"
    case = {"case_id": "hb_t1", "raw_question": q,
            "question_sha256": hashlib.sha256(q.encode()).hexdigest(),
            "split": "calibration", "causal_scale": "small_group_decision",
            "domain": "courts", "question_open_ts": 1730000000.0,
            "resolution_deadline": "2025-06-30", "market_snapshots": {},
            "forecast_cutoffs": ["2025-01-15T00:00:00Z"]}
    model = {"registry_model_id": "m", "temporal_safety_tier": "TIER_B",
             "openrouter_slug": "s", "openrouter_provider": "wandb",
             "openrouter_provider_display": "WandB", "quantization": "bf16",
             "request_configuration": {"temperature": 0.2, "max_tokens": 100}}
    from historical_backtests.models import registry
    monkeypatch.setattr(registry, "assert_temporal_ordering",
                        lambda *a, **k: {"ordering": "test"})
    monkeypatch.setattr(runner, "assert_temporal_ordering",
                        lambda *a, **k: {"ordering": "test"})
    row = runner.run_row(case, "2025-01-15T00:00:00Z", model, results_dir=tmp_path,
                         capsule_dir=tmp_path, run_baselines=False)
    assert seen["question"] == q                             # byte-for-byte into the facade
    assert row["qualified"], row.get("disqualify_reasons")
    assert row["p_yes"] == 0.4
    # tampered question → row fails closed
    case2 = dict(case, question_sha256="0" * 64)
    row2 = runner.run_row(case2, "2025-01-15T00:00:00Z", model, results_dir=tmp_path,
                          capsule_dir=tmp_path, run_baselines=False)
    assert not row2["qualified"]


def test_scorer_locked_one_open(tmp_path, monkeypatch):
    code = f"""
import json, os, sys, time, hashlib
sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
os.environ['REPLAY_SCORER'] = '1'
from historical_backtests.framework import scorer, resolution_store as RS
import historical_backtests.framework.scorer as S
tmp = {str(tmp_path)!r}
from pathlib import Path
S.ROOT = Path(tmp)
RS.VAULT_DIR = Path(tmp) / 'resolution_vault'
bdir = Path(tmp) / 'benchmark_versions' / 'b1'; bdir.mkdir(parents=True)
case = {{'case_id': 'c1', 'split': 'rotating_locked', 'cluster_id': 'cl1',
        'causal_scale': 's', 'domain': 'd', 'resolution_deadline_ts': 2e9}}
qv = bdir / 'question_vault.json'
qv.write_text(json.dumps({{'cases': [case]}}))
RS.seal_file(qv)
RS.write_resolutions('b1', {{'c1': {{'actual_outcome': 1, 'resolution_ts': 1.9e9}}}})
rdir = Path(tmp) / 'results' / 'b1' / 'r1'; rdir.mkdir(parents=True)
led = rdir / 'forecast_ledger.jsonl'
led.write_text(json.dumps({{'case_id': 'c1', 'cutoff': '2025-01-01T00:00:00Z',
    'qualified': True, 'p_yes': 0.7, 'event_time': {{}}, 'baselines': [],
    'causal_scale': 's', 'domain': 'd'}}) + '\\n')
led.with_suffix('.jsonl.seal').write_text(json.dumps(
    {{'sha256': hashlib.sha256(led.read_bytes()).hexdigest()}}))
out = S.score('b1', 'r1', splits=(), open_locked=True)
assert out['label'] == 'ROTATING_SEALED_HOLDOUT' and out['n_scored'] == 1, out
try:
    S.score('b1', 'r1', splits=(), open_locked=True)
    print('SECOND-OPEN-ALLOWED')
except PermissionError:
    print('ONE-OPEN-OK')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parents[2]))
    assert "ONE-OPEN-OK" in r.stdout, r.stdout + r.stderr


def test_crps_and_coverage_functions():
    from historical_backtests.framework.metrics import crps_event_time, interval_cover
    evt = {"cdf_grid_ts": [100.0, 200.0, 300.0], "cdf": [0.1, 0.5, 0.9],
           "occurrence_resolves": "yes",
           "first_passage_quantiles_ts": {"0.1": 120.0, "0.9": 280.0,
                                          "0.25": 150.0, "0.75": 250.0}}
    c_hit = crps_event_time(evt, outcome=1, resolution_ts=150.0, deadline_ts=300.0)
    c_censored = crps_event_time(evt, outcome=0, resolution_ts=9e9, deadline_ts=300.0)
    assert c_hit is not None and c_censored is not None and c_hit < c_censored
    assert interval_cover(evt, outcome=1, resolution_ts=200.0, q_lo="0.1", q_hi="0.9") == 1
    assert interval_cover(evt, outcome=1, resolution_ts=500.0, q_lo="0.1", q_hi="0.9") == 0
    assert interval_cover(evt, outcome=0, resolution_ts=9e9, q_lo="0.1", q_hi="0.9") is None
