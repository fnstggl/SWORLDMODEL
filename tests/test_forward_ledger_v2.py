"""Phase 16: forward ledger is append-only, versioned, never edits a locked forecast; re-forecasting
creates a NEW version; scoring only touches resolved rows."""
from swm.world_model_v2.forward_ledger_v2 import ForwardLedgerV2, ForwardLock


def _lock(qid="q1", commit="abc123", raw=0.7):
    return ForwardLock(qid=qid, question="Will X happen?", as_of="2024-01-01", horizon="2024-06-01",
                       evidence_bundle_hash="ev1", plan_hash="pl1",
                       mechanisms=[{"mech_id": "poisson_arrival", "version": "1.0.0", "status": "prior"}],
                       n_particles=30, code_commit=commit, model_versions={"llm": "deepseek-chat"},
                       raw_probability=raw, calibrated_probability=raw, confidence_grade="supported")


def test_lock_is_versioned_and_appended(tmp_path):
    led = ForwardLedgerV2(path=str(tmp_path / "l.jsonl"))
    v = led.lock(_lock())
    assert len(v) == 16
    rows = led.load()
    assert len(rows) == 1 and rows[0]["kind"] == "lock" and rows[0]["lock_version"] == v


def test_reforecast_creates_new_version_never_edits(tmp_path):
    led = ForwardLedgerV2(path=str(tmp_path / "l.jsonl"))
    v1 = led.lock(_lock(commit="aaa", raw=0.7))
    v2 = led.lock(_lock(commit="bbb", raw=0.4))            # new code commit → new version
    assert v1 != v2
    rows = led.load()
    assert len(rows) == 2                                   # both preserved, first NOT edited
    assert rows[0]["raw_probability"] == 0.7 and rows[1]["raw_probability"] == 0.4


def test_resolution_is_a_new_line_not_an_edit(tmp_path):
    led = ForwardLedgerV2(path=str(tmp_path / "l.jsonl"))
    v = led.lock(_lock())
    led.resolve("q1", v, 1.0)
    rows = led.load()
    assert len(rows) == 2
    assert rows[0]["kind"] == "lock" and rows[0]["resolution"] is None  # lock untouched (never edited)
    assert rows[1]["kind"] == "resolution" and rows[1]["outcome"] == 1.0
    assert led.open_locks() == []                          # now resolved


def test_open_locks_and_scoring(tmp_path):
    led = ForwardLedgerV2(path=str(tmp_path / "l.jsonl"))
    for i in range(6):
        v = led.lock(_lock(qid=f"q{i}", raw=0.8))
        led.resolve(f"q{i}", v, 1.0)
    led.lock(_lock(qid="open", raw=0.5))                   # unresolved
    s = led.score(min_n=5)
    assert s["n_resolved"] == 6 and s["n_open"] == 1
    assert s["brier_raw"] is not None
