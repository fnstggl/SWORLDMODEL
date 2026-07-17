"""Frozen event-time vault — the FULL protocol exercised offline with a stubbed market API:
build → seal → time gate → tamper gate → single-open gate. The only thing these tests cannot do is
the real freeze (network); everything the protocol must refuse or verify is proven here. Plus the
placebo-controlled fitter and stratified-pack surfaces."""
import datetime as dt
import hashlib
import json
import time

import pytest

import experiments.replay_v3.build_event_time_vault as BV
import experiments.replay_v3.score_event_time_vault as SV
from experiments.replay_v3.build_event_time_vault import canonical_bytes

T0 = 1_700_000_000.0
DAY = 86400.0


def _iso(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()


def _fixture_market(i, end_ts, p_yes=0.4):
    return {"question": f"Will event {i} happen by the deadline?", "conditionId": f"cid_{i}",
            "outcomes": json.dumps(["Yes", "No"]), "clobTokenIds": json.dumps([f"tok_{i}", f"ntok_{i}"]),
            "endDate": _iso(end_ts), "volumeNum": 50000.0,
            "outcomePrices": json.dumps([str(p_yes), str(1 - p_yes)])}


def _build_fixture_vault(tmp_path, monkeypatch, n=15, near_price=0.4):
    """Run the REAL builder main() against a stubbed gamma API into tmp paths. n=15 → 5 near
    (resolving within ~30h) + 10 far (30..100d out)."""
    import experiments.replay_v2.build_vault as V2B
    now = time.time()
    page = ([_fixture_market(i, now + (6 + i * 5) * 3600.0, p_yes=near_price) for i in range(5)]
            + [_fixture_market(100 + i, now + (30 + i * 7) * DAY) for i in range(10)])
    monkeypatch.setattr(V2B, "_get", lambda url: page if "offset=0" in url else [])
    monkeypatch.setattr(V2B, "_history", lambda tok: [], raising=False)
    monkeypatch.setattr(BV, "OUT", tmp_path / "event_time_frozen_vault.json")
    monkeypatch.setattr(BV, "SEAL", tmp_path / "event_time_vault_seal.json")
    monkeypatch.setattr(BV, "N_TARGET", n)
    BV.main()
    return BV.OUT, BV.SEAL


def test_build_seals_two_tranche_vault_and_never_rebuilds(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    vault = json.loads(out.read_text())
    seal = json.loads(seal_p.read_text())
    assert vault["n_questions"] == 15 and vault["n_near"] == 5 and vault["n_far"] == 10
    near = [q for q in vault["questions"] if q["tranche"] == "near"]
    far = [q for q in vault["questions"] if q["tranche"] == "far"]
    now = time.time()
    assert all(q["end_ts"] <= now + 2.1 * DAY for q in near)   # resolves today/tomorrow
    assert all(q["end_ts"] >= now + 21 * DAY for q in far)
    assert seal["sha256"] == hashlib.sha256(canonical_bytes(vault)).hexdigest()
    tr = seal["tranches"]
    assert tr["near"]["opened"] is False and tr["far"]["opened"] is False
    near_opens = dt.datetime.fromisoformat(tr["near"]["opens_after"]).timestamp()
    far_opens = dt.datetime.fromisoformat(tr["far"]["opens_after"]).timestamp()
    assert near_opens < now + 3.5 * DAY < far_opens            # near scoreable within days
    assert near_opens > max(q["end_ts"] for q in near)         # ... but only after ITS window
    # a frozen vault is never rebuilt in place
    with pytest.raises(SystemExit, match="never rebuilt"):
        BV.main()


def test_near_tranche_excludes_already_effectively_decided_markets(tmp_path, monkeypatch):
    """'resolve today/tomorrow if they HAVEN'T happened yet': a near-dated market whose freeze
    price is pinned at 0.99 has effectively happened — it must not enter the near tranche."""
    with pytest.raises(SystemExit, match="too small"):
        _build_fixture_vault(tmp_path, monkeypatch, near_price=0.99)


def test_scorer_time_gate_refuses_early_per_tranche(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    monkeypatch.setattr(SV, "OUT", out)
    monkeypatch.setattr(SV, "SEAL", seal_p)
    for tranche in ("near", "far", "all"):
        with pytest.raises(SystemExit, match="refusing to score early"):
            SV.main(tranche=tranche)


def test_scorer_tamper_gate_and_per_tranche_single_open(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    monkeypatch.setattr(SV, "OUT", out)
    monkeypatch.setattr(SV, "SEAL", seal_p)
    # move the clock past the NEAR window: rewrite its opens_after into the past (the gate itself
    # is what we bypass here, to reach the TAMPER gate behind it)
    seal = json.loads(seal_p.read_text())
    seal["tranches"]["near"]["opens_after"] = _iso(time.time() - DAY)
    seal_p.write_text(json.dumps(seal))
    vault = json.loads(out.read_text())
    vault["questions"][0]["market_p_yes_at_freeze"] = 0.9      # tamper
    out.write_text(json.dumps(vault, indent=1))
    with pytest.raises(SystemExit, match="SEAL MISMATCH"):
        SV.main(tranche="near")
    # single-open PER TRANCHE: a consumed near tranche refuses; far stays gated by ITS window
    seal["tranches"]["near"]["opened"] = True
    seal_p.write_text(json.dumps(seal))
    with pytest.raises(SystemExit, match="already opened"):
        SV.main(tranche="near")
    with pytest.raises(SystemExit, match="refusing to score early"):
        SV.main(tranche="far")


# ---------------------------------------------------------------- placebo-controlled measurement
def _hist(points):
    return [{"t": T0 + d * DAY, "p": p} for d, p in points]


def test_placebo_control_removes_secular_drift():
    """A market whose implied hazard drifts up mechanically (deadline approach) must NOT credit a
    mid-life statement with the drift: the placebo-normalized ratio lands near 1, the raw one
    doesn't."""
    from experiments.replay_v3.fit_intention_hr import statement_hazard_ratio_placebo
    deadline = T0 + 100 * DAY
    flat = [(d, 0.5) for d in range(2, 99, 2)]                # constant price ⇒ λ rises toward T
    meas = statement_hazard_ratio_placebo(_hist(flat), T0 + 50 * DAY, deadline)
    assert meas is not None and meas["placebo_controlled"]
    assert meas["raw"] > 1.05                                 # the naive measure sees the drift
    assert 0.9 <= meas["hazard_ratio"] <= 1.1                 # the controlled measure does not
    # a REAL post-statement move survives the control
    real = ([(d, 0.5) for d in range(2, 50, 2)]
            + [(50 + i, max(0.1, 0.5 - 0.05 * i)) for i in range(1, 9)]
            + [(d, 0.12) for d in range(60, 99, 4)])
    meas2 = statement_hazard_ratio_placebo(_hist(real), T0 + 50 * DAY, deadline)
    assert meas2 is not None and meas2["hazard_ratio"] < 0.75


def test_stratified_fit_and_pathway_table_overlay(tmp_path, monkeypatch):
    from swm.world_model_v2.event_time import (fit_intention_hazard_ratios, _hr_table,
                                               hr_pack_info)
    import swm.world_model_v2.event_time as ET
    rows = ([{"commitment_level": "committed_to_prevent", "hazard_ratio": 0.5,
              "pathway": "cooperative_agreement"}] * 8
            + [{"commitment_level": "committed_to_prevent", "hazard_ratio": 0.9,
                "pathway": "institutional_procedure"}] * 8)
    pack = fit_intention_hazard_ratios(rows)
    pooled = pack["hazard_ratios"]["committed_to_prevent"][0]
    coop = pack["hazard_ratios_by_pathway"]["cooperative_agreement"]["committed_to_prevent"][0]
    inst = pack["hazard_ratios_by_pathway"]["institutional_procedure"]["committed_to_prevent"][0]
    assert coop < pooled < inst                               # strata differ, pooled sits between
    pack["fitted_at"] = "2026-07-17T00:00:00+00:00"
    p = tmp_path / "intention_hr_pack.json"
    p.write_text(json.dumps(pack))
    monkeypatch.setattr(ET, "INTENTION_HR_PACK", p)
    t_coop = _hr_table("cooperative_agreement")["committed_to_prevent"][0]
    t_inst = _hr_table("institutional_procedure")["committed_to_prevent"][0]
    assert t_coop == pytest.approx(coop) and t_inst == pytest.approx(inst)
    info = hr_pack_info()
    assert info["source"] == "fitted_pack" and info["stratified"] is True
    assert info["fitted_at"].startswith("2026-07-17")


def test_sensitivity_harness_importable_and_arms_defined():
    import experiments.replay_v3.sensitivity_harness as SH
    assert callable(SH.sweep) and callable(SH._run_pinned_world)


# ---------------------------------------------------------------- pre-registration protocol
import experiments.replay_v3.predict_event_time_vault as PV


def _patch_prereg_paths(tmp_path, monkeypatch, out, seal_p):
    monkeypatch.setattr(PV, "OUT", out)
    monkeypatch.setattr(PV, "SEAL", seal_p)
    monkeypatch.setattr(PV, "VAULT", tmp_path)
    monkeypatch.setattr(PV, "PRED_DIR", tmp_path / "predictions")
    monkeypatch.setattr(PV, "LOG_DIR", tmp_path / "prereg_logs")


def test_prereg_openness_gate_never_forecasts_after_the_fact():
    now = time.time()
    w = {"end_ts": now + 6 * 3600.0}
    assert PV.openness_gate(w, {"closed": False, "p_yes": 0.4}, now) == ""
    # scheduled end passed → refuse
    assert "end passed" in PV.openness_gate({"end_ts": now - 60.0}, {"closed": False, "p_yes": 0.4}, now)
    # market reports closed → refuse
    assert "closed" in PV.openness_gate(w, {"closed": True, "p_yes": 0.4}, now)
    # world already knows (pinned price near resolution) → refuse, both directions
    assert "decided" in PV.openness_gate(w, {"closed": False, "p_yes": 0.985}, now)
    assert "decided" in PV.openness_gate(w, {"closed": False, "p_yes": 0.01}, now)
    # unknown price is not proof of decision → allowed (recorded on the row either way)
    assert PV.openness_gate(w, {"closed": False, "p_yes": None}, now) == ""
    # TIME-AWARE: the same extreme price on a FAR-dated market is a longshot forecast, not a
    # decided outcome — refusing it would bias the tranche toward mid-price rows
    far = {"end_ts": now + 30 * 86400.0}
    assert PV.openness_gate(far, {"closed": False, "p_yes": 0.01}, now) == ""
    assert PV.openness_gate(far, {"closed": False, "p_yes": 0.985}, now) == ""
    assert "closed" in PV.openness_gate(far, {"closed": True, "p_yes": 0.5}, now)


def test_prereg_refuses_tampered_vault(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    _patch_prereg_paths(tmp_path, monkeypatch, out, seal_p)
    vault = json.loads(out.read_text())
    vault["questions"][0]["market_p_yes_at_freeze"] = 0.9
    out.write_text(json.dumps(vault, indent=1))
    with pytest.raises(SystemExit, match="SEAL MISMATCH"):
        PV.verify_vault_seal()


def test_p_at_market_end_interpolation_and_polarity():
    evt = {"cdf_grid_ts": [100.0, 200.0, 300.0], "cdf": [0.1, 0.3, 0.5]}
    assert PV.p_at_ts(evt, 250.0) == pytest.approx(0.4)
    assert PV.p_at_ts(evt, 300.0) == pytest.approx(0.5)
    assert PV.p_at_ts(evt, 1000.0) == pytest.approx(0.5)       # clamped at the last grid point
    evt_surv = dict(evt, occurrence_resolves="no")
    assert PV.p_at_ts(evt_surv, 250.0) == pytest.approx(0.6)   # survival polarity inverts F
    assert PV.p_at_ts({"cdf_grid_ts": [], "cdf": []}, 250.0) is None


def _fake_prediction_row(w, *, status="predicted", completed_at=None):
    return {"condition_id": w["condition_id"], "question": w["question"], "tranche": w["tranche"],
            "end_date": w["end_date"], "end_ts": w["end_ts"], "event_cluster": w.get("event_cluster"),
            "market_p_yes_at_freeze": w["market_p_yes_at_freeze"], "as_of": "2026-01-01",
            "model": "test", "seed": 1, "code_git_sha": "deadbeef",
            "started_at": _iso(time.time() - 120.0),
            "live_market_at_prediction": {"fetched_at": _iso(time.time() - 120.0),
                                          "closed": False, "p_yes": 0.4},
            "status": status,
            "completed_at": completed_at or _iso(time.time() - 60.0),
            "p_yes": 0.35, "p_at_market_end": 0.3,
            "event_time": {"cdf_grid_ts": [w["end_ts"] - 3600.0, w["end_ts"] + 3600.0],
                           "cdf": [0.2, 0.4], "first_passage_quantiles_ts": {"0.1": None},
                           "occurrence_resolves": "yes"}}


def test_prereg_finalize_seals_append_only(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    _patch_prereg_paths(tmp_path, monkeypatch, out, seal_p)
    vault = json.loads(out.read_text())
    near = [w for w in vault["questions"] if w["tranche"] == "near"]
    # finalize with a missing checkpoint refuses
    for w in near[:-1]:
        PV._write_atomic(PV._checkpoint("near", w["condition_id"]), _fake_prediction_row(w))
    with pytest.raises(SystemExit, match="no checkpoint"):
        PV.finalize("near", vault)
    PV._write_atomic(PV._checkpoint("near", near[-1]["condition_id"]),
                     _fake_prediction_row(near[-1]))
    PV.finalize("near", vault)
    doc = json.loads(PV.predictions_path("near").read_text())
    pseal = json.loads(PV.predictions_seal_path("near").read_text())
    assert doc["n_rows"] == len(near) and doc["n_predicted"] == len(near)
    assert doc["vault_sha256"] == json.loads(seal_p.read_text())["sha256"]
    assert pseal["sha256"] == hashlib.sha256(canonical_bytes(doc)).hexdigest()
    # sealed predictions are never overwritten
    with pytest.raises(SystemExit, match="never overwritten"):
        PV.finalize("near", vault)


def _open_near_gate(seal_p):
    seal = json.loads(seal_p.read_text())
    seal["tranches"]["near"]["opens_after"] = _iso(time.time() - DAY)
    seal_p.write_text(json.dumps(seal))


def test_scorer_refuses_without_preregistered_predictions(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    _patch_prereg_paths(tmp_path, monkeypatch, out, seal_p)
    monkeypatch.setattr(SV, "OUT", out)
    monkeypatch.setattr(SV, "SEAL", seal_p)
    _open_near_gate(seal_p)
    with pytest.raises(SystemExit, match="PREREGISTERED PREDICTIONS REQUIRED"):
        SV.main(tranche="near")
    # the refusal must NOT have consumed the tranche
    assert json.loads(seal_p.read_text())["tranches"]["near"]["opened"] is False


def test_scorer_refuses_touched_or_foreign_predictions(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    _patch_prereg_paths(tmp_path, monkeypatch, out, seal_p)
    monkeypatch.setattr(SV, "OUT", out)
    monkeypatch.setattr(SV, "SEAL", seal_p)
    vault = json.loads(out.read_text())
    near = [w for w in vault["questions"] if w["tranche"] == "near"]
    for w in near:
        PV._write_atomic(PV._checkpoint("near", w["condition_id"]), _fake_prediction_row(w))
    PV.finalize("near", vault)
    _open_near_gate(seal_p)
    # touched predictions file → seal mismatch
    doc = json.loads(PV.predictions_path("near").read_text())
    doc["rows"][0]["p_at_market_end"] = 0.99
    PV.predictions_path("near").write_text(json.dumps(doc, indent=1, default=str))
    with pytest.raises(SystemExit, match="PREDICTIONS SEAL MISMATCH"):
        SV.main(tranche="near")
    # re-sealed but registered against a DIFFERENT vault → refuse
    doc["rows"][0]["p_at_market_end"] = 0.3
    doc["vault_sha256"] = "0" * 64
    PV.predictions_path("near").write_text(json.dumps(doc, indent=1, default=str))
    PV.predictions_seal_path("near").write_text(json.dumps(
        {"file": PV.predictions_path("near").name,
         "sha256": hashlib.sha256(canonical_bytes(doc)).hexdigest(), "sealed_at": _iso(time.time())}))
    with pytest.raises(SystemExit, match="PREDICTIONS/VAULT MISMATCH"):
        SV.main(tranche="near")


def test_scorer_scores_preregistered_rows_with_no_llm(tmp_path, monkeypatch):
    """End-to-end offline: sealed pre-registered rows are scored with NO model anywhere — a late
    forecast (completed after the scheduled end) is excluded, valid ones produce CRPS/Brier."""
    import experiments.replay_v2.build_vault as V2B
    import experiments.replay_v3.fit_survival_pack as FS
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    _patch_prereg_paths(tmp_path, monkeypatch, out, seal_p)
    monkeypatch.setattr(SV, "OUT", out)
    monkeypatch.setattr(SV, "SEAL", seal_p)
    monkeypatch.setattr(SV, "RESULTS", tmp_path / "scores.json")
    vault = json.loads(out.read_text())
    near = [w for w in vault["questions"] if w["tranche"] == "near"]
    for i, w in enumerate(near):
        row = _fake_prediction_row(w) if i else _fake_prediction_row(
            w, completed_at=_iso(w["end_ts"] + 3600.0))        # row 0: registered too late
        PV._write_atomic(PV._checkpoint("near", w["condition_id"]), row)
    PV.finalize("near", vault)
    _open_near_gate(seal_p)
    monkeypatch.setattr(FS, "_market_by_condition", lambda cid: {})
    monkeypatch.setattr(V2B, "_history", lambda tok: [], raising=False)
    SV.main(tranche="near")
    res = json.loads((tmp_path / "event_time_vault_scores_near.json").read_text())
    assert res["n_questions"] == len(near)
    assert res["n_excluded"] == 1                              # the late row is out, with the reason
    assert res["n_scored"] == len(near) - 1
    late = [r for r in res["rows"] if r.get("excluded")]
    assert "did not complete before" in late[0]["excluded"]
    assert all("crps_v2" in r and "crps_market" in r and "brier_v2" in r
               for r in res["rows"] if "excluded" not in r)
    # the tranche is consumed exactly once
    assert json.loads(seal_p.read_text())["tranches"]["near"]["opened"] is True
    with pytest.raises(SystemExit, match="already opened"):
        SV.main(tranche="near")
