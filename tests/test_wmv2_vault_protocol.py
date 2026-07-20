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


def _build_fixture_vault(tmp_path, monkeypatch, n=12):
    """Run the REAL builder main() against a stubbed gamma API into tmp paths."""
    import experiments.replay_v2.build_vault as V2B
    now = time.time()
    page = [_fixture_market(i, now + (30 + i * 7) * DAY) for i in range(n)]
    monkeypatch.setattr(V2B, "_get", lambda url: page if "offset=0" in url else [])
    monkeypatch.setattr(V2B, "_history", lambda tok: [], raising=False)
    monkeypatch.setattr(BV, "OUT", tmp_path / "event_time_frozen_vault.json")
    monkeypatch.setattr(BV, "SEAL", tmp_path / "event_time_vault_seal.json")
    monkeypatch.setattr(BV, "N_TARGET", n)
    BV.main()
    return BV.OUT, BV.SEAL


def test_build_seals_future_window_vault_and_never_rebuilds(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    vault = json.loads(out.read_text())
    seal = json.loads(seal_p.read_text())
    assert vault["n_questions"] == 12
    assert all(q["market_p_yes_at_freeze"] == pytest.approx(0.4) for q in vault["questions"])
    assert seal["sha256"] == hashlib.sha256(canonical_bytes(vault)).hexdigest()
    assert seal["opened"] is False
    opens = dt.datetime.fromisoformat(seal["opens_after"]).timestamp()
    assert opens > max(q["end_ts"] for q in vault["questions"])   # opens only after the window
    # a frozen vault is never rebuilt in place
    with pytest.raises(SystemExit, match="never rebuilt"):
        BV.main()


def test_scorer_time_gate_refuses_early(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    monkeypatch.setattr(SV, "OUT", out)
    monkeypatch.setattr(SV, "SEAL", seal_p)
    with pytest.raises(SystemExit, match="refusing to score early"):
        SV.main()


def test_scorer_tamper_gate_and_single_open_gate(tmp_path, monkeypatch):
    out, seal_p = _build_fixture_vault(tmp_path, monkeypatch)
    monkeypatch.setattr(SV, "OUT", out)
    monkeypatch.setattr(SV, "SEAL", seal_p)
    # move the clock past the window: rewrite opens_after into the past (the gate itself is what we
    # bypass here, to reach the TAMPER gate behind it)
    seal = json.loads(seal_p.read_text())
    seal["opens_after"] = _iso(time.time() - DAY)
    seal_p.write_text(json.dumps(seal))
    vault = json.loads(out.read_text())
    vault["questions"][0]["market_p_yes_at_freeze"] = 0.9      # tamper
    out.write_text(json.dumps(vault, indent=1))
    with pytest.raises(SystemExit, match="SEAL MISMATCH"):
        SV.main()
    # single-open: an already-opened seal refuses a second scoring run
    seal["opened"] = True
    seal_p.write_text(json.dumps(seal))
    with pytest.raises(SystemExit, match="already opened"):
        SV.main()


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


def test_stratified_fit_stays_offline_and_the_channel_stays_quarantined(tmp_path):
    """§NAP: the intention-HR fitting utility remains an OFFLINE artifact producer, but there is
    NO production stance→hazard channel to overlay it onto — and the artifact it produces does
    not pass the provenance-eligibility gate until it carries the full contract."""
    from swm.world_model_v2.event_time import fit_intention_hazard_ratios, hr_pack_info
    from swm.world_model_v2.numeric_provenance import fitted_artifact_eligible
    rows = ([{"commitment_level": "committed_to_prevent", "hazard_ratio": 0.5,
              "pathway": "cooperative_agreement"}] * 8
            + [{"commitment_level": "committed_to_prevent", "hazard_ratio": 0.9,
                "pathway": "institutional_procedure"}] * 8)
    pack = fit_intention_hazard_ratios(rows)
    pooled = pack["hazard_ratios"]["committed_to_prevent"][0]
    coop = pack["hazard_ratios_by_pathway"]["cooperative_agreement"]["committed_to_prevent"][0]
    inst = pack["hazard_ratios_by_pathway"]["institutional_procedure"]["committed_to_prevent"][0]
    assert coop < pooled < inst                               # strata differ, pooled sits between
    info = hr_pack_info()
    assert info["source"] == "quarantined_no_production_stance_hazard_channel"
    ok, why = fitted_artifact_eligible(pack)
    assert not ok and "missing required provenance keys" in why


def test_sensitivity_harness_importable_and_arms_defined():
    import experiments.replay_v3.sensitivity_harness as SH
    assert callable(SH.sweep) and callable(SH._run_pinned_world)
