"""Pins the ensemble-compile-collapse fix (frozen-5: 4/5 questions rejected EVERY well-formed candidate).

Root cause: the compile-time executability check instantiated operators with llm=None, and the default-on
qualitative actor policy refuses to construct without a backend (§19) — which aborted operator
instantiation BEFORE the terminal outcome writer, so every candidate looked non-executable even though it
runs fine at rollout where the backend is present. Fix: the check DEFERS runtime-backend operators (they
are runtime resources, not plan defects, and are not outcome writers) and REPAIRS a missing outcome writer
instead of rejecting. These are fast and deterministic; a backend-gated live check is separate."""
import os

import pytest

# ensure the outcome operators are registered before get_operator is used
import swm.world_model_v2.fallback  # noqa: F401
import swm.world_model_v2.event_time  # noqa: F401
import swm.world_model_v2.phase_consumers  # noqa: F401
from types import SimpleNamespace

from swm.world_model_v2.materialize import operators_from_plan
from swm.world_model_v2.result import CompilerExecutionError
from swm.world_model_v2.ensemble_compiler import _executability_check


def _plan_with_actor_policy():
    """A minimal plan whose FIRST mechanism is the qualitative actor policy (needs a runtime backend) and
    whose terminal writer is generic_outcome_prior — the exact shape the compiler produced for BoJ/Knesset.
    Materializes: one 'outcome' quantity so the readout binds; a resolve_outcome event."""
    return SimpleNamespace(
        as_of=1_700_000_000.0, horizon_ts=1_704_000_000.0,
        entities=[], populations=[], relations=[], institutions=[],
        quantities=[{"name": "outcome", "qtype": "outcome", "units": "unit"}],
        accepted_mechanisms=[{"operator": "production_actor_policy", "mech_id": "actors"},
                             {"operator": "generic_outcome_prior", "mech_id": "resolve"}],
        scheduled_events=[{"etype": "resolve_outcome", "ts": 1_703_999_999.0, "participants": [],
                           "payload": {"outcome_var": "outcome", "family": "binary",
                                       "options": ["Yes", "No"], "lean": "neutral"}}],
        first_passage_processes=[], posterior_rate_particles=None,
        provenance={"outcome_lean": "neutral"}, _outside_world=None, _world_boundary=None,
        _unresolved_mechanisms=[],
        outcome_contract=SimpleNamespace(readout_var="outcome", family="binary", options=["Yes", "No"]))


def _strict(monkeypatch):
    """Force the PRODUCTION strict-actor-policy mode: the qualitative actor policy refuses to run without a
    backend (§19). The test suite's conftest sets SWM_ALLOW_NUMERIC_BASELINE=1 (offline numeric allowance),
    which masks the production failure — these tests must reproduce it."""
    monkeypatch.delenv("SWM_ALLOW_NUMERIC_BASELINE", raising=False)
    monkeypatch.delenv("SWM_ACTOR_POLICY", raising=False)
    monkeypatch.delenv("SWM_LLM_ACTORS", raising=False)


def test_rollout_path_still_refuses_actor_policy_without_backend(monkeypatch):
    # the REAL rollout path (defer_backend_operators=False) keeps the loud §19 refusal — no silent numeric
    _strict(monkeypatch)
    with pytest.raises(CompilerExecutionError):
        operators_from_plan(_plan_with_actor_policy(), llm=None)


def test_check_path_defers_actor_policy_and_keeps_outcome_writer(monkeypatch):
    # the executability check defers the backend-only actor policy and STILL instantiates the terminal writer
    _strict(monkeypatch)
    ops, rej = operators_from_plan(_plan_with_actor_policy(), llm=None,
                                   defer_backend_operators=True, allow_experimental=True)
    names = {getattr(o, "name", "") for o in ops}
    assert "generic_outcome_prior" in names                    # the outcome writer survived the deferral
    assert any("deferred_to_runtime_backend" in (r.get("reason") or "") for r in rej)
    assert all(getattr(o, "name", "") != "production_actor_policy" for o in ops)   # actor policy deferred


def test_executability_check_passes_for_actor_policy_plan(monkeypatch):
    # END TO END for the fix: in STRICT mode (production), a well-formed plan with a qualitative-actor-policy
    # mechanism is EXECUTABLE (was rejected as nonexecutable_after_bounded_repair before the fix)
    _strict(monkeypatch)
    ok, why = _executability_check(_plan_with_actor_policy(), llm=None)
    assert ok is True, why


def test_executability_check_repairs_missing_outcome_writer(monkeypatch):
    # a plan that lost its terminal writer is REPAIRED (canonical resolve_outcome synthesized), not rejected
    _strict(monkeypatch)
    plan = _plan_with_actor_policy()
    plan.accepted_mechanisms = [{"operator": "production_actor_policy", "mech_id": "actors"}]  # no writer
    plan.scheduled_events = [{"etype": "actor_action", "ts": 1_701_000_000.0, "participants": [], "payload": {}}]
    ok, why = _executability_check(plan, llm=None)
    assert ok is True, why
    assert any(e["etype"] == "resolve_outcome" for e in plan.scheduled_events)   # resolver synthesized


# ----- backend-gated live check: the five frozen questions must each produce >=1 executable model -----
FROZEN5 = [("BoJ", "7279494c-a775-5a57-a5f2-ac22252fb286"),
           ("visionOS", "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6"),
           ("Wale", "741b4bed-7502-5cd2-9cbe-949fbc70f857"),
           ("Hormuz", "017e64ef-7354-56c4-8a4d-e27121bc639a"),
           ("Banxico", "cfb43147-d9d2-5bd9-903f-f449e9a5aecf")]


@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"),
                    reason="live compile requires DEEPSEEK_API_KEY (integration check)")
@pytest.mark.parametrize("label,qid", FROZEN5)
def test_frozen5_produces_at_least_one_executable_model(label, qid):
    from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
    from swm.world_model_v2 import ensemble_compiler as EC
    from swm.world_model_v2.llm_call_cache import CallLedger
    from swm.api.deepseek_backend import default_chat_fn
    rows = {r["question_id"]: r for r in fetch_btf3()}
    q = _forecast_input(rows[qid])
    llm = default_chat_fn(system="Reply ONLY JSON.", max_tokens=8000, temperature=0.2)
    ledger = CallLedger()
    ens = EC.compile_world_ensemble(q["question"], llm=llm, as_of=str(q["present_date"])[:10],
                                    horizon=str(q["expected_resolution_date"])[:10], seed=0, ledger=ledger)
    executable = [c for c in ens.surviving() if c.executable_plan is not None]
    assert executable, f"{label}: no executable structural candidate — ensemble would collapse"
