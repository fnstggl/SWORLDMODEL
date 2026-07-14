"""Integration completion — rule-normalization / requirement-inference / fingerprint tests (Part Q)."""
from __future__ import annotations
import inspect
from types import SimpleNamespace

from swm.world_model_v2.integration_completion import (normalize_institution_rules, executable_rule_count,
                                                       infer_required_phases, completeness_diagnostics)
from swm.world_model_v2.institutions import EXECUTABLE_RULE_KINDS
from swm.world_model_v2 import runtime_fingerprint as rf
from swm.world_model_v2 import unified_runtime as U


def _plan(institutions=None, populations=None, relations=None, actor_decisions=None, mechs=None):
    return SimpleNamespace(institutions=institutions or [], populations=populations or [],
                           relations=relations or [], actor_decisions=actor_decisions or [],
                           accepted_mechanisms=mechs or [])


def test_normalization_maps_noncanonical_kinds_to_executable():
    plan = _plan(institutions=[{"id": "senate", "rules": [
        {"kind": "voting_rule", "params": {"quorum": 51}},
        {"kind": "confirmation_process", "params": {}},
        {"kind": "committee_vote", "params": {}}]}])
    assert executable_rule_count(plan) == 0                   # none executable before
    rep = normalize_institution_rules(plan)
    assert executable_rule_count(plan) == 3                   # all executable after
    for ru in plan.institutions[0]["rules"]:
        assert ru["kind"] in EXECUTABLE_RULE_KINDS
        assert ru["_original_kind"] in ("voting_rule", "confirmation_process", "committee_vote")
    assert rep["rules_total"] == 3 and rep["already_executable"] == 0


def test_normalization_never_drops_a_rule_and_is_idempotent():
    plan = _plan(institutions=[{"id": "x", "rules": [{"kind": "weird_unknown_kind", "params": {}}]}])
    normalize_institution_rules(plan)
    assert executable_rule_count(plan) == 1                   # unmapped → generic executable procedure
    before = [dict(r) for r in plan.institutions[0]["rules"]]
    normalize_institution_rules(plan)                         # idempotent
    assert plan.institutions[0]["rules"] == before


def test_normalization_preserves_already_executable():
    plan = _plan(institutions=[{"id": "x", "rules": [{"kind": "quorum", "params": {"n": 51}}]}])
    rep = normalize_institution_rules(plan)
    assert rep["already_executable"] == 1 and executable_rule_count(plan) == 1


def test_completeness_flags_ornamental_institution():
    plan = _plan(institutions=[{"id": "x", "rules": [{"kind": "voting_rule", "params": {}}]}])
    diags = completeness_diagnostics(plan)
    kinds = {d["issue"] for d in diags}
    assert "institution_declared_but_no_executable_rule" in kinds
    assert "institution_declared_but_no_operator" in kinds


def test_infer_required_phases():
    plan = _plan(institutions=[{"id": "s", "rules": []}], populations=[{"id": "p"}],
                 relations=[{"src": "a", "rel": "trusts", "dst": "b"}],
                 mechs=[{"operator": "nonlinear_contagion"}, {"operator": "production_actor_policy"}])
    req = infer_required_phases(plan)
    assert req["phase10_institutions"] and req["phase9_populations"] and req["phase9_networks"]
    assert req["phase7_nonlinear"] and req["phase4_actor_policy"]


def test_runtime_fingerprint_deterministic_and_corpus_status():
    f1 = rf.runtime_fingerprint(); f2 = rf.runtime_fingerprint()
    assert f1["fingerprint_hash"] == f2["fingerprint_hash"]
    assert rf.corpus_status(f1["fingerprint_hash"]) == "product_eligible"
    assert rf.corpus_status("some_old_hash") == "diagnostic_only"


def test_no_benchmark_question_id_hardcoding():
    """The runtime/completion code must not branch on specific corpus question IDs (Part K/gate 12)."""
    from experiments.integration_corpus import QUESTIONS
    qids = [q[0] for q in QUESTIONS]
    src = inspect.getsource(U) + inspect.getsource(
        __import__("swm.world_model_v2.integration_completion", fromlist=["x"]))
    leaked = [qid for qid in qids if f'"{qid}"' in src or f"'{qid}'" in src]
    assert not leaked, f"runtime hardcodes benchmark question IDs: {leaked}"


def test_unified_runtime_records_institution_normalization():
    """The unified runtime must invoke normalization (default-on) — source references it."""
    assert "normalize_institution_rules" in inspect.getsource(U)
