"""Pins the evidence-sufficiency gate: a run whose posterior saw 0 as-of observations is flagged
`starved` so it can never be silently mistaken for an evidence-driven forecast (the EXP-104 failure)."""
from types import SimpleNamespace

from swm.world_model_v2.unified_runtime import evidence_sufficiency_signal


def _bundle(n_docs, n_claims):
    return SimpleNamespace(documents=list(range(n_docs)), included_claim_ids=list(range(n_claims)))


def _posterior(n_eff):
    return SimpleNamespace(n_effective_observations=n_eff)


def test_starved_when_asof_supplied_but_no_effective_observations():
    s = evidence_sufficiency_signal(_bundle(10, 8), _posterior(0), as_of="2026-05-07")
    assert s["starved"] is True and s["n_documents"] == 10 and s["n_effective_observations"] == 0


def test_not_starved_when_evidence_reaches_posterior():
    s = evidence_sufficiency_signal(_bundle(34, 33), _posterior(16), as_of="2026-05-07")
    assert s["starved"] is False and s["n_effective_observations"] == 16


def test_empty_bundle_is_starved():
    s = evidence_sufficiency_signal(None, None, as_of="2026-05-07")
    assert s["starved"] is True and s["n_documents"] == 0


def test_no_asof_is_not_flagged_starved():
    # no as_of => evidence was never expected => not a starvation failure (e.g. pure counterfactual)
    s = evidence_sufficiency_signal(None, None, as_of="")
    assert s["starved"] is False


def test_dropped_evidence_phase_is_not_flagged_starved():
    # explicit ablation (phase2 dropped) must not be reported as an accidental starvation
    s = evidence_sufficiency_signal(_bundle(0, 0), _posterior(0), as_of="2026-05-07", evidence_dropped=True)
    assert s["starved"] is False
