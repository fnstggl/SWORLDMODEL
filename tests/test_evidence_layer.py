"""Phase 2 evidence layer: as-of gate (zero slack), quarantine, bundle hashes, rendering,
leakage auditor (resolution terms, future dates, retrospective language, duplicates, timestamp basis)."""
import pytest

from swm.world_model_v2.evidence import EvidenceBundle, EvidenceGateError, EvidenceItem
from swm.world_model_v2.leakage_audit import audit_bundle

AS_OF = 1.6e9   # 2020-09-13


def _item(iid, text, *, pub=AS_OF - 86400, verified=False, title="", source="outlet"):
    return EvidenceItem(item_id=iid, text=text, title=title, source=source,
                        retrieved_at=AS_OF + 5 * 86400, published_at=pub, published_verified=verified)


def test_gate_refuses_post_asof_items_hard():
    b = EvidenceBundle(question_id="q1", as_of=AS_OF)
    with pytest.raises(EvidenceGateError):
        b.add(_item("late", "the result was announced", pub=AS_OF + 3600))
    assert len(b.items) == 0 and len(b.quarantine) == 1
    assert b.quarantine[0].quarantine_reason.startswith("published_at")


def test_gate_quarantines_undated_items_unless_explicit():
    b = EvidenceBundle(question_id="q1", as_of=AS_OF)
    assert b.add(_item("und", "context", pub=None)) is False
    assert len(b.quarantine) == 1
    assert b.add(_item("und2", "user context", pub=None), allow_undated=True) is True
    assert "undated_admitted_explicitly" in b.items[0].leakage_flags


def test_zero_slack_default_and_nonzero_slack_is_flagged():
    b = EvidenceBundle(question_id="q1", as_of=AS_OF)
    assert b.slack_s == 0.0
    b2 = EvidenceBundle(question_id="q1", as_of=AS_OF, slack_s=3600.0)
    b2.add(_item("x", "fine", pub=AS_OF - 10))
    assert any(f.startswith("nonzero_asof_slack") for f in b2.items[0].leakage_flags)
    rep = audit_bundle(b2)
    assert rep.summary["nonzero_slack"] is True
    assert rep.summary["evidence_quality_grade"].startswith("C")


def test_bundle_hash_is_content_sensitive_and_order_insensitive():
    b1 = EvidenceBundle(question_id="q", as_of=AS_OF)
    b1.add(_item("a", "alpha"))
    b1.add(_item("b", "beta"))
    b2 = EvidenceBundle(question_id="q", as_of=AS_OF)
    b2.add(_item("b", "beta"))
    b2.add(_item("a", "alpha"))
    assert b1.bundle_hash() == b2.bundle_hash()
    b2.add(_item("c", "gamma"))
    assert b1.bundle_hash() != b2.bundle_hash()


def test_render_excludes_quarantine_and_orders_verified_first():
    b = EvidenceBundle(question_id="q", as_of=AS_OF)
    b.add(_item("claimed", "rss-dated story", pub=AS_OF - 86400))
    b.add(_item("wiki", "revision text", pub=AS_OF - 5 * 86400, verified=True))
    b.add(_item("und", "mystery", pub=None))                 # quarantined
    txt = b.render()
    assert "mystery" not in txt
    assert txt.index("revision text") < txt.index("rss-dated story")
    assert "verified" in txt.splitlines()[0]


def test_auditor_catches_planted_resolution_term_and_future_date():
    b = EvidenceBundle(question_id="q", as_of=AS_OF)
    b.add(_item("leak1", "Officials declared the winner after a landslide."))
    b.add(_item("leak2", "The committee will meet on 2021-03-05 to review."))   # date after as_of
    b.add(_item("ok", "Polls open tomorrow across the state."))
    rep = audit_bundle(b, resolution_terms=["declared the winner"])
    assert "leak1" in rep.hard_leaks and "leak2" in rep.hard_leaks and "ok" not in rep.hard_leaks
    assert rep.summary["evidence_quality_grade"] == "F (leaked)"
    assert not rep.clean()


def test_auditor_flags_retrospective_language_and_duplicates():
    b = EvidenceBundle(question_id="q", as_of=AS_OF)
    b.add(_item("retro", "At the time, few believed it; the plan would go on to succeed."))
    dup = "Senator Smith announced a new infrastructure package on the steps of the capitol today."
    b.add(_item("d1", dup, source="ap"))
    b.add(_item("d2", dup + " (syndicated)", source="local-paper"))
    rep = audit_bundle(b)
    assert any(f.startswith("retrospective_language") for f in rep.item_flags["retro"])
    assert rep.duplicates and rep.duplicates[0][0] in ("d1", "d2")
    assert rep.clean()                                        # soft flags are not hard leaks


def test_persist_is_append_only(tmp_path):
    b = EvidenceBundle(question_id="q", as_of=AS_OF)
    b.add(_item("a", "alpha"))
    p1 = b.persist(root=str(tmp_path))
    p2 = b.persist(root=str(tmp_path))
    assert p1 != p2                                           # versioned, never overwritten
