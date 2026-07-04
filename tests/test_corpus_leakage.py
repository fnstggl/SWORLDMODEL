"""EXP-008: the leakage-safe corpus must never surface a post-as_of document."""
import pytest

from swm.retrieval.corpus import Document, LeakageError, TimestampedCorpus

T0 = 1_000_000.0
DAY = 86400.0


def _corpus():
    return TimestampedCorpus([
        Document("past1", T0 - 3 * DAY, "election poll result", {"resolution": 1}),
        Document("past2", T0 - 2 * DAY, "election market resolved", {"resolution": 0}),
        Document("future", T0 + 5 * DAY, "election ANSWER leaks here", {"resolution": 1}),
    ])


def test_asof_excludes_future():
    c = _corpus()
    visible = c.as_of(T0)
    assert {d.doc_id for d in visible} == {"past1", "past2"}
    assert all(d.timestamp < T0 for d in visible)


def test_reference_class_never_returns_future():
    c = _corpus()
    # query strongly overlaps the FUTURE doc's text — similarity would rank it first if reachable
    ref = c.reference_class("election ANSWER leaks here", T0, k=10)
    assert "future" not in {d.doc_id for d in ref}
    assert all(d.timestamp < T0 for d in ref)


def test_document_requires_timestamp():
    with pytest.raises(ValueError):
        Document("bad", None, "no timestamp")


def test_guard_rejects_smuggled_future_doc():
    # even if a future doc is forced past the filter, the final guard raises rather than leak
    from swm.retrieval.corpus import _assert_no_future
    fut = Document("f", T0 + DAY, "x")
    with pytest.raises(LeakageError):
        _assert_no_future([fut], T0)


def test_empty_reference_class_is_safe():
    c = _corpus()
    assert c.reference_class("anything", T0 - 10 * DAY, k=5) == []   # nothing before this as_of
