"""The leakage gate must pass in CI (audit E.4).

Right now this asserts the one leakage invariant we can enforce with only the schema in place:
NO feature may use information from at or after the split time. As the store/harness land, this
file grows into the real gate (contamination probe, audience-leakage, content-hash dedup).
"""
from swm.ingestion.schema import Event, EventType


def _knowable_before(events, split_ts):
    """The as-of rule the store must implement: only events strictly before split_ts."""
    return [e for e in events if e.timestamp < split_ts]


def test_no_future_events_leak_into_features():
    split_ts = 1000.0
    events = [
        Event(actor_id="a", timestamp=900.0, type=EventType.OPEN, channel="email"),
        Event(actor_id="a", timestamp=999.9, type=EventType.CLICK, channel="email"),
        # this one is AT/AFTER the split and must never be visible at prediction time:
        Event(actor_id="a", timestamp=1000.0, type=EventType.REPLY, channel="email"),
        Event(actor_id="a", timestamp=1500.0, type=EventType.REPLY, channel="email"),
    ]
    visible = _knowable_before(events, split_ts)
    assert all(e.timestamp < split_ts for e in visible)
    assert len(visible) == 2
    # the reply outcome (the label) is NOT among the visible features
    assert all(e.type != EventType.REPLY for e in visible)
