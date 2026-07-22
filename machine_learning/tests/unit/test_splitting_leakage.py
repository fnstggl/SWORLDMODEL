"""Split assignment determinism + isolation + leakage detection."""
from machine_learning.normalization.base import load_converter
from machine_learning.config import FIXTURES_DIR
from machine_learning.splitting.policies import SplitPolicy, assign_split
from machine_learning.splitting.leakage_checks import LeakageReport


def _casino_records():
    conv = load_converter("casino", timestamp="2026-07-22T00:00:00Z")
    return list(conv.iter_records(FIXTURES_DIR / "casino"))


def test_split_is_deterministic():
    pol = SplitPolicy.from_registry("casino")
    recs = _casino_records()
    a = [assign_split(r, pol)[0] for r in recs]
    b = [assign_split(r, pol)[0] for r in recs]
    assert a == b


def test_episode_maps_to_single_split():
    pol = SplitPolicy.from_registry("casino")
    recs = _casino_records()
    by_ep = {}
    for r in recs:
        sp, _ = assign_split(r, pol)
        by_ep.setdefault(r["episode"]["episode_id"], set()).add(sp)
    for ep, splits in by_ep.items():
        assert len(splits) == 1, (ep, splits)


def test_eval_only_all_cross_dataset():
    pol = SplitPolicy.from_registry("craigslistbargain")
    assert pol.eval_only
    conv = load_converter("craigslistbargain", timestamp="2026-07-22T00:00:00Z")
    recs = list(conv.iter_records(FIXTURES_DIR / "craigslistbargain"))
    for r in recs:
        assert assign_split(r, pol)[0] == "test_cross_dataset"


def test_leakage_check_flags_planted_violation():
    # simulate a split table where one episode straddles two splits
    rows = [
        {"episode_id": "ep1", "split": "train", "content_hash": "h1",
         "isolation_keys_json": '{"conversation": "ep1"}'},
        {"episode_id": "ep1", "split": "test_in_domain", "content_hash": "h2",
         "isolation_keys_json": '{"conversation": "ep1"}'},
    ]
    from machine_learning.splitting import leakage_checks as LC
    orig = LC.load_split_table
    LC.load_split_table = lambda _did: rows
    try:
        rep = LC.check_dataset("fake")
    finally:
        LC.load_split_table = orig
    assert isinstance(rep, LeakageReport)
    assert not rep.ok
    assert rep.episode_violations
    assert rep.unit_violations
