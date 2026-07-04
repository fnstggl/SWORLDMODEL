"""End-to-end: synthetic threads -> store -> personas -> readout -> ladder -> world.

The synthetic population has REAL structure (per-person response functions that interact with
message features), so the test can assert the pipeline recovers signal — the same shape of claim
exp001 makes on real data.
"""
import math
import random

from swm.entities.persona import apply_correction, build_persona
from swm.ingestion.importers import import_threads
from swm.ingestion.store import EventStore
from swm.worlds.world import World

DAY = 86400.0
T0 = 1_700_000_000.0


def synth_threads(n_contacts=12, sends_per_contact=25, seed=7):
    """Each contact has a latent response function: base responsiveness + a penalty for
    long messages if they prefer short ones (the interaction the model must find)."""
    rng = random.Random(seed)
    threads = []
    for c in range(n_contacts):
        contact = f"contact{c}@example.com"
        base = rng.uniform(0.15, 0.65)              # latent responsiveness
        prefers_short = rng.random() < 0.5          # latent style preference
        msgs = []
        t = T0 + c * 1000
        for k in range(sends_per_contact):
            t += rng.uniform(1, 3) * DAY
            long_msg = rng.random() < 0.5
            body = ("Hey, quick one — are you free this week to chat? " if not long_msg else
                    "Hello, I hope this message finds you well. I wanted to reach out regarding "
                    "a number of topics we have been considering, including the roadmap, the "
                    "budget, and several other items that I believe warrant a longer discussion "
                    "at your earliest convenience. Please let me know your availability. " * 2)
            p = base - (0.25 if (long_msg and prefers_short) else 0.0)
            msgs.append({"from": "me@example.com", "to": [contact], "timestamp": t, "text": body})
            if rng.random() < max(0.02, p):
                reply_len = 8 if prefers_short else 60
                msgs.append({"from": contact, "to": ["me@example.com"],
                             "timestamp": t + rng.uniform(0.1, 0.8) * DAY,
                             "text": "ok sounds good " * (reply_len // 3)})
        threads.append({"thread_id": f"th-{c}", "channel": "email", "messages": msgs})
    return threads


def _loaded_store():
    store = EventStore(":memory:")
    counts = import_threads(store, synth_threads(), owner_id="me@example.com")
    assert counts["messages_out"] > 200
    return store


def test_labels_derived_correctly():
    store = _loaded_store()
    sends = store.labeled_sends()
    rate = sum(s.replied for s in sends) / len(sends)
    assert 0.05 < rate < 0.8  # sane, non-degenerate labels


def test_persona_asof_uses_no_future():
    store = _loaded_store()
    sends = store.labeled_sends()
    early = sends[0]
    p = build_persona(early.recipient_id, store.history_asof(early.recipient_id, early.timestamp))
    # before the first send there is no evidence: posterior ~= prior
    assert p.n_sends == 0
    assert abs(p.responsiveness.mean - 0.3) < 0.05


def test_correction_shifts_posterior_but_is_outvotable():
    store = _loaded_store()
    rid = store.recipients()[0]
    p = build_persona(rid, store.history_asof(rid, T0 + 400 * DAY))
    before = p.verbosity.mean
    apply_correction(p, "verbosity", 2.3, confidence=1.0)
    assert p.verbosity.mean != before
    # strong contrary evidence outweighs one correction
    for _ in range(50):
        p.verbosity.update(math.log(90))
    assert p.verbosity.mean > 3.5


def test_ladder_runs_and_person_signal_exists():
    store = _loaded_store()
    from swm.eval.harness import run_ladder

    result = run_ladder(store)
    assert "error" not in result, result
    names = [r["name"] for r in result["rungs"]]
    assert names == ["L0_base_rate", "L1", "L2", "L3", "L4"]
    l0 = result["rungs"][0]["log_loss"]
    l3 = next(r for r in result["rungs"] if r["name"] == "L3")["log_loss"]
    # synthetic data HAS per-person signal by construction; the ladder must find it
    assert l3 < l0, f"person rung {l3} should beat base rate {l0}"


def test_world_fit_predict_compare_voi():
    store = _loaded_store()
    w = World(store=store)
    out = w.fit()
    assert out.get("fitted"), out
    rid = store.recipients()[0]
    pred = w.predict(rid, "Hey — quick one: free to chat this week?")
    assert pred["report_type"] == "prediction"
    assert 0.0 < pred["p_mean"] < 1.0
    assert pred["p_interval80"][0] <= pred["p_mean"] <= pred["p_interval80"][1]
    cmp_out = w.compare(rid, ["Hey — free this week?", "Dear colleague, " + "very long text " * 60])
    assert len(cmp_out["ranked"]) == 2
    # VOI either returns a well-formed question or honestly declines
    q = w.voi(rid, "Hey — free this week?")
    assert q is None or (q["report_type"] == "insight" and q["question"])
    # a correction round-trips
    c = w.correct(rid, "verbosity", "short & punchy")
    assert "persona" in c
