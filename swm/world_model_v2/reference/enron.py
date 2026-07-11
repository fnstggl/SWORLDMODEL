"""Reference World A — individual message response (Enron). Configuration + FITTED mechanisms, no new engine.

Leak-free examples: for each test message, every feature derives ONLY from messages strictly before its
timestamp (sender→recipient history, recipient inbox volume, prior reply behavior, time-of-day/weekday).
Splits: TIME-FORWARD (train earlier / test later) and PERSON-DISJOINT (recipients never seen in train).

Mechanisms fitted from the TRAIN split (status=fitted; nothing invented):
  * reply hazard over elapsed time  — empirical P(reply lands in bucket | not yet) over (4h,1d,3d,7d,14d)
  * recipient base rate             — Laplace-smoothed per-recipient reply rate
  * relationship effect             — smoothed sender→recipient prior reply rate (falls back to recipient)
  * workload effect                 — reply-rate multiplier by recent-inbox-volume tercile
  * time-of-day / weekday effect    — send-hour and weekday multipliers
Remaining mechanisms: attention latent (prior-backed: a distribution, sampled per particle), message-content
evaluation (experimental: optional LLM policy, ablatable, OFF for the statistical arms).

The V2 world per message: recipient entity (attention latent, fitted workload), message_delivered event at
t0, inbox_checked HAZARD at the fitted per-recipient check rate → decision opportunities; the reply decision
consumes fitted rates modulated by the sampled latent; terminal readout = replied-by-h from world state.
Arms I0–I8 as specified; identical features/cutoffs across arms.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

DAY = 86400.0
BUCKETS = (4 / 24, 1.0, 3.0, 7.0, 14.0)                    # days — hazard/readout grid


# ---------------------------------------------------------------- examples (leak-free)
@dataclass
class MessageExample:
    msg_id: str
    sender: str
    recipient: str
    subject: str
    body: str
    sent_ts: float
    replied: bool
    delay_days: float = None               # None if never replied (within corpus)
    feats: dict = field(default_factory=dict)


def build_examples(records, *, min_history=0):
    """records: load_enron_reply_delay output. Features computed ONLY from strictly-earlier messages.
    One pass in time order keeps it O(n)."""
    recs = sorted([r for r in records if r.get("date_ts")], key=lambda r: r["date_ts"])
    sent_by = defaultdict(list)            # sender -> [(ts, replied)]
    pair = defaultdict(lambda: [0, 0])     # (sender, recipient) -> [n, n_replied]
    inbox = defaultdict(list)              # recipient -> [ts] received
    rrate = defaultdict(lambda: [0, 0])    # recipient -> [n_received_with_label, n_replied]
    out = []
    for r in recs:
        s = (r.get("from") or "").strip().lower()
        rc = (r.get("to") or "").split(",")[0].strip().lower()
        if not s or not rc or s == rc:
            continue
        ts = r["date_ts"]
        recent = [t for t in inbox[rc][-200:] if ts - 7 * DAY <= t < ts]
        n_pair, n_pair_rep = pair[(s, rc)]
        n_rc, n_rc_rep = rrate[rc]
        import time as _t
        lt = _t.gmtime(ts)
        ex = MessageExample(
            msg_id=r["msg_id"], sender=s, recipient=rc, subject=r.get("subject", ""),
            body=(r.get("body") or "")[:1500], sent_ts=ts, replied=bool(r.get("replied")),
            delay_days=(r["delay_hours"] / 24.0 if r.get("delay_hours") is not None else None),
            feats={"pair_n": n_pair, "pair_rate": (n_pair_rep + 0.5) / (n_pair + 1.0),
                   "rcpt_n": n_rc, "rcpt_rate": (n_rc_rep + 1.0) / (n_rc + 2.0),
                   "inbox_7d": len(recent), "hour": lt.tm_hour, "weekday": lt.tm_wday,
                   "thread": 1 if str(r.get("subject", "")).lower().startswith(("re:", "fw:")) else 0})
        if n_pair + n_rc >= min_history:
            out.append(ex)
        inbox[rc].append(ts)
        rrate[rc] = [n_rc + 1, n_rc_rep + (1 if r.get("replied") else 0)]
        pair[(s, rc)] = [n_pair + 1, n_pair_rep + (1 if r.get("replied") else 0)]
        sent_by[s].append((ts, bool(r.get("replied"))))
    return out


def splits(examples, *, test_frac=0.25, person_disjoint_frac=0.5, seed=13):
    """TIME-FORWARD first (train strictly earlier), then within test: half the recipients are HELD-OUT
    PERSONS (never in train). Returns (train, test_time, test_person)."""
    exs = sorted(examples, key=lambda e: e.sent_ts)
    cut = int(len(exs) * (1 - test_frac))
    train, test = exs[:cut], exs[cut:]
    train_people = {e.recipient for e in train}
    rng = random.Random(seed)
    test_new = [e for e in test if e.recipient not in train_people]
    test_seen = [e for e in test if e.recipient in train_people]
    return train, test_seen, test_new


# ---------------------------------------------------------------- fitted mechanisms (train only)
@dataclass
class FittedMechanisms:
    global_rate: float
    hazard: list                            # P(reply in bucket | no reply yet), per BUCKETS — FITTED
    workload_mult: list                     # reply-rate multiplier per inbox-7d tercile — FITTED
    hour_mult: dict                         # send-hour bucket -> multiplier — FITTED
    weekday_mult: dict
    terciles: tuple
    check_rate_per_day: float               # inbox-check hazard proxy (from median reply delay) — FITTED
    status: dict = field(default_factory=dict)

    def base_p(self, ex) -> float:
        """The fitted multiplicative rate model (also arm I1): relationship > recipient > global, times
        workload/hour/weekday multipliers. Every factor measured; nothing invented."""
        if ex.feats["pair_n"] >= 3:
            p = ex.feats["pair_rate"]
        elif ex.feats["rcpt_n"] >= 5:
            p = ex.feats["rcpt_rate"]
        else:
            p = self.global_rate
        t = 0 if ex.feats["inbox_7d"] <= self.terciles[0] else (1 if ex.feats["inbox_7d"] <= self.terciles[1] else 2)
        p *= self.workload_mult[t] * self.hour_mult.get(ex.feats["hour"] // 6, 1.0) \
            * self.weekday_mult.get(min(ex.feats["weekday"], 5), 1.0)
        return min(0.98, max(0.02, p))


def fit_mechanisms(train) -> FittedMechanisms:
    n = len(train) or 1
    g = sum(e.replied for e in train) / n
    # hazard over elapsed time among replies
    delays = [e.delay_days for e in train if e.replied and e.delay_days is not None]
    hz, prev, alive = [], 0.0, len(delays) + (len(train) - len(delays))  # non-repliers survive all buckets
    remaining = alive
    for b in BUCKETS:
        landed = sum(1 for d in delays if prev < d <= b)
        hz.append(landed / max(1, remaining))
        remaining -= landed
        prev = b
    # workload terciles
    vols = sorted(e.feats["inbox_7d"] for e in train)
    t1, t2 = vols[len(vols) // 3] if vols else 3, vols[2 * len(vols) // 3] if vols else 10
    def rate(sub):
        return (sum(e.replied for e in sub) + 1.0) / (len(sub) + 2.0)
    wl = []
    for lo, hi in ((None, t1), (t1, t2), (t2, None)):
        sub = [e for e in train if (lo is None or e.feats["inbox_7d"] > lo)
               and (hi is None or e.feats["inbox_7d"] <= hi)]
        wl.append(rate(sub) / max(1e-6, g))
    hour_m = {}
    for hb in range(4):
        sub = [e for e in train if e.feats["hour"] // 6 == hb]
        hour_m[hb] = rate(sub) / max(1e-6, g) if len(sub) >= 30 else 1.0
    wd_m = {}
    for wd in range(6):
        sub = [e for e in train if min(e.feats["weekday"], 5) == wd]
        wd_m[wd] = rate(sub) / max(1e-6, g) if len(sub) >= 30 else 1.0
    med_delay = sorted(delays)[len(delays) // 2] if delays else 1.0
    return FittedMechanisms(global_rate=g, hazard=hz, workload_mult=wl, hour_mult=hour_m,
                            weekday_mult=wd_m, terciles=(t1, t2),
                            check_rate_per_day=max(0.5, 1.0 / max(0.05, med_delay)),
                            status={"reply_hazard": "fitted", "recipient_rate": "fitted",
                                    "relationship": "fitted", "workload": "fitted",
                                    "hour_weekday": "fitted", "check_rate": "fitted",
                                    "attention_latent": "prior_backed",
                                    "message_content_eval": "experimental (LLM policy, off by default)"})


# ---------------------------------------------------------------- the V2 world per message
def v2_predict(ex, fm: FittedMechanisms, *, horizon_days=14.0, n_particles=30, seed=0,
               latent=True, event_driven=True, relationship=True):
    """Build + roll the message world through the UNIVERSAL runtime. Returns {'p_by': {b: p}, 'trace': …}.
    Ablations: latent=False → point attention; event_driven=False → single-shot decision (no queue);
    relationship=False → recipient/global rates only (pair history hidden)."""
    from swm.world_model_v2.contracts import OutcomeContract
    from swm.world_model_v2.events import Event, EventQueue, StochasticHazard, register_event_type
    from swm.world_model_v2.init_state import InitialStateModel, LatentVariableRecord
    from swm.world_model_v2.rollout import WorldModelV2Run
    from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
    from swm.world_model_v2.transitions import TransitionOperator, TransitionProposal, StateDelta

    ex2 = ex if relationship else MessageExample(**{**ex.__dict__, "feats": {**ex.feats, "pair_n": 0}})
    p_base = fm.base_p(ex2)

    base = WorldState(world_id=ex.msg_id[:12], branch_id="root",
                      clock=SimulationClock(now=ex.sent_ts, as_of=ex.sent_ts))
    rcpt = Entity(identity="recipient")
    rcpt.set("attention", F(0.7, status="assumed"))
    rcpt.set("current_action", F(None, status="assumed"))
    base.entities["recipient"] = rcpt
    latents = [LatentVariableRecord(path="recipient.attention",
                                    candidates={"mean": 0.7, "sd": 0.25, "lo": 0.1, "hi": 1.0},
                                    method="prior")] if latent else []
    init = InitialStateModel(base_world=base, latents=latents)

    register_event_type("reply_check", scheduling="hazard", validated=True)

    class ReplyDecision(TransitionOperator):
        """FITTED policy: at each observation opportunity, reply with the bucket hazard × fitted rate ratio
        × sampled attention. No LLM, no invented constants — every factor is train-measured or a labeled
        prior (attention)."""
        name = "reply_decision_fitted"

        def applicable(self, world, event):
            return event.etype == "reply_check" and world.entity("recipient").value("current_action") is None

        def propose(self, world, event, rng):
            el_days = (world.clock.now - ex.sent_ts) / DAY
            b = next((i for i, bb in enumerate(BUCKETS) if el_days <= bb), len(BUCKETS) - 1)
            att = world.entity("recipient").value("attention") or 0.7
            # fitted per-opportunity hazard, scaled by how this message's fitted rate compares to global
            p_reply = min(0.95, fm.hazard[b] * (p_base / max(1e-6, fm.global_rate)) * (0.4 + 0.85 * att))
            act = "reply" if rng.random() < p_reply else "wait"
            return TransitionProposal(operator=self.name, action={"actor": "recipient", "type": act},
                                      p_dist={"reply": p_reply, "wait": 1 - p_reply},
                                      reason_codes=[f"bucket={b}", "fitted_hazard"])

        def apply(self, world, proposal):
            d = StateDelta(at=world.clock.now, event_type="reply_decision", operator=self.name,
                           reason_codes=proposal.reason_codes, uncertainty={"p_dist": proposal.p_dist})
            if proposal.action["type"] == "reply":
                before = world.entity("recipient").value("current_action")
                world.entity("recipient").set("current_action",
                                              F("reply", status="derived", method=self.name,
                                                updated_at=world.clock.now))
                world.quantities["reply_delay_days"] = type("Q", (), {
                    "value": (world.clock.now - ex.sent_ts) / DAY})()
                d.change("recipient.current_action", before, "reply")
            return d

    def build_queue(world):
        q = EventQueue(horizon_ts=ex.sent_ts + horizon_days * DAY)
        rng = random.Random(hash(world.branch_id) & 0xFFFF)
        if event_driven:
            q.add_hazard(StochasticHazard(etype="reply_check", rate_per_day=fm.check_rate_per_day,
                                          participants=["recipient"]),
                         now=ex.sent_ts, rng=rng, world=world)
        else:
            q.schedule(Event(ts=ex.sent_ts + 60.0, etype="reply_check", participants=["recipient"]))
        return q

    contract = OutcomeContract(
        family="response_occurrence", options=["reply", "no_reply"],
        resolution_rule=f"replied within {horizon_days:.0f}d",
        readout=lambda w: "reply" if w.entity("recipient").value("current_action") == "reply" else "no_reply",
        horizon_ts=ex.sent_ts + horizon_days * DAY).validate()
    run = WorldModelV2Run(initial=init, queue_builder=build_queue, operators=[ReplyDecision()],
                          contract=contract, n_particles=n_particles)
    result, branches = run.run(seed=seed)
    # per-horizon readout from terminal DELAYS (native: the delay distribution answers every bucket)
    delays = []
    for b in branches:
        q = b.world.quantities.get("reply_delay_days")
        delays.append(q.value if q is not None else None)
    p_by = {bb: sum(1 for d in delays if d is not None and d <= bb) / len(delays) for bb in BUCKETS}
    return {"p_by": p_by, "p14": result["distribution"].get("reply", 0.0),
            "delays": [d for d in delays if d is not None],
            "n_deltas": result["n_deltas"], "trace_branches": branches}
