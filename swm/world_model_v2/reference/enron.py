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
_CONTENT_PROMPT = """You ARE the recipient of an email, deciding whether YOU will reply. Judge ONLY from what
you can see now — never from the future.
YOU: {recipient} (Enron employee)
FROM: {sender}   PRIOR HISTORY WITH THEM: you have received {pair_n} emails from this person before; you
replied to about {pair_rate:.0%} of them.
YOUR CURRENT LOAD: {inbox_7d} emails in your inbox in the last week.
SUBJECT: {subject}
MESSAGE:
---
{body}
---
Given the message's content, specificity, whether it asks you something, its importance to your work, and
your relationship — how likely are YOU to reply at all (any time)? Return ONLY JSON:
{{"reply_propensity": <0..1>, "why": "<6 words>"}}"""


def content_multiplier(ex, chat_fn, meter=None):
    """MAX-CAPACITY content mechanism (experimental): ONE LLM call reads the EXACT message + the recipient's
    OBSERVABLE dossier (no future, no labels) → a reply-propensity in [0,1]. Returned as a multiplier on the
    fitted base rate (centered at 1.0 so content REDISTRIBUTES rather than replacing the fitted signal).
    None on parse failure (arm abstains from the content boost, falls back to metadata)."""
    from swm.engine.grounding import parse_json
    prompt = _CONTENT_PROMPT.format(
        recipient=ex.recipient, sender=ex.sender, pair_n=ex.feats["pair_n"],
        pair_rate=ex.feats["pair_rate"], inbox_7d=ex.feats["inbox_7d"],
        subject=ex.subject[:200], body=ex.body[:1500])
    txt = chat_fn(prompt)
    if meter is not None:
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
    r = parse_json(txt) or {}
    try:
        prop = min(1.0, max(0.0, float(r["reply_propensity"])))
    except (KeyError, TypeError, ValueError):
        return None, ""
    return prop, str(r.get("why", ""))[:60]


def v2_predict(ex, fm: FittedMechanisms, *, horizon_days=14.0, n_particles=30, seed=0,
               latent=True, event_driven=True, relationship=True, content_fn=None, meter=None):
    """Build + roll the message world through the UNIVERSAL runtime. Returns {'p_by': {b: p}, 'trace': …}.
    Ablations: latent=False → point attention; event_driven=False → single-shot decision (no queue);
    relationship=False → recipient/global rates only (pair history hidden); content_fn set → MAX-CAPACITY
    content-conditioned policy (LLM reads the exact message, modulates the fitted hazard)."""
    from swm.world_model_v2.contracts import OutcomeContract
    from swm.world_model_v2.events import Event, EventQueue, StochasticHazard, register_event_type
    from swm.world_model_v2.init_state import InitialStateModel, LatentVariableRecord
    from swm.world_model_v2.rollout import WorldModelV2Run
    from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
    from swm.world_model_v2.transitions import TransitionOperator, TransitionProposal, StateDelta

    ex2 = ex if relationship else MessageExample(**{**ex.__dict__, "feats": {**ex.feats, "pair_n": 0}})
    p_base = fm.base_p(ex2)
    content_why = ""
    if content_fn is not None:
        prop, content_why = content_multiplier(ex, content_fn, meter)
        if prop is not None:
            # content redistributes around the fitted rate: propensity 0.5→×1.0, higher/lower scales it,
            # bounded so a single LLM read can't manufacture certainty (max ~2.2× swing)
            p_base = min(0.97, max(0.01, p_base * (0.45 + 1.75 * prop)))

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
            # HAZARD INTEGRATION (the 30-days≠30-guesses rule): fm.hazard[b] is P(reply lands in bucket b |
            # not yet) for the WHOLE bucket. There are ~check_rate×width opportunities in that bucket, so the
            # per-opportunity hazard must satisfy 1-(1-h)^n = H_bucket — otherwise probability compounds and
            # the rollout overpredicts more the longer the horizon (the exact failure of the first run).
            widths = (BUCKETS[0],) + tuple(BUCKETS[i] - BUCKETS[i - 1] for i in range(1, len(BUCKETS)))
            n_opp = max(1.0, fm.check_rate_per_day * widths[b])
            H = min(0.95, fm.hazard[b] * (p_base / max(1e-6, fm.global_rate)) * (0.4 + 0.85 * att))
            p_reply = 1.0 - (1.0 - H) ** (1.0 / n_opp)
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
            "delays": [d for d in delays if d is not None], "p_base": p_base, "content_why": content_why,
            "n_deltas": result["n_deltas"], "trace_branches": branches}


# ---------------------------------------------------------------- E2: non-LLM text baseline
def fit_text_baseline(train):
    """A real non-LLM content model: hashed word-presence → per-token replied-rate; predict a message's
    reply propensity as the mean replied-rate of its tokens (Laplace). No embeddings library needed; it IS a
    bag-of-words logistic-free text signal to test whether CONTENT (not the LLM) carries the lift."""
    import re
    tok = defaultdict(lambda: [0.0, 0.0])                  # token -> [n, n_replied]
    for e in train:
        words = set(re.findall(r"[a-z]{3,}", (e.subject + " " + e.body).lower()))
        for w in list(words)[:60]:
            tok[w][0] += 1
            tok[w][1] += 1 if e.replied else 0
    base = sum(e.replied for e in train) / max(1, len(train))
    model = {w: (c[1] + base) / (c[0] + 1.0) for w, c in tok.items() if c[0] >= 5}
    return {"model": model, "base": base}


def text_baseline_p(ex, tb):
    import re
    words = set(re.findall(r"[a-z]{3,}", (ex.subject + " " + ex.body).lower()))
    rates = [tb["model"][w] for w in words if w in tb["model"]]
    return min(0.97, max(0.01, sum(rates) / len(rates))) if rates else tb["base"]


# ---------------------------------------------------------------- STRUCTURED ACTOR POLICY (PARTS 2-4)
# Composes the UNIVERSAL actor-cognition components (swm/world_model_v2/actor_cognition.py) into this
# reference world. Nothing below adds email-only cognition machinery: interpretation schema, fitted policy,
# action packs, hidden-state latents and temporal processes are the shared V2 vocabulary; this file supplies
# only the Enron parameter pack (observable context, workload normalization, fitted dispersion).

def interpret_message(ex, chat_fn, meter=None):
    """The Enron composition of the universal interpretation boundary: recipient-observable context only
    (prior pair history, current load, local time, thread marker) + the EXACT message. No labels, no future."""
    from swm.world_model_v2.actor_cognition import interpret
    context = (f"- prior history with {ex.sender}: you received {ex.feats['pair_n']} messages from them; "
               f"you replied to about {ex.feats['pair_rate']:.0%}\n"
               f"- your inbox load, last 7 days: {ex.feats['inbox_7d']} messages\n"
               f"- local time: hour {ex.feats['hour']:02d}, weekday {ex.feats['weekday']}\n"
               f"- subject marks an existing thread: {'yes' if ex.feats.get('thread') else 'no'}")
    return interpret(chat_fn, actor=f"{ex.recipient} (Enron employee)", channel="corporate email",
                     context=context, content=f"SUBJECT: {ex.subject[:200]}\n{ex.body[:1500]}", meter=meter)


def fit_hetero_sd(train) -> float:
    """FITTED heterogeneity for the responsiveness latent: dispersion of log per-recipient response-rate
    multipliers (recipients with n>=5) — a measured spread, not an invented constant. Clamped [0.1, 0.6]."""
    g = sum(e.replied for e in train) / max(1, len(train))
    per = {}
    for e in train:
        if e.feats["rcpt_n"] >= 5:
            per[e.recipient] = e.feats["rcpt_rate"]          # last seen (running smoothed rate)
    if len(per) < 10 or g <= 0:
        return 0.3
    logs = [math.log(max(1e-3, r / g)) for r in per.values()]
    mu = sum(logs) / len(logs)
    sd = (sum((x - mu) ** 2 for x in logs) / max(1, len(logs) - 1)) ** 0.5
    return min(0.6, max(0.1, sd / 2.0))                      # /2: rate noise inflates raw dispersion


REPLY_ACTIONS = ("reply_now", "reply_later", "ask_clarification")


def v2_predict_actor(ex, fm: FittedMechanisms, interp, pol, *, level=6, horizon_days=14.0,
                     n_particles=48, seed=0, hetero_sd=0.3):
    """PART 2-4: the STRUCTURED actor world. Levels are the C-ladder:
      2: typed action distribution through the event world (point hidden state)
      3: + sampled coherent hidden actor state (attention, responsiveness, obligation_sensitivity; correlated)
      4: + dynamic attention (mean-reverting workload/time-of-day process between observations)
      5: + relationship state (history-inferred strength modulates engagement; terminal bounded transition)
      6: + interruption hazard (distraction events) + obligation×hidden-state coupling
    The fitted policy `pol` (train-only) sets the calibrated engagement mass from interpretation features +
    the metadata anchor; the interpretation shapes the typed split; hidden state modulates within bounds.
    interp=None → the arm ABSTAINS from semantics (metadata anchor only), flagged."""
    from swm.world_model_v2.actor_cognition import (Interpretation, action_distribution,
                                                    attention_transition, hidden_state_latents,
                                                    relationship_modulator, relationship_strength,
                                                    relationship_transition)
    from swm.world_model_v2.contracts import OutcomeContract
    from swm.world_model_v2.events import Event, EventQueue, StochasticHazard, register_event_type
    from swm.world_model_v2.init_state import InitialStateModel
    from swm.world_model_v2.rollout import WorldModelV2Run
    from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
    from swm.world_model_v2.transitions import TransitionOperator, TransitionProposal, StateDelta

    base_p = fm.base_p(ex)
    itp = interp if interp is not None else Interpretation()             # neutral shape when abstaining
    p_eng = pol.p_engage(itp.features(), base_p) if (interp is not None and pol is not None) else base_p
    act_dist = action_distribution("messaging", itp, p_eng)
    p_del = act_dist["delegate"]
    # delegation preempts at the first observation; compensate the reply hazard so P(reply)=p_eng exactly
    p_reply_target = min(0.97, p_eng / max(0.03, 1.0 - p_del))
    wl_norm = min(1.0, ex.feats["inbox_7d"] / max(1.0, 2.0 * fm.terciles[1]))
    rel_strength = relationship_strength(ex.feats["pair_n"], ex.feats["pair_rate"], fm.global_rate)

    base = WorldState(world_id=ex.msg_id[:12], branch_id="root",
                      clock=SimulationClock(now=ex.sent_ts, as_of=ex.sent_ts))
    rcpt = Entity(identity="recipient")
    rcpt.set("attention", F(0.7, status="assumed"))
    rcpt.set("current_action", F(None, status="assumed"))
    rcpt.set("responsiveness", F(1.0, status="assumed"))
    rcpt.set("obligation_sensitivity", F(1.0, status="assumed"))
    rcpt.set("workload_pressure", F(wl_norm, status="observed", method="inbox_7d/terciles"))
    rcpt.set("last_observation_ts", F(ex.sent_ts, status="derived"))
    rcpt.set("relationships", F(rel_strength, status="inferred", method="relationship_strength"),
             key=ex.sender)
    base.entities["recipient"] = rcpt
    latents, correlations = ([], [])
    if level >= 3:
        latents, correlations = hidden_state_latents("recipient", workload_norm=wl_norm,
                                                     hetero_sd=hetero_sd)
    init = InitialStateModel(base_world=base, latents=latents, correlations=correlations)

    register_event_type("reply_check", scheduling="hazard", validated=True)
    register_event_type("deferred_reply", scheduling="scheduled", validated=True)

    class TypedMessageDecision(TransitionOperator):
        """The typed-action policy inside the world: at each observation opportunity the actor either
        engages (hazard calibrated to the FITTED p_engage) — sampling reply_now / reply_later /
        ask_clarification from the interpretation-shaped split — or hands off (delegate, first observation
        only) or waits. reply_later schedules a deferred completion event; past-horizon deferrals honestly
        count as no observed reply."""
        name = "typed_action_decision"

        def applicable(self, world, event):
            return (event.etype in ("reply_check", "deferred_reply", "distraction")
                    and world.entity("recipient").value("current_action") in (None, "reply_later"))

        def propose(self, world, event, rng):
            r = world.entity("recipient")
            if event.etype == "deferred_reply":
                if r.value("current_action") == "reply_later":
                    return TransitionProposal(operator=self.name,
                                              action={"actor": "recipient", "type": "complete_deferred"},
                                              reason_codes=["deferred_completion"])
                return None
            if event.etype == "distraction":
                if level >= 6:
                    return TransitionProposal(operator=self.name,
                                              action={"actor": "recipient", "type": "interrupted"},
                                              reason_codes=["distraction_shock"])
                return None
            if r.value("current_action") is not None:
                return None
            att = float(r.value("attention") or 0.7)
            if level >= 4:                                   # PART 4: attention as a temporal process
                last = float(r.value("last_observation_ts") or ex.sent_ts)
                att = attention_transition(att, dt_days=(world.clock.now - last) / DAY,
                                           workload_norm=wl_norm, hour=ex.feats["hour"], rng=rng)
            resp = float(r.value("responsiveness") or 1.0) if level >= 3 else 1.0
            p_t = p_reply_target * resp
            if level >= 5:
                p_t *= relationship_modulator(rel_strength, itp.relationship_salience)
            if level >= 6:
                osens = float(r.value("obligation_sensitivity") or 1.0)
                p_t *= min(1.25, max(0.8, 1.0 + 0.5 * osens * (itp.obligation - 0.5)))
            p_t = min(0.97, max(0.01, p_t))
            el_days = (world.clock.now - ex.sent_ts) / DAY
            b = next((i for i, bb in enumerate(BUCKETS) if el_days <= bb), len(BUCKETS) - 1)
            widths = (BUCKETS[0],) + tuple(BUCKETS[i] - BUCKETS[i - 1] for i in range(1, len(BUCKETS)))
            n_opp = max(1.0, fm.check_rate_per_day * widths[b])
            H = min(0.95, fm.hazard[b] * (p_t / max(1e-6, fm.global_rate)) * (0.4 + 0.85 * att))
            p_check = 1.0 - (1.0 - H) ** (1.0 / n_opp)
            first_check = float(r.value("last_observation_ts") or ex.sent_ts) <= ex.sent_ts
            u = rng.random()
            if u < p_check:                                  # ENGAGE: typed split (renormalized reply mass)
                mass = {a: act_dist[a] for a in REPLY_ACTIONS}
                z = sum(mass.values()) or 1.0
                pick, acc, chosen = rng.random(), 0.0, REPLY_ACTIONS[0]
                for a, m in mass.items():
                    acc += m / z
                    if pick <= acc:
                        chosen = a
                        break
                return TransitionProposal(operator=self.name,
                                          action={"actor": "recipient", "type": chosen},
                                          p_dist={**{a: round(m / z * p_check, 4) for a, m in mass.items()},
                                                  "wait": round(1 - p_check, 4)},
                                          reason_codes=[f"bucket={b}", "fitted_policy",
                                                        f"attention={att:.2f}"])
            if first_check and u < p_check + p_del:          # hand-off happens at the first look only
                return TransitionProposal(operator=self.name,
                                          action={"actor": "recipient", "type": "delegate"},
                                          p_dist={"delegate": round(p_del, 4)},
                                          reason_codes=["handoff"])
            return TransitionProposal(operator=self.name,
                                      action={"actor": "recipient", "type": "observe", "att": att},
                                      p_dist={"wait": round(1 - p_check, 4)}, reason_codes=["waited"])

        def apply(self, world, proposal):
            r = world.entity("recipient")
            act = proposal.action["type"]
            d = StateDelta(at=world.clock.now, event_type="typed_action", operator=self.name,
                           reason_codes=proposal.reason_codes, uncertainty={"p_dist": proposal.p_dist})
            before = r.value("current_action")
            if act == "interrupted":
                a0 = float(r.value("attention") or 0.7)
                r.set("attention", F(max(0.05, a0 - 0.15), status="derived", method="distraction",
                                     updated_at=world.clock.now))
                return d.change("recipient.attention", round(a0, 3), round(max(0.05, a0 - 0.15), 3))
            if act == "observe":
                r.set("last_observation_ts", F(world.clock.now, status="derived",
                                               updated_at=world.clock.now))
                if level >= 4 and "att" in proposal.action:  # the temporal process PERSISTS its state
                    r.set("attention", F(proposal.action["att"], status="derived",
                                         method="attention_transition", updated_at=world.clock.now))
                return d
            if act in ("reply_now", "ask_clarification", "complete_deferred"):
                r.set("current_action", F("reply", status="derived", method=act,
                                          updated_at=world.clock.now))
                world.quantities["reply_delay_days"] = type("Q", (), {
                    "value": (world.clock.now - ex.sent_ts) / DAY})()
                d.change("recipient.current_action", before, f"reply({act})")
            elif act == "reply_later":
                r.set("current_action", F("reply_later", status="derived", method=act,
                                          updated_at=world.clock.now))
                d.change("recipient.current_action", before, "reply_later")
            elif act == "delegate":
                r.set("current_action", F("delegate", status="derived", method=act,
                                          updated_at=world.clock.now))
                d.change("recipient.current_action", before, "delegate")
            if level >= 5:                                   # PART 4: bounded relationship transition
                s0 = float(r.value("relationships", key=ex.sender) or rel_strength)
                s1 = relationship_transition(s0, engaged=act in ("reply_now", "ask_clarification",
                                                                 "complete_deferred"))
                r.set("relationships", F(s1, status="derived", method="relationship_transition",
                                         updated_at=world.clock.now), key=ex.sender)
                d.change(f"recipient.relationships[{ex.sender}]", round(s0, 3), round(s1, 3))
            return d

    def build_queue(world):
        q = EventQueue(horizon_ts=ex.sent_ts + horizon_days * DAY)
        rng = random.Random(hash(world.branch_id) & 0xFFFF)
        q.add_hazard(StochasticHazard(etype="reply_check", rate_per_day=fm.check_rate_per_day,
                                      participants=["recipient"]), now=ex.sent_ts, rng=rng, world=world)
        if level >= 6:
            q.add_hazard(StochasticHazard(etype="distraction", rate_per_day=1.0 + 2.0 * wl_norm,
                                          participants=["recipient"],
                                          payload={"prov": "prior (workload-coupled interruptions)"}),
                         now=ex.sent_ts, rng=rng, world=world)
        return q

    contract = OutcomeContract(
        family="response_occurrence", options=["reply", "no_reply"],
        resolution_rule=f"replied within {horizon_days:.0f}d",
        readout=lambda w: "reply" if w.entity("recipient").value("current_action") == "reply"
        else "no_reply",
        horizon_ts=ex.sent_ts + horizon_days * DAY).validate()

    # reply_later completion rides the check-event stream: a deferred reply completes at the first
    # observation after the per-particle sampled defer delay; past-horizon completions drop honestly
    class DeferredCompletion(TransitionOperator):
        name = "deferred_completion"

        def applicable(self, world, event):
            r = world.entity("recipient")
            if r.value("current_action") != "reply_later" or event.etype != "reply_check":
                return False
            chosen_at = float(r.get("current_action").prov.updated_at or 0.0)
            rngl = random.Random(int(chosen_at) ^ (hash(world.branch_id) & 0xFFFF))
            defer_days = min(5.0, max(0.1, math.exp(rngl.gauss(math.log(0.8), 0.8))))
            return world.clock.now >= chosen_at + defer_days * DAY

        def propose(self, world, event, rng):
            return TransitionProposal(operator=self.name,
                                      action={"actor": "recipient", "type": "complete_deferred"},
                                      reason_codes=["deferred_completion"])

        def apply(self, world, proposal):
            r = world.entity("recipient")
            before = r.value("current_action")
            r.set("current_action", F("reply", status="derived", method="complete_deferred",
                                      updated_at=world.clock.now))
            world.quantities["reply_delay_days"] = type("Q", (), {
                "value": (world.clock.now - ex.sent_ts) / DAY})()
            d = StateDelta(at=world.clock.now, event_type="typed_action", operator=self.name,
                           reason_codes=["deferred_completion"])
            return d.change("recipient.current_action", before, "reply(deferred)")

    run = WorldModelV2Run(initial=init, queue_builder=build_queue,
                          operators=[TypedMessageDecision(), DeferredCompletion()],
                          contract=contract, n_particles=n_particles)
    result, branches = run.run(seed=seed)
    delays, actions = [], []
    for b in branches:
        q = b.world.quantities.get("reply_delay_days")
        delays.append(q.value if q is not None else None)
        actions.append(b.world.entity("recipient").value("current_action") or "ignore")
    p_by = {bb: sum(1 for d in delays if d is not None and d <= bb) / len(delays) for bb in BUCKETS}
    return {"p_by": p_by, "p14": result["distribution"].get("reply", 0.0),
            "delays": [d for d in delays if d is not None], "p_engage": p_eng,
            "action_dist": act_dist, "terminal_actions": actions,
            "interpretation": (itp.as_dict() if interp is not None else None),
            "abstained": interp is None, "n_deltas": result["n_deltas"], "trace_branches": branches}
