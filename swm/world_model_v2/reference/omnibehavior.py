"""Reference World C — longitudinal individual engagement (OmniBehavior, CC-BY-NC-SA, benchmark use).

REPAIRED TASK (the prior harness was invalid twice over):
  1. last-8-events sampling oversampled engaged periods (documented earlier), AND
  2. the label is nearly DETERMINED by event type: E-commerce/Customer-Service/Advertisement events are
     action RECORDS (pooled rate ≈ 1.000) — logged because the user acted — while Video/Live/Search are
     passive EXPOSURES (rates 0.00–0.22). Predicting "acted" on an E-commerce event is a type lookup.
Repair: targets are PASSIVE-EXPOSURE events only (Video Browsing / Live Streaming / Search Behavior),
sampled UNIFORMLY over each user's chronological test segment; action-record events remain visible as
HISTORY context. Split: per-user chronological 70/30 (immutable).

This world is the portfolio's PERSISTENCE test: engagement is a latent state carried ACROSS the user's
real event sequence (bursts/refractoriness), not an i.i.d. per-event draw. The momentum statistic is
measured on train first — if the data shows no burstiness, persistence is structurally unexercised and the
run says so rather than manufacturing an arm.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

PASSIVE = ("Video Browsing", "Live Streaming", "Search Behavior")
POS = ("conversion", "like", "follow", "comment", "share", "collect", "click", "purchase", "order")


def acted(e) -> bool:
    return any(str(a.get("type", "")).lower() in POS for a in (e.get("action") or [])
               if isinstance(a, dict))


def user_events(udata) -> list:
    return sorted(udata.get("action_history") or [], key=lambda e: str(e.get("timestamp", "")))


def split_user(events, frac=0.7):
    cut = int(len(events) * frac)
    return events[:cut], events[cut:]


def fit_stats(train_by_user: dict) -> dict:
    """Train-only fitted statistics: per-user passive rate, pooled per-type rate, and the POOLED momentum
    lift — P(act | ≥1 act in the previous 3 passive events) / P(act | none). The burstiness measurement
    that decides whether persistence is structurally exercised at all."""
    by_type, by_user = defaultdict(lambda: [0, 0]), {}
    mom = {True: [0, 0], False: [0, 0]}
    for uid, evs in train_by_user.items():
        pas = [e for e in evs if e.get("type") in PASSIVE]
        k = sum(1 for e in pas if acted(e))
        by_user[uid] = (len(pas), k)
        recent = []
        for e in pas:
            key = any(recent[-3:])
            mom[key][0] += 1
            mom[key][1] += acted(e)
            recent.append(acted(e))
        for e in evs:
            t = e.get("type", "?")
            by_type[t][0] += 1
            by_type[t][1] += acted(e)
    p_hot = (mom[True][1] + 0.5) / (mom[True][0] + 1.0)
    p_cold = (mom[False][1] + 0.5) / (mom[False][0] + 1.0)
    g = sum(k for _, k in by_user.values()) / max(1, sum(n for n, _ in by_user.values()))
    return {"user_rate": {u: (k + g * 4) / (n + 4) for u, (n, k) in by_user.items()},   # shrunk to global
            "type_rate": {t: (k + 0.5) / (n + 1.0) for t, (n, k) in by_type.items()},
            "global_rate": g,
            "momentum_lift": p_hot / max(1e-4, p_cold), "p_hot": p_hot, "p_cold": p_cold,
            "momentum_n": {"hot": mom[True][0], "cold": mom[False][0]},
            "status": {"user_rate": "fitted", "type_rate": "fitted", "momentum": "fitted",
                       "engagement_latent": "prior_backed"}}


def momentum_state(prior_passive_events, k=3) -> float:
    """Observable engagement state at a target: fraction of the last k passive exposures acted on."""
    last = [acted(e) for e in prior_passive_events[-k:]]
    return sum(last) / max(1, len(last))


def item_features(uid, target, prior_passive, stats, interp=None) -> list:
    """The typed feature vector for the universal fitted policy: metadata + persistence state
    (+ interpretation dims appended by the caller when semantics are on)."""
    ur = stats["user_rate"].get(uid, stats["global_rate"])
    tr = stats["type_rate"].get(target.get("type", "?"), stats["global_rate"])
    m = momentum_state(prior_passive)
    hour = 0
    ts = str(target.get("timestamp", ""))
    if len(ts) >= 13:
        try:
            hour = int(ts[11:13])
        except ValueError:
            hour = 0
    return [_logit(ur), _logit(tr), m, 1.0 if 8 <= hour <= 22 else 0.0]


def _logit(p):
    p = min(1 - 1e-4, max(1e-4, p))
    return math.log(p / (1 - p))


def v2_engagement_predict(x, base, pol, *, latent=True, n_particles=32, seed=0):
    """The V2 particle readout over the calibrated per-event probability: each particle draws a latent
    responsiveness (logit-space, mean-preserving; broad prior) around the FITTED policy's p. Feature
    assembly (metadata + momentum persistence state + interpretation dims) is the caller's contract with
    the specific fitted policy — a policy is only ever scored on the feature layout it was fitted on."""
    p0 = pol.p_engage(x, base) if pol is not None else base
    if not latent:
        return {"p": p0, "particles": [], "n_particles": 0}
    rng = random.Random(seed)
    ps = []
    for _ in range(n_particles):
        z = _logit(p0) + rng.gauss(0.0, 0.35)                # latent responsiveness (broad prior, logit)
        ps.append(1.0 / (1.0 + math.exp(-max(-30, min(30, z)))))
    return {"p": sum(ps) / len(ps),
            "particles": [round(v, 4) for v in ps[:3]], "n_particles": n_particles}
