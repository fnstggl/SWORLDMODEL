"""Reference World D — information diffusion on a real follower graph (SNAP higgs-twitter).

The portfolio's NETWORK + TEMPORAL-ROLLOUT test. Data: 456k-node follower graph + timestamped activity
(RT/MT/RE) around the 2012 Higgs announcement. NO message content exists in this dataset → LLM/semantic
arms are STRUCTURALLY ABSENT (logged per the omission rule; nothing to interpret).

Task (leak-free, time-forward): at a pre-announcement cutoff, among users who are NOT yet active but
follow ≥1 active user, predict who activates in the next 24h. Features are strictly as-of the cutoff.
The fit uses an EARLIER cutoff's transitions (train window precedes the eval window; both windows end
before the announcement regime break).

Structural mechanisms under test:
  * NETWORK: does per-user exposure (active followees) beat a population hazard?
  * TEMPORAL ROLLOUT: does simulating within-window contagion (newly-activated sample users adding
    exposure to their sampled followers, event-driven) beat the static closed form 1−exp(−q·k)?
  * LATENT: per-particle transmission-rate uncertainty vs a point estimate.
"""
from __future__ import annotations

import gzip
import math
import random
from collections import defaultdict

ACTIVITY = "data/higgs/higgs-activity_time.txt.gz"
SOCIAL = "data/higgs/higgs-social_network.edgelist.gz"


def load_activation_times():
    """user -> first ts they ACT (source side of any RT/MT/RE event)."""
    first = {}
    t_min = None
    with gzip.open(ACTIVITY, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            u, ts = int(p[0]), int(p[2])
            if u not in first or ts < first[u]:
                first[u] = ts
            t_min = ts if t_min is None else min(t_min, ts)
    return first, t_min


def exposure_snapshot(first, cutoffs):
    """ONE streaming pass over the 14M-edge follower graph (edge a b = a follows b). For every cutoff ts:
    active_followees[a], latest_followee_activation[a]; plus out_degree[a] (cutoff-free)."""
    active_by = {c: {u for u, t in first.items() if t <= c} for c in cutoffs}
    n_act = {c: defaultdict(int) for c in cutoffs}
    last_act = {c: {} for c in cutoffs}
    out_deg = defaultdict(int)
    with gzip.open(SOCIAL, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            a, b = int(p[0]), int(p[1])
            out_deg[a] += 1
            for c in cutoffs:
                t_b = first.get(b)
                if t_b is not None and t_b <= c:
                    n_act[c][a] += 1
                    if t_b > last_act[c].get(a, 0):
                        last_act[c][a] = t_b
    return {"active_by": active_by, "n_act": n_act, "last_act": last_act, "out_deg": out_deg}


def build_cohort(first, snap, cutoff, window_s, *, n_sample, seed):
    """At-risk cohort at `cutoff`: exposed (≥1 active followee), not yet active. Label: activates within
    (cutoff, cutoff+window]. Deterministic sample."""
    at_risk = [a for a, k in snap["n_act"][cutoff].items()
               if k >= 1 and a not in snap["active_by"][cutoff]]
    rng = random.Random(seed)
    rng.shuffle(at_risk)
    rows = []
    for a in at_risk[:n_sample]:
        t = first.get(a)
        rows.append({"u": a, "k": snap["n_act"][cutoff][a], "deg": snap["out_deg"].get(a, 0),
                     "recency_h": (cutoff - snap["last_act"][cutoff].get(a, cutoff)) / 3600.0,
                     "y": int(t is not None and cutoff < t <= cutoff + window_s)})
    return rows


def fit_logistic(rows, feats_fn, *, iters=400, lr=0.3, l2=1e-3):
    """The strongest simple statistical baseline: logistic on as-of features, fitted on the TRAIN cohort."""
    X = [feats_fn(r) for r in rows]
    Y = [r["y"] for r in rows]
    k = len(X[0])
    w, b = [0.0] * k, math.log(max(1e-4, sum(Y) / max(1, len(Y))) / (1 - sum(Y) / max(1, len(Y)) + 1e-9))
    n = len(X)
    for _ in range(iters):
        gw, gb = [0.0] * k, 0.0
        for x, y in zip(X, Y):
            z = sum(wi * xi for wi, xi in zip(w, x)) + b
            q = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = q - y
            for i in range(k):
                gw[i] += e * x[i]
            gb += e
        w = [wi - lr * (gi / n + l2 * wi) for wi, gi in zip(w, gw)]
        b -= lr * gb / n
    def predict(r):
        z = sum(wi * xi for wi, xi in zip(w, feats_fn(r))) + b
        return 1 / (1 + math.exp(-max(-30, min(30, z))))
    return predict, {"w": [round(x, 4) for x in w], "b": round(b, 4)}


def feats_full(r):
    return [math.log1p(r["k"]), math.log1p(r["deg"]),
            r["k"] / max(1.0, r["deg"]), math.exp(-r["recency_h"] / 24.0)]


def fit_q(train_rows, window_s):
    """Per-exposure transmission hazard q (per active followee per day), moment-matched on the TRAIN
    cohort: solve mean(1−exp(−q·k·W)) = train activation rate by bisection. status: fitted."""
    rate = sum(r["y"] for r in train_rows) / max(1, len(train_rows))
    W = window_s / 86400.0
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        pred = sum(1 - math.exp(-mid * r["k"] * W) for r in train_rows) / len(train_rows)
        if pred < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def exposure_snapshot_aged(first, cutoffs, taus_h=(6.0, 12.0, 24.0, 48.0)):
    """exposure_snapshot + age-weighted exposure per candidate half-life τ (information-aging mechanism):
    k_tau[c][τ][a] = Σ_{b active followee of a} exp(−(c − t_b)/τ). Same single streaming pass."""
    active_by = {c: {u for u, t in first.items() if t <= c} for c in cutoffs}
    n_act = {c: defaultdict(int) for c in cutoffs}
    last_act = {c: {} for c in cutoffs}
    k_tau = {c: {tau: defaultdict(float) for tau in taus_h} for c in cutoffs}
    out_deg = defaultdict(int)
    with gzip.open(SOCIAL, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            a, b = int(p[0]), int(p[1])
            out_deg[a] += 1
            t_b = first.get(b)
            if t_b is None:
                continue
            for c in cutoffs:
                if t_b <= c:
                    n_act[c][a] += 1
                    if t_b > last_act[c].get(a, 0):
                        last_act[c][a] = t_b
                    age_h = (c - t_b) / 3600.0
                    for tau in taus_h:
                        k_tau[c][tau][a] += math.exp(-age_h / tau)
    return {"active_by": active_by, "n_act": n_act, "last_act": last_act, "out_deg": out_deg,
            "k_tau": k_tau, "taus_h": taus_h}


def build_cohort_aged(first, snap, cutoff, window_s, *, n_sample, seed):
    """build_cohort + per-τ age-weighted exposure (k_tau) and k_eff0 left unset until a τ is chosen."""
    rows = build_cohort(first, snap, cutoff, window_s, n_sample=n_sample, seed=seed)
    for r in rows:
        r["k_tau"] = {tau: snap["k_tau"][cutoff][tau].get(r["u"], 0.0) for tau in snap["taus_h"]}
    return rows


def load_activity_stream():
    """All activity timestamps (any type) — the Hawkes self-excitation validation target."""
    ts = []
    with gzip.open(ACTIVITY, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                ts.append(int(p[2]))
    ts.sort()
    return ts


def sample_subgraph_edges(sample_ids):
    """Edges among the SAMPLED cohort (b→followers within sample) for within-window contagion rollout.
    One more streaming pass; only sample×sample edges are kept (the rollout's honest scope)."""
    ids = set(sample_ids)
    followers_of = defaultdict(list)                        # b -> [a in sample who follow b]
    with gzip.open(SOCIAL, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            a, b = int(p[0]), int(p[1])
            if a in ids and b in ids:
                followers_of[b].append(a)
    return followers_of


def v2_contagion_predict(rows, q_fit, window_s, followers_of, *, n_particles=30, seed=0,
                         network=True, rollout=True, latent=True, base_rate=None):
    """The V2 diffusion world: per particle, draw a transmission rate q around the fitted value (latent
    parameter uncertainty, lognormal sd 0.4 — broad labeled prior), then run the window as an event-driven
    contagion over the sampled cohort: each user's activation hazard is q·k_t where k_t GROWS when their
    sampled in-sample followees activate mid-window (rollout=True). Discretized in 12 steps (2h each for a
    24h window) — an explicit, auditable integration of the same hazard the closed form uses.
    network=False → population hazard (base rate) for everyone. rollout=False → k frozen at t0.
    Returns per-user mean activation probability across particles."""
    W = window_s / 86400.0
    n_steps = 12
    dt = W / n_steps
    idx = {r["u"]: i for i, r in enumerate(rows)}
    p_acc = [0.0] * len(rows)
    rng = random.Random(seed)
    for pi in range(n_particles):
        q = q_fit * math.exp(rng.gauss(0.0, 0.4)) if latent else q_fit
        if not network:
            p0 = base_rate if base_rate is not None else 0.01
            for i in range(len(rows)):
                p_acc[i] += p0
            continue
        k = [float(r["k"]) for r in rows]
        active = [False] * len(rows)
        log_surv = [0.0] * len(rows)
        for _ in range(n_steps):
            newly = []
            for i, r in enumerate(rows):
                if active[i]:
                    continue
                haz = q * k[i] * dt
                log_surv[i] -= haz                           # Rao-Blackwellized readout (no 1/N floor)
                if rng.random() < 1.0 - math.exp(-haz):      # sampled activation drives the CONTAGION
                    active[i] = True
                    newly.append(r["u"])
            if rollout:
                for u in newly:                              # new activations add exposure in-sample
                    for a in followers_of.get(u, []):
                        j = idx.get(a)
                        if j is not None and not active[j]:
                            k[j] += 1.0
        for i in range(len(rows)):
            p_acc[i] += 1.0 - math.exp(log_surv[i])
    return [min(0.97, max(1e-4, p / n_particles)) for p in p_acc]
