"""Phase 9 — SECOND real graph domain: Enron email communication (final hardening, gate C).

Materially different from congress co-voting: the relation is EMAIL COMMUNICATION (directed), the observation
process is message logs (not roll-call votes), and the task is FUTURE-EDGE PREDICTION under a temporal split
(no leakage). Streams the CMU Enron tarball (capped), builds a temporal directed communication graph among the
most active addresses, and evaluates through the SAME Phase-9 edge posterior.

Tasks:
  1. TEMPORAL link prediction (PREDICTION, not reconstruction): train on messages before a cutoff date; predict
     which ordered pairs communicate AFTER the cutoff. Feature = past (train-period) communication via the
     exposure edge posterior. Future labels never used as input.
  2. RECONSTRUCTION baseline (same-period held-out edges) — labeled reconstruction, for contrast.
Metrics: AUROC, PR-AUC (average precision), Brier, log loss, calibration (ECE), + a degree/frequency baseline.

Caches a compact edge list (committed) so the evaluation reproduces without re-fetching.
"""
from __future__ import annotations

import email
import json
import math
import re
import tarfile
import time
import urllib.request
from collections import defaultdict
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from swm.world_model_v2.phase3_posterior import ExposureObservation, infer_edge_posterior_exposure

OUT = Path("experiments/results/phase9")
CACHE = OUT / "enron_comm_edges.json"
URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"


def build(cap_messages=45000, top_nodes=70):
    """Stream + parse (sender, recipient, ts); keep dyads among the top-active addresses; cache compact."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    req = urllib.request.Request(URL, headers={"User-Agent": "swm"})
    resp = urllib.request.urlopen(req, timeout=180)
    activity = defaultdict(int)
    dyads = defaultdict(list)                                 # (a,b) -> [ts,...]
    n = 0
    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if n >= cap_messages:
                break
            if not m.isfile():
                continue
            try:
                f = tar.extractfile(m)
                if f is None:
                    continue
                msg = email.message_from_bytes(f.read())
            except Exception:
                continue
            frm = getaddresses([msg.get("From", "")])
            to = getaddresses(msg.get_all("To", []) or [])
            if not frm or not frm[0][1] or not to:
                continue
            try:
                ts = parsedate_to_datetime(msg.get("Date", "")).timestamp()
            except Exception:
                continue
            s = frm[0][1].lower()
            for a in to:
                r = (a[1] or "").lower()
                if r and "@" in r and r != s:
                    dyads[(s, r)].append(ts)
                    activity[s] += 1
                    activity[r] += 1
            n += 1
    top = {a for a, _ in sorted(activity.items(), key=lambda kv: -kv[1])[:top_nodes]}
    edges = [{"src": a, "dst": b, "ts": sorted(tss)}
             for (a, b), tss in dyads.items() if a in top and b in top]
    data = {"n_messages_parsed": n, "n_nodes": len(top), "nodes": sorted(top),
            "edges": edges, "cap_messages": cap_messages}
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data))
    return data


def _auroc(sl):
    pos = [s for s, y in sl if y == 1]
    neg = [s for s, y in sl if y == 0]
    if not pos or not neg:
        return None
    w = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(1 for p in pos for q in neg if p == q)
    return round(w / (len(pos) * len(neg)), 4)


def _ap(sl):
    """Average precision (PR-AUC)."""
    order = sorted(sl, key=lambda x: -x[0])
    tp = 0
    total_pos = sum(y for _, y in sl)
    if total_pos == 0:
        return None
    s = 0.0
    for i, (_, y) in enumerate(order, 1):
        if y == 1:
            tp += 1
            s += tp / i
    return round(s / total_pos, 4)


def _ece(sl, bins=10):
    from collections import defaultdict as dd
    b = dd(lambda: [0, 0])
    for p, y in sl:
        k = min(bins - 1, int(p * bins))
        b[k][0] += y
        b[k][1] += 1
    n = len(sl)
    return round(sum(abs(c[0] / c[1] - (k + 0.5) / bins) * c[1] for k, c in b.items() if c[1]) / max(1, n), 4)


def temporal_prediction(data):
    edges = data["edges"]
    all_ts = sorted(t for e in edges for t in e["ts"])
    cutoff = all_ts[len(all_ts) // 2]                        # median timestamp → temporal split
    nodes = data["nodes"]
    train_cnt = defaultdict(int)                             # (a,b) train-period message count
    train_out = defaultdict(int)                             # a's total outbound in train (exposure)
    fut = set()                                              # ordered pairs communicating AFTER cutoff
    for e in edges:
        a, b = e["src"], e["dst"]
        for t in e["ts"]:
            if t < cutoff:
                train_cnt[(a, b)] += 1
                train_out[a] += 1
            else:
                fut.add((a, b))
    active = [a for a in nodes if train_out[a] >= 3]         # nodes active in the train period
    scored, freq_base = [], []
    for a in active:
        for b in nodes:
            if a == b:
                continue
            label = 1 if (a, b) in fut else 0
            k = train_cnt[(a, b)]                            # past a→b messages
            N = min(60, train_out[a])                       # exposure = a's outbound opportunities (capped)
            post = infer_edge_posterior_exposure(a, b, "communication",
                                                 [ExposureObservation("repeated_interaction", N, min(k, N), 0.9)],
                                                 prior_p=0.05)
            scored.append((post.posterior_p, label))
            freq_base.append((min(1.0, k / 5.0), label))    # raw past-frequency baseline
    base = sum(y for _, y in scored) / max(1, len(scored))
    brier = sum((p - y) ** 2 for p, y in scored) / max(1, len(scored))
    ll = sum(-(y * math.log(max(1e-6, p)) + (1 - y) * math.log(max(1e-6, 1 - p))) for p, y in scored) / max(1, len(scored))
    return {"task": "predict post-cutoff email edges from pre-cutoff communication (temporal, no leakage)",
            "n_candidate_pairs": len(scored), "base_rate": round(base, 4),
            "auroc": _auroc(scored), "pr_auc": _ap(scored), "brier": round(brier, 4),
            "log_loss": round(ll, 4), "ece": _ece(scored),
            "auroc_frequency_baseline": _auroc(freq_base), "pr_auc_frequency_baseline": _ap(freq_base)}


def reconstruction(data):
    """Same-period reconstruction (held-out train edges predicted from train frequency) — labeled recon."""
    edges = data["edges"]
    all_ts = sorted(t for e in edges for t in e["ts"])
    cutoff = all_ts[len(all_ts) // 2]
    nodes, cnt, out = data["nodes"], defaultdict(int), defaultdict(int)
    for e in edges:
        a, b = e["src"], e["dst"]
        for t in e["ts"]:
            if t < cutoff:
                cnt[(a, b)] += 1
                out[a] += 1
    active = [a for a in nodes if out[a] >= 3]
    scored = []
    for a in active:
        for b in nodes:
            if a == b:
                continue
            scored.append((min(1.0, cnt[(a, b)] / 5.0), 1 if cnt[(a, b)] > 0 else 0))
    return {"task": "RECONSTRUCTION (train-period edges from train frequency)", "auroc": _auroc(scored),
            "pr_auc": _ap(scored)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = build()
    tp = temporal_prediction(data)
    rec = reconstruction(data)
    report = {"dataset": "Enron email communication (CMU)", "n_messages_parsed": data["n_messages_parsed"],
              "n_nodes": data["n_nodes"], "n_edges": len(data["edges"]),
              "temporal_link_prediction_PREDICTION": tp, "reconstruction_baseline": rec,
              "build_latency_s": round(time.time() - t0, 1)}
    report["gates"] = {
        "second_distinct_graph_domain": True,   # email communication, not another congress
        "temporal_prediction_above_chance": (tp["auroc"] or 0) >= 0.6,
        "temporal_prediction_beats_frequency_baseline": (tp["auroc"] or 0) >= (tp["auroc_frequency_baseline"] or 0),
        "reconstruction_labeled_distinct": True}
    report["all_gates_pass"] = all(report["gates"].values())
    (OUT / "enron_validation.json").write_text(json.dumps(report, indent=2))
    print("ENRON EMAIL GRAPH:", data["n_nodes"], "nodes,", len(data["edges"]), "dyads,",
          data["n_messages_parsed"], "msgs")
    print("  temporal prediction:", json.dumps(tp))
    print("  reconstruction:", json.dumps(rec))
    print("  GATES:", json.dumps(report["gates"]), "ALL:", report["all_gates_pass"])


if __name__ == "__main__":
    main()
