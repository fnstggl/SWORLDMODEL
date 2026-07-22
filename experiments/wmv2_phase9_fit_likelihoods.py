"""Phase 9 — FIT observation likelihoods on real data vs the fixed tables (final hardening, Part 3).

The Phase-9 edge observation rates (EDGE_OBS_MODELS detect/false) are hand-set broad assumptions. Here we FIT
the per-opportunity detection rate for the `repeated_interaction` edge class on real Enron communication with a
NODE-DISJOINT train/test split, and compare the fitted likelihood against the fixed table on held-out
future-edge prediction. Fitting uses train nodes only; the test nodes are never seen during fitting.

Honest by design: we report whether the fitted model beats the fixed table on held-out log-loss / Brier / ECE.
If it does not, we say so — the fixed broad table may already be adequate, and we do not tune on the test set.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from swm.world_model_v2.phase3_observation import _edge_rates

OUT = Path("experiments/results/phase9")
ENRON = OUT / "enron_comm_edges.json"


def _logit(p):
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + math.exp(-x)) if x > -700 else 0.0


def _exposure_post(prior_p, N, k, detect, false):
    """Explicit Binomial exposure posterior with given per-opportunity (detect, false)."""
    if N == 0:
        return prior_p
    detect = min(1 - 1e-6, max(1e-6, detect))
    false = min(1 - 1e-6, max(1e-6, false))
    lo = _logit(prior_p) + (k * math.log(detect) + (N - k) * math.log(1 - detect)
                            - (k * math.log(false) + (N - k) * math.log(1 - false)))
    return _sig(lo)


def _split_counts(data, node_set):
    """Train-period per-dyad counts + per-sender outbound + future-edge set, restricted to node_set."""
    edges = data["edges"]
    all_ts = sorted(t for e in edges for t in e["ts"])
    cutoff = all_ts[len(all_ts) // 2]
    cnt, out, fut = defaultdict(int), defaultdict(int), set()
    for e in edges:
        a, b = e["src"], e["dst"]
        if a not in node_set or b not in node_set:
            continue
        for t in e["ts"]:
            if t < cutoff:
                cnt[(a, b)] += 1
                out[a] += 1
            else:
                fut.add((a, b))
    return cnt, out, fut


def _fit_rates(nodes, cnt, out, fut):
    """Fitted per-opportunity detect/false: mean(train a→b fraction of a's outbound) over future-edge dyads
    (detect) vs non-future-edge dyads (false). Real, node-restricted, no test leakage."""
    det_num = det_den = fal_num = fal_den = 0.0
    for a in nodes:
        if out[a] < 3:
            continue
        for b in nodes:
            if a == b:
                continue
            frac = cnt[(a, b)] / out[a]
            if (a, b) in fut:
                det_num += frac
                det_den += 1
            else:
                fal_num += frac
                fal_den += 1
    detect = det_num / det_den if det_den else 0.1
    false = fal_num / fal_den if fal_den else 0.01
    return max(1e-4, detect), max(1e-5, false)


def _eval(nodes, cnt, out, fut, detect, false):
    scored = []
    for a in nodes:
        if out[a] < 3:
            continue
        for b in nodes:
            if a == b:
                continue
            N = min(60, out[a])
            k = min(cnt[(a, b)], N)
            p = _exposure_post(0.05, N, k, detect, false)
            scored.append((p, 1 if (a, b) in fut else 0))
    n = max(1, len(scored))
    brier = sum((p - y) ** 2 for p, y in scored) / n
    ll = sum(-(y * math.log(max(1e-6, p)) + (1 - y) * math.log(max(1e-6, 1 - p))) for p, y in scored) / n
    b = defaultdict(lambda: [0, 0])
    for p, y in scored:
        kk = min(9, int(p * 10))
        b[kk][0] += y
        b[kk][1] += 1
    ece = sum(abs(c[0] / c[1] - (kk + 0.5) / 10) * c[1] for kk, c in b.items() if c[1]) / n
    return {"brier": round(brier, 4), "log_loss": round(ll, 4), "ece": round(ece, 4), "n": len(scored)}


def main(seed=0):
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(ENRON.read_text())
    nodes = list(data["nodes"])
    rng = random.Random(seed)
    rng.shuffle(nodes)
    half = len(nodes) // 2
    fit_nodes, test_nodes = set(nodes[:half]), set(nodes[half:])   # NODE-DISJOINT
    # fit on fit_nodes
    cnt_f, out_f, fut_f = _split_counts(data, fit_nodes)
    detect_fit, false_fit = _fit_rates(fit_nodes, cnt_f, out_f, fut_f)
    # the FIXED table (repeated_interaction, strong, rel 0.9) effective rates
    detect_fixed, false_fixed = _edge_rates("repeated_interaction", "strong", 0.9)
    # evaluate BOTH on the held-out test_nodes
    cnt_t, out_t, fut_t = _split_counts(data, test_nodes)
    fitted = _eval(test_nodes, cnt_t, out_t, fut_t, detect_fit, false_fit)
    fixed = _eval(test_nodes, cnt_t, out_t, fut_t, detect_fixed, false_fixed)
    report = {
        "dataset": "Enron email (node-disjoint fit/test)", "evidence_class": "repeated_interaction",
        "fitted_rates": {"detect": round(detect_fit, 4), "false": round(false_fit, 4)},
        "fixed_table_rates": {"detect": round(detect_fixed, 4), "false": round(false_fixed, 4)},
        "held_out_fixed": fixed, "held_out_fitted": fitted,
        "fitted_beats_fixed": {
            "log_loss": fitted["log_loss"] < fixed["log_loss"],
            "brier": fitted["brier"] < fixed["brier"], "ece": fitted["ece"] < fixed["ece"]},
        "verdict": ("fitted likelihood beats the fixed table on held-out test"
                    if fitted["log_loss"] < fixed["log_loss"] else
                    "fitted likelihood does NOT beat the fixed table on held-out log-loss (honest null)")}
    (OUT / "fitted_likelihoods.json").write_text(json.dumps(report, indent=2))
    print("FIT vs FIXED observation likelihoods (Enron, node-disjoint):")
    print("  fitted rates:", report["fitted_rates"], "| fixed:", report["fixed_table_rates"])
    print("  held-out FIXED :", fixed)
    print("  held-out FITTED:", fitted)
    print("  fitted beats fixed:", report["fitted_beats_fixed"])
    print("  verdict:", report["verdict"])


if __name__ == "__main__":
    main()
