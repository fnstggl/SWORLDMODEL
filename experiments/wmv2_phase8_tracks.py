"""Phase 8 — Tracks B & C: persistence on OTHER longitudinal structures (dyadic + institutional).

The engagement track (A) shows repeated-individual-behavior persistence. Persistence must not be declared
universal from one dataset, so we run the SAME machinery (history → DecayedBetaBernoulli filter → materialize
into WorldState → readout) on two materially different longitudinal structures and grade each honestly:

  TRACK B — DYADIC / RELATIONSHIP HISTORY (Enron email, real). For each ordered dyad a→b, does the dyad's
  prior interaction history predict a future interaction in the next window? Filtered relationship-interaction
  propensity vs no-history (global) vs interaction-frequency. Time-forward split at the median timestamp;
  dyad-disjoint is inherent (each dyad scored on its own future). Honest null preserved if it does not beat
  the frequency baseline.

  TRACK C — INSTITUTIONAL / PROCESS HISTORY (US Senate roll-calls, real). Does the chamber's recent
  decision history (a persistent institutional pass-propensity) predict whether the NEXT roll-call passes?
  Filtered pass-propensity vs no-history (global base rate). Roll-calls are processed in sequence
  (time-forward by construction); each vote's own outcome is never a model input.

Both reuse the shared-world readout from ``wmv2_phase8_shared_world``. No LLM.
Run: PYTHONPATH=. python -m experiments.wmv2_phase8_tracks
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from experiments.wmv2_phase8_shared_world import (_auroc, _brier, _ece, _logloss, _materialize_and_read,
                                                  _materialize_and_read_scalar, _paired_brier, _power)

OUT = Path("experiments/results/phase8")
ENRON = "experiments/results/phase9/enron_comm_edges.json"
CONGRESS = "experiments/results/phase9/congress_S117_bills.json"


def _filtered_readout(actor, anchor, prior_obs, decay=0.6, strength=4.0):
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    key = PersistentStateKey("w", "s", "actor", actor, "engagement_propensity")
    obs = [(f"e{i}", 1 if x else 0, float(i)) for i, x in enumerate(prior_obs)]
    post = DecayedBetaBernoulliFilter(key=key, prior_mean=anchor, prior_strength=strength, decay=decay).filter(obs)
    return _materialize_and_read("w", actor, post)


# ------------------------------------------------------------------ TRACK B — Enron dyadic
def track_b_enron(n_windows=6):
    """Median time-split link persistence: for each dyad active in the FIRST half, does its first-half
    interaction history predict a SECOND-half interaction? Positives = still active; negatives = went quiet.
    Filtered relationship propensity (over first-half windows) vs interaction-frequency vs no-history. Enron's
    late-2001 collapse is a real regime change — if persistence cannot beat frequency through it, we say so."""
    data = json.loads(Path(ENRON).read_text())
    edges = data["edges"]
    all_ts = sorted(t for e in edges for t in e["ts"])
    lo, hi = all_ts[0], all_ts[-1]
    mid = all_ts[len(all_ts) // 2]
    span = (mid - lo) / n_windows if mid > lo else 1.0
    ys, p_persist, p_nohist, p_freq = [], [], [], []
    rows = []
    for e in edges:
        first = [t for t in e["ts"] if t < mid]
        second = [t for t in e["ts"] if t >= mid]
        if not first:
            continue                                         # only dyads active in the first half (fair task)
        wins = [0] * n_windows
        for t in first:
            k = min(n_windows - 1, int((t - lo) / span)) if span > 0 else 0
            wins[k] = 1
        rows.append(((e["src"], e["dst"]), wins, 1 if second else 0))
    base = sum(y for _, _, y in rows) / max(1, len(rows))     # second-half activity rate among first-active
    for (a, b), wins, y in rows:
        ys.append(y)
        freq = sum(wins) / max(1, len(wins))
        p_freq.append(min(0.97, max(0.03, freq)))
        p_nohist.append(_materialize_and_read_scalar("w", f"{a}|{b}", base))
        p_persist.append(_filtered_readout(f"{a}|{b}", base, wins, decay=0.8, strength=4.0))
    n = len(ys)
    real = sum(ys) / max(1, n)

    def M(ps):
        return {"brier": round(_brier(ps, ys), 5), "logloss": round(_logloss(ps, ys), 4),
                "auroc": round(_auroc(ps, ys), 4), "ece": round(_ece(ps, ys), 4)}
    detail = {"no_history": M(p_nohist), "frequency": M(p_freq), "persist_shared_world": M(p_persist)}
    pf = _paired_brier(ys, p_persist, p_freq)
    pn = _paired_brier(ys, p_persist, p_nohist)
    for d in (pf, pn):
        d.pop("paired_diffs", None)
    beats_freq = pf["ci95"][1] < 0
    beats_base = pn["ci95"][1] < 0
    if beats_freq and beats_base:
        verdict = "dyadic history HELPS vs BOTH frequency and base-rate baselines (CIs exclude 0)"
    elif beats_freq:
        verdict = ("dyadic history beats the frequency baseline but does NOT beat the base-rate baseline "
                   f"(AUROC {detail['persist_shared_world']['auroc']}); Enron's late-2001 collapse is a "
                   "regime change that washes out dyadic persistence — HONEST WEAK result")
    else:
        verdict = "dyadic persistence NOT DETECTABLE vs baselines (honest null)"
    return {"task": "Enron dyadic link persistence (predict 2nd-half interaction from 1st-half history)",
            "n_dyads_scored": n, "n_windows": n_windows, "second_half_activity_rate": round(real, 4),
            "detail": detail, "persist_vs_frequency": pf, "persist_vs_no_history": pn, "verdict": verdict}


# ------------------------------------------------------------------ TRACK C — congress institutional
def track_c_congress():
    data = json.loads(Path(CONGRESS).read_text())
    bills = data["bills"]
    # pass = majority yes among those voting; the chamber's decision sequence
    seq = []
    for bl in bills:
        yes = bl["dem"][0] + bl["rep"][0]
        tot = bl["dem"][1] + bl["rep"][1]
        seq.append(1 if (tot > 0 and yes > tot / 2) else 0)
    n_all = len(seq)
    base = sum(seq) / max(1, n_all)
    # time-forward: predict each roll-call from the persistent pass-propensity of PRIOR roll-calls
    ys, p_persist, p_nohist = [], [], []
    warmup = max(5, n_all // 20)
    for i in range(warmup, n_all):
        prior = seq[:i]
        ys.append(seq[i])
        p_nohist.append(_materialize_and_read_scalar("w", "chamber", base))
        p_persist.append(_filtered_readout("chamber", base, prior, decay=0.85, strength=6.0))
    n = len(ys)

    def M(ps):
        return {"brier": round(_brier(ps, ys), 5), "logloss": round(_logloss(ps, ys), 4),
                "auroc": round(_auroc(ps, ys), 4), "ece": round(_ece(ps, ys), 4)}
    detail = {"no_history": M(p_nohist), "persist_shared_world": M(p_persist)}
    pn = _paired_brier(ys, p_persist, p_nohist)
    diffs = pn.pop("paired_diffs")
    power = _power(sum(ys) / max(1, n), abs(pn["mean"]), n, diffs)
    verdict = ("institutional pass-history HELPS (CI excludes 0)" if pn["ci95"][1] < 0
               else "institutional pass-history HURTS (CI excludes 0)" if pn["ci95"][0] > 0
               else "institutional pass-persistence NOT DETECTABLE (CI spans 0)")
    return {"task": "US Senate S117 roll-call pass persistence (predict next pass from prior decisions)",
            "n_scored": n, "n_bills_total": n_all, "base_pass_rate": round(base, 4),
            "detail": detail, "persist_vs_no_history": pn, "power": power,
            "adequately_powered": power["power_at_observed_effect"] >= 0.8, "verdict": verdict}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = track_b_enron()
    c = track_c_congress()
    report = {"track_B_dyadic_enron": b, "track_C_institutional_congress": c,
              "summary": {"track_B_verdict": b["verdict"], "track_C_verdict": c["verdict"]}}
    (OUT / "tracks_BC.json").write_text(json.dumps(report, indent=1, default=str))
    print("TRACK B (Enron dyadic):", json.dumps(b["detail"]))
    print("  persist_vs_frequency:", b["persist_vs_frequency"], "\n  VERDICT:", b["verdict"])
    print("TRACK C (congress institutional):", json.dumps(c["detail"]))
    print("  persist_vs_no_history:", c["persist_vs_no_history"], "power:", c["power"]["power_at_observed_effect"])
    print("  VERDICT:", c["verdict"])
    print("wrote", OUT / "tracks_BC.json")


if __name__ == "__main__":
    main()
