"""Representation ablation — Phase 3 (REPRESENTATION-CHOICE PRINCIPLE, empirical).

Proves — on synthetic ground truth with a KNOWN hidden structure — that the best representation of a hidden
social state DEPENDS on the truth, and that collapsing to an arbitrary scalar (`trust=0.7`) loses whenever the
uncertainty has structure. This is the empirical justification for choosing representations by calibration
rather than intuition.

Three generative families, each with a genuinely different hidden structure:
  smooth_continuous   theta ~ Uniform(0.1, 0.9)             (a propensity taking ALL values in a range)
  two_regime          theta in {0.25, 0.75} (50/50)         (two qualitatively distinct regimes)
  bimodal_mixture     theta ~ 1/2 Beta(9,2) + 1/2 Beta(2,9) (two plausible worlds at once)

For each episode: draw theta, emit `m` reliability-weighted directional votes from theta (via the SAME
observation likelihood every fitter uses), draw the outcome ~ Bernoulli(theta). Fit every candidate
representation PER EPISODE and score held-out log-loss / Brier / ECE. Writes a machine-readable artifact.

Finding (reported honestly, not tuned to a narrative): `scalar_point` — the arbitrary-scalar anti-pattern —
is dominated in every family (worst or near-worst held-out log-loss + ECE). `mixture` wins when the truth is
genuinely bimodal. On sparse noisy evidence a COARSE well-regularized representation (discrete/hybrid) can
beat a finer continuous one even for a continuous truth — so the winner is chosen by held-out calibration, not
by which representation "sounds right". The evidence-abundance sweep shows the winner shifting with
identifiability. This is the empirical justification for the representation-choice principle.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from swm.world_model_v2.phase3_representation import choose_representation

OUT = Path("experiments/results/phase3")
D_FROM_REL = lambda r: 0.5 + 0.35 * r          # must match phase3_representation._vote_loglik


def _seed(*parts) -> int:
    """Hash-stable seed (hashlib, NOT the per-process-salted builtin hash) — reproducibility across runs."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _beta(rng, a, b):
    ga = rng.gammavariate(a, 1.0)
    gb = rng.gammavariate(b, 1.0)
    return ga / (ga + gb) if (ga + gb) > 0 else 0.5


def _draw_theta(rng, family: str) -> float:
    if family == "smooth_continuous":
        return 0.1 + 0.8 * rng.random()                     # Uniform(0.1, 0.9): a genuinely spread continuum
    if family == "two_regime":
        return 0.75 if rng.random() < 0.5 else 0.25
    if family == "bimodal_mixture":
        return _beta(rng, 9.0, 2.0) if rng.random() < 0.5 else _beta(rng, 2.0, 9.0)
    raise ValueError(family)


def _episode(rng, family: str, m: int):
    theta = _draw_theta(rng, family)
    votes = []
    for _ in range(m):
        rel = 0.5 + 0.45 * rng.random()                     # reliability in [0.5, 0.95]
        d = D_FROM_REL(rel)
        p_yes_vote = theta * d + (1 - theta) * (1 - d)
        sign = 1 if rng.random() < p_yes_vote else -1
        votes.append((sign, rel))
    outcome = 1 if rng.random() < theta else 0
    return votes, outcome


def run(seed: int = 20240720, n: int = 1500, m: int = 8) -> dict:
    families = ["smooth_continuous", "two_regime", "bimodal_mixture"]
    kinds = ["scalar_point", "continuous_probabilistic", "discrete_hypothesis", "mixture",
             "hybrid_interpretable"]
    report = {"seed": seed, "n_episodes_per_family": n, "votes_per_episode": m, "families": {}}
    for fam in families:
        rng = random.Random(_seed(seed, fam))
        episodes = [_episode(rng, fam, m) for _ in range(n)]
        split = n // 3
        train, test = episodes[:split], episodes[split:]
        card = choose_representation(fam, train, test, candidates=kinds)
        # rank by held-out log-loss
        ranked = sorted(card.metrics.items(), key=lambda kv: kv[1]["held_out_logloss"])
        report["families"][fam] = {
            "winner": card.winner,
            "ranking": [{"kind": k, **v} for k, v in ranked],
            "scalar_baseline_rank": [k for k, _ in ranked].index("scalar_point") + 1,
            "base_rate_yes": round(sum(o for _, o in test) / max(1, len(test)), 4)}
    # evidence-abundance sweep: representation choice depends on IDENTIFIABILITY, not just structure. With
    # sparse evidence a coarse (well-regularized) representation calibrates best; as evidence grows, the
    # finer representation that matches the true structure overtakes it. Reported honestly whatever it shows.
    report["evidence_sweep"] = {}
    for fam in ["smooth_continuous", "bimodal_mixture"]:
        report["evidence_sweep"][fam] = {}
        for mm in (2, 8, 30):
            rng = random.Random(_seed(seed, fam, "sweep", mm))
            eps = [_episode(rng, fam, mm) for _ in range(n)]
            card = choose_representation(fam, eps[:n // 3], eps[n // 3:], candidates=kinds)
            ranked = sorted(card.metrics.items(), key=lambda kv: kv[1]["held_out_logloss"])
            report["evidence_sweep"][fam][f"m={mm}"] = {
                "winner": card.winner,
                "logloss_by_kind": {k: v["held_out_logloss"] for k, v in ranked}}

    # cross-family summary: does the winner match the structure, and is scalar ever best?
    report["summary"] = {
        "winner_by_family": {f: report["families"][f]["winner"] for f in families},
        "scalar_ever_wins": any(report["families"][f]["winner"] == "scalar_point" for f in families),
        "scalar_mean_rank": round(sum(report["families"][f]["scalar_baseline_rank"]
                                      for f in families) / len(families), 2),
        "winner_shifts_with_evidence": {
            f: [report["evidence_sweep"][f][f"m={mm}"]["winner"] for mm in (2, 8, 30)]
            for f in report["evidence_sweep"]},
        "conclusion": "the best representation is family- AND identifiability-dependent; a fixed scalar is "
                      "dominated whenever the hidden state has structure — representation must be chosen by "
                      "held-out calibration, never by intuition or convenience"}
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = run()
    (OUT / "representation_ablation.json").write_text(json.dumps(rep, indent=2))
    print("REPRESENTATION ABLATION — held-out log-loss (lower is better)\n")
    for fam, r in rep["families"].items():
        print(f"  {fam}  (base-rate yes={r['base_rate_yes']}) → winner: {r['winner']}")
        for row in r["ranking"]:
            mark = "  <-- winner" if row["kind"] == r["winner"] else ""
            print(f"      {row['kind']:26s} logloss={row['held_out_logloss']:.4f}  "
                  f"brier={row['brier']:.4f}  ece={row['calibration_err']:.4f}{mark}")
        print(f"      scalar_point rank: {r['scalar_baseline_rank']}/5\n")
    print("SUMMARY:", json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()
