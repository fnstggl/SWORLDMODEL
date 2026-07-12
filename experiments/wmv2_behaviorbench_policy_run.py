"""BehaviorBench — Phase 4 re-run: UNIVERSAL learned policy (shared Fehr-Schmidt population + QRE +
CH + conditional cooperation) replacing the per-game hand-crafted structural policies.

What changed vs wmv2_behaviorbench_run.py (same immutable splits, seed 13, 50/50):
  * ONE population-preference mixture (12 FS types) fitted JOINTLY across games (partial pooling)
    drives every two-player game through the same utility machinery — no per-game decision rules;
  * choice noise is quantal response with payoff-scale-normalized precision (transfers across games);
  * cross-game interaction is structural: proposer acceptance beliefs and investor return beliefs
    derive from the SAME fitted preference population (not separately fitted per game);
  * genuine COLD-START arm: published packs only (FS 1999 + CHC 2004 τ=1.5 + FGF 2001), zero local fit;
  * LEAVE-ONE-GAME-OUT transfer: preference mixture + pooled response params fitted WITHOUT the held-out
    game's train data — the regime where no on-distribution statistical model exists at all;
  * distributional calibration (PIT/KS + central coverage) reported per game.

Arms:
  A0 uniform | A1 train histogram | A2 train KDE   (specialist ceilings, recomputed on same splits)
  P_pub  published packs only (cold start, $0, no data)
  P_fit  full hierarchical fit (in-distribution)
  P_logo leave-one-game-out transfer (per game)
  T_pool pooled-other-games histogram (transfer baseline for LOGO)
  ablations of P_fit: no_interaction (uninformative partner beliefs) | selfish (α=β=0) | no_qre (λ→∞)
Context: prior arms (direct LLM A3 0.185, elicitation A4 0.123, hand-crafted V2 A5 0.058) from
experiments/results/wmv2_behaviorbench.json — identical splits, quoted not rerun.

LLM calls: ZERO (the semantic channel added nothing here in the prior round — quoted, not re-bought).
Run: PYTHONPATH=. python -m experiments.wmv2_behaviorbench_policy_run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_behaviorbench_policy.json"
GAMES = ["dictator", "ultimatum_responder", "trust_banker",
         "ultimatum_proposer", "trust_investor", "guessing", "public_goods"]
INTERACTION_GAMES = ("ultimatum_proposer", "trust_investor", "guessing", "public_goods")


def w1(a, b):
    if not a or not b:
        return None
    a, b = sorted(a), sorted(b)
    n = 50
    return sum(abs(a[min(len(a) - 1, int(t / n * len(a)))] -
                   b[min(len(b) - 1, int(t / n * len(b)))]) for t in range(n)) / n


def paired_w1_ci(sample_a, sample_b, test, scale, *, n_boot=500, seed=17):
    rng = random.Random(seed)
    n = len(test)
    ds = []
    for _ in range(n_boot):
        t = [test[rng.randrange(n)] for _ in range(n)]
        ds.append((w1(sample_a, t) - w1(sample_b, t)) / scale)
    ds.sort()
    return {"mean": round(sum(ds) / len(ds), 4),
            "ci95": [round(ds[int(0.025 * len(ds))], 4), round(ds[int(0.975 * len(ds))], 4)]}


def run(n_pred):
    from swm.eval.behavior_eval import load_game
    from swm.world_model_v2.policy import pit_calibration, sample_from_dist
    from swm.world_model_v2.registry.families.choice import (
        GAME_GRID, GamePolicyModel, PAYOFF_SCALE, fit_game_policy, fs_published_pop)
    from swm.world_model_v2.reference.behavior_games import fit_dist

    t0 = time.time()
    train_by, test_by = {}, {}
    for g in GAMES:
        rows = load_game(g)
        vals = [r["human_answer"] for r in rows]
        rng = random.Random(13)                       # the immutable split — identical to prior run
        idx = list(range(len(vals)))
        rng.shuffle(idx)
        cut = len(idx) // 2
        train_by[g] = [vals[i] for i in idx[:cut]]
        test_by[g] = [vals[i] for i in idx[cut:]]
        print(f"{g}: n={len(vals)} train_mean={sum(train_by[g])/cut:.1f} "
              f"test_mean={sum(test_by[g])/len(test_by[g]):.1f}", flush=True)

    rng = random.Random(29)
    out = {"games": {}, "logo": {}, "_splits": {g: {"seed": 13, "n_train": len(train_by[g]),
                                                    "n_test": len(test_by[g])} for g in GAMES}}

    # ---------------- fits ----------------
    print("fitting P_fit (joint hierarchical) …", flush=True)
    m_fit = fit_game_policy(train_by)
    m_pub = GamePolicyModel(fs_published_pop(),
                            lam={g: 0.1 * 100.0 / PAYOFF_SCALE[g] for g in GAME_GRID},
                            sd={g: 0.10 * (GAME_GRID[g][1] - GAME_GRID[g][0]) for g in GAME_GRID})
    # ablations (each refits response params at its own configuration on train — best-config-per-arm)
    from swm.world_model_v2.policy import PopulationPreferences, PreferenceAtom
    m_selfish = fit_game_policy(train_by, fit_weights=False,
                                base_pop=PopulationPreferences(
                                    atoms=[PreferenceAtom({"alpha": 0.0, "beta": 0.0}, 1.0)],
                                    source="ablation: selfish point preferences"))
    m_noqre = fit_game_policy(train_by)
    for g in GAME_GRID:
        m_noqre.lam[g] = 100.0 * 100.0 / PAYOFF_SCALE[g]    # λ→large: near-best-response
    m_noqre.invalidate()
    m_noint = fit_game_policy(train_by)
    _make_uninformative_beliefs(m_noint)

    # LOGO transfer models: one per game, fitted without that game
    m_logo = {}
    for g in GAMES:
        tr = {k: v for k, v in train_by.items() if k != g}
        m = fit_game_policy(tr)
        # pooled response params for the held-out game: payoff-normalized median of fitted games
        lam100 = statistics.median(m.lam[k] * PAYOFF_SCALE[k] / 100.0 for k in tr)
        sdfrac = statistics.median(m.sd[k] / (GAME_GRID[k][1] - GAME_GRID[k][0]) for k in tr)
        m.lam[g] = lam100 * 100.0 / PAYOFF_SCALE[g]
        m.sd[g] = sdfrac * (GAME_GRID[g][1] - GAME_GRID[g][0])
        if g == "guessing":
            m.tau_ch, m.tau_src = 1.5, "published_research (CHC 2004) — held-out fallback"
        if g == "public_goods":
            m.pg_slope, m.pg_src = 0.8, "published_research (FGF 2001) — held-out fallback"
        m.invalidate()
        m_logo[g] = m

    agg = {a: [] for a in ("A0", "A1", "A2", "P_pub", "P_fit", "P_logo", "T_pool",
                           "P_no_interaction", "P_selfish", "P_no_qre")}
    for g in GAMES:
        lo, hi, _ = GAME_GRID[g]
        scale = hi - lo
        test = test_by[g]
        kde = fit_dist(train_by[g], lo, hi)
        arms = {
            "A0": [rng.uniform(lo, hi) for _ in range(n_pred)],
            "A1": list(train_by[g]),
            "A2": [kde.sample(rng) for _ in range(n_pred)],
            "P_pub": sample_from_dist(m_pub.population_dist(g), rng, n_pred),
            "P_fit": sample_from_dist(m_fit.population_dist(g), rng, n_pred),
            "P_logo": sample_from_dist(m_logo[g].population_dist(g), rng, n_pred),
            "P_no_interaction": sample_from_dist(m_noint.population_dist(g), rng, n_pred),
            "P_selfish": sample_from_dist(m_selfish.population_dist(g), rng, n_pred),
            "P_no_qre": sample_from_dist(m_noqre.population_dist(g), rng, n_pred),
        }
        # transfer baseline: pooled other-games histogram rescaled into this game's range
        pool = []
        for k in GAMES:
            if k == g:
                continue
            klo, khi, _ = GAME_GRID[k]
            pool += [lo + (v - klo) / (khi - klo) * scale for v in train_by[k]]
        arms["T_pool"] = pool

        res = {}
        for a, s in arms.items():
            if s:
                res[a] = {"w1_norm": round(w1(s, test) / scale, 4), "n": len(s),
                          "mean": round(sum(s) / len(s), 1)}
        res["_test_mean"] = round(sum(test) / len(test), 1)
        res["_calibration"] = {
            "P_fit": pit_calibration(m_fit.population_dist(g), test),
            "P_logo": pit_calibration(m_logo[g].population_dist(g), test)}
        res["_paired"] = {
            "P_fit_vs_A1": paired_w1_ci(arms["P_fit"], arms["A1"], test, scale),
            "P_fit_vs_A2": paired_w1_ci(arms["P_fit"], arms["A2"], test, scale),
            "P_logo_vs_T_pool": paired_w1_ci(arms["P_logo"], arms["T_pool"], test, scale),
            "P_logo_vs_A1": paired_w1_ci(arms["P_logo"], arms["A1"], test, scale),
            "P_fit_vs_no_interaction": paired_w1_ci(arms["P_fit"], arms["P_no_interaction"], test, scale),
            "P_fit_vs_selfish": paired_w1_ci(arms["P_fit"], arms["P_selfish"], test, scale),
            "P_fit_vs_no_qre": paired_w1_ci(arms["P_fit"], arms["P_no_qre"], test, scale),
            "P_pub_vs_A0": paired_w1_ci(arms["P_pub"], arms["A0"], test, scale)}
        out["games"][g] = res
        for a in agg:
            if res.get(a):
                agg[a].append(res[a]["w1_norm"])
        print(f"\n== {g} ==")
        for a in agg:
            if res.get(a):
                print(f"  {a:18s} W1n={res[a]['w1_norm']:.4f} mean={res[a]['mean']}")
        print(f"  P_fit vs A1: {res['_paired']['P_fit_vs_A1']}  "
              f"P_logo vs T_pool: {res['_paired']['P_logo_vs_T_pool']}", flush=True)

    out["aggregate_w1_norm"] = {a: round(sum(v) / len(v), 4) for a, v in agg.items() if v}
    out["aggregate_interaction_games"] = {
        a: round(sum(out["games"][g][a]["w1_norm"] for g in INTERACTION_GAMES) / len(INTERACTION_GAMES), 4)
        for a in agg if all(out["games"][g].get(a) for g in INTERACTION_GAMES)}
    out["fitted"] = {
        "pop_weights": [{"params": a.params, "w": round(a.weight, 4)} for a in m_fit.pop.atoms],
        "pop_source": m_fit.pop.source,
        "lam_per_game": {g: round(m_fit.lam[g], 4) for g in GAMES},
        "sd_per_game": {g: round(m_fit.sd[g], 2) for g in GAMES},
        "tau_ch": {"value": m_fit.tau_ch, "source": m_fit.tau_src},
        "pg_slope": {"value": m_fit.pg_slope, "source": m_fit.pg_src}}
    out["_context_prior_run"] = {
        "artifact": "experiments/results/wmv2_behaviorbench.json (identical splits)",
        "aggregates_quoted": {"A3_direct_llm": 0.185, "A4_elicitation_ensemble": 0.123,
                              "A5_handcrafted_v2": 0.058, "V2_no_interp": 0.053, "A1": 0.038},
        "note": "LLM arms not re-bought; the semantic channel added no value on this benchmark "
                "(prior round, preserved)"}
    out["_structurally_not_exercised"] = {
        "person_level_shrinkage": "one choice per person — no repeated measures; tested in OmniBehavior",
        "persistence": "one-shot games", "temporal_rollout": "single decision event"}
    out["_meta"] = {"llm_calls": 0, "est_cost_usd": 0.0, "runtime_s": round(time.time() - t0, 1)}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nAGGREGATE: {out['aggregate_w1_norm']}")
    print(f"INTERACTION GAMES: {out['aggregate_interaction_games']}")
    print(f"wrote {RESULT} ({out['_meta']['runtime_s']}s)")
    return out


def _make_uninformative_beliefs(model):
    """Ablation: sever the cross-game interaction structure — acceptance 0.5 flat, banker return
    fraction point 1/3, PG belief fixed midpoint — while keeping preferences and noise identical."""
    from swm.world_model_v2.registry.families.choice import GAME_GRID, _grid
    lo, hi, step = GAME_GRID["ultimatum_proposer"]
    model._cache["acc_curve"] = {o: 0.5 for o in _grid(lo, hi, step)}
    model._cache["banker_phi"] = [(1.0 / 3.0, 1.0)]
    model._cache["pg_belief"] = 10.0
    real_invalidate = model.invalidate
    model.invalidate = lambda: None                     # pin the ablated beliefs
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pred", type=int, default=400)
    a = ap.parse_args()
    run(a.n_pred)
