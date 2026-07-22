"""BehaviorBench interaction benchmark — Reference World B, leak-free, paired (portfolio item 1).

Task: predict the DISTRIBUTION of ~100 held-out real human choices per economic game (7 games,
moblab/game_behavior; one condition per game; immutable 50/50 train/test split, seed 13).

Arms (identical test rows, identical evidence):
  A0 uniform prior            A1 train empirical (persistence)        A2 train KDE (strongest statistical)
  A3 direct LLM sampling      — K=40 TRUE separate calls/game, temp 1.0, NEVER memoized
  A4 LLM distribution-elicitation ensemble — 3 calls predicting the 200-person histogram, pooled
  A5 V2 MAX-CAPACITY          — fitted social-preference latents (cross-game), simulated-partner
                                interaction, LLM interpretation modulation, trembling fitted on train;
                                readout via the universal runtime (entities/events/deltas/terminal states)
Ablations of A5: no structured interpretation | no latent heterogeneity | no interaction.

STRUCTURALLY NOT EXERCISED here (logged, not faked): persistence (one-shot games), long-horizon temporal
rollout, institutions. Those get their fair tests in OmniBehavior / Higgs / ForecastBench.

Metric: normalized Wasserstein-1 to the held-out human sample; paired bootstrap CIs (test resampled, both
arms scored on the same resample). Plus mean-abs-error of means, cost, latency.

Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_behaviorbench_run
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_behaviorbench.json"
GAMES = ["dictator", "ultimatum_responder", "trust_banker",                  # interaction N/A (controls)
         "ultimatum_proposer", "trust_investor", "guessing", "public_goods"]  # interaction STRUCTURAL
INTERACTION_GAMES = ("ultimatum_proposer", "trust_investor", "guessing", "public_goods")
DS_IN, DS_OUT = 0.27e-6, 1.10e-6


def w1(a, b):
    if not a or not b:
        return None
    a, b = sorted(a), sorted(b)
    n = 50
    return sum(abs(a[min(len(a) - 1, int(t / n * len(a)))] -
                   b[min(len(b) - 1, int(t / n * len(b)))]) for t in range(n)) / n


def paired_w1_ci(sample_a, sample_b, test, scale, *, n_boot=500, seed=17):
    """ΔW1_norm = W1(a,test*) − W1(b,test*) over bootstrap resamples of the SAME test draw."""
    rng = random.Random(seed)
    n = len(test)
    ds = []
    for _ in range(n_boot):
        t = [test[rng.randrange(n)] for _ in range(n)]
        ds.append((w1(sample_a, t) - w1(sample_b, t)) / scale)
    ds.sort()
    return {"mean": round(sum(ds) / len(ds), 4),
            "ci95": [round(ds[int(0.025 * len(ds))], 4), round(ds[int(0.975 * len(ds))], 4)]}


def run(k_samples, n_pred, particles):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.grounding import parse_json
    from swm.eval.behavior_eval import load_game, _num
    from swm.world_model_v2.actor_cognition import interpret
    from swm.world_model_v2.reference.behavior_games import (GAME_SPECS, fit_beta, fit_priors,
                                                             simulate_game, v2_game_world)
    t0 = time.time()
    meter = {"calls": 0, "tokens": 0}
    _llm_sample = default_chat_fn(system="You are a player who is playing an economics game.",
                                  max_tokens=120, temperature=1.0)
    _llm_json = default_chat_fn(system="You are a careful behavioral scientist. Reply ONLY compact JSON.",
                                max_tokens=500, temperature=0.7)
    if _llm_sample is None:
        raise SystemExit("needs DEEPSEEK_API_KEY")

    def call(fn, prompt):                                    # metered, NEVER memoized (true sampling)
        txt = fn(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
        return txt

    # ---- immutable splits ----
    data, train_by, test_by = {}, {}, {}
    for g in GAMES:
        rows = load_game(g)
        vals = [r["human_answer"] for r in rows]
        rng = random.Random(13)
        idx = list(range(len(vals)))
        rng.shuffle(idx)
        cut = len(idx) // 2
        train_by[g] = [vals[i] for i in idx[:cut]]
        test_by[g] = [vals[i] for i in idx[cut:]]
        data[g] = rows[0]                                    # the single shared prompt
        print(f"{g}: n={len(vals)} train={len(train_by[g])} test={len(test_by[g])} "
              f"train_mean={sum(train_by[g])/cut:.1f} test_mean={sum(test_by[g])/len(test_by[g]):.1f}",
              flush=True)

    gp = fit_priors(train_by)
    print(f"fitted: levelk_w={[round(w, 2) for w in gp.levelk_w]} pg_slope={gp.pg_slope}", flush=True)

    out = {"games": {}, "_splits": {g: {"seed": 13, "n_train": len(train_by[g]), "n_test": len(test_by[g])}
                                    for g in GAMES}}
    agg = {a: [] for a in ("A0", "A1", "A2", "A3", "A4", "A5", "V2_no_interp", "V2_no_latent",
                           "V2_no_interaction")}
    lat = {"A3_s": 0.0, "A4_s": 0.0, "A5_s": 0.0}
    for g in GAMES:
        spec = GAME_SPECS[g]
        lo, hi = spec["lo"], spec["hi"]
        scale = hi - lo
        test = test_by[g]
        prompt = f"{data[g]['system']}\n\n{data[g]['user']}"
        rng = random.Random(29)
        arms = {}
        arms["A0"] = [rng.uniform(lo, hi) for _ in range(n_pred)]
        arms["A1"] = list(train_by[g])
        # A2: KDE resample of the game's own train dist (the honest strongest statistical arm)
        arms["A2"] = [gp.dists[g].sample(rng) for _ in range(n_pred)]
        # A3 true sampling: K separate calls
        t = time.time()
        a3 = []
        for _ in range(k_samples):
            v = _num(call(_llm_sample, prompt))
            if v is not None:
                a3.append(min(hi, max(lo, v)))
        arms["A3"] = a3
        lat["A3_s"] += time.time() - t
        # A4 distribution elicitation ×3, pooled
        t = time.time()
        a4 = []
        for j in range(3):
            r = parse_json(call(_llm_json,
                f"200 different people each played this economics game once (anonymously, real stakes):\n"
                f"---\n{data[g]['user'][:900]}\n---\n"
                f"Predict the DISTRIBUTION of their 200 answers (range {lo:.0f}-{hi:.0f}). Return ONLY "
                f'JSON: {{"histogram": {{"<answer_value>": <count>, ...}}}} with 5-15 bins, counts summing '
                f"to about 200. Attempt {j + 1}.")) or {}
            for k, c in (r.get("histogram") or {}).items():
                try:
                    a4 += [min(hi, max(lo, float(k)))] * max(0, min(200, int(c)))
                except (TypeError, ValueError):
                    continue
        arms["A4"] = a4
        lat["A4_s"] += time.time() - t
        # A5 V2 max-capacity
        t = time.time()
        itp = interpret(lambda p: call(_llm_json, p), actor="a participant in an economics experiment",
                        channel="one-shot anonymous economics game, real stakes",
                        context="- you play once; no reputation; the other player is a stranger",
                        content=data[g]["user"][:900])
        beta = fit_beta(g, gp, train_by[g], itp, interaction=True)
        arms["A5"] = simulate_game(g, gp, itp, n=n_pred, seed=37, beta=beta, interaction=True,
                                   latent=True, interp_on=True)
        # the anti-cheating runtime path (typed world, events, deltas) + consistency check
        wtrace = v2_game_world(g, gp, itp, n_particles=particles, seed=41, beta=beta, interaction=True)
        consist = w1(arms["A5"], wtrace["sample"])
        lat["A5_s"] += time.time() - t
        # ablations (same fitted β refit per configuration on train — each arm at its best train config)
        arms["V2_no_interp"] = simulate_game(g, gp, None, n=n_pred, seed=37,
                                             beta=fit_beta(g, gp, train_by[g], None), interaction=True,
                                             latent=True, interp_on=False)
        arms["V2_no_latent"] = simulate_game(g, gp, itp, n=n_pred, seed=37, beta=beta, interaction=True,
                                             latent=False, interp_on=True)
        arms["V2_no_interaction"] = simulate_game(g, gp, itp, n=n_pred, seed=37,
                                                  beta=fit_beta(g, gp, train_by[g], itp,
                                                                interaction=False),
                                                  interaction=False, latent=True, interp_on=True)
        res = {}
        for a, s in arms.items():
            if s:
                res[a] = {"w1_norm": round(w1(s, test) / scale, 4), "n": len(s),
                          "mean": round(sum(s) / len(s), 1)}
        res["_test_mean"] = round(sum(test) / len(test), 1)
        res["_paired"] = {
            "A5_vs_A1": paired_w1_ci(arms["A5"], arms["A1"], test, scale),
            "A5_vs_A2": paired_w1_ci(arms["A5"], arms["A2"], test, scale),
            "A5_vs_A3": paired_w1_ci(arms["A5"], arms["A3"], test, scale) if arms["A3"] else None,
            "A5_vs_no_interaction": paired_w1_ci(arms["A5"], arms["V2_no_interaction"], test, scale),
            "A5_vs_no_interp": paired_w1_ci(arms["A5"], arms["V2_no_interp"], test, scale),
            "A5_vs_no_latent": paired_w1_ci(arms["A5"], arms["V2_no_latent"], test, scale)}
        res["_v2"] = {"beta": beta, "interp": (itp.as_dict() if itp else None),
                      "runtime_consistency_w1norm": round(consist / scale, 4),
                      "n_deltas_runtime": wtrace["n_deltas"],
                      "interaction_structural": g in INTERACTION_GAMES}
        out["games"][g] = res
        for a in agg:
            if res.get(a):
                agg[a].append(res[a]["w1_norm"])
        print(f"\n== {g} (interaction {'STRUCTURAL' if g in INTERACTION_GAMES else 'n/a'}) ==")
        for a in ("A0", "A1", "A2", "A3", "A4", "A5", "V2_no_interp", "V2_no_latent", "V2_no_interaction"):
            if res.get(a):
                print(f"  {a:17s} W1n={res[a]['w1_norm']:.4f} mean={res[a]['mean']}")
        print(f"  A5 vs A2 (stat): {res['_paired']['A5_vs_A2']}  "
              f"A5 vs no_interaction: {res['_paired']['A5_vs_no_interaction']}", flush=True)

    out["aggregate_w1_norm"] = {a: round(sum(v) / len(v), 4) for a, v in agg.items() if v}
    # interaction-structural subset aggregate (the fair test)
    out["aggregate_interaction_games"] = {
        a: round(sum(out["games"][g][a]["w1_norm"] for g in INTERACTION_GAMES if out["games"][g].get(a))
                 / len(INTERACTION_GAMES), 4)
        for a in ("A1", "A2", "A3", "A4", "A5", "V2_no_interaction") if
        all(out["games"][g].get(a) for g in INTERACTION_GAMES)}
    out["_structurally_not_exercised"] = {
        "persistence": "one-shot games — no repeated state; fair test: OmniBehavior",
        "temporal_rollout": "single decision event; fair test: Higgs/SEISMIC cascades",
        "institutions": "no rule systems beyond game payoffs; fair test: ForecastBench institutional Qs"}
    out["_meta"] = {"llm_calls": meter["calls"], "llm_tokens_est": meter["tokens"],
                    "est_cost_usd": round(meter["tokens"] * (DS_IN + DS_OUT) / 2, 4),
                    "model_name": "deepseek-chat (DeepSeek V3)", "k_samples": k_samples,
                    "runtime_s": round(time.time() - t0, 1), "latency_s": {k: round(v, 1)
                                                                           for k, v in lat.items()},
                    "license_note": "moblab/game_behavior cc-by-nc-nd — benchmark use only"}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nAGGREGATE W1_norm: {out['aggregate_w1_norm']}")
    print(f"INTERACTION GAMES: {out['aggregate_interaction_games']}")
    print(f"wrote {RESULT} (calls={meter['calls']}, ~${out['_meta']['est_cost_usd']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-samples", type=int, default=40)
    ap.add_argument("--n-pred", type=int, default=400)
    ap.add_argument("--particles", type=int, default=100)
    a = ap.parse_args()
    run(a.k_samples, a.n_pred, a.particles)
