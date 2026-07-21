"""Higgs diffusion — Phase 7 re-run: NONLINEAR mechanism families vs the fitted-logistic ceiling.

Prior verdict (experiments/results/wmv2_higgs.json): the linear-hazard contagion world was significantly
WORSE than the as-of exposure logistic (Δ+0.00234 [0.00117,0.00349]) because the fitted model learned a
concave exposure response and a negative degree effect the linear form cannot express. This run tests
whether nonlinear, heterogeneous, aging hazards — executing inside the shared world — close or reverse
that gap.

Protocol (identical cohorts to the prior run — comparable numbers):
  activity from t_min; TRAIN cohort at t0+24h labeled on (24,48]h (seed 13); EVAL cohort at t0+48h
  labeled on (48,72]h (seed 17); n=4000 each; both windows pre-announcement.

Form selection is done ONLY on a train-internal split (fit half / val half, deterministic): the selected
form becomes the preregistered primary arm M_sel; test metrics for ALL arms are reported transparently.

Arms:
  H0 base rate | H1 fitted logistic (the ceiling to beat)
  M1 linear-hazard world (prior V2_max, reproduced)
  M2 complex-contagion Hill world (fitted α, c)
  M3 exposure-response log-linear hazard world (concavity + degree conditioning)
  M4 = M3 + fitted lognormal frailty (susceptibility heterogeneity)
  M5 = aging: age-weighted exposure k_τ (fitted τ) + log-linear hazard
  M6 = M5 + frailty  (full nonlinear stack)
  ablations on M_sel: no_rollout (closed form), rollout (event-driven)
  Hawkes self-excitation: fitted on train activity hours, held-out count-forecast vs Poisson baseline
    (validates the family on this dataset; separate from the per-user task)

Resumable: stage outputs cached under experiments/results/higgs_nl_cache/. Deterministic under fixed
seeds. No LLM calls (SNAP Higgs has no content — semantic arms remain STRUCTURALLY ABSENT, logged).
Run: PYTHONPATH=. python -m experiments.wmv2_higgs_nonlinear_run
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_higgs_nonlinear.json"
CACHE = Path("experiments/results/higgs_nl_cache")


def _metrics(rows, preds):
    pr = list(zip(preds, [r["y"] for r in rows]))
    n = len(pr)
    brier = sum((p - y) ** 2 for p, y in pr) / n
    ll = -sum(y * math.log(max(1e-6, p)) + (1 - y) * math.log(max(1e-6, 1 - p)) for p, y in pr) / n
    pos = [p for p, y in pr if y == 1]
    neg = [p for p, y in pr if y == 0]
    auroc = (sum(1 for a in pos for c in neg if a > c) + 0.5 * sum(1 for a in pos for c in neg if a == c)) \
        / max(1, len(pos) * len(neg)) if pos and neg else None
    ap = None
    if pos:
        ranked = sorted(pr, key=lambda t: -t[0])
        tp, ap = 0, 0.0
        for i, (_, y) in enumerate(ranked, 1):
            if y == 1:
                tp += 1
                ap += tp / i
        ap /= len(pos)
    return {"brier": round(brier, 6), "logloss": round(ll, 4),
            "auroc": round(auroc, 3) if auroc else None, "pr_auc": round(ap, 3) if ap else None,
            "real_rate": round(sum(y for _, y in pr) / n, 4),
            "pred_rate": round(sum(p for p, _ in pr) / n, 4), "n": n}


def run(n_sample, particles):
    from swm.world_model_v2.reference.higgs import (build_cohort_aged, exposure_snapshot_aged,
                                                    feats_full, fit_logistic, load_activation_times,
                                                    load_activity_stream, sample_subgraph_edges)
    from swm.world_model_v2.registry.families.diffusion import (
        LogLinearHazard, closed_form_window_p, contagion_window_predict, fit_aging_tau, fit_frailty_sigma,
        fit_hawkes, fit_hill, fit_linear_q, fit_loglinear, hawkes_forecast_counts)
    from swm.world_model_v2.registry.ingestion import paired_bootstrap_delta

    t0w = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cohort_cache = CACHE / f"cohorts_{n_sample}.json"
    if cohort_cache.exists():
        blob = json.loads(cohort_cache.read_text())
        train, test = blob["train"], blob["test"]
        for rows in (train, test):
            for r in rows:
                r["k_tau"] = {float(k): v for k, v in r["k_tau"].items()}
        fol = {int(k): v for k, v in blob["followers_of"].items()}
        print(f"cohorts from cache: train n={len(train)} test n={len(test)}", flush=True)
    else:
        print("parsing activity …", flush=True)
        first, t_min = load_activation_times()
        H = 3600.0
        cut_tr, cut_ev, win = t_min + 24 * H, t_min + 48 * H, 24 * H
        print(f"users_ever_active={len(first)}; streaming follower graph (aged, 2 cutoffs) …", flush=True)
        snap = exposure_snapshot_aged(first, [cut_tr, cut_ev])
        train = build_cohort_aged(first, snap, cut_tr, win, n_sample=n_sample, seed=13)
        test = build_cohort_aged(first, snap, cut_ev, win, n_sample=n_sample, seed=17)
        fol = sample_subgraph_edges([r["u"] for r in test])
        cohort_cache.write_text(json.dumps(
            {"train": train, "test": test,
             "followers_of": {str(k): v for k, v in fol.items()}}, default=str))
        print("cohorts built + cached", flush=True)

    W = 1.0                                   # window in days
    ys_te = [r["y"] for r in test]
    r_tr = sum(r["y"] for r in train) / len(train)
    print(f"train rate={r_tr:.4f} test rate={sum(ys_te)/len(ys_te):.4f}", flush=True)

    # ---------------- fits (train only) ----------------
    fit_half = [r for i, r in enumerate(train) if i % 2 == 0]
    val_half = [r for i, r in enumerate(train) if i % 2 == 1]

    hz_lin = fit_linear_q(train, W)
    hz_hill = fit_hill(train, W)
    hz_ll = fit_loglinear(train, W)
    sig_ll, sig_profile = fit_frailty_sigma(hz_ll, train, W)
    tau, hz_age, tau_lls = fit_aging_tau(train, W)
    for rows in (train, test):
        for r in rows:
            r["k_eff0"] = r["k_tau"][tau]
    sig_age, sig_age_profile = fit_frailty_sigma(hz_age, train, W, k_key="k_eff0")
    # NOTE fit set for form SELECTION: fit on fit_half, score on val_half (no test contact)
    hz_ll_h = fit_loglinear(fit_half, W)
    hz_hill_h = fit_hill(fit_half, W)
    hz_lin_h = fit_linear_q(fit_half, W)
    sig_h, _ = fit_frailty_sigma(hz_ll_h, fit_half, W)
    tau_h, hz_age_h, _ = fit_aging_tau(fit_half, W)
    for r in val_half:
        r["k_eff0_h"] = r["k_tau"][tau_h]

    def cf(rows, hz, sigma=0.0, aging=False, kkey=None):
        if kkey:
            saved = [r.get("k_eff0") for r in rows]
            for r in rows:
                r["k_eff0"] = r[kkey]
        out = closed_form_window_p(rows, hz, W, frailty_sigma=sigma,
                                   aging_tau_h=(tau_h if aging else None))
        if kkey:
            for r, s in zip(rows, saved):
                r["k_eff0"] = s
        return out

    val_scores = {}
    for name, preds in {
        "M1_linear": cf(val_half, hz_lin_h),
        "M2_hill": cf(val_half, hz_hill_h),
        "M3_loglinear": cf(val_half, hz_ll_h),
        "M4_frailty": cf(val_half, hz_ll_h, sigma=sig_h),
        "M5_aging": cf(val_half, hz_age_h, aging=True, kkey="k_eff0_h"),
        "M6_aging_frailty": cf(val_half, hz_age_h, sigma=sig_h, aging=True, kkey="k_eff0_h"),
    }.items():
        val_scores[name] = round(sum((p - r["y"]) ** 2 for p, r in zip(preds, val_half)) / len(val_half), 6)
    m_sel = min(val_scores, key=val_scores.get)
    print(f"form selection (train-internal val): {val_scores} → M_sel={m_sel}", flush=True)

    # ---------------- test-set arms (all reported; primary = M_sel) ----------------
    h0 = [max(1e-4, r_tr)] * len(test)
    pred_h1, coef = fit_logistic(train, feats_full)
    h1 = [pred_h1(r) for r in test]

    t1 = time.time()
    arms = {
        "H0_base": h0,
        "H1_logistic": h1,
        "M1_linear": contagion_window_predict(test, hz_lin, W, fol, n_particles=particles, seed=7,
                                              latent_scale_sd=0.4),
        "M2_hill": contagion_window_predict(test, hz_hill, W, fol, n_particles=particles, seed=7),
        "M3_loglinear": contagion_window_predict(test, hz_ll, W, fol, n_particles=particles, seed=7),
        "M4_frailty": contagion_window_predict(test, hz_ll, W, fol, n_particles=particles, seed=7,
                                               frailty_sigma=sig_ll),
        "M5_aging": contagion_window_predict(test, hz_age, W, fol, n_particles=particles, seed=7,
                                             aging_tau_h=tau),
        "M6_aging_frailty": contagion_window_predict(test, hz_age, W, fol, n_particles=particles, seed=7,
                                                     aging_tau_h=tau, frailty_sigma=sig_age),
    }
    sim_latency = time.time() - t1
    # ablation: M_sel without rollout (closed form) — what does event-driven execution add?
    sel_hz = {"M1_linear": (hz_lin, 0.0, None), "M2_hill": (hz_hill, 0.0, None),
              "M3_loglinear": (hz_ll, 0.0, None), "M4_frailty": (hz_ll, sig_ll, None),
              "M5_aging": (hz_age, 0.0, tau), "M6_aging_frailty": (hz_age, sig_age, tau)}[m_sel]
    arms[f"{m_sel}_no_rollout"] = closed_form_window_p(test, sel_hz[0], W, frailty_sigma=sel_hz[1],
                                                       aging_tau_h=sel_hz[2])

    detail = {a: _metrics(test, p) for a, p in arms.items()}
    paired = {
        "Msel_vs_H1": paired_bootstrap_delta(ys_te, arms[m_sel], h1),
        "Msel_vs_H0": paired_bootstrap_delta(ys_te, arms[m_sel], h0),
        "Msel_vs_M1_linear": paired_bootstrap_delta(ys_te, arms[m_sel], arms["M1_linear"]),
        "M3_vs_M1_nonlinearity": paired_bootstrap_delta(ys_te, arms["M3_loglinear"], arms["M1_linear"]),
        "M4_vs_M3_frailty": paired_bootstrap_delta(ys_te, arms["M4_frailty"], arms["M3_loglinear"]),
        "M5_vs_M3_aging": paired_bootstrap_delta(ys_te, arms["M5_aging"], arms["M3_loglinear"]),
        "Msel_vs_no_rollout": paired_bootstrap_delta(ys_te, arms[m_sel], arms[f"{m_sel}_no_rollout"]),
        "H1_vs_H0": paired_bootstrap_delta(ys_te, h1, h0),
    }

    # ---------------- Hawkes self-excitation (family validation on real cascade timing) ----------------
    print("fitting Hawkes on activity stream …", flush=True)
    stream = load_activity_stream()
    t_min = stream[0]
    tr_a, tr_b = t_min, t_min + 48 * 3600
    ev_a, ev_b = tr_b, t_min + 72 * 3600
    mu, al, om, ll = fit_hawkes(stream, tr_a, tr_b)
    n_bins = 24
    actual = [0] * n_bins
    binw = (ev_b - ev_a) / n_bins
    for t in stream:
        if ev_a <= t < ev_b:
            b = int((t - ev_a) / binw)
            if b < n_bins:
                actual[b] += 1
    fc = hawkes_forecast_counts(mu, al, om, stream, ev_a, ev_b, n_bins, n_sims=60, seed=3)
    train_rate = sum(1 for t in stream if tr_a <= t < tr_b) / (tr_b - tr_a)
    pois = [train_rate * binw] * n_bins
    mae_h = sum(abs(f - a) for f, a in zip(fc, actual)) / n_bins
    mae_p = sum(abs(f - a) for f, a in zip(pois, actual)) / n_bins
    hawkes_out = {"fitted": {"mu_per_s": mu, "branching_alpha": round(al, 4),
                             "omega_per_s": om, "train_ll": round(ll, 1),
                             "source": "fitted (train window 0-48h, EM/MLE exponential kernel)"},
                  "held_out_24_72h": {"mae_hawkes": round(mae_h, 1), "mae_poisson": round(mae_p, 1),
                                      "actual_total": sum(actual), "hawkes_total": round(sum(fc), 1),
                                      "poisson_total": round(sum(pois), 1)},
                  "passed": mae_h < mae_p,
                  "limits": "aggregate stream only (not per-user); constant background μ — circadian "
                            "baseline NOT modeled (logged omission); exponential kernel only"}
    print(f"hawkes: mae {mae_h:.1f} vs poisson {mae_p:.1f} (actual {sum(actual)})", flush=True)

    out = {"protocol": {"cohorts": "identical to wmv2_higgs.json (train@24h/eval@48h, 24h windows, "
                                   "seeds 13/17, both pre-announcement)",
                        "form_selection": "train-internal fit/val split (even/odd rows); "
                                          "primary arm preregistered as val winner before test scoring",
                        "n_sample": n_sample, "particles": particles},
           "fitted": {"linear_q": hz_lin.params(),
                      "hill": hz_hill.params(),
                      "loglinear_theta": hz_ll.params(),
                      "frailty_sigma": {"value": sig_ll, "profile_ll": sig_profile,
                                        "source": "fitted (profile likelihood, train)"},
                      "aging_tau_h": {"value": tau, "grid_ll": {str(k): v for k, v in tau_lls.items()},
                                      "source": "fitted (grid likelihood, train)"},
                      "aging_frailty_sigma": sig_age,
                      "logistic_baseline": coef},
           "val_selection": {"scores_brier": val_scores, "selected": m_sel},
           "detail": detail, "paired": paired,
           "hawkes_family_validation": hawkes_out,
           "llm_arms": "STRUCTURALLY ABSENT — no message content in SNAP Higgs",
           "_meta": {"sim_latency_s": round(sim_latency, 1),
                     "runtime_s": round(time.time() - t0w, 1), "llm_calls": 0, "est_cost_usd": 0.0}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    for a, m in detail.items():
        print(f"  {a:22s} brier={m['brier']} auroc={m['auroc']} pr_auc={m['pr_auc']}")
    for k, v in paired.items():
        print(f"  {k}: Δ={v['mean']:+.6f} CI{v['ci95']}")
    print(f"wrote {RESULT} ({out['_meta']['runtime_s']}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=4000)
    ap.add_argument("--particles", type=int, default=30)
    a = ap.parse_args()
    run(a.n_sample, a.particles)
