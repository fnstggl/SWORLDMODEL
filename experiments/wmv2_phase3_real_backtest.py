"""Phase 3 posterior — REAL RESOLVED historical backtest (validation only).

Runs the ACTUAL production path on held-out, already-resolved historical questions with KNOWN binary
outcomes, then scores forecasting accuracy AGAINST the realized outcome. This is the accuracy test the live
harness (wmv2_phase3_live_validation.py) deliberately does NOT do: that harness measures whether the posterior
*moved* the number; this one measures whether posterior consumption makes the forecast *more right*.

Production path (identical to serving):
    historical question
      -> compile_world (Phase-2 CODE)
      -> gather_evidence  (strict as-of retrieval: Google News RSS after:/before:, per-doc temporal
         verification, claim-level leakage audit)  ==> one frozen EvidenceBundleV2, REUSED across all arms
      -> tag_claims (qualitative, no numbers)      ==> frozen tags, REUSED across all arms
      -> infer_posterior (particle posterior)
      -> materialize onto plan (or not)            <== THE ONLY THING THAT VARIES BETWEEN THE KEY TWO ARMS
      -> rollout terminal

Arms (all share the SAME frozen plan + bundle + tags + seed + outcome contract):
    prior_only            reference-class prior mean, NO evidence assimilation at the terminal
    phase2_no_posterior   Phase-2 evidence path, Phase-3 posterior NOT consumed (consume_posterior=False)
    phase3_posterior      Phase-3 posterior-conditioned terminal        (consume_posterior=True)
    point_estimate        posterior collapsed to its scalar mean        (anti-scalar ablation)
    market                crowd/prediction-market implied prob where reliably available (else null)

THE KEY PAIRED COMPARISON is phase3_posterior vs phase2_no_posterior: identical everything except whether the
Phase-3 posterior is consumed. Scored with Brier, log-loss, calibration (ECE), directional accuracy, and a
paired bootstrap CI on the per-question Brier/log-loss difference. Per-question deltas are recorded so an
aggregate win cannot hide a per-question regression.

Networked + LLM-backed; per-question failures are recorded, never fatal; the artifact is written incrementally
so a timeout leaves a scorable partial. "the probability changed" is NOT counted as improvement.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

OUT = Path("experiments/results/phase3")
ART = OUT / "real_backtest.json"

# ------------------------------------------------------------------ held-out RESOLVED question set
# (qid, question, as_of, horizon, domain, outcome[1=yes/0=no], market_prob_or_None, resolution_note)
# as_of is STRICTLY BEFORE the resolution date. Outcomes are matters of public record. market_prob is left
# None unless a defensible as-of crowd/market number is available offline — fabricating market prices would
# poison the baseline, so we record null and say so rather than guess.
QUESTIONS = [
    # --- elections / politics ---
    ("trump_2024",   "Will Donald Trump win the 2024 US presidential election?", "2024-10-20", "2024-11-06", "elections", 1, None, "Trump won; called 2024-11-06."),
    ("harris_2024",  "Will Kamala Harris win the 2024 US presidential election?", "2024-10-20", "2024-11-06", "elections", 0, None, "Harris lost the 2024 election."),
    ("biden_nominee","Will Joe Biden be the Democratic nominee for the 2024 US presidential election?", "2024-07-05", "2024-08-22", "elections", 0, None, "Biden withdrew 2024-07-21; Harris nominated."),
    ("uk_labour",    "Will the Labour Party win the 2024 United Kingdom general election?", "2024-06-25", "2024-07-04", "elections", 1, None, "Labour won a majority 2024-07-04."),
    ("shutdown_oct24","Will there be a US federal government shutdown on October 1, 2024?", "2024-09-20", "2024-10-01", "politics", 0, None, "CR signed 2024-09-26; no shutdown."),
    ("shutdown_dec24","Will there be a US federal government shutdown before the end of 2024?", "2024-12-10", "2024-12-31", "politics", 0, None, "CR passed 2024-12-21; no shutdown."),
    # --- economics / finance ---
    ("fed_sep24",    "Will the US Federal Reserve cut interest rates at its September 2024 meeting?", "2024-09-10", "2024-09-19", "econ", 1, None, "FOMC cut 50bp 2024-09-18."),
    ("fed_nov24",    "Will the US Federal Reserve cut interest rates at its November 2024 meeting?", "2024-10-28", "2024-11-08", "econ", 1, None, "FOMC cut 25bp 2024-11-07."),
    ("fed_dec24",    "Will the US Federal Reserve cut interest rates at its December 2024 meeting?", "2024-12-05", "2024-12-19", "econ", 1, None, "FOMC cut 25bp 2024-12-18."),
    ("fed_jan25",    "Will the US Federal Reserve cut interest rates at its January 2025 meeting?", "2025-01-20", "2025-01-30", "econ", 0, None, "FOMC held steady 2025-01-29."),
    ("btc_100k",     "Will Bitcoin exceed one hundred thousand US dollars by the end of 2024?", "2024-11-15", "2024-12-31", "finance", 1, None, "BTC passed $100k 2024-12-04."),
    ("recession_24", "Will the United States enter a recession in 2024?", "2024-07-01", "2024-12-31", "macro", 0, None, "No NBER recession in 2024."),
    ("nvda_split",   "Will Nvidia announce a stock split in 2024?", "2024-05-01", "2024-06-30", "finance", 1, None, "Nvidia announced 10-for-1 split 2024-05-22."),
    ("sp500_6000",   "Will the S&P 500 index close above 6000 in 2024?", "2024-11-01", "2024-12-31", "finance", 1, None, "S&P 500 first closed >6000 2024-11-08."),
    # --- technology ---
    ("gpt5_2024",    "Will OpenAI release a model called GPT-5 in 2024?", "2024-08-01", "2024-12-31", "tech", 0, None, "No GPT-5 in 2024."),
    ("gpt5_2025",    "Will OpenAI release a model called GPT-5 in 2025?", "2025-06-01", "2025-12-31", "tech", 1, None, "OpenAI released GPT-5 2025-08."),
    ("apple_intel",  "Will Apple release its Apple Intelligence features in 2024?", "2024-08-01", "2024-12-31", "tech", 1, None, "Apple Intelligence launched 2024-10."),
    # --- geopolitics ---
    ("gaza_ceasefire24","Will Israel and Hamas agree to a ceasefire by the end of 2024?", "2024-10-01", "2024-12-31", "geopolitics", 0, None, "No ceasefire in 2024; deal reached Jan 2025."),
    ("gaza_ceasefire25","Will an Israel-Hamas ceasefire take effect in January 2025?", "2025-01-10", "2025-01-31", "geopolitics", 1, None, "Ceasefire took effect 2025-01-19."),
    ("assad_fall",   "Will Bashar al-Assad's government fall in Syria in 2024?", "2024-11-25", "2024-12-31", "geopolitics", 1, None, "Assad government fell 2024-12-08."),
    ("ru_ua_cf24",   "Will Russia and Ukraine agree to a ceasefire in 2024?", "2024-06-01", "2024-12-31", "geopolitics", 0, None, "No ceasefire in 2024."),
    # --- sports ---
    ("india_t20",    "Will India win the 2024 ICC Men's T20 Cricket World Cup?", "2024-06-20", "2024-06-29", "sports", 1, None, "India won the final 2024-06-29."),
    ("real_ucl",    "Will Real Madrid win the 2024 UEFA Champions League final?", "2024-05-25", "2024-06-01", "sports", 1, None, "Real Madrid won the final 2024-06-01."),
    # --- science / space ---
    ("starship_catch","Will SpaceX catch a Starship booster with the launch tower in 2024?", "2024-10-01", "2024-12-31", "science", 1, None, "Booster caught on Flight 5, 2024-10-13."),
]

_EPS = 1e-6


def _clip(p):
    return min(1.0 - _EPS, max(_EPS, p))


def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = _clip(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    llm0 = default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)
    if llm0 is None:
        return None, None
    meter = {"calls": 0}

    def llm(p):
        meter["calls"] += 1
        return llm0(p)
    return llm, meter


def _bundle_trace(bundle, art):
    """Leakage/temporal facts for the manual audit (published dates, temporal status, excluded/leaked)."""
    docs = []
    for d in bundle.documents[:12]:
        docs.append({
            "id": d.get("id"), "source": d.get("source"), "url": (d.get("url") or "")[:120],
            "published_at": d.get("published_at"),
            "published_iso": (time.strftime("%Y-%m-%d", time.gmtime(d["published_at"]))
                              if isinstance(d.get("published_at"), (int, float)) and d.get("published_at") else None),
            "temporal_status": d.get("temporal_status")})
    return {
        "bundle_hash": bundle.bundle_hash(),
        "as_of_iso": time.strftime("%Y-%m-%d", time.gmtime(bundle.as_of)),
        "n_documents": len(bundle.documents),
        "n_included": len(bundle.included_claim_ids),
        "n_excluded": len(bundle.excluded_claim_ids),
        "n_suspicious": len(bundle.suspicious_claim_ids),
        "n_leakage_flags": len(bundle.leakage_flags),
        "leakage_flags": bundle.leakage_flags[:8],
        "documents": docs}


def run_question(q, llm, cfg, seed=0):
    from swm.world_model_v2.phase3_pipeline import simulate_with_posterior
    qid, question, as_of, horizon, domain, outcome, market, note = q
    rec = {"qid": qid, "question": question, "domain": domain, "as_of": as_of, "horizon": horizon,
           "outcome": outcome, "market_prob": market, "resolution_note": note}
    t0 = time.time()
    try:
        # ---- ARM: phase3_posterior — full production path; freezes plan+bundle+tags for the other arms
        res3, art = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                            config=cfg, consume_posterior=True)
        plan, bundle, tags = art.get("plan"), art.get("bundle"), art.get("tags")
        pi = res3.posterior_inference or {}
        prior_mean = (pi.get("outcome_rate") or {}).get("prior_mean")
        post_mean = (pi.get("outcome_rate") or {}).get("posterior_mean")

        # ---- ARM: phase2_no_posterior — SAME plan+bundle+tags, posterior computed but NOT consumed
        res2, _ = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                          config=cfg, consume_posterior=False, plan=plan, bundle=bundle, tags=tags)
        # ---- ARM: point_estimate — SAME everything, posterior collapsed to scalar mean
        resP, _ = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                          config=cfg, consume_posterior=True, posterior_point_estimate=True,
                                          plan=plan, bundle=bundle, tags=tags)
        # ---- within-run reproducibility of the numeric posterior (same frozen inputs -> identical hash)
        res3b, artb = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                              config=cfg, consume_posterior=True, plan=plan, bundle=bundle, tags=tags)

        arms = {
            "prior_only": prior_mean,                                  # pre-evidence reference-class prior
            "phase2_no_posterior": res2.raw_probability,
            "phase3_posterior": res3.raw_probability,
            "point_estimate": resP.raw_probability,
            "market": market}
        rec.update({
            "status": res3.simulation_status, "support_grade": res3.support_grade,
            "arms": arms,
            "prior_mean": prior_mean, "posterior_mean": post_mean,
            "posterior_shift": (None if prior_mean is None or post_mean is None
                                else round(post_mean - prior_mean, 5)),
            "n_included_claims": art["planes"]["evidence"].get("n_included_claims"),
            "n_effective_observations": art["planes"]["posterior"].get("n_effective_observations"),
            "rate_source": art["planes"].get("execution", {}).get("rate_source"),
            "posterior_consumed": pi.get("consumed_by_simulator"),
            "reproducible_hash": art.get("posterior_hash") == artb.get("posterior_hash"),
            "posterior_hash": art.get("posterior_hash"),
            "trace": _bundle_trace(bundle, art),
            "latency_s": round(time.time() - t0, 1)})
    except Exception as e:  # noqa: BLE001
        rec.update({"status": "harness_error", "error": f"{type(e).__name__}: {e}"[:200],
                    "arms": {}, "latency_s": round(time.time() - t0, 1)})
    return rec


# ------------------------------------------------------------------ scoring
ARM_NAMES = ["prior_only", "phase2_no_posterior", "phase3_posterior", "point_estimate", "market"]


def _score_arm(rows, arm):
    pts = [(r["arms"][arm], r["outcome"]) for r in rows
           if r.get("arms", {}).get(arm) is not None and r.get("outcome") in (0, 1)]
    if not pts:
        return {"n": 0}
    briers = [_brier(p, y) for p, y in pts]
    lls = [_logloss(p, y) for p, y in pts]
    dirok = [1 if (p > 0.5) == (y == 1) else (0 if p != 0.5 else 0) for p, y in pts]
    # 10-bin ECE
    bins = {}
    for p, y in pts:
        b = min(9, int(p * 10))
        bins.setdefault(b, []).append((p, y))
    ece = sum((len(v) / len(pts)) * abs(sum(p for p, _ in v) / len(v) - sum(y for _, y in v) / len(v))
              for v in bins.values())
    return {"n": len(pts), "brier": round(sum(briers) / len(briers), 4),
            "log_loss": round(sum(lls) / len(lls), 4),
            "directional_acc": round(sum(dirok) / len(pts), 4),
            "ece": round(ece, 4),
            "mean_p": round(sum(p for p, _ in pts) / len(pts), 4)}


def _paired_bootstrap(rows, arm_a, arm_b, n_boot=10000, seed=12345):
    """Paired bootstrap on per-question (Brier_a - Brier_b) and (LogLoss_a - LogLoss_b).
    Positive mean => arm_a WORSE (higher loss) than arm_b. We report a=phase3, b=phase2 so a NEGATIVE
    difference means Phase-3 improves. Deterministic LCG (no Date/random dependence)."""
    pairs = [(r["arms"][arm_a], r["arms"][arm_b], r["outcome"]) for r in rows
             if r.get("arms", {}).get(arm_a) is not None and r.get("arms", {}).get(arm_b) is not None
             and r.get("outcome") in (0, 1)]
    if len(pairs) < 3:
        return {"n": len(pairs), "insufficient": True}
    db = [_brier(a, y) - _brier(b, y) for a, b, y in pairs]
    dl = [_logloss(a, y) - _logloss(b, y) for a, b, y in pairs]
    n = len(pairs)
    state = seed & 0xFFFFFFFF

    def _rand():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF
    mb, ml = [], []
    for _ in range(n_boot):
        idx = [int(_rand() * n) % n for _ in range(n)]
        mb.append(sum(db[i] for i in idx) / n)
        ml.append(sum(dl[i] for i in idx) / n)
    mb.sort(); ml.sort()

    def ci(v):
        return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v))], 4)]
    return {"n": n,
            "mean_brier_diff": round(sum(db) / n, 4), "brier_diff_ci95": ci(mb),
            "mean_logloss_diff": round(sum(dl) / n, 4), "logloss_diff_ci95": ci(ml),
            "brier_diff_prob_negative": round(sum(1 for x in mb if x < 0) / len(mb), 3),
            "note": "arm_a=phase3_posterior, arm_b=phase2_no_posterior; negative => Phase-3 lowers loss (improves)"}


def _per_question_deltas(rows):
    out = []
    for r in rows:
        a = r.get("arms", {})
        if a.get("phase3_posterior") is None or a.get("phase2_no_posterior") is None or r.get("outcome") not in (0, 1):
            continue
        y = r["outcome"]
        b3, b2 = _brier(a["phase3_posterior"], y), _brier(a["phase2_no_posterior"], y)
        out.append({"qid": r["qid"], "outcome": y,
                    "p_phase3": round(a["phase3_posterior"], 4), "p_phase2": round(a["phase2_no_posterior"], 4),
                    "brier_phase3": round(b3, 4), "brier_phase2": round(b2, 4),
                    "brier_delta": round(b3 - b2, 4),
                    "verdict": ("phase3_better" if b3 < b2 - 1e-9 else
                                "phase2_better" if b2 < b3 - 1e-9 else "tie")})
    return out


def aggregate(rows, meter):
    ok = [r for r in rows if r.get("status", "").startswith("completed") and r.get("arms")]
    scored = [r for r in ok if r.get("arms", {}).get("phase3_posterior") is not None]
    per_arm = {a: _score_arm(scored, a) for a in ARM_NAMES}
    deltas = _per_question_deltas(scored)
    n_better = sum(1 for d in deltas if d["verdict"] == "phase3_better")
    n_worse = sum(1 for d in deltas if d["verdict"] == "phase2_better")
    n_tie = sum(1 for d in deltas if d["verdict"] == "tie")
    boot = _paired_bootstrap(scored, "phase3_posterior", "phase2_no_posterior")
    n = len(scored)
    preliminary = n < 30
    # verdict from the paired key comparison (accuracy, not movement)
    verdict = "inconclusive"
    if isinstance(boot, dict) and not boot.get("insufficient"):
        hi = boot["brier_diff_ci95"][1]
        lo = boot["brier_diff_ci95"][0]
        if hi < 0:
            verdict = "phase3_improves"           # whole CI below 0 => improvement
        elif lo > 0:
            verdict = "phase3_harms"              # whole CI above 0 => regression
        else:
            verdict = "inconclusive"
    return {
        "n_questions": len(rows), "n_completed": len(ok), "n_scored": n,
        "n_harness_error": sum(1 for r in rows if not r.get("status", "").startswith("completed")),
        "preliminary": preliminary,
        "reproducible_hash_rate": round(sum(1 for r in ok if r.get("reproducible_hash")) / max(1, len(ok)), 3),
        "posterior_consumed_rate": round(sum(1 for r in ok if r.get("posterior_consumed")) / max(1, len(ok)), 3),
        "per_arm_scores": per_arm,
        "key_comparison_phase3_vs_phase2": {
            "per_question_phase3_better": n_better, "phase2_better": n_worse, "tie": n_tie,
            "paired_bootstrap": boot},
        "per_question_deltas": deltas,
        "llm_calls": meter["calls"] if meter else None,
        "verdict": verdict,
        "verdict_meaning": {
            "phase3_improves": "Phase 3 improves real held-out forecasting (paired Brier CI entirely < 0)",
            "inconclusive": "result is inconclusive (paired Brier CI spans 0)",
            "phase3_harms": "Phase 3 harms forecasting (paired Brier CI entirely > 0)"}[verdict]}


def run(limit=None, seed=0):
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig
    OUT.mkdir(parents=True, exist_ok=True)
    llm, meter = _make_llm()
    if llm is None:
        return {"error": "no DEEPSEEK_API_KEY / llm unavailable"}
    cfg = OrchestratorConfig()
    qs = QUESTIONS[:limit] if limit else QUESTIONS
    rows = []
    for q in qs:
        rec = run_question(q, llm, cfg, seed=seed)
        rows.append(rec)
        payload = {"rows": rows, "aggregate": aggregate(rows, meter),
                   "retrieval_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "seed": seed}
        ART.write_text(json.dumps(payload, indent=2))     # incremental: a timeout still leaves a scorable file
        a = rec.get("arms", {})
        print(f"[{rec['domain']:11s}] {rec['qid']:16s} y={rec.get('outcome')} "
              f"prior={a.get('prior_only')} ph2={a.get('phase2_no_posterior')} "
              f"ph3={a.get('phase3_posterior')} status={rec.get('status')} "
              f"neff={rec.get('n_effective_observations')} t={rec.get('latency_s')}s")
    return aggregate(rows, meter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    agg = run(limit=args.limit, seed=args.seed)
    print("\nAGGREGATE:", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
