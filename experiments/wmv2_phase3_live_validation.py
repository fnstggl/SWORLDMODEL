"""Phase 3 posterior — LIVE held-out general-path validation (real DeepSeek + live Google News RSS).

Proves the full production path on REAL evidence across domains, through the UNIVERSAL WMv2 path (no
benchmark-specific engine): for each held-out question we run `simulate_with_posterior` and record whether the
posterior actually crossed every plane and was CONSUMED, plus a posterior-IGNORED ablation arm on the same
evidence to isolate the posterior's causal effect on the terminal.

Metrics (Part Q + anti-scaffolding):
  posterior_consumed_rate   fraction of runs whose resolver drew from the posterior (rate_source=='posterior')
  prior->posterior shift     |posterior_mean - prior_mean| distribution (evidence moved the number)
  dependence collapse        n_included_claims -> n_effective_observations
  structural_updated_rate    fraction whose structural posterior differs from the structural prior
  terminal_effect            |P(yes)_consumed - P(yes)_posterior_ignored| (the posterior changed the answer)
  reproducibility            same seed twice -> identical posterior_hash + identical terminal
  no_abstention              every coherent question produced a forecast (weak evidence widened, never blocked)

Each question is independent; a per-question failure is recorded, not fatal. Networked + non-deterministic
across wall-clock (live news changes), so the artifact records the retrieval date. Reproducibility is asserted
WITHIN a run (same-process, same-bundle) where it must hold.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

OUT = Path("experiments/results/phase3")

# held-out, cross-domain; as_of chosen so contemporaneous news exists at retrieval time
QUESTIONS = [
    ("Will the Federal Reserve cut interest rates at its next meeting?", "2024-09-01", "2024-09-20", "econ"),
    ("Will OpenAI release a model called GPT-5 this year?", "2024-08-01", "2024-12-31", "tech"),
    ("Will the United States enter a recession this year?", "2024-07-01", "2024-12-31", "macro"),
    ("Will there be a US federal government shutdown?", "2024-09-15", "2024-10-15", "politics"),
    ("Will Bitcoin exceed one hundred thousand dollars?", "2024-10-01", "2024-12-31", "finance"),
    ("Will a ceasefire be agreed between Israel and Hamas?", "2024-08-01", "2024-11-01", "geopolitics"),
]


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


def run(limit=None, seed=0):
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig
    from swm.world_model_v2.phase3_pipeline import simulate_with_posterior
    llm, meter = _make_llm()
    if llm is None:
        return {"error": "no DEEPSEEK_API_KEY / llm unavailable"}
    cfg = OrchestratorConfig()
    rows, qs = [], QUESTIONS[:limit] if limit else QUESTIONS
    for q, as_of, horizon, domain in qs:
        t0 = time.time()
        rec = {"question": q, "domain": domain, "as_of": as_of, "horizon": horizon}
        try:
            res, art = simulate_with_posterior(q, llm=llm, as_of=as_of, horizon=horizon, seed=seed, config=cfg)
            pi = res.posterior_inference or {}
            planes = art.get("planes", {})
            plan, bundle, tags = art.get("plan"), art.get("bundle"), art.get("tags")  # reuse compiled world + evidence + tags
            # posterior-IGNORED ablation arm on the SAME plan+bundle+tags (posterior computed but NOT consumed)
            res0, _ = simulate_with_posterior(q, llm=llm, as_of=as_of, horizon=horizon, seed=seed, config=cfg,
                                              consume_posterior=False, plan=plan, bundle=bundle, tags=tags)
            # within-run reproducibility: same plan + seed + bundle + tags → byte-identical NUMERIC posterior.
            # This isolates ALL stochastic LLM steps (compile + tag) to shared inputs; the numeric posterior
            # pipeline (infer_posterior) must be exactly reproducible — the core Phase-3 determinism claim.
            res2, art2 = simulate_with_posterior(q, llm=llm, as_of=as_of, horizon=horizon, seed=seed, config=cfg,
                                                 plan=plan, bundle=bundle, tags=tags)
            rec.update({
                "status": res.simulation_status, "support_grade": res.support_grade,
                "has_forecast": res.has_forecast(), "p_yes_consumed": res.raw_probability,
                "p_yes_posterior_ignored": res0.raw_probability,
                "terminal_effect": (None if res.raw_probability is None or res0.raw_probability is None
                                    else round(abs(res.raw_probability - res0.raw_probability), 4)),
                "n_included_claims": planes.get("evidence", {}).get("n_included_claims"),
                "n_effective_observations": planes.get("posterior", {}).get("n_effective_observations"),
                "prior_mean": pi.get("outcome_rate", {}).get("prior_mean"),
                "posterior_mean": pi.get("outcome_rate", {}).get("posterior_mean"),
                "posterior_shift": pi.get("outcome_rate", {}).get("shift"),
                "posterior_sd": pi.get("outcome_rate", {}).get("posterior_sd"),
                "rate_source": planes.get("execution", {}).get("rate_source"),
                "posterior_consumed": pi.get("consumed_by_simulator"),
                "structural_updated": bool(pi.get("structural", {}).get("posterior") and
                                           pi["structural"]["posterior"] != pi["structural"].get("prior")),
                "posterior_hash": art.get("posterior_hash"),
                "reproducible_hash": art.get("posterior_hash") == art2.get("posterior_hash"),
                "reproducible_terminal": res.raw_probability == res2.raw_probability,
                "warnings": pi.get("warnings", []), "latency_s": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001
            rec.update({"status": "harness_error", "error": f"{type(e).__name__}: {e}"[:200],
                        "latency_s": round(time.time() - t0, 1)})
        rows.append(rec)
        (OUT / "live_validation.json").write_text(json.dumps({"rows": rows}, indent=2))   # incremental
        print(f"[{domain:11s}] {q[:52]:52s} consumed={rec.get('posterior_consumed')} "
              f"shift={rec.get('posterior_shift')} src={rec.get('rate_source')} "
              f"repro={rec.get('reproducible_hash')} grade={rec.get('support_grade')}")
    return _aggregate(rows, meter)


def _aggregate(rows, meter):
    ok = [r for r in rows if r.get("status", "").startswith("completed")]
    consumed = [r for r in ok if r.get("posterior_consumed")]
    shifts = [abs(r["posterior_shift"]) for r in consumed if r.get("posterior_shift") is not None]
    effects = [r["terminal_effect"] for r in consumed if r.get("terminal_effect") is not None]
    n = max(1, len(ok))
    agg = {
        "n_questions": len(rows), "n_completed": len(ok), "n_harness_error": len(rows) - len(ok),
        "no_abstention_rate": round(sum(1 for r in ok if r.get("has_forecast")) / n, 3),
        "posterior_consumed_rate": round(len(consumed) / n, 3),
        "mean_prior_to_posterior_shift": round(sum(shifts) / len(shifts), 4) if shifts else None,
        "max_shift": round(max(shifts), 4) if shifts else None,
        "mean_terminal_effect_vs_ignored": round(sum(effects) / len(effects), 4) if effects else None,
        "structural_updated_rate": round(sum(1 for r in ok if r.get("structural_updated")) / n, 3),
        "reproducible_hash_rate": round(sum(1 for r in ok if r.get("reproducible_hash")) / n, 3),
        "dependence_collapse_examples": [
            {"q": r["question"][:40], "claims": r.get("n_included_claims"),
             "effective": r.get("n_effective_observations")} for r in consumed[:6]],
        "llm_calls": meter["calls"],
        "gates": {}}
    agg["gates"] = {
        "all_coherent_questions_forecast": agg["no_abstention_rate"] == 1.0,
        "posterior_consumed_when_evidence_present": agg["posterior_consumed_rate"] >= 0.5,
        "evidence_moves_the_number": (agg["mean_prior_to_posterior_shift"] or 0) > 0.02,
        "posterior_changes_terminal_vs_ignored": (agg["mean_terminal_effect_vs_ignored"] or 0) > 0.01,
        "within_run_reproducible": agg["reproducible_hash_rate"] == 1.0}
    agg["all_gates_pass"] = all(agg["gates"].values())
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    agg = run(limit=args.limit, seed=args.seed)
    payload = json.loads((OUT / "live_validation.json").read_text()) if (OUT / "live_validation.json").exists() else {}
    payload["aggregate"] = agg
    payload["retrieval_date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "live_validation.json").write_text(json.dumps(payload, indent=2))
    print("\nAGGREGATE:", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
