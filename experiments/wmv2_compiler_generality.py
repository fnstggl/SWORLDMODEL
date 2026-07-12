"""Phase 1 compiler generality validation — the FIRST real-LLM exercise of the general path.

The audit's central finding: `compile_world` had never run against a real LLM anywhere; every benchmark
number came from hand-built worlds. This harness fixes that. It feeds ≥100 held-out natural-language
questions across 16 domains through the ONE compiler (real DeepSeek), then attempts full end-to-end
execution (materialize → rollout → terminal projection). NO scripted target plans: the compiler generates
its own plan for every question; we score the plans structurally + with a jury rubric on a stratified
sample.

Metrics (all automated from real runs):
  compile_success  — parsed, typed outcome contract, ≥1 executable registry mechanism accepted
  abstention       — CompileAbstention rate + reason histogram (abstention is a valid, desired outcome)
  executes_e2e     — run_from_plan produced a native terminal distribution without crashing
  mechanism_validity — every accepted mechanism resolves to an executable operator (should be 100% by
                       construction post-Tier-A; measured to prove it)
  readout_resolves — the plan's terminal readout binds to the materialized world
  omissions_logged — dropped fields/relations/rules recorded (loud-failure check)
  materialize_abstain — MaterializeAbstention rate (dangling readout / no executable mechanism)
  unsupported_precision — LLM-proposed entity fields entering as 'inferred' not 'observed' (provenance)

Jury rubric (stratified sample, separate cheap LLM, NEVER writes the plan): rates outcome-contract
correctness, actor relevance, mechanism appropriateness, missing high-sensitivity variables — a validation
aid, not production authority.

Resumable: per-question results cached under experiments/results/compiler_generality/. Deterministic given
the cache. Metered (cost + latency).
Run: DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_compiler_generality
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_compiler_generality.json"
CACHE = Path("experiments/results/compiler_generality")

# ---- ≥100 held-out NL questions across 16 domains. Questions only; NO plans. as-of before any outcome. ----
# (domain, question, as_of, horizon)
QUESTIONS = [
    # individual messaging
    ("messaging", "Will my manager reply to the budget-approval email I sent this morning by end of week?", "2023-05-01", "2023-05-08"),
    ("messaging", "If I send a follow-up nudge tomorrow, will the vendor respond within 48 hours?", "2023-06-10", "2023-06-13"),
    ("messaging", "Will the recruiter answer my scheduling question before the weekend?", "2023-03-15", "2023-03-19"),
    ("messaging", "Is my co-founder likely to read and reply to the long strategy memo I just sent?", "2023-08-01", "2023-08-05"),
    ("messaging", "Will the professor respond to my request for a recommendation letter within two weeks?", "2023-09-01", "2023-09-15"),
    ("messaging", "Will the support team escalate my ticket after my second message?", "2023-07-20", "2023-07-25"),
    # negotiation
    ("negotiation", "Will the buyer accept our counteroffer of $4.2M for the office building?", "2023-04-01", "2023-05-01"),
    ("negotiation", "In the salary negotiation, will the candidate accept an offer 8% below their stated ask?", "2023-06-01", "2023-06-15"),
    ("negotiation", "Will the union and management reach a wage agreement before the contract expires?", "2023-08-15", "2023-09-30"),
    ("negotiation", "Will the two companies agree on licensing terms within the next month?", "2023-05-10", "2023-06-10"),
    ("negotiation", "Will the divorcing parties settle the property split out of court?", "2023-02-01", "2023-06-01"),
    ("negotiation", "Will the supplier agree to a 15% volume discount if we double our order?", "2023-10-01", "2023-10-20"),
    # organizational approval
    ("organizational_decision", "Will the engineering VP approve the request to hire two more backend engineers this quarter?", "2023-07-01", "2023-09-30"),
    ("organizational_decision", "Will the finance department sign off on the new marketing budget increase?", "2023-04-05", "2023-04-30"),
    ("organizational_decision", "Will the product team greenlight the proposed feature for the next release?", "2023-06-01", "2023-06-30"),
    ("organizational_decision", "Will the CEO approve the acquisition proposal presented at the leadership offsite?", "2023-09-10", "2023-10-10"),
    ("organizational_decision", "Will the IT security team approve the third-party integration request?", "2023-03-01", "2023-03-21"),
    ("organizational_decision", "Will the department head authorize remote work for the new hire?", "2023-05-15", "2023-05-29"),
    # election
    ("election", "Will the incumbent mayor win re-election in the upcoming city vote?", "2023-09-01", "2023-11-07"),
    ("election", "Will voter turnout in the special election exceed 40%?", "2023-06-01", "2023-08-01"),
    ("election", "Will the challenger flip the swing district in the general election?", "2023-08-01", "2023-11-05"),
    ("election", "Will the ballot referendum on the transit levy pass?", "2023-07-01", "2023-11-07"),
    ("election", "Will the third-party candidate reach 10% of the vote?", "2023-09-15", "2023-11-05"),
    ("election", "Will the party retain its majority in the state legislature?", "2023-08-20", "2023-11-07"),
    # legislation
    ("legislation", "Will the infrastructure bill pass the Senate before the recess?", "2023-06-01", "2023-08-01"),
    ("legislation", "Will the committee advance the data-privacy bill to a floor vote?", "2023-04-01", "2023-06-01"),
    ("legislation", "Will the amendment to the housing bill be adopted?", "2023-05-10", "2023-06-10"),
    ("legislation", "Will the appropriations bill clear both chambers before the fiscal deadline?", "2023-08-15", "2023-09-30"),
    ("legislation", "Will the governor sign the education-funding bill into law?", "2023-07-01", "2023-07-31"),
    ("legislation", "Will the minimum-wage bill survive the veto override attempt?", "2023-09-01", "2023-10-01"),
    # acquisition
    ("acquisition", "Will the proposed merger between the two regional banks be completed?", "2023-05-01", "2023-12-31"),
    ("acquisition", "Will regulators approve the tech acquisition without requiring divestitures?", "2023-06-01", "2023-11-30"),
    ("acquisition", "Will the target company's board accept the hostile takeover bid?", "2023-04-01", "2023-06-01"),
    ("acquisition", "Will the private-equity firm close the buyout by year end?", "2023-07-01", "2023-12-31"),
    ("acquisition", "Will the acquisition be blocked on antitrust grounds?", "2023-08-01", "2024-02-01"),
    # product launch
    ("product_launch", "Will the new smartphone model launch on the announced date?", "2023-08-01", "2023-10-01"),
    ("product_launch", "Will the streaming service hit 1M subscribers within three months of launch?", "2023-05-01", "2023-08-01"),
    ("product_launch", "Will the electric vehicle begin deliveries this quarter?", "2023-07-01", "2023-09-30"),
    ("product_launch", "Will the app reach the top 10 in its category within a month of release?", "2023-06-01", "2023-07-01"),
    ("product_launch", "Will the game ship without a major delay?", "2023-09-01", "2023-11-01"),
    # social-media diffusion
    ("social_media_diffusion", "Will the campaign hashtag trend nationally within 24 hours of launch?", "2023-05-01", "2023-05-02"),
    ("social_media_diffusion", "Will the product-announcement post exceed 100k shares in a week?", "2023-06-01", "2023-06-08"),
    ("social_media_diffusion", "Will the viral video reach 5 million views in three days?", "2023-07-10", "2023-07-13"),
    ("social_media_diffusion", "Will the petition gather 50,000 signatures online within a month?", "2023-04-01", "2023-05-01"),
    ("social_media_diffusion", "Will the influencer's endorsement drive a measurable spike in brand mentions?", "2023-08-01", "2023-08-08"),
    # protest / strike
    ("protest", "Will the planned climate march draw more than 10,000 participants?", "2023-09-01", "2023-09-20"),
    ("protest", "Will the teachers' strike spread to neighboring districts within two weeks?", "2023-05-01", "2023-05-15"),
    ("protest", "Will the factory workers walk out over the pay dispute?", "2023-06-01", "2023-06-30"),
    ("strike", "Will the transit strike be resolved before Monday's commute?", "2023-07-14", "2023-07-17"),
    ("strike", "Will the writers' strike end before the fall season?", "2023-06-01", "2023-09-15"),
    # court / regulatory
    ("court_ruling", "Will the appeals court uphold the lower court's ruling in the antitrust case?", "2023-05-01", "2023-10-01"),
    ("court_ruling", "Will the judge grant the injunction to halt the pipeline construction?", "2023-04-01", "2023-05-01"),
    ("court_ruling", "Will the regulator impose a fine in the data-breach investigation?", "2023-06-01", "2023-12-01"),
    ("court_ruling", "Will the patent-infringement suit be dismissed on summary judgment?", "2023-07-01", "2023-11-01"),
    ("court_ruling", "Will the environmental agency approve the drilling permit?", "2023-08-01", "2023-12-31"),
    # fundraising
    ("fundraising", "Will the startup close its Series B round within the quarter?", "2023-06-01", "2023-09-30"),
    ("fundraising", "Will the nonprofit hit its $2M annual campaign goal by December?", "2023-01-01", "2023-12-31"),
    ("fundraising", "Will the crowdfunding project reach its funding target before the deadline?", "2023-05-01", "2023-06-01"),
    ("fundraising", "Will the candidate out-raise their opponent this reporting period?", "2023-07-01", "2023-09-30"),
    # coalition formation
    ("coalition", "Will the three parties form a governing coalition after the election?", "2023-09-01", "2023-11-01"),
    ("coalition", "Will the swing senators join the bipartisan compromise bloc?", "2023-06-01", "2023-07-15"),
    ("coalition", "Will the industry groups unite behind the single lobbying position?", "2023-05-01", "2023-06-15"),
    # market reaction
    ("market", "Will the stock rise more than 5% the day after the earnings call?", "2023-07-25", "2023-07-27"),
    ("market", "Will the central bank raise interest rates at the next meeting?", "2023-06-01", "2023-07-27"),
    ("market", "Will the commodity price fall below its six-month low this month?", "2023-08-01", "2023-08-31"),
    # reputation crisis
    ("reputation_crisis", "Will the company's CEO resign within a month of the scandal breaking?", "2023-05-01", "2023-06-01"),
    ("reputation_crisis", "Will the brand recover its pre-crisis social sentiment within a quarter?", "2023-06-01", "2023-09-01"),
    ("reputation_crisis", "Will the public apology stem the customer boycott?", "2023-07-01", "2023-08-01"),
    # best-action recommendation
    ("best_action", "Should we send the discount offer now or wait until the customer's renewal date to maximize retention?", "2023-06-01", "2023-07-01"),
    ("best_action", "What sequence of concessions should we offer to close the deal before quarter end?", "2023-08-01", "2023-09-30"),
    ("best_action", "Which coalition partner should we approach first to build a majority?", "2023-05-01", "2023-06-15"),
    ("best_action", "Should we launch the campaign now or after the competitor's announcement?", "2023-07-01", "2023-08-15"),
]


def _expand_to_100(base):
    """Deterministically extend the base set past 100 by paraphrasing horizons/quantities — still
    scenario-specific NL questions, no plans. Kept explicit and auditable."""
    extra = []
    variants = [
        ("messaging", "Will the client confirm the meeting time I proposed by tomorrow afternoon?", "2023-10-01", "2023-10-03"),
        ("negotiation", "Will the landlord accept a rent reduction in exchange for a longer lease?", "2023-09-01", "2023-09-20"),
        ("organizational_decision", "Will the board approve the stock buyback proposal at the next meeting?", "2023-06-15", "2023-07-15"),
        ("election", "Will the incumbent governor survive the recall vote?", "2023-08-01", "2023-11-01"),
        ("legislation", "Will the tax-reform package pass before the session ends?", "2023-05-01", "2023-06-30"),
        ("acquisition", "Will the airline merger clear its final regulatory hurdle?", "2023-07-01", "2023-12-01"),
        ("product_launch", "Will the wearable device meet its holiday-season ship date?", "2023-09-01", "2023-12-01"),
        ("social_media_diffusion", "Will the meme cross over from one platform to another within a week?", "2023-06-01", "2023-06-08"),
        ("protest", "Will the sit-in force the university to reopen negotiations?", "2023-04-01", "2023-04-21"),
        ("court_ruling", "Will the class-action settlement receive court approval?", "2023-05-01", "2023-09-01"),
        ("fundraising", "Will the museum's capital campaign reach 75% of goal by mid-year?", "2023-01-01", "2023-06-30"),
        ("coalition", "Will the trade bloc admit the new member state this year?", "2023-03-01", "2023-12-31"),
        ("market", "Will the IPO price above its indicated range?", "2023-08-01", "2023-08-20"),
        ("reputation_crisis", "Will the airline's on-time reputation recover after the meltdown?", "2023-07-01", "2023-10-01"),
        ("best_action", "Should the campaign spend its final ad budget on turnout or persuasion?", "2023-10-01", "2023-11-05"),
        ("messaging", "Will the busy executive delegate my request to an assistant?", "2023-06-01", "2023-06-05"),
        ("negotiation", "Will the freelancer accept a milestone-based payment structure?", "2023-08-01", "2023-08-15"),
        ("organizational_decision", "Will the hospital committee approve the new triage protocol?", "2023-05-01", "2023-06-01"),
        ("legislation", "Will the rider be stripped from the bill in conference?", "2023-06-01", "2023-07-01"),
        ("election", "Will the primary go to a runoff?", "2023-05-01", "2023-06-01"),
        ("acquisition", "Will the founders accept the all-stock acquisition offer?", "2023-06-01", "2023-08-01"),
        ("product_launch", "Will the beta convert at least 20% of testers to paid?", "2023-07-01", "2023-08-15"),
        ("social_media_diffusion", "Will the recall notice reach affected customers within 72 hours?", "2023-09-01", "2023-09-05"),
        ("protest", "Will the boycott cut quarterly sales by a measurable margin?", "2023-06-01", "2023-09-30"),
        ("court_ruling", "Will the regulator open a formal probe into the merger?", "2023-05-01", "2023-07-01"),
        ("fundraising", "Will the challenger clear $1M in small-dollar donations this quarter?", "2023-07-01", "2023-09-30"),
        ("coalition", "Will the holdout faction join the leadership's whip count?", "2023-06-01", "2023-06-30"),
        ("market", "Will the bond yield invert relative to the shorter maturity this week?", "2023-08-01", "2023-08-08"),
        ("reputation_crisis", "Will the sponsor withdraw after the controversy?", "2023-05-01", "2023-06-01"),
        ("best_action", "Which message framing should we A/B test first to lift open rates?", "2023-06-01", "2023-06-15"),
        ("strike", "Will the dockworkers ratify the tentative agreement?", "2023-07-01", "2023-08-01"),
        ("organizational_decision", "Will the grant committee fund the pilot program?", "2023-04-01", "2023-05-15"),
    ]
    return base + variants


def _digest(q, as_of, horizon):
    return hashlib.sha1(f"{q}|{as_of}|{horizon}".encode()).hexdigest()[:12]


def run(limit, jury_sample):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.grounding import parse_json
    from swm.world_model_v2.compiler import CompileAbstention, compile_world
    from swm.world_model_v2.materialize import MaterializeAbstention, build_world, run_from_plan
    from swm.world_model_v2.transitions import _OPERATORS
    from swm.world_model_v2 import registry as reg

    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    reg.load_registry()                                       # mirror production families into the vocabulary
    questions = _expand_to_100(QUESTIONS)[:limit] if limit else _expand_to_100(QUESTIONS)
    meter = {"calls": 0, "tokens": 0}
    llm = default_chat_fn(system="You are the world-slice compiler proposal stage. Reply ONLY JSON.",
                          max_tokens=1400, temperature=0.2)
    jury = default_chat_fn(system="You are a careful reviewer. Reply ONLY compact JSON.",
                           max_tokens=300, temperature=0.2)
    if llm is None:
        raise SystemExit("needs DEEPSEEK_API_KEY")

    def call(fn, prompt):
        txt = fn(prompt)
        meter["calls"] += 1
        meter["tokens"] += (len(prompt) + len(txt or "")) // 4
        return txt

    rows = []
    for i, (domain, q, as_of, horizon) in enumerate(questions):
        cache_f = CACHE / f"{_digest(q, as_of, horizon)}.json"
        if cache_f.exists():
            rows.append(json.loads(cache_f.read_text()))
            continue
        rec = {"domain": domain, "question": q, "as_of": as_of, "horizon": horizon,
               "compiled": False, "abstained": False, "abstain_reason": "", "executed_e2e": False,
               "materialize_abstained": False, "n_accepted_mech": 0, "n_rejected_mech": 0,
               "mechanisms": [], "all_mech_executable": None, "readout_var": "", "readout_resolves": None,
               "n_entities": 0, "n_institutions": 0, "n_populations": 0, "n_latents": 0,
               "provenance_ok": None, "n_omissions": 0, "terminal_distribution": None, "error": ""}
        t_q = time.time()
        try:
            plan = compile_world(q, llm=lambda p: call(llm, p), evidence="", as_of=as_of, horizon=horizon)
            rec["compiled"] = True
            rec["n_accepted_mech"] = len(plan.accepted_mechanisms)
            rec["n_rejected_mech"] = len(plan.rejected_mechanisms)
            rec["mechanisms"] = [m["mech_id"] for m in plan.accepted_mechanisms]
            rec["all_mech_executable"] = all(m["operator"] in _OPERATORS for m in plan.accepted_mechanisms)
            rec["readout_var"] = plan.outcome_contract.readout_var
            rec["n_entities"] = len(plan.entities)
            rec["n_institutions"] = len(plan.institutions)
            rec["n_populations"] = len(plan.populations)
            rec["n_latents"] = len(plan.latents)
            # provenance check: build the world and confirm entity fields are 'inferred', not fabricated
            w = build_world(plan)
            rec["n_omissions"] = len(w.omissions)
            statuses = [sf.prov.status for e in w.entities.values() for sf in e.fields.values()
                        if hasattr(sf, "prov")]
            rec["provenance_ok"] = all(s != "observed" for s in statuses)   # no fabricated observation
            rec["readout_resolves"] = (plan.outcome_contract.readout_var in w.quantities
                                       or plan.outcome_contract.readout_var.split(".")[0] in w.entities)
            # end-to-end execution attempt
            try:
                result, branches = run_from_plan(plan, n_particles=12, seed=7)
                rec["executed_e2e"] = True
                rec["terminal_distribution"] = result.get("distribution") or result.get("quantiles")
            except MaterializeAbstention as e:
                rec["materialize_abstained"] = True
                rec["abstain_reason"] = f"materialize: {str(e)[:150]}"
        except CompileAbstention as e:
            rec["abstained"] = True
            rec["abstain_reason"] = str(e)[:200]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        rec["latency_s"] = round(time.time() - t_q, 2)
        cache_f.write_text(json.dumps(rec, indent=1, default=str))
        rows.append(rec)
        print(f"  [{i+1}/{len(questions)}] {domain:22s} compiled={rec['compiled']} "
              f"e2e={rec['executed_e2e']} abstain={rec['abstained']} mech={rec['mechanisms'][:3]}",
              flush=True)

    # ---- jury rubric on a stratified sample (validation aid; never writes plans) ----
    jury_rows = []
    by_domain = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    sample = []
    for d, rs in by_domain.items():
        sample += [r for r in rs if r["compiled"]][:max(1, jury_sample // max(1, len(by_domain)))]
    for r in sample[:jury_sample]:
        prompt = (f"A world-model compiler produced this plan for the question below. Rate 0-1 each:\n"
                  f"QUESTION: {r['question']} (as-of {r['as_of']}, horizon {r['horizon']})\n"
                  f"OUTCOME readout variable: {r['readout_var']}\n"
                  f"ACCEPTED MECHANISMS: {r['mechanisms']}\n"
                  f"#entities={r['n_entities']} #institutions={r['n_institutions']} "
                  f"#populations={r['n_populations']} #latents={r['n_latents']}\n"
                  f'Return ONLY JSON: {{"outcome_contract_ok": <0..1>, "actors_relevant": <0..1>, '
                  f'"mechanisms_appropriate": <0..1>, "missing_high_sensitivity_var": <0..1>, '
                  f'"note": "<12 words>"}}')
        j = parse_json(call(jury, prompt)) or {}
        jury_rows.append({"question": r["question"][:80], "domain": r["domain"], **j})

    # ---- aggregate ----
    n = len(rows)
    comp = [r for r in rows if r["compiled"]]
    agg = {
        "n_questions": n, "n_domains": len(by_domain),
        "compile_success_rate": round(len(comp) / n, 3),
        "abstention_rate": round(sum(r["abstained"] for r in rows) / n, 3),
        "executes_e2e_rate": round(sum(r["executed_e2e"] for r in rows) / n, 3),
        "materialize_abstain_rate": round(sum(r["materialize_abstained"] for r in rows) / n, 3),
        "error_rate": round(sum(bool(r["error"]) for r in rows) / n, 3),
        "mechanism_validity": round(sum(1 for r in comp if r["all_mech_executable"]) / max(1, len(comp)), 3),
        "readout_resolves_rate": round(sum(1 for r in comp if r["readout_resolves"]) / max(1, len(comp)), 3),
        "provenance_ok_rate": round(sum(1 for r in comp if r["provenance_ok"]) / max(1, len(comp)), 3),
        "mean_mechanisms_per_plan": round(sum(len(r["mechanisms"]) for r in comp) / max(1, len(comp)), 2),
        "mean_entities_per_plan": round(sum(r["n_entities"] for r in comp) / max(1, len(comp)), 2),
        "mean_latents_per_plan": round(sum(r["n_latents"] for r in comp) / max(1, len(comp)), 2),
    }
    abstain_reasons = {}
    for r in rows:
        if r["abstained"] or r["materialize_abstained"]:
            key = r["abstain_reason"][:60]
            abstain_reasons[key] = abstain_reasons.get(key, 0) + 1
    mech_hist = {}
    for r in comp:
        for m in r["mechanisms"]:
            mech_hist[m] = mech_hist.get(m, 0) + 1
    jury_agg = {}
    if jury_rows:
        for k in ("outcome_contract_ok", "actors_relevant", "mechanisms_appropriate",
                  "missing_high_sensitivity_var"):
            vals = [float(j[k]) for j in jury_rows if isinstance(j.get(k), (int, float))]
            jury_agg[k] = round(sum(vals) / len(vals), 3) if vals else None

    out = {"aggregate": agg, "per_domain": {d: {"n": len(rs),
            "compile_rate": round(sum(r["compiled"] for r in rs) / len(rs), 2),
            "e2e_rate": round(sum(r["executed_e2e"] for r in rs) / len(rs), 2)}
            for d, rs in by_domain.items()},
           "abstention_reasons": dict(sorted(abstain_reasons.items(), key=lambda kv: -kv[1])),
           "mechanism_histogram": dict(sorted(mech_hist.items(), key=lambda kv: -kv[1])),
           "jury_rubric": {"n_reviewed": len(jury_rows), "means": jury_agg, "rows": jury_rows[:20]},
           "forensic_examples": [r for r in rows if r["executed_e2e"]][:3]
                                + [r for r in rows if r["abstained"]][:2],
           "_meta": {"llm_calls": meter["calls"], "llm_tokens_est": meter["tokens"],
                     "est_cost_usd": round(meter["tokens"] * (0.27e-6 + 1.10e-6) / 2, 4),
                     "model": "deepseek-chat", "runtime_s": round(time.time() - t0, 1),
                     "note": "FIRST real-LLM exercise of compile_world→materialize→rollout; no scripted "
                             "plans; benchmark adapters absent (pure NL questions)"}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print("\nAGGREGATE:", json.dumps(agg, indent=1))
    print("JURY:", jury_agg)
    print(f"wrote {RESULT} (calls={meter['calls']}, ~${out['_meta']['est_cost_usd']}, "
          f"{out['_meta']['runtime_s']}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jury-sample", type=int, default=16)
    a = ap.parse_args()
    run(a.limit, a.jury_sample)
