"""Phase 3B — LOCKED final test (Parts I/J/K).

A completely separate, event-family- and temporally-disjoint held-out set of resolved questions, NEVER used to
fit or select anything. Run ONCE, after all repair parameters are frozen (`repair_params.json`). Produces the
final paired comparison: Phase-2 vs REPAIRED Phase-3 (and vs current Phase-3), plus every baseline, scored
against realized outcomes with paired-bootstrap CIs and per-domain breakdowns.

Each question runs the real production path once (phase3 arm freezes plan+bundle+tags), then the phase2 arm
reuses them; the repaired forecast is computed by the FROZEN production repair module from the live tags. No
parameter is chosen here.

Leakage discipline vs the diagnostic set: DISJOINT event families (different institutions, entities, contests),
mostly different countries/leagues, and the manual as-of leakage audit is re-run on a stratified sample.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

OUT = Path("experiments/results/phase3b")
ART = OUT / "locked_test.json"

# (qid, question, as_of, horizon, domain, outcome, family, note). Families are DISJOINT from the diagnostic set.
QUESTIONS = [
    # elections — non-US-2024, non-UK-2024 families
    ("modi_pm",      "Will Narendra Modi remain Prime Minister of India after the 2024 general election?", "2024-05-20", "2024-06-09", "elections", 1, "india_ge", "NDA won; Modi sworn in Jun 2024."),
    ("afd_first_de", "Will the AfD finish first in Germany in the 2024 European Parliament election?", "2024-05-25", "2024-06-09", "elections", 0, "eu_parl", "CDU/CSU first; AfD second."),
    ("fr_nfp",       "Will the New Popular Front win the most seats in the 2024 French legislative election?", "2024-06-28", "2024-07-07", "elections", 1, "france_leg", "NFP won most seats Jul 7 2024."),
    ("mx_sheinbaum", "Will Claudia Sheinbaum win the 2024 Mexican presidential election?", "2024-05-25", "2024-06-02", "elections", 1, "mexico_pres", "Sheinbaum won Jun 2 2024."),
    ("jp_ldp_maj",   "Will the Liberal Democratic Party keep a single-party majority in Japan's 2024 general election?", "2024-10-20", "2024-10-27", "elections", 0, "japan_ge", "LDP lost its majority Oct 27 2024."),
    ("ven_maduro",   "Will Venezuela's electoral authority declare Nicolas Maduro the winner of the July 2024 presidential election?", "2024-07-20", "2024-07-29", "elections", 1, "venezuela_pres", "CNE declared Maduro winner Jul 29 2024 (disputed)."),
    # central banks — non-FOMC institutions
    ("ecb_jun24",    "Will the European Central Bank cut interest rates at its June 2024 meeting?", "2024-06-01", "2024-06-06", "econ", 1, "ecb", "ECB cut 25bp Jun 6 2024."),
    ("boe_aug24",    "Will the Bank of England cut its policy interest rate at its August 2024 meeting?", "2024-07-25", "2024-08-01", "econ", 1, "boe", "BoE cut 25bp Aug 1 2024."),
    ("boj_jul24",    "Will the Bank of Japan raise interest rates at its July 2024 meeting?", "2024-07-20", "2024-07-31", "econ", 1, "boj", "BoJ hiked Jul 31 2024."),
    # macro / markets — non-index-6000, non-btc-100k families
    ("cpi_sep24",    "Will US headline CPI inflation be below 3 percent year-over-year in the September 2024 report?", "2024-09-01", "2024-10-10", "macro", 1, "us_cpi", "Sep 2024 CPI 2.4% YoY."),
    ("unemp_jul24",  "Will the US unemployment rate be at or above 4.0 percent in the July 2024 jobs report?", "2024-07-15", "2024-08-02", "macro", 1, "us_jobs", "Jul 2024 unemployment 4.3%."),
    ("gold_2500",    "Will gold close above 2500 US dollars per ounce by the end of 2024?", "2024-07-01", "2024-12-31", "finance", 1, "gold", "Gold passed $2500 in Aug 2024."),
    ("us10y_5pct",   "Will the US 10-year Treasury yield exceed 5 percent at any point in 2024?", "2024-05-01", "2024-12-31", "finance", 0, "ust_yield", "10y peaked ~4.7% in 2024."),
    ("eth_5000",     "Will Ethereum's price exceed 5000 US dollars in 2024?", "2024-05-01", "2024-12-31", "finance", 0, "eth_threshold", "ETH peaked ~$4100 in 2024."),
    ("nvda_3t",      "Will Nvidia's market capitalization exceed 3 trillion US dollars in 2024?", "2024-05-01", "2024-12-31", "finance", 1, "mktcap", "Nvidia passed $3T Jun 2024."),
    # tech / regulatory — non-openai, non-apple families
    ("eth_etf",      "Will the US SEC approve spot Ethereum exchange-traded funds in 2024?", "2024-05-01", "2024-12-31", "tech", 1, "sec_etf", "SEC approved spot ETH ETFs May 2024."),
    ("grok2",        "Will xAI release Grok-2 in 2024?", "2024-06-01", "2024-12-31", "tech", 1, "xai_release", "Grok-2 released Aug 2024."),
    ("llama3",       "Will Meta release Llama 3 in 2024?", "2024-03-01", "2024-12-31", "tech", 1, "meta_release", "Llama 3 released Apr 2024."),
    ("tiktok_law",   "Will the United States enact a law requiring the sale or ban of TikTok in 2024?", "2024-03-01", "2024-12-31", "tech", 1, "tiktok", "Divest-or-ban law signed Apr 24 2024."),
    ("tesla_robotaxi","Will Tesla unveil a dedicated robotaxi vehicle in 2024?", "2024-08-01", "2024-12-31", "tech", 1, "tesla", "Cybercab unveiled Oct 10 2024."),
    # geopolitics — non-gaza, non-syria, non-ruua families
    ("iran_israel",  "Will Iran and Israel exchange direct military strikes on each other's territory in 2024?", "2024-03-01", "2024-12-31", "geopolitics", 1, "iran_israel", "Direct strikes in Apr and Oct 2024."),
    ("taiwan",       "Will China conduct a military invasion or blockade of Taiwan in 2024?", "2024-01-15", "2024-12-31", "geopolitics", 0, "taiwan", "No invasion or blockade in 2024."),
    ("guyana",       "Will Venezuela militarily invade Guyana's Essequibo region in 2024?", "2024-01-15", "2024-12-31", "geopolitics", 0, "guyana", "No invasion in 2024."),
    # sports — non-cricket, non-UCL families
    ("nba_celtics",  "Will the Boston Celtics win the 2024 NBA Finals?", "2024-06-01", "2024-06-24", "sports", 1, "nba", "Celtics won Jun 17 2024."),
    ("euro_spain",   "Will Spain win UEFA Euro 2024?", "2024-07-10", "2024-07-14", "sports", 1, "euro", "Spain won Jul 14 2024."),
    ("f1_verstappen","Will Max Verstappen win the 2024 Formula 1 World Drivers' Championship?", "2024-10-01", "2024-12-08", "sports", 1, "f1", "Verstappen clinched the 2024 title."),
    ("masters_schef","Will Scottie Scheffler win the 2024 Masters golf tournament?", "2024-04-08", "2024-04-14", "sports", 1, "golf", "Scheffler won Apr 14 2024."),
    ("nhl_panthers", "Will the Florida Panthers win the 2024 Stanley Cup?", "2024-06-01", "2024-06-24", "sports", 1, "nhl", "Panthers won Jun 24 2024."),
    # corporate / other — non-nvda-split families
    ("disney_peltz", "Will Nelson Peltz's Trian win a seat on Disney's board at the 2024 shareholder meeting?", "2024-03-15", "2024-04-03", "finance", 0, "disney_proxy", "Disney defeated Trian Apr 3 2024."),
    ("paramount",    "Will Paramount Global agree to a merger or sale in 2024?", "2024-05-01", "2024-12-31", "finance", 1, "ma", "Skydance merger agreed Jul 2024."),
    ("boeing_ceo",   "Will Boeing name a new chief executive officer in 2024?", "2024-05-01", "2024-12-31", "finance", 1, "boeing_ceo", "Kelly Ortberg named CEO Aug 2024."),
    # science / space — non-starship families
    ("moon_crew",    "Will a crewed mission land humans on the Moon in 2024?", "2024-01-01", "2024-12-31", "science", 0, "moon", "No crewed lunar landing in 2024."),
    ("starliner",    "Will Boeing's Starliner return its crew from the ISS aboard Starliner in 2024?", "2024-07-01", "2024-12-31", "science", 0, "starliner", "Crew returned on Dragon in 2025; Starliner returned uncrewed."),
    ("hurricane_c5", "Will the 2024 Atlantic hurricane season produce a Category 5 hurricane?", "2024-06-15", "2024-11-30", "science", 1, "hurricane", "Hurricane Beryl reached Cat 5 in Jul 2024."),
    ("cannabis",     "Will the United States enact federal legalization of cannabis in 2024?", "2024-03-01", "2024-12-31", "politics", 0, "cannabis", "No federal legalization in 2024."),
]

_EPS = 1e-6


def _clip(p):
    return min(1 - _EPS, max(_EPS, p))


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = _clip(p); return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    llm0 = default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)
    if llm0 is None:
        return None, None
    meter = {"calls": 0}

    def llm(p):
        meter["calls"] += 1; return llm0(p)
    return llm, meter


def _tag_rows(bundle, tags):
    ctext = {}
    for c in bundle.included_claims():
        ctext[c.get("claim_id")] = c.get("claim_class")
    rows = []
    for t in tags:
        rows.append({"claim_id": t.claim_id, "outcome_direction": t.outcome_direction, "strength": t.strength,
                     "reliability": float(t.reliability), "is_strategic": t.is_strategic,
                     "dependence_group": t.dependence_group,
                     "supports_hypotheses": list(t.supports_hypotheses),
                     "opposes_hypotheses": list(t.opposes_hypotheses)})
    return rows


def run_one(q, llm, cfg, params, seed=0):
    from swm.world_model_v2.phase3_pipeline import simulate_with_posterior
    from swm.world_model_v2.phase3b_repair import repaired_from_capture_row
    qid, question, as_of, horizon, domain, outcome, family, note = q
    rec = {"qid": qid, "question": question, "as_of": as_of, "horizon": horizon, "domain": domain,
           "family": family, "outcome": outcome, "resolution_note": note}
    t0 = time.time()
    try:
        res3, art = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                            config=cfg, consume_posterior=True)
        plan, bundle, tags = art["plan"], art["bundle"], art["tags"]
        res2, _ = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                          config=cfg, consume_posterior=False, plan=plan, bundle=bundle, tags=tags)
        pi = res3.posterior_inference or {}
        prov = (pi.get("prior_provenance") or {}).get("outcome_rate", {})
        row = {"qid": qid, "question": question, "as_of": as_of, "domain": domain, "outcome": outcome,
               "outcome_lean": str((plan.provenance or {}).get("outcome_lean", "neutral")),
               "prior": {"alpha": prov.get("alpha"), "beta": prov.get("beta")},
               "tags": _tag_rows(bundle, tags), "p_phase2": res2.raw_probability}
        rep = repaired_from_capture_row(row, params)
        rec.update({
            "status": res3.simulation_status,
            "arms": {
                "prior_only": (pi.get("outcome_rate") or {}).get("prior_mean"),
                "phase2": res2.raw_probability,
                "phase3_current": res3.raw_probability,
                "phase3_repaired": rep["repaired_p"]},
            "repair": rep,
            "n_effective_observations": pi.get("n_effective_observations"),
            "n_included_claims": len(bundle.included_claim_ids),
            "posterior_hash": art.get("posterior_hash"),
            "latency_s": round(time.time() - t0, 1)})
    except Exception as e:  # noqa: BLE001
        rec.update({"status": "harness_error", "error": f"{type(e).__name__}: {e}"[:200], "arms": {},
                    "latency_s": round(time.time() - t0, 1)})
    return rec


ARMS = ["prior_only", "phase2", "phase3_current", "phase3_repaired"]


def _score(rows, arm):
    pts = [(r["arms"][arm], r["outcome"]) for r in rows if r.get("arms", {}).get(arm) is not None]
    if not pts:
        return {"n": 0}
    b = sum(brier(p, y) for p, y in pts) / len(pts)
    l = sum(logloss(p, y) for p, y in pts) / len(pts)
    d = sum(1 for p, y in pts if (p > 0.5) == (y == 1)) / len(pts)
    bins = {}
    for p, y in pts:
        bins.setdefault(min(9, int(p * 10)), []).append((p, y))
    ece = sum((len(v) / len(pts)) * abs(sum(p for p, _ in v) / len(v) - sum(y for _, y in v) / len(v))
              for v in bins.values())
    return {"n": len(pts), "brier": round(b, 4), "log_loss": round(l, 4), "directional_acc": round(d, 4),
            "ece": round(ece, 4)}


def _paired_bootstrap(rows, arm_a, arm_b, n_boot=10000, seed=777):
    pairs = [(r["arms"][arm_a], r["arms"][arm_b], r["outcome"]) for r in rows
             if r.get("arms", {}).get(arm_a) is not None and r.get("arms", {}).get(arm_b) is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "insufficient": True}
    db = [brier(a, y) - brier(b, y) for a, b, y in pairs]
    dl = [logloss(a, y) - logloss(b, y) for a, b, y in pairs]
    n = len(pairs); state = seed & 0xFFFFFFFF

    def rnd():
        nonlocal state; state = (1103515245 * state + 12345) & 0x7FFFFFFF; return state / 0x7FFFFFFF
    mb, ml = [], []
    for _ in range(n_boot):
        idx = [int(rnd() * n) % n for _ in range(n)]
        mb.append(sum(db[i] for i in idx) / n); ml.append(sum(dl[i] for i in idx) / n)
    mb.sort(); ml.sort()

    def ci(v): return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v))], 4)]
    return {"n": n, "mean_brier_diff": round(sum(db) / n, 4), "brier_diff_ci95": ci(mb),
            "mean_logloss_diff": round(sum(dl) / n, 4), "logloss_diff_ci95": ci(ml),
            "note": f"arm_a={arm_a}, arm_b={arm_b}; negative => {arm_a} lowers loss (improves) vs {arm_b}"}


def aggregate(rows):
    ok = [r for r in rows if r.get("status", "").startswith("completed") and r.get("arms")
          and r["arms"].get("phase3_repaired") is not None]
    per_arm = {a: _score(ok, a) for a in ARMS}
    # domain breakdown for the key comparison
    domains = {}
    for r in ok:
        domains.setdefault(r["domain"], []).append(r)
    dom = {}
    for dname, rs in domains.items():
        dom[dname] = {"n": len(rs), "phase2_brier": _score(rs, "phase2").get("brier"),
                      "repaired_brier": _score(rs, "phase3_repaired").get("brier")}
    # per-question deltas repaired vs phase2
    deltas = []
    for r in ok:
        y = r["outcome"]; a = r["arms"]
        b_rep, b_p2 = brier(a["phase3_repaired"], y), brier(a["phase2"], y)
        deltas.append({"qid": r["qid"], "outcome": y, "p_phase2": round(a["phase2"], 4),
                       "p_repaired": round(a["phase3_repaired"], 4), "brier_delta": round(b_rep - b_p2, 4),
                       "verdict": "repaired_better" if b_rep < b_p2 - 1e-9 else
                                  "phase2_better" if b_p2 < b_rep - 1e-9 else "tie"})
    boot_rep = _paired_bootstrap(ok, "phase3_repaired", "phase2")
    boot_cur = _paired_bootstrap(ok, "phase3_current", "phase2")
    gates = _eval_gates(per_arm, boot_rep)
    return {"n_completed": len(ok), "n_questions": len(rows), "per_arm_scores": per_arm,
            "domain_breakdown": dom, "per_question_deltas": deltas,
            "paired_repaired_vs_phase2": boot_rep, "paired_current_vs_phase2": boot_cur,
            "repaired_better": sum(1 for d in deltas if d["verdict"] == "repaired_better"),
            "phase2_better": sum(1 for d in deltas if d["verdict"] == "phase2_better"),
            "tie": sum(1 for d in deltas if d["verdict"] == "tie"),
            "preregistered_gates": gates}


def _eval_gates(per_arm, boot_rep):
    """Evaluate the PRE-REGISTERED Part-K gates (from PREREGISTERED_GATES.json). Never changed after results."""
    if boot_rep.get("insufficient"):
        return {"insufficient": True}
    bd, ld = boot_rep["mean_brier_diff"], boot_rep["mean_logloss_diff"]
    bci, lci = boot_rep["brier_diff_ci95"], boot_rep["logloss_diff_ci95"]
    ece_rep = (per_arm.get("phase3_repaired") or {}).get("ece")
    ece_p2 = (per_arm.get("phase2") or {}).get("ece")
    g = {
        "G1_brier_not_worse": bd <= 0,
        "G2_logloss_not_worse": ld <= 0,
        "G3_one_primary_CI_favorable": (bci[1] < 0) or (lci[1] < 0),
        "G4_no_significant_regression": not (bci[0] > 0 or lci[0] > 0),
        "G5_ece_not_materially_worse": (ece_rep is not None and ece_p2 is not None and ece_rep <= ece_p2 + 0.05)}
    if g["G3_one_primary_CI_favorable"] and g["G4_no_significant_regression"] and g["G5_ece_not_materially_worse"]:
        verdict = "phase3b_improves"
    elif (bci[0] > 0 or lci[0] > 0):
        verdict = "phase3b_harms"
    else:
        verdict = "inconclusive"
    return {"gates": g, "verdict": verdict,
            "production_default": "phase2" if verdict != "phase3b_improves" else "phase3b_repaired"}


def main():
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig
    from swm.world_model_v2.phase3b_repair import load_params
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_params()
    llm, meter = _make_llm()
    if llm is None:
        print(json.dumps({"error": "no llm"})); return
    cfg = OrchestratorConfig()
    qs = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    rows = []
    for q in qs:
        rec = run_one(q, llm, cfg, params, seed=args.seed)
        rows.append(rec)
        payload = {"rows": rows, "aggregate": aggregate(rows), "frozen_params": params,
                   "retrieval_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": args.seed}
        ART.write_text(json.dumps(payload, indent=2))
        a = rec.get("arms", {})
        print(f"[{rec['domain']:11s}] {rec['qid']:15s} y={rec.get('outcome')} p2={a.get('phase2')} "
              f"cur={a.get('phase3_current')} rep={a.get('phase3_repaired')} "
              f"mode={(rec.get('repair') or {}).get('mode')} t={rec.get('latency_s')}s")
    agg = aggregate(rows)
    print("\nLOCKED-TEST AGGREGATE:", json.dumps(agg["per_arm_scores"], indent=2))
    print("repaired vs phase2:", json.dumps(agg["paired_repaired_vs_phase2"]))


if __name__ == "__main__":
    main()
