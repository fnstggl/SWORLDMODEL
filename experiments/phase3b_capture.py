"""Phase 3B — diagnostic capture: freeze the full decomposition of every diagnostic question.

Re-runs the production path on the 23 diagnostic questions (the frozen negative-backtest set) and dumps, per
question, EVERYTHING needed to (a) forensically trace the regression and (b) fit/validate the repair OFFLINE
without further network/LLM calls:

  question, as_of, horizon, domain, realized outcome,
  compiled outcome_lean + structural hypotheses (with priors),
  every claim tag (text + direction + strength + reliability + strategic + hypotheses),
  prior (alpha,beta,mean,provenance), assimilation ledger, posterior mean/sd, structural prior->posterior,
  p_phase2 (consume_posterior=False terminal), p_phase3 (consume_posterior=True terminal),
  point-estimate terminal, n_included, n_effective, posterior_hash.

Written to experiments/results/phase3b/diagnostic_capture.json (incremental). This is the frozen substrate
for the offline diagnosis and calibration/stacking fits — so the fits never touch the network and are exactly
reproducible. Live retrieval can drift vs the committed backtest; the aggregate is compared to the committed
numbers and any drift is recorded, not hidden.
"""
from __future__ import annotations
import json, time
from pathlib import Path

OUT = Path("experiments/results/phase3b")
ART = OUT / "diagnostic_capture.json"

# the 23 diagnostic questions, identical to the committed backtest (imported to avoid drift in wording)
from experiments.wmv2_phase3_real_backtest import QUESTIONS


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


def _claim_text_map(bundle):
    out = {}
    for c in bundle.included_claims():
        cid = c.get("claim_id")
        svo = " ".join(str(c.get(k, "")) for k in ("subject", "predicate", "object", "value") if c.get(k))
        span = str(c.get("supporting_span", ""))
        txt = (svo + (" | " + span if span else "")).strip()
        out[cid] = {"text": txt[:280], "source": c.get("source_id"),
                    "source_type": c.get("claim_class"), "modality": c.get("modality"),
                    "polarity": c.get("polarity")}
    return out


def capture_one(q, llm, cfg, seed=0):
    from swm.world_model_v2.phase3_pipeline import simulate_with_posterior
    qid, question, as_of, horizon, domain, outcome, market, note = q
    t0 = time.time()
    rec = {"qid": qid, "question": question, "as_of": as_of, "horizon": horizon, "domain": domain,
           "outcome": outcome, "resolution_note": note}
    try:
        res3, art = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                            config=cfg, consume_posterior=True)
        plan, bundle, tags = art["plan"], art["bundle"], art["tags"]
        res2, _ = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                          config=cfg, consume_posterior=False, plan=plan, bundle=bundle, tags=tags)
        resP, _ = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                          config=cfg, consume_posterior=True, posterior_point_estimate=True,
                                          plan=plan, bundle=bundle, tags=tags)
        pi = res3.posterior_inference or {}
        ctext = _claim_text_map(bundle)
        tag_rows = []
        for t in tags:
            c = ctext.get(t.claim_id, {})
            tag_rows.append({
                "claim_id": t.claim_id, "text": c.get("text", ""), "source_type": c.get("source_type"),
                "outcome_direction": t.outcome_direction, "strength": t.strength,
                "reliability": round(t.reliability, 3), "is_strategic": t.is_strategic,
                "dependence_group": t.dependence_group,
                "supports_hypotheses": list(t.supports_hypotheses), "opposes_hypotheses": list(t.opposes_hypotheses)})
        prov = (pi.get("prior_provenance") or {}).get("outcome_rate", {})
        hyps = [{"id": str(h.get("id")), "label": h.get("label") or h.get("name"),
                 "prior": h.get("prior")} for h in (getattr(plan, "structural_hypotheses", []) or [])
                if isinstance(h, dict)]
        rec.update({
            "status": res3.simulation_status,
            "outcome_lean": str((plan.provenance or {}).get("outcome_lean", "neutral")),
            "structural_hypotheses": hyps,
            "prior": {"alpha": prov.get("alpha"), "beta": prov.get("beta"),
                      "mean": (pi.get("outcome_rate") or {}).get("prior_mean"),
                      "class": prov.get("class"), "source": prov.get("source")},
            "tags": tag_rows,
            "assimilation_ledger": pi.get("assimilation_ledger", []),
            "posterior_mean": (pi.get("outcome_rate") or {}).get("posterior_mean"),
            "posterior_sd": (pi.get("outcome_rate") or {}).get("posterior_sd"),
            "structural_prior": (pi.get("structural") or {}).get("prior"),
            "structural_posterior": (pi.get("structural") or {}).get("posterior"),
            "n_effective_observations": pi.get("n_effective_observations"),
            "n_claims_collapsed": pi.get("n_claims_collapsed"),
            "n_included_claims": len(bundle.included_claim_ids),
            "p_phase2": res2.raw_probability, "p_phase3": res3.raw_probability,
            "p_point_estimate": resP.raw_probability,
            "posterior_hash": art.get("posterior_hash"),
            "warnings": pi.get("warnings", []),
            "latency_s": round(time.time() - t0, 1)})
    except Exception as e:  # noqa: BLE001
        rec.update({"status": "harness_error", "error": f"{type(e).__name__}: {e}"[:200],
                    "latency_s": round(time.time() - t0, 1)})
    return rec


def main():
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig
    OUT.mkdir(parents=True, exist_ok=True)
    llm, meter = _make_llm()
    if llm is None:
        print(json.dumps({"error": "no llm"})); return
    cfg = OrchestratorConfig()
    rows = []
    for q in QUESTIONS:
        rec = capture_one(q, llm, cfg)
        rows.append(rec)
        ART.write_text(json.dumps({"rows": rows, "retrieval_date_utc":
                                   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
        print(f"{rec['qid']:16s} y={rec.get('outcome')} p2={rec.get('p_phase2')} p3={rec.get('p_phase3')} "
              f"neff={rec.get('n_effective_observations')} lean={rec.get('outcome_lean')} "
              f"nhyp={len(rec.get('structural_hypotheses',[]))} t={rec.get('latency_s')}s")
    print("DONE", len(rows), "captured; llm_calls", meter["calls"])


if __name__ == "__main__":
    main()
