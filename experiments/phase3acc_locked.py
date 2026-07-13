"""Phase 3 accuracy — LOCKED-test capture (the new adequately-powered held-out set).

Runs the real production path on the 93-question locked corpus (all event-family/temporally disjoint from the
frozen 23- and 34-question sets). Per question it captures EVERYTHING needed to score every arm OFFLINE from a
frozen record — so fitting/selection (done separately on the frozen dev set) never touches these questions:

  tags (+ claim class + text), prior(alpha,beta), structural prior/posterior, n_effective,
  p_phase2 (posterior ignored), p_phase3 (raw generic posterior consumed),
  causal_proposal (typed latents + combination) + causal_claim_map (claim->latent links),
  realized outcome, family, posterior_hash.

Incremental writes; per-question failures preserved. Networked + LLM-backed; strict as-of retrieval unchanged.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

from experiments.phase3acc_corpus import QUESTIONS
from swm.world_model_v2.phase3_causal_latents import propose_latents, map_claims_to_latents

OUT = Path("experiments/results/phase3acc")
ART = OUT / "locked_capture.json"


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


def _tag_rows(bundle, tags):
    cls = {c.get("claim_id"): c.get("claim_class") for c in bundle.included_claims()}
    txt = {}
    for c in bundle.included_claims():
        svo = " ".join(str(c.get(k, "")) for k in ("subject", "predicate", "object", "value") if c.get(k))
        span = str(c.get("supporting_span", ""))
        txt[c.get("claim_id")] = (svo + (" | " + span if span else "")).strip()[:280]
    rows = []
    for t in tags:
        rows.append({"claim_id": t.claim_id, "text": txt.get(t.claim_id, ""),
                     "claim_class": cls.get(t.claim_id), "source_type": cls.get(t.claim_id),
                     "outcome_direction": t.outcome_direction, "strength": t.strength,
                     "reliability": float(t.reliability), "is_strategic": t.is_strategic,
                     "dependence_group": t.dependence_group,
                     "supports_hypotheses": list(t.supports_hypotheses),
                     "opposes_hypotheses": list(t.opposes_hypotheses)})
    return rows


def capture_one(q, llm, cfg, seed=0):
    from swm.world_model_v2.phase3_pipeline import simulate_with_posterior
    qid, question, as_of, horizon, domain, outcome, family, note = q
    rec = {"qid": qid, "question": question, "as_of": as_of, "horizon": horizon, "domain": domain,
           "outcome": outcome, "family": family, "resolution_note": note}
    t0 = time.time()
    try:
        res3, art = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                            config=cfg, consume_posterior=True)
        plan, bundle, tags = art["plan"], art["bundle"], art["tags"]
        res2, _ = simulate_with_posterior(question, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                          config=cfg, consume_posterior=False, plan=plan, bundle=bundle, tags=tags)
        pi = res3.posterior_inference or {}
        prov = (pi.get("prior_provenance") or {}).get("outcome_rate", {})
        trows = _tag_rows(bundle, tags)
        # causal-latent proposal + claim mapping (qualitative LLM; numeric inference is offline)
        prop = propose_latents(question, horizon=horizon, llm=llm)
        briefs = [{"claim_id": t["claim_id"], "text": t["text"]} for t in trows if t["text"]]
        cmap = map_claims_to_latents(prop["latents"], briefs, llm=llm) if prop["latents"] else {}
        rec.update({
            "status": res3.simulation_status,
            "outcome_lean": str((plan.provenance or {}).get("outcome_lean", "neutral")),
            "prior": {"alpha": prov.get("alpha"), "beta": prov.get("beta"),
                      "mean": (pi.get("outcome_rate") or {}).get("prior_mean")},
            "tags": trows,
            "structural_prior": (pi.get("structural") or {}).get("prior"),
            "structural_posterior": (pi.get("structural") or {}).get("posterior"),
            "n_effective_observations": pi.get("n_effective_observations"),
            "n_included_claims": len(bundle.included_claim_ids),
            "posterior_mean": (pi.get("outcome_rate") or {}).get("posterior_mean"),
            "p_phase2": res2.raw_probability, "p_phase3": res3.raw_probability,
            "causal_proposal": prop, "causal_claim_map": cmap,
            "posterior_hash": art.get("posterior_hash"),
            "latency_s": round(time.time() - t0, 1)})
    except Exception as e:  # noqa: BLE001
        rec.update({"status": "harness_error", "error": f"{type(e).__name__}: {e}"[:200],
                    "latency_s": round(time.time() - t0, 1)})
    return rec


def main():
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    llm, meter = _make_llm()
    if llm is None:
        print(json.dumps({"error": "no llm"})); return
    cfg = OrchestratorConfig()
    rows = []
    existing = {}
    if ART.exists():                                            # resume: keep completed rows
        for r in json.loads(ART.read_text()).get("rows", []):
            if r.get("status", "").startswith("completed"):
                existing[r["qid"]] = r
    qs = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    for q in qs:
        qid = q[0]
        rec = existing.get(qid) or capture_one(q, llm, cfg, seed=args.seed)
        rows.append(rec)
        ART.write_text(json.dumps({"rows": rows, "retrieval_date_utc":
                       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": args.seed}, indent=2))
        a2, a3 = rec.get("p_phase2"), rec.get("p_phase3")
        nlat = len((rec.get("causal_proposal") or {}).get("latents", []))
        print(f"[{rec['domain']:11s}] {qid:18s} y={rec.get('outcome')} p2={a2} p3={a3} "
              f"nlat={nlat} status={rec.get('status')} t={rec.get('latency_s')}s")
    print("DONE", len(rows), "captured; llm_calls", meter["calls"])


if __name__ == "__main__":
    main()
