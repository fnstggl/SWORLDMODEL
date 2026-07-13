"""Phase 3 — forensic trace dump: every number traced from a verified claim to the terminal.

For a few held-out questions, dump the COMPLETE chain a reviewer needs to audit a single forecast:
  claim → LLM qualitative tag → dependence-collapsed effective observation → per-observation likelihood
  reweight (assimilation ledger: ESS before/after, resample) → outcome-rate posterior (prior→posterior) →
  structural prior→posterior → the resolve_outcome StateDelta that records rate_source=='posterior' →
  terminal distribution. Writes a machine-readable artifact + a human-readable trace.

This is the anti-scaffolding evidence: it shows the posterior CROSSING every plane and being CONSUMED, with
the numbers attributable to specific claims. Networked (real DeepSeek + live RSS); records the retrieval date.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

OUT = Path("experiments/results/phase3")
QUESTIONS = [
    ("Will the Federal Reserve cut interest rates at its next meeting?", "2024-09-01", "2024-09-20"),
    ("Will there be a US federal government shutdown?", "2024-09-15", "2024-10-15"),
]


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    llm0 = default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)
    return (lambda p: llm0(p)) if llm0 else None


def dump_one(q, as_of, horizon, llm, seed=0):
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig
    from swm.world_model_v2.phase3_pipeline import simulate_with_posterior
    res, art = simulate_with_posterior(q, llm=llm, as_of=as_of, horizon=horizon, seed=seed,
                                       config=OrchestratorConfig())
    bundle, tags, post = art["bundle"], art["tags"], art["posterior"]
    pi = res.posterior_inference or {}
    tag_by_id = {t.claim_id: t for t in tags}
    claims = []
    for c in bundle.included_claims()[:12]:
        t = tag_by_id.get(c["claim_id"])
        claims.append({"claim_id": c["claim_id"], "class": c.get("claim_class"),
                       "span": (c.get("supporting_span", "") or "")[:110],
                       "dependence_group": c.get("dependence_group", ""),
                       "tag_direction": getattr(t, "outcome_direction", None),
                       "tag_strength": getattr(t, "strength", None),
                       "is_strategic": getattr(t, "is_strategic", None),
                       "reliability": round(getattr(t, "reliability", 0.0), 3) if t else None})
    return {
        "question": q, "as_of": as_of, "horizon": horizon,
        "planes": art.get("planes", {}),
        "prior_provenance": post.prior_provenance,
        "claims_to_tags": claims,
        "assimilation_ledger": post.assimilation_ledger,
        "outcome_rate": pi.get("outcome_rate"),
        "structural_prior": post.structural_prior,
        "structural_posterior": post.structural_posterior,
        "terminal_distribution": res.raw_distribution,
        "raw_probability": res.raw_probability,
        "rate_source_in_terminal_delta": art["planes"].get("execution", {}).get("rate_source"),
        "posterior_consumed": pi.get("consumed_by_simulator"),
        "posterior_hash": art.get("posterior_hash"),
        "support_grade": res.support_grade, "simulation_status": res.simulation_status,
        "latent_specs": [s.as_dict() for s in art.get("latent_specs", [])]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    llm = _make_llm()
    if llm is None:
        print("no llm")
        return
    traces = []
    for q, a, h in QUESTIONS:
        try:
            traces.append(dump_one(q, a, h, llm))
            print(f"traced: {q[:50]} → consumed={traces[-1]['posterior_consumed']} "
                  f"src={traces[-1]['rate_source_in_terminal_delta']}")
        except Exception as e:  # noqa: BLE001
            traces.append({"question": q, "error": f"{type(e).__name__}: {e}"[:200]})
            print(f"ERROR on {q[:40]}: {e}")
        (OUT / "forensic_traces.json").write_text(
            json.dumps({"traces": traces, "retrieval_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                              time.gmtime())}, indent=2))
    print(f"\nwrote {OUT/'forensic_traces.json'}")


if __name__ == "__main__":
    main()
