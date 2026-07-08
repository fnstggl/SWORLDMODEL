"""EXP-081: the continuous harvest — rebuild the calibrated-prior registry from all data, idempotently.

The flywheel as a standing job. Because elasticities combine PRECISION-WEIGHTED, re-running the harvest onto
the existing registry would DOUBLE-COUNT the same data (spuriously shrinking the CIs). So the continuous
harvest rebuilds the registry FROM SCRATCH each run over all current sources — deterministic and idempotent:
same data ⇒ same registry, new data ⇒ a tighter registry, no double-counting. It also refreshes real
embeddings for any new keys and appends a run to `harvest_manifest.json`, so the registry keeps compounding
correctly as datasets are added. This is the entry point a scheduled Routine invokes (then commits + pushes).

Run: python -m experiments.exp081_continuous_harvest
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from experiments.exp076_corpus_harvest import (_cmv, _extra_generic, _fomc, _gss, _oqa, _upworthy)
from swm.variables.prior_registry import PriorRegistry

REGISTRY_PATH = "swm/variables/learned_priors.json"
EMBED_CACHE = "swm/variables/prior_embeddings.json"
MANIFEST = "experiments/results/harvest_manifest.json"
RESULT = "experiments/results/exp081_continuous_harvest.json"


def harvest_all(*, refresh_embeddings=True) -> dict:
    """Rebuild the registry from scratch over every source (idempotent) and commit it + a manifest entry."""
    reg = PriorRegistry()                              # FRESH — never accumulate onto a prior run (no double-count)
    sources = [_gss(reg), _oqa(reg), _cmv(reg), _fomc(reg), _upworthy(reg),
               _extra_generic(reg, "stackexchange", "question_answered"),
               _extra_generic(reg, "telco_churn", "customer_churn"),
               _extra_generic(reg, "globalopinions", "opinion_consensus")]
    reg.save(REGISTRY_PATH)
    ocs = sorted({k.split("|")[1] for k in reg.records})

    new_embeddings = 0
    if refresh_embeddings and os.environ.get("HF_TOKEN"):
        try:
            from swm.variables.embeddings import EmbeddingCache, hf_embed_fn
            cache = EmbeddingCache.load(EMBED_CACHE)
            before = len(cache.vecs)
            cache.precompute([k.replace("|", " ") for k in reg.records], hf_embed_fn())
            cache.save()
            new_embeddings = len(cache.vecs) - before
        except Exception as e:                         # embedding refresh must never break the harvest
            new_embeddings = -1
            print(f"  (embedding refresh skipped: {str(e)[:70]})")

    run = {"time": datetime.now().isoformat(timespec="seconds"),
           "sources": [{"source": s["source"], "fits": s.get("fits")} for s in sources],
           "n_priors": len(reg.records), "n_outcome_classes": len(ocs), "new_embeddings": new_embeddings}
    manifest = []
    if os.path.exists(MANIFEST):
        try:
            manifest = json.loads(Path(MANIFEST).read_text())
        except ValueError:
            manifest = []
    manifest.append(run)
    Path(MANIFEST).write_text(json.dumps(manifest[-50:], indent=1))     # keep the last 50 runs
    Path(RESULT).write_text(json.dumps({"latest": run, "outcome_classes": ocs}, indent=1))
    return run, ocs


def run() -> dict:
    run_rec, ocs = harvest_all()
    print("EXP-081  continuous harvest (fresh rebuild, idempotent)")
    print(f"  {run_rec['time']}")
    for s in run_rec["sources"]:
        print(f"    {s['source']:14s} fits={s['fits']}")
    print(f"  -> {run_rec['n_priors']} priors across {run_rec['n_outcome_classes']} outcome-classes; "
          f"embeddings refreshed (+{run_rec['new_embeddings']}); committed {REGISTRY_PATH}")
    print(f"  manifest appended -> {MANIFEST}")
    return {"latest": run_rec, "outcome_classes": ocs}


if __name__ == "__main__":
    run()
