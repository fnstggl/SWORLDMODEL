"""Phase 3 accuracy — augment a frozen capture with causal-latent proposals (LLM-only, fast).

Reads a capture that already has tags + outcomes (e.g. the frozen 23-question diagnostic capture), and for each
question calls the causal-latent LLM to (a) propose typed latents + a combination and (b) map the question's
claims to those latents. No rollouts, no retrieval — just the qualitative proposal, so the NUMERIC causal-latent
inference can then run OFFLINE and reproducibly. Writes <source>_enriched.json.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

from swm.world_model_v2.phase3_causal_latents import propose_latents, map_claims_to_latents


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    llm0 = default_chat_fn(system="Reply ONLY JSON.", max_tokens=1400, temperature=0.2)
    if llm0 is None:
        return None
    return llm0


def enrich(src, out, limit=None):
    d = json.loads(Path(src).read_text())
    llm = _make_llm()
    if llm is None:
        print("no llm"); return
    rows = d["rows"]
    rows = rows[:limit] if limit else rows
    for r in rows:
        if r.get("causal_proposal") is not None:
            continue
        if not r.get("tags"):
            r["causal_proposal"] = {"latents": [], "combination": "weighted_mean"}
            r["causal_claim_map"] = {}
            continue
        prop = propose_latents(r["question"], horizon=r.get("horizon", ""), llm=llm)
        briefs = [{"claim_id": t["claim_id"], "text": t.get("text", "")} for t in r["tags"] if t.get("text")]
        cmap = map_claims_to_latents(prop["latents"], briefs, llm=llm) if prop["latents"] else {}
        r["causal_proposal"] = prop
        r["causal_claim_map"] = cmap
        Path(out).write_text(json.dumps({"rows": rows, "retrieval_date_utc":
                             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
        print(f"{r['qid']:16s} latents={[L['type'] for L in prop['latents']]} comb={prop['combination']} "
              f"mapped={len(cmap)}")
    print("DONE enriched", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/results/phase3b/diagnostic_capture.json")
    ap.add_argument("--out", default="experiments/results/phase3acc/dev_enriched.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    enrich(args.src, args.out, args.limit)


if __name__ == "__main__":
    main()
