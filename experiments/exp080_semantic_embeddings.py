"""EXP-080: real embeddings unlock SEMANTIC elasticity transfer across the corpus.

The registry has 592 elasticities keyed by exact strings. A real sentence embedding lets a weight learned for
one variable inform a semantically-equivalent-but-lexically-different one — the thing the lexical default
cannot do. This embeds every corpus key once (HuggingFace all-MiniLM-L6-v2), commits the vectors to
`swm/variables/prior_embeddings.json` (so it works offline/reproducibly), and tests paraphrase probes:
does querying "price growth → monetary tightening" recover the "inflation → rate_hike" elasticity? Real
embeddings should; lexical should not.

Run: python -m experiments.exp080_semantic_embeddings   (uses HF_TOKEN once, then the committed cache)
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.variables.embedding_registry import EmbeddingPriorRegistry, lexical_embed
from swm.variables.embeddings import EmbeddingCache, hf_embed_fn
from swm.variables.prior_registry import PriorRegistry, semantic_key

REG = "swm/variables/learned_priors.json"
CACHE = "swm/variables/prior_embeddings.json"
RESULT = "experiments/results/exp080_semantic_embeddings.json"

# (query variable, query outcome-class, expected registry variable, expected outcome-class) — paraphrases with
# minimal lexical overlap with the stored key, same meaning.
PROBES = [
    ("consumer price growth", "monetary tightening", "inflation", "rate_hike"),
    ("joblessness level", "central bank raising interest rates", "unemployment", "rate_hike"),
    ("how long the subscriber has stayed", "subscription cancellation", "tenure", "customer_churn"),
    ("the monthly bill amount", "account cancellation", "monthly_charges", "customer_churn"),
    ("a curiosity-provoking title", "click engagement", "curiosity", "headline_engagement"),
]


def run() -> dict:
    reg = PriorRegistry.load(REG)
    cache = EmbeddingCache.load(CACHE)
    live = hf_embed_fn()
    key_texts = [k.replace("|", " ") for k in reg.records]
    probe_texts = [f"{qv} {qo}" for qv, qo, *_ in PROBES]
    before = len(cache.vecs)
    cache.precompute(key_texts + probe_texts, live)
    cache.save()

    real = EmbeddingPriorRegistry(reg, embed_fn=cache.embed_fn(), threshold=0.45).build_index()
    lex = EmbeddingPriorRegistry(reg, embed_fn=lexical_embed, threshold=0.45).build_index()

    rows, real_ok, lex_ok = [], 0, 0
    for qv, qo, ev, eo in PROBES:
        exp = reg.records.get(semantic_key(ev, eo))
        r, l = real.get(qv, qo), lex.get(qv, qo)
        real_hit = r is not None and exp is not None and (r.mean > 0) == (exp.mean > 0)
        lex_hit = l is not None and exp is not None and (l.mean > 0) == (exp.mean > 0)
        real_ok += real_hit
        lex_ok += lex_hit
        rows.append({"probe": f"{qv} -> {qo}", "expected": f"{ev}->{eo}",
                     "expected_elasticity": round(exp.mean, 3) if exp else None,
                     "real_embed": {"transferred": r is not None, "elasticity": round(r.mean, 3) if r else None,
                                    "correct_sign": real_hit, "src": r.source if r else None},
                     "lexical": {"transferred": l is not None, "elasticity": round(l.mean, 3) if l else None,
                                 "correct_sign": lex_hit}})
    res = {"n_probes": len(PROBES), "corpus_keys_embedded": len(cache.vecs), "new_embeddings": len(cache.vecs) - before,
           "real_embedding_correct": real_ok, "lexical_correct": lex_ok, "cache_path": CACHE, "probes": rows}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print("EXP-080  semantic elasticity transfer via real embeddings (all-MiniLM-L6-v2)")
    print(f"  embedded {len(cache.vecs)} corpus keys -> committed {CACHE}")
    print(f"  paraphrase transfer (correct-sign elasticity recovered): real {real_ok}/{len(PROBES)} vs "
          f"lexical {lex_ok}/{len(PROBES)}")
    for row in rows:
        rr, ll = row["real_embed"], row["lexical"]
        print(f"    {row['probe'][:44]:44s} exp {row['expected_elasticity']:+.2f} | "
              f"real {'OK' if rr['correct_sign'] else '--'} ({rr['elasticity']}) | "
              f"lexical {'OK' if ll['correct_sign'] else '--'} ({ll['elasticity']})")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
