"""EXP-047: semantic (LLM-judge) stance vs lexical extraction — closing the EXP-044 content gap.

EXP-044 proved lexical reading recovers only ~13% of the signal the market extracts from the same news,
because outcome polarity is question-specific and term-matching can't read it. This tests the fix: an LLM
stance judge (swm/variables/semantic_stance.py) reads the as-of news for THIS question's specific YES
resolution and returns a signed stance. Same module runs in production (Anthropic API backend) and here
(committed judgments replayed via cached_judge_fn) — so this validates the production system.

Leakage discipline (the contamination concern is real — a strong LLM may recall dated outcomes):
  - the judge saw ONLY question + as-of news headlines, never the price or the outcome;
  - the PRIMARY metric is MARKET-CONSISTENCY — correlation of the semantic stance with the as-of market
    PRICE the judge never saw. The price is a function of the same news, independent of the future
    outcome, so agreement with it measures reading skill, not outcome recall (the EXP-037 robustness
    argument);
  - stance-vs-outcome is reported too but flagged as contamination-susceptible; the post-cutoff subset
    (resolving after the model's knowledge cutoff) is called out separately as the cleaner signal.

Compares, on the judged sample: semantic stance vs the EXP-044 lexical stance vs the market — correlation
with the as-of price (robust) and with the outcome (caveated). Writes JSON.
Run: python -m experiments.exp047_semantic_stance
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.variables.content_extractor import extract as lexical_extract
from experiments.datasets_swm import load
from experiments.exp043_news_grounding import _outcome

STANCE_CACHE = "experiments/results/exp047_stance/stance_cache.json"
RESULT = "experiments/results/exp047_semantic_stance.json"


def _rebuild_batch():
    """Deterministically reconstruct the exact sample the judge scored (must match the dump in the
    scratchpad that produced the cache): test_kalshi, news>=4, target+future present, every `step`-th."""
    cand = [x for x in load("test_kalshi")
            if x.get("question") and x.get("news") and len(x["news"]) >= 4
            and x.get("target") and x.get("future")]
    step = max(1, len(cand) // 70)
    return cand[::step][:70]


def _corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den > 1e-9 else 0.0


def run():
    cache = json.loads(Path(STANCE_CACHE).read_text())
    batch = _rebuild_batch()
    rows = []
    for i, x in enumerate(batch):
        key = str(i)
        if key not in cache:
            continue
        sem_stance, sem_conf, sem_rel = cache[key]
        lex = lexical_extract(x["question"], x["news"], x["target"]["t"])
        y = _outcome(x)
        rows.append({
            "sem_stance": sem_stance, "sem_conf_stance": sem_stance * sem_conf,
            "lex_stance": lex["recent_subject_stance"],       # EXP-044's best lexical feature
            "price": x["target"]["p"], "lean": x["target"]["p"] - 0.5,
            "outcome": y, "resdate": None,
            "market_id": x["market_id"],
        })
        # post-cutoff flag from the market_id date suffix (resolves after model knowledge cutoff)
        import re
        m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", x.get("market_id", ""))
        rows[-1]["post_cutoff"] = bool(m and int(m.group(1)) >= 26)

    n = len(rows)
    price = [r["price"] for r in rows]
    outc = [r["outcome"] for r in rows]

    def corr_with(field, target):
        xs = [r[field] for r in rows]
        return round(_corr(xs, target), 4)

    # PRIMARY: market consistency (stance vs as-of price the judge never saw)
    market_consistency = {
        "semantic_stance": corr_with("sem_stance", price),
        "semantic_confident_stance": corr_with("sem_conf_stance", price),
        "lexical_stance": corr_with("lex_stance", price),
    }
    # SECONDARY (caveated): stance vs outcome
    outcome_corr = {
        "semantic_stance": corr_with("sem_stance", outc),
        "lexical_stance": corr_with("lex_stance", outc),
        "market_lean": round(_corr([r["lean"] for r in rows], outc), 4),
    }
    # cleaner outcome signal: post-cutoff subset only
    post = [r for r in rows if r.get("post_cutoff")]
    post_outcome = None
    if len(post) >= 5:
        po = [r["outcome"] for r in post]
        post_outcome = {"n": len(post),
                        "semantic_stance": round(_corr([r["sem_stance"] for r in post], po), 4),
                        "market_lean": round(_corr([r["lean"] for r in post], po), 4)}

    sc = market_consistency["semantic_stance"]; lc = market_consistency["lexical_stance"]
    out = {"dataset": "kalshi", "n_judged": n, "judge": "LLM (Claude) — blind to price/outcome",
           "market_consistency_corr_with_price": market_consistency,
           "outcome_corr": outcome_corr, "post_cutoff_outcome_corr": post_outcome,
           "semantic_beats_lexical_on_market_consistency": abs(sc) > abs(lc),
           "semantic_vs_lexical_ratio": round(abs(sc) / max(1e-9, abs(lc)), 2)}

    print(f"EXP-047 semantic stance vs lexical — Kalshi, {n} LLM-judged questions (blind to price/outcome)")
    print("  PRIMARY — MARKET CONSISTENCY (corr of stance with the as-of PRICE the judge never saw; robust):")
    for k, v in market_consistency.items():
        print(f"    {k:<28} corr {v}")
    print(f"  -> semantic beats lexical on market-consistency: {out['semantic_beats_lexical_on_market_consistency']} "
          f"({out['semantic_vs_lexical_ratio']}x)")
    print("  SECONDARY — corr with OUTCOME (contamination-susceptible; the market lean is the ceiling):")
    for k, v in outcome_corr.items():
        print(f"    {k:<28} corr {v}")
    if post_outcome:
        print(f"  POST-CUTOFF subset (cleaner, n={post_outcome['n']}): semantic {post_outcome['semantic_stance']}"
              f"  market_lean {post_outcome['market_lean']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
