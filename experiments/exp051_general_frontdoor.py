"""EXP-051: the general front door end-to-end — any question flows all the way through.

Demonstrates GeneralSimulator routing + fusing real questions through the assembled pipeline:
  - POPULATION questions (GSS opinion items) -> GroundedSimulator bottom-up simulation, scored vs the true
    held-out population share;
  - MARKET questions (Kalshi) with as-of news -> the semantic stance judge (EXP-047 committed judgments),
    scored vs the as-of market price the judge never saw.

One `GeneralSimulator.answer(...)` handles both by routing to the machinery that carries each question's
signal — the concrete "put in any scenario -> simulate/read -> calibrated outcome." Writes JSON.
Run: python -m experiments.exp051_general_frontdoor
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from swm.api.general_simulate import GeneralSimulator
from swm.api.grounded_simulate import GroundedSimulator
from experiments.datasets_gss import load as load_gss
from experiments.exp045_population_rollout import ATTRS
from experiments.exp050_grounded_readout import _rows, _split

RESULT = "experiments/results/exp051_general_frontdoor.json"


def _key(question, news):
    """Instance-unique key (question + first headline) so duplicate markets stay distinct — the EXP-047
    alignment was per-instance, not per-question-text."""
    t = (news[0].get("title", "") if news else "")[:40]
    return f"{question}||{t}"


class _CachedJudge:
    """Replays the EXP-047 semantic stance judgments (LLM read the news, blind to price/outcome)."""
    def __init__(self, by_key):
        self.by = by_key

    def stance(self, question, news, resolution_hint=""):
        return self.by.get(_key(question, news), {"stance": 0.0, "confidence": 0.0, "relevant": 0})


def _market_cases():
    from experiments.exp047_semantic_stance import _rebuild_batch   # EXACT same sample the cache was built on
    cache = json.loads(Path("experiments/results/exp047_stance/stance_cache.json").read_text())
    batch = _rebuild_batch()
    by_key = {}
    for i, x in enumerate(batch):
        if str(i) in cache:
            s = cache[str(i)]
            by_key[_key(x["question"], x["news"])] = {"stance": s[0], "confidence": s[1], "relevant": s[2]}
    return batch, by_key


def run():
    # fit the population simulator on GSS train, hold out test respondents
    tr_recs, te_recs = _split(load_gss())
    sim = GroundedSimulator(attrs=ATTRS).fit(_rows(tr_recs))
    batch, stance_by_q = _market_cases()
    gen = GeneralSimulator(grounded=sim, stance_judge=_CachedJudge(stance_by_q))

    # A. population questions end-to-end (route -> population_simulation)
    te_by_item = defaultdict(list)
    for r in _rows(te_recs):
        te_by_item[r["qid"]].append(r)
    pop_results, pop_err = [], []
    labels = {"grass": "Should marijuana be legal?", "cappun": "Favor the death penalty?",
              "homosex": "Is homosexuality wrong?", "abany": "Abortion legal for any reason?",
              "natenvir": "Spend more on the environment?"}
    for item, q in labels.items():
        rs = te_by_item.get(item, [])
        if len(rs) < 30:
            continue
        true = sum(r["answer_idx"] for r in rs) / len(rs)
        fc = gen.answer(q, known_item=item, population=[r["demo"] for r in rs])
        pop_err.append(abs(fc.p_outcome - true))
        pop_results.append({"question": q, "method": fc.method, "predicted": round(fc.p_outcome, 4),
                            "true_share": round(true, 4), "n": len(rs)})

    # B. market questions end-to-end (route -> news_stance), scored vs the as-of price
    mkt_results, mkt_pred, mkt_price = [], [], []
    for x in batch:
        q = x["question"]
        if _key(q, x["news"]) not in stance_by_q:
            continue
        fc = gen.answer(q, news=x["news"], base_rate=0.5, resolution_hint="")
        price = x["target"]["p"]
        mkt_pred.append(fc.p_outcome); mkt_price.append(price)
        if len(mkt_results) < 6:
            mkt_results.append({"question": q[:60], "method": fc.method,
                                "predicted": round(fc.p_outcome, 4), "as_of_price": round(price, 4)})

    def _corr(xs, ys):
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        return num / den if den > 1e-9 else 0.0
    mkt_err = [abs(p - pr) for p, pr in zip(mkt_pred, mkt_price)]
    mkt_corr = round(_corr(mkt_pred, mkt_price), 4)

    out = {"population_questions": {
               "mae_vs_true_share": round(sum(pop_err) / len(pop_err), 4) if pop_err else None,
               "examples": pop_results},
           "market_questions": {
               "n": len(mkt_err),
               "corr_with_as_of_price": mkt_corr,       # the honest signal: direction (EXP-047)
               "mae_vs_as_of_price": round(sum(mkt_err) / len(mkt_err), 4) if mkt_err else None,
               "level_calibration_caveat": "news stance recovers DIRECTION (corr), not the absolute level; "
                                           "low-base-rate events need a reference-class base rate the stance "
                                           "judge does not supply — MAE is inflated by the 0.5 anchor",
               "examples": mkt_results},
           "one_front_door": True}

    print("EXP-051 general front door — one answer() routes any question end-to-end")
    print(f"  A. POPULATION questions (route -> bottom-up simulation), MAE vs true share "
          f"{out['population_questions']['mae_vs_true_share']}:")
    for r in pop_results:
        print(f"     [{r['method']}] {r['question']:<34} predicted {r['predicted']}  true {r['true_share']}")
    print(f"  B. MARKET questions (route -> semantic news reading), corr with as-of price {mkt_corr} "
          f"(direction; MAE {out['market_questions']['mae_vs_as_of_price']} inflated by 0.5 anchor) "
          f"over {len(mkt_err)}:")
    for r in mkt_results:
        print(f"     [{r['method']}] {r['question']:<42} predicted {r['predicted']}  price {r['as_of_price']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
