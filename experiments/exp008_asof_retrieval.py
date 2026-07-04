"""EXP-008 — does leakage-safe as-of retrieval help, and can it beat the market?

Domain: Manifold binary event markets (the only domain with a real market price at a fixed
horizon). Test set = the EXP-006 markets: 140 BINARY markets that resolved AFTER the Jan-2026 model
cutoff, each with a reconstructed market price at horizon T (48h after creation) and a hidden 0/1
resolution.

The retrieval source is a LEAKAGE-SAFE reference class (`swm/retrieval/corpus.py`): every sibling
market is a timestamped Document (timestamp = its close/resolution date). For a target market
created at C, the only reachable siblings are those that RESOLVED strictly before C — a document
that resolved before the target even existed cannot encode the target's outcome. Any attempt to
return a post-as_of doc raises LeakageError. This is the whole point: the retriever is structurally
forbidden from touching the post-creation information the market aggregates.

Four predictors, identical test markets, identical metrics:
  1. no retrieval            — the blind LLM forecast (question text only; data/mf_pred_agent*)
  2. LLM + as-of retrieval   — #1 shrunk toward the leakage-safe reference-class base rate
  3. state-model + retrieval — a Posterior (the real state machinery) seeded at the population base
                               rate and .observe()'d with each retrieved sibling's resolution
  4. market price at T       — the reconstructed market@T (data/mf_truth market_at_T)

Reports Brier / log loss / ECE / decision lift (uplift@20) and the head-to-head vs the market. No
web search, no post-as_of data — enforced by the corpus gate.

  python -m experiments.exp008_asof_retrieval score
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import date
from pathlib import Path

from swm.eval.metrics import (brier_score, expected_calibration_error, log_loss, uplift_at_k)
from swm.retrieval.corpus import Document, TimestampedCorpus
from swm.state.state import Posterior

CLIP = lambda p: min(0.99, max(0.01, p))


def _ts(iso: str) -> float:
    y, m, d = (int(x) for x in iso.split("-"))
    return (date(y, m, d) - date(2000, 1, 1)).days * 86400.0


def _logit(p): return math.log(CLIP(p) / (1 - CLIP(p)))
def _sig(z): return 1 / (1 + math.exp(-z))


def _load():
    pk = {p["id"]: p for p in json.loads(Path("data/mf_packets.json").read_text())}
    truth = {t["id"]: t for t in json.loads(Path("data/mf_truth.json").read_text())}
    preds: dict[str, list[float]] = {}
    for f in glob.glob("data/mf_pred_agent*.json") + glob.glob("data/mf_predB_agent*.json"):
        for x in json.loads(Path(f).read_text()):
            preds.setdefault(x["id"], []).append(CLIP(x["p_yes"]))
    llm = {i: sum(v) / len(v) for i, v in preds.items()}          # pool the blind swarm
    return pk, truth, llm


def _build_corpus(pk: dict, truth: dict) -> TimestampedCorpus:
    """Each resolved market is a Document timestamped at its close/resolution date. The outcome is
    in meta — reachable ONLY for siblings that closed before the query's as_of."""
    c = TimestampedCorpus()
    for i, p in pk.items():
        c.add(Document(doc_id=i, timestamp=_ts(p["closes_iso"]), text=p["question"],
                       meta={"resolution": truth[i]["resolution"]}))
    return c


def score():
    pk, truth, llm = _load()
    corpus = _build_corpus(pk, truth)
    ids = [i for i in pk if i in truth and i in llm]
    base = sum(truth[i]["resolution"] for i in ids) / len(ids)     # population YES rate (prior mean)

    rows = {"1 no retrieval (LLM)": [], "2 LLM + as-of retrieval": [],
            "3 state-model + retrieval": [], "4 market price @T": []}
    y, n_ref = [], []
    for i in ids:
        as_of = _ts(pk[i]["created_iso"])                          # forecast made at creation
        # leakage-safe reference class: siblings RESOLVED before this market was created
        ref = corpus.reference_class(pk[i]["question"], as_of, k=25, min_sim=0.0)
        ref = [d for d in ref if d.doc_id != i]                    # never retrieve self
        res = [d.meta["resolution"] for d in ref]
        n_ref.append(len(res))
        ref_rate = sum(res) / len(res) if res else base            # retrieved reference-class rate

        p_llm = llm[i]
        # #2: shrink the LLM toward the reference class, weighted by how much evidence it carries
        w = len(res) / (len(res) + 8.0)                            # 8-market shrinkage prior
        p_ret = _sig((1 - w) * _logit(p_llm) + w * _logit(ref_rate))
        # #3: pure state machinery — a Posterior at the population prior, updated by each sibling
        post = Posterior(base, 4.0)                                # prior worth 4 pseudo-markets
        for r in res:
            post.observe(float(r))
        p_state = post.mean

        y.append(truth[i]["resolution"])
        rows["1 no retrieval (LLM)"].append(CLIP(p_llm))
        rows["2 LLM + as-of retrieval"].append(CLIP(p_ret))
        rows["3 state-model + retrieval"].append(CLIP(p_state))
        rows["4 market price @T"].append(CLIP(truth[i]["market_at_T"]))

    print(f"EXP-008  Manifold event forecasting, n={len(ids)} markets, base YES rate {base:.3f}.")
    print(f"leakage-safe reference class: median {sorted(n_ref)[len(n_ref)//2]} siblings/market, "
          f"{sum(1 for n in n_ref if n==0)} markets with zero (fall back to population prior).\n")
    print(f"   {'method':<28}{'logloss':>9}{'brier':>8}{'ece':>7}{'uplift@20':>11}{'beats mkt':>11}")
    mkt = rows["4 market price @T"]
    for name, p in rows.items():
        ll, br, ece, up = (log_loss(y, p), brier_score(y, p),
                           expected_calibration_error(y, p), uplift_at_k(y, p, 0.2))
        beats = sum(1 for k in range(len(y))
                    if abs(p[k] - y[k]) < abs(mkt[k] - y[k])) / len(y)
        tag = "  (reference)" if name.startswith("4") else ""
        print(f"   {name:<28}{ll:>9.4f}{br:>8.4f}{ece:>7.4f}{up:>11.4f}{beats:>10.0%}{tag}")

    # does retrieval improve over no-retrieval? (paired Brier delta on the retrieval-rich subset)
    rich = [k for k in range(len(ids)) if n_ref[k] >= 5]
    print(f"\n== retrieval-rich subset (>=5 siblings, n={len(rich)}) ==")
    if rich:
        for name in ("1 no retrieval (LLM)", "2 LLM + as-of retrieval", "3 state-model + retrieval",
                     "4 market price @T"):
            p = [rows[name][k] for k in rich]
            yy = [y[k] for k in rich]
            print(f"   {name:<28} brier {brier_score(yy, p):.4f}  logloss {log_loss(yy, p):.4f}")
    print("\n(honest read below in exp008_asof_retrieval.md)")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("score")
    ap.parse_args()
    score()


if __name__ == "__main__":
    main()
