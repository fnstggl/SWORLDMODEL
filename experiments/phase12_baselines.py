"""Phase 12 — direct-model baselines + critic inputs (Part L).

For each corpus question, computes non-simulation comparison arms on the SAME question + as_of (no post-outcome
info): a grounded one-shot direct-LLM forecast (B1) and a call-matched grounded direct ensemble (B2, log-linear
pooled). The base rate (B0) and analogical arm are computed offline in the evaluator. These let Phase 12 answer
the second required question — is the full simulator adding predictive signal vs realistic non-simulation
alternatives — and feed the critic (V2 vs direct/ensemble disagreement). Incremental + resumable.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

OUT = Path("experiments/results/phase12")
ART = OUT / "baselines.json"

_PROMPT = ("You are a careful forecaster. As of {as_of}, estimate the probability (0..1) that the following "
           "resolves YES. Use only what was plausibly known by that date; do not assume knowledge of the "
           "outcome. Reply ONLY JSON: {{\"p\": <0..1>, \"why\": \"<one line>\"}}\nQUESTION: {q}")


def _make_llm(temp):
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(system="Reply ONLY JSON.", max_tokens=300, temperature=temp)


def _ask(llm, q, as_of):
    from swm.engine.grounding import parse_json
    try:
        raw = parse_json(llm(_PROMPT.format(as_of=as_of, q=q))) or {}
        p = float(raw.get("p"))
        return min(1.0, max(0.0, p))
    except Exception:  # noqa: BLE001
        return None


def main():
    import math
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((OUT / "corpus.json").read_text())["rows"]
    llms = [_make_llm(0.2), _make_llm(0.5), _make_llm(0.8)]
    if llms[0] is None:
        print("no llm"); return
    existing = {}
    if ART.exists():
        for r in json.loads(ART.read_text()).get("rows", []):
            if r.get("direct_p") is not None:
                existing[r["row_id"]] = r
    out = []
    qs = rows[:args.limit] if args.limit else rows
    for r in qs:
        if r["row_id"] in existing:
            out.append(existing[r["row_id"]]); continue
        q, as_of = r.get("question", ""), r.get("as_of", "")
        rec = {"row_id": r["row_id"], "qid": r["qid"], "outcome": r["outcome"], "split": r["split"]}
        if not q:
            rec.update({"direct_p": None, "ensemble_p": None, "note": "no_question_text"})
        else:
            ps = [_ask(l, q, as_of) for l in llms]
            ps = [p for p in ps if p is not None]
            if not ps:
                rec.update({"direct_p": None, "ensemble_p": None})
            else:
                # log-linear (geometric) pool for the ensemble
                lg = sum(math.log(min(1 - 1e-6, max(1e-6, p)) / (1 - min(1 - 1e-6, max(1e-6, p)))) for p in ps) / len(ps)
                ens = 1 / (1 + math.exp(-lg))
                rec.update({"direct_p": round(ps[0], 4), "ensemble_p": round(ens, 4), "n_ensemble": len(ps)})
        out.append(rec)
        ART.write_text(json.dumps({"rows": out, "retrieval_date_utc":
                       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
        print(f"{r['row_id'][:40]:40s} y={r['outcome']} direct={rec.get('direct_p')} ens={rec.get('ensemble_p')}")
    print("DONE baselines", len(out))


if __name__ == "__main__":
    main()
