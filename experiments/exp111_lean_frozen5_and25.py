"""EXP-111: the LEAN deadline-aware forecaster on BOTH the five frozen questions AND the 25-question set.

Forecast = the §8-9 grounded deadline-aware prior MEAN (build_outcome_rate_prior), no rich rollout
(~2 LLM calls/question). Step 6 of the mandate: evaluate the lean path on the same five questions the
full-actor pass (EXP-110) uses, and on the larger 25-question set, so the two architectures are compared
on identical inputs. Leakage-quarantined; resolution/SOTA join at scoring only.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp111_lean_frozen5_and25
"""
from __future__ import annotations
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
from swm.api.deepseek_backend import default_chat_fn
from swm.world_model_v2.phase3_priors import build_outcome_rate_prior
from swm.world_model_v2.state import parse_time

FROZEN5 = [("BoJ", "7279494c-a775-5a57-a5f2-ac22252fb286"),
           ("visionOS", "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6"),
           ("Wale", "741b4bed-7502-5cd2-9cbe-949fbc70f857"),
           ("Hormuz", "017e64ef-7354-56c4-8a4d-e27121bc639a"),
           ("Banxico", "cfb43147-d9d2-5bd9-903f-f449e9a5aecf")]
OUT = Path("experiments/results/exp111_lean_frozen5_and25.json")


def _work(rows, llm, qid, label=""):
    try:
        q = _forecast_input(rows[qid])
        as_of, hz = str(q["present_date"])[:10], str(q["expected_resolution_date"])[:10]
        plan = SimpleNamespace(question=q["question"], as_of=parse_time(as_of), horizon_ts=parse_time(hz),
                               provenance={"outcome_lean": "neutral", "as_of": as_of})
        spec = build_outcome_rate_prior(plan, llm=llm)
        row = rows[qid]
        return {"label": label, "qid": qid, "p": round(float(spec.mean), 4),
                "outcome": int(row["resolution"]), "stage": spec.provenance.get("stage"),
                "src": spec.source_class, "reference_class": spec.reference_class[:60],
                "sota": None if row.get("sota_forecast_probability") is None
                else round(float(row["sota_forecast_probability"]) / 100.0, 4),
                "question": q["question"][:80]}
    except Exception as e:  # noqa: BLE001
        return {"label": label, "qid": qid, "p": None, "outcome": int(rows[qid]["resolution"]),
                "err": f"{type(e).__name__}: {e}"[:120]}


def _metrics(recs):
    sc = [r for r in recs if r.get("p") is not None]
    if not sc:
        return {"n": 0}
    ps, ys = [r["p"] for r in sc], [r["outcome"] for r in sc]

    def auc(ps, ys):
        P = [p for p, y in zip(ps, ys) if y == 1]
        N = [p for p, y in zip(ps, ys) if y == 0]
        return round(sum((a > b) + 0.5 * (a == b) for a in P for b in N) / (len(P) * len(N)), 4) \
            if P and N else None
    ym = [p for p, y in zip(ps, ys) if y == 1]
    nm = [p for p, y in zip(ps, ys) if y == 0]
    sota = [(r["sota"], r["outcome"]) for r in sc if r.get("sota") is not None]
    return {"n": len(sc), "base_rate_yes": round(sum(ys) / len(ys), 4),
            "brier": round(sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps), 4),
            "acc@0.5": round(sum((p > 0.5) == y for p, y in zip(ps, ys)) / len(ys), 4),
            "AUC": auc(ps, ys),
            "mean_p_yes": round(statistics.mean(ym), 4) if ym else None,
            "mean_p_no": round(statistics.mean(nm), 4) if nm else None,
            "const_base_brier": round(sum((sum(ys) / len(ys) - y) ** 2 for y in ys) / len(ys), 4),
            "sota_brier": round(sum((p - y) ** 2 for p, y in sota) / len(sota), 4) if sota else None,
            "sota_n": len(sota)}


def run():
    rows = {r["question_id"]: r for r in fetch_btf3()}
    ids25 = json.loads(Path("experiments/results/exp107_sample_ids.json").read_text())
    llm = default_chat_fn(system="Reply ONLY JSON.", max_tokens=1500, temperature=0.2)

    jobs = [(qid, label) for label, qid in FROZEN5] + [(qid, "") for qid in ids25]
    recs = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_work, rows, llm, qid, label): qid for qid, label in jobs}
        for f in as_completed(futs):
            r = f.result()
            recs[r["qid"]] = r
    frozen = [recs[qid] for _, qid in FROZEN5]
    set25 = [recs[qid] for qid in ids25]

    summary = {"frozen5": {"metrics": _metrics(frozen), "results": frozen},
               "set25": {"metrics": _metrics(set25)},
               "combined30": {"metrics": _metrics(frozen + set25)},
               "results25": set25}
    OUT.write_text(json.dumps(summary, indent=1, default=str))

    print("=== EXP-111 LEAN §8-9 deadline-prior forecaster ===")
    print("\n-- frozen 5 (same qs as the full-actor EXP-110) --")
    for r in frozen:
        print(f"  {r['label']:9s} p={r.get('p')} outcome={r['outcome']} SOTA={r.get('sota')} "
              f"stage={r.get('stage')} | {r.get('question','')[:52]}")
    print(f"  metrics: {json.dumps(_metrics(frozen))}")
    print(f"\n-- 25-set --\n  metrics: {json.dumps(_metrics(set25))}")
    print(f"\n-- combined 30 --\n  metrics: {json.dumps(_metrics(frozen + set25))}")
    print(f"  (ref: EXP-107 rich-numeric 25-q Brier 0.310/AUC0.413 [OLDER code]; constant≈0.24)")
    print(f"  wrote {OUT}")
    return summary


if __name__ == "__main__":
    run()
