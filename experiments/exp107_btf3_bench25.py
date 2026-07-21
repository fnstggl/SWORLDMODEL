"""EXP-107: the 25-question BTF-3 stack-up. All fixes on (evidence targeting + escalated retry,
grounded evidence-quality prior, recurrence calendar, evidence-sufficiency gate, rollout-viability
invariant), MEAN-OF-3 per question (the ~0.6 single-run variance makes single runs uninterpretable).

Fidelity: NUMERIC actors (SWM_LLM_ACTORS=off) so 25 x 3 = 75 full pipelines is tractable (~2 min/run vs
~25). The actor-cognition layer is NOT among the fixes under test (and hurt accuracy in EXP-102); every
structural fix (evidence, prior, calendar, institutions, populations, scheduled reality) still executes.
Full-LLM-actor validation on a subset is a separate, slower check.

Scored vs the FutureSearch SOTA baseline (per-question) and constant baselines. Leakage-quarantined:
the forecaster sees only allowlisted as-of fields; resolution/sota join at scoring. Per-question checkpoint.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp107_btf3_bench25 [n=25] [k=3]
"""
from __future__ import annotations
import dataclasses
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("SWM_LLM_ACTORS", "off")               # numeric actors for tractability (labeled)

from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input

CKPT = Path("experiments/results/exp107_checkpoints")
OUT = Path("experiments/results/exp107_bench25.json")
IDS = Path("experiments/results/exp107_sample_ids.json")


def _sample(by_id, n):
    if IDS.exists():
        return json.loads(IDS.read_text())[:n]
    ids = sorted(random.Random(107).sample(sorted(by_id), n))
    IDS.write_text(json.dumps(ids, indent=1))
    return ids


def _brier(ps, ys):
    return round(sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps), 4) if ps else None


def _auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    if not pos or not neg:
        return None
    return round(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)), 4)


def _one(qid, rows, k):
    from swm.world_model_v2.unified_runtime import simulate_world_stable
    from swm.api.deepseek_backend import default_chat_fn
    cp = CKPT / f"{qid}.json"
    if cp.exists():
        return json.loads(cp.read_text())
    q = _forecast_input(rows[qid])
    llm = default_chat_fn(system="Reply ONLY JSON.", max_tokens=8000, temperature=0.2)
    try:
        res = simulate_world_stable(q["question"], n_runs=k, as_of=str(q["present_date"])[:10],
                                    horizon=str(q["expected_resolution_date"])[:10], llm=llm, seed=0)
        d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
        prov = d.get("provenance") or {}
        p = d.get("calibrated_probability") if d.get("calibrated_probability") is not None else d.get("raw_probability")
        rec = {"qid": qid, "question": q["question"][:90], "p": p,
               "status": d.get("simulation_status"),
               "mean_of_k": prov.get("mean_of_k"),
               "evidence_sufficiency": prov.get("evidence_sufficiency"),
               "outcome_pathway_repaired": (prov.get("outcome_pathway") or {}).get("repaired")}
    except Exception as e:  # noqa: BLE001
        rec = {"qid": qid, "question": q["question"][:90], "p": None, "error": f"{type(e).__name__}: {e}"[:200]}
    cp.write_text(json.dumps(rec, default=str))
    return rec


def run(n=25, k=3):
    CKPT.mkdir(parents=True, exist_ok=True)
    rows = {r["question_id"]: r for r in fetch_btf3()}
    ids = _sample(rows, n)
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_one, qid, rows, k): qid for qid in ids}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            print(f"  {rec['qid'][:8]} p={rec.get('p')} status={rec.get('status')} "
                  f"spread={(rec.get('mean_of_k') or {}).get('spread')} ({len(results)}/{len(ids)})")

    # ---- scoring (answers + SOTA join here) ----
    for r in results:
        row = rows[r["qid"]]
        r["outcome"] = int(row["resolution"])
        r["p_sota"] = None if row.get("sota_forecast_probability") is None \
            else round(float(row["sota_forecast_probability"]) / 100.0, 4)
    scored = [r for r in results if r.get("p") is not None]
    ys = [r["outcome"] for r in scored]
    ps = [r["p"] for r in scored]
    base = sum(int(rows[r["qid"]]["resolution"]) for r in results) / len(results)
    sota = [(r["p_sota"], r["outcome"]) for r in scored if r["p_sota"] is not None]
    summary = {
        "n": len(results), "n_scored": len(scored), "k": k, "fidelity": "numeric_actors",
        "base_rate_yes": round(base, 4),
        "wmv2": {"brier": _brier(ps, ys),
                 "acc@0.5": round(sum((p > 0.5) == y for p, y in zip(ps, ys)) / len(ys), 4) if ys else None,
                 "AUC": _auc(ps, ys),
                 "mean_spread": round(sum((r.get("mean_of_k") or {}).get("spread", 0) for r in scored)
                                      / max(1, len(scored)), 4)},
        "sota_futuresearch": {"n": len(sota), "brier": _brier([p for p, _ in sota], [y for _, y in sota]),
                              "acc@0.5": round(sum((p > 0.5) == y for p, y in sota) / len(sota), 4) if sota else None,
                              "AUC": _auc([p for p, _ in sota], [y for _, y in sota])},
        "const_baselines": {"p=0.5": _brier([0.5] * len(ys), ys),
                            "p=base_rate": _brier([base] * len(ys), ys)},
        "results": sorted(results, key=lambda r: r["qid"]),
    }
    OUT.write_text(json.dumps(summary, indent=1, default=str))
    print("\n=== EXP-107 BTF-3 stack-up (n={}, mean-of-{}, numeric actors) ===".format(len(results), k))
    print(json.dumps({kk: vv for kk, vv in summary.items() if kk != "results"}, indent=1))
    print(f"wrote {OUT}")
    return summary


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 25,
        int(sys.argv[2]) if len(sys.argv) > 2 else 3)
