"""EXP-101: BTF-3 pastcasting pilot — 50 questions, no-evidence arm (WMv2 mechanism forecaster + DeepSeek).

First run of the stack on FutureSearch's public pastcasting benchmark (HF: BTF-2/BTF-3, binary config,
1,515 resolved questions anchored Apr-May 2026, resolved mid-May-Jul 2026, CC-BY-NC-4.0). This arm is
deliberately retrieval-free: it measures the compile->simulate pipeline exactly as the clean 660-question
Manifold backtest did (EXP-091/095), and doubles as the BTF "No Evidence" contamination probe — the LLM
(deepseek-v4-flash, self-reported cutoff May 2025, logged not trusted) predates the question window, and
an implausibly strong score here would itself be evidence of weight leakage, not skill.

Leakage protocol (the whole point — enforced in code, not by care):
  * the raw dataset lands in data/btf3/ (gitignored); the forecaster receives ONLY fields from
    ALLOWED_FIELDS (question / resolution_criteria / background / dates — all authored as-of present_date
    by the benchmark's own pipeline). `resolution`, `resolution_explanation`, `sota_*` never leave the
    scoring section; a hard assert refuses to build a prompt from a record carrying forbidden keys.
  * as-of = present_date, horizon = expected_resolution_date - present_date; no grounders, no web.
  * SOTA baseline (their published forecast per question) is joined only at scoring time.

Arms (same 50 question ids -> paired comparison):
  v4flash (default)  DeepSeek-V4-Flash direct. Official cutoff Apr 2026: outcomes (May-Jul 2026) are
                     post-cutoff, and Apr-2026 knowledge approximates the intended as-of state — but the
                     cutoff is fuzzy at week granularity, hence the second arm.
  v3or               deepseek/deepseek-chat-v3-0324 via OpenRouter (cutoff ~mid-2024): unambiguously
                     clean for BTF-3 AND for BTF-2's Oct-Dec 2025 windows, where V4 is contaminated.

Run: DEEPSEEK_API_KEY=.. python -m experiments.exp101_btf3_pilot [n_questions=50] [arm=v4flash]
     OPENROUTER_API_KEY=.. python -m experiments.exp101_btf3_pilot 50 v3or
"""
from __future__ import annotations

import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from swm.api.mechanisms import mechanism_forecast
from swm.eval.metrics import log_loss

ROWS_API = ("https://datasets-server.huggingface.co/rows"
            "?dataset=BTF-2%2FBTF-3&config=binary&split=test&offset={off}&length=100")
RAW = Path("data/btf3/binary_full.json")                    # gitignored (data/): CC-BY-NC, not redistributed
SAMPLE_IDS = Path("experiments/results/exp101_btf3_sample_ids.json")
ARMS = {"v4flash": lambda system: __import__("swm.api.deepseek_backend", fromlist=["x"])
        .deepseek_chat_fn(system=system, max_tokens=700),
        "v3or": lambda system: __import__("swm.api.openrouter_backend", fromlist=["x"])
        .openrouter_chat_fn("deepseek/deepseek-chat-v3-0324", system=system, max_tokens=700)}


def _paths(arm):
    sfx = "" if arm == "v4flash" else f"_{arm}"
    return (Path(f"experiments/results/exp101_btf3_predictions{sfx}.json"),
            Path(f"experiments/results/exp101_btf3_pilot{sfx}.json"))

ALLOWED_FIELDS = {"question_id", "question", "resolution_criteria", "background",
                  "present_date", "date_cutoff_end", "expected_resolution_date"}
FORBIDDEN_FIELDS = {"resolution", "resolution_explanation", "sota_forecast_probability",
                    "sota_summary_rationale"}


def fetch_btf3() -> list[dict]:
    if RAW.exists():
        return json.loads(RAW.read_text())
    rows, off = [], 0
    while True:
        with urllib.request.urlopen(ROWS_API.format(off=off), timeout=60) as r:
            page = json.loads(r.read())
        rows += [x["row"] for x in page["rows"]]
        off += 100
        if off >= page["num_rows_total"]:
            break
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(rows))
    return rows


def _ts(s: str) -> float:
    return datetime.fromisoformat(str(s).split(".")[0]).replace(tzinfo=timezone.utc).timestamp()


def _forecast_input(row: dict) -> dict:
    q = {k: row[k] for k in ALLOWED_FIELDS if k in row}
    assert not (set(q) & FORBIDDEN_FIELDS), "leak: forbidden field reached the forecaster"
    return q


def _question_text(q: dict) -> str:
    assert not (set(q) & FORBIDDEN_FIELDS)
    return (f"{q['question']}\n\nResolution criteria: {q['resolution_criteria'][:900]}\n\n"
            f"Background (as of {str(q['present_date'])[:10]}): {q['background'][:1200]}")


def _auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    if not pos or not neg:
        return None
    return round(sum((p > n_) + 0.5 * (p == n_) for p in pos for n_ in neg) / (len(pos) * len(neg)), 4)


def _brier(ps, ys):
    return round(sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps), 4)


def run(n_questions: int = 50, arm: str = "v4flash") -> dict:
    PRED, RESULT = _paths(arm)
    rows = fetch_btf3()
    by_id = {r["question_id"]: r for r in rows}
    if SAMPLE_IDS.exists():
        ids = json.loads(SAMPLE_IDS.read_text())
    else:
        ids = sorted(random.Random(42).sample(sorted(by_id), n_questions))
        SAMPLE_IDS.write_text(json.dumps(ids, indent=1))

    llm = ARMS[arm]("You are a careful superforecaster. Reply with ONLY compact JSON.")
    done = {r["question_id"]: r for r in json.loads(PRED.read_text())} if PRED.exists() else {}
    preds = list(done.values())

    def _work(qid):
        q = _forecast_input(by_id[qid])
        as_of, resolve = _ts(q["present_date"]), _ts(q["expected_resolution_date"])
        if resolve <= as_of:
            resolve = as_of + 30 * 86400
        p, info = mechanism_forecast(_question_text(q), as_of, resolve, llm, n=4000)
        return {"question_id": qid, "p_model": 0.5 if p is None else round(float(p), 4),
                "compile_failed": p is None,
                "mechanism": (info or {}).get("mechanism"),
                "horizon_days": round((resolve - as_of) / 86400, 1),
                "question": q["question"][:120]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(_work, qid) for qid in ids if qid not in done]):
            preds.append(fut.result())
            if len(preds) % 10 == 0:
                PRED.write_text(json.dumps(preds, indent=1))
                print(f"  {len(preds)}/{len(ids)} forecast")
    PRED.write_text(json.dumps(preds, indent=1))

    # ---- scoring: answers + SOTA baseline are touched only from here on ----
    for r in preds:
        row = by_id[r["question_id"]]
        r["outcome"] = int(row["resolution"])
        r["p_sota"] = None if row.get("sota_forecast_probability") is None \
            else round(float(row["sota_forecast_probability"]) / 100.0, 4)
    ys = [r["outcome"] for r in preds]
    ps = [r["p_model"] for r in preds]
    base = sum(ys) / len(ys)
    sota_pairs = [(r["p_sota"], r["outcome"]) for r in preds if r["p_sota"] is not None]

    res = {"n": len(preds), "arm": arm, "base_rate_yes": round(base, 4),
           "compile_failures": sum(r["compile_failed"] for r in preds),
           "model": {"brier": _brier(ps, ys), "log_loss": round(log_loss(ys, ps), 4),
                     "accuracy_at_0.5": round(sum((p > 0.5) == y for p, y in zip(ps, ys)) / len(ys), 4),
                     "AUC": _auc(ps, ys),
                     "frac_extreme": round(sum(p > 0.9 or p < 0.1 for p in ps) / len(ps), 3)},
           "sota_futuresearch": {"n": len(sota_pairs),
                                 "brier": _brier([p for p, _ in sota_pairs], [y for _, y in sota_pairs]),
                                 "accuracy_at_0.5": round(sum((p > 0.5) == y for p, y in sota_pairs)
                                                          / len(sota_pairs), 4),
                                 "AUC": _auc([p for p, _ in sota_pairs], [y for _, y in sota_pairs])},
           "const_baselines": {"p=0.5": _brier([0.5] * len(ys), ys),
                               "p=sample_base_rate": _brier([base] * len(ys), ys),
                               "p=0.33_global_prior": _brier([0.33] * len(ys), ys)},
           "by_mechanism": {}}
    for m in sorted({r["mechanism"] for r in preds if r["mechanism"]}):
        sub = [r for r in preds if r["mechanism"] == m]
        res["by_mechanism"][m] = {"n": len(sub),
                                  "brier": _brier([r["p_model"] for r in sub], [r["outcome"] for r in sub])}
    RESULT.write_text(json.dumps(res, indent=1))
    PRED.write_text(json.dumps(preds, indent=1))

    M, S, C = res["model"], res["sota_futuresearch"], res["const_baselines"]
    print(f"\nEXP-101  BTF-3 pilot, no-evidence arm={arm}, n={res['n']} (yes-rate {res['base_rate_yes']})")
    print(f"  WMv2+{arm}   brier {M['brier']}  log-loss {M['log_loss']}  acc@0.5 {M['accuracy_at_0.5']}"
          f"  AUC {M['AUC']}  extreme {M['frac_extreme']}")
    print(f"  FutureSearch SOTA (same qs, n={S['n']})  brier {S['brier']}  acc {S['accuracy_at_0.5']}"
          f"  AUC {S['AUC']}")
    print(f"  const: 0.5 -> {C['p=0.5']}   sample-base -> {C['p=sample_base_rate']}   0.33 -> {C['p=0.33_global_prior']}")
    print(f"  by mechanism: " + ", ".join(f"{m}({a['n']})={a['brier']}" for m, a in res["by_mechanism"].items()))
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 50,
        sys.argv[2] if len(sys.argv) > 2 else "v4flash")
