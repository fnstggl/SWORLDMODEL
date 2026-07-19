"""CLI: python -m historical_backtests.tools.compare_runs BENCH RUN_A RUN_B
Compares two scored runs (dev scores) of the same benchmark: aggregate + by scale/domain +
failure/qualification rates + per-case prediction deltas + cost."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(bench, run):
    rdir = ROOT / "results" / bench / run
    scores = json.loads((rdir / "scores_dev.json").read_text()) \
        if (rdir / "scores_dev.json").exists() else {}
    rows = {}
    led = rdir / "forecast_ledger.jsonl"
    if led.exists():
        for line in led.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["case_id"], r["cutoff"])] = r
    return scores, rows


def main():
    bench, ra, rb = sys.argv[1], sys.argv[2], sys.argv[3]
    sa, rows_a = _load(bench, ra)
    sb, rows_b = _load(bench, rb)
    out = {"benchmark": bench, "run_a": ra, "run_b": rb,
           "overall": {"a": sa.get("overall"), "b": sb.get("overall")},
           "by_scale": {k: {"a": (sa.get("by_scale") or {}).get(k),
                            "b": (sb.get("by_scale") or {}).get(k)}
                        for k in sorted(set(sa.get("by_scale") or {})
                                        | set(sb.get("by_scale") or {}))},
           "qualification_rate": {
               "a": (sum(1 for r in rows_a.values() if r.get("qualified"))
                     / max(1, len(rows_a))),
               "b": (sum(1 for r in rows_b.values() if r.get("qualified"))
                     / max(1, len(rows_b)))},
           "cost_usd": {
               "a": round(sum((r.get("llm_usage") or {}).get("cost_usd", 0)
                              for r in rows_a.values()), 2),
               "b": round(sum((r.get("llm_usage") or {}).get("cost_usd", 0)
                              for r in rows_b.values()), 2)}}
    deltas = []
    for key in set(rows_a) & set(rows_b):
        pa, pb = rows_a[key].get("p_yes"), rows_b[key].get("p_yes")
        if isinstance(pa, (int, float)) and isinstance(pb, (int, float)):
            deltas.append({"case": key[0], "cutoff": key[1][:10],
                           "p_a": pa, "p_b": pb, "delta": round(pb - pa, 3)})
    deltas.sort(key=lambda d: -abs(d["delta"]))
    out["largest_prediction_changes"] = deltas[:15]
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
