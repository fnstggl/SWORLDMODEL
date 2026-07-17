"""Build all frozen evidence capsules for a benchmark (resumable; forecast-side safe — reads the
QUESTION vault only)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from historical_backtests.framework.evidence_build import build_capsule_for
from historical_backtests.framework.runner import load_question_vault

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/N — process-parallel sharding")
    a = ap.parse_args()
    vault = load_question_vault(a.benchmark)
    out_dir = ROOT / "evidence_archives" / a.benchmark
    todo = [(c, cut) for c in vault["cases"] for cut in c["forecast_cutoffs"]
            if not (out_dir / f"{c['case_id']}__{cut[:10]}.json").exists()]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        todo = [t for k, t in enumerate(todo) if k % n == i]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[{time.strftime('%H:%M:%S')}] capsules to build: {len(todo)}", flush=True)
    stats = {"items": 0, "wiki": 0, "wayback": 0, "rejected_no_archive": 0,
             "rejected_contaminated": 0}
    for i, (c, cut) in enumerate(todo):
        s = build_capsule_for(c, cut, out_dir=out_dir)
        stats["items"] += s["n_items"]
        stats["wiki"] += s["n_wiki"]
        stats["wayback"] += s["n_wayback"]
        stats["rejected_no_archive"] += s["rejected"]["no_archived_version"]
        stats["rejected_contaminated"] += s["rejected"]["contaminated"]
        print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(todo)} {s['case_id']} @{cut[:10]} "
              f"items={s['n_items']} (wiki {s['n_wiki']} / wayback {s['n_wayback']}; "
              f"rej {s['rejected']})", flush=True)
    (out_dir / "_build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
