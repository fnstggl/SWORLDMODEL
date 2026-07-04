"""EXP-001 runner: person-vs-segment go/no-go on a real corpus (see exp001_person_vs_segment.md).

Usage:
    python -m experiments.exp001_run --pages data/gmail_page1.json data/gmail_page2.json \
        --owner beckett@runaurelius.com [--window-hours 24]

Outputs the ladder, noise floor, and an integrity report. Prints honestly — including when
the verdict is NO-GO or the data is too small/degenerate to conclude anything.
"""
from __future__ import annotations

import argparse
import time

from swm.eval.baselines import noise_floor_brier
from swm.eval.harness import run_ladder
from swm.ingestion import store as store_mod
from swm.ingestion.gmail_search import convert_files
from swm.ingestion.importers import import_threads
from swm.ingestion.store import EventStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="+", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--window-hours", type=float, default=None,
                    help="override email reply window (justify from observed latencies!)")
    args = ap.parse_args()

    threads, integrity = convert_files(args.pages, args.owner)
    store = EventStore(":memory:")
    counts = import_threads(store, threads, args.owner)

    if args.window_hours:
        store_mod.REPLY_WINDOWS["email"] = args.window_hours * 3600.0

    now = time.time()
    sends = store.labeled_sends(censor_at=now)
    all_sends = store.labeled_sends()
    n_censored = len(all_sends) - len(sends)
    replies = [s for s in sends if s.replied]
    latencies_h = sorted((s.reply_latency or 0) / 3600 for s in replies)

    print("== integrity ==")
    print(f"threads in: {integrity['threads_in']}  dropped (bounced): {integrity['dropped_bounced']}")
    print(f"imported: {counts}")
    print(f"labeled sends: {len(sends)} (censored out: {n_censored})  "
          f"replies: {len(replies)}  rate: {len(replies)/max(1,len(sends)):.3f}")
    if latencies_h:
        print(f"reply latencies (h): min {latencies_h[0]:.2f}  median "
              f"{latencies_h[len(latencies_h)//2]:.2f}  max {latencies_h[-1]:.2f}")
        window_h = store_mod.REPLY_WINDOWS["email"] / 3600
        if latencies_h[-1] > 0.8 * window_h:
            print(f"WARNING: max latency near window ({window_h:.0f}h) — window may truncate replies")

    print("\n== ladder (temporal split) ==")
    # monkeypatch labeled_sends inside the harness path via a censored view
    orig = store.labeled_sends
    store.labeled_sends = lambda **kw: orig(censor_at=now, **{k: v for k, v in kw.items() if k != "censor_at"})  # type: ignore
    result = run_ladder(store)
    if "error" in result:
        print("cannot run:", result["error"])
        return
    for r in result["rungs"]:
        print(f"{r['name']:<14} logloss {r['log_loss']:.4f}  brier {r['brier']:.4f}  "
              f"ece {r['ece']:.4f}  uplift@20 {r['uplift_at_20']:+.4f}")
    print(f"n_train {result['n_train']}  n_test {result['n_test']}  "
          f"test base rate {result['test_base_rate']:.3f}")
    print(f"bootstrap p(L3 fails to beat L2): {result['bootstrap_p_L3_beats_L2']}")
    print("verdict:", result["verdict"])
    print("\nnoise floor:", noise_floor_brier(sends, min_sends_per_recipient=3))


if __name__ == "__main__":
    main()
