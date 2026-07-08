"""Rebuild the data caches for the EXP-074/075/077 data-scaling program from their public sources.

The two large CMV caches (cmv_pairs.json ~14MB, cmv_perop.json ~12MB) are re-derivable and gitignored;
this script regenerates them so the experiments run offline after one download. The small Upworthy A/B
cache is committed but is also rebuilt here. No API key needed — these are public research corpora.

  - CMV: Cornell ConvoKit "winning-args-corpus" (Tan et al. 2016), matched winner/loser arguments to CMV
    OPs. Source: http://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/
  - Upworthy: the Upworthy Research Archive exploratory release (headline A/B tests). Source: OSF jd64p.

Run: python -m experiments.build_scaling_caches
"""
from __future__ import annotations

import csv
import io
import json
import os
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

WA_URL = "http://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/winning-args-corpus.zip"
UPWORTHY_URL = "https://osf.io/download/3vqmp/"   # exploratory-packages CSV (OSF)

PAIRS = "experiments/results/exp074/cmv_pairs.json"
PEROP = "experiments/results/exp075/cmv_perop.json"
UPWORTHY = "experiments/results/exp077/upworthy_ab.json"


def _download(url, dst):
    if Path(dst).exists():
        return dst
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)
    return dst


def _load_wa_utterances():
    z = _download(WA_URL, "/tmp/winning-args-corpus.zip")
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith("utterances.jsonl"))
        with zf.open(name) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                if line.strip():
                    yield json.loads(line)


def build_cmv():
    """cmv_pairs.json (matched pairs, EXP-074) and cmv_perop.json (per-OP candidate sets, EXP-075)."""
    roots, byroot, pairs_raw = {}, defaultdict(list), defaultdict(dict)
    for u in _load_wa_utterances():
        m = u["meta"]
        if u["reply-to"] is None:
            roots[u["id"]] = u.get("text", "")
        # per-OP candidate arguments: direct top-level replies with a known outcome
        if u["reply-to"] and str(u["reply-to"]).startswith("t3_") and m.get("success") in (0, 1):
            byroot[u["root"]].append((m["success"], u.get("text", "")))
        # matched pairs: gather by pair_id -> {success: text}
        for pid in (m.get("pair_ids") or []):
            if m.get("success") in (0, 1) and u.get("text"):
                pairs_raw[(u["root"], pid)][m["success"]] = u.get("text", "")

    # EXP-075 per-OP sets
    perop = []
    for r, args in byroot.items():
        if len(args) < 2 or not (0 < sum(s for s, _ in args) < len(args)):
            continue
        perop.append({"op_id": r, "op_text": roots.get(r, "")[:1200],
                      "args": [{"success": int(s), "text": t[:1500]} for s, t in args]})
    Path(PEROP).parent.mkdir(parents=True, exist_ok=True)
    Path(PEROP).write_text(json.dumps(perop))
    print(f"  wrote {PEROP}: {len(perop)} OPs, {sum(len(o['args']) for o in perop)} args")

    # EXP-074 matched pairs (both a winner and a loser present)
    pairs = []
    for (root, pid), d in pairs_raw.items():
        if 0 in d and 1 in d:
            pairs.append({"pair": f"{root}:{pid}", "op_text": roots.get(root, "")[:1200],
                          "pos": d[1][:1500], "neg": d[0][:1500]})
    Path(PAIRS).parent.mkdir(parents=True, exist_ok=True)
    Path(PAIRS).write_text(json.dumps(pairs))
    print(f"  wrote {PAIRS}: {len(pairs)} matched pairs")


def build_upworthy():
    """upworthy_ab.json: per-test distinct-wording headline sets with CTR winner labels (EXP-077)."""
    csv_path = _download(UPWORTHY_URL, "/tmp/upworthy_exploratory.csv")
    tests = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                imp, clk = int(row["impressions"]), int(row["clicks"])
            except (ValueError, KeyError):
                continue
            h = row["headline"].strip()
            if not h:
                continue
            tests[row["clickability_test_id"]][h][0] += imp
            tests[row["clickability_test_id"]][h][1] += clk
    out = []
    for t, hs in tests.items():
        dh = {h: (i, c) for h, (i, c) in hs.items() if i >= 1000}   # trustworthy CTR only
        if len(dh) < 2:
            continue
        ctrs = {h: c / i for h, (i, c) in dh.items()}
        if max(ctrs.values()) == min(ctrs.values()):
            continue
        win = max(ctrs, key=ctrs.get)
        out.append({"test_id": t, "headlines": [
            {"text": h[:300], "impressions": i, "clicks": c, "ctr": round(c / i, 5), "success": int(h == win)}
            for h, (i, c) in dh.items()]})
    Path(UPWORTHY).parent.mkdir(parents=True, exist_ok=True)
    Path(UPWORTHY).write_text(json.dumps(out))
    print(f"  wrote {UPWORTHY}: {len(out)} tests, {sum(len(o['headlines']) for o in out)} headlines")


def main():
    print("building CMV caches (ConvoKit winning-args)...")
    build_cmv()
    print("building Upworthy A/B cache (OSF exploratory release)...")
    build_upworthy()
    print("done.")


if __name__ == "__main__":
    main()
