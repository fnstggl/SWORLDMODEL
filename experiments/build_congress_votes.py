"""Build the VoteView Senate roll-call cache for EXP-085 (the grounded committee-vote world model).

For each roll-call vote in a range of Senate congresses, record every member's vote (Yea/Nay) plus their
ideology GROUNDED from the PRIOR congress (their measured ideal point BEFORE this vote — leakage-free; a
freshman with no prior record falls back to their party's prior-congress median). Public data, no key.

Source: voteview.com static CSVs (members / rollcalls / votes), per chamber per congress.
Run: python -m experiments.build_congress_votes
"""
from __future__ import annotations

import csv
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import median

BASE = "https://voteview.com/static/data/out"
CONGRESSES = list(range(105, 119))          # 105..118; targets 106..118 (105 seeds the first prior)
OUT = "experiments/results/exp085/senate_bills.json"


def _csv(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return list(csv.DictReader(io.TextIOWrapper(r, encoding="latin-1")))


def _members(c):
    """icpsr -> (nominate_dim1, party_code) for Senate congress c."""
    out = {}
    for m in _csv(f"{BASE}/members/S{c}_members.csv"):
        try:
            out[m["icpsr"]] = (float(m["nominate_dim1"]), m["party_code"])
        except (ValueError, KeyError):
            continue
    return out


def _party_median(members):
    byp = defaultdict(list)
    for x, p in members.values():
        byp[p].append(x)
    return {p: median(xs) for p, xs in byp.items()}


def _yea(cast):
    return 1 if cast in ("1", "2", "3") else (0 if cast in ("4", "5", "6") else None)


def main():
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    mem = {}
    for c in CONGRESSES:
        try:
            mem[c] = _members(c)
            print(f"  members S{c}: {len(mem[c])}")
        except Exception as e:
            print(f"  members S{c} FAILED: {str(e)[:60]}")
            mem[c] = {}

    bills = []
    for c in CONGRESSES:
        if c - 1 not in mem or not mem[c - 1]:
            continue                                    # need a prior congress for grounding
        prior, cur = mem[c - 1], mem.get(c, {})
        pmed = _party_median(prior)
        try:
            rolls = {r["rollnumber"]: r for r in _csv(f"{BASE}/rollcalls/S{c}_rollcalls.csv")}
            votes = _csv(f"{BASE}/votes/S{c}_votes.csv")
        except Exception as e:
            print(f"  votes S{c} FAILED: {str(e)[:60]}")
            continue
        byroll = defaultdict(list)
        for v in votes:
            y = _yea(v["cast_code"])
            if y is None:
                continue
            icpsr = v["icpsr"]
            party = cur.get(icpsr, (None, None))[1] or prior.get(icpsr, (None, None))[1]
            if party is None:
                continue
            if icpsr in prior:                          # GROUNDED: prior-congress measured ideal point
                x = prior[icpsr][0]
            elif party in pmed:                          # freshman: party's prior median (pre-vote estimate)
                x = pmed[party]
            else:
                continue
            byroll[v["rollnumber"]].append({"icpsr": icpsr, "x": round(x, 4), "party": party, "vote": y})
        for rn, ms in byroll.items():
            yea = sum(m["vote"] for m in ms)
            n = len(ms)
            if n < 40 or yea == 0 or yea == n:           # need a real division of the chamber
                continue
            minority = min(yea, n - yea) / n
            bills.append({"congress": c, "rollnumber": rn, "n": n, "yea": yea,
                          "contested": minority >= 0.2,   # within ~60-40 -> party-line can fail
                          "members": ms})
        print(f"  S{c}: {sum(1 for b in bills if b['congress'] == c)} usable divided votes")

    Path(OUT).write_text(json.dumps(bills))
    print(f"wrote {OUT}: {len(bills)} bills, {sum(len(b['members']) for b in bills)} member-votes, "
          f"{sum(b['contested'] for b in bills)} contested")


if __name__ == "__main__":
    main()
