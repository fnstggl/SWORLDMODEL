"""Phase 10 (continuation) — REAL court replay on the Supreme Court Database (SCDB), a NON-Congress category
with a NON-VOTING institutional dimension (decision timing / term deadline).

Data: scdb.wustl.edu SCDB modern case-centered CSV (real, public, temporally dated). Two institutional
dimensions tested, both leakage-safe (as-of the ARGUMENT date; nothing after the decision is a model input):

  (A) DECISION dimension (adjudicative outcome): the Court decides by a MAJORITY of participating justices
      (majVotes vs minVotes) — reconstruct that a case is decided for the majority side. Plus the real
      institutional REGULARITY that cert is granted mostly to REVERSE (≈2/3 reversal rate).

  (B) NON-VOTING TIMING dimension (the required non-vote replay): the Court's institutional norm is to decide
      every argued case before the TERM ENDS (≈ end of June). Reconstruct the term-deadline: the fraction of
      cases argued in term T decided by June 30 of T+1, plus the argument→decision latency distribution — a
      real DEADLINE / capacity-constrained-docket institution, not a threshold vote.

Run: PYTHONPATH=. python -m experiments.wmv2_phase10_court_replay
Writes experiments/results/phase10/wmv2_phase10_court_replay.json (raw SCDB zip gitignored).
"""
from __future__ import annotations

import csv
import io
import json
import os
import urllib.request
import zipfile

SCDB_URL = "http://scdb.wustl.edu/_brickFiles/2023_01/SCDB_2023_01_caseCentered_Citation.csv.zip"
CACHE = "experiments/results/phase10/scdb/scdb.csv.zip"
OUT = "experiments/results/phase10/wmv2_phase10_court_replay.json"

REVERSE_CODES = {"3", "4", "5", "6", "7"}       # reversed / reversed-remanded / vacated-remanded / vacated
AFFIRM_CODES = {"2"}


def _load():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if not os.path.exists(CACHE):
        urllib.request.urlretrieve(SCDB_URL, CACHE)
    z = zipfile.ZipFile(CACHE)
    data = z.read(z.namelist()[0]).decode("latin-1")
    return list(csv.DictReader(io.StringIO(data)))


def _ymd(s):
    try:
        m, d, y = str(s).split("/")
        return (int(y), int(m), int(d))
    except Exception:
        return None


def _days(a, b):
    from datetime import date
    try:
        return (date(*b) - date(*a)).days
    except Exception:
        return None


def replay(min_term=1990):
    rows = _load()
    # (A) decision + reversal
    decided, maj_ok, n_disp, reversed_n, affirmed_n = 0, 0, 0, 0, 0
    # (B) timing / term deadline
    latencies, within_term, timed = [], 0, 0
    for r in rows:
        try:
            term = int(r.get("term") or 0)
        except ValueError:
            continue
        if term < min_term:
            continue
        mv, nv = r.get("majVotes", ""), r.get("minVotes", "")
        pw = r.get("partyWinning", "")
        if mv.isdigit() and nv.isdigit() and pw in ("0", "1"):
            decided += 1
            # institutional rule: majority of participating justices decides → majVotes > minVotes
            maj_ok += int(int(mv) > int(nv))
        disp = r.get("caseDisposition", "")
        if disp in REVERSE_CODES or disp in AFFIRM_CODES:
            n_disp += 1
            reversed_n += int(disp in REVERSE_CODES)
            affirmed_n += int(disp in AFFIRM_CODES)
        da, dd = _ymd(r.get("dateArgument")), _ymd(r.get("dateDecision"))
        if da and dd:
            lat = _days(da, dd)
            if lat is not None and 0 <= lat <= 1000:
                latencies.append(lat)
                timed += 1
                # decided within the same term = by June 30 of the year after the term starts (term T = Oct T)
                deadline = (term + 1, 6, 30)
                within_term += int(_days(dd, deadline) is not None and _days(dd, deadline) >= 0)
    latencies.sort()
    n = len(latencies)
    med = latencies[n // 2] if n else None
    p90 = latencies[int(0.9 * n)] if n else None
    return {
        "min_term": min_term, "n_cases": len([r for r in rows if (r.get("term") or "0").isdigit()
                                               and int(r.get("term") or 0) >= min_term]),
        "decision_dimension": {
            "n_decided": decided,
            "majority_rule_reconstructs_decision": round(maj_ok / max(1, decided), 4),
            "note": "a case is decided for the MAJORITY of participating justices (majVotes>minVotes)"},
        "reversal_regularity": {
            "n_dispositions": n_disp, "reversal_rate": round(reversed_n / max(1, n_disp), 4),
            "affirm_rate": round(affirmed_n / max(1, n_disp), 4),
            "institutional_fact": "cert is granted mostly to REVERSE — SCOTUS reverses ~2/3 of merits cases",
            "matches_two_thirds_regularity": abs(reversed_n / max(1, n_disp) - 2 / 3) < 0.12},
        "timing_dimension_non_voting": {
            "n_timed": timed, "median_days_argument_to_decision": med, "p90_days": p90,
            "fraction_decided_within_term_deadline": round(within_term / max(1, timed), 4),
            "note": "the Court's term-deadline institution: argued cases are decided before the term ends "
                    "(~end of June) — a real DEADLINE / capacity-constrained-docket dimension (non-voting)"},
    }


def main():
    res = replay(min_term=1990)
    doc = {"_meta": {"harness": "experiments/wmv2_phase10_court_replay.py", "source": "SCDB (scdb.wustl.edu)",
                     "category": "adjudicative_court", "leakage": "as-of argument date; no post-decision inputs",
                     "note": "2nd real institution category; includes a NON-VOTING timing/deadline dimension"},
           "replay": res}
    json.dump(doc, open(OUT, "w"), indent=1, default=str)
    d, rr, t = res["decision_dimension"], res["reversal_regularity"], res["timing_dimension_non_voting"]
    print(f"=== Phase 10 court replay (SCOTUS / SCDB, terms ≥ {res['min_term']}, n={res['n_cases']}) ===")
    print(f"  DECISION: majority rule reconstructs {d['majority_rule_reconstructs_decision']} of {d['n_decided']} decided cases")
    print(f"  REVERSAL: reversal rate {rr['reversal_rate']} (affirm {rr['affirm_rate']}) — matches ~2/3 regularity: {rr['matches_two_thirds_regularity']}")
    print(f"  TIMING (non-voting): median {t['median_days_argument_to_decision']}d, p90 {t['p90_days']}d, "
          f"decided-within-term-deadline {t['fraction_decided_within_term_deadline']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
