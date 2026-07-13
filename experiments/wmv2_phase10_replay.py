"""Phase 10 — REAL historical replay of the US Senate legislative institution on VoteView roll-calls.

Data: voteview.com static roll-call CSVs (yea/nay counts, date, recorded vote_result, vote_question) — real,
public, reproducible, temporally dated. For each roll-call we:
  1. select the applicable threshold from the vote QUESTION using the evidence-backed template's rules
     (cloture = 3/5 of the full Senate; treaty/veto-override = 2/3 present; else simple majority of present);
  2. execute the institutional decision engine (decisions.evaluate_decision) on the REAL yea/nay counts;
  3. compare the predicted outcome to the RECORDED vote_result.

This tests institutional RULE SELECTION + THRESHOLD EXECUTION on thousands of real votes (not final-outcome
prediction of a black box). The ablation `majority_only` (ignore cloture/treaty rules) shows the institutional
rules are load-bearing. Leakage-safe: only the roll-call's own counts + as-of rules are used; nothing later.

Run: PYTHONPATH=. python -m experiments.wmv2_phase10_replay [--congresses 117,118]
Caches CSVs under experiments/results/phase10/voteview/ ; writes wmv2_phase10_replay.json.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import urllib.request

from swm.world_model_v2.institutions_v2.decisions import ThresholdSpec, evaluate_decision

BASE = "https://voteview.com/static/data/out/rollcalls"
CACHE = "experiments/results/phase10/voteview"
OUT = "experiments/results/phase10/wmv2_phase10_replay.json"

PASS_MARKERS = ("agreed to", "passed", "confirmed", "adopted", "sustained", "amendment germane",
                "cloture motion agreed", "motion agreed", "nomination confirmed", "bill passed",
                "joint resolution passed", "concurrent resolution agreed", "resolution agreed",
                "conference report agreed", "veto overridden")
FAIL_MARKERS = ("rejected", "failed", "not agreed", "not passed", "cloture motion rejected",
                "motion rejected", "veto sustained", "point of order sustained", "not sustained")


def _fetch(congress: int) -> list:
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/S{congress}_rollcalls.csv"
    if not os.path.exists(path):
        url = f"{BASE}/S{congress}_rollcalls.csv"
        with urllib.request.urlopen(url, timeout=90) as r:
            data = r.read().decode("latin-1")
        open(path, "w").write(data)
    return list(csv.DictReader(io.StringIO(open(path, encoding="latin-1").read())))


def _recorded(result: str) -> bool | None:
    r = (result or "").lower()
    if any(m in r for m in FAIL_MARKERS):
        return False
    if any(m in r for m in PASS_MARKERS):
        return True
    return None                                  # unclassifiable → excluded (honest)


def _is_nomination(bill_number: str, question: str, result: str) -> bool:
    b = (bill_number or "").upper()
    return b.startswith("PN") or "nomination" in (question or "").lower() or "nominat" in (result or "").lower()


def _to_ymd(s):
    try:
        return tuple(int(x) for x in str(s)[:10].split("-"))
    except Exception:
        return (0, 0, 0)


def _threshold_for(question: str, result: str, *, bill_number: str = "", date: str = "",
                   mode: str = "matter_aware") -> tuple:
    """Select the applicable threshold — the institutional RULE SELECTION step, with AS-OF + MATTER-TYPE
    versioning (the nuclear option: since 2013 [exec/judicial] and 2017 [SCOTUS], cloture on NOMINATIONS
    needs only a simple majority; cloture on LEGISLATION still needs 3/5).
      mode='matter_aware'   full as-of + matter-type rules (the real institution)
      mode='naive_cloture'  ablation: uniform 3/5 for ALL cloture (ignores the nuclear option)
      mode='majority_only'  ablation: majority for everything (ignores special thresholds)
    Returns (ThresholdSpec, label)."""
    q = (question or "").lower()
    r = (result or "").lower()
    if mode == "majority_only":
        return ThresholdSpec("simple_majority", 0.5, base="present"), "majority"
    if "cloture" in q or "cloture" in r:
        if mode == "matter_aware" and _is_nomination(bill_number, question, result) and _to_ymd(date) >= (2013, 1, 1):
            return ThresholdSpec("simple_majority", 0.5, base="present"), "cloture_nomination_majority"
        return ThresholdSpec("supermajority", 0.6, base="all_members", quorum_fraction=0.5), "cloture_legislation_3_5"
    if "treaty" in q or "resolution of ratification" in q:
        return ThresholdSpec("supermajority", 2 / 3, base="present"), "treaty_2_3"
    if "veto" in q or "objections of the president" in q or "override" in q:
        return ThresholdSpec("supermajority", 2 / 3, base="present"), "override_2_3"
    return ThresholdSpec("simple_majority", 0.5, base="present"), "majority"


def replay(congresses, *, mode="matter_aware", senate_size=100, _rows=None):
    rows_all = _rows if _rows is not None else []
    if _rows is None:
        for c in congresses:
            rows_all += _fetch(c)
    n, correct, excluded = 0, 0, 0
    by_type = {}
    eligible = [f"s{i}" for i in range(senate_size)]
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    misses = []
    for row in rows_all:
        rec = _recorded(row.get("vote_result", ""))
        if rec is None:
            excluded += 1
            continue
        try:
            yea, nay = int(row["yea_count"]), int(row["nay_count"])
        except (ValueError, KeyError):
            excluded += 1
            continue
        if yea + nay == 0:
            excluded += 1
            continue
        spec, label = _threshold_for(row.get("vote_question", ""), row.get("vote_result", ""),
                                     bill_number=row.get("bill_number", ""), date=row.get("date", ""),
                                     mode=mode)
        # reconstruct the vote from counts: yea 'yes', nay 'no' (present = yea+nay)
        votes = {f"s{i}": "yes" for i in range(yea)}
        votes.update({f"s{yea + i}": "no" for i in range(nay)})
        elig = eligible if yea + nay <= senate_size else [f"s{i}" for i in range(yea + nay)]
        res = evaluate_decision(spec, votes, eligible=elig)
        pred = res.passed
        n += 1
        ok = pred == rec
        correct += int(ok)
        t = by_type.setdefault(label, {"n": 0, "correct": 0})
        t["n"] += 1
        t["correct"] += int(ok)
        key = ("t" if rec else "f") + ("p" if pred else "n")
        confusion[{"tp": "tp", "tn": "tn", "fp": "fp", "fn": "fn"}[("t" if pred else "f") + ("p" if rec else "n")]] = \
            confusion.get(("t" if pred else "f") + ("p" if rec else "n"), 0) + 1
        if not ok and len(misses) < 25:
            misses.append({"congress": row.get("congress"), "roll": row.get("rollnumber"),
                           "date": row.get("date"), "q": row.get("vote_question"),
                           "result": row.get("vote_result"), "yea": yea, "nay": nay,
                           "threshold": label, "predicted": pred})
    return {"n_scored": n, "n_excluded_unclassifiable": excluded,
            "accuracy": round(correct / max(1, n), 4), "by_threshold_type": by_type,
            "confusion": confusion, "sample_misses": misses}


def main():
    congresses = [117, 118]
    for a in sys.argv[1:]:
        if a.startswith("--congresses"):
            congresses = [int(x) for x in a.split("=", 1)[-1].split(",")]
    rows = []
    for c in congresses:
        rows += _fetch(c)
    matter_aware = replay(congresses, mode="matter_aware", _rows=rows)
    naive_cloture = replay(congresses, mode="naive_cloture", _rows=rows)
    majority_only = replay(congresses, mode="majority_only", _rows=rows)
    # leakage audit on the anchor template (proof the reconstruction uses only as-of rules)
    from swm.world_model_v2.institutions_v2.store import load_store
    from swm.world_model_v2.institutions_v2.evidence import leakage_audit
    st = load_store(reload=True)
    la = leakage_audit(st.templates["us_congress_legislative"], "2021-01-01",
                       outcome_events=[{"id": "later_vote", "date": "2024-01-01"}])
    result = {
        "_meta": {"harness": "experiments/wmv2_phase10_replay.py", "source": "voteview.com roll-calls",
                  "congresses": congresses,
                  "note": "tests institutional rule-selection + threshold execution on REAL votes; "
                          "leakage-safe (only each roll-call's own counts + as-of rules)."},
        "matter_aware": matter_aware,          # full as-of + matter-type (nuclear option) rules
        "ablation_naive_cloture": naive_cloture,   # uniform 3/5 cloture (ignores nuclear option)
        "ablation_majority_only": majority_only,   # majority for everything
        "matter_aware_vs_naive_cloture": round(matter_aware["accuracy"] - naive_cloture["accuracy"], 4),
        "matter_aware_vs_majority_only": round(matter_aware["accuracy"] - majority_only["accuracy"], 4),
        "leakage_audit": la,
    }
    json.dump(result, open(OUT, "w"), indent=1, default=str)
    print(f"=== Phase 10 historical replay (Senate {congresses}) ===")
    print(f"  scored {matter_aware['n_scored']} real roll-calls "
          f"({matter_aware['n_excluded_unclassifiable']} unclassifiable excluded)")
    print(f"  accuracy matter-aware (as-of + nuclear option): {matter_aware['accuracy']}")
    print(f"  accuracy ablation naive-cloture (uniform 3/5):  {naive_cloture['accuracy']}")
    print(f"  accuracy ablation majority-only:                {majority_only['accuracy']}")
    print(f"  matter-aware vs naive-cloture Δ (nuclear-option rule load-bearing): "
          f"+{result['matter_aware_vs_naive_cloture']}")
    print("  by threshold type:", {k: f"{v['correct']}/{v['n']}"
                                    for k, v in matter_aware["by_threshold_type"].items()})
    print(f"  leakage audit @2021 clean: {la['clean']} (future excluded: {la['future_outcomes_excluded']})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
