"""Phase 10 (continuation) — automatic rule reconstruction on 2 real documents vs verified ground truth (#4).

Runs the extract → ground → validate pipeline on the VERBATIM text of two materially different authoritative
documents (US Constitution Art I §5/§7; Delaware GCL §141(b)) and scores the extracted typed rules against
the manually-verified ground truth (the rules hand-encoded in institutions_v2/build.py). The LLM proposes
formalizations; only source-span-grounded, deterministically-valid rules are accepted.

Run: PYTHONPATH=. python -m experiments.wmv2_phase10_extract
Writes experiments/results/phase10/wmv2_phase10_extraction.json
"""
from __future__ import annotations

import json

from swm.world_model_v2.institutions_v2.extract import extract_rules

OUT = "experiments/results/phase10/wmv2_phase10_extraction.json"

ART1 = ("US Constitution, Article I. Section 5: a Majority of each House shall constitute a Quorum to do "
        "Business. Section 7: Every Bill which shall have passed the House of Representatives and the Senate, "
        "shall, before it become a Law, be presented to the President of the United States; If he approve he "
        "shall sign it, but if not he shall return it, with his Objections to that House in which it shall "
        "have originated, who shall proceed to reconsider it. If after such Reconsideration two thirds of "
        "that House shall agree to pass the Bill, it shall be sent, together with the Objections, to the "
        "other House, by which it shall likewise be reconsidered, and if approved by two thirds of that "
        "House, it shall become a Law.")

DGCL = ("Delaware General Corporation Law Section 141(b): A majority of the total number of directors shall "
        "constitute a quorum for the transaction of business unless the certificate of incorporation or the "
        "bylaws require a greater number. The vote of the majority of the directors present at a meeting at "
        "which a quorum is present shall be the act of the board of directors unless the certificate of "
        "incorporation or the bylaws shall require a vote of a greater number.")

# manually-verified ground truth: (rule_type, fraction) pairs the document actually states
GROUND_TRUTH = {
    "art1": [("quorum", 0.5), ("threshold", 0.5), ("override", 2 / 3)],
    "dgcl": [("quorum", 0.5), ("threshold", 0.5)],
}


def _score(extracted_rules, truth):
    got = set()
    for r in extracted_rules:
        rt = "threshold" if r.kind == "threshold" else r.kind
        got.add((rt, round(float(r.params.get("fraction", 0.5)), 2)))
    truth_set = {(t, round(f, 2)) for t, f in truth}
    tp = len(got & truth_set)
    precision = tp / max(1, len(got))
    recall = tp / max(1, len(truth_set))
    return {"extracted": sorted(str(g) for g in got), "ground_truth": sorted(str(g) for g in truth_set),
            "true_positives": tp, "precision": round(precision, 3), "recall": round(recall, 3),
            "missed": sorted(str(g) for g in (truth_set - got)),
            "spurious": sorted(str(g) for g in (got - truth_set))}


def main():
    docs = {"art1": (ART1, "usconst_art1", "US-federal", "1789-03-04"),
            "dgcl": (DGCL, "dgcl_141b", "US-DE", "1969-07-03")}
    out = {"_meta": {"harness": "experiments/wmv2_phase10_extract.py",
                     "note": "LLM proposes formalizations; only source-span-grounded + deterministically-valid "
                             "rules accepted; scored vs manually-verified ground truth"}, "documents": {}}
    for name, (text, sid, jur, eff) in docs.items():
        res = extract_rules(text, source_id=sid, jurisdiction=jur, effective_date=eff)
        score = _score(res["rules"], GROUND_TRUTH[name])
        out["documents"][name] = {
            "source_id": sid, "source_tag": res["source_tag"], "n_candidates": res["n_candidates"],
            "n_accepted": len(res["rules"]), "n_rejected": len(res["rejected"]),
            "score": score,
            "accepted_rules": [{"rule_id": r.rule_id, "kind": r.kind, "params": r.params} for r in res["rules"]],
            "rejected": [{"reason": x["reason"][:100]} for x in res["rejected"]],
        }
    macro_p = sum(d["score"]["precision"] for d in out["documents"].values()) / len(out["documents"])
    macro_r = sum(d["score"]["recall"] for d in out["documents"].values()) / len(out["documents"])
    out["macro_precision"] = round(macro_p, 3)
    out["macro_recall"] = round(macro_r, 3)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print("=== Phase 10 automatic rule extraction (vs verified ground truth) ===")
    for name, d in out["documents"].items():
        s = d["score"]
        print(f"  {name} [{d['source_tag']}]: {d['n_accepted']} accepted / {d['n_rejected']} rejected "
              f"| precision {s['precision']} recall {s['recall']} | missed {s['missed']} spurious {s['spurious']}")
    print(f"  macro precision {out['macro_precision']}  recall {out['macro_recall']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
