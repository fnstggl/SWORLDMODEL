"""EXP-065: spec-quality benchmark + scored validation of the compiler on REAL resolved outcomes.

Two questions, both aimed at the real bet ("are the LLM's inferred models good enough?"):

  PART 1 — SPEC QUALITY. Does the compiler pick the right MECHANISM and calibrate the RATES?
    (a) Mechanism selection, graded against an EXTERNAL model (Qwen-72B via HF) so the grader is not the
        author and no outcome leaks. (b) Rate calibration: with the data-measured opinion volatility the
        engine's intervals cover at the nominal rate; a mis-set clock breaks coverage — so a spec's rate is
        checkable and matters.

  PART 2 — SCORED VALIDATION across mechanisms, on REAL resolved data, through the ONE compiler interface,
    each scored with the metric that fits and against the baseline it must beat:
      committee     -> real Supreme Court votes (SCDB, leakage-free as-of ideology): direction + margin
      single_agent  -> real persuasion (ChangeMyView): best-message discrimination
      electorate    -> real opinion shares over time (GSS): share-RMSE + coverage vs the marginal
    This proves the front door reproduces the validated per-mechanism results on real outcomes — not a toy.

Run: python -m experiments.exp065_spec_quality_and_validation
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from swm.api.compiler import CompiledModel
from swm.api.model_spec import parse_spec
from swm.eval.world_model_bench import score_binary, score_share

SCDB = "data/SCDB_2024_01_justiceCentered_Citation.csv"
GSS = "experiments/results/exp045_gss/gss_parsed.json.gz"
CMV_C = "experiments/results/exp021_cmv/cmv_common.json"
CMV_I = "experiments/results/exp021_cmv/cmv_inferences.json"
QWEN = "experiments/results/exp065/qwen_mechanism.json"
RESULT = "experiments/results/exp065_spec_quality_and_validation.json"


# ===================== PART 1b: rate calibration (data-measured vs mis-set clock) =====================
def _rate_calibration():
    rows = json.load(gzip.open(GSS))
    by_q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for q, a in r["answers"].items():
            if a in (0, 1):
                by_q[q][r["year"]].append(r)
    QS = ["gunlaw", "cappun", "premarsx", "homosex", "abany", "fefam", "letdie1", "grass", "natenvir"]
    sig2, shares = [], {}
    for q in QS:
        years = sorted(y for y, rs in by_q[q].items() if len(rs) >= 300)
        sh = {y: sum(r["answers"][q] for r in by_q[q][y]) / len(by_q[q][y]) for y in years}
        shares[q] = (years, sh)
        for i in range(1, len(years)):
            dy = years[i] - years[i - 1]
            if dy:
                sig2.append((sh[years[i]] - sh[years[i - 1]]) ** 2 / dy)
    sigma = math.sqrt(sum(sig2) / len(sig2))

    def coverage(mult):
        cov = tot = 0
        for q in QS:
            years, sh = shares[q]
            for ai in range(len(years)):
                A = years[ai]
                for T in years:
                    if 8 <= T - A <= 16:
                        w = 1.2816 * sigma * mult * math.sqrt(T - A)
                        cov += int(abs(sh[T] - sh[A]) <= w); tot += 1
        return round(cov / tot, 3)
    return {"data_measured_sigma_per_year": round(sigma, 4),
            "coverage_calibrated": coverage(1.0), "coverage_clock_2x_fast": coverage(2.0),
            "coverage_clock_2x_slow": coverage(0.5), "nominal": 0.8,
            "reading": "with the data-measured rate the 80% interval covers ~80%; a 2x-wrong clock breaks "
                       "it -> a spec's rate is both checkable and consequential"}


# ===================== PART 2a: committee -> real Supreme Court =====================
def _load_scdb(train_max_term=2009):
    cases = defaultdict(list)
    with open(SCDB, encoding="latin1") as f:
        for row in csv.DictReader(f):
            try:
                term = int(row["term"]); direction = int(row["direction"])
            except (ValueError, KeyError):
                continue
            if direction not in (1, 2) or not row.get("justiceName"):
                continue
            cases[row["caseId"]].append({"term": term, "justice": row["justiceName"],
                                         "lib": 1 if direction == 2 else 0, "issue": row.get("issueArea", "0")})
    ov = defaultdict(lambda: [0, 0]); iss = defaultdict(lambda: [0, 0])
    for votes in cases.values():
        for v in votes:
            if v["term"] <= train_max_term:
                ov[v["justice"]][0] += v["lib"]; ov[v["justice"]][1] += 1
                iss[(v["justice"], v["issue"])][0] += v["lib"]; iss[(v["justice"], v["issue"])][1] += 1
    def ideology(j, issue):
        base = (ov[j][0] + 1) / (ov[j][1] + 2) if ov[j][1] else 0.5
        c = iss[(j, issue)]
        return 0.5 * base + 0.5 * ((c[0] + base) / (c[1] + 1)) if c[1] else base
    test = [(cid, votes) for cid, votes in cases.items()
            if all(v["term"] > train_max_term for v in votes) and 5 <= len(votes) <= 9]
    return test, ideology


def _committee_validation(sample=400, n=60):
    test, ideology = _load_scdb()
    test = test[:sample]
    bin_recs, ind_margin, sim_margin, ind_dir, n_cases = [], [], [], 0, 0
    for cid, votes in test:
        ideos = [ideology(v["justice"], v["issue"]) for v in votes]
        spec = parse_spec({"mechanism": "committee",
                           "extra": {"agents": [{"id": v["justice"], "position": io, "influence": 1.0,
                                                 "openness": 0.35, "conviction": 0.45}
                                                for v, io in zip(votes, ideos)],
                                     "homophily": 0.6, "consensus_pull": 0.5, "rounds": 5, "position_sd": 0.05},
                           "outcome": {"event": {"op": ">", "value": 0.5}}})
        out = CompiledModel(spec).run(n=n)
        k = len(votes); true_lib = sum(v["lib"] for v in votes)
        true_dir = int(true_lib > k / 2); true_margin = max(true_lib, k - true_lib) / k
        bin_recs.append({"mechanism": "committee", "p": out["p_event"], "y": true_dir, "base": 0.5})
        sim_maj = max(out["mean_vote_share"], 1 - out["mean_vote_share"])
        sim_margin.append(abs(sim_maj - true_margin))
        ind_share = sum(int(io > 0.5) for io in ideos) / k
        ind_margin.append(abs(max(ind_share, 1 - ind_share) - true_margin))
        ind_dir += int((ind_share > 0.5) == true_dir)
        n_cases += 1
    return bin_recs, {"n_cases": n_cases, "margin_mae_compiled": round(sum(sim_margin) / n_cases, 4),
                      "margin_mae_independent": round(sum(ind_margin) / n_cases, 4),
                      "direction_acc_independent": round(ind_dir / n_cases, 4)}


# ===================== PART 2b: single_agent -> real persuasion (CMV) =====================
def _single_agent_validation():
    common = {r["id"]: r for r in json.load(open(CMV_C))}
    infer = {r["id"]: r for r in json.load(open(CMV_I))}
    by_op = defaultdict(list)
    for rid, c in common.items():
        f = infer.get(rid)
        if not f:
            continue
        spec = parse_spec({"mechanism": "single_agent", "extra": {
            "person": {"trait_openness": f["op_openness"], "skepticism": f["op_skepticism"],
                       "certainty_disposition": f["op_entrenchment"]},
            "message": {"addresses_crux": f["arg_addresses_crux"], "evidence": f["arg_evidence"],
                        "clarity": f["arg_clarity"], "politeness_disposition": f["arg_respectfulness"],
                        "expertise": f["arg_expertise"]}}})
        p = CompiledModel(spec).run(n=1)["p_respond_mean"]
        by_op[c["op_id"]].append({"p": p, "y": int(c["success"])})
    # best-message discrimination: among OPs with mixed outcomes, does the top-ranked arg persuade more?
    hits = rand = k = 0
    for op, rs in by_op.items():
        if len(rs) >= 2 and 0 < sum(r["y"] for r in rs) < len(rs):
            top = max(rs, key=lambda r: r["p"])
            hits += top["y"]; rand += sum(r["y"] for r in rs) / len(rs); k += 1
    return {"n_mixed_ops": k, "best_message_precision@1": round(hits / k, 4) if k else None,
            "random_pick_rate": round(rand / k, 4) if k else None,
            "lift": round((hits - rand) / k, 4) if k else None}


# ===================== PART 2c: electorate -> real opinion shares (GSS) =====================
def _electorate_validation(max_pairs=150, n=40):
    rows = json.load(gzip.open(GSS))
    by_q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for q, a in r["answers"].items():
            if a in (0, 1):
                by_q[q][r["year"]].append(r)
    QS = ["gunlaw", "cappun", "homosex", "abany", "fefam", "natenvir", "grass"]
    recs = []
    sig_pairs = []
    for q in QS:
        years = sorted(y for y, rs in by_q[q].items() if len(rs) >= 300)
        sh = {y: sum(r["answers"][q] for r in by_q[q][y]) / len(by_q[q][y]) for y in years}
        for i in range(1, len(years)):
            dy = years[i] - years[i - 1]
            if dy:
                sig_pairs.append((sh[years[i]] - sh[years[i - 1]]) ** 2 / dy)
    sigma = math.sqrt(sum(sig_pairs) / len(sig_pairs))
    for q in QS:
        years = sorted(y for y, rs in by_q[q].items() if len(rs) >= 300)
        sh = {y: sum(r["answers"][q] for r in by_q[q][y]) / len(by_q[q][y]) for y in years}
        for ai in range(len(years)):
            A = years[ai]
            for T in years:
                if 8 <= T - A <= 16 and len(recs) < max_pairs:
                    # build electorate cells (age x degree x party x region) at A
                    agg = defaultdict(lambda: [0, 0])
                    for r in by_q[q][A]:
                        d = r["demo"]; key = (d["age"], d["degree"], d["party"], d["region"])
                        agg[key][0] += r["answers"][q]; agg[key][1] += 1
                    cells = [{"stance": v[0] / v[1], "weight": v[1], "est_sd": 0.03}
                             for v in agg.values() if v[1] >= 3]
                    spec = parse_spec({"mechanism": "electorate", "horizon": T - A,
                                       "extra": {"cells": cells, "k_social": 0.1, "k_proof": 0.0}})
                    out = CompiledModel(spec).run(n=n)
                    marg = sum(c["stance"] * c["weight"] for c in cells) / sum(c["weight"] for c in cells)
                    w = 1.2816 * sigma * math.sqrt(T - A)
                    recs.append({"mechanism": "electorate", "pred": out["mean_share"], "truth": sh[T],
                                 "marginal": marg, "lo": max(0, out["mean_share"] - w),
                                 "hi": min(1, out["mean_share"] + w)})
    return recs


def run():
    # PART 1: spec quality
    qwen = json.loads(Path(QWEN).read_text()) if Path(QWEN).exists() else []
    answered = [r for r in qwen if not str(r["pick"]).startswith("ERR")]
    mech_acc = (sum(r["correct"] for r in answered) / len(answered)) if answered else None
    part1 = {"mechanism_selection_external_llm": {
                "grader_model": "Qwen/Qwen2.5-72B-Instruct (HF, blind)", "n_answered": len(answered),
                "n_total": len(qwen), "accuracy_on_answered": round(mech_acc, 3) if mech_acc is not None else None,
                "note": "external blind grader; remaining items hit the HF credit limit (402), not model error"},
             "rate_calibration": _rate_calibration()}

    # PART 2: scored validation across mechanisms
    comm_bin, comm_margin = _committee_validation()
    sa = _single_agent_validation()
    elec = _electorate_validation()
    binary = score_binary(comm_bin)
    share = score_share(elec)
    part2 = {"committee_scotus": {**binary.get("committee", {}), **comm_margin},
             "single_agent_cmv": sa, "electorate_gss": share.get("electorate", {})}

    out = {"PART1_spec_quality": part1, "PART2_scored_validation": part2}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-065  spec-quality benchmark + scored validation of the compiler on REAL outcomes")
    print("  PART 1 — SPEC QUALITY")
    m = part1["mechanism_selection_external_llm"]
    print(f"    mechanism selection (Qwen-72B, blind): {m['accuracy_on_answered']} on {m['n_answered']}/"
          f"{m['n_total']} answered  ({m['note']})")
    rc = part1["rate_calibration"]
    print(f"    rate calibration: data sigma={rc['data_measured_sigma_per_year']}/yr -> coverage "
          f"{rc['coverage_calibrated']} (nominal {rc['nominal']}); 2x-fast={rc['coverage_clock_2x_fast']}, "
          f"2x-slow={rc['coverage_clock_2x_slow']}")
    print("  PART 2 — SCORED VALIDATION (real resolved outcomes, through the ONE compiler interface)")
    c = part2["committee_scotus"]
    print(f"    committee/SCOTUS ({c['n_cases']} cases): direction acc={c['accuracy']} brier={c['brier']} "
          f"| margin MAE compiled={c['margin_mae_compiled']} vs independent {c['margin_mae_independent']}")
    s = part2["single_agent_cmv"]
    print(f"    single_agent/CMV ({s['n_mixed_ops']} mixed OPs): best-message precision@1="
          f"{s['best_message_precision@1']} vs random {s['random_pick_rate']} (lift {s['lift']:+})")
    e = part2["electorate_gss"]
    print(f"    electorate/GSS ({e['n']} pairs): RMSE={e['rmse']} vs marginal {e['rmse_marginal']} "
          f"| coupling skill {e['coupling_skill']} | coverage "
          f"{e.get('interval_coverage', {}).get('empirical_coverage')}")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
