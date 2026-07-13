"""Phase 3B — forensic diagnosis (Parts A, B, H) on the frozen diagnostic capture.

Produces machine-readable artifacts:
  forensic_decomposition.json  per-question: prior -> ledger -> posterior -> p_phase2 -> p_phase3, whether the
                               movement was causally sensible, and whether the movement helped or hurt vs the
                               realized outcome. Largest |Brier delta| regressions flagged.
  double_counting.json         redundancy analysis: corr(logit p2, logit p3), the learned stack's Phase-3
                               coefficient c (Phase-3 information BEYOND Phase-2), claim-flow counts, and the
                               mechanism verdict (override vs additive double-count).
  dev_repaired_eval.json       DEV-set scores for prior/phase2/phase3_current/phase3_repaired (optimistic —
                               repaired is fit on this set; the honest test is the LOCKED set).

Also reproduces the committed backtest scoring and reports live-retrieval DRIFT of this capture vs committed.
"""
from __future__ import annotations
import json, math
from pathlib import Path

from experiments.phase3b_offline import load_capture, logloss, brier, rate_posterior, fidelity_check
from swm.world_model_v2.phase3b_repair import load_params, repaired_from_capture_row, logit

OUT = Path("experiments/results/phase3b")


def _corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _sensible(direction_net, moved_up):
    """Was the movement causally sensible given the net evidence direction?"""
    if direction_net == 0:
        return "no_net_direction"
    if (direction_net > 0 and moved_up) or (direction_net < 0 and not moved_up):
        return "consistent_with_evidence"
    return "INVERTED_vs_evidence"


def forensic(rows):
    out = []
    for r in rows:
        led = r.get("assimilation_ledger", [])
        n_yes = sum(1 for e in led if e.get("direction") == "supports_yes")
        n_no = sum(1 for e in led if e.get("direction") == "supports_no")
        net = n_yes - n_no
        prior_m = r["prior"].get("mean") or 0.5
        post_m = r.get("posterior_mean")
        moved_up = (post_m is not None and post_m > prior_m)
        y = r["outcome"]
        b2, b3 = brier(r["p_phase2"], y), brier(r["p_phase3"], y)
        out.append({
            "qid": r["qid"], "domain": r["domain"], "outcome": y,
            "prior_mean": round(prior_m, 4), "posterior_mean": post_m,
            "n_effective": r.get("n_effective_observations"),
            "ledger_supports_yes": n_yes, "ledger_supports_no": n_no, "net_direction": net,
            "movement_sensible": _sensible(net, moved_up),
            "p_phase2": r["p_phase2"], "p_phase3": r["p_phase3"],
            "brier_phase2": round(b2, 4), "brier_phase3": round(b3, 4),
            "brier_delta_phase3_minus_phase2": round(b3 - b2, 4),
            "phase3_hurt": b3 > b2 + 1e-9,
            "phase3_terminal_vs_phase2_move": round(r["p_phase3"] - r["p_phase2"], 4),
            "large_regression": (b3 - b2) > 0.10})
    out.sort(key=lambda x: -x["brier_delta_phase3_minus_phase2"])
    return out


def double_counting(rows, params):
    x2 = [logit(r["p_phase2"]) for r in rows]
    x3 = [logit(r["p_phase3"]) for r in rows]
    corr = _corr(x2, x3)
    st = params.get("stack", {})
    # claim flow: every non-neutral effective obs updates Phase-3; Phase-2's recompile also reads the bundle
    total_nonneutral = sum(sum(1 for e in r.get("assimilation_ledger", []) if e.get("direction") != "neutral")
                           for r in rows)
    return {
        "n": len(rows),
        "corr_logit_phase2_phase3": round(corr, 4) if corr is not None else None,
        "learned_stack": {"a": st.get("a"), "b_phase2": st.get("b"), "c_phase3": st.get("c")},
        "interpretation": (
            "Mechanism: when consumed, the Phase-3 posterior particles OVERRIDE the terminal rate (materialize "
            "-> _inject_posterior_rate), so Phase-3 does not ADD to Phase-2 — it REPLACES the rate with its own "
            "assimilation of the SAME bundle. This is redundant/competing assimilation, not additive double-"
            "counting. The learned stack coefficient c on logit(p_phase3) measures the INDEPENDENT information "
            "Phase-3 carries beyond Phase-2: c near 0 => Phase-3 adds ~nothing beyond Phase-2 (redundant); "
            "high corr(logit p2, logit p3) confirms the two forecasts move together off the shared evidence."),
        "total_nonneutral_effective_obs_across_dev": total_nonneutral,
        "mechanism_verdict": "override_not_additive"}


def dev_eval(rows, params):
    def score(getp):
        pts = [(getp(r), r["outcome"]) for r in rows]
        pts = [(p, y) for p, y in pts if p is not None]
        b = sum(brier(p, y) for p, y in pts) / len(pts)
        l = sum(logloss(p, y) for p, y in pts) / len(pts)
        bins = {}
        for p, y in pts:
            bins.setdefault(min(9, int(p * 10)), []).append((p, y))
        ece = sum((len(v) / len(pts)) * abs(sum(p for p, _ in v) / len(v) - sum(yy for _, yy in v) / len(v))
                  for v in bins.values())
        return {"n": len(pts), "brier": round(b, 4), "log_loss": round(l, 4), "ece": round(ece, 4)}
    reps = {r["qid"]: repaired_from_capture_row(r, params) for r in rows}
    return {
        "NOTE": "DEV set — repaired is FIT on this set; optimistic. The honest number is the LOCKED test.",
        "prior_only": score(lambda r: r["prior"].get("mean")),
        "phase2": score(lambda r: r["p_phase2"]),
        "phase3_current": score(lambda r: r["p_phase3"]),
        "phase3_repaired_devfit": score(lambda r: reps[r["qid"]]["repaired_p"]),
        "gate_modes": {m: sum(1 for v in reps.values() if v["mode"] == m)
                       for m in set(v["mode"] for v in reps.values())}}


def committed_reproduction():
    p = Path("experiments/results/phase3/real_backtest.json")
    if not p.exists():
        return {"error": "committed artifact missing"}
    d = json.loads(p.read_text())
    rows = [r for r in d["rows"] if r.get("arms", {}).get("phase3_posterior") is not None
            and r.get("outcome") in (0, 1)]
    def sc(arm):
        pts = [(r["arms"][arm], r["outcome"]) for r in rows if r["arms"].get(arm) is not None]
        return {"brier": round(sum(brier(p, y) for p, y in pts) / len(pts), 4),
                "log_loss": round(sum(logloss(p, y) for p, y in pts) / len(pts), 4)}
    return {"n": len(rows), "phase2": sc("phase2_no_posterior"), "phase3": sc("phase3_posterior"),
            "committed_verdict": d["aggregate"]["verdict"]}


def main():
    rows = load_capture()
    params = load_params()
    fid = fidelity_check(rows)
    forensic_rows = forensic(rows)
    dc = double_counting(rows, params)
    de = dev_eval(rows, params)
    committed = committed_reproduction()
    # drift: fresh capture aggregate vs committed
    def agg(getp):
        pts = [(getp(r), r["outcome"]) for r in rows]
        return {"brier": round(sum(brier(p, y) for p, y in pts) / len(pts), 4),
                "log_loss": round(sum(logloss(p, y) for p, y in pts) / len(pts), 4)}
    drift = {"fresh_capture_phase2": agg(lambda r: r["p_phase2"]),
             "fresh_capture_phase3": agg(lambda r: r["p_phase3"]),
             "committed": committed,
             "note": "live retrieval drifts vs the committed backtest; the committed negative result stays "
                     "frozen. This capture is the DEV substrate for diagnosis + fitting only."}
    (OUT / "forensic_decomposition.json").write_text(json.dumps(
        {"fidelity_offline_vs_captured": fid, "rows": forensic_rows}, indent=2))
    (OUT / "double_counting.json").write_text(json.dumps(dc, indent=2))
    (OUT / "dev_repaired_eval.json").write_text(json.dumps({"drift": drift, "dev_eval": de}, indent=2))
    print("fidelity:", fid)
    print("committed reproduction:", json.dumps(committed))
    print("drift:", json.dumps(drift, indent=2))
    print("double-counting corr(logit p2,p3):", dc["corr_logit_phase2_phase3"], "stack c:", dc["learned_stack"])
    print("\nlargest regressions (fresh capture):")
    for x in forensic_rows[:8]:
        print(f"  {x['qid']:15s} y={x['outcome']} p2={x['p_phase2']} p3={x['p_phase3']} "
              f"dB={x['brier_delta_phase3_minus_phase2']:+.3f} net={x['net_direction']:+d} "
              f"{x['movement_sensible']}")
    print("\nDEV eval:", json.dumps(de, indent=2))


if __name__ == "__main__":
    main()
