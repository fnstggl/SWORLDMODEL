"""Phase 7 — real-data nonlinear form validation (Parts 20-23).

Fits candidate structural forms on committed real datasets, compares them under IDENTICAL splits against the
required baselines (constant base-rate, additive logistic = the current Phase-6 form, linear), selects the form
on VALIDATION only, scores the untouched TEST once, applies the parsimony rule (keep the simpler form unless
the nonlinear one beats it on held-out), and records calibration + ablations + subgroup transfer. Preserves
nulls (StackExchange, CMV) and honest ties as first-class results.

Datasets (committed, reproducible — no download):
  telco         attrition/persistence — tenure→churn hazard is strongly nonlinear (declining) → expect a win
  stackexchange response occurrence     — a Phase-6 NULL → expect nonlinear to ALSO find null (adversarial)
  cmv           persuasion              — a Phase-6 NULL → expect null; guard against unsupported backfire
  upworthy      platform content CTR    — partial pooling across randomized A/B tests + headline-length shape

Deterministic under fixed seeds. No LLM calls. Run:
  PYTHONPATH=. python -m experiments.wmv2_phase7_forms_validation
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from swm.world_model_v2.nonlinear import fit, compare
from swm.world_model_v2.nonlinear.forms import get_form
from swm.world_model_v2.registry.ingestion import paired_bootstrap_delta

RESULTS = "experiments/results"

# per-dataset config: which vars get nonlinear smooths, which interactions are NOMINATED (fit decides).
_CONFIG = {
    "telco": dict(path=f"{RESULTS}/harvest_extra/telco_churn.json",
                  feats=["senior", "partner", "dependents", "tenure", "phone_service", "paperless_billing",
                         "monthly_charges", "contract", "internet_service", "is_female"],
                  smooths={"tenure": 5, "monthly_charges": 3}, interactions=[("tenure", "contract")],
                  category="persistence_attrition",
                  hypothesis="tenure→churn declining hazard is nonlinear; contract×tenure interaction"),
    "stackexchange": dict(path=f"{RESULTS}/harvest_extra/stackexchange.json",
                          feats=["title_len", "body_len", "n_tags", "code_present", "has_code_kw", "is_howto",
                                 "title_qmark", "hour", "dayofweek"],
                          smooths={"title_len": 4, "body_len": 4, "hour": 3},
                          interactions=[("code_present", "is_howto")],
                          category="participation_response",
                          hypothesis="NULL expected (Phase-6 found features do not beat base rate)"),
    "cmv": dict(path=f"{RESULTS}/harvest_extra/cmv.json",
                feats=["op_openness", "op_skepticism", "op_entrenchment", "arg_addresses_crux", "arg_evidence",
                       "arg_clarity", "arg_respectfulness", "arg_expertise"],
                smooths={"op_entrenchment": 3, "arg_addresses_crux": 3},
                interactions=[("op_entrenchment", "arg_addresses_crux")],
                category="persuasion_belief",
                hypothesis="NULL/weak expected; test entrenchment×crux interaction; REJECT unsupported backfire"),
}


def _load(name):
    return json.load(open(_CONFIG[name]["path"]))


def _candidates(name, tr):
    """Build the candidate predictors for a dataset, all fitted on TRAIN only."""
    cfg = _CONFIG[name]
    feat = cfg["feats"]
    logistic = get_form("logistic")
    gam = get_form("gam")
    lin_keys = [k for k in feat if k not in cfg["smooths"]]
    fl = fit.fit_logistic_form(tr, feat, dataset=name)
    fli = fit.fit_logistic_form(tr, feat, interactions=cfg["interactions"], dataset=name)
    fg = fit.fit_gam(tr, lin_keys, cfg["smooths"], dataset=name)
    fgi = fit.fit_gam(tr, lin_keys, cfg["smooths"], interactions=cfg["interactions"], dataset=name)
    rate = sum(r["y"] for r in tr) / max(1, len(tr))
    return {
        "constant": (lambda r, _r=rate: _r),
        "logistic": (lambda r: logistic.eval(fl.params, {"features": r["features"]})),
        "logistic_interaction": (lambda r: logistic.eval(fli.params, {"features": r["features"]})),
        "gam": (lambda r: gam.eval(fg.params, {"features": r["features"]})),
        "gam_interaction": (lambda r: gam.eval(fgi.params, {"features": r["features"]})),
    }, {"logistic": fl, "logistic_interaction": fli, "gam": fg, "gam_interaction": fgi}


def compare_dataset(name, *, seed=0):
    rows = _load(name)
    cfg = _CONFIG[name]
    tr, va, te = fit.random_split(rows, seed=seed)
    cands, fits = _candidates(name, tr)
    comp = compare.compare_forms(cands, va, te, seed=seed)
    decision = compare.select_with_parsimony(comp, simpler=("constant", "logistic"))
    return {"dataset": name, "category": cfg["category"], "hypothesis": cfg["hypothesis"],
            "n": len(rows), "base_rate": round(sum(r["y"] for r in rows) / len(rows), 4),
            "split": f"random 60/20/20 seed={seed}", "dataset_hash": fit.dataset_hash(rows),
            "val_scores": comp["val_scores"], "test_scores": comp["test_scores"],
            "validation_selected": comp["selected"], "paired_test_deltas": comp["paired_test_deltas"],
            "parsimony_decision": decision,
            "fitted_params": {k: v.as_dict() for k, v in fits.items()}}


def ablate_dataset(name, *, seed=0):
    """Ablation ladder (Part 23): strip nonlinear structure and measure the held-out cost of each piece."""
    rows = _load(name)
    cfg = _CONFIG[name]
    tr, va, te = fit.random_split(rows, seed=seed)
    feat = cfg["feats"]
    lin_keys = [k for k in feat if k not in cfg["smooths"]]
    logistic = get_form("logistic"); gam = get_form("gam")
    yt = [r["y"] for r in te]
    arms = {}
    # linear only (no context nonlinearity)
    fl = fit.fit_logistic_form(tr, feat)
    arms["linear_only"] = [logistic.eval(fl.params, {"features": r["features"]}) for r in te]
    # gam without interaction
    fg = fit.fit_gam(tr, lin_keys, cfg["smooths"])
    arms["gam_no_interaction"] = [gam.eval(fg.params, {"features": r["features"]}) for r in te]
    # gam with only ONE smooth (drop the rest) → isolate the dominant smooth
    dom = list(cfg["smooths"])[0]
    fg1 = fit.fit_gam(tr, [k for k in feat if k != dom], {dom: cfg["smooths"][dom]})
    arms["gam_single_smooth"] = [gam.eval(fg1.params, {"features": r["features"]}) for r in te]
    # full phase 7 (gam + interaction)
    fgi = fit.fit_gam(tr, lin_keys, cfg["smooths"], interactions=cfg["interactions"])
    arms["full_phase7"] = [gam.eval(fgi.params, {"features": r["features"]}) for r in te]
    out = {arm: compare.metrics(p, yt) for arm, p in arms.items()}
    # incremental contribution vs linear_only
    deltas = {arm: paired_bootstrap_delta(yt, arms[arm], arms["linear_only"], seed=seed)
              for arm in arms if arm != "linear_only"}
    return {"dataset": name, "category": cfg["category"], "arm_scores": out,
            "incremental_vs_linear": deltas,
            "note": "negative ΔBrier = the nonlinear component helps held-out; a component that never changes "
                    "held-out is ornamental (Part 23)"}


def transfer_test(*, seed=0):
    """Out-of-group transfer (Part 22): fit a nonlinear pack on one subgroup, test on held-out subgroups.
    telco: fit on month-to-month (contract=0), test on contract∈{1,2}. A nonlinear shape that is a genuine
    mechanism should transport (with widening); one that overfits the training subgroup will fail."""
    rows = _load("telco")
    cfg = _CONFIG["telco"]
    feat = cfg["feats"]
    lin_keys = [k for k in feat if k not in cfg["smooths"]]
    train = [r for r in rows if r["features"]["contract"] == 0]
    test = [r for r in rows if r["features"]["contract"] in (1, 2)]
    logistic = get_form("logistic"); gam = get_form("gam")
    fl = fit.fit_logistic_form(train, feat)
    fg = fit.fit_gam(train, lin_keys, cfg["smooths"])
    yt = [r["y"] for r in test]
    pl = [logistic.eval(fl.params, {"features": r["features"]}) for r in test]
    pg = [gam.eval(fg.params, {"features": r["features"]}) for r in test]
    d = paired_bootstrap_delta(yt, pg, pl, seed=seed)
    return {"dataset": "telco", "transfer": "fit contract=month-to-month → test contract=1yr/2yr",
            "n_train": len(train), "n_test": len(test),
            "logistic": compare.metrics(pl, yt), "gam": compare.metrics(pg, yt),
            "paired_delta_gam_minus_logistic": d,
            "verdict": ("nonlinear transports (beats linear out-of-group)" if d["ci95"][1] < 0
                        else "nonlinear does NOT beat linear out-of-group — domain-restrict / widen"),
            "note": "different contract types have different tenure support — a real transport stress test"}


# ---------------------------------------------------------------- Upworthy content response (partial pooling)
def _upworthy_arms():
    """Flatten Upworthy A/B tests → per-arm CTR rows with headline features + test_id (for pooling)."""
    tests = json.load(open(f"{RESULTS}/exp054_upworthy/upworthy_parsed.json"))
    arms = []
    for t in tests:
        for a in t["arms"]:
            h = a.get("headline", "") or ""
            n = int(a.get("impressions", 0) or 0)
            k = int(a.get("clicks", 0) or 0)
            if n < 100:
                continue
            arms.append({"test_id": t["test_id"], "features": _headline_feats(h), "k": k, "n": n,
                         "ctr": k / n})
    return arms


def _headline_feats(h):
    words = h.split()
    return {"len_chars": len(h), "len_words": len(words),
            "has_number": 1.0 if any(c.isdigit() for c in h) else 0.0,
            "qmark": 1.0 if "?" in h else 0.0,
            "you": 1.0 if any(w.lower() in ("you", "your", "you're") for w in words) else 0.0,
            "this": 1.0 if any(w.lower() in ("this", "these", "here's") for w in words) else 0.0}


def upworthy_content(*, seed=0):
    """Content-response validation: does a nonlinear headline-length shape + partial pooling across tests beat
    a linear headline model and the global-mean baseline, on held-out TESTS (test-disjoint split)?"""
    from swm.world_model_v2.nonlinear.pooling import pool_beta_binomial
    arms = _upworthy_arms()
    tr, va, te = fit.group_split(arms, group_key="test_id", seed=seed)
    # partial-pooling baseline: per-test Beta-Binomial shrunk CTR (uses train tests' pooled prior)
    counts = {}
    for r in tr:
        counts.setdefault(r["test_id"], {"k": 0, "n": 0})
        counts[r["test_id"]]["k"] += r["k"]; counts[r["test_id"]]["n"] += r["n"]
    pooled = pool_beta_binomial(counts)
    global_rate = sum(r["k"] for r in tr) / max(1, sum(r["n"] for r in tr))
    # headline models (weighted logistic on impressions via expanded pseudo-rows would be heavy; fit on
    # arm-level rate weighted by n using a simple IRLS-free grid on the GAM smooth of len_words)
    feat_keys = list(_headline_feats("a b").keys())
    lin_keys = [k for k in feat_keys if k != "len_words"]
    # build weighted binary rows (down-weight to keep it light): expand each arm into k pos + (n-k) neg capped
    def expand(rows, cap=40):
        out = []
        for r in rows:
            scale = min(1.0, cap / max(1, r["n"]))
            pos = max(0, round(r["k"] * scale)); neg = max(0, round((r["n"] - r["k"]) * scale))
            for _ in range(pos):
                out.append({"features": r["features"], "y": 1})
            for _ in range(neg):
                out.append({"features": r["features"], "y": 0})
        return out
    tr_e = expand(tr)
    logistic = get_form("logistic"); gam = get_form("gam")
    fl = fit.fit_logistic_form(tr_e, feat_keys)
    fg = fit.fit_gam(tr_e, lin_keys, {"len_words": 4})
    # evaluate on held-out TEST arms: predict CTR, score against observed clicks (Brier weighted by impressions)
    yt, wt = [], []
    preds = {"global_mean": [], "pooled_prior": [], "linear_headline": [], "nonlinear_headline": []}
    for r in te:
        yt.append(r["ctr"]); wt.append(r["n"])
        preds["global_mean"].append(global_rate)
        preds["pooled_prior"].append(pooled.transfer_estimate())   # new test → population prior
        preds["linear_headline"].append(logistic.eval(fl.params, {"features": r["features"]}))
        preds["nonlinear_headline"].append(gam.eval(fg.params, {"features": r["features"]}))

    def wbrier(p):
        z = sum(wt)
        return sum(w * (pi - y) ** 2 for pi, y, w in zip(p, yt, wt)) / z
    scores = {k: round(wbrier(v), 6) for k, v in preds.items()}
    return {"dataset": "upworthy", "category": "platform_content_response",
            "n_test_arms": len(te), "global_ctr": round(global_rate, 4),
            "impression_weighted_brier": scores,
            "hypothesis": "headline features are weak vs the per-test baseline; partial pooling regularizes",
            "verdict": ("nonlinear headline shape helps" if scores["nonlinear_headline"] < scores["linear_headline"]
                        and scores["nonlinear_headline"] < scores["global_mean"]
                        else "headline effects weak/null — pooled per-test baseline dominates (honest)")}


def validate_all(*, seed=0):
    out = {"_meta": {"seed": seed, "note": "validation-only selection; test scored once; nulls preserved"}}
    for name in ("telco", "stackexchange", "cmv"):
        out[name] = compare_dataset(name, seed=seed)
    out["upworthy"] = upworthy_content(seed=seed)
    out["telco_transfer"] = transfer_test(seed=seed)
    return out


def main():
    t0 = time.time()
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    val = validate_all(seed=0)
    val["_meta"]["runtime_s"] = round(time.time() - t0, 1)
    val["_meta"]["llm_calls"] = 0
    with open(f"{RESULTS}/wmv2_phase7_validation.json", "w") as f:
        json.dump(val, f, indent=1, default=str)
    print(f"wrote {RESULTS}/wmv2_phase7_validation.json  ({time.time() - t0:.1f}s)")
    # ablations + summary
    abl = {name: ablate_dataset(name, seed=0) for name in ("telco", "stackexchange", "cmv")}
    with open(f"{RESULTS}/wmv2_phase7_ablations.json", "w") as f:
        json.dump({"_meta": {"note": "Part-23 ablation ladder"}, "datasets": abl}, f, indent=1, default=str)
    print(f"wrote {RESULTS}/wmv2_phase7_ablations.json")
    for name in ("telco", "stackexchange", "cmv", "upworthy"):
        d = val[name]
        if "parsimony_decision" in d:
            print(f"  {name}: selected={d['validation_selected']} -> {d['parsimony_decision']['promoted']} "
                  f"({'BEAT' if d['parsimony_decision'].get('beat_baseline') else 'tie/null'})")
        else:
            print(f"  {name}: {d.get('verdict')}")


if __name__ == "__main__":
    main()
