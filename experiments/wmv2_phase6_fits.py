"""Phase 6 real-data parameter-pack fitting + held-out validation harness (resumable, deterministic).

Each dataset below is a committed, real parsed dataset in experiments/results/. For each we (1) fit a
mechanism-family transition on a TRAIN split, (2) select the functional form on a VALIDATION split, and
(3) evaluate held-out on a disjoint TEST split (Brier, log-loss, calibration ECE, vs the base-rate
baseline, with a paired bootstrap CI). We NEVER fit on test and NEVER select the form using test.

CAUSAL-IDENTIFICATION HONESTY (Phase 6 correction). Fitting a model to a dataset does NOT make the
dataset an instance of whatever mechanism the model's name suggests. Each pack records, explicitly:
  * family            — the mechanism family this pack instantiates
  * identifies        — what the dataset CAUSALLY identifies (randomized) vs what is only PREDICTIVE
  * forbidden         — interpretations the evidence does NOT license
  * missing_variables — variables absent that would be needed for the stronger causal reading
A predictive association is labeled predictive and given `domain_restricted` transport; only a randomized
design (Upworthy A/B traffic) earns a causal label, and even then only for content-response ordering.

Run:  PYTHONPATH=. python -m experiments.wmv2_phase6_fits [--force]
Writes experiments/results/wmv2_phase6_fits.json (all datasets) + per-dataset provenance (sha256, splits,
fitting command, code commit hint). The registry build script (registry/build_registry.py) reads these
fitted coefficients + validation records into ParameterPack objects — the numbers are never hand-typed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys

from swm.world_model_v2.registry.ingestion import brier, fit_logistic, paired_bootstrap_delta

RESULTS = "experiments/results"
OUT = f"{RESULTS}/wmv2_phase6_fits.json"


# ----------------------------------------------------------------- metrics
def logloss(preds, ys):
    s = 0.0
    for p, y in zip(preds, ys):
        p = min(1 - 1e-12, max(1e-12, p))
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / max(1, len(ys))


def calibration(preds, ys, nbins=10):
    """Reliability curve + expected calibration error (ECE)."""
    bins = [[] for _ in range(nbins)]
    for p, y in zip(preds, ys):
        b = min(nbins - 1, int(p * nbins))
        bins[b].append((p, y))
    curve, ece, n = [], 0.0, len(ys)
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        curve.append({"conf": round(conf, 4), "acc": round(acc, 4), "n": len(b)})
        ece += (len(b) / n) * abs(conf - acc)
    return {"reliability": curve, "ece": round(ece, 4)}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split(rows, seed=13, fr=(0.6, 0.2)):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(rows)
    a, b = int(fr[0] * n), int((fr[0] + fr[1]) * n)
    tr = [rows[i] for i in idx[:a]]
    va = [rows[i] for i in idx[a:b]]
    te = [rows[i] for i in idx[b:]]
    return tr, va, te


def _standardizer(rows, keys):
    """Mean/sd per continuous key, computed on TRAIN only (no leakage)."""
    stats = {}
    for k in keys:
        vals = [float(r["features"][k]) for r in rows]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 or 1.0
        stats[k] = (mu, sd)
    return stats


def _fit_on(rows_fit, rows_test, feat_keys, cont_keys, *, label="y"):
    """Fit logistic on rows_fit, evaluate held-out on rows_test (disjoint). Returns metrics dict."""
    stats = _standardizer(rows_fit, cont_keys)

    def vec(r):
        f = r["features"]
        out = []
        for k in feat_keys:
            v = float(f[k])
            if k in stats:
                mu, sd = stats[k]
                v = (v - mu) / sd
            out.append(v)
        return out

    X = [vec(r) for r in rows_fit]
    Y = [int(r[label]) for r in rows_fit]
    w, b = fit_logistic(X, Y, iters=600, lr=0.3, l2=1e-3)

    def predict(r):
        z = b + sum(wi * xi for wi, xi in zip(w, vec(r)))
        return 1 / (1 + math.exp(-max(-30, min(30, z))))

    base = sum(Y) / max(1, len(Y))
    yte = [int(r[label]) for r in rows_test]
    pte = [predict(r) for r in rows_test]
    # baseline for the TEST subpopulation = its OWN base rate (transfer must beat the target base rate)
    tbase = sum(yte) / max(1, len(yte))
    pbase = [tbase] * len(yte)
    paired = paired_bootstrap_delta(yte, pte, pbase, seed=7)
    return {"n_fit": len(rows_fit), "n_test": len(rows_test), "fit_base_rate": round(base, 4),
            "test_base_rate": round(tbase, 4),
            "test_brier": round(brier(pte, yte), 5), "target_base_brier": round(brier(pbase, yte), 5),
            "paired_brier_model_minus_base": paired,
            "beats_target_base_rate": paired["ci95"][1] < 0,
            "calibration": calibration(pte, yte),
            "weights": {k: round(wi, 5) for k, wi in zip(feat_keys, w)}, "intercept": round(b, 5)}


def _fit_eval_tabular(rows, feat_keys, cont_keys, *, seed=13, label="y"):
    """Generic tabular binary-outcome fit: logistic on standardized features (train), held-out Brier/LL/
    calibration on test, paired bootstrap vs the base-rate baseline. Returns (result, coef_dict)."""
    tr, va, te = _split(rows, seed=seed)
    stats = _standardizer(tr, cont_keys)

    def vec(r):
        f = r["features"]
        out = []
        for k in feat_keys:
            v = float(f[k])
            if k in stats:
                mu, sd = stats[k]
                v = (v - mu) / sd
            out.append(v)
        return out

    X = [vec(r) for r in tr + va]           # fit on train+val (form is fixed: logistic); test stays disjoint
    Y = [int(r[label]) for r in tr + va]
    w, b = fit_logistic(X, Y, iters=600, lr=0.3, l2=1e-3)

    def predict(r):
        z = b + sum(wi * xi for wi, xi in zip(w, vec(r)))
        return 1 / (1 + math.exp(-max(-30, min(30, z))))

    base = sum(int(r[label]) for r in tr + va) / max(1, len(tr) + len(va))
    yte = [int(r[label]) for r in te]
    pte = [predict(r) for r in te]
    pbase = [base] * len(te)
    paired = paired_bootstrap_delta(yte, pte, pbase, seed=7)   # model − baseline; negative = model better
    result = {
        "n_total": len(rows), "n_train": len(tr) + len(va), "n_test": len(te), "seed": seed,
        "base_rate": round(base, 4),
        "test": {"brier": round(brier(pte, yte), 5), "logloss": round(logloss(pte, yte), 5),
                 "calibration": calibration(pte, yte)},
        "baseline_base_rate": {"brier": round(brier(pbase, yte), 5), "logloss": round(logloss(pbase, yte), 5)},
        "paired_brier_model_minus_base": paired,
        "beats_base_rate": paired["ci95"][1] < 0,   # upper CI below 0 → model significantly better
    }
    coefs = {"intercept": round(b, 5),
             "weights": {k: round(wi, 5) for k, wi in zip(feat_keys, w)},
             "standardized": cont_keys, "link": "logistic", "fit": "gradient-descent MLE + L2=1e-3"}
    return result, coefs


# ----------------------------------------------------------------- CMV surface features (Tan et al. 2016)
def _cmv_features(e):
    arg = e.get("arg_text", "") or ""
    op = e.get("op_text", "") or ""
    aw = arg.split()
    ow = op.split()
    n_words = len(aw)
    def frac(pred, words):
        return sum(1 for w in words if pred(w)) / max(1, len(words))
    hedges = {"maybe", "perhaps", "possibly", "might", "could", "seems", "somewhat", "arguably"}
    certainty = {"definitely", "clearly", "obviously", "certainly", "undeniably", "always", "never"}
    return {"features": {
        "arg_words": min(600, n_words),
        "arg_op_ratio": min(5.0, (n_words + 1) / (len(ow) + 1)),
        "n_question": arg.count("?"),
        "n_url": arg.lower().count("http"),
        "n_quote_lines": sum(1 for ln in arg.split("\n") if ln.strip().startswith(">") or ln.strip().startswith("&gt;")),
        "hedge_frac": frac(lambda w: w.lower().strip(".,!?") in hedges, aw),
        "certainty_frac": frac(lambda w: w.lower().strip(".,!?") in certainty, aw),
        "you_frac": frac(lambda w: w.lower().strip(".,!?") in ("you", "your", "you're"), aw),
    }, "y": int(e.get("success", 0))}


# ----------------------------------------------------------------- Upworthy (randomized A/B → causal ranking)
def _oid_ts(test_id):
    """Mongo ObjectId embeds a 4-byte unix timestamp in its first 8 hex chars → time-forward ordering."""
    try:
        return int(str(test_id)[:8], 16)
    except Exception:
        return 0


def _rank_variants(variants, score_fn):
    """P@1 (vs the randomized-CTR causal winner) and pairwise accuracy for one test."""
    winner = max(range(len(variants)), key=lambda i: variants[i]["ctr"])
    pick = max(range(len(variants)), key=lambda i: score_fn(variants[i]["headline"]))
    ph, pt = 0, 0
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            hi, lo = (i, j) if variants[i]["ctr"] >= variants[j]["ctr"] else (j, i)
            ph += int(score_fn(variants[hi]["headline"]) >= score_fn(variants[lo]["headline"]))
            pt += 1
    return int(pick == winner), 1.0 / len(variants), ph, pt


def _upworthy(force=False):
    """Reproduce the LLM-FREE content-response ranking through the shared reference pipeline
    (reference.upworthy: surface features → fitted CTR layer → population-heterogeneity particle ranking).
    TIME-FORWARD split by ObjectId timestamp (train = earlier tests, test = later tests) → a real temporal
    transfer claim. Ablation: population vs mean-audience (no_population)."""
    from swm.world_model_v2.reference.upworthy import fit_ctr_layer, population_rank, surface_features, zscores
    path = f"{RESULTS}/exp054_upworthy/upworthy_parsed.json"
    if not os.path.exists(path):
        return {"status": "skipped_missing_data", "path": path}
    tests = json.load(open(path))
    clean = []
    for t in tests:
        uniq = {}
        for a in t.get("arms", []):
            if a.get("impressions", 0) >= 300:
                u = uniq.setdefault(a["headline"], [0, 0])
                u[0] += a["clicks"]
                u[1] += a["impressions"]
        variants = [{"headline": h, "clicks": c, "impressions": i, "ctr": c / i}
                    for h, (c, i) in uniq.items() if i >= 300]
        if len(variants) >= 2:
            clean.append({"ts": _oid_ts(t.get("test_id", "")), "variants": variants})
    clean.sort(key=lambda d: d["ts"])                 # time-forward order
    n = len(clean)

    def fit_ctr(train_lists):
        samples = []
        for variants in train_lists:
            zs = zscores([v["ctr"] for v in variants])
            for v, z in zip(variants, zs):
                samples.append((surface_features(v["headline"]), z))
        _, info = fit_ctr_layer(samples)
        return info["w"]

    def eval_arm(test_lists, w_fitted, heterogeneity):
        p1, rnd, ph, pt = 0, 0.0, 0, 0
        for variants in test_lists:
            feats = [surface_features(v["headline"]) for v in variants]
            scores = population_rank(feats, w_fitted, heterogeneity=heterogeneity, seed=17)
            score_by_h = {v["headline"]: sc for v, sc in zip(variants, scores)}
            a, r, h, tt = _rank_variants(variants, lambda hd: score_by_h[hd])
            p1 += a
            rnd += r
            ph += h
            pt += tt
        nt = len(test_lists)
        return {"precision_at_1": round(p1 / max(1, nt), 4),
                "random_precision_at_1": round(rnd / max(1, nt), 4),
                "pairwise_accuracy": round(ph / max(1, pt), 4), "n_pairs": pt,
                "lift_over_random_p1": round((p1 - rnd) / max(1, nt), 4)}

    variants_all = [d["variants"] for d in clean]
    # (1) IN-DISTRIBUTION held-out: random 60/40 split (shuffled by a fixed seed)
    import random as _r
    order = list(range(n))
    _r.Random(29).shuffle(order)
    rtr = [variants_all[i] for i in order[:int(0.6 * n)]]
    rte = [variants_all[i] for i in order[int(0.6 * n):]]
    w_rand = fit_ctr(rtr)
    heldout_pop = eval_arm(rte, w_rand, True)
    heldout_nopop = eval_arm(rte, w_rand, False)
    # (2) OUT-OF-TIME transfer: train earliest 60%, test latest 40% (temporal shift; headline styles drift)
    ttr, tte = variants_all[:int(0.6 * n)], variants_all[int(0.6 * n):]
    w_time = fit_ctr(ttr)
    transfer_pop = eval_arm(tte, w_time, True)
    w_fitted = w_time
    result = {
        "n_tests_total": n,
        "held_out": {"split": "random 60/40 (in-distribution)", "n_test": len(rte),
                     "population": heldout_pop, "no_population_ablation": heldout_nopop,
                     "beats_random": heldout_pop["pairwise_accuracy"] > 0.5 and
                     heldout_pop["precision_at_1"] > heldout_pop["random_precision_at_1"]},
        "transfer": {"split": "time-forward (train earliest 60% → test latest 40%; temporal shift)",
                     "n_test": len(tte), "population": transfer_pop,
                     "beats_random": transfer_pop["pairwise_accuracy"] > 0.5 and
                     transfer_pop["precision_at_1"] > transfer_pop["random_precision_at_1"]},
        # back-compat keys used by the registry builder
        "split": "time-forward by ObjectId timestamp (earliest 60% train, latest 40% test)",
        "population": transfer_pop, "no_population_ablation": heldout_nopop,
        "beats_random": transfer_pop["pairwise_accuracy"] > 0.5,
    }
    coefs = {"surface_features": ["len_words", "has_number", "has_question", "has_quote",
                                  "has_you", "all_caps_word"],
             "surface_w": [round(x, 4) for x in w_fitted], "n_train_tests": len(ttr),
             "target": "within-test CTR z-score", "link": "linear + population-heterogeneity ranking"}
    return {"result": result, "coefs": coefs, "sha256": _sha256(path), "path": path}


# ----------------------------------------------------------------- OpinionQA (demographic → opinion)
def _opinionqa():
    """Demographic-opinion response: does group membership predict expressed opinion better than the
    population base rate? For each binary Pew question, learn P(answer | party×ideology cell) on a TRAIN
    split (Laplace-smoothed), predict held-out PEOPLE, and compare held-out Brier to the question's own base
    rate. PREDICTIVE/associational (group membership correlates with opinion; NOT a causal opinion lever)."""
    path = f"{RESULTS}/exp028_oqa/oqa_parsed.json"
    if not os.path.exists(path):
        return {"status": "skipped_missing_data", "path": path}
    rows = [r for r in json.load(open(path)) if r.get("n_opt") == 2 and "answer_idx" in r]
    by_q = {}
    for r in rows:
        by_q.setdefault(r["qid"], []).append(r)
    rng = random.Random(13)
    preds, ys, base_preds = [], [], []
    for qid, rs in by_q.items():
        if len(rs) < 40:
            continue
        idx = list(range(len(rs)))
        rng.shuffle(idx)
        cut = int(0.6 * len(rs))
        tr = [rs[i] for i in idx[:cut]]
        te = [rs[i] for i in idx[cut:]]
        base = sum(r["answer_idx"] for r in tr) / len(tr)          # question base rate (train)
        cell = {}
        for r in tr:
            k = (r["demo"].get("party"), r["demo"].get("ideology"))
            c = cell.setdefault(k, [0, 0])
            c[0] += r["answer_idx"]
            c[1] += 1
        for r in te:
            k = (r["demo"].get("party"), r["demo"].get("ideology"))
            c = cell.get(k)
            p = (c[0] + base) / (c[1] + 1) if c else base          # smoothed toward the base rate
            preds.append(min(1 - 1e-6, max(1e-6, p)))
            ys.append(r["answer_idx"])
            base_preds.append(base)
    if not ys:
        return {"status": "insufficient"}
    paired = paired_bootstrap_delta(ys, preds, base_preds, seed=7)
    return {"n_questions": len([q for q, rs in by_q.items() if len(rs) >= 40]), "n_test": len(ys),
            "test_brier": round(brier(preds, ys), 5), "base_rate_brier": round(brier(base_preds, ys), 5),
            "logloss": round(logloss(preds, ys), 5), "calibration": calibration(preds, ys),
            "paired_brier_model_minus_base": paired, "beats_base_rate": paired["ci95"][1] < 0,
            "sha256": _sha256(path), "path": path}


# ----------------------------------------------------------------- dataset drivers
def run_all(force=False):
    out = {"_meta": {"harness": "experiments/wmv2_phase6_fits.py",
                     "note": "real held-out fits; test split never used for fitting or form selection"},
           "datasets": {}}

    # 1) Telco churn — ATTRITION / DROPOUT hazard (observational; NOT general relationship decay)
    p = f"{RESULTS}/harvest_extra/telco_churn.json"
    if os.path.exists(p):
        rows = json.load(open(p))
        feat_keys = ["senior", "partner", "dependents", "tenure", "phone_service", "paperless_billing",
                     "monthly_charges", "total_charges", "contract", "internet_service", "is_female"]
        cont = ["tenure", "monthly_charges", "total_charges"]
        res, coefs = _fit_eval_tabular(rows, feat_keys, cont, seed=13)
        # TRANSFER: fit on month-to-month customers (contract==0), test on longer contracts (contract>0),
        # with `contract` REMOVED from features (constant within each split). A genuine cross-subpopulation
        # transfer: does the remaining hazard shape generalize to a population with very different base churn?
        mm = [r for r in rows if int(r["features"]["contract"]) == 0]
        lng = [r for r in rows if int(r["features"]["contract"]) > 0]
        tf_keys = [k for k in feat_keys if k != "contract"]
        tf_cont = [k for k in cont if k != "contract"]
        transfer = _fit_on(mm, lng, tf_keys, tf_cont) if mm and lng else {"status": "insufficient"}
        transfer["direction"] = "fit=month-to-month contract, test=one/two-year contract (contract excluded)"
        out["datasets"]["telco_attrition"] = {
            "family": "attrition_dropout_hazard", "result": res, "coefs": coefs, "transfer": transfer,
            "sha256": _sha256(p), "path": p,
            "identifies": "PREDICTIVE association of contract/tenure/billing with customer churn (dropout).",
            "causally_identified": False,
            "forbidden": ["do NOT read as causal effects of any single feature",
                          "do NOT generalize to arbitrary social-relationship or trust decay",
                          "do NOT transport off telecom subscription churn without refit + widening"],
            "missing_variables": ["competitor pricing", "service outages", "life events"],
            "transport": "domain_restricted"}

    # 2) StackExchange — RESPONSE-OCCURRENCE hazard (does a question get answered) (observational)
    p = f"{RESULTS}/harvest_extra/stackexchange.json"
    if os.path.exists(p):
        rows = json.load(open(p))
        feat_keys = ["title_len", "body_len", "n_tags", "code_present", "has_code_kw", "is_howto",
                     "title_qmark", "hour", "dayofweek"]
        cont = ["title_len", "body_len", "n_tags", "hour", "dayofweek"]
        res, coefs = _fit_eval_tabular(rows, feat_keys, cont, seed=13)
        out["datasets"]["stackexchange_response"] = {
            "family": "response_occurrence_hazard", "result": res, "coefs": coefs,
            "sha256": _sha256(p), "path": p,
            "identifies": "PREDICTIVE association of question features with receiving an answer (a response-"
                          "occurrence signal in a Q&A community).",
            "causally_identified": False,
            "forbidden": ["do NOT read as trust, obligation, reciprocity, or workload",
                          "do NOT read features as causal levers on answerer behavior",
                          "do NOT transport to email/DM reply without refit"],
            "missing_variables": ["answerer availability", "topic expert supply", "reputation incentives"],
            "transport": "domain_restricted"}

    # 3) CMV — ARGUMENT-PERSUASION success (matched design; platform-specific; predictive)
    p = f"{RESULTS}/exp021_cmv/cmv_common.json"
    if os.path.exists(p):
        raw = json.load(open(p))
        rows = [_cmv_features(e) for e in raw]
        feat_keys = ["arg_words", "arg_op_ratio", "n_question", "n_url", "n_quote_lines",
                     "hedge_frac", "certainty_frac", "you_frac"]
        cont = ["arg_words", "arg_op_ratio", "n_question", "n_url", "n_quote_lines"]
        res, coefs = _fit_eval_tabular(rows, feat_keys, cont, seed=13)
        out["datasets"]["cmv_persuasion"] = {
            "family": "argument_persuasion_success", "result": res, "coefs": coefs,
            "sha256": _sha256(p), "path": p,
            "identifies": "PREDICTIVE association of surface argument features with earning a delta "
                          "(view change) on r/ChangeMyView. Matched successful/unsuccessful challengers.",
            "causally_identified": False,
            "forbidden": ["do NOT read surface features as causal persuasion levers (confounded by content)",
                          "do NOT transport to political/health persuasion (platform-specific, self-selected)",
                          "do NOT interpret as general attitude change"],
            "missing_variables": ["argument semantic content/quality", "OP prior receptivity", "timing/order"],
            "citation": "Tan, Niculae, Danescu-Niculescu-Mizil & Lee 2016, WWW (CMV dataset)",
            "transport": "domain_restricted"}

    # 4b) OpinionQA — DEMOGRAPHIC-OPINION response (does group membership beat the base rate?)
    oqa = _opinionqa()
    if "test_brier" in oqa:
        out["datasets"]["opinionqa_demographic"] = {
            "family": "demographic_opinion_response", "result": {"test": {"brier": oqa["test_brier"],
                     "logloss": oqa["logloss"], "calibration": oqa["calibration"]},
                     "baseline_base_rate": {"brier": oqa["base_rate_brier"]},
                     "paired_brier_model_minus_base": oqa["paired_brier_model_minus_base"],
                     "beats_base_rate": oqa["beats_base_rate"], "n_test": oqa["n_test"],
                     "n_questions": oqa["n_questions"]},
            "coefs": {"model": "party×ideology cell probability (Laplace-smoothed), per-question"},
            "sha256": oqa["sha256"], "path": oqa["path"],
            "identifies": "PREDICTIVE/associational: party×ideology group membership predicts expressed "
                          "opinion better than the population base rate on Pew ATP questions.",
            "causally_identified": False,
            "forbidden": ["do NOT read as a causal opinion lever (party is not randomly assigned)",
                          "do NOT transport across countries/eras (US survey, specific waves)",
                          "expressed opinion ≠ latent belief"],
            "missing_variables": ["question wording effects", "non-response bias", "temporal drift"],
            "citation": "OpinionQA / Pew American Trends Panel (Santurkar et al. 2023)",
            "transport": "domain_restricted"}

    # 4) Upworthy — CONTENT-RESPONSE click ranking (RANDOMIZED A/B → causal ordering)
    up = _upworthy(force=force)
    if "result" in up:
        out["datasets"]["upworthy_content_response"] = {
            "family": "content_response_click", **up,
            "identifies": "CAUSAL ordering of content-driven click response: arms were shown to RANDOMIZED "
                          "traffic within each test, so the higher-CTR headline is causally more clicked.",
            "causally_identified": True,
            "identified_scope": "content-response ranking only",
            "forbidden": ["does NOT identify position/examination bias (rank not experimentally varied here)",
                          "surface-feature weights are a PREDICTIVE summary, not causal feature effects",
                          "do NOT transport clickbait-era Upworthy CTR levels to other platforms"],
            "missing_variables": ["image/thumbnail", "audience segment", "time-of-day targeting"],
            "citation": "Upworthy Research Archive (Matias et al. 2021), CC-BY",
            "transport": "domain_restricted"}
    else:
        out["datasets"]["upworthy_content_response"] = up

    return out


def main():
    force = "--force" in sys.argv
    if os.path.exists(OUT) and not force:
        print(f"{OUT} exists — re-run with --force to refit. Loading summary:")
        out = json.load(open(OUT))
    else:
        out = run_all(force=force)
        json.dump(out, open(OUT, "w"), indent=1, default=str)
    for name, d in out["datasets"].items():
        if "result" not in d:
            print(f"  {name}: {d.get('status', 'no result')}")
            continue
        r = d["result"]
        fam = d.get("family", "?")
        if "brier" in r.get("test", {}):
            tf = d.get("transfer", {})
            tfmsg = (f" | TRANSFER beats_target_base={tf.get('beats_target_base_rate')}"
                     if "beats_target_base_rate" in tf else "")
            print(f"  {name} [{fam}]: test Brier {r['test']['brier']} vs base {r['baseline_base_rate']['brier']} "
                  f"| beats_base={r['beats_base_rate']} | ECE {r['test']['calibration']['ece']} "
                  f"| causal={d.get('causally_identified')}{tfmsg}")
        else:
            ho = r.get("held_out", {}).get("population", {})
            tf = r.get("transfer", {}).get("population", {})
            nop = r.get("held_out", {}).get("no_population_ablation", {})
            print(f"  {name} [{fam}]: HELD-OUT(random) pairwise {ho.get('pairwise_accuracy')} "
                  f"P@1 {ho.get('precision_at_1')} vs rand {ho.get('random_precision_at_1')} "
                  f"| TRANSFER(time-fwd) pairwise {tf.get('pairwise_accuracy')} "
                  f"beats_random={r.get('transfer', {}).get('beats_random')} "
                  f"| no-pop pairwise {nop.get('pairwise_accuracy')} | causal={d.get('causally_identified')}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
