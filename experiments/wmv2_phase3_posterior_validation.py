"""Phase 3 posterior — synthetic hidden-state RECOVERY + CALIBRATION + ABLATIONS + ACCEPTANCE GATES.

This is the offline, deterministic empirical backbone (Parts O/P/Q/R). It exercises the SAME production code
(`infer_posterior` + the registered observation models) on scenarios whose hidden state is KNOWN, so we can
measure whether the posterior is REAL:

  RECOVERY      does the posterior mean track the true hidden rate θ? (corr, RMSE vs a prior-only baseline)
  CALIBRATION   are posterior-predictive probabilities calibrated against realized outcomes? (ECE)
  COVERAGE      does the 80% credible interval contain θ ~80% of the time? (honest uncertainty)
  STRUCTURE     does the structural posterior concentrate on the TRUE competing structure?
  ABLATIONS     posterior vs prior-only; full-posterior vs point-estimate; dependence-corrected vs
                independent (on syndicated data); likelihood-structural vs the Phase-2 heuristic.
  GATES         explicit numerical + empirical acceptance gates with pass/fail — no result hidden.

Well-specified by construction (the generator uses the model's own likelihood), so recovery failure would
indicate a real defect, not a modeling artifact; misspecification robustness is probed separately by injecting
neutral noise claims and unreliable sources. Deterministic under hash-stable seeds.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

from swm.world_model_v2.phase3_observation import _STRENGTH_SENS_SPEC
from swm.world_model_v2.phase3_latent_spec import ClaimTag
from swm.world_model_v2.phase3_posterior import infer_posterior

OUT = Path("experiments/results/phase3")
STRUCTURES = [("H_no", 0.18, "strong_no"), ("H_mid", 0.5, "neutral"), ("H_yes", 0.82, "strong_yes")]


def _seed(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _stub_plan(lean="neutral", hyps=None):
    return SimpleNamespace(provenance={"outcome_lean": lean},
                           structural_hypotheses=hyps or [], question="synthetic recovery scenario")


def _gen_vote_direction(rng, theta, strength, reliability):
    """Generate a claim direction from θ using the SAME likelihood infer_posterior inverts (well-specified)."""
    sens, spec = _STRENGTH_SENS_SPEC.get(strength, (0.72, 0.72))
    sens = 0.5 + reliability * (sens - 0.5)
    spec = 0.5 + reliability * (spec - 0.5)
    p_yes_vote = theta * sens + (1 - theta) * (1 - spec)
    return "supports_yes" if rng.random() < p_yes_vote else "supports_no"


def _gen_scenario(rng, *, n_claims=6, syndicate=False, noise_frac=0.0, structural=False):
    """One scenario with known ground truth. Returns (theta, true_struct, tags, outcome)."""
    if structural:
        true_struct, theta_c, _lean = STRUCTURES[rng.randrange(len(STRUCTURES))]
        theta = min(0.95, max(0.05, theta_c + rng.gauss(0, 0.06)))
        hyps = [{"id": s, "prior": 1.0 / len(STRUCTURES), "lean": ln} for s, _, ln in STRUCTURES]
    else:
        true_struct, theta, hyps = "", 0.1 + 0.8 * rng.random(), []
    tags, strengths, syn_seed = [], ("weak", "moderate", "strong"), None
    for i in range(n_claims):
        rel = 0.55 + 0.4 * rng.random()
        strength = strengths[rng.randrange(3)]
        if syndicate and i >= 1 and syn_seed is not None:
            # TRUE syndication: copies i>=1 re-publish the SAME report — identical direction + strength as the
            # first syndicated member (this is what dependence correction is designed to de-duplicate).
            direction, strength = syn_seed
            dg = "wire-syn"
        else:
            if noise_frac and rng.random() < noise_frac:
                direction, strength = "neutral", "weak"                # injected irrelevant claim
            else:
                direction = _gen_vote_direction(rng, theta, strength, rel)
            dg = "wire-syn" if syndicate else f"src-{i}"
            if syndicate and syn_seed is None:
                syn_seed = (direction, strength)                       # the original report the rest copy
        sup = [true_struct] if (structural and direction != "neutral" and rng.random() < 0.7) else []
        tags.append(ClaimTag(claim_id=f"c{i}", outcome_direction=direction, strength=strength,
                             reliability=rel, dependence_group=dg, supports_hypotheses=sup))
    outcome = 1 if rng.random() < theta else 0
    return theta, true_struct, tags, hyps, outcome


# ---------------------------------------------------------------- metrics
def _ece(preds, bins=10):
    if not preds:
        return 0.0
    buck = [[] for _ in range(bins)]
    for p, o in preds:
        buck[min(bins - 1, int(p * bins))].append((p, o))
    n, e = len(preds), 0.0
    for b in buck:
        if b:
            e += len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(o for _, o in b) / len(b))
    return e


def _corr(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def recovery_and_calibration(n=800, seed=7, n_claims=6, noise_frac=0.15):
    thetas, post_means, prior_means, preds_post, preds_prior, covered = [], [], [], [], [], 0
    for i in range(n):
        rng = random.Random(_seed(seed, "recov", i))
        theta, _, tags, _, outcome = _gen_scenario(rng, n_claims=n_claims, noise_frac=noise_frac)
        post = infer_posterior(_stub_plan("neutral"), None, tags, seed=_seed(seed, "inf", i) % 100000)
        thetas.append(theta)
        post_means.append(post.outcome_rate_mean)
        prior_means.append(post.outcome_rate_prior_mean)
        preds_post.append((post.outcome_rate_mean, outcome))
        preds_prior.append((post.outcome_rate_prior_mean, outcome))
        # 80% credible interval from the particle set
        lo, hi = _cred_interval(post.outcome_rate_particles, 0.1, 0.9)
        covered += 1 if lo <= theta <= hi else 0
    rmse_post = math.sqrt(sum((p - t) ** 2 for p, t in zip(post_means, thetas)) / n)
    rmse_prior = math.sqrt(sum((p - t) ** 2 for p, t in zip(prior_means, thetas)) / n)
    return {"n": n, "recovery_corr": round(_corr(post_means, thetas), 4),
            "rmse_posterior": round(rmse_post, 4), "rmse_prior_only": round(rmse_prior, 4),
            "rmse_improvement": round(rmse_prior - rmse_post, 4),
            "ece_posterior": round(_ece(preds_post), 4), "ece_prior_only": round(_ece(preds_prior), 4),
            "ci80_coverage": round(covered / n, 4), "brier_posterior": round(
                sum((p - o) ** 2 for p, o in preds_post) / n, 4),
            "brier_prior_only": round(sum((p - o) ** 2 for p, o in preds_prior) / n, 4)}


def recovery_curve(seed=7):
    """Recovery + calibration as a function of EVIDENCE ABUNDANCE. Recovery rises and credible intervals
    tighten toward the nominal 0.80 as evidence grows — the honest picture a single weak-evidence point hides."""
    curve = {}
    for m in (3, 6, 12, 24):
        r = recovery_and_calibration(n=600, seed=_seed(seed, "curve", m) % 100000, n_claims=m, noise_frac=0.1)
        curve[f"n_claims={m}"] = {"recovery_corr": r["recovery_corr"], "rmse_improvement": r["rmse_improvement"],
                                  "ece_posterior": r["ece_posterior"], "ci80_coverage": r["ci80_coverage"]}
    return curve


def _cred_interval(particles, ql, qh):
    pts = sorted(particles, key=lambda pw: pw[0])
    acc, lo, hi = 0.0, pts[0][0], pts[-1][0]
    for r, w in pts:
        acc += w
        if acc >= ql:
            lo = r
            break
    acc = 0.0
    for r, w in pts:
        acc += w
        if acc >= qh:
            hi = r
            break
    return lo, hi


def structural_recovery(n=600, seed=11):
    hit_post, hit_prior = 0, 0
    for i in range(n):
        rng = random.Random(_seed(seed, "struct", i))
        _, true_struct, tags, hyps, _ = _gen_scenario(rng, n_claims=7, structural=True)
        post = infer_posterior(_stub_plan("neutral", hyps), None, tags, seed=_seed(seed, "si", i) % 100000)
        if post.structural_posterior:
            top = max(post.structural_posterior, key=post.structural_posterior.get)
            hit_post += 1 if top == true_struct else 0
        top_prior = max(post.structural_prior, key=post.structural_prior.get) if post.structural_prior else ""
        hit_prior += 1 if top_prior == true_struct else 0
    return {"n": n, "true_structure_recovered": round(hit_post / n, 4),
            "prior_baseline": round(hit_prior / n, 4), "chance": round(1.0 / len(STRUCTURES), 4)}


def ablations(n=800, seed=13):
    """Each arm scored on the SAME scenarios by predictive log-loss + calibration + RMSE to θ."""
    arms = {"full_posterior": [], "prior_only": [], "point_estimate": [], "independent_on_syndicated": [],
            "dependence_corrected_on_syndicated": []}
    rmse = {k: [] for k in arms}
    for i in range(n):
        rng = random.Random(_seed(seed, "abl", i))
        theta, _, tags, _, outcome = _gen_scenario(rng, n_claims=6, noise_frac=0.1)
        post = infer_posterior(_stub_plan("neutral"), None, tags, seed=_seed(seed, "ai", i) % 100000)
        _score(arms, rmse, "full_posterior", post.outcome_rate_mean, outcome, theta)
        _score(arms, rmse, "prior_only", post.outcome_rate_prior_mean, outcome, theta)
        _score(arms, rmse, "point_estimate", _point_estimate(tags), outcome, theta)
        # dependence ablation: a scenario deliberately flooded with syndicated copies of ONE report
        rng2 = random.Random(_seed(seed, "syn", i))
        theta_s, _, tags_s, _, out_s = _gen_scenario(rng2, n_claims=8, syndicate=True)
        indep = infer_posterior(_stub_plan("neutral"), None, tags_s, seed=_seed(seed, "ii", i) % 100000,
                                use_dependence=False)
        deps = infer_posterior(_stub_plan("neutral"), None, tags_s, seed=_seed(seed, "di", i) % 100000,
                               use_dependence=True)
        _score(arms, rmse, "independent_on_syndicated", indep.outcome_rate_mean, out_s, theta_s)
        _score(arms, rmse, "dependence_corrected_on_syndicated", deps.outcome_rate_mean, out_s, theta_s)
    out = {}
    for k, preds in arms.items():
        nn = len(preds)
        ll = sum(-(o * math.log(max(1e-6, p)) + (1 - o) * math.log(max(1e-6, 1 - p))) for p, o in preds) / nn
        out[k] = {"logloss": round(ll, 5), "ece": round(_ece(preds), 5),
                  "rmse_to_theta": round(math.sqrt(sum(rmse[k]) / len(rmse[k])), 5)}
    return out


def _score(arms, rmse, key, p, outcome, theta):
    p = max(1e-6, min(1 - 1e-6, p))
    arms[key].append((p, outcome))
    rmse[key].append((p - theta) ** 2)


def _point_estimate(tags):
    """The scalar anti-pattern arm: reliability-weighted directional vote share, no uncertainty."""
    num = sum((1.0 if t.outcome_direction == "supports_yes" else 0.0) * t.reliability
              for t in tags if t.outcome_direction != "neutral")
    den = sum(t.reliability for t in tags if t.outcome_direction != "neutral") or 1.0
    return max(0.02, min(0.98, num / den))


def acceptance_gates(recov, curve, struct, abl):
    hi = curve["n_claims=24"]                                          # recovery is a gate at ADEQUATE evidence
    gates = {
        "recovery_rises_with_evidence":
            curve["n_claims=24"]["recovery_corr"] > curve["n_claims=3"]["recovery_corr"] + 0.15,
        "recovery_corr>=0.55_at_adequate_evidence": hi["recovery_corr"] >= 0.55,
        "posterior_beats_prior_rmse_at_all_evidence_levels":
            all(v["rmse_improvement"] > 0 for v in curve.values()),
        "posterior_calibrated_ece<=0.08": recov["ece_posterior"] <= 0.08,
        "ci80_coverage_conservative_[0.78,0.95]": 0.78 <= recov["ci80_coverage"] <= 0.95,
        "ci80_coverage_tightens_toward_nominal_with_evidence":
            hi["ci80_coverage"] <= curve["n_claims=3"]["ci80_coverage"] + 1e-9,
        "structure_recovered_above_prior": struct["true_structure_recovered"] > struct["prior_baseline"] + 0.15,
        "full_posterior_beats_point_estimate_logloss":
            abl["full_posterior"]["logloss"] < abl["point_estimate"]["logloss"],
        "full_posterior_beats_prior_only_logloss":
            abl["full_posterior"]["logloss"] < abl["prior_only"]["logloss"],
        "dependence_correction_improves_calibration_on_syndicated":
            abl["dependence_corrected_on_syndicated"]["ece"] <= abl["independent_on_syndicated"]["ece"],
        "dependence_correction_reduces_overconfidence_on_syndicated":
            abl["dependence_corrected_on_syndicated"]["logloss"] <= abl["independent_on_syndicated"]["logloss"] + 1e-9,
    }
    return {"gates": {k: bool(v) for k, v in gates.items()},
            "n_pass": sum(1 for v in gates.values() if v), "n_total": len(gates),
            "all_pass": all(gates.values())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recov = recovery_and_calibration(n_claims=24, noise_frac=0.1)      # headline recovery at adequate evidence
    curve = recovery_curve()
    struct = structural_recovery()
    abl = ablations()
    gates = acceptance_gates(recov, curve, struct, abl)
    report = {"recovery_at_adequate_evidence": recov, "recovery_curve": curve, "structural_recovery": struct,
              "ablations": abl, "acceptance_gates": gates,
              "note": "well-specified generator (model's own likelihood) + injected neutral/unreliable noise; "
                      "syndicated arm uses TRUE identical copies; hash-stable seeds → reproducible"}
    (OUT / "posterior_validation.json").write_text(json.dumps(report, indent=2))
    print("RECOVERY @ adequate evidence (24 claims):", json.dumps(recov, indent=2))
    print("\nRECOVERY CURVE (vs evidence abundance):")
    for k, v in curve.items():
        print(f"  {k:14s} corr={v['recovery_corr']:.3f}  rmse_impr={v['rmse_improvement']:+.4f}  "
              f"ece={v['ece_posterior']:.4f}  ci80={v['ci80_coverage']:.3f}")
    print("\nSTRUCTURAL RECOVERY:", json.dumps(struct, indent=2))
    print("\nABLATIONS (logloss / ece / rmse_to_theta):")
    for k, v in abl.items():
        print(f"  {k:38s} {v}")
    print("\nACCEPTANCE GATES:", json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
