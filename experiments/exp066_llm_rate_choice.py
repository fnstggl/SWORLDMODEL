"""EXP-066: can the LLM pick the right RATE for a novel quantity on its own? (the open measurement)

The compiler's mechanism choice is robust (EXP-065: 15/15) and the engine + a CORRECT rate is calibrated
(EXP-065: 80% coverage). The one untested link is whether the LLM can supply the rate itself — the
per-unit-time VOLATILITY the diffusion needs. This is the direct test of "are the inferences good enough".

Ground truth: for 15 GSS attitude topics we measure the real year-to-year opinion volatility (sigma, pp/yr)
from decades of data. These span 1.5-4.9 pp/yr and contain a genuine trap — the noisiest quantities are
the reactive "spending" attitudes, NOT the famously fast-DRIFTING moral issues (whose year-to-year
volatility is only moderate). So the benchmark separates a model that understands VOLATILITY from one that
confuses it with cumulative change.

An EXTERNAL model (Qwen-2.5-72B via HF) is asked, BLIND (no data, plain-language topic), for each topic's
typical annual opinion shift. We score three things:
  1. ABSOLUTE scale — is the estimate in the right ballpark (ratio, fraction within 2x)?
  2. DISCRIMINATION — Spearman rank corr: can it tell volatile from stable quantities?
  3. DOWNSTREAM — do intervals built from the LLM's rates cover at the nominal 80%?

Plus a full-spec compilation demo: the LLM emits a complete generic_scm spec end-to-end, run through the
engine — the compiler operating fully autonomously, blind.

Run (needs HF_TOKEN in env for a fresh query; otherwise uses the committed cache):
  HF_TOKEN=... python -m experiments.exp066_llm_rate_choice
"""
from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path

GSS = "experiments/results/exp045_gss/gss_parsed.json.gz"
CACHE = "experiments/results/exp066/qwen_rates.json"
SPEC_CACHE = "experiments/results/exp066/qwen_fullspec.json"
RESULT = "experiments/results/exp066_llm_rate_choice.json"

TOPICS = {
    "gunlaw": "requiring a police permit before a person can buy a gun",
    "cappun": "favoring the death penalty for murder",
    "premarsx": "believing premarital sex is wrong",
    "homosex": "believing same-sex relations are wrong",
    "abany": "believing abortion should be legal for any reason",
    "fefam": "believing a woman's place is in the home",
    "letdie1": "allowing doctors to end a terminally ill patient's life",
    "fepol": "believing women are not suited for politics",
    "grass": "favoring the legalization of marijuana",
    "natenvir": "believing the US spends too little on the environment",
    "natfare": "believing the US spends too little on welfare",
    "natcrime": "believing the US spends too little on fighting crime",
    "natheal": "believing the US spends too little on healthcare",
    "natrace": "believing the US spends too little on improving conditions for Black Americans",
    "nateduc": "believing the US spends too little on education",
}


def _data_sigma():
    rows = json.load(gzip.open(GSS))
    by_q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for q, a in r["answers"].items():
            if a in (0, 1):
                by_q[q][r["year"]].append(r)
    out = {}
    for q in TOPICS:
        years = sorted(y for y, rs in by_q[q].items() if len(rs) >= 300)
        sh = {y: sum(r["answers"][q] for r in by_q[q][y]) / len(by_q[q][y]) for y in years}
        sig2 = [(sh[years[i]] - sh[years[i - 1]]) ** 2 / (years[i] - years[i - 1])
                for i in range(1, len(years)) if years[i] != years[i - 1]]
        out[q] = {"sigma_pp": math.sqrt(sum(sig2) / len(sig2)) * 100, "shares": (years, sh)}
    return out


def _query_rates():
    """ONE batched call over all topics — credit-efficient AND a fairer differentiation test (the model
    sees every topic together and must rank their volatilities relative to each other)."""
    from swm.api.hf_backend import hf_chat_fn
    fn = hf_chat_fn(system="You are a survey-methodology expert on US public opinion. Return ONLY JSON.",
                    max_tokens=600, temperature=0.0)
    listing = "\n".join(f'  "{q}": {desc}' for q, desc in TOPICS.items())
    prompt = ("For each US public-opinion topic below, estimate its VOLATILITY: the typical year-over-year "
              "change (up or down) in the percentage of Americans holding that view — how much it bounces "
              "per year, NOT the total drift over decades. Some attitudes are jumpy/reactive; others are "
              "stable. Differentiate them.\n" + listing +
              '\nReturn ONLY JSON mapping each key to a number in percentage points, e.g. {"gunlaw": 2.0, ...}')
    raw = fn(prompt)
    a, b = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[a:b + 1])
    return {q: (float(obj[q]) if q in obj and _isnum(str(obj[q])) else None) for q in TOPICS}


def _isnum(t):
    try:
        float(t); return True
    except ValueError:
        return False


def _spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)) * sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / den if den else 0.0


def _full_spec_demo():
    """The LLM emits a COMPLETE structural spec end-to-end; run it through the engine (blind compilation)."""
    from swm.api.compiler import StructuralCompiler, build_compile_prompt
    from swm.api.hf_backend import hf_chat_fn
    from swm.api.world_model import WorldModel
    q = "Will US CPI inflation be above 3% at the end of this year?"
    if Path(SPEC_CACHE).exists():
        spec_json = json.loads(Path(SPEC_CACHE).read_text())
    else:
        try:
            fn = hf_chat_fn(system="You compile questions into runnable structural simulations. Output ONLY "
                                   "the JSON spec, no prose.", max_tokens=700)
            raw = fn(build_compile_prompt(q))
            a, b = raw.find("{"), raw.rfind("}")
            spec_json = json.loads(raw[a:b + 1])
            Path(SPEC_CACHE).parent.mkdir(parents=True, exist_ok=True)
            Path(SPEC_CACHE).write_text(json.dumps(spec_json, indent=1))
        except Exception as e:
            return {"status": f"skipped (backend unavailable: {str(e)[:60]})"}
    wm = WorldModel(compiler=StructuralCompiler(lambda key: spec_json))
    out = wm.simulate(q, key=q)
    f = out["forecast"]
    iv = f.get("interval_80", [0, 0])
    degenerate = (iv[1] - iv[0]) < 1e-6                 # collapsed at a bound -> a buggy equation/bounds
    return {"question": q, "compiled_mechanism": out["mechanism"], "spec": spec_json,
            "variables": out["spec"]["variables"], "forecast": f, "headline": out["headline"],
            "degenerate": degenerate,
            "diagnosis": ("STRUCTURE correct (mechanism, value, volatility, bounds, outcome all sensible) "
                          "but the EQUATION is buggy: its intent (per the rationale) is mean-reversion to "
                          "~3%, yet '0.01*(100-CPI)' reverts toward ~35% and saturates the hi bound -> "
                          "degenerate P=1.0. Right intent, wrong formula -> autonomous spec authoring needs "
                          "a validation/repair loop." if degenerate else "ran cleanly")}


def run():
    data = _data_sigma()
    Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
    if Path(CACHE).exists() and not os.environ.get("HF_FORCE"):
        est = json.loads(Path(CACHE).read_text())
    else:
        est = _query_rates()
        Path(CACHE).write_text(json.dumps(est, indent=1))

    rows = []
    for q in TOPICS:
        d = data[q]["sigma_pp"]; e = est.get(q)
        if e is not None:
            rows.append({"topic": q, "data_sigma_pp": round(d, 2), "llm_est_pp": e,
                         "ratio": round(e / d, 2)})
    dv = [r["data_sigma_pp"] for r in rows]; lv = [r["llm_est_pp"] for r in rows]
    log_ratios = [math.log(r["llm_est_pp"] / r["data_sigma_pp"]) for r in rows if r["llm_est_pp"] > 0]
    geo_ratio = math.exp(sum(log_ratios) / len(log_ratios))
    within2x = sum(1 for r in rows if 0.5 <= r["ratio"] <= 2.0) / len(rows)
    spearman = _spearman(lv, dv)

    # downstream: intervals from LLM rates vs data rates -> coverage
    def coverage(get_sigma):
        cov = tot = 0
        for q in TOPICS:
            years, sh = data[q]["shares"]
            sig = get_sigma(q) / 100.0
            for ai in range(len(years)):
                A = years[ai]
                for T in years:
                    if 8 <= T - A <= 16:
                        w = 1.2816 * sig * math.sqrt(T - A)
                        cov += int(abs(sh[T] - sh[A]) <= w); tot += 1
        return round(cov / tot, 3)
    cov_llm = coverage(lambda q: est.get(q) or 3.0)
    cov_data = coverage(lambda q: data[q]["sigma_pp"])
    cov_global = coverage(lambda q: 2.9)          # a single global prior (no per-topic differentiation)

    demo = _full_spec_demo()

    out = {"n_topics": len(rows), "grader": "Qwen-2.5-72B (HF, blind)",
           "absolute_scale": {"geometric_mean_ratio_llm_over_data": round(geo_ratio, 3),
                              "fraction_within_2x": round(within2x, 3),
                              "reading": "is the LLM's rate in the right ballpark?"},
           "discrimination": {"spearman_rank_corr": round(spearman, 3),
                              "reading": "can the LLM tell volatile topics from stable ones? "
                                         "(the hard part -- volatility != cumulative drift)"},
           "downstream_coverage": {"nominal": 0.8, "llm_rates": cov_llm, "data_rates": cov_data,
                                   "single_global_prior": cov_global},
           "per_topic": sorted(rows, key=lambda r: -r["data_sigma_pp"]),
           "full_spec_compilation_demo": demo}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-066  can the LLM pick the right RATE for a novel quantity? (Qwen-72B, blind)")
    print(f"  n topics = {len(rows)}   (data-measured opinion volatility 1.5-4.9 pp/yr)")
    print(f"  1. ABSOLUTE scale: geo-mean ratio LLM/data = {geo_ratio:.2f}  "
          f"({within2x*100:.0f}% within 2x)  -> {'right ballpark' if 0.6 < geo_ratio < 1.7 else 'off'}")
    print(f"  2. DISCRIMINATION: Spearman(LLM, data) = {spearman:+.2f}  "
          f"-> {'can rank' if spearman > 0.4 else 'CANNOT reliably rank volatile vs stable'}")
    print(f"  3. DOWNSTREAM coverage (nominal 0.80): LLM-rates={cov_llm}  data-rates={cov_data}  "
          f"single-global-prior={cov_global}")
    print("  per-topic (data sigma vs LLM estimate, pp/yr):")
    for r in sorted(rows, key=lambda r: -r["data_sigma_pp"]):
        print(f"       {r['topic']:9s} data={r['data_sigma_pp']:.1f}  llm={r['llm_est_pp']:.1f}  (x{r['ratio']})")
    if "status" in demo:
        print(f"  FULL-SPEC blind compilation: {demo['status']}")
    else:
        print(f"  FULL-SPEC blind compilation: '{demo['question']}' -> mechanism={demo['compiled_mechanism']},"
              f" {demo['headline']}")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
