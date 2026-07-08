"""EXP-068: a SCORED end-to-end run of the full autonomous pipeline on forecastable questions.

The whole system, driven by an LLM, scored against real outcomes. For each question the pipeline is fully
autonomous: an LLM (Qwen-72B via HF; swap in `anthropic_compile_fn` with ANTHROPIC_API_KEY for a frontier
paid backend) reads the question + as-of context and COMPILES a structural model — no human picks the
mechanism, variables, equation, or rate. The `WorldModel` front door then VALIDATES and (if flagged)
REPAIRS the spec (EXP-067), runs the Monte-Carlo, and returns a forecast. We score it against the resolved
outcome.

Task: forecastable opinion trajectories from the GSS. For 15 attitude topics we take a real as-of year A
(share known) and ask the pipeline to forecast the share ~12 years later; we score against the realized
share at T. This is genuinely FORECASTABLE (decade-horizon opinion is structure/persistence-dominated,
inside the predictability horizon) and the LLM writes the ENTIRE spec — including the volatility, the piece
EXP-066 isolated. Leakage control: the LLM is given only the AS-OF value, never the future; and we compare
to the persistence baseline (share stays at A), which any real forecast must beat.

Robust to the free-tier credit limit: each compiled spec is cached incrementally, so the run resumes and
scores whatever completed. Reproducible offline from the committed cache.

Run (live): HF_TOKEN=... python -m experiments.exp068_scored_end_to_end
"""
from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from swm.api.compiler import StructuralCompiler, build_compile_prompt
from swm.api.model_spec import parse_spec
from swm.api.world_model import WorldModel

GSS = "experiments/results/exp045_gss/gss_parsed.json.gz"
SPECS = "experiments/results/exp068/specs.json"
RESULT = "experiments/results/exp068_scored_end_to_end.json"

TOPICS = {
    "gunlaw": "US adults who favor requiring a police permit before buying a gun",
    "cappun": "US adults who favor the death penalty for murder",
    "premarsx": "US adults who believe premarital sex is not wrong at all",
    "homosex": "US adults who believe same-sex relations are not wrong at all",
    "abany": "US adults who believe abortion should be legal for any reason",
    "fefam": "US adults who agree a woman's place is in the home",
    "letdie1": "US adults who favor allowing doctors to end a terminally ill patient's life",
    "fepol": "US adults who agree women are not suited for politics",
    "grass": "US adults who favor legalizing marijuana",
    "natenvir": "US adults who say the US spends too little on the environment",
    "natfare": "US adults who say the US spends too little on welfare",
    "natcrime": "US adults who say the US spends too little on fighting crime",
    "natheal": "US adults who say the US spends too little on healthcare",
    "natrace": "US adults who say the US spends too little on improving the conditions of Black Americans",
    "nateduc": "US adults who say the US spends too little on education",
}


def _pairs():
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
        pair = None
        for A in years:
            cand = [T for T in years if 10 <= T - A <= 14]
            if cand:
                pair = (A, cand[len(cand) // 2]); break
        if pair:
            A, T = pair
            out[q] = {"A": A, "T": T, "share_A": sh[A], "share_T": sh[T], "H": T - A}
    return out


def _norm(x):
    return x / 100.0 if x is not None and x > 1.5 else x


def run():
    pairs = _pairs()
    Path(SPECS).parent.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(SPECS).read_text()) if Path(SPECS).exists() else {}

    live = None
    if not all(q in specs for q in pairs) and os.environ.get("HF_TOKEN"):
        from swm.api.hf_backend import hf_chat_fn
        live = hf_chat_fn(system="You compile questions into runnable structural-simulation JSON specs. "
                                 "Output ONLY the JSON spec.", max_tokens=700)
    repair = None
    if os.environ.get("HF_TOKEN"):
        from swm.api.hf_backend import hf_chat_fn
        repair = hf_chat_fn(system="You fix bugs in structural-model JSON specs. Output ONLY corrected JSON.",
                            max_tokens=700)

    # 1) COMPILE each spec (live LLM, cached incrementally)
    for q, p in pairs.items():
        if q in specs:
            continue
        if live is None:
            break
        question = f"What percentage of {TOPICS[q]} will there be in the year {p['T']}?"
        context = f"As of {p['A']}, {p['share_A']*100:.0f}% held this view."
        try:
            raw = live(build_compile_prompt(question, context))
            spec_json = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            specs[q] = spec_json
            Path(SPECS).write_text(json.dumps(specs, indent=1))
        except Exception as e:
            print(f"  compile stopped at {q}: {str(e)[:60]}")
            break

    # 2) RUN each through the validated WorldModel front door and SCORE
    def repair_fn(prompt):
        obj = json.loads(repair(prompt)) if repair else None
        return obj if obj else {}
    scored, flagged = [], []
    for q, p in pairs.items():
        if q not in specs:
            continue
        spec_json = specs[q]
        wm = WorldModel(compiler=StructuralCompiler(lambda key, s=spec_json: s),
                        repair_fn=(repair_fn if repair else None))
        try:
            out = wm.simulate(f"forecast {q}", key=q, n=4000)
            f = out["forecast"]
            # point estimate + interval are mechanism-specific keys; normalize
            pt = f.get("mean", f.get("mean_share", f.get("mean_vote_share")))
            pred = _norm(pt)
            if pred is None or not out.get("validation", {}).get("clean", True):
                flagged.append({"topic": q, "clean": out.get("validation", {}).get("clean"),
                                "issues": [i["code"] for i in out.get("validation", {}).get("issues", [])]})
                continue
            lo, hi = [_norm(x) for x in f.get("interval_80", [pred, pred])]
            scored.append({"topic": q, "A": p["A"], "T": p["T"], "share_A": round(p["share_A"], 3),
                           "truth": round(p["share_T"], 3), "pred": round(pred, 3),
                           "persistence": round(p["share_A"], 3),
                           "covered": bool(lo <= p["share_T"] <= hi),
                           "mechanism": f["mechanism"], "repairs": out.get("validation", {}).get("repairs")})
        except Exception as e:
            flagged.append({"topic": q, "error": str(e)[:80]})

    n = len(scored)
    if n:
        mae = sum(abs(s["pred"] - s["truth"]) for s in scored) / n
        mae_persist = sum(abs(s["persistence"] - s["truth"]) for s in scored) / n
        cov = sum(s["covered"] for s in scored) / n
        clean = sum(1 for s in scored if not s["repairs"])
        repaired = sum(1 for s in scored if s["repairs"])
        skill = 1 - mae / mae_persist if mae_persist else 0.0
        # fair breakdown: the specs the LLM compiled CLEANLY (no repair needed) vs the repaired ones
        cs = [s for s in scored if not s["repairs"]]
        mae_clean = sum(abs(s["pred"] - s["truth"]) for s in cs) / len(cs) if cs else None
        mae_clean_persist = sum(abs(s["persistence"] - s["truth"]) for s in cs) / len(cs) if cs else None
    else:
        mae = mae_persist = cov = clean = repaired = skill = mae_clean = mae_clean_persist = None

    out = {"n_scored": n, "n_topics": len(pairs), "n_specs_compiled": len(specs),
           "MAE_pipeline": round(mae, 4) if mae else None,
           "MAE_persistence_baseline": round(mae_persist, 4) if mae_persist else None,
           "skill_vs_persistence": round(skill, 4) if skill is not None else None,
           "MAE_clean_specs_only": round(mae_clean, 4) if mae_clean else None,
           "MAE_clean_specs_persistence": round(mae_clean_persist, 4) if mae_clean_persist else None,
           "interval_coverage_80": round(cov, 3) if cov is not None else None,
           "specs_clean_no_repair": clean, "specs_repaired": repaired,
           "flagged_by_validator": flagged,
           "backend": "Qwen-2.5-72B via HF (swap anthropic_compile_fn for a paid frontier backend)",
           "per_topic": scored}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-068  scored end-to-end: full autonomous pipeline on forecastable GSS opinion questions")
    print(f"  scored {n}/{len(pairs)} topics (LLM compiled the whole spec; validated + repaired; ~12yr horizon)")
    if n:
        print(f"  MAE pipeline (all {n}) = {mae:.4f}   vs persistence {mae_persist:.4f}   "
              f"-> skill {skill:+.3f} ({'beats' if skill > 0 else 'does not beat'} persistence)")
        if mae_clean is not None:
            print(f"  MAE on the {clean} CLEANLY-compiled specs = {mae_clean:.4f} vs persistence "
                  f"{mae_clean_persist:.4f} (skill {1 - mae_clean/mae_clean_persist:+.3f})")
        print(f"  80% interval coverage = {cov:.2f}   |  specs clean(no-repair)={clean}, repaired={repaired}")
        if flagged:
            print(f"  validator flagged/excluded {len(flagged)}: "
                  f"{[(x['topic'], x.get('issues') or x.get('error','')) for x in flagged]}")
        for s in scored:
            mark = "OK " if abs(s["pred"] - s["truth"]) <= abs(s["persistence"] - s["truth"]) else "   "
            print(f"    {mark}{s['topic']:9s} {s['A']}->{s['T']}  truth={s['truth']}  pred={s['pred']}  "
                  f"persist={s['persistence']}  covered={s['covered']}")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
