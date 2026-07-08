"""The no-cheat backtest harness — run the world-model over resolved forecasting questions and score it against
the crowd, with a built-in parametric-leakage meter.

For each item the forecaster (a) COMPILES the question into a structural spec AS OF the question date (the LLM
is told to use only information available then, and to define the outcome event so P(event) = P(the question
resolves YES)); (b) RUNS the Monte-Carlo — so the LLM states variables and a mechanism, never the outcome; the
simulation produces the probability (the decomposition defense against leakage). Then:

  - SKILL vs the CROWD: Brier/log-loss of the model vs the crowd probability at the same as-of, vs the base
    rate. skill = 1 − loss_model/loss_crowd (>0 ⇒ the simulation beat the market).
  - LEAKAGE METER: `direct_estimate` asks the LLM for P(YES) directly (pure parametric). If the simulated
    forecast tracks the direct one AND both crush the crowd on pre-cutoff items, we are measuring memory, not
    forecasting — so the honest headline uses the CUTOFF-CLEAN slice and reports the model-vs-direct gap.

Ablation toggles operate on the compiled spec (cheap re-simulation, LLM calls served from cache): `ground`
(use the compiler's as-of state estimates vs neutral values), `force_readout` (compiler-chosen mechanism vs a
forced calibrated readout), `max_vars` (variable-count via variance triage).
"""
from __future__ import annotations

import datetime as _dt
import math

from swm.api.compiler import CompiledModel, build_compile_prompt
from swm.api.adaptive_fidelity import triage
from swm.api.model_spec import parse_spec
from swm.api.retrieval_grounding import parse_json_lenient
from swm.eval.metrics import brier_score, log_loss


def _clip(p, lo=0.02, hi=0.98):
    return min(hi, max(lo, float(p)))


def as_of_str(ts):
    return _dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")


def _asof_context(item):
    return (f"TODAY'S DATE IS {as_of_str(item.as_of)}. Use ONLY information that was available on or before "
            f"this date; do NOT use any knowledge of events after it. The outcome.event MUST be defined so that "
            f"P(event) equals the probability that the question resolves YES.")


def _p_from_forecast(fc):
    # For a calibrated_readout the readout MEAN is P(YES) directly. Using p_event = P(readout > 0.5) is a
    # category error: it BINARIZES a probability (0.7 -> 1.0, 0.5 -> 0.0), manufacturing false extremes.
    if fc.get("mechanism") == "calibrated_readout" and fc.get("mean") is not None:
        return fc["mean"]
    if fc.get("p_event") is not None:
        return fc["p_event"]
    if fc.get("p_target") is not None:
        return fc["p_target"]
    if fc.get("mean") is not None and 0.0 <= fc["mean"] <= 1.0:
        return fc["mean"]
    return None


def _apply_toggles(spec, *, ground=True, max_vars=None):
    """Cheap architecture ablations on a compiled spec (no new LLM calls)."""
    if not ground:                                    # neutralize the state estimates -> weights/intercept only
        for v in spec.variables:
            v.value = getattr(v, "center", 0.5) if v.weight is not None else v.value
            v.est_sd = 0.0
    if max_vars and spec.variables and any(v.weight is not None for v in spec.variables):
        keep = set(triage(spec, keep_frac=1.0)["invest_in"][:max_vars]) if len(spec.variables) > max_vars else None
        if keep:
            for v in spec.variables:
                if v.name not in keep and v.weight is not None:
                    v.weight = 0.0
    return spec


def compile_spec(item, llm, *, force_readout=False):
    prompt = build_compile_prompt(item.question, _asof_context(item))
    if force_readout:
        prompt += ("\nUse the calibrated_readout mechanism: list the variables that press the outcome toward "
                   "YES with signed elasticities and an intercept at the base rate.")
    try:
        return parse_spec(llm(prompt))
    except Exception:
        return None


def forecast_item(item, llm, *, n=2500, ground=True, force_readout=False, max_vars=None, seed=0):
    """Compile as-of -> simulate -> P(YES). Returns (p, meta) or (None, meta) if it could not produce one."""
    spec = compile_spec(item, llm, force_readout=force_readout)
    if spec is None:
        return None, {"error": "compile"}
    spec = _apply_toggles(spec, ground=ground, max_vars=max_vars)
    try:
        fc = CompiledModel(spec).run(n=n, seed=seed)
    except Exception as e:
        return None, {"error": f"run:{str(e)[:40]}", "mechanism": spec.mechanism}
    p = _p_from_forecast(fc)
    return (None if p is None else _clip(p)), {"mechanism": spec.mechanism,
                                               "n_vars": len(spec.variables), "raw_p": p}


def direct_estimate(item, llm):
    """LEAKAGE METER: the LLM's direct P(YES) (pure parametric, no simulation)."""
    q = (f"{_asof_context(item)}\nQuestion: {item.question}\n"
         f'Give your probability that this resolves YES. Return ONLY JSON: {{"p": <0..1>}}.')
    r = parse_json_lenient(llm(q))
    if not r or r.get("p") is None:
        return None
    try:
        return _clip(r["p"])
    except Exception:
        return None


def _skill(loss_model, loss_base):
    return round(1 - loss_model / loss_base, 4) if loss_base > 1e-9 else None


def score(rows):
    """rows: list of {outcome, p_model, p_crowd, (p_direct), category, cutoff_clean}. Returns overall +
    per-slice Brier/log-loss and SKILL vs crowd and base rate, for the model and (if present) the direct LLM."""
    def _agg(sub):
        if not sub:
            return None
        y = [r["outcome"] for r in sub]
        base = sum(y) / len(y)
        out = {"n": len(sub), "base_rate": round(base, 4),
               "brier_model": round(brier_score(y, [r["p_model"] for r in sub]), 4),
               "brier_crowd": round(brier_score(y, [r["p_crowd"] for r in sub]), 4),
               "ll_model": round(log_loss(y, [r["p_model"] for r in sub]), 4),
               "ll_crowd": round(log_loss(y, [r["p_crowd"] for r in sub]), 4)}
        bll = log_loss(y, [base] * len(sub))
        out["skill_vs_crowd"] = _skill(out["ll_model"], out["ll_crowd"])
        out["skill_vs_base"] = _skill(out["ll_model"], bll)
        out["crowd_skill_vs_base"] = _skill(out["ll_crowd"], bll)
        if all(r.get("p_direct") is not None for r in sub):
            out["ll_direct"] = round(log_loss(y, [r["p_direct"] for r in sub]), 4)
            out["direct_skill_vs_crowd"] = _skill(out["ll_direct"], out["ll_crowd"])
        return out

    res = {"overall": _agg(rows), "clean": _agg([r for r in rows if r.get("cutoff_clean")]),
           "by_category": {}, "by_crowd_confidence": {}}
    for c in sorted({r["category"] for r in rows}):
        res["by_category"][c] = _agg([r for r in rows if r["category"] == c])
    # where the crowd is UNSURE (0.35-0.65) is where a real model can add the most
    res["by_crowd_confidence"]["uncertain(.35-.65)"] = _agg([r for r in rows if 0.35 <= r["p_crowd"] <= 0.65])
    res["by_crowd_confidence"]["confident"] = _agg([r for r in rows if not 0.35 <= r["p_crowd"] <= 0.65])
    return res
