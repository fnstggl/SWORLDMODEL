"""EXP-067: the spec validator + repair loop — a linter and smoke-test for LLM-compiled models.

Closes the one real failure EXP-066 found: the LLM gets the structure right but makes numeric bugs in the
equations. This is the linter + test-run that catches them before you trust the spec.

  A. THE REAL BUG, CAUGHT AND REPAIRED. Qwen's actual inflation spec (committed from EXP-066) — whose
     mean-reversion equation has an equilibrium at ~35% against a hi bound of 10, so it pinned to the bound
     and gave P=1.0 — is run through `validate`. It flags the exact defect (no interior equilibrium ->
     saturates the bound; degenerate outcome; trivial event). Then the repair loop hands the spec + issues
     back to an LLM to fix; the repaired spec passes clean and forecasts sanely.

  B. NO FALSE POSITIVES. Clean specs (the EXP-064 incumbent SCM, a committee, a bracket) pass with zero
     errors — the validator does not cry wolf.

  C. EACH CHECK FIRES. A battery of deliberately-broken specs shows every check triggers: equilibrium out
     of bounds, event threshold outside support, value out of bounds, volatility too large.

Run (HF_TOKEN in env attempts a LIVE LLM repair; otherwise a committed reference fix is used):
  HF_TOKEN=... python -m experiments.exp067_spec_validator
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from swm.api.compiler import CompiledModel, StructuralCompiler
from swm.api.model_spec import parse_spec
from swm.api.spec_validator import ValidatingCompiler, build_repair_prompt, validate

BUGGY = "experiments/results/exp066/qwen_fullspec.json"
REPAIR_CACHE = "experiments/results/exp067/repaired_spec.json"
RESULT = "experiments/results/exp067_spec_validator.json"

# committed reference fix (used if no live LLM backend): mean-revert toward 3, inside the [0,10] bound
REFERENCE_FIX = {"mechanism": "generic_scm",
                 "variables": [{"name": "CPI_inflation", "value": 4.2, "est_sd": 0.5, "volatility": 0.3,
                                "lo": 0, "hi": 10}],
                 "equations": {"CPI_inflation": "-0.3*(CPI_inflation - 3)"},
                 "outcome": {"variable": "CPI_inflation", "event": {"op": ">", "value": 3}},
                 "horizon": 12, "dt": 1, "rationale": "mean-reversion toward 3%, inside bounds"}


def _repair_fn():
    """A repair backend: try a live LLM (HF) if a token is present, else return the committed reference fix."""
    if Path(REPAIR_CACHE).exists():
        cached = json.loads(Path(REPAIR_CACHE).read_text())
        return lambda prompt: cached, "cached"
    if os.environ.get("HF_TOKEN"):
        from swm.api.hf_backend import hf_chat_fn
        raw_fn = hf_chat_fn(system="You fix bugs in structural-model JSON specs. Output ONLY corrected JSON.",
                            max_tokens=700)
        def fn(prompt):
            raw = raw_fn(prompt)
            obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            Path(REPAIR_CACHE).parent.mkdir(parents=True, exist_ok=True)
            Path(REPAIR_CACHE).write_text(json.dumps(obj, indent=1))
            return obj
        return fn, "live-hf"
    return lambda prompt: REFERENCE_FIX, "reference"


def _part_a():
    buggy = json.loads(Path(BUGGY).read_text())
    pre_issues = [i.as_dict() for i in validate(parse_spec(buggy))]
    pre_forecast = CompiledModel(parse_spec(buggy)).run(n=2000)

    repair, backend = _repair_fn()
    try:
        vc = ValidatingCompiler(compiler=StructuralCompiler(lambda key: buggy), repair_fn=repair, max_repairs=2)
        compiled = vc.compile("Will US CPI inflation be above 3% at year-end?", key="cpi")
        report = vc.last_report
    except Exception as e:                                  # live backend failed -> committed reference fix
        backend = f"{backend}->reference (live failed: {str(e)[:40]})"
        vc = ValidatingCompiler(compiler=StructuralCompiler(lambda key: buggy),
                                repair_fn=lambda p: REFERENCE_FIX, max_repairs=2)
        compiled = vc.compile("Will US CPI inflation be above 3% at year-end?", key="cpi")
        report = vc.last_report
    post_forecast = compiled.run(n=4000)
    return {"repair_backend": backend, "issues_flagged": pre_issues,
            "buggy_forecast": {"p_event": pre_forecast["p_event"], "interval": pre_forecast["interval_80"]},
            "repaired_clean": report["clean"], "repairs": report["repairs"],
            "repaired_equation": compiled.spec.equations,
            "repaired_forecast": {"p_event": post_forecast["p_event"], "mean": post_forecast["mean"],
                                  "interval": post_forecast["interval_80"]}}


def _part_b():
    clean = {
        "incumbent_scm": {"mechanism": "generic_scm",
            "variables": [{"name": "approval", "value": 0.47, "est_sd": 0.03, "volatility": 0.02},
                          {"name": "vote", "value": 0.49, "est_sd": 0.01, "volatility": 0.015}],
            "equations": {"vote": "0.3*(0.5*approval + 0.25 - vote)"},
            "outcome": {"variable": "vote", "event": {"op": ">", "value": 0.5}}, "horizon": 26},
        "committee": {"mechanism": "committee", "outcome": {"event": {"op": ">", "value": 0.5}},
            "extra": {"agents": [{"id": f"a{i}", "position": p} for i, p in enumerate([0.8, 0.7, 0.3])]}},
        "bracket": {"mechanism": "bracket", "outcome": {"target": "A"},
            "extra": {"competitors": [{"name": "A", "strength": 1650}, {"name": "B", "strength": 1550}]}}}
    return {name: {"errors": [i.as_dict() for i in validate(parse_spec(s)) if i.severity == "error"]}
            for name, s in clean.items()}


def _part_c():
    battery = {
        "equilibrium_out_of_bounds": {"mechanism": "generic_scm",
            "variables": [{"name": "x", "value": 0.5, "volatility": 0.01, "lo": 0, "hi": 1}],
            "equations": {"x": "-0.4*(x - 2.0)"}, "outcome": {"variable": "x"}},
        "event_threshold_outside_support": {"mechanism": "generic_scm",
            "variables": [{"name": "x", "value": 0.5, "volatility": 0.02, "lo": 0, "hi": 1}],
            "equations": {"x": "-0.2*(x - 0.5)"}, "outcome": {"variable": "x", "event": {"op": ">", "value": 5}}},
        "value_out_of_bounds": {"mechanism": "generic_scm",
            "variables": [{"name": "x", "value": 3.0, "volatility": 0.01, "lo": 0, "hi": 1}],
            "equations": {"x": "-0.2*(x - 0.5)"}, "outcome": {"variable": "x"}},
        "volatility_too_large": {"mechanism": "generic_scm",
            "variables": [{"name": "x", "value": 0.5, "volatility": 2.0, "lo": 0, "hi": 1}],
            "equations": {"x": "-0.2*(x - 0.5)"}, "outcome": {"variable": "x"}, "horizon": 9}}
    return {name: sorted({i.code for i in validate(parse_spec(s))}) for name, s in battery.items()}


def run():
    A, B, C = _part_a(), _part_b(), _part_c()
    out = {"A_real_bug_caught_and_repaired": A, "B_no_false_positives": B, "C_each_check_fires": C}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-067  spec validator + repair loop (linter + test-run for LLM-compiled models)")
    print("  A. THE REAL INFLATION BUG:")
    print(f"     validator flagged: {[i['code'] for i in A['issues_flagged']]}")
    print(f"     buggy forecast: P={A['buggy_forecast']['p_event']} interval={A['buggy_forecast']['interval']}")
    print(f"     repair backend={A['repair_backend']} -> clean={A['repaired_clean']} in {A['repairs']} round(s)")
    print(f"     repaired equation: {A['repaired_equation']}")
    print(f"     repaired forecast: P={A['repaired_forecast']['p_event']} "
          f"mean={A['repaired_forecast']['mean']} interval={A['repaired_forecast']['interval']}")
    print("  B. NO FALSE POSITIVES (clean specs -> 0 errors):")
    for name, r in B.items():
        print(f"     {name:16s} errors={len(r['errors'])}")
    print("  C. EACH CHECK FIRES:")
    for name, codes in C.items():
        print(f"     {name:32s} -> {codes}")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
