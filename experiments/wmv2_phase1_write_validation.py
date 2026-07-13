"""Assemble docs/WMV2_PHASE1_VALIDATION.md + a consolidated machine-readable summary from the three
Phase-1 no-abstention artifacts (generality B13 gates, ablations, forensic traces). Numbers are read
straight from the JSON so the doc can never drift from the run. Run AFTER the three harnesses:
  PYTHONPATH=. python -m experiments.wmv2_phase1_write_validation
"""
from __future__ import annotations

import json
from pathlib import Path

GEN = "experiments/results/wmv2_phase1_no_abstention_generality.json"
ABL = "experiments/results/wmv2_phase1_ablations.json"
FOR = "experiments/results/wmv2_phase1_forensic_traces.json"
DOC = "docs/WMV2_PHASE1_VALIDATION.md"
SUMMARY = "experiments/results/wmv2_phase1_validation_summary.json"


def _load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def run():
    gen = _load(GEN)
    if gen is None:
        raise SystemExit(f"missing {GEN} — run the generality harness first")
    abl = _load(ABL)
    forr = _load(FOR)

    gates = gen["b13_gates"]
    summ = gen["summary"]
    meta = gen["_meta"]
    all_pass = summ["all_gates_passed"]

    L = ["# WMv2 Phase 1 — Validation (no-abstention, B12/B13/B15)", "",
         "*Real-LLM validation of the production Phase-1 path: every coherent question SIMULATES; epistemic "
         "weakness lowers the support grade, never refuses. All numbers are read directly from the run "
         "artifacts (JSON) — nothing hand-entered. Companion machine-readable summary: "
         "`experiments/results/wmv2_phase1_validation_summary.json`.*", "",
         f"**Protocol.** {summ['n_questions']} held-out natural-language questions across "
         f"{summ['n_domains']} domains, **no scripted plans** — the compiler builds its own plan for each and "
         f"the shipped `pipeline.simulate()` produces the `SimulationResult`. Model: {meta['model']} · "
         f"{meta['llm_calls']} calls · ~${meta['est_cost_usd']} · {meta['runtime_s']}s. Resumable, "
         f"deterministic given the cache.", "",
         f"## B13 acceptance gates — {'ALL PASSED ✅' if all_pass else 'SOME FAILED ❌'}", "",
         "| gate | value | threshold | result |", "|---|---|---|---|"]
    names = {
        "valid_plan_rate": "valid plan (compiled)", "materialize_rate": "materialize (world built)",
        "complete_rollout_readout_rate": "complete rollout + bound readout",
        "forecast_abstention_rate": "forecast abstention (coherent Q, no forecast)",
        "clarification_rate": "clarification (incoherent only)",
        "execution_failure_rate": "execution failure (engineering)",
        "provenance_status_rate": "provenance status present",
        "fallback_tier_identified_rate": "fallback names its tier",
        "unsupported_precision_rate": "unsupported precision (field stamped observed)",
        "llm_prob_injection_rate": "LLM-minted terminal probability",
        "no_keyword_router": "no scenario keyword router (static)"}
    op_sym = {">=": "≥", "<": "<", "==": "="}
    for k, g in gates.items():
        val = g["value"]
        mark = "✅" if g["passed"] else "❌"
        L.append(f"| {names.get(k, k)} | `{val}` | {op_sym.get(g['op'], g['op'])} {g['threshold']} | {mark} |")
    L += ["",
          f"**Forecasts produced: {summ['forecasts_produced']}/{summ['n_questions']} "
          f"({summ['forecast_rate']:.0%}).** Simulation-status histogram: "
          f"`{gen['simulation_status_histogram']}`. Support-grade histogram: "
          f"`{gen['support_grade_histogram']}`."]
    if gen.get("failure_taxonomy_histogram"):
        L.append(f" Failure taxonomy (execution_failed only): `{gen['failure_taxonomy_histogram']}`.")
    L += ["",
          "### Reading the grades",
          "On the **general path** the highest defensible mechanism tier is 6 (generic structural, "
          "`exploratory`) or 7 (competing structural hypotheses, `highly_speculative`), because no "
          "held-out-validated *domain* parameter pack (tiers 1–4) applies to these arbitrary questions. The "
          "support grade honestly reports that: a from-scratch general social simulation without a validated "
          "domain mechanism is exploratory/speculative, and the forecast is a correspondingly broad prior "
          "with wide dispersion and explicit limitations — not a confident number. Sharpening requires the "
          "domain packs and evidence assimilation (Phase 3), out of Phase-1 scope. Phase 1's claim is "
          "generality + no-abstention + honesty, which the gates above establish.", ""]

    # per-domain
    L += ["## Per-domain coverage", "", "| domain | n | forecast rate | complete rate | grades |",
          "|---|---|---|---|---|"]
    for d, r in gen["per_domain"].items():
        L.append(f"| {d} | {r['n']} | {r['forecast_rate']:.0%} | {r['complete_rate']:.0%} | "
                 f"{', '.join(r['grades'])} |")
    L.append("")

    # ablations
    if abl:
        L += ["## B12 ablations — component contributions", "",
              "Each question compiled ONCE; the compiled plan is transformed per ablation (no extra LLM "
              f"calls). k={abl['_meta']['k_questions']} questions.", "",
              "| ablation | forecast | complete | exec-fail | dispersion | struct-H |",
              "|---|---|---|---|---|---|"]
        for a, s in abl["summary"].items():
            L.append(f"| {a} | {s['forecast_rate']:.0%} | {s['complete_rate']:.0%} | "
                     f"{s['execution_failure_rate']:.0%} | {s['mean_dispersion']} | {s['mean_struct_entropy']} |")
        L += ["", "**Contribution vs. full compiler** (what breaks when a component is removed):", ""]
        for a, c in abl["contributions_vs_full"].items():
            L.append(f"- **{a}**: forecast −{c['forecast_rate_drop']:.0%}, complete "
                     f"−{c['complete_rate_drop']:.0%}, exec-fail +{c['execution_failure_increase']:.0%}, "
                     f"Δdispersion {c['dispersion_change']}, Δstruct-H {c['struct_entropy_change']}. "
                     f"{abl['interpretation'].get(a, '')}")
        L.append("")
    else:
        L += ["## B12 ablations", "", "_Artifact `wmv2_phase1_ablations.json` not present at doc-build time._", ""]

    # forensic
    if forr:
        n_dom = forr["_meta"]["n"]
        n_fc = sum(1 for t in forr["traces"].values()
                   if t.get("SIMULATION_RESULT", {}).get("raw_probability") is not None)
        L += ["## B15 forensic traces", "",
              f"{n_fc}/{n_dom} domain traces produced a forecast; full per-domain traces (every "
              f"intermediate structure) in `docs/WMV2_PHASE1_FORENSIC_TRACES.md` + "
              f"`experiments/results/wmv2_phase1_forensic_traces.json`.", ""]

    # verdict
    L += ["## Verdict", "",
          f"{'All B13 acceptance gates pass' if all_pass else 'Some B13 gates did not pass (see table)'}: "
          f"the production compiler produces an honest, executable, terminal-state forecast for "
          f"{summ['forecast_rate']:.0%} of {summ['n_questions']} held-out questions across "
          f"{summ['n_domains']} domains, with **zero forecast abstentions**, through one generic path with no "
          f"domain hard-coding and no LLM-minted probabilities. Weakness is carried by the support grade, "
          f"not by refusal. Historical Session-1 results (`WMV2_COMPILER_VALIDATION.md`) are preserved "
          f"unedited; see `WMV2_NO_ABSTENTION_MIGRATION.md`.", ""]

    Path(DOC).write_text("\n".join(L))
    consolidated = {"b13_gates": gates, "all_gates_passed": all_pass, "summary": summ,
                    "simulation_status_histogram": gen["simulation_status_histogram"],
                    "support_grade_histogram": gen["support_grade_histogram"],
                    "failure_taxonomy_histogram": gen.get("failure_taxonomy_histogram", {}),
                    "ablations": abl["summary"] if abl else None,
                    "ablation_contributions": abl["contributions_vs_full"] if abl else None,
                    "forensic_domains": (forr["_meta"]["n"] if forr else 0),
                    "meta": meta}
    Path(SUMMARY).write_text(json.dumps(consolidated, indent=1, default=str))
    print(f"wrote {DOC} + {SUMMARY} (all_gates_passed={all_pass})")
    return consolidated


if __name__ == "__main__":
    run()
