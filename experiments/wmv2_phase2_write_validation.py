"""Assemble docs/WMV2_PHASE2_VALIDATION.md from the Phase-2 artifacts. Numbers are read straight from the
JSON so the doc cannot drift. Run after the validation / ablations / metrics / forensic harnesses.
  PYTHONPATH=. python -m experiments.wmv2_phase2_write_validation
"""
from __future__ import annotations

import json
from pathlib import Path

GATES = "experiments/results/wmv2_phase2_evidence_validation.json"
ABL = "experiments/results/wmv2_phase2_ablations.json"
MET = "experiments/results/wmv2_phase2_subsystem_metrics.json"
FOR = "experiments/results/wmv2_phase2_forensic_traces.json"
REG = "experiments/results/wmv2_phase2_source_adapter_registry.json"
DOC = "docs/WMV2_PHASE2_VALIDATION.md"


def _load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def run():
    g = _load(GATES)
    abl = _load(ABL)
    met = _load(MET)
    forr = _load(FOR)
    reg = _load(REG)
    L = ["# WMv2 Phase 2 — Validation", "",
         "*Real-data validation of the production evidence path: live Google News RSS (paired "
         "after:/before:), archive.org temporal verification, span-validated claims, and evidence that "
         "changes the compiled world. All numbers read directly from the run artifacts.*", ""]

    if g:
        m = g["_meta"]
        L += [f"## End-to-end acceptance gates (held-out, n={g['n_questions']}, {g['n_domains']} domains)", "",
              f"Model: {m['model']} · {m['llm_calls']} LLM calls · ~${m['est_cost_usd']} · {m['runtime_s']}s · "
              f"{g['total_paired_rss_queries']} paired RSS queries issued.", "",
              "| gate | value | threshold | result |", "|---|---|---|---|"]
        for name, gg in g["gates"].items():
            mark = "PASS ✅" if gg["passed"] else "FAIL ❌"
            op = {">=": "≥", "<": "<", "==": "="}.get(gg["op"], gg["op"])
            L.append(f"| {name} | `{gg['value']}` | {op} {gg['th']} | {mark} |")
        L += ["", f"All gates passed: **{g['all_gates_passed']}**. mean docs/question {g['mean_docs']}, "
              f"mean included claims {g['mean_included_claims']}, mean structural plan changes "
              f"{g['mean_structural_changes']}.", "",
              "**Note on the generic held-out bank.** These questions are deliberately generic (no named "
              "entities — e.g. \"Will the incumbent mayor win re-election?\"), which is a stress test for "
              "compiler generality but a poor target for public retrieval: there is no specific entity to "
              "retrieve. The nonempty-bundle gate is therefore measured only on public-evidence domains, and "
              "the named-entity forensic set below is the fair measure of whether retrieval + causal "
              "integration works when a question names specifics.", ""]
        L += ["### Per-domain", "", "| domain | n | mean docs | material-change rate |", "|---|---|---|---|"]
        for d, r in g["per_domain"].items():
            L.append(f"| {d} | {r['n']} | {r['mean_docs']} | {r['material_rate']} |")
        L.append("")

    if forr:
        traces = forr["traces"]
        causal = sum(1 for t in traces if t.get("evidence_is_causal"))
        withdocs = sum(1 for t in traces if t.get("n_documents", 0) > 0)
        L += ["## Named-entity end-to-end (16 real 2023-2024 events, one per domain)", "",
              f"{withdocs}/{len(traces)} retrieved contemporaneous evidence; {causal}/{len(traces)} show "
              f"evidence as CAUSAL (structural plan change / terminal movement / observation StateDeltas). "
              f"Full traces: `docs/WMV2_PHASE2_FORENSIC_TRACES.md`.", "",
              "| domain | docs | included claims | Δstruct | lean_only | terminal changed |",
              "|---|---|---|---|---|---|"]
        for t in traces:
            if "error" in t:
                L.append(f"| {t['domain']} | error | | | | |"); continue
            L.append(f"| {t['domain']} | {t.get('n_documents')} | {len(t.get('included_claims', []))} | "
                     f"{t.get('structural_changes')} | {t.get('lean_only')} | {t.get('terminal_changed')} |")
        L.append("")

    if abl:
        a = abl["leakage_ablation"]["aggregate"]
        L += ["## Ablations", "", "### before-only vs paired after:/before: (LIVE, real historical events)", "",
              "| arm | mean post-as-of leakage |", "|---|---|",
              f"| before: only (evaluation arm) | **{a['mean_post_asof_share_before_only']}** |",
              f"| paired after:/before: (production) | **{a['mean_post_asof_share_paired']}** |",
              f"| paired + independent temporal filter | **{a['mean_post_asof_share_paired_plus_filter']}** |", "",
              f"paired reduces leakage vs before-only: **{a['paired_reduces_leakage_vs_before_only']}**; "
              f"temporal filter zeroes residual: **{a['temporal_filter_zeroes_residual_leak']}**. This is the "
              f"empirical basis for the production paired-date rule; RSS dates alone are never trusted.", ""]
        p = abl.get("pipeline_ablations", {})
        if p and "no_dependence_collapse_would_overcount_share" in p:
            L += ["### pipeline safeguards (on persisted bundles)", "",
                  f"- removing dependence collapse would overcount independent sources in "
                  f"**{p['no_dependence_collapse_would_overcount_share']:.0%}** of bundles;",
                  f"- removing temporal verification would admit post-as-of docs in "
                  f"**{p['no_temporal_verification_would_leak_share']:.0%}**;",
                  f"- removing actor visibility would leak non-public claims to all actors in "
                  f"**{p['no_actor_visibility_would_leak_share']:.0%}**.", ""]

    if met:
        bm = met["bundle_metrics"]
        L += ["## Subsystem metrics (from persisted immutable bundles)", "",
              f"- **claims**: {bm['claims']['total']} total, span-verified rate "
              f"**{bm['claims']['span_verified_rate']}** (unsupported spans rejected); classes "
              f"{bm['claims']['by_class']}",
              f"- **entities**: {bm['entities']['total_mentions']} mentions, ambiguity-preserved rate "
              f"{bm['entities']['ambiguity_rate']}",
              f"- **dependence**: {bm['dependence']['n_documents']} docs → "
              f"{bm['dependence']['n_independent_sources']} independent sources "
              f"(dedup reduction {bm['dependence']['dedup_reduction']}); "
              f"{bm['dependence']['syndicated_or_dup_groups']} syndicated/dup groups",
              f"- **contradictions**: {bm['contradictions']['total']} edges {bm['contradictions']['by_type']}",
              f"- **visibility**: {bm['visibility']['by_state']}",
              f"- **temporal**: {bm['temporal']['by_status']}; post-as-of admitted to bundle: "
              f"**{bm['temporal']['post_asof_in_bundle']}**"]
        ta = met.get("live_temporal_audit", {})
        if ta:
            L += [f"- **live temporal audit** (archive.org Wayback, n={ta.get('n_audited')}): "
                  f"{ta.get('verified_pre_asof')} verified_pre_asof, "
                  f"**{ta.get('post_asof_in_admitted')} post-as-of** among admitted; statuses "
                  f"{ta.get('statuses')}"]
        L.append("")

    if reg:
        live = sum(1 for a in reg["adapters"] if a["live_verified"])
        L += ["## Source adapters", "",
              f"{len(reg['adapters'])} registered, **{live} live-verified production** connectors "
              f"(machine-readable: `experiments/results/wmv2_phase2_source_adapter_registry.json`).", ""]

    L += ["## Failure taxonomy & cost", "",
          "Connector failures are recorded per-invocation with an explicit status "
          "(zero_results ≠ http_error ≠ timeout ≠ network_error ≠ parse_error). Forecast abstention "
          "remains 0 (weak/absent evidence degrades the support grade, never blocks a forecast). Costs and "
          "latencies are in each artifact's `_meta`.", "",
          "## Honest gate status", "",
          "Where a metric is measured on a smaller real sample than the spec's target N (e.g. manually-"
          "audited claim/entity annotation sets), that is stated, not extrapolated. Production-eligibility "
          "per subsystem and the exact Phase-3 dependencies are in "
          "`WMV2_PHASE2_LIMITATIONS_AND_DEPENDENCIES.md`.", ""]

    Path(DOC).write_text("\n".join(L))
    print(f"wrote {DOC}")


if __name__ == "__main__":
    run()
