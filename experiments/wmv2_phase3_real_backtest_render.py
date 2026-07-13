"""Render the two Phase-3 real-backtest Markdown docs from the machine-readable artifact.

Reads experiments/results/phase3/real_backtest.json (the only source of truth) and writes:
  docs/WMV2_PHASE3_REAL_BACKTEST.md          — results, per-arm scores, key paired comparison, verdict
  docs/WMV2_PHASE3_REAL_BACKTEST_TRACES.md   — per-question forecasts + as-of leakage/temporal audit traces

Every number here is a copy of a field in the JSON. If the JSON says preliminary, the docs say preliminary.
"""
from __future__ import annotations
import json
from pathlib import Path

ART = Path("experiments/results/phase3/real_backtest.json")
D = json.loads(ART.read_text())
agg = D["aggregate"]
rows = D["rows"]


def f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


VERDICT_LINE = {
    "phase3_improves": "**Phase 3 improves real held-out forecasting.**",
    "inconclusive": "**Result is inconclusive.**",
    "phase3_harms": "**Phase 3 harms forecasting.**",
}[agg["verdict"]]

# ---------------------------------------------------------------- main doc
scored = [r for r in rows if r.get("arms", {}).get("phase3_posterior") is not None]
pa = agg["per_arm_scores"]
key = agg["key_comparison_phase3_vs_phase2"]
boot = key["paired_bootstrap"]

main = []
main.append("# WMv2 Phase 3 — REAL Resolved Historical Backtest\n")
main.append("*Validation only. This run does not redesign Phase 3, does not start Phase 4/9, and does not "
            "weaken any prior result. Every number below is a field of the committed machine-readable artifact "
            "`experiments/results/phase3/real_backtest.json`; failures and regressions are preserved, not "
            "hidden.*\n")
main.append("## Verdict\n")
main.append(VERDICT_LINE + "\n")
main.append(f"> {agg['verdict_meaning']}\n")
if agg["preliminary"]:
    main.append(f"\n**PRELIMINARY** — scored on **{agg['n_scored']}** resolved questions "
                f"(< 30). Treat as directional, not definitive.\n")
main.append("\n## What this measures (and what it refuses to count)\n")
main.append(
    "The key comparison is the **identical production path with the Phase-3 posterior consumed vs. ignored** "
    "— same questions, same frozen `EvidenceBundleV2`, same compiled plan, same qualitative claim tags, same "
    "seed, same outcome contract. The ONLY thing that varies is whether the Phase-3 particle posterior is "
    "materialized onto the plan before rollout. Accuracy is scored against the **realized, resolved outcome** "
    "(not a synthetic one). Posterior *movement* (\"the probability changed\") is explicitly **not** counted "
    "as improvement — only lower loss against the real outcome is.\n")
main.append("\nProduction path per question:\n\n```\nhistorical question\n  -> compile_world (Phase-2 CODE)\n"
            "  -> gather_evidence  [strict as-of: Google News RSS after:/before:, per-doc temporal "
            "verification, claim-level leakage audit]  => ONE frozen EvidenceBundleV2 (reused by all arms)\n"
            "  -> tag_claims [qualitative, no numbers]  => frozen tags (reused by all arms)\n"
            "  -> infer_posterior [particle posterior]\n"
            "  -> materialize onto plan  <== the only thing that varies between the two key arms\n"
            "  -> rollout terminal\n  -> score vs realized outcome\n```\n")

main.append("\n## Arms\n")
main.append("| arm | what it is |\n|---|---|\n"
            "| `prior_only` | reference-class prior mean; no evidence assimilation at the terminal |\n"
            "| `phase2_no_posterior` | Phase-2 evidence path; Phase-3 posterior computed but **not** consumed |\n"
            "| `phase3_posterior` | Phase-3 posterior-conditioned terminal |\n"
            "| `point_estimate` | posterior collapsed to its scalar mean (anti-scalar ablation) |\n"
            "| `market` | crowd / prediction-market implied probability where reliably available |\n")

main.append("\n## Aggregate scores (vs realized outcome, lower is better except directional accuracy)\n")
main.append("| arm | n | Brier ↓ | log-loss ↓ | ECE ↓ | directional acc ↑ | mean p |\n|---|---|---|---|---|---|---|\n")
for a in ["prior_only", "phase2_no_posterior", "phase3_posterior", "point_estimate", "market"]:
    s = pa[a]
    if s.get("n", 0) == 0:
        main.append(f"| `{a}` | 0 | — | — | — | — | — |\n")
    else:
        main.append(f"| `{a}` | {s['n']} | {f(s['brier'])} | {f(s['log_loss'])} | {f(s['ece'])} | "
                    f"{f(s['directional_acc'])} | {f(s['mean_p'])} |\n")

main.append("\n## The key paired comparison — Phase 3 vs. Phase 2 (posterior consumed vs. ignored)\n")
main.append(f"- per-question: **Phase-3 better on {key['per_question_phase3_better']}**, "
            f"Phase-2 better on {key['phase2_better']}, tie on {key['tie']}\n")
if not boot.get("insufficient"):
    main.append(f"- paired bootstrap (n={boot['n']}, arm_a=phase3, arm_b=phase2; **negative = Phase-3 lowers "
                f"loss**):\n")
    main.append(f"  - mean Brier difference **{f(boot['mean_brier_diff'])}**, 95% CI "
                f"**[{f(boot['brier_diff_ci95'][0])}, {f(boot['brier_diff_ci95'][1])}]**\n")
    main.append(f"  - mean log-loss difference **{f(boot['mean_logloss_diff'])}**, 95% CI "
                f"**[{f(boot['logloss_diff_ci95'][0])}, {f(boot['logloss_diff_ci95'][1])}]**\n")
    main.append(f"  - P(bootstrap Brier difference < 0) = **{f(boot['brier_diff_prob_negative'],3)}**\n")
else:
    main.append(f"- paired bootstrap: insufficient paired points (n={boot['n']})\n")

main.append("\n### Per-question Brier deltas (aggregate wins cannot hide a per-question regression)\n")
main.append("| qid | outcome | p(phase3) | p(phase2) | Brier phase3 | Brier phase2 | Δ (ph3−ph2) | verdict |\n"
            "|---|---|---|---|---|---|---|---|\n")
for d in agg["per_question_deltas"]:
    main.append(f"| `{d['qid']}` | {d['outcome']} | {f(d['p_phase3'])} | {f(d['p_phase2'])} | "
                f"{f(d['brier_phase3'])} | {f(d['brier_phase2'])} | {f(d['brier_delta'])} | {d['verdict']} |\n")

main.append("\n## Integrity / reproducibility\n")
main.append(f"- questions attempted: **{agg['n_questions']}**, completed: **{agg['n_completed']}**, "
            f"scored: **{agg['n_scored']}**, harness errors: **{agg['n_harness_error']}**\n")
main.append(f"- within-run numeric-posterior reproducibility (same frozen inputs → identical hash): "
            f"**{f(agg['reproducible_hash_rate'],3)}**\n")
main.append(f"- posterior consumed when evidence present: **{f(agg['posterior_consumed_rate'],3)}**\n")
main.append(f"- retrieval date (UTC): **{D.get('retrieval_date_utc')}**, seed **{D.get('seed')}**\n")
main.append("\n## Honest reading\n")
main.append(
    "This is a real resolved-outcome backtest on the production path, not a synthetic recovery test. "
    "Per-question deltas are reported precisely so an aggregate number cannot mask an individual regression. "
    "Per the acceptance rule, Phase 3 is **NOT** declared empirically validated unless the paired Brier CI "
    "lies entirely below zero. " + VERDICT_LINE.replace("**", "") + "\n")
main.append("\n## Reproduce\n```\nPYTHONPATH=. python experiments/wmv2_phase3_real_backtest.py\n"
            "PYTHONPATH=. python experiments/wmv2_phase3_real_backtest_render.py\n```\n")

Path("docs/WMV2_PHASE3_REAL_BACKTEST.md").write_text("".join(main))

# ---------------------------------------------------------------- traces doc
tr = []
tr.append("# WMv2 Phase 3 — Real Backtest Forecast + Leakage-Audit Traces\n")
tr.append("*Per-question forecasts for every arm plus the strict as-of evidence trace (retrieval window, "
          "document publication dates, per-document temporal status, and claim-level leakage partitions) used "
          "for the manual leakage audit. Source of truth: `experiments/results/phase3/real_backtest.json`.*\n")
tr.append("\n## Manual leakage audit — stratified sample\n")
tr.append("For each question below, the `as_of` is strictly before the resolution date. The retrieval layer "
          "pairs `after:`/`before:` on Google News RSS, runs per-document temporal verification, and a "
          "claim-level leakage audit (post-as-of publication, resolution-term language, retrospective phrasing) "
          "before freezing the bundle. The audit columns below are what a human checks: are any admitted "
          "documents published **after** `as_of`? Do any leakage flags fire? \n")

for r in rows:
    tr.append(f"\n### `{r['qid']}` — {r['question']}\n")
    tr.append(f"- domain **{r.get('domain')}**, as_of **{r.get('as_of')}**, horizon **{r.get('horizon')}**, "
              f"realized outcome **{r.get('outcome')}** — {r.get('resolution_note')}\n")
    tr.append(f"- status **{r.get('status')}**, support grade **{r.get('support_grade')}**, "
              f"latency {f(r.get('latency_s'),1)}s\n")
    a = r.get("arms", {})
    if a:
        tr.append(f"- forecasts — prior_only **{f(a.get('prior_only'))}**, "
                  f"phase2_no_posterior **{f(a.get('phase2_no_posterior'))}**, "
                  f"phase3_posterior **{f(a.get('phase3_posterior'))}**, "
                  f"point_estimate **{f(a.get('point_estimate'))}**, "
                  f"market **{f(a.get('market'))}**\n")
        tr.append(f"- posterior: prior_mean {f(r.get('prior_mean'))} → posterior_mean {f(r.get('posterior_mean'))} "
                  f"(shift {f(r.get('posterior_shift'))}); included claims {r.get('n_included_claims')} → "
                  f"{r.get('n_effective_observations')} effective observations; consumed "
                  f"{r.get('posterior_consumed')}; reproducible_hash {r.get('reproducible_hash')}\n")
    if r.get("error"):
        tr.append(f"- ERROR: {r['error']}\n")
    t = r.get("trace")
    if t:
        tr.append(f"- **as-of audit**: bundle_hash `{t['bundle_hash'][:16]}`, as_of {t['as_of_iso']}, "
                  f"{t['n_documents']} docs, included {t['n_included']} / excluded {t['n_excluded']} / "
                  f"suspicious {t['n_suspicious']} claims, leakage flags {t['n_leakage_flags']}\n")
        if t.get("leakage_flags"):
            tr.append(f"  - leakage flags: {t['leakage_flags']}\n")
        if t.get("documents"):
            tr.append("\n  | doc | source | published | temporal_status |\n  |---|---|---|---|\n")
            for d in t["documents"]:
                tr.append(f"  | {d.get('id')} | {d.get('source')} | {d.get('published_iso') or '—'} | "
                          f"{d.get('temporal_status')} |\n")

Path("docs/WMV2_PHASE3_REAL_BACKTEST_TRACES.md").write_text("".join(tr))
print("wrote docs/WMV2_PHASE3_REAL_BACKTEST.md and docs/WMV2_PHASE3_REAL_BACKTEST_TRACES.md")
print("verdict:", agg["verdict"], "| scored:", agg["n_scored"], "| preliminary:", agg["preliminary"])
