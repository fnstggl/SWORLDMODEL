"""Final benchmark report + per-case detail files (scorer-side: imports the resolution store)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from historical_backtests.framework.resolution_store import read_resolutions

ROOT = Path(__file__).resolve().parents[1]


def _rows(rdir: Path) -> list:
    out = []
    for line in (rdir / "forecast_ledger.jsonl").read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _fmt_ts(ts):
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(float(ts)))
    except (TypeError, ValueError):
        return "—"


def _b(p):
    return "—" if not isinstance(p, (int, float)) else f"{p:.2f}"


def _row_baseline(r, arm):
    for b in (r.get("baselines") or []):
        if b.get("arm") == arm and isinstance(b.get("p"), (int, float)):
            return b["p"]
    return None


def build_report(benchmark_id: str, run_id: str) -> Path:
    bdir = ROOT / "benchmark_versions" / benchmark_id
    rdir = ROOT / "results" / benchmark_id / run_id
    vault = json.loads((bdir / "question_vault.json").read_text())
    cases = {c["case_id"]: c for c in vault["cases"]}
    model_reg = json.loads((ROOT / "models" / "historical_model_registry.json").read_text())
    model = model_reg["models"][0]
    dev = json.loads((rdir / "scores_dev.json").read_text()) \
        if (rdir / "scores_dev.json").exists() else {}
    lock = json.loads((rdir / "scores_locked.json").read_text()) \
        if (rdir / "scores_locked.json").exists() else {}
    rows = _rows(rdir)
    reso = read_resolutions(benchmark_id, purpose=f"report:{run_id}")
    qual = [r for r in rows if r.get("qualified")]
    fails = [r for r in rows if not r.get("qualified")]
    fail_causes = {}
    for r in fails:
        key = (r.get("disqualify_reasons") or ["unknown"])[0].split(":")[0]
        fail_causes[key] = fail_causes.get(key, 0) + 1
    cost = sum((r.get("llm_usage") or {}).get("cost_usd", 0) for r in rows) \
        + sum((r.get("baseline_usage") or {}).get("cost_usd", 0) for r in rows)

    def _verdict():
        pb = (lock.get("paired_vs_baselines") or {}).get("direct_same_model") \
            or (dev.get("paired_vs_baselines") or {}).get("direct_same_model")
        skill = lock.get("capability_normalized_skill", dev.get("capability_normalized_skill"))
        if not pb:
            return "No paired comparison available — no verdict claimed.", False
        lo, hi = pb["ci95"]
        better = pb["mean_diff"] < 0 and hi < 0
        if better:
            return (f"World Model V2 IMPROVED on the direct same-model baseline: paired Brier "
                    f"diff {pb['mean_diff']:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}], excludes 0); "
                    f"capability-normalized skill {skill:+.3f}."), True
        return (f"World Model V2 did NOT demonstrably improve on the direct same-model baseline: "
                f"paired Brier diff {pb['mean_diff']:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}] "
                f"includes 0 or favors direct). Capability-normalized skill "
                f"{skill if skill is not None else '—'}. Reported as-is."), False
    verdict, _improved = _verdict()

    m = [f"# WMv2 OpenRouter Historical Backtest — RESULTS ({benchmark_id}, {run_id})\n",
         f"*Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}*\n",
         "\n## 1–4. Model, provider, tier, entrypoint\n",
         f"1. Historical model: **{model['exact_checkpoint']}** "
         f"(OpenRouter `{model['openrouter_slug']}`, HF rev `{model['hf_revision_sha'][:12]}…`), "
         f"knowledge cutoff {model['base_model_knowledge_cutoff']}, "
         f"released {model['public_release_timestamp'][:10]}.\n",
         f"2. Provider/quantization: **{model['openrouter_provider_display']} / "
         f"{model['quantization']}** — pinned, no fallbacks, per-call audit ledgers.\n",
         f"3. Tier: **{model['temporal_safety_tier']}** — {model['mutable_serving_limitation']}\n",
         "4. Entrypoint: `swm.world_model_v2.unified_runtime.simulate_world` for EVERY row "
         "(sentinel tests prove the legacy simplified path is never used).\n",
         "\n## 5–9. Execution accounting\n",
         f"5. Full-run proof per row: PhaseExecutionRecords for the complete phase contract, "
         f"operator delta census, particle counts, terminal source — stored in every ledger row; "
         f"qualification gates in `framework/qualify.py`.\n",
         f"6. Questions selected: **{vault['n_cases']}** (4 cutoffs each).\n",
         f"7. Expected complete runs: **{vault['n_cases'] * 4}**.\n",
         f"8. Attempted: **{len(rows)}**; qualified full-system runs: **{len(qual)}**.\n",
         f"9. Incomplete/disqualified: **{len(fails)}** — causes: {fail_causes}.\n",
         "\n## 10–12. Scores\n"]
    if lock:
        o = lock.get("overall") or {}
        m.append(f"10. **Rotating locked test** (opened ONCE): n={o.get('n')}, "
                 f"Brier **{o.get('brier')}**, log-loss {o.get('log_loss')}, "
                 f"AUROC {o.get('auroc')}, ECE {o.get('ece')}, "
                 f"CRPS {o.get('event_time_crps')}, 80% coverage "
                 f"{o.get('interval_coverage_80')}.\n")
        m.append("    Baselines (same model, same evidence): \n")
        for arm, blk in (lock.get("baseline_blocks") or {}).items():
            pb = (lock.get("paired_vs_baselines") or {}).get(arm) or {}
            m.append(f"    - {arm}: Brier {blk.get('brier')} | paired diff "
                     f"{pb.get('mean_diff')} CI95 {pb.get('ci95')}\n")
    if dev:
        o = dev.get("overall") or {}
        m.append(f"\n    Dev splits (REUSABLE_DEVELOPMENT_BACKTEST): n={o.get('n')}, "
                 f"Brier {o.get('brier')}, AUROC {o.get('auroc')}; "
                 f"skill vs direct {dev.get('capability_normalized_skill')}.\n")
    m.append("\n## 13. Results by causal scale\n")
    src = lock or dev
    for scale, blk in sorted((src.get("by_scale") or {}).items()):
        m.append(f"- {scale}: n={blk.get('n')}, Brier {blk.get('brier')}, "
                 f"AUROC {blk.get('auroc')}\n")
    m.append("\n## 14. Leakage census\n")
    m.append("- Evidence: archived-bytes only (Wayback capture proofs + Wikipedia revids); "
             "contamination scrub counts in `evidence_archives/_build_stats.json`; capsules "
             "sealed before simulation; deterministic query generation (no frontier model "
             "anywhere in the case-dependent path).\n")
    m.append(f"\n## 15. Verdict\n\n**{verdict}**\n")
    m.append("\n## Locked-row table\n\n| question | cutoff | outcome | WMv2 | direct | market "
             "| dominant mode | median t | qualified |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        c = cases.get(r["case_id"]) or {}
        if c.get("split") != "rotating_locked":
            continue
        rz = reso.get(r["case_id"]) or {}
        evt = r.get("event_time") or {}
        md = evt.get("mode_distribution") or {}
        dom = max(md, key=md.get) if md else "—"
        med = (evt.get("first_passage_quantiles_ts") or {}).get("0.5")
        m.append(f"| {r['raw_question'][:60]} | {r['cutoff'][:10]} | "
                 f"{rz.get('actual_outcome', '—')} | {_b(r.get('p_yes'))} | "
                 f"{_b(_row_baseline(r, 'direct_same_model'))} | "
                 f"{_b(_row_baseline(r, 'market_price_at_cutoff'))} | {dom} | "
                 f"{_fmt_ts(med) if isinstance(med, (int, float)) else '—'} | "
                 f"{'Y' if r.get('qualified') else 'N: ' + str((r.get('disqualify_reasons') or ['?'])[0])[:40]} |\n")
    m.append(f"\n## Cost\n\nTotal OpenRouter spend this run: **${cost:.2f}** "
             f"across {sum((r.get('llm_usage') or {}).get('n_calls', 0) for r in rows)} "
             f"primary calls + baselines.\n")
    out = rdir / "final_report.md"
    out.write_text("".join(m))
    # ---- per-case detail files: biggest hits/misses vs direct ----
    det = rdir / "case_details"
    det.mkdir(exist_ok=True)
    scored = []
    for r in qual:
        rz = reso.get(r["case_id"]) or {}
        d = _row_baseline(r, "direct_same_model")
        if isinstance(r.get("p_yes"), (int, float)) and rz:
            y = rz["actual_outcome"]
            e_w = (r["p_yes"] - y) ** 2
            e_d = (d - y) ** 2 if isinstance(d, (int, float)) else None
            scored.append((r, y, e_w, e_d))
    scored.sort(key=lambda x: x[2])
    picks = {"most_accurate": scored[:5], "largest_errors": scored[-5:],
             "wmv2_beats_direct": sorted([s for s in scored if s[3] is not None],
                                         key=lambda x: x[2] - x[3])[:5],
             "wmv2_trails_direct": sorted([s for s in scored if s[3] is not None],
                                          key=lambda x: x[3] - x[2])[:5]}
    for group, items in picks.items():
        for r, y, e_w, e_d in items:
            evt = r.get("event_time") or {}
            lin = r.get("lineage_event_time") or {}
            body = {"group": group, "question": r["raw_question"], "cutoff": r["cutoff"],
                    "actual_outcome": y, "wmv2_p": r.get("p_yes"),
                    "direct_p": _row_baseline(r, "direct_same_model"),
                    "market_p": _row_baseline(r, "market_price_at_cutoff"),
                    "brier_wmv2": round(e_w, 4),
                    "brier_direct": round(e_d, 4) if e_d is not None else None,
                    "modes": lin.get("modes"), "hazard_ratio_by_mode":
                        lin.get("hazard_ratio_by_mode"),
                    "decision_structures": lin.get("decision_structures"),
                    "stances": (r.get("actor_intentions") or {}).get("intentions"),
                    "mode_distribution": evt.get("mode_distribution"),
                    "cdf": evt.get("cdf"), "support_grade": r.get("support_grade"),
                    "limitations": r.get("limitations"),
                    "full_run_proof": r.get("full_run_proof")}
            (det / f"{group}__{r['case_id']}__{r['cutoff'][:10]}.json").write_text(
                json.dumps(body, indent=1, default=str))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    print(build_report(a.benchmark, a.run))
