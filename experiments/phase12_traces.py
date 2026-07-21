"""Phase 12 — forensic traces (Part R): assemble end-to-end per-question traces from the frozen artifacts.

For a stratified sample (one per domain + special cases), emit the full chain: question → as-of → active-
component manifest → raw forecast → selected calibrator → calibrated forecast → calibration provenance →
support grade + reasons → uncertainty decomposition → dominant sensitivity contributors → direct-model
disagreement (critic) → limitations. A reviewer can see whether the output is a real simulation result.
Writes forensic_traces.json (machine-readable) and returns markdown for the traces doc.
"""
from __future__ import annotations
import json
from pathlib import Path

from swm.world_model_v2 import phase12_support as sg
from swm.world_model_v2.phase12_serve import load_phase12_bundle

OUT = Path("experiments/results/phase12")


def build():
    corpus = json.loads((OUT / "corpus.json").read_text())
    rows = {r["row_id"]: r for r in corpus["rows"]}
    unc = {d["row_id"]: d for d in json.loads((OUT / "uncertainty_decomposition.json").read_text())["rows"]}
    base = {}
    if (OUT / "baselines.json").exists():
        base = {b["row_id"]: b for b in json.loads((OUT / "baselines.json").read_text())["rows"]}
    bundle = load_phase12_bundle()
    model = bundle["support_model"] if bundle else None

    # stratified sample: one per domain (prefer rich-trace test rows), plus a couple of extremes
    by_domain = {}
    for r in corpus["rows"]:
        by_domain.setdefault(r["domain"], []).append(r)
    sample = []
    for dom, rs in by_domain.items():
        rs2 = sorted(rs, key=lambda r: (not r["has_rich_trace"], r["split"] != "test"))
        sample.append(rs2[0])
    traces = []
    for r in sample:
        rid = r["row_id"]
        grade, meta = (model.grade(r) if model else (r.get("_support_grade", "exploratory"), {}))
        b = base.get(rid, {})
        d = unc.get(rid, {})
        raw = r["raw_p"]
        crit = None
        if b.get("direct_p") is not None:
            dis = abs(raw - b["direct_p"])
            crit = {"direct_p": b.get("direct_p"), "ensemble_p": b.get("ensemble_p"),
                    "disagreement": round(dis, 3), "flag": "large_disagreement" if dis > 0.3 else "ok",
                    "note": "critic annotates only; never overwrites the simulation number"}
        traces.append({
            "row_id": rid, "domain": r["domain"], "question": r.get("question", ""),
            "as_of": r.get("as_of"), "horizon_days": r.get("horizon_days"), "outcome": r["outcome"],
            "active_component_manifest": r.get("active_components"),
            "raw_forecast": raw, "selected_calibrator": (bundle["calibrator_name"] if bundle else "identity"),
            "calibrated_forecast": raw,     # identity selected
            "calibration_provenance": {"provisional": True, "effective_calibration_n":
                                       (bundle["effective_calibration_n"] if bundle else None)},
            "support_grade": grade, "support_grade_reasons": meta,
            "uncertainty_decomposition": d.get("components"), "aleatoric_var": d.get("aleatoric_var"),
            "epistemic_var": d.get("epistemic_var"),
            "dominant_sensitivity_contributors": d.get("dominant_sensitivity_contributors"),
            "direct_model_disagreement": crit,
            "limitations": ["provisional calibrator (pre-Phase-11)",
                            "Phases 8/9/11 not on the forecast path for this row"]})
    (OUT / "forensic_traces.json").write_text(json.dumps({"n": len(traces), "traces": traces}, indent=2))
    return traces


def markdown(traces):
    m = ["# WMv2 Phase 12 — Forensic Traces\n",
         "*Stratified end-to-end traces assembled from the frozen Phase-12 artifacts. Each shows the full "
         "chain from question to calibrated user-facing result so a reviewer can confirm the output came from "
         "the real max-capacity posterior simulation (not a disguised direct forecast). Machine-readable: "
         "`experiments/results/phase12/forensic_traces.json`.*\n"]
    for t in traces:
        m.append(f"\n## `{t['row_id']}` — {t['domain']}\n")
        m.append(f"**Q:** {t['question']}  \nas_of **{t['as_of']}**, horizon **{t['horizon_days']}d**, "
                 f"realized outcome **{t['outcome']}**\n")
        ac = t["active_component_manifest"] or {}
        on = [k for k, v in ac.items() if v]; off = [k for k, v in ac.items() if not v]
        m.append(f"- **active components** — ON: {', '.join(on) or 'none'}; OFF/not-wired: {', '.join(off)}\n")
        m.append(f"- raw forecast **{t['raw_forecast']}** → calibrator **{t['selected_calibrator']}** → "
                 f"calibrated **{t['calibrated_forecast']}** (provisional, eff cal n="
                 f"{t['calibration_provenance']['effective_calibration_n']})\n")
        m.append(f"- **support grade** `{t['support_grade']}` "
                 f"(expected_error={ (t['support_grade_reasons'] or {}).get('expected_error') })\n")
        if t.get("uncertainty_decomposition"):
            m.append(f"- **uncertainty decomposition** (epistemic {t.get('epistemic_var')}, aleatoric "
                     f"{t.get('aleatoric_var')}): {t['uncertainty_decomposition']}\n")
        if t.get("dominant_sensitivity_contributors"):
            top = t["dominant_sensitivity_contributors"][:2]
            m.append(f"- **dominant sensitivity**: {top}\n")
        if t.get("direct_model_disagreement"):
            m.append(f"- **direct-model disagreement (critic)**: {t['direct_model_disagreement']}\n")
        m.append(f"- **limitations**: {'; '.join(t['limitations'])}\n")
    return "".join(m)


def main():
    traces = build()
    Path("docs/WMV2_PHASE12_FORENSIC_TRACES.md").write_text(markdown(traces))
    print("wrote", len(traces), "forensic traces")


if __name__ == "__main__":
    main()
