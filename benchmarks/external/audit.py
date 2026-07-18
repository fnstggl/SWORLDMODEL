"""Machine-readable external-benchmark audit (verified 2026-07-18).

Writes benchmarks/external/benchmark_audit.json: one row per candidate benchmark with a fixed field
schema, so downstream tooling (and reviewers) can see at a glance what is usable NOW, what is
contaminated, and what blocks each option. Facts were verified against the public sources on the
audit date; re-verify before relying on registration windows or release status.
"""
from __future__ import annotations

import json
import os

AUDIT_FIELDS = [
    "benchmark", "official_source", "access_status", "license", "question_types", "resolved_or_live",
    "contamination_risk", "as_of_evidence_available", "human_baseline_available",
    "market_baseline_available", "supports_external_submission", "supports_full_production_runtime",
    "estimated_cost", "estimated_latency", "recommended_use", "blocking_issues",
]

AUDIT = [
    {
        "benchmark": "ForecastBench",
        "official_source": "https://www.forecastbench.org ; question sets + resolutions: "
                           "https://github.com/forecastingresearch/forecastbench-datasets ; code: "
                           "https://github.com/forecastingresearch/forecastbench (MIT)",
        "access_status": "open — question sets and resolution files are public (raw fetch verified "
                         "2026-07-18; latest round 2026-07-05, 500 questions); official tournament "
                         "entry requires email registration (out of scope for this run)",
        "license": "datasets CC BY-SA 4.0; code MIT",
        "question_types": "binary probability questions: prediction markets (Manifold, Metaculus, "
                          "Polymarket, RAND Forecasting Initiative/infer) + dataset questions "
                          "(FRED, ACLED, Wikipedia, DBnomics, yfinance) at multiple horizons",
        "resolved_or_live": "live biweekly rounds; per-round resolution files are updated in place as "
                            "outcomes resolve (2026-07-05 round: 213/428 rows resolved as of 2026-07-18)",
        "contamination_risk": "low for live rounds — outcomes lie in the future of any training cutoff; "
                              "non-zero only for a freeze made after the round due date (mitigated here "
                              "by excluding already-resolved questions and scoring only resolutions "
                              "strictly after predicted_at)",
        "as_of_evidence_available": True,
        "human_baseline_available": True,
        "market_baseline_available": True,
        "supports_external_submission": True,
        "supports_full_production_runtime": True,
        "estimated_cost": "LLM cost only (DeepSeek): roughly $0.01-0.05 per question through the "
                          "production runtime; datasets free",
        "estimated_latency": "1-5 min per question through the full production runtime; scoring "
                             "becomes valid at the first post-freeze horizon (days to weeks)",
        "recommended_use": "primary: sealed preregistered self-scored track now; official tournament "
                           "entry after email registration",
        "blocking_issues": ["official leaderboard uses difficulty-adjusted Brier — self-scored "
                            "unadjusted Brier is not directly comparable",
                            "tournament submission requires email registration (not done this run)",
                            "in this sandboxed environment the GitHub API listing endpoint is "
                            "session-restricted; the adapter falls back to probing raw file URLs "
                            "by date (raw.githubusercontent.com is reachable)"],
    },
    {
        "benchmark": "ForecastBench-Sim",
        "official_source": "https://github.com/forecastingresearch/forecastbench-sim ; "
                           "arXiv:2606.18686",
        "access_status": "open code (generate-it-yourself simulated forecasting worlds); no hosted "
                         "question service or leaderboard",
        "license": "GPL-3.0 (code)",
        "question_types": "synthetic binary forecasting questions in simulated worlds with known "
                          "generating processes",
        "resolved_or_live": "self-generated — resolution is computable at generation time",
        "contamination_risk": "none by construction (synthetic outcomes), but sim-to-real transfer "
                              "is unproven",
        "as_of_evidence_available": True,
        "human_baseline_available": False,
        "market_baseline_available": False,
        "supports_external_submission": False,
        "supports_full_production_runtime": True,
        "estimated_cost": "compute + LLM only; no data cost",
        "estimated_latency": "immediate scoring (outcomes known at generation)",
        "recommended_use": "internal ablation/regression harness only; not evidence of real-world "
                           "forecasting skill (sim-to-real unproven, no external baselines)",
        "blocking_issues": ["no public leaderboard or external baselines",
                            "sim-to-real validity unproven",
                            "GPL-3.0 code license requires care if code is vendored"],
    },
    {
        "benchmark": "FutureSearch BTF (Bench to the Future, BTF-2/BTF-3)",
        "official_source": "https://huggingface.co/datasets/BTF-2/BTF-2 ; futuresearch.ai",
        "access_status": "BTF-2 downloadable from Hugging Face; BTF-3 (1907 questions) advertised but "
                         "not yet released; frozen full corpora not public",
        "license": "CC BY-NC 4.0 (BTF-2)",
        "question_types": "1417 resolved binary forecasting questions with archived as-of evidence "
                          "(pre-resolution web snapshots)",
        "resolved_or_live": "resolved — all BTF-2 questions resolved by Dec 2025",
        "contamination_risk": "HIGH: outcomes resolved Dec 2025 are inside the training window of any "
                              "model trained past ~Oct 2025 — the production LLM (DeepSeek, 2026) is "
                              "contaminated for accuracy claims",
        "as_of_evidence_available": True,
        "human_baseline_available": False,
        "market_baseline_available": True,
        "supports_external_submission": False,
        "supports_full_production_runtime": True,
        "estimated_cost": "LLM cost only; dataset free (non-commercial license)",
        "estimated_latency": "immediate scoring (already resolved)",
        "recommended_use": "diagnostic only (pipeline plumbing, failure-mode analysis) — NOT accuracy "
                           "claims, because the production LLM's training data postdates the outcomes",
        "blocking_issues": ["contaminated for any model with training past ~Oct 2025",
                            "CC BY-NC 4.0 restricts commercial use",
                            "BTF-3 not yet released; frozen full corpora not public"],
    },
    {
        "benchmark": "Metaculus (API / bot tournaments)",
        "official_source": "https://www.metaculus.com ; API: https://www.metaculus.com/api/ ; "
                           "FutureEval seasonal bot tournaments",
        "access_status": "API requires an account token; bot tournaments (FutureEval) are seasonal — "
                         "Summer 2026 season already underway as of the audit date",
        "license": "proprietary; Terms of Use restrict use of API data for AI evaluation without "
                   "permission",
        "question_types": "binary, numeric, multiple-choice, date questions; live community "
                          "predictions",
        "resolved_or_live": "live questions plus a large resolved archive",
        "contamination_risk": "low for live questions; high for the resolved archive",
        "as_of_evidence_available": False,
        "human_baseline_available": True,
        "market_baseline_available": True,
        "supports_external_submission": True,
        "supports_full_production_runtime": True,
        "estimated_cost": "free API within rate limits + LLM cost",
        "estimated_latency": "seasonal tournament cadence; question resolution weeks to months",
        "recommended_use": "not usable this run; revisit with an account token, ToU permission for "
                           "AI-eval use, and the next FutureEval season entry window",
        "blocking_issues": ["API requires account token (none provisioned this run)",
                            "ToU restricts AI-evaluation use of API data without permission",
                            "Summer 2026 tournament season already underway — mid-season entry "
                            "not comparable"],
    },
]

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_audit.json")


def write_audit(path: str = DEFAULT_PATH) -> dict:
    for row in AUDIT:   # schema guard: exactly the required fields, all present
        missing = [f for f in AUDIT_FIELDS if f not in row]
        extra = [k for k in row if k not in AUDIT_FIELDS]
        if missing or extra:
            raise ValueError(f"audit row {row.get('benchmark')}: missing={missing} extra={extra}")
    doc = {"audit_date": "2026-07-18", "fields": AUDIT_FIELDS, "benchmarks": AUDIT}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    return doc


if __name__ == "__main__":
    d = write_audit()
    print(f"wrote {DEFAULT_PATH} with {len(d['benchmarks'])} benchmarks")
