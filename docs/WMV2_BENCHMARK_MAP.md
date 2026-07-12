# WMv2 benchmark map — existing assets, honest status, staged execution plan

*Audit of every benchmark asset in the repo (Part 1B). Nothing here is duplicated; V2 evaluation adapts these
harnesses without changing evaluation populations. All prior results and negative findings preserved (see
`ARCHITECTURE_AGENT_ENGINE.md`, `docs/AUDIT_BEHAVIOR_MODELS.md`).*

| Benchmark | Entry points | Data | n / labels | Leak protection | Prior result | Status for V2 |
|---|---|---|---|---|---|---|
| **ForecastBench** rounds | `swm/eval/forecastbench.py`, `exp098/exp100` | remote JSON, auto-fetch | 13 rounds used; ~87 deliberation binary + resolutions | as-of due date + bounded before/after retrieval | pooled n=87: whole-stack vs grounded-1shot Δ=−0.0009 (p=0.98) | **runnable now** (compatibility + supported-subset modes) |
| **Manifold/Polymarket crowd** | `swm/eval/grade_vs_crowd.py`, `crowd_sets.py`, `exp092` | main's `forecasting_corpus` (checked in) | n≈46-127 cleaned; crowd price at as-of | `cutoff_clean` + as-of search | dir 0.72, recal skill −0.06, unsure-slice +0.049 | **runnable now** |
| **Upworthy** | `swm/eval/response_datasets.py`, `experiments/behavior_pilot/upworthy_eval.py` | OSF CSV, loader auto-downloads (14MB) | 32k tests; randomized CTR winners | randomized (causal by design) | DeepSeek p@1 0.56 (rand 0.34); OSim 0.35 | **runnable after data download**; V2 audience-world arms need population mechanisms → *architecture-validity mode first* |
| **BehaviorBench** | `swm/eval/behavior_eval.py`, `behaviorbench_eval.py` | HF resolve URLs (public) | 9 games × ~200 human choices | n/a (static human data) | matched: OSim W1 0.191 < DeepSeek 0.228; responder failure | **runnable now** (distributional fidelity of V2 agents) |
| **OmniBehavior** | `swm/eval/omnibehavior_eval.py`, `omnibehavior_run.py` | HF (released 2026-05); smallest-EN-users slice cached | first slice n=72, real_rate 0.54 (**BIASED SLICE** — last-8-events sampling oversamples engagement) | time-ordered prefixes | DeepSeek acc 0.50 ≈ chance — **NOT a valid grade until sampling repaired** | **harness needs repair** (uniform chronological event sampling + per-user base-rate baselines) before any V2 claim |
| **Enron reply+delay** | `swm/eval/response_datasets.py::load_enron_reply_delay`, `time_forward_split` | needs ~1.7GB maildir download | reply-occurrence + delay reconstructable from headers | time-forward split (tested) | none — never run | **runnable after data download** — the designated Reference World A |
| **Higgs/SEISMIC diffusion** | none yet (registry entries only) | SNAP (open) | cascades + follower graph | early-observation cutoffs needed | none | **runnable only after harness build** (cascade reconstruction + labels) |
| **Forward ledger** | `swm/engine/forward_ledger.py`, `experiments/flywheel_forward.py` | git-committed JSONL | append-only, versioned | contamination-proof by construction | wired, low n | **runnable now** — V2 runs must lock `{B0, grounded_1shot, V2}` + plan hash + mechanism versions |

## Execution order (per the directive; adjusted only by data blockers)

1. **Enron Reference World A** — the decisive development benchmark (needs maildir download; loaders + splits
   exist). Arms I0–I8 as specified; time-forward + person-disjoint splits; reply@1/3/7/14d + delay.
2. **Upworthy** — audience world via the universal compiler (no separate headline engine); full arm ladder.
3. **ForecastBench** — compatibility audit on every leak-free round (compiled? abstained? why? which
   mechanisms? terminal-state answer?) + supported-subset comparison vs all v1 baselines, coverage reported.
4. **BehaviorBench** — V2 independent vs interacting vs persistent agents vs the OSim/DeepSeek results.
5. **OmniBehavior** — repair sampling first; then base-rate/user-history/direct/V2±persistence ladder.
6. **Higgs/SEISMIC** — build the cascade harness; graph and agent mechanisms independently ablated.
7. **Manifold/Polymarket expanded** — V2 on validation-passing questions only, same evidence snapshots.

**Standing rule:** a benchmark adapter may not bypass V2 (no direct-LLM answer copied into a terminal state);
every V2 prediction must carry an execution trace (plan compiled → world materialized → mechanisms executed →
deltas → clock advanced → terminal projection). Status labels: interface-only → architecture-validated →
no-evidence → preliminary-positive → replicated → statistically-supported → production-validated.

**Current V2 status: architecture-validated on all software invariants; NO predictive evidence yet.**
