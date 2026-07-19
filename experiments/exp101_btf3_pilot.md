# EXP-101 — BTF-3 pastcasting pilot: first run on FutureSearch's public benchmark (n=50, no evidence)

**Benchmark.** [BTF-2/BTF-3](https://huggingface.co/datasets/BTF-2/BTF-3) (Hugging Face, CC-BY-NC-4.0,
raw copy kept local-only in `data/btf3/`, gitignored): 1,515 resolved binary pastcasting questions,
present dates 2026-04-20 → late May, resolutions **exclusively May–July 2026** (audited: zero questions
resolve before 2026-04). Each row ships FutureSearch's own SOTA forecast, which becomes a per-question
reference baseline. Sample: 50 questions, `random.Random(42)`, ids committed in
`results/exp101_btf3_sample_ids.json`.

**Protocol.** No-evidence arm (deliberately retrieval-free — same posture as the clean 660-question
Manifold backtest): `mechanism_forecast` (compile → route to mechanism simulator → simulate, EXP-095
stack), as-of = `present_date`, horizon = `expected_resolution_date`. Leakage is enforced in code, not by
care: the forecaster receives an **allowlist** of fields (question / criteria / background — all authored
as-of present_date by the benchmark itself); `resolution`, `resolution_explanation` and `sota_*` join at
scoring time only, behind an assert.

**Cutoff forensics (the BTF-1 lesson, replayed on us).**
- DeepSeek-V4-Flash *self-reports* "May 2025"; its **official cutoff is April 2026**. Self-reports are
  worthless — treat vendor statements + empirical probes as the only evidence.
- For BTF-3 this is still clean-by-construction: all outcomes post-date April 2026, and April-2026
  crystallized knowledge ≈ the *intended* information state (present dates are late April/May 2026).
- **BTF-2 is a different story: its questions resolve Oct–Dec 2025 — inside V4's training window. BTF-2
  must only ever be run with old-cutoff models** (e.g. `deepseek/deepseek-chat-v3-0324`, cutoff
  ~mid-2024, now wired via `swm/api/openrouter_backend.py`). A stray BTF-2 question ("U.S. v. Comey gag
  order, Oct 15–Dec 31 2025") surfaced during review and was confirmed absent from BTF-3's binary config.
- Belt-and-suspenders: the second arm re-runs the identical 50 questions on V3-0324 (clean for
  everything). If V4's recency were leaking outcomes, V4 would show inflated discrimination vs V3.

## Results (identical 50 questions, paired)

| forecaster | brier | log-loss | acc@0.5 | AUC | extreme (p<.1 or >.9) |
|---|---|---|---|---|---|
| WMv2 mechanism + **V4-Flash** (Apr-2026 cutoff) | 0.2634 | 1.137 | 0.66 | **0.497** | 0.48 |
| WMv2 mechanism + **V3-0324** (mid-2024 cutoff) | 0.3367 | 1.450 | 0.62 | **0.508** | 0.36 |
| **FutureSearch SOTA** (research + ensemble; n=45 with forecasts) | **0.0918** | — | 0.889 | 0.918 | — |
| const p=0.5 | 0.2500 | — | — | — | — |
| const p=sample base rate (0.26) | 0.1924 | — | — | — | — |
| const p=0.33 (BTF-3 global prior) | 0.1973 | — | — | — | — |

By mechanism (brier, n): V4 arm — arrival (25) 0.121, diffusion (2) 0.114, contest (5) 0.262,
escalation (2) 0.210, whipcount (8) 0.406, aggregation (7) **0.570**, persistence (1) 0.945.
V3 arm — arrival (20) 0.121, whipcount (13) **0.743**, aggregation (5) 0.513.

## Findings

1. **Both arms are at chance discrimination (AUC ≈ 0.50) and below every constant baseline.** On
   evidence-hungry questions with zero retrieval, the mechanism stack cannot rank YES above NO. This is
   the EXP-089 result reproduced on an external benchmark — no surprise, and no leakage: chance-level
   performance is itself the No-Evidence contamination probe coming back clean (leakage inflates, it
   never nullifies).
2. **The damage is concentrated: 24/50 extreme predictions, 8 on the wrong side (~1.0 brier each).**
   They decompose into exactly two failure families:
   - **Evidence deficit** (unwinnable without news): Strait-of-Hormuz suspension — we said 0.03 on the
     historical base rate, SOTA said 0.94 from May-2026 reporting, resolved YES. BoJ June hike (we 0.00,
     SOTA 0.74, YES). These measure the value of retrieval, not a model defect.
   - **Confident confabulation** (the EXP-089 disease, back in the mechanism sims): `aggregation` and
     `whipcount` invent vote shares / committed-vote counts with zero declared uncertainty, and the
     threshold kernel converts invented point estimates into 0.00/0.98. The honest-ignorance posture the
     latent sim earned in EXP-091 was never inherited by these two mechanism kernels.
   - Plus one pure reasoning miss: visionOS-at-WWDC (annual cadence ⇒ base rate ≥0.9; we said 0.02).
3. **Model recency bought little: V4 beats V3 by 0.073 brier with both at chance AUC.** Two extra years
   of crystallized knowledge (V3 mid-2024 → V4 Apr-2026, i.e. knowledge up to the question's as-of date)
   is worth far less on this benchmark than the two fixable failure families above. The binding
   constraint is evidence + honest uncertainty, not the LLM's vintage — consistent with EXP-092's
   "grounding is the lever" verdict.
4. **The gap to FutureSearch SOTA is 0.263 → 0.092 brier.** Their published per-question forecasts make
   BTF-3 a permanent regression target: every future arm (uncertainty-widened kernels, retrieval, inner
   crowd, multi-model panel) can be scored against the same 45 reference forecasts on the same ids.

## Mandated next steps

1. **Inherit honest ignorance in `aggregation`/`whipcount`**: when shares/counts are LLM-invented
   (no grounded poll/whip evidence), widen parameter uncertainty so the kernel cannot emit p<0.05 or
   p>0.95 — the EXP-091 fix, applied to the two kernels that skipped it. Re-run this exact 50 to measure.
2. **Retrieval arm** (RetroSearch-lite or BTF-2's inlined research on a BTF-2-safe model): quantify the
   evidence-deficit share of the gap directly.
3. **Scale to n=200+** before drawing conclusions beyond direction — at n=50 the brier CI is wide.
4. **BTF-2 runs: old-cutoff models only** (V3-0324 via `openrouter_backend`); BTF-3 remains safe for
   current frontier models until vendors' cutoffs pass ~April 2026.

Artifacts: `results/exp101_btf3_pilot{,_v3or}.json`, `results/exp101_btf3_predictions{,_v3or}.json`,
sample ids, this report. Cost: ~100 LLM calls (~$0.10 total). Keys read from env only, never stored.
