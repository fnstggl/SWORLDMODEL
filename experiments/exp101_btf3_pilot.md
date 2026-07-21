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

---

## Addendum: the widened-kernel arms (provenance fix, same protocol)

**Fix** (`swm/api/mechanisms.py`): provenance tiers (grounded / quoted / invented) for
`aggregation`/`whipcount` — sd floors, count/lean estimate noise, no hard arithmetic gates on conjectured
counts, log-odds shrinkage toward the base-rate anchor; compile prompt now invites OMITTING invented
numbers (nulls land on the base-rate fallback, not on "zero votes"). Grounded direct calls unchanged
(pinned by tests, 9 passing).

| run (v4flash, no evidence) | brier | log-loss | acc@0.5 | AUC | extreme | vs const-base | vs SOTA |
|---|---|---|---|---|---|---|---|
| 50 pre-fix | 0.2634 | 1.137 | 0.66 | 0.497 | 0.48 | +0.071 worse | 0.0918 |
| 50 widened (paired) | 0.2008 | 0.737 | 0.76 | 0.632 | 0.28 | +0.008 worse | 0.0918 |
| **200 widened** | **0.2168** | 0.695 | 0.695 | **0.677** | 0.37 | **−0.008 better** | 0.1048 |

Paired 50: net brier delta **+0.063**; 18 improved / 16 worsened / 16 same. Gains are precisely the four
confabulated extremes (BoJ, Banxico, Slovakia, Farm Bill: 0.00/0.98 → 0.50, ~+0.7 each); losses are the
honesty tax (confident-and-right calls de-sharpened to 0.5, ~−0.25 each). Wrong-side extremes cost ~4x
what right-side extremes earn, so the trade is strongly net-positive exactly when the model cannot tell
which is which — the no-evidence regime. With grounded params the kernels keep full sharpness, which is
why this is a provenance tier and not a global clamp.

At n=200 the stack **beats every constant baseline for the first time** (0.2168 vs 0.2244/0.2245/0.25)
with real discrimination (AUC 0.677) — i.e., the no-evidence arm now contains honest signal, no longer
self-inflicted damage. Remaining gap to FutureSearch SOTA: **0.112 brier** (0.2168 vs 0.1048 on the same
187 questions), which decomposes as ~0.06 recovered by uncertainty discipline (this fix) and a remainder
that is evidence + research + ensembling — to be measured next by an evidence arm (BTF-2's inlined
research on an old-cutoff model, then RetroSearch-lite).

The 200-question by-mechanism table (arrival n=107 at 0.167 is the workhorse; escalation/persistence
small-n and weak; aggregation still worst at 0.294) is in `results/exp101_btf3_pilot_widened200.json`.

## Under the hood (traced, 5 questions) — and two wiring gaps found by tracing

Full-transparency traces (temperature 0, same calls as the run) show the per-question anatomy: ONE LLM
call (~5,000-char prompt = question + resolution criteria + truncated background + the 7-mechanism menu),
one JSON reply (a mechanism label + 1–5 scalars), then a ~4,000-draw Monte-Carlo integration of a
one-line kernel. No actors, institutions, populations, or state transitions run in this path — per the
EXP-089 mandate (deeper simulation machinery measurably hurt on open-domain questions).

Gaps found by tracing (to fix in the next paired run, NOT retro-fixed into reported results):
1. **Nested-params drop:** the LLM sometimes returns `{"params": {...}}`; the router reads only top-level
   keys, so those params vanish and the kernel lands on the base-rate fallback (BoJ, Banxico → 0.5).
   Honest by accident; the fix is to flatten `params` into the top level.
2. **`contest.win_prob` bypasses provenance:** an invented win_prob is returned verbatim (pre-fix Wale
   0.98 was the LLM's raw guess passing straight through). Needs the same anchor/widening treatment.
