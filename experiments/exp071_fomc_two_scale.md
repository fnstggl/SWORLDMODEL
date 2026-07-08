# EXP-071 — environment → individuals → institution (FOMC), on real data: does the middle scale earn its place?

The coupling the World substrate was built for, on the case where cross-scale feedback is real and
scoreable: the **FOMC**. The macro ENVIRONMENT (inflation, unemployment) drives the INDIVIDUALS (members'
desired policy), who vote to produce the INSTITUTION's decision (the rate move). Real monthly data 1985–2026
from FRED (FEDFUNDS, CPI, UNRATE), leakage-free — each month's decision predicted from macro known that
month, scored on a held-out 40%.

Held to the EXP-070 discipline: **a coupling must beat its ablation.** The honest, non-trivial question:
does inserting the **middle scale** — heterogeneous voting members with an inaction zone — beat going
straight from the environment to the decision?

| model | what it is | MAE | direction acc |
|---|---|---|---|
| **STATIC** (coupling cut) | members frozen → always hold | 0.185 | 0.641 |
| **momentum** | next move = last move (pure inertia) | **0.071** | **0.919** |
| **DIRECT** (env → decision) | representative partial-adjustment Taylor rule | 0.177 | **0.707** |
| **TWO-SCALE** (env → members → committee) | heterogeneous voting members + inaction zone | 0.186 | 0.641 |

## What the numbers say — three honest findings

1. **The environment→decision coupling EARNS its place.** The macro Taylor pressure lifts direction accuracy
   from the always-hold floor **0.641 → 0.707**. Knowing the economy genuinely tells you *which way* the Fed
   will move. The substrate's own ablation (coupling cut → static hold) confirms the environment edge is
   load-bearing.

2. **The middle MEMBER scale does NOT earn its place — it degrades the signal.** Routing the graded macro
   pressure through discrete/saturating voting members collapses direction back to the static floor
   (**0.707 → 0.641**). The member layer throws away the magnitude information the direct rule keeps; the
   extra structure *hurts*. Same call as SCOTUS (EXP-070): tested, didn't beat the simpler model, so we do
   **not** scale it up here.

3. **The dominant real signal is policy INERTIA.** A momentum baseline (next move = last move) crushes every
   macro model (MAE **0.071** vs 0.177; direction **0.92**). The Fed moves in *runs*, and that persistence —
   not the contemporaneous macro pressure — is the strongest single predictor of the near-term move. Any
   serious Fed model is momentum-first; the macro pressure adds directional signal at turning points, and the
   member scale adds nothing on top.

## Why this is the right result, not a disappointing one

The brief asked to build this coupling **and prove whether it beats separate — before scaling up.** We did:
- Built the environment→individuals→institution world in the substrate on real FRED data.
- The environment coupling passed its ablation (direction 0.64 → 0.71).
- The individual/member scale **failed** its test (0.71 → 0.64) — so the honest, disciplined move is to keep
  the environment coupling and **drop the member scale for FOMC**, exactly as the substrate's scoreboard is
  designed to enforce.

This is the digital-twin discipline working as intended: **you wire a scale in only if it beats its
ablation on real data.** Across the two cases we've now scored (SCOTUS EXP-070, FOMC EXP-071), the honest
verdict is the same and important: the coupling *machinery* is real and captures genuine cross-scale feedback
(the bank-run cascade), but the specific individual↔institution couplings we can score do **not** beat
simpler models — the intermediate human scale mostly reproduces (or degrades) what a direct
environment→outcome rule and a momentum term already capture. Where a shared world would win is a case with
strong endogenous cross-scale feedback and weak simple baselines; FOMC and SCOTUS have strong simple
baselines (inertia, static ideology), so they don't clear the bar. That is a real, measured finding about
*when* to reach for the world substrate — which is worth more than a forced win.

## Tests — `tests/test_fomc_two_scale.py` (3, all pass)

The environment entity drives the members when coupled (guards the fix where a step-less environment entity
silently dropped its external input); coupling-cut collapses to hold; scoring works. Full suite green.

## Data

`experiments/results/exp071/fomc_macro.json` — 494 months (1985–2026), real FRED FEDFUNDS/CPI/UNRATE with
the derived as-of inflation and the realized next-3-month policy move. Rerun offline from the committed panel.
