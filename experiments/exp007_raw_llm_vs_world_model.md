# EXP-007 — Is this genuinely more than raw LLM + context?

The uncomfortable audit. Same no-cheat temporal HN split (train = March 2026, n=129; test =
round 4, May 2026, n=114 posts — all after the Jan 2026 model cutoff, so the LLM forecasts are
contamination-free). Five predictors, identical test posts, identical metrics. No spin: where the
state model loses, it is reported as a loss.

`experiments/raw_llm_vs_world_model.py` — `packets` writes title+timing-only batches for a blind
swarm; `score --titleonly` runs the head-to-head. The five predictors:

1. **raw LLM, title only** — a blind swarm (4 Claude agents, no web search) forecasting from the
   title and submit-time ONLY. No author, no domain, no history.
2. **raw LLM + retrieved context** — the committed round-4 forecasts, made WITH author priors and
   domain (the "real" product path from EXP-002/006).
3. **statistical, no LLM** — logistic over content + entity factors, fit on March, no LLM anywhere.
4. **calibrated system** — #2 passed through the fitted Platt layer (`data/calibration.json`).
5. **state-transition model** — #3 plus the *transitioned* context state (domain_reputation,
   topic_salience evolving after every post), the full `swm/state` machinery.

## Results (blind, no leakage)

```
-- P(score >= 10) | base rate 0.123 --
   method                      logloss   brier    ece  uplift@20
   1 raw LLM title-only         0.3181  0.0915 0.0717     0.0946   <- best logloss & brier
   2 raw LLM + context          0.3702  0.1093 0.0756     0.0511
   3 statistical (no LLM)       0.4019  0.1203 0.0747    -0.0793
   4 calibrated (=#2+Platt)     0.3593  0.1061 0.0539     0.0511   <- best ECE
   5 state-transition           0.4077  0.1239 0.0774    -0.0359

-- P(score >= 40) | base rate 0.053 --
   method                      logloss   brier    ece  uplift@20
   1 raw LLM title-only         0.1854  0.0486 0.0156     0.0778   <- best
   2 raw LLM + context          0.1922  0.0487 0.0191     0.0778
   3 statistical (no LLM)       0.2354  0.0561 0.0481     0.0343
   4 calibrated (=#2+Platt)     0.1922  0.0487 0.0191     0.0778
   5 state-transition           0.2453  0.0614 0.0836     0.0343
```

Decision lift (uplift@20 at ≥40) vs raw LLM + context: title-only, context, and calibrated all tie
at **+0.0778**; the two non-LLM branches (statistical, state) are **worse** at +0.0343.

## The honest answers to the questions asked

**Is this actually proprietary beyond raw LLM + context? On this HN benchmark, no.** The two
non-LLM branches (#3 statistical, #5 state-transition) are the *worst* two predictors on every
metric at both thresholds. The signal that separates a good HN forecast from the base rate is
overwhelmingly the LLM reading the title — not the state machinery. At ~130 training examples a
from-scratch logistic cannot out-resolve a frontier model's prior over what HN upvotes.

**Did explicit state improve prediction here? No — it made the no-LLM baseline slightly worse.**
State-transition (#5) is worse than the plain statistical model (#3) on this test (logloss 0.2453
vs 0.2354 at ≥40). Adding the transitioned context factors to the tiny March-trained logistic added
variance, not signal, on the round-4 window. This *contradicts* the EXP-005 result on the larger
2,810-post single-domain sequence, where state helped by −1.4% logloss. The difference is sample
size and author overlap: EXP-005 had deep repeat-author sequences within one world; round 4 is a
fresh window with mostly new/low-history authors, exactly where an evolving latent state has nothing
to evolve from.

**Did transitions improve prediction, or only static features?** Neither beat the LLM. *Within* the
non-LLM family, ablation still says the stateful factors earn their keep (below) — but the family
they belong to is dominated by the LLM, so "transitions vs static" is a contest for second-to-last
place on this benchmark.

**Which variables survived ablation** (KEEP = removing it worsens held-out logloss at ≥40):

```
   domain_reputation    +0.0047  KEEP   (stateful — transitions after every post)
   is_show              +0.0043  KEEP
   author_standing      +0.0017  KEEP   (stateful)
   is_weekend           +0.0017  KEEP
   author_ceiling       +0.0009  KEEP   (stateful)
   author_volume        +0.0004  KEEP   (stateful)
   is_ask               -0.0005  EXPERIMENTAL
   author_quality       -0.0007  EXPERIMENTAL  (redundant with standing/ceiling)
```

The stateful factors (domain_reputation, author_standing/ceiling/volume) are consistently KEEPs and
`author_quality` is consistently flagged redundant — the ablation filter works and its verdicts are
stable across EXP-005 and EXP-007. But surviving ablation *within the no-LLM branch* is not the same
as beating the LLM, and it doesn't.

**The most surprising honest finding: title-only beat context.** Raw LLM with *only the title* beat
raw LLM *with* retrieved author/domain context on both thresholds (logloss 0.3181 vs 0.3702 at ≥10).
The retrieved context did not help — it slightly hurt — because most round-4 authors are new or
low-signal, and feeding thin priors into the forecast nudged some predictions the wrong way while
adding confidence. This is a real, uncomfortable result: on this window, *more context made the LLM
worse*. The one thing that reliably helped was the **calibration layer** (#4 has the best ECE,
0.0539 vs 0.0756), which is a genuine, cheap, keep-able piece of proprietary value.

**Where does the system still collapse into LLM guessing?** Everywhere the LLM wins, which here is
everywhere. The "world model" contributes measurable value only in EXP-005's deep single-domain
repeat-author regime and via the calibration layer. On a fresh cross-author window it does not beat
the LLM reading a title.

## Is /v1/rollout using state transitions or just calling the LLM?

**It uses state transitions, not the LLM.** `api/app.py::rollout_ep` builds a `TransitionModel(reg,
PriorHead())` and calls `swm.state.trajectory.rollout` — the outcome distribution comes from the
statistical `PriorHead`/`OutcomeHead` and the state evolves via deterministic factor update rules
(`registry.apply_update`). No LLM is invoked in the rollout path. Because it uses the uncalibrated
`PriorHead` (no fitted, backtested head for arbitrary domains), the honesty gate labels every
rollout `report_type: "simulation"`, `calibration_grade: "unvalidated"` except HN horizon 1.

## Exactly where LLMs are used (verified by grep)

- `swm/llm.py` — trait extraction from message text and draft generation. Feature/prior only.
- `swm/entities/persona.py` — one comment noting the lexical formality proxy is "replaceable by the
  LLM extractor." No live call in the scored path.
- The forecast **experiments** (EXP-002/003/006 round predictions, and the EXP-007 blind swarm) —
  the LLM *is* predictors #1 and #2.
- **NOT** in `swm/state/` or `swm/transition/`. The state-transition statistical path is 100%
  LLM-free (the only `llm` string in `swm/state/factors.py` is the keyword in the topic tagger).

So the architecture cleanly separates "LLM as feature/prior extractor" from "statistical
state-transition as the probability source." That separation is real and is the honest design. What
this experiment shows is that on HN, at this scale, the LLM branch carries the prediction and the
statistical branch does not add to it.

## One-step vs multi-step rollout degradation

Unchanged from EXP-005 and still the weakest claim: one-step HN transition is validated (grade C);
multi-step degradation is not cleanly measurable at this per-author depth (horizon logloss is
dominated by the shifting realized hit-rate, not model error). The API reflects this — `/rollout`
validates only HN horizon 1 and labels everything else unvalidated.

## What needs to be built next (honest priorities)

1. **Keep the calibration layer — it's the one component that beat the LLM here (best ECE).** Ship
   calibrated LLM forecasts as the product; do not ship the from-scratch statistical head as the
   headline predictor on sparse cross-author windows.
2. **Stop feeding thin context to the LLM.** Title-only beat title+context. Retrieval must be gated
   on *sufficiency* (enough author history to move the prior) or it degrades the forecast. This is
   the direct motivation for EXP-008: as-of retrieval that only helps when it has real signal.
3. **The state machinery's value is regime-specific** — deep, repeat-actor, single-world sequences
   (EXP-005), not fresh cross-author windows (EXP-007). Target the next backtests there (individual
   email threads with the same recipients over time), where state actually evolves.
4. **The honest positioning holds:** the moat is not "we beat the LLM at reading a title." It is
   (a) calibration, (b) counterfactual/no-market decision questions where no LLM baseline is
   well-defined, and (c) repeat-entity state where history compounds. This experiment rules out the
   claim that the state-transition model beats raw LLM + context on open HN prediction. Reported as
   such.
