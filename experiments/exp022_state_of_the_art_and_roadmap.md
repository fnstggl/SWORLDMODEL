# EXP-022 — How accurate is this really, vs. the best social simulation out there? + roadmap

An honest assessment: what our numbers actually are, where the field's best work stands, whether we
match it, and — mapped concretely — what it would take to get there (with the first steps implemented).

## Where we are, measured (no-cheat backtests)
| regime | outcome | result | vs baseline |
|---|---|---|---|
| Individual response | GitHub issue → maintainer responds (6h) | log loss 0.132 on deep-history repos; **beats a raw LLM given the same context** (0.258 vs 0.304) | halves the segment baseline; scales with entity depth |
| Individual response | Enron email → recipient replies | log loss 0.275, calibrated; scales with recipient depth | beats segment/global |
| **Persuasion** | CMV argument → changes the OP's view (delta) | **LLM-inferred variables cut log loss to 0.592 vs 0.635** surface/no-inference (+0.043, 6.8% rel; uplift@20 2.2×) (EXP-021) | inferred latent variables earn their place where there is no entity state |
| Aggregate | HN post → front-page hit | grade-A calibration (ECE 0.019); hybrid beats raw LLM+context overall | small edge (aleatoric ceiling) |
| Market | Manifold event @ fair horizon | **loses** (Brier 0.260 vs market 0.178) | blocked on as-of data |

So: **well-calibrated, no-cheat, real-outcome prediction that beats strong baselines and a raw LLM
where entity state or inferable latent variables carry the signal** — and honest losses where the
constraint is information access (markets) or aleatoric noise (HN).

## What the best work out there is (grounded)
- **Generative Agent Simulations of 1,000 People** (Park, Bernstein, et al., 2024) — the current
  high-water mark for *individual* behavioral simulation. They interviewed 1,000 real people for ~2
  hours each, built an LLM agent per person from the transcript, and predicted each person's answers
  to the **General Social Survey**, Big Five, and economic games. Headline: ~**85% normalized
  accuracy** (normalized to human test–retest reliability ~0.81). Their power comes from **rich
  per-person data** (the interview) + an LLM conditioned on it.
- **Generative Agents / Smallville** (Park et al., 2023) — believable emergent agent behavior;
  evaluated on believability, not calibrated prediction.
- **OpinionQA / "Whose Opinions Do LLMs Reflect?"** (Santurkar et al., 2023) — the standard benchmark
  for opinion-distribution alignment (Pew ATP surveys). Measures how well a model matches US
  demographic groups' opinion distributions; shows LLMs are systematically misaligned.
- **Synthetic-respondent startups** (Aaru, Artificial Societies, etc.) — claim poll/market/consumer
  prediction; validation is largely proprietary/opaque.

## Are we at that level? Honestly: no — narrower and smaller, but more rigorous, and architecturally aligned
- **Breadth/scale:** the 1,000-person work validated across many behavioral instruments with rich
  interview data. We have validated on response/reply + persuasion on public data, with **inferred**
  (not interviewed) variables. We are **not** at their breadth or scale.
- **Rigor:** we are arguably **more rigorous on what they under-report** — calibrated probabilistic
  prediction (log loss / ECE), strict as-of no-leakage backtests, and an **explicit,
  provenance-tracked, auditable variable map** rather than an opaque interview transcript.
- **Architecture:** the key insight — our **VariableMap** is an *automated, structured* version of
  their **interview**. They elicit a person's variables by talking to them for two hours; we infer
  the same behavioral variables from available context (message, history, platform) with provenance
  and confidence. Their approach cannot scale (you can't interview everyone); ours can, and EXP-021
  shows the inferred variables genuinely predict a hard human outcome (persuasion). This is the right
  bet for a *general* social world model.

## What it would take to match/exceed the best — mapped, with first steps implemented
1. **Rich per-person inference (close the "interview" gap).** Their edge is 2h of interview per
   person. Our analog: ingest a person's available writing/history and run a *deep* variable
   inference over it, not a one-shot title read. **Implemented foundation:** the inference engine
   accepts arbitrarily rich context (`llm_infer_fn`) and EXP-021 shows deep inference from full OP+
   argument text works. **Next:** a multi-pass inference that reads a person's history corpus.
2. **A standard survey benchmark (claim comparability).** Backtest VariableMap-conditioned prediction
   on a public opinion/behavior benchmark (OpinionQA / GSS-style) to be directly comparable to the
   SOTA on their turf. **Status: blocked here on data access** — OpinionQA's data is HF-gated (401)
   and codalab TLS-fails in this environment; GSS/WVS need a download. This is the single most
   important step to *claim* parity, and it is a data-access task, not a modeling one.
3. **Breadth of validated outcomes.** We now span response (GitHub/Enron) and persuasion (CMV). Add
   choice/purchase/vote/attitude outcomes as timestamped datasets become reachable.
4. **Aggregate/population validation** (elections, polls, markets) — blocked on as-of feeds (EXP-017).
5. **Scale the variable schema + confidence modeling** — richer disposition/values variables; better
   uncertainty propagation. Calibration/incremental, not architectural.

## The honest bottom line
The **core architecture is right and now empirically supported**: mapping known + inferred variables
into one provenance-tracked state, then predicting from it, matches our best models on response
outcomes at ~zero cost AND wins on persuasion where inference is the only signal (EXP-021). We are
**not yet at the scale or breadth of the best published social simulation** (the 1,000-person work),
and the gap is **data**: rich per-person corpora and a standard survey benchmark — both largely
access-blocked in this environment, both squarely on the roadmap. What separates us positively is
rigor (calibration + no-cheat backtests) and an auditable variable map that scales where interviews
cannot. Implemented this round: the persuasion validation (EXP-021) and the full variable-mapping
core (EXP-020); blocked-and-mapped: the survey benchmark and rich per-person ingestion.
