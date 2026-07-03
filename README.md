# SWORLDMODEL — a social world model, built the honest way

A **social world model** predicts the *distribution* of human/social responses to a proposed
action (a message, product, policy, or event), under **partial observation**, with **calibrated
uncertainty**, and — crucially — **backtested against real, time-forward outcomes.**

This repository starts from a research audit (**[`docs/social-world-model-audit.md`](docs/social-world-model-audit.md)**)
and scaffolds the build it recommends. The audit is deliberately anti-hype: it separates what is
**established**, what is **speculative**, and what would require **original research**, and it argues
for starting with a narrow, paid, backtestable wedge instead of "ChatGPT for simulating the future."

## The thesis in five lines
1. **Believable ≠ accurate.** The category's failure mode is confident, well-narrated, *uncalibrated* output. So the **evaluator is the product**: build the harness that can embarrass the model *before* the model.
2. **Start narrow.** First wedge: **outbound-message response prediction & optimization** (B2B email reply, then marketing copy) — clean outcomes, timestamps, a baseline to beat, a paying buyer.
3. **Partial + probabilistic.** Every output is a distribution with an interval and a calibration grade. No prophecy.
4. **The moat is data + evaluation, not the model.** No public dataset pairs readable message content with observed outcomes; proprietary content→outcome logs are the scarce asset. Own the public, contamination-controlled benchmark too.
5. **Earn generality.** Generality is stacked calibrated wedges, each with its own backtest — never asserted up front.

## Repo layout (see Section I of the audit)
```
swm/         core library — modules map 1:1 to the architecture (Section C)
  eval/      FIRST-CLASS: harness, metrics, baselines, leakage gate  (built before the model)
api/         FastAPI service (Section J): /predict /compare-actions /simulate /backtest ...
benchmarks/  public-data harnesses (Upworthy, Criteo) for credibility results
experiments/ dated, reproducible result scripts
tests/       incl. the leakage gate that must pass in CI
```
`swm/eval/metrics.py` and `swm/ingestion/schema.py` are implemented; the rest are honest stubs to
be filled in build-order: `schema → store → actions.encoder → transition.readout →
uncertainty.calibration → eval.* → benchmarks/upworthy`.

## Status
Research + scaffold. Not a product. The hardest claims (does LLM-agent rollout add *calibrated* lift
over a boring gradient-boosted readout?) are **open research problems**, tracked in the audit (Section H).

See the audit for the full literature map, data-acquisition and evaluation plans, wedge specs,
30/90/365-day roadmap, API spec, competitive analysis (Aaru, Simile, and the synthetic-respondent
field), moats, and a brutal critique of why this probably fails and how to de-risk it.
