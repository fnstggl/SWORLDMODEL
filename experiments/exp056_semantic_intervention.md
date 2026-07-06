# EXP-056 — The semantic interventional model: an LLM picks the causally-better action

EXP-054 stood up the interventional KPI (choosing a headline is a real `do(x)` on randomized A/B data) and
found lexical features capture only **9.5%** of achievable uplift and rank arms at **chance** — the frontier
is semantic, exactly like EXP-044/047. This builds the fix: an LLM judge
(`swm/api/intervention_selector.py`, the same pluggable-backend pattern as `semantic_stance`) reads the
candidate headlines and picks the one it expects to win, **blind to the realized CTRs**, scored on the same
causal scoreboard.

## Setup
45 held-out A/B tests with **distinct** headlines (a real text choice, not image variants). The LLM picks
are committed; the lexical model (EXP-054) is re-run on the same sample as the baseline to beat.

## Result

| model | picked-arm CTR | uplift over random | **% of achievable uplift captured** |
|---|---|---|---|
| **semantic (LLM)** | 0.01872 | +0.199 pp | **36.5%** |
| lexical (EXP-054) | 0.01749 | +0.076 pp | 14.0% |

*(oracle 0.0222, random 0.0167.)*

**The semantic model captures 2.6× more of the achievable headline uplift than the lexical model** (36.5%
vs 14.0%), and roughly doubles the CTR uplift over a random pick. Reading the headlines *semantically* —
for curiosity gap, emotional hook, specificity, provocation — picks the causally-better intervention far
better than length/question-mark/keyword features.

## Why this matters
- It **directly attacks the humbling EXP-054 numbers** on the one KPI that tests the thesis ("what happens
  if I do X"). The interventional task was not hopeless — it was *semantic*, and an LLM judge closes much
  of the gap, the same lesson as EXP-047 (stance) now proven for `do(x)` *selection*.
- It completes the pattern the whole project kept finding: **gestalt-without-evidence fails (EXP-037),
  lexical reading fails (EXP-044/054), semantic reading of the real content succeeds (EXP-047/056).** The
  general engine's front door should route interventional questions to the semantic selector, held to
  policy-regret.

## Honest limits
- 45 tests, one LLM pass; a production run judges thousands via the API backend (`InterventionSelector`
  with `anthropic_judge_fn`).
- Self-judged here (same model family describes and is scored); a separate-key API judge is the stricter
  production setup the module already supports.
- Still only 36.5% of the oracle — headline CTR has real irreducible noise and image effects the text
  cannot see; the point is that semantic **more than doubles** lexical, not that it is solved.
- Public data; a mechanism benchmark for the selector, not a leakage-free skill number.

## Reproduce
`python -m experiments.exp056_semantic_intervention` → `experiments/results/exp056_semantic_intervention.json`
(replays committed LLM picks). `python -m pytest tests/test_intervention_selector.py`.
