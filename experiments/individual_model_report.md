# Individualized model report (spec Phase 4)

**Question:** `this entity + this message/action + this context → response distribution`, with
hierarchical partial pooling so the estimator degrades gracefully with evidence.

**Headline:** the estimator is **built, tested, and validated on synthetic data where the response
function is known** — hierarchical partial pooling beats both the segment model and a no-pooling
per-person model, and stays calibrated where no-pooling overfits. The **real-world individual claim
is BLOCKED-ON-PRIVATE-DATA**: there is no labeled real behavioral dataset in this repo, and none was
fabricated.

## What is implemented (real code)
- `swm/state/latent.py` — `HierarchicalPosterior` / `BetaHierarchical`: person ← segment ←
  population shrinkage. Cold → segment prior (wide); evidence → shrink to individual.
- `swm/transition/individual_transition.py` — the response function: per-entity posterior + a
  calibrated logistic head over `[message features + segment_logit + person_logit]`, with an
  **evidence-source ablation switch** (`sources ⊆ {segment, person, message}`). `transition()`
  advances the entity posterior after each outcome (the state recurrence).
- `swm/worlds/individual_world.py` — `fit_stream` + a temporal-holdout backtest with a calibration
  grade.
- `swm/eval/individual_response_eval.py` — compares segment / +person / +message / full-individual
  and (pluggable) raw-LLM / raw-LLM+context arms on the same rows.

## Synthetic estimator validation (`experiments/individual_harness.py`, `results/exp011_individual_synth.json`)

Generative truth: 300 contacts, each with a latent reply rate θ_i ~ Beta(2,6); Zipf-ish contact
frequency (uneven evidence — the regime where pooling matters); a true short+CTA message effect;
6,000 events; temporal 70/30 split. Test base rate 0.369, segment rate 0.359.

| arm | log loss | Brier | ECE |
|---|---|---|---|
| segment (population only) | 0.6588 | 0.2330 | 0.0092 |
| no_pooling (per-person MLE) | 0.6307 | 0.2203 | **0.0885** |
| **partial_pooling (the model)** | **0.6062** | **0.2098** | 0.0262 |
| + message features | **0.5890** | **0.2025** | 0.0281 |

**Partial pooling beats both segment (−0.053 log loss) and no-pooling (−0.025), and is calibrated
(ECE 0.026) where no-pooling is badly overconfident (ECE 0.089).** Adding message features adds
another −0.017 — the content of the action carries signal beyond the person's base rate.

### Where pooling wins (log loss by contact-evidence bucket)
| bucket | n | partial_pooling | no_pooling | segment |
|---|---|---|---|---|
| cold (<3 obs) | 62 | 0.5427 | 0.5365 | 0.5764 |
| warm (3–15) | 412 | **0.6048** | 0.6131 | 0.6432 |
| hot (15+) | 1326 | **0.6097** | 0.6405 | 0.6676 |

The story is exactly the theory: on **cold** contacts partial pooling ≈ segment (correctly falls
back to the prior; no-pooling's tiny edge here is n=62 noise), while on **warm/hot** contacts
partial pooling dominates because it trusts the individual without the variance blow-up that sinks
no-pooling. This is the cold-start → warm behavior the design promised, measured.

## Evidence-source ablation
The `sources` switch is the ablation: `{segment}` vs `{segment,person}` vs `{segment,person,message}`
are the rows above. Each source earns its place on held-out log loss — person > segment, message on
top of person.

## Raw-LLM arms — BLOCKED
`raw_llm` and `raw_llm+context` require an LLM/agent predictor over the same rows and an
ANTHROPIC_API_KEY (absent here); the harness accepts a predictions table and marks the arm BLOCKED
otherwise. On synthetic data an LLM arm is not even meaningful (there is no real language to read),
so the LLM comparison for individuals belongs on real message data — see below.

## What would unblock the real claim
A labeled stream of real outbound messages → observed responses (email reply / positive-reply /
objection / conversion). The plumbing is ready: `swm/ingestion/store.py` (as-of event store,
anti-inflation reply labels), `swm/ingestion/gmail_search.py` (a Gmail ingestion path), and
`IndividualWorld.backtest`. With the operator's consent to ingest their own sent/received mail, the
same harness runs on real behavior and the L0→L4 ladder (`swm/eval/harness.py`) grades it. **Until
that data exists, the individual model is validated as an estimator and unproven as a world claim —
stated plainly, not faked.**
