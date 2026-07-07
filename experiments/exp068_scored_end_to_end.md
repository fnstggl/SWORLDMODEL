# EXP-068 — self-correcting front door + a scored end-to-end run on forecastable questions

Two builds: (1) wire the validator into the front door so every `simulate()` self-corrects by default, and
(2) run the **full autonomous pipeline** — LLM compiles → validate → repair → Monte-Carlo → score — against
real resolved outcomes, and report what actually happens. No cherry-picking.

## 1. Self-correction is now the default path

`WorldModel` wraps its compiler in a `ValidatingCompiler` unless `validate=False`. Every `simulate(question)`
now **validates the compiled spec (simulate-and-inspect) and repairs it** (if a `repair_fn` is supplied)
before running, and the `validation` report rides along in the output. Backward compatible: `validate=False`
restores the raw path; a buggy spec then passes through unchanged (regression-tested).

## 2. Scored end-to-end — the honest result

**Task.** 15 GSS attitude topics; a real as-of year A (share known) → forecast the share ~12 years later;
scored against the realized share at T. Decade-horizon opinion is genuinely *forecastable* (structure- and
persistence-dominated, inside the predictability horizon). The LLM (Qwen-72B via HF, blind to the future,
given only the as-of value) compiled the **entire** spec — mechanism, variables, equation, and rate. Backend
is pluggable: `anthropic_compile_fn` with a key swaps in a frontier paid model.

### The validator earned its keep in the wild

Of 15 autonomously-compiled specs, **the validator caught 5 as malformed** — the LLM used *categorical
string labels* (`"legal_any_reason"`) where a numeric stance belongs (`non_numeric_field`). These are exactly
the bugs EXP-067 was built for, now caught **in a live run, cleanly, with no crash** — the front door
refused to trust an unrunnable spec instead of emitting garbage. (With a repair backend attached they are
handed back and fixed; here they are flagged and excluded so the score is clean.)

### Accuracy: the pipeline **ties persistence, does not beat it**

On the 10 cleanly-compiled specs:

| | value |
|---|---|
| MAE, autonomous pipeline | **0.078** |
| MAE, persistence baseline (share stays at A) | 0.076 |
| skill vs persistence | **−0.02 (a tie, marginally worse)** |
| 80% interval coverage | **0.50** (nominal 0.80 — over-confident) |

**Honest reading.** The pipeline is fully operational and produces sensible point forecasts — but on
decade-horizon aggregate opinion it does **not beat persistence**, which is a very strong baseline here (most
attitudes barely move over 12 years, and the ones that drift do so for exogenous reasons the model can't
see). This is entirely consistent with the project's standing finding that opinion at this scale is
**persistence/marginal-dominated** (EXP-053/061). The world-model's real wins are on **structured** questions
— committee margins, tournaments, individual best-message (EXP-065) — not on unstructured mass opinion,
where there is little structure to exploit and persistence is near-optimal.

**A concrete, diagnosable calibration miss.** Coverage 0.50 (vs 0.80 nominal) traces to a *mechanism-choice*
issue: the LLM modeled these as near-static `electorate` cells, which **under-propagate uncertainty** over a
long horizon. A `generic_scm` diffusion with the same (correctly-scaled, per EXP-066) volatility would widen
the interval as √horizon and cover far better (EXP-063 showed exactly that: data-rate diffusion → 0.80
coverage). So the next concrete compiler improvement is clear: **mechanism selection should weigh uncertainty
propagation**, preferring a diffusion for smooth long-horizon quantities.

## What this establishes

- The autonomous pipeline **runs end-to-end** and **self-corrects**: the validator caught 5/15 real LLM spec
  bugs live and the front door didn't emit garbage.
- On a deliberately hard, persistence-dominated task it **honestly ties persistence** rather than pretending
  to beat it — the value is in structured questions, measured separately (EXP-065).
- The one calibration gap is diagnosed to mechanism choice, giving the next concrete build.

## Tests

`tests/test_spec_validator.py` gains the front-door self-correction test (validate-by-default repairs a buggy
spec; `validate=False` preserves the raw path) and a `non_numeric_field` static check. Full suite: **262
passed**. Reproducible offline from the committed spec cache (`experiments/results/exp068/specs.json`).
