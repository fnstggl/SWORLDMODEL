# EXP-064 — the world-model compiler: `simulate(question)`, the whole system underneath

The keystone. Stage ② of the architecture (`ARCHITECTURE_WORLDMODEL.md`) is built: the layer that reads a
question and **compiles the right structural model**, then runs it through the calibrated-time Monte-Carlo
engine. Everything built before — Levels 1–3, the bracket, the diffusion SCM — is now this compiler's
**mechanism library**. One call handles anything:

```python
wm = WorldModel(compiler=StructuralCompiler(compile_fn), retriever=retriever)
wm.simulate("Will the FOMC raise rates in July?")
```

## The pipeline (one call)

```
question ─▶ retrieve context ─▶ COMPILE structural model ─▶ run mechanism ─▶ Monte-Carlo ─▶ forecast
                                 (LLM: mechanism + variables      (bracket / committee /     (distribution +
                                  + equations + timescales)        electorate / SCM / agent)  reducible/irreducible)
```

## What was built

- **`swm/api/model_spec.py`** — the spec IR + a **safe** structural-equation evaluator. The LLM writes
  equations like `"0.3*(0.5*approval + 0.35*economy - vote)"`; they are evaluated through a **whitelisted
  AST walker** (arithmetic, declared variables, a small math-function set) — never `eval()`. Attribute
  access, imports, and arbitrary calls raise. (Security-tested.)
- **`swm/api/compiler.py`** — `StructuralCompiler` (question → `ModelSpec` via a pluggable LLM backend:
  `anthropic_compile_fn` in prod, `cached_compile_fn` in dev) and the **mechanism library** that
  instantiates a spec into a Monte-Carlo forecast:

  | mechanism | real process | engine |
  |---|---|---|
  | `bracket` | competition / tournament | seeded best-of-k bracket |
  | `committee` | named decision-makers deliberate | AgentSociety (Level 2) |
  | `electorate` | population by segment | PopulationSimulator (Level 3) |
  | `single_agent` | one person's response | Level-1 quantities |
  | `generic_scm` | coupled quantitative variables | calibrated-time diffusion SCM |

- **`swm/api/world_model.py`** — `WorldModel.simulate(question)`: retrieve → compile → run → a calibrated
  forecast with the mechanism, the outcome distribution, the reducible/irreducible split, a forecastability
  verdict, and the compiled spec as an audit trail.

## One call, five question types, five mechanisms

Committed specs (the LLM compiler's structural judgment, cached for reproducibility); production swaps in
the API backend behind the identical interface.

| question | compiled mechanism | forecast |
|---|---|---|
| Will OKC win the 2026 title? | **bracket** | P(OKC) **0.36** (0.43 even with strengths known → rest irreducible) |
| Will the FOMC raise rates? | **committee** | P(hike) **0.32** (leans hold) |
| Will the referendum pass? | **electorate** | P(pass) **0.86**, turnout-weighted, coupled |
| Will the incumbent hold the seat? | **generic_scm** | P(hold) **0.28**, **irreducible 93% → forecastable: False** (an honest coin-flip) |
| Will this person reply? | **single_agent** | P(reply) **0.75** |

The same call dispatches each to its real generative process — the thing a single generic mechanism could
not do, and precisely why the NBA composite failed. The incumbent case is the honesty payoff: the compiler
returns a wide distribution and **flags it unforecastable** rather than bluffing a confident number.

## Tests — `tests/test_compiler.py` (8, all pass)

Safe-eval arithmetic/functions **and rejection of unsafe input** (`__import__`, attribute access, `open`),
spec parsing, every mechanism handler, the front-door dispatch, and the cached backend raising on a miss.
Full suite: **252 passed**.

## What this is, and what remains

This is the vision realized as an interface: *ask anything → the system maps the relevant slice of the
world into the right runnable model and rolls it forward at calibrated time → a calibrated distribution
with an honest uncertainty split.* Two honest edges remain:

1. **The specs' quality is the LLM's job.** The compiler is only as good as the structural models the LLM
   emits; that is now the central thing to measure and improve (a spec-quality benchmark), and it is where
   the "inference is good enough" bet gets tested directly.
2. **Scored validation** should run on forecastable questions (inside the predictability horizon) where the
   thesis predicts a win — the compiler makes that a single uniform harness across mechanisms.
