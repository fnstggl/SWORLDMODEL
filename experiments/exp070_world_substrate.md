# EXP-070 — the persistent World substrate: a coupled two-scale world, scored

The architectural leap toward the full vision: not a throwaway per-question model, but **one persistent
world** where entities at different scales coexist on a single clock and *affect each other*. Built to the
brief's discipline — **score coupled-vs-separate before scaling up.**

## What was built — `swm/world/substrate.py`

- **`Entity`** — a node (person / institution / population / environment), carrying STATE, advanced by its
  own mechanism (`step_fn`) and read by its own `readout_fn`. The mechanism library (single-agent /
  committee / electorate / SCM) supplies these.
- **`Coupling`** — a directed edge: the OUTPUT of one entity `wire`d into the INPUT of another. This is what
  makes the world non-separable *across scales*.
- **`World`** — `advance(dt)` steps every entity by the same elapsed time on ONE shared clock (inputs
  gathered before any entity steps, so a tick is well-defined); `query(entity)` reads an outcome;
  `without_couplings()` cuts the edges for the ablation; `rollout` / `montecarlo_world` make a question a
  **query against forward-simulations of the shared world**.

This is the thing that was missing: a single time axis and cross-scale wiring, instead of independent
per-question models.

## A. Cross-scale feedback the substrate captures — and separate models cannot

A **bank-run world**: an environment RUMOR → depositors' (individuals) withdrawal intent → the BANK
(institution) distress → which feeds **back** into the rumor and the depositors. From an *identical* one-time
shock:

| world | bank outcome | distress trajectory |
|---|---|---|
| **coupled** | **FAILED** | 0 → 0 → 0 → **1.0** → 1.0 … (cascade ignites) |
| **separate scales** | stable | 0 → 0 → 0 → 0 … (fizzles) |

The same shock produces **opposite** outcomes. Contagion-to-failure is an emergent, cross-scale property of
the wiring — **not reachable by simulating the scales independently**. This is the qualitative capability a
shared world adds.

## B. A real, scored two-scale test: individuals → the institution they sit in

The Supreme Court as a World of 9 justice entities (individual scale — each carries an ideology STATE that
**drifts** as its record accumulates) coupled UP into the Court entity (institution scale — a committee vote
over the justices' current states). On real SCDB cases, leakage-free, the honest question: does coupling the
individual-scale **dynamics** up to the institution beat treating each justice as a **static** input?

| configuration | vote-margin MAE | direction acc |
|---|---|---|
| **SEPARATE** (justices frozen at train-era ideology) | **0.170** | 0.521 |
| **COUPLED** (as-of drifting ideology, wired up) | 0.182 | 0.522 |

**Verdict: coupling does not beat separate here (it ties/slightly hurts on margin).** So — per the discipline
— **we do not scale up on this case.** This is consistent with everything: SCOTUS is robust to these
couplings (EXP-062's public backdrop was also neutral), and the individual drift adds noise the static
train-era estimate already captured well enough. The scoring did its job: it told us *not* to build more here.

## What this establishes

- **The substrate is real and works.** Entities, cross-scale couplings, one clock, coupled Monte-Carlo
  rollout, `without_couplings` ablation — and it captures genuine cross-scale feedback (A) that independent
  models structurally cannot.
- **Coupling must earn its place, and here it didn't.** The two-scale SCOTUS coupling ties the separate
  baseline, so the honest move is to hold — exactly the "score before scaling up" discipline the brief asked
  for. The substrate is the *machinery* for the full digital-twin vision; whether to wire any given pair of
  scales is now an empirical question with a scoreboard, not a leap of faith.
- **The next coupling to test** is one where cross-scale feedback plausibly manifests on scoreable data — an
  environment→individuals→institution loop like FOMC (macro economy → members' hawkishness → the rate vote),
  where the feedback A demonstrates is real. The bank-run shows the substrate can represent it; the SCOTUS
  result shows we only keep a coupling that beats its ablation.

## Tests — `tests/test_world_substrate.py` (6, all pass)

Shared-clock advance, coupling wiring, query-without-mutation, `without_couplings` isolation, the
cross-scale feedback (coupled cascades / separate fizzles), and `rollout`. Full suite: **292 passed**.
