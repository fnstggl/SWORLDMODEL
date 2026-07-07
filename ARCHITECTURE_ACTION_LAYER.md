# The Action Layer + the Navigable Object — the build spec

*The design for the two core value props before a line of product code: (1) the general best-action
layer (`argmax_a E[U(outcome) | do(a)]`), and (2) the navigable outcome object that replaces a scalar
probability with reasons + branches. Plus the hard demotion of the compositing flagship.*

This is a spec, not a summary. It is written to be implemented against.

---

## 0. Ground truth — what exists today (adjudicated from code, EXP-063 state)

- **The navigable object is not built.** Only `variance_decomposition` in `swm/simulation/structural.py`
  (the reducible/irreducible split) exists, and it is wired to nothing. No pivotal-branch extraction, no
  "worlds," no question front door.
- **The compiler (Stage ②) is not built.** Confirmed: no `compile` step turns a question into a structural
  model. A human still picks the mechanism. It therefore has no verification/selection loop either.
- **The action layer today is three narrow, disconnected prototypes:**
  - `swm/api/individual_simulate.py::best_message` — ranks a *fixed* candidate list on *one* person via a
    deterministic `response_fn`. No Monte-Carlo, no outcome distribution, no utility beyond `p_respond`, no
    action generation. Validated (+22pt precision@1 on CMV, EXP-060) but Level-1-only.
  - `swm/api/intervention_selector.py::select` — an LLM *judges text* and ranks. **No simulation.** The
    fragile shortcut, not a `do`-operator.
  - `swm/simulation/counterfactuals.py::best_of` — `do(A)` vs `do(B)` on the aggregate regression head;
    picks max P(hit) from a fixed list.

None runs the structural engine, generates actions, returns a distribution per action, or generalizes
across mechanisms. Fixing this fragmentation is the point of this spec.

---

## 1. Why the action layer is a top-2 value prop

A market or an LLM answers *"what will happen."* The action layer answers *"what should I do, and what
happens if I do it"* — a `do`-query neither can touch. It is off-market by construction (no crowd prices
"the best email to send Andreessen"), usually short-horizon and structure-dominated (the forecastable
regime where a simulation genuinely wins), and we already have one real validated instance of it
(`best_message`, +22pt causal lift). The job is to generalize that single instance into one robust,
mechanism-agnostic layer.

The core query:

```
best_action  =  argmax_a  E[ U(outcome) | do(a) ]
```

where the expectation is over the outcome distribution produced by **rolling the simulation forward with
action `a` injected as an intervention** — the structural Monte-Carlo engine that already exists.

---

## 2. The seven components (each framed robust-vs-fragile)

### C1 — An action is a *typed intervention on the model*, not a string
The single most important robustness decision. `intervention_selector` is fragile precisely because it lets
an LLM judge *text* instead of simulating. An `Action` is a first-class transform `Model → Model`:

- **Parameter intervention** `do(X := x)` — set price = 49, tone = warm, launch-timing = pre-conference.
- **Structural intervention** — add/remove an entity or edge: a new competitor, an endorser in the
  influence graph, a removed blocker.
- **Temporal/exogenous intervention** — inject an event at time *t*: "send the follow-up on day 3."

The LLM *proposes* actions; scoring is *always* simulation. LLM never scores.

### C2 — Candidate generation + search (not a fixed list)
An `ActionSpace` with three regimes:
- **Enumerable** (which of 5 emails / 3 features) → score all.
- **Continuous** (price, discount, timing) → grid then local refine (Bayesian optimization over the
  parameter), inner Monte-Carlo per point.
- **Combinatorial / open-ended** (what to *say*) → the LLM is the **action generator** (it has read how
  people negotiate/pitch/price), proposes a *diverse* candidate set; the simulator scores each; a refine
  step mutates the top few ("shorter", "lead with the ask", "add social proof") and re-scores.

### C3 — The nested-loop architecture (the "exact best architecture")
Two nested loops with **adaptive budget**:

```
OUTER: search over actions a ∈ A          (best-arm identification / racing)
   INNER: for fixed a, Monte-Carlo the outcome → F_a = { outcome | do(a) }
          → E[U|a], its standard error, the full distribution, reducible/irreducible split
```

The inner loop is the existing structural ensemble. The **outer loop is what keeps it from being fragile.**
Naïve "fixed N per action, take the argmax mean" will crown a lucky loser and waste compute equally on
obviously-bad and too-close-to-call actions. Use **best-arm identification** (successive halving / UCB /
Thompson over actions): pour inner sims into the promising-and-statistically-close actions, prune the
dominated early. This buys:
1. **Efficiency** — compute goes where the decision is contested.
2. **A confidence statement** — "A beats B at 95%".
3. **Honest abstention** — "A and B are a tie within noise at this budget" instead of a fabricated winner.

The `montecarlo` in `structural.py` is the inner primitive; the outer racing loop is new
(`swm/decision/best_action.py`).

### C4 — Explicit utility + risk preference
`argmax E[U]` needs a real `utility_fn(outcome) → ℝ`, pluggable:
- `P(desired)` — best_message: P(reply).
- `E[value]` — pricing: `E[profit] = price · P(buy | price) · volume`, swept over price.
- **Risk-adjusted** — maximize `P(good)` subject to a cap on `P(disaster)`, or optimize a quantile / CVaR.
Do not hardcode "maximize the mean."

### C5 — Every action returns the navigable object, not a scalar
The action layer and the navigable object (§3) are the **same renderer applied to `F_a`.** The output:

> "Send B. +14pt reply vs A. Wins in the worlds where she's busy (shorter beats detailed there); loses only
> if she's already decided — ~12%. Ranking validated on CMV-like persuasion (precision@1 0.74); pricing is a
> hypothesis until backtested."

i.e. E[U], the distribution, reducible/irreducible split, pivotal branches, contrast vs next-best **and vs
do-nothing**, and a calibration grade. Never a naked "do B."

### C6 — Reflexivity and sequences (policies)
The action can change the world (a follow-up lands on the person the first message left behind —
`simulate_thread` already carries state). The general case is `argmax over policies π` (a short sequential
plan). **Start with single actions; make the interface admit a policy** so negotiations / multi-touch
outreach / pricing schedules are an extension, not a rewrite.

### C7 — Identifiability honesty + the scoreboard
Model-based counterfactuals are not identified causal effects (as `counterfactuals.py` already states); they
are only as good as the head's calibration. The action layer ships with its **own KPI — policy regret +
CATE-sign + off-policy value (IPS / doubly-robust)**, never log-loss. Validate on observed-intervention
data: **Upworthy A/B (EXP-054), the CMV natural experiment (reproduce +22pt, EXP-060), and a pricing sim.**
Every recommendation carries "validated on domain X / hypothesis on domain Y".

### Where it sits
Defined **generically over "a simulator"** (anything exposing `simulate_once` / a `StructuralModel`):
- Works **today** on hand-built models (individual, pricing, NBA bracket) — no compiler needed to start.
- **Automatically generalizes** the moment the compiler emits models. Built once, against the engine —
  never per-mechanism. This is what stops it from becoming a fourth bolted-on prototype.

---

## 3. The Navigable Object (shared output layer)

Small, high-value, shared by forecasting *and* every action. On the existing ensemble:

1. **Point + distribution** — mode, mean, histogram.
2. **Reducible / irreducible split** — `variance_decomposition`, already built; surface it.
3. **Pivotal-branch extraction ("the worlds") — the one new bit.** Keep each trajectory's terminal outcome
   *and* its sampled latent draws (which variables/events took which value). Compute each variable's
   **attribution to the outcome** (conditional outcome given its branch — Sobol-style, cheap because the
   ensemble already exists). Highest-attribution variables *are* the pivots; condition on their branches:
   "in the worlds where X → 85%, else → 25%, and P(X) = 0.4." Automatic pivotal-branch discovery from the
   ensemble — same machinery as the variance split, extended to per-variable attribution.
4. **What-to-watch** — pivots ranked by how much resolving them would sharpen the forecast (reducible mass
   each unlocks).

Lives in `swm/report/navigable.py`, consumes any ensemble from `montecarlo`.

---

## 4. Demote the compositing flagship (hard)

- **Rename** `GroundedSimulator.simulate_population`: it is no longer "simulate." It becomes a named library
  primitive — `IndependentPopulationReadout` / `MarginalAggregator` — invoked *only* when the compiler
  determines the question is a non-interacting marginal (opinion shares, etc.).
- **Reserve the word "simulate"** for "the compiler selected and ran the right mechanism, rolled it at
  calibrated time, returned the navigable object." A mean of regressions is a `readout`, never a simulation.
- Update README/ROADMAP: the front door is the engine + (eventually) compiler; the compositor is a demoted
  leaf carrying its honest label.

---

## 5. Build order

The action layer and navigable object sit on the engine and **do not need the compiler to start**:

1. **Navigable object** on `structural.py` (distribution + reducible/irreducible + pivotal branches).
   Shared by everything; unblocks the UI vision.
2. **General action layer** on the engine (typed interventions → nested Monte-Carlo + best-arm racing →
   utility/risk → returns the navigable object per action → policy-regret scoreboard). Validate by
   *reproducing best_message on CMV*, adding a *pricing* model, scoring *Upworthy* regret. The main value
   prop, built once, generic.
3. **Demote the flagship** (rename + reframe) — mechanical, done alongside.
4. **Compiler (Stage ②)** with a candidate-and-verify loop — the bigger build; feeds models into 1 & 2.

## 6. Proposed module layout

```
swm/report/navigable.py     — NavigableOutcome from any ensemble (§3); pivotal-branch attribution
swm/decision/action.py      — Action = typed intervention (parameter/structural/temporal): Model → Model
swm/decision/space.py       — ActionSpace + proposers (enumerable / continuous / LLM-generated)
swm/decision/best_action.py — nested loop: inner montecarlo + outer best-arm racing → ranked actions
swm/decision/utility.py     — utility_fn library + risk preferences (mean / quantile / CVaR / constrained)
swm/eval/policy_regret.py   — policy regret, CATE-sign, IPS / doubly-robust off-policy value
```
