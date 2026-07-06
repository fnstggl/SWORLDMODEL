# Simulation audit — are we simulating a world, or compositing numbers?

Commissioned as an adversarial 5-lens agent swarm (compositing-skeptic, simulation-advocate, KPI-critic,
world-model-theorist, benchmark-designer) that read the actual code and results, plus a synthesis judge.
The verdict is blunt and the code adjudicates it.

## Verdict: ~85% compositing, ~15% genuine dynamics — and the shipped flagship is 100% composite

**The end-to-end deliverable is a mean of independent regressions.** `GroundedSimulator.simulate_population`
is literally `ps = [readout.predict(q, d) for d in population]; p_outcome = sum(ps)/n`. Every
`readout.predict` is a sigmoid of a linear combination of PCA-projected one-hot demographics, shrunk toward
the question's own training marginal. **No person's prediction reads any other person's — `∂pᵢ/∂pⱼ = 0`
everywhere.** There is no state, no iteration, no feedback, no emergence. The "value_drivers" are a
linear-model attribution, not an emergent property. "Simulate the event" is, arithmetically, a linear pool
of a fitted logistic regression.

The other front doors are the same shape: `QuestionEngine.aggregate` is an explicit log-linear opinion
pool (a shrunk weighted **sum** of non-interacting drivers); `GeneralSimulator.answer` is a
confidence-weighted average of `logit(p)` across sources. Compositing all the way down.

**Genuine dynamics exist but are disconnected from the flagship:**
- `UnifiedBeliefDynamics.update_person` — the per-agent temporal operator — **has zero call sites**. It is
  never iterated over a population.
- `MultiStepRollout.rollout` is the only real state evolution (`path.append(nxt)` across steps), but it
  evolves a **single scalar belief**, not a population, and its "Monte-Carlo" is a deterministic symmetric
  grid (the mean path is fully determined). And belief is a near-martingale (EXP-033) where it can't beat
  persistence anyway.
- The actual agent substrate (`swm/simulation/actors.py`, `swm/graph/diffusion.py` with `IMPLEMENTED =
  False`) is unwired — scenery that makes the repo look like an ABM without running as one.

**The one genuinely dynamical win, EXP-045, is honest about being compositional:** `Ŝ(t) = S(last) +
[ĝ(demo_t) − ĝ(demo_last)]` beats persistence and the edge grows with horizon — but `ĝ` is one static
regression evaluated at two **exogenous real-census** demographic compositions. The census supplies the
forward motion, not simulated agents.

**The interaction that would make this a world model — agent×agent, or agent×evolving-aggregate — exists
nowhere on any scored path.** The one wired agent×event interaction (EXP-042's responsiveness gating) is
real as *mechanism* (slope 0.71 responsive vs 0.51 entrenched) but predictively **inert** (+0.0007 over an
additive model). The parts of an agent-based model are individually built and validated; they have never
been composed into one loop.

## The benchmark is measuring the wrong thing

The headline KPIs — log-loss/accuracy, **MAE-on-population-share**, market-consistency, CRPS-on-a-belief-
path — measure *predictive reconstruction of a marginal*, which is exactly what compositing is good at:
- **MAE-on-share is near-tautological.** `simulate_population` averages predictions each shrunk toward the
  per-question training marginal (`tau=40`), so "marijuana 0.336 vs true 0.337" substantially measures
  *recovery of a marginal the model was fit on*, not simulation of an unobserved outcome.
- **Market-consistency (corr 0.57) grades crowd-mimicry** — rewarding the model for matching the price is
  the inverse of the stated value proposition (differentiated skill on off-market questions).

A "what-would-actually-happen" simulator must be scored on what these miss:
- **KPI-A — Interventional / policy-regret skill.** Predict how the outcome *changes* when you change an
  input. Nothing scores this today (`counterfactuals.py` self-labels its `do()` outputs "not identified";
  ROADMAP stage G is ❌). The one in-repo dataset with *genuine randomized interventions* — Upworthy
  headline A/B tests (`benchmarks/upworthy`, `IMPLEMENTED=False`) — should be built and scored with
  off-policy value / regret (IPS or doubly-robust) + sign-of-CATE recovery. **This is the only KPI that
  directly tests the thesis.**
- **KPI-B — Skill-vs-persistence at each horizon.** Generalize EXP-045's rolling-origin design into the
  *rule*: report `skill = 1 − loss/loss_persistence` at each horizon with CIs, per-step calibration, and
  **turning-point / trajectory accuracy** (current KPIs score only the terminal share, never path shape).
  Measurable now on the 406 GSS rolling-origin forecasts.

## The one architectural change that would earn the word "simulate"

**Make aggregation non-separable: a mean-field coupling term so each agent's readout depends on the
evolving aggregate state, rolled forward over sampled events.** Replace `sum(ps)/n` with a loop that
(i) samples an event stream over the horizon (the missing event model), (ii) at each step applies
`UnifiedBeliefDynamics.update_person` to every agent's VariableMap — reusing the operator that currently
has zero call sites — where each agent's update depends on the current population aggregate
(`∂pᵢ/∂pⱼ ≠ 0`), and (iii) aggregates bottom-up each step, carrying state forward as `rollout.py` already
does for the scalar case. This composes three separately-validated components into one genuine forward
loop — and its output could no longer be reproduced by a mean of independent logistic predictions.

**Held honestly to KPI-B:** if the coupled roll-forward cannot beat both the independent mean and
persistence (EXP-042's inert interaction and EXP-045's exogenous-demographics result warn this is a live
possibility), the honest move is to **rename** — the code is a calibrated, correlation-aware regression
with bottom-up averaging and an auditable decomposition. A good thing; just not "simulate a world."

## Ranked next steps (from the audit)
1. **Wire the mean-field coupling loop into `simulate_population`** — turns the flagship from a
   mean-of-regressions into a coupled roll-forward; reuses the dead `update_person`. Gate on KPI-B.
2. **Build the Upworthy randomized-intervention harness → KPI-A** (off-policy regret / CATE-sign). The only
   test of "what happens if I do X"; data already in-repo.
3. **Make skill-vs-persistence-at-each-horizon + trajectory fidelity the primary scoreboard** (KPI-B).
4. **Couple the population rollout to the event operator** (EXP-045's own next step) — endogenous dynamics,
   not re-averaged exogenous census.
5. **Demote the perverse KPIs** (MAE-on-share, market-consistency) to calibration diagnostics.
6. **Wire or delete the dead agent substrate** so the repo stops implying capability it doesn't run.

## Bottom line
Today the system is a well-engineered, honestly-documented **calibrated compositor** with two genuine but
disconnected dynamical fragments. The scaffolding for a real agent-based world model is fully present and
individually validated — but it has never been composed into one coupled, forward-rolling loop, and until
it is (and beats the independent mean on KPI-B), "simulate what would actually happen" overstates what the
shipped path does. This audit is the mandate to either build the coupling loop or rename the claim.
