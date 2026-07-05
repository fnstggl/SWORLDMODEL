# EXP-012 — Simulation architecture audit: where the "world model" was still a classifier

Brutal, no spin. Written before/while building the real simulation engine, and confirmed by the
benchmark (`experiments/results/exp013_simulation_benchmark.json`).

## Where was the model still just `state -> classifier`?
Everywhere that mattered for the EXP-009 headline. Concretely:
- `swm/worlds/aggregate_world.py` + `swm/transition/aggregate_transition.py`: the "state-transition
  model" builds a feature vector from the current `PopulationState` and feeds it to **one logistic
  per threshold** (`OutcomeHead`). That is `state -> feature vector -> logistic -> P(outcome)`. The
  state *evolves* between items (EMA updates), but a single prediction is a classifier over the
  current features. No actors, no reactions, no within-item timesteps.
- `swm/worlds/individual_world.py`: same shape — pooled person logit + message features -> logistic.
- EXP-009's "aggregate_world" / "structured" / "calibrated" tiers are all this. So when EXP-009
  concluded "world model ties raw LLM," it was really "a logistic over as-of features ties the LLM."
  That is not a fair test of *simulation*, and this audit's whole point is that it was never run.

## Where were transitions hand-coded rather than learned/simulated?
- The factor `update` rules in `swm/state/factors.py` and `PopulationState.observe_outcome` are
  **hand-coded EMA/sufficient-statistic updates**. That is defensible as a *state update* (it is the
  correct Bayesian/empirical update), but it is not a simulated dynamic — nothing reacts.
- `swm/state/transition.py::TransitionModel.step` samples a one-step outcome from the head and
  applies those hand-coded updates. It never simulates actors interacting.

## Did `/v1/rollout` simulate actors reacting, or just update scalars?
Just scalars. Pre-fix, `/v1/rollout` sampled a one-step band distribution and applied factor
updates; there was **no actor, no exposure, no social proof, no front-page dynamic** — and (the
EXP-008 audit found) it even used a state-ignoring `PriorHead`. It was a fixed-distribution Monte
Carlo, not a simulation.

## Did rollout simulate steps or sample a one-step outcome?
One-step. `swm/simulation/rollout.py::simulate` (the aggregate one) does advance state across an
*action plan* (multiple posts), but within a single prediction it draws one outcome per step from
the head — it does not simulate a *population reacting over time to one action*. The multi-step axis
was "more posts", not "the reaction cascade of one post".

## Which parts were NOT a real simulator?
- The entire aggregate/individual "world model" prediction path (classifier).
- `/v1/rollout` (scalar updates).
- `swm/transition/diffusion.py` (IC/LT/Hawkes) existed but was **never wired into a prediction** —
  a real mechanic sitting unused.

## Which parts are acceptable as baselines / readout heads?
- The logistic/GBDT heads are fine **as calibration readouts or as explicit baselines to beat** —
  which is exactly how they are now used (the benchmark's `old_classifier` / `learned_gbdt` tiers).
- The EMA/conjugate state updates are the correct *state* transition; they stay.
- The as-of retrieval + leakage gate are correct and reused unchanged.

## Minimum implementation to make it an actual simulation engine
Exactly what was built (`swm/simulation/{actors,reactions,event_queue,trajectory_state,policies,
engine}.py`):
1. **Explicit actors/segments** — 8 weighted HN community segments with affinities + attention.
2. **Multi-step reactions** — each step, exposed segments upvote with a propensity from their
   affinity·(action features) · author reputation · social proof · novelty.
3. **World-state updates each step** — score, exposure pool, social proof, novelty/fatigue, and a
   **stochastic front-page transition** (early-velocity logistic) that jumps exposure ~30×.
4. **Probability from the trajectory distribution** — P(hit) = fraction of simulated trajectories
   that cross the cascade, then a thin Platt readout. The readout only reads the simulated outcome;
   it does not replace the simulation.

## Did the rebuild change the verdict? (from the benchmark)
- The old `state -> classifier` (`old_classifier`) is the **worst** non-LLM tier (log loss 0.3379).
- The **simulation** is the **best-calibrated** tier (ECE 0.0185) and, on **high-context posts,
  beats raw LLM + context** (0.258 vs 0.285).
- The **hybrid** (simulation gated with the calibrated LLM) **beats raw LLM and raw LLM + context
  overall** (0.3160 vs 0.3200 / 0.3229).
So: replacing the classifier with an actual simulation did not make it win *everywhere* (it still
loses on cold-start/semantics-dominant posts, where the LLM's prior dominates), but it produced a
better-calibrated model that wins where state is rich — which is the honest, hypothesis-shaped
result. Full numbers and the "is it a real simulator now" verdict: `exp013_real_simulation_world_model.md`.
