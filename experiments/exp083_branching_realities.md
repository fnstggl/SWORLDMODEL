# EXP-083 — branching-realities rollout: modeling FORWARD through future events, never faking a point

**The question.** The ROADMAP's "hard core" and EXP-033's verdict — *"you cannot forecast forward without
forecasting the events; long-horizon forecasting is a future-event-forecasting problem, not a better-
dynamics problem"* — say the missing capability is not smoother dynamics but the **discrete future events
that fork reality**. The existing `EventModel` (EXP-035) and the calibrated transition-operator work are the
*continuous* term (a smooth diffusion of a scalar belief). This experiment builds and validates the
complementary **discrete / jump term**: sample pivotal future events forward, branch on their outcomes, and
return a *branching distribution* + the *pivotal fork* — so we model past the near-horizon without collapsing
to a false-confident point and without abstaining.

## What was built

- **`swm/transition/future_events.py`** — the event model. `FutureEvent` (a dated event with a categorical
  outcome distribution; each outcome either JUMPS the belief via `impact` or RESOLVES the question via
  `resolves`), `SurpriseHazard` (a Poisson base-rate of unscheduled mean-zero shocks — the unknown-unknowns
  floor), `EventCalendar` (known events + hazard over a horizon). `events_from_records` for structured/test
  input; `EventImpactJudge` wraps a pluggable `judge_fn` (the EXP-030 channel) to author the calendar from a
  retrieved timeline in production.
- **`swm/simulation/branching_rollout.py`** — the engine. `BranchingRollout` Monte-Carlos K trajectories:
  between events the belief evolves under a **pluggable `continuous_step`** (a martingale by default — *the
  calibrated transition operator drops in here*, which is what keeps this from colliding with that work), and
  at each dated event it samples the discrete outcome, applies the jump (or resolves and stops), and records
  the branch. `pivotal_branches` reads the per-trajectory branch record and returns, for each event, the
  conditional outcome per branch + the event's share of total variance (η² = *what watching it would
  resolve*). `forward_forecast` is the front object: distribution + pivotal forks + a reducible/irreducible
  split that **replaces abstention** — past the horizon it still returns the full branching distribution and
  says which forks are reducible (watchable) vs irreducible (the surprise + resolution floor).
- **Wired into `swm/api/world_model.py`** — `WorldModel.simulate(question, events=…)` now dispatches to
  `simulate_forward`, returning the branching distribution + pivotal branches + (optional) **best action**
  by P(desired), never a bare point.

**Composition contract (so it does NOT double-build the transition operator):** this engine owns the
discrete jumps + branching; the `continuous_step` you pass owns the between-event drift/volatility. Pass a
calibrated transition operator there and the two become one jump-diffusion. A test asserts the pluggable
continuous step composes.

## Results (no-cheat generative validation; `python -m experiments.exp083_branching_realities`)

Generative truth (a "will the Fed cut?" shape): each instance has b0~U(0.42,0.58); a pivotal CPI print forks
it (hot −0.25 / cool +0.25, 50/50); the decision then resolves ~Bernoulli(post-CPI belief). The model is
given only b0 + the event *structure*, never the realized outcome. 1,500 instances.

| Test | Result | Reading |
|---|---|---|
| **1. Blind marginal** (pre-event) | Brier skill vs persistence **−0.001** | ties the martingale — **never fakes a point** (EXP-033 respected) |
| **2. Conditional** (once CPI resolves) | Brier **0.185 vs 0.245 = +0.245 skill**; log-loss **+0.188** | the payoff of modeling the event: the fork the marginal can't express |
| **3. Pivotal recovery** | CPI named top pivot **100%**; P(yes\|hot)=**0.239**, P(yes\|cool)=**0.751** (true ≈ b0∓0.25) | the decomposition recovers the real fork |
| **4. Best-action** | picks `push_cool` at P(yes)=0.655, **+0.159** over do-nothing | evaluates a `do()` persistence *cannot* |
| **5. Honest negative** | no events → Brier skill **0.000** | collapses to persistence — no spurious lift |

## Verdict

- **The marginal is honest and the conditional is the win.** On the blind marginal an efficient belief is a
  martingale, so branching ties persistence (−0.001) — it does not manufacture confidence. The value shows
  up the moment a pivotal event resolves: +0.245 Brier skill, because branching is the only model that knows
  what the event *means* for the answer. This is exactly "model forward through the events."
- **It models past the horizon as a branching distribution, not a point.** The pivotal-branch decomposition
  returns "25% if hot, 80% if cool, and here's P(each)" — the object a smooth diffusion structurally cannot
  produce — and the reducible/irreducible split replaces abstention with an honest "here's the fork, here's
  the floor." (For the Fed shape, CPI resolves ~25% of the outcome variance; the FOMC coin-flip resolution is
  the irreducible remainder — the honest split.)
- **It unlocks best-action on forward questions**: a `do()` that shifts an event's odds is scored by rolling
  the world forward under it — something persistence/nowcast cannot evaluate at all.

**Relationship to the other event/dynamics work (the "is this already built?" question):** there are now
three forward-model pieces and they are genuinely distinct, composing rather than colliding — (1) EXP-035
`swm/transition/event_model.py`: continuous heteroskedastic diffusion; (2) EXP-077 `swm/simulation/event_model.py`:
calibrated continuous-Gaussian-JUMP variance placement (interval coverage — 82% vs persistence's 3%); (3) this,
EXP-083: the *discrete categorical* branching + pivotal decomposition + conditional forecast + forward
best-action — the multimodal fork ("hold/cut") that a summed-Gaussian jump model structurally cannot produce.
(1)/(2) and the calibrated transition operator are the *continuous* term and drop into this engine's pluggable
`continuous_step` seam; this is the *discrete* term. Together: one jump-diffusion. EXP-077's calibrated impact
distributions can parameterize this model's branch impacts — the named unification.

**Honest boundary / next step:** validated here on a transparent generative process (as EXP-063's bracket and
EXP-053's cascade were). The named next step is a real forward calendar (FOMC/election dates + an LLM
`EventImpactJudge` authoring the impacts from retrieved news) scored no-cheat on resolved outcomes — the
event→impact authoring is the LLM's job and the piece that then needs its own calibration.

Tests: `tests/test_future_events.py` (9), `tests/test_branching_rollout.py` (7). Full suite 365 green.
