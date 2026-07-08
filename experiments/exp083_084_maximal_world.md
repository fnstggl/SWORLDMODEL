# EXP-083/084 — The Maximal World: does modeling every person as an agent actually beat the shortcut?

You asked us to *rebuild the world itself as a calibrated simulation* — model every relevant person as an
agent with their desires/predisposition/mood/situation, put them on a shared social plane, and roll it
forward with real time-physics to see what happens. You said to run it **as an experiment, fully.** So we
did — as a falsifiable, no-cheat head-to-head in the two regimes you named: a **cascade/tipping-point**
question and a **best-launch-messaging** question. In both, the "maximal world" (agents with full context)
is scored against the calibrated *shortcut* (the compact model that does **not** model individuals) and the
dumb baselines, on data whose answer we already know.

The result is consistent and, I think, genuinely important: **the maximal world does NOT beat the compact
calibrated shortcut in either regime.** In the cascade it recovers the real tipping signal (it beats the dumb
baselines) but not one bit more than the two-parameter mean-field already had; for the individual-target
messaging question it does *worse* than simply scoring each message on its own. World-completeness is not the
lever we were missing — identifiability and irreducible noise set the ceiling, not how much world you simulate.

## Part A — the cascade (baby-name fashion tipping, 481 names, forecast 10 years out, leakage-free)

Every person modeled as an **agent with their own adoption threshold** (Granovetter: some jump early, some
need everyone else first) and their own **fashion-fatigue** hazard, all seeing the shared prevalence (the
social plane), rolled forward — the S-curve-then-crash *emerges* from the agent distribution. Its ablation
is the **mean-field shortcut** (the EXP-072 coupled ODE: two numbers, `g = ρg − λp`, the whole population
collapsed to one prevalence). Both fit their parameters on the same train names, scored on the same held-out
names.

| regime | persistence | trend | **mean-field shortcut** | **agent world** | winner |
|---|---|---|---|---|---|
| all test points | 0.140 | 0.136 | **0.123** | 0.144 | mean-field |
| **turning points** (near peak) | 0.264 | 0.570 | **0.152** | 0.182 | mean-field |
| rising | 0.441 | 0.630 | 0.498 | 0.514 | persistence |
| stable | 0.107 | 0.081 | 0.085 | 0.105 | trend |

(MAE, lower is better.) Read it honestly:

- **The agent world is real, not a strawman.** At the turning points — where persistence and trend are
  structurally wrong — the agent world **beats the simple baselines by +31%** (0.182 vs 0.264). The coupled
  cascade dynamics it models genuinely exist.
- **But it loses to the mean-field shortcut** (0.152 vs 0.182, −20% at the turning points; worse overall).
  Modeling every individual recovered *the same tipping signal the two-parameter ODE already had*, at vastly
  higher complexity — and slightly worse, because the extra machinery adds estimation variance.

**Why — and this is the general law:** for an **aggregate observable** (a population share), the individual
threshold distribution is *unidentifiable*. You only ever see the sum, and infinitely many agent
populations produce the same sum, so the agent detail cannot be fit from the data — it collapses to the
mean-field. Individual-level modeling can only pay when you have, and are predicting, individual-level data.
That sets up Part B.

## Part B — best-launch-messaging (CMV: which reply changes THIS person's mind, 200 held-out targets)

Here the target is a single **individual**, so agent modeling is identifiable — the regime where it should
win if anywhere. The **agent world** models the target first (one call builds a structured persona — their
values, the crux of their view, what moves them vs bounces off), then, *as that person*, rates each candidate
reply. Its comparison is the **shortcut**: score each reply by the LLM's one-shot judgment, ignoring who the
target is. Precision@1 (top-ranked reply is a real delta winner) on the same 200 held-out OPs as EXP-076:

| approach | precision@1 |
|---|---|
| **shortcut — LLM one-shot, each reply scored INDEPENDENTLY** | **0.640** |
| control — same LLM, all replies scored JOINTLY, no persona | 0.495 |
| **agent world — model the person, then score jointly** | 0.440 |
| random pick | 0.512 |

The agent world **loses** — it lands *below random*, 0.20 under the shortcut. And the controlled decomposition
says exactly why, in two parts:

- **Modeling the person added nothing** — worse than the no-persona control by 0.055. Even a rich, sensible
  model of the target (the personas read as accurate) did not recover any signal about which argument would
  actually flip them. The deciding factor (their state of mind that day) isn't in the text, so no model of
  the person — however detailed — can read it. This is the ~0.64 irreducible ceiling again, from the other side.
- **A bonus, actionable finding:** the real damage (0.640 → 0.495) is the **joint-comparative framing**.
  Asking the LLM to score all candidates *together* is far worse than scoring each *independently* — the
  comparison biases it toward the more elaborate/confrontational reply, which is not what wins deltas. For
  the product this is concrete: **score each candidate message on its own; never ask the model to rank them
  head-to-head.**

## What this answers, directly

**"Should we model every person as an agent with all their context?"** — Only where it's identifiable and
you're predicting an individual, and even then only up to the irreducible ceiling. For population/aggregate
questions (adoption shares, election margins, click rates) the compact calibrated model is *better*, not
just cheaper — the agent detail washes out in the sum and only adds noise. This is the same lesson EXP-072
taught (calibration beats completeness), now demonstrated by building the maximal thing and watching it lose.

**"Are we not modeling the whole world yet — is that the ceiling?"** — No. We built the maximal world and it
did **not** break the ceiling in either regime. The ceiling is set by (a) what's *identifiable* from the data
you have and (b) irreducible noise — not by how much world you simulate. A hyper-realistic sim still can't
read a fact that was never recorded (which name a toddler gets; the mood someone's in when they read your
message).

**"Open-source world models / a social plane / just Python?"** — The social plane is the right abstraction
and we have it (`swm/world/substrate.py`: entities on one clock, couplings, `without_couplings()` ablation).
It's pure Python; there is no pretrained "social world model" worth connecting to (the famous ones model
pixels/robotics). The maximal world **is** the composition of calibrated mechanisms + agents — we build it,
and — crucially — we now have measured *when it earns its place and when it doesn't.*

## Where the maximal world DOES pay (the honest yes)

It's not "never." Model individuals as agents when: (1) the target is an individual or a small group, (2) you
have individual-level signal to identify them, and (3) interaction between them drives the outcome (small
committees, negotiations, a specific person's decision). That's exactly the regime `AgentSociety` +
`substrate.py` are for. For the big aggregate forecasts, route to the calibrated compact model — which is
what the EXP-078 regime router already does. The experiment turned "model everything" from a belief into a
*decision rule*: **simulate individuals where they're identifiable and interacting; use the calibrated
shortcut everywhere else.**

## Reproducibility

- `experiments/exp083_maximal_world_cascade.py` (Part A), `exp084_maximal_world_messaging.py` (Part B).
- Part A reuses the EXP-072 baby-name samples (leakage-free, fit-on-train). Part B reuses the EXP-076 200-OP
  held-out split; the agent simulation is cached to `experiments/results/exp084/agent_sim.json` (resumable,
  `DEEPSEEK_API_KEY` from env only, never committed).
