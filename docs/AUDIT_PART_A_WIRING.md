# PART A — Ground-truth production wiring audit

*Method: read of the actual runtime code (imports, constructors, branches, call graphs) on
`claude/agent-engine-on-main` @ `e57a613`. No READMEs, no prior summaries. Every claim below is traceable to
a file:line in `swm/engine/`.*

## 0. The one entry point

`AgentWorldModel.simulate(question, **kw)` → `_simulate(...)` → (flywheel log wrapper). All routing lives in
`front_door.py:_simulate` (lines 87–156). The decision tree, verified:

```
_simulate(question, message, recipient, channel, evidence, as_of, binary, search_fn)
│
├─ recipient AND message ............................. → _individual          (individual.py)
├─ NOT binary AND parametric AND router.route()==parametric → parametric.simulate  (swm.api.world_model)
├─ SceneGrounder(...).ground(question, evidence) ...... dossier
│    ├─ dossier.abstain ............................... → Forecast(abstain)    [loud refusal]
│    └─ dossier.resolved ............................. → Forecast(0.97 resolved_by_evidence)
├─ binary:
│    ├─ binary_kind ∈ {contest, announcement} AND route_contests → _parametric_binary  (compiler kernel)
│    ├─ event_engine=="panel" (DEFAULT) .............. → _panel               (observer_panel.py)  ★
│    └─ else .......................................... → _society (society:event)   (society.py)
├─ any _DIFFUSION_WORDS .............................. → _diffusion           (diffusion.py)
├─ cast.process=="individual_reaction" ............... → _individual
├─ cast.process=="artifact_optimization" | artifact words → _artifacts        (actions.py)
└─ else (collective_choice / population_share) ....... → _society             (society.py)
```

★ **This is the branch the pilot ablation (44 binary deliberation questions) actually took.**

---

## 1. CRITICAL AUDIT — what did the prior "FULL" arm actually execute?

**Answer: taxonomy #3 — a role-prompted observer panel that pools INDEPENDENT forecasting probabilities. It
is NOT a society simulation.**

The pilot ran `binary=True` deliberation questions. With the default `event_engine="panel"`
(`front_door.py:49`, `137`), every one routed to `_panel` → `ObserverPanel.forecast` (`observer_panel.py:99`).
What that does, line by line:

- Builds `jobs` = `LENSES (5) × families (1 default) × reps_per_lens (2)` = **10 independent LLM calls**
  (`observer_panel.py:102–104`).
- **`common_scene = dossier.brief()`** — every forecaster reads the *same* dossier (`:105`).
  `partition_evidence=False` by default (measured negative, EXP-096), so there is **no private information**;
  all 10 see identical evidence (`:108–110`).
- Each call independently emits `{base_rate, p, why}` as a superforecaster under one reasoning lens
  (`:107–119`). **No call reads another call's output. No shared state. No rounds. No time advances.**
- Aggregation: `pool_distribution` (log-linear/geometric-mean-of-odds) over the 10 `p`s (`:126`), then a
  *confidence-tracks-evidence* shrink toward the mean base-rate anchor (`:129–149`), then `clamp_p`.
- `_panel` then applies a per-domain out-of-sample temperature and re-clamps (`front_door.py:193–195`).

Against the prompt's taxonomy:

| Property of a genuine simulation | Present in the prior FULL arm? |
|---|---|
| Actors correspond to the real process (candidates, voters) | ❌ No — the "agents" are generic forecaster *lenses* (base-rater/insider/skeptic/momentum/market-aware), not stakeholders in the question |
| Persistent state | ❌ None |
| Actions affect later states | ❌ No sequential structure at all |
| Observation of other actors | ❌ Each call is blind to the others |
| Sequential transitions / rounds | ❌ Single shot |
| Emergent outcome in native answer space | ❌ The outcome is a *pooled probability*, not an emergent state |

**Classification: #3 role-prompted observer panel (independent grounded forecasts, log-linear pooled +
base-rate shrink + per-domain temperature).** Per the prompt's own rule, we must **not** call this a "society
simulation." The +0.0095 Brier edge of FULL over one grounded call in the pilot is the value of **ensembling
10 grounded forecasts under diverse lenses + calibration** — *not* evidence for stakeholder simulation,
interaction, persistent state, or temporal rollout, none of which executed.

This is exactly why the new **B3 arm (call-matched grounded direct ensemble)** is mandatory: it pools ~10
grounded direct forecasts with **no lens roles and no base-rate-anchor machinery**, so `FULL − B3` isolates
whether the *lens diversity + calibration* adds anything over plain averaging of the same budget, and
`B3 − B1` isolates the value of ensembling itself.

### One complete execution trace — "Will Zohran Mamdani win the 2025 NYC mayoral election?" (as-of 2025-08-17, y=1)

| # | agent identity/role | evidence supplied | private vs common | forecast emitted | state read | state changed | others observed | sim. time elapsed | aggregation |
|---|---|---|---|---|---|---|---|---|---|
| — | `SceneGrounder.ground` | as-of RSS passages | — | dossier (standing, facts, relations) | — | — | — | — | — |
| 1 | lens=outside_view rep0 | `dossier.brief()` | **common** | `{base_rate, p, why}` | none | none | none | none | ↓ |
| 2 | lens=outside_view rep1 | same | common | `{…, p}` | none | none | none | none | ↓ |
| 3–4 | lens=insider ×2 | same | common | `{…, p}` | none | none | none | none | ↓ |
| 5–6 | lens=skeptic ×2 | same | common | `{…, p}` | none | none | none | none | ↓ |
| 7–8 | lens=momentum ×2 | same | common | `{…, p}` | none | none | none | none | ↓ |
| 9–10 | lens=market_aware ×2 | same | common | `{…, p}` | none | none | none | none | ↓ |
| agg | `pool_distribution` → base-rate shrink (`w=f(conf,agreement)`) → `clamp_p` → per-domain `T` → clamp | — | — | **p=0.84** | — | — | — | — | log-linear pool |

Ten independent grounded forecasts, pooled. No message passing, no persistence, no calendar. That is the
whole of the prior "FULL."

---

## 2. The society rollout (the path the benchmark NEVER touched)

`SocietyRollout.run` (`society.py:67`) — invoked only for **non-binary** `collective_choice`/`population_share`,
or binary questions if `event_engine` is switched to `"society"` (non-default). Its actual properties:

- **Actors = real cast** (candidates/segments from the grounded dossier), instantiated as diverse personas
  with rotated private evidence slices (`society.py:76–88`, `agents.py:draw_variants`). ✅ #4/#5 casting.
- **Interaction is real**: each branch runs dated rounds; the next round's `public` signal is a *sampled
  realization* of this round's aggregate poll reading + sampled persona statements (`society.py:117–125`).
  Cascades/bandwagons can emerge; different sampled histories diverge across `branches`. ✅ #5 interaction.
- **Calendar time**: rounds are real dated steps (`_dates`, `society.py:31–36`) with a cast-derived cadence;
  each round tells the persona "this advances the world to {date} (~N days of real change)". ✅ partial #8.
- **BUT persistent agent state = ❌**: `decide()` (`agents.py:91`) is **stateless per call** — it receives
  `(date, dossier, variant, private_facts, public)`. The persona object persists across rounds, but its
  `private_facts` are fixed at instantiation and it accumulates **no memory, no updated beliefs, no
  commitments, no relationship state, no action history**. The only thing that changes round-to-round is the
  shared `public` string. So this is **NOT #6 (persistent-state)** and **NOT full #8** (transitions are
  poll-reading advances, not scheduled real event opportunities — debates, endorsements, scandals).

**Honest classification of the society path: #5 interacting stakeholder simulation (partial), with real
calendar rounds but no persistent per-agent state and no real-event-opportunity transitions.** This is the
genuinely-simulation part of the system — and it has **never been graded**, because the benchmark used the
panel.

---

## 3. Pathway-by-pathway audit table

| Pathway | Entry | Outcome space | Actors / agents | Interact? | State? | Rounds | Aggregation | Calibration | Class label | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Binary social event** (deliberation) | `_panel` | binary yes/no | 10 forecaster *lenses* (not stakeholders) | ❌ | ❌ | 1 | log-linear pool + base-rate shrink | per-domain OOS T | `society:event` | **working-as-intended, but mislabeled "society"** — it's an ensemble |
| **Named-candidate collective choice** | `_society` | named options | real cast personas | ✅ | ❌ | 1–`max_rounds` dated | log-linear pool over personas×branches | class shrink | `society:collective_choice` | **partially implemented** (interaction yes, persistent state no); **unwired from the benchmark** |
| **Individual response** | `_individual` | responds / not | k latent states × reps of ONE person | ❌ | ❌ | 1 | weighted per-state mix + Laplace | class shrink | `individual:response` | **partial** — no follow-up timing, valence, objection, meeting-booked, or relationship history (Part F gaps) |
| **Best message / headline** | `_artifacts` | generated texts ranked | audience personas | (paired) | ❌ | 1 | audience-scored ranking | class shrink (ungraded) | `artifact:engagement` | **unvalidated** — no randomized-outcome grade yet |
| **Diffusion / virality** | `_diffusion` | reach distribution | archetypes on heavy-tailed graph | ✅ (cascade) | ❌ | MC worlds | Monte-Carlo cascade | ungraded | `diffusion:reach` | **unvalidated** — ships flagged, never graded on real cascades |
| **Non-human process** | `parametric.simulate` | native (price/rate/date) | — (mechanism kernel) | — | — | MC | — | main's calibration | parametric | **working** for non-human; **legacy-logistic inside** (see §4) |
| **Contest / announcement binary** | `_parametric_binary` | binary | compiler kernel (`ground=False`) | — | — | MC (n=3000) | — | `society:event` T | parametric | **partially dangerous** — announcements are institutional decisions routed to a logistic-over-invented-variables mechanism (§4) |
| **Outcome logging** | `simulate` wrapper | — | — | — | — | — | — | — | flywheel | **working but thin metadata** (§5) |
| **Resolution** | `flywheel.auto_resolve` | — | — | — | — | — | — | — | flywheel | working (cited, conservative) |
| **Calibration refit** | `flywheel.refit` | — | — | — | — | — | per-class + per-domain T | into live registry | flywheel | working |
| **Routing** | `ParadigmRouter.route` / `binary_kind` | — | — | — | — | — | — | — | router | working; misclassification risk (§4) |

---

## 4. Legacy & dangerous — residual logistic/ODE reachability for HUMAN questions

The banned pattern (LLM-invented coefficients as the social generative process) lives in
`swm/api/compiler.py:262` — *"the outcome is a logistic over the variables, where each variable carries its
value AND its elasticity `weight`."* Reachability for a **human-cognition** question:

- **Deliberation (the core human class): NOT reachable.** Binary deliberation → `_panel`; non-binary human →
  `_society`. Neither imports the compiler. ✅
- **Announcement binaries → REACHABLE** via `_parametric_binary` → `parametric_binary_p` → `CompiledModel`
  (`front_door.py:135`, `203`, `317–331`). An announcement ("will company X launch/release Y by date") is
  partly an *institutional decision* — sending it to a logistic-over-invented-variables kernel is the closest
  surviving thing to the banned pattern. Mitigation in place: `_apply_toggles(spec, ground=False)` neutralizes
  live state estimates, so it runs as base-rate + mechanism structure, not a fitted human model. **Still worth
  a guardrail**: this should be pinned by a test and reconsidered (route announcements to the panel, or to a
  base-rate-only kernel).
- **Router misclassification → REACHABLE** via the non-binary `route()==parametric` branch
  (`front_door.py:103`). If the router wrongly tags a human question as a non-human process, it hits the
  logistic compiler. Mitigation: the router is biased hard toward agents; but this is an untested failure
  mode. **Recommend a test** asserting a battery of human questions never route to `parametric`.

The large legacy tree (`swm/transition/*`, `swm/variables/bayes_logistic.py`, `swm/simulation/*`,
`swm/worlds/*`) is **NOT imported by the agent-engine front door** (verified: `swm/engine/*.py` import only
`swm.engine.*` and 6 `swm.api.*` symbols; none pull `transition`/`variables`/`odeint`). It is dead code from
the perspective of the human path — but it remains in the tree and should be either quarantined or
clearly marked non-production to prevent accidental re-wiring.

---

## 5. Flywheel metadata — what is logged vs what Part C requires

Logged today (`front_door.py:74–82`): `question, question_class, domain, mechanism, p, distribution, as_of,
resolve_by, engine_config{branches,panel_reps,event_engine}, grounding{coverage,n_passages,abstain}`.

**Missing for the forward-locked multi-arm ledger (Part C):** per-arm predictions (B0/B1/B2/B3…), evidence
hash, prompt hashes, code commit, model version/params, router explanation, n_agents, n_rounds, interaction
structure, call/token counts, estimated cost, latency, raw-vs-calibrated split, abstention reason, resolution
source. And the log must become **append-only with versioning** (never overwrite a forecast after any
prompt/model/calibration/routing/code change → write a new version row).

---

## 6. Verdict & immediate implications

1. **The pilot's "FULL" was an observer-panel ensemble, not a simulation.** The defining architectural claim
   (simulation adds value) remains **untested** — the simulation code (`_society`) never ran in the
   benchmark. All prior "society" language for the binary benchmark is retired here.
2. **The correct next experiment** isolates the ladder: `raw → grounded-one-shot → call-matched grounded
   ensemble (B3) → generic forecaster panel (B4) → independent stakeholders (B5) → interacting stakeholders
   (B6) → persistent-state (B7) → time-evolving (B8)`. B3 is the arm the pilot lacked and the one that
   decides whether *any* of the panel machinery beats plain averaging.
3. **Two guardrails to add** (tests): human questions never reach the logistic compiler; announcements
   reconsidered.
4. **`_society` needs a real persistent-state option (B7) and real-event-opportunity transitions (B8)** before
   #6/#8 can be claimed or tested — currently they do not exist and must not be described as if they do.

*Do not change architecture beyond the two guardrail tests until Part B's harness produces held-out
component-level effects.*
