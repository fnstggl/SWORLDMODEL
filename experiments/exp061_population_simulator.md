# EXP-061 — Level 3: large-scale demographic simulation, and the honest coupling KPI

**Level 3 of the framework, built general** (not election-specific) and measured with the *right* KPI. The
brief was explicit: only worth it *after we show the coupling changes the answer — otherwise it's a fancy
poll average.* So this experiment's job is to measure that honestly, and it does.

---

## First: the KPI. Is log-loss the best measure here? No.

Log-loss is a proper score for a probabilistic **binary label**. A large-scale demographic outcome is a
**continuous population share** (support %, adoption %, vote share). Leaning on log-loss has three problems:
it scores a probability against a 0/1 label when our target is a proportion; it conflates **calibration**
and **sharpness**; and it does not isolate *the* question — does the coupling beat the marginal composite?

`swm/eval/population_metrics.py` leads with three things log-loss can't give:

| KPI | what it answers |
|---|---|
| **share-RMSE / MAE** | accuracy on the proportion (the thing we predict) |
| **coupling skill** = 1 − RMSE(coupled)/RMSE(marginal) | **the decisive number**: does interaction beat the poll average? |
| **interval coverage** | real calibration for a continuous outcome (do 80% intervals contain truth 80% of the time?) |

Winner-accuracy / Brier are kept as *secondary* (the "who wins" framing).

---

## What was built (general — an election is just one instance)

- **`swm/simulation/population_simulator.py`** — `PopulationSimulator`: real demographic **cells** (each a
  mean-field agent: belief, responsiveness, influence = size × turnout) → two coupled channels forward →
  a pluggable **aggregator**.
  - **Opinion coupling** (mean-field): conformity + bandwagon/social-proof.
  - **Participation coupling** (new): a cell's **turnout is not fixed** — enthusiasm rises when the
    aggregate moves its way (mobilization) and falls when it's losing. Differential, stance-coupled
    turnout is the general analogue of turnout surges, viral adoption, and protest cascades.
  - **Aggregators**: `share_aggregator` (participation-weighted share — the general default);
    `winner_take_all_aggregator` (regional majority roll-up — the electoral-college *shape*, fully
    general). "Who wins the election" = the general machinery + this one aggregator.

Validated on the **General Social Survey** — real attitudes, hundreds of demographic cells (age × degree ×
party × region), **15 topics × ~30 years** — a *general* large-scale-demographic benchmark, not elections.

---

## A. Does coupling beat the marginal poll average? (1,927 real predictions)

For each (topic, as-of year A → target year T), build real cells at A and predict the population share at T.

| | marginal (poll average) | coupled |
|---|---|---|
| **share-RMSE** | **0.095** | 0.109 |
| coupling skill | — | **−0.15** |
| interval coverage (nom. 0.80) | — | 0.72 |

**The honest, decisive result — and it's precise:**

| coupled model | coupling skill |
|---|---|
| pure conformity (mean-field, no bandwagon) | **−0.000** — *identical* to the poll average |
| + bandwagon / social proof | **−0.15** — *worse* than the poll average |

So on full-population opinion, coupling is **exactly a fancy poll average** (pure conformity is
mean-preserving, so it reproduces the marginal), and adding bandwagon **hurts** — because most GSS attitudes
are stable or drift for *exogenous* reasons, not endogenous bandwagon. **Coupling only earns its place when
the real process actually has the coupling.** This confirms the EXP-053 finding at scale (1,927 predictions,
15 topics) with the proper KPI, and it is the direct, honest answer to the brief's condition. (A frozen
"no change" marginal is genuinely hard to beat over a decade; the temporal-*trend* axis is a separate
lever from cross-agent *coupling*, and is reported only as a labeled reference.)

## B. Where coupling BITES: participation-weighted outcomes

The same real cells, but the outcome is participation-weighted with **real turnout differentials**
(older / more-educated participate more — published Census CPS constants) plus stance-coupled
**mobilization**:

| topic | raw marginal | + real turnout | + mobilization | outcome |
|---|---|---|---|---|
| gunlaw | 0.790 | +0.002 | **+0.043** | 0.835 |
| homosex | 0.607 | −0.000 | **+0.042** | 0.649 |
| natenvir | 0.587 | −0.004 | **+0.039** | 0.622 |
| abany | 0.452 | +0.001 | **−0.015** | 0.438 |

Mobilization moves the outcome by 3–4 points in the direction of the enthused majority — a swing that can
flip a close decision, and that a marginal average **cannot** produce. This is the general shape of turnout
surges / adoption cascades. (GSS is a full-population survey, so it has no participation-weighted *ground
truth* to score this against — this is a mechanism demonstration; a scored win needs real turnout-weighted
data. See the honest gap below.)

## C. Aggregation layer (the electoral shape, general)

`winner_take_all_aggregator` rolls region-level majorities up over the census regions (which regions support
the position). Verified on real GSS regional outcomes — the arithmetic an election needs, general to any
"majority per region, then count regions" outcome.

---

## Tests — `tests/test_population_simulator.py` (9, all pass)

KPI (coupling-skill sign, interval coverage, scorecard, winner accuracy), aggregators (share / marginal /
winner-take-all), the simulator (marginal vs coupled; participation coupling moves the outcome), and the
Level-2 backdrop.

## The honest state, and the named gap

- **Built and general**: real cells at scale, a turnout/participation model, coupled opinion + mobilization,
  pluggable aggregators including the electoral shape. Calibrated (interval coverage 0.72 vs 0.80 nominal).
- **The decisive finding, measured honestly**: on full-population opinion, **coupling does not beat the
  marginal** — it's a fancy poll average (conformity) or worse (bandwagon). That's the truth the brief
  asked for, not a forced win.
- **Where the win lives**: participation/cascade-weighted outcomes (B). To *score* a coupling win we need a
  dataset where the outcome is genuinely participation-weighted **with ground truth** — real election
  returns + turnout by group, or an adoption panel. That is the next data acquisition, and the machinery to
  consume it is now built.
