# EXP-074 — the portfolio backtest: mapping WHERE fidelity wins, across six real domains, no-cheat

The general social world model's honest scoreboard is not one number — it's a MATRIX of domain × fidelity ×
skill-vs-baselines. This runs the event-backtest harness across six real domains (three downloaded this
session: elections, adoption, referenda) at low vs high fidelity, to learn *from data* where the rich
calibrated simulation beats the skeptic's free baselines and where a simple model already wins.

## The map (skill vs persistence; higher fidelity = the richer model)

| domain | kind | n | low-fidelity | high-fidelity | fidelity helps? | beats ALL baselines? |
|---|---|---|---|---|---|---|
| **gss_social** | population opinion | 79 | −0.053 | **+0.150** | ✅ | no (ties trend on some items) |
| gss_spend | population opinion | 54 | −0.003 | −0.039 | ❌ | no |
| **adoption** | diffusion (OWID, 37 techs) | 890 | +0.000 | **+0.316** | ✅ | no (ties trend mid-curve) |
| referenda | Swiss ballot measures | 352 | +0.885* | +0.914* | ✅ | yes* |
| senate | US Senate 1976–2024 | 464 | +0.000 | +0.031 | ✅ (small) | yes |
| fomc | FRED macro rate moves | 198 | +0.006 | +0.170† | ✅† | no |

\* referenda "skill vs persistence" is inflated — consecutive referenda are unrelated, so the persistence
baseline is near-random; the meaningful baseline is the base rate, which the type-conditioned model beats
only marginally (+0.885 → +0.914). Read: **base-rate-dominated, fidelity adds little.**
† fomc skill is on **log-loss**, where a calibrated probabilistic macro model beats a hard momentum call;
on direction-accuracy (EXP-071) momentum still wins. Read: **fidelity helps the *probability*, not the
directional call.**

## What the map says — the honest, general lesson

**Fidelity wins big where the outcome is the aggregate of a modelable, evolving population or a diffusion
process, and the simple baselines are weak:**
- **Population opinion (social issues): +0.150 skill** — demographic composition drives the change; the
  calibrated multi-variable population beats persistence (reproduces EXP-073 on the social subset).
- **Technology adoption: +0.316 skill** — the S-curve diffusion mechanism captures acceleration-then-
  saturation that persistence structurally misses. The single strongest fidelity win in the portfolio.

**Fidelity adds little or nothing where a strong simple baseline already captures the outcome:**
- **Elections (Senate): +0.031** — states are stable; persistence is strong, uniform-swing adds a sliver.
- **Referenda: ~base-rate** — without campaign/text features, pass-rate is dominated by the base rate.
- **FOMC: momentum-dominated** on the directional call (fidelity only sharpens the probability).
- **Spending attitudes: −0.039** — these move on *period* shocks, not demographic composition, so the
  composition model doesn't help (an honest null *within* the opinion domain).

## The rule this establishes (data, not assertion)

> Reach for the high-fidelity calibrated simulation when the outcome is the aggregate of a **modelable,
> evolving population or diffusion** and the simple baselines (persistence / momentum / base-rate / market)
> are **weak**. Where a strong simple baseline exists, the honest move is to defer to it (or blend) — the
> rich sim sharpens the probability but rarely beats the point.

This is the empirical answer to "where does modeling more variables win," across six domains — and it is
exactly the regime the digital-twin bet was built for. The next architecture step (see `ARCHITECTURE_PEAK.md`)
is to turn this map into an automatic **router** (rich sim vs baseline, per question) and to run the
**calibration harvest** that fits elasticities across all these datasets into the learned-prior registry, so
every new question's variables arrive pre-calibrated.

Data (committed caches for reproducibility): `experiments/results/exp074/{adoption,referenda,senate}.json`.
Run: `python -m experiments.exp074_portfolio_backtest`.
