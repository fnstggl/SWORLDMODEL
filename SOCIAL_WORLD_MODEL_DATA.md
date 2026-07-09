# The grounding stack for a high-fidelity social world model

*The strategic data map — is there a "full social world model" dataset, is GDELT the closest, and is the gap
just more data? Researched across primary dataset pages + 2024–2026 academic/benchmark sources.*

## Bottom line

**There is no single "full social world model" dataset, and there structurally cannot be one.** GDELT is the
closest thing to a single high-frequency feed of "what is happening in the social world right now" — but it is
only ONE layer (the noisy *event/flow* layer). A world model factorizes into a **latent state** (the slow
"position" of each society — institutions, capacity, cleavages, beliefs, economy) plus **transitions** (the fast
"velocity" — the event stream that moves the state), and **no single collection instruments both**:

- **Event feeds** (GDELT, POLECAT, ACLED) capture *flow* at high frequency but carry almost no calibrated *state*.
- **Structural indices** (V-Dem, World Bank, Polity) capture *state* but only annually, heavily smoothed.
- **Surveys** (WVS, Gallup) capture *beliefs* but sparsely in time and expensively.
- **Markets/benchmarks** (Metaculus, FRED, prediction markets) supply *calibrated probabilities/outcomes* to
  train transitions against — but only for the narrow set of questions someone chose to ask.

**So the gap is not more GDELT — it is more LAYERS, fused and calibrated.** The event flow is the cheapest,
noisiest part; durable fidelity comes from grounding the slow latent state + calibrating transitions against
markets and gold-standard labels. This is exactly why we built the **structural-state layer** (V-Dem + World
Bank) this turn — it is the single highest-leverage addition to a GDELT-only system.

## The five layers

### 1. Event / flow — "what is happening now"
| Dataset | Scale / freq | Coverage | Accuracy | Access |
|---|---|---|---|---|
| **GDELT 2.0** | >364M events, ~1M art/day, **every 15 min** | 1979–, global, CAMEO | **Low** (actor ~54%, cat ~49%; 40–88% redundancy; cooperation drift) | **Free** — CSV, BigQuery, APIs |
| **POLECAT** (ICEWS successor) | smaller, **weekly** | ~2018– (internal ~2010–) | **High** (actor ~89%, cat 80–90%; ~half GDELT redundancy); PLOVER/NGEC | **Free** — Harvard Dataverse, Cline Center |
| **ACLED** | human-coded, **weekly** | Africa 1997–, **world real-time since 2022** | **Gold standard**, narrow (violence/protest) | Free non-commercial (OAuth API); licensed commercial |
| **UCDP GED** | fatal-violence events, **annual**(+monthly candidate) | 1989–2024, global | **Gold standard** (fatal only) | **Free** API + bulk |

GDELT and POLECAT are the **scale-vs-accuracy poles of the same idea**; ACLED + UCDP are the ground-truth anchors.
A serious event layer runs all four fused: GDELT for breadth/recency, POLECAT for clean signal, ACLED/UCDP as
gold labels. **We use GDELT today** (broadest, free, no rate limit on the bulk files).

### 2. Structural / state — the slow latent "position" (mostly free bulk)
| Dataset | Measures | Coverage | Access |
|---|---|---|---|
| **V-Dem** | ~500 indicators / ~245 indices | ~200 countries, **1789–**, annual (v16) | Free bulk; OWID CSVs live-keyless ← **we use this** |
| **World Bank WDI** | ~1,400 dev indicators | 217 economies, 1960– | **Free live API** ← **we use this** |
| **World Bank WGI** | 6 governance dims | 200+, 1996– | Free API/bulk (v2 REST currently flaky) |
| **QoG** | *pre-merged* V-Dem+WGI+WDI+… | global, long panels | **Free bulk — the fastest full state matrix** |
| **Polity5 / Freedom House / Fragile States / CoW / Penn World Table / IMF WEO** | regime, rights, fragility, war, PPP GDP, macro projections | long panels | Free bulk / APIs |

**V-Dem is the richest single state descriptor; QoG is the fastest way to a harmonized country-year state
matrix.** This layer is cheap, free, and buildable now — the highest fidelity-per-effort win.

### 3. Belief / opinion — what people think (sparse; one paid item)
WVS (free, ~64 countries wave 7), **Gallup World Poll (paid microdata)**, Afro/Latino/Eurobarometer (free
regional anchors), and **GDELT GKG tone/emotion** as the free high-frequency mood *proxy* that fills the gaps
between survey waves.

### 4. Economic / markets — best-instrumented, mostly free live-API
**FRED** (800k+ series), World Bank, IMF (WEO + Data API), OECD (SDMX) — near-complete free live macro. Plus
**calibrated outcome probabilities**: **Polymarket + Manifold expose fully public no-auth APIs** (we already use
Manifold for the backtest), Metaculus + Good Judgment for expert/community calibration.

### 5. Forecasting benchmarks & world-model efforts — where transitions get calibrated
- **ViEWS** (PRIO + Uppsala) — **the closest thing to a working social world model**: open, AI-driven,
  **monthly probabilistic conflict forecasts up to 36 months ahead** at country-month AND subnational
  0.5°-grid resolution, grounded on UCDP, with an open Prediction Challenge + MLOps pipeline. **This is our
  reference architecture** — latent state + calibrated probabilistic transitions on real data, uncertainty
  quantified. We should copy its shape.
- **ForecastBench** (2024) — dynamic, contamination-free benchmark of 1,000 auto-generated questions from real
  time series (ACLED, FRED, DBnomics, Yahoo, Wikipedia) + markets (Manifold, Metaculus, Polymarket) with human
  superforecaster comparison. **The canonical scoreboard** for a world model's forecasts.
- **DARPA/IARPA lineage**: ICEWS → POLECAT, INCAS (influence campaigns), SocialSim (multi-agent online sim),
  Mercury.
- **LLM agent-societies** (AgentSociety, Socioverse "a world model for social simulation" over 10M real users,
  GenSim, LLM Economist) — the frontier, but 2025–26 critique literature ("Too Human to Model", "Static
  Sandboxes Are Inadequate") shows they are **not yet calibrated/validated forecasters**. They explore
  mechanisms but still need grounding on layers 1–4 — **exactly the gap we are filling.**

## The recommended grounding map (latent state + transitions)
| World-model component | Ground on | Status |
|---|---|---|
| latent state — institutions/regime | **V-Dem** + QoG | ✅ **built this turn** (V-Dem via OWID) |
| latent state — governance/capacity | World Bank WGI + WDI | ✅ **WDI built** (WGI API flaky; QoG bulk is the fix) |
| latent state — economy | FRED + IMF WEO + Penn World Table | ⚠️ WDI economy built; FRED/IMF next |
| latent state — beliefs/mood | WVS + barometers + **GDELT GKG tone** | ○ GKG tone is the cheap high-freq fill |
| transition — fast events | **POLECAT + GDELT** fused | ✅ GDELT built; POLECAT is the accuracy upgrade |
| transition — ground-truth violence | ACLED + UCDP GED | ○ free anchors/validation labels |
| calibration targets | Metaculus + Polymarket + Manifold, scored via **ForecastBench** | ✅ Manifold backtest live; ForecastBench = the scoreboard to adopt |
| reference architecture | **ViEWS** pipeline | ○ copy its probabilistic, uncertainty-quantified shape |

## Biggest fidelity-per-integration-effort gains (in order)
1. **Merge V-Dem + QoG + World Bank as the standing latent state** — days of work, free, gives the model a real
   *position* instead of only *velocity*. **The single highest-leverage addition to a GDELT-only system.**
   (Started this turn.)
2. **Fuse POLECAT + GDELT and deduplicate** — biggest event-layer accuracy jump for modest effort.
3. **Anchor/validate on ACLED + UCDP; calibrate on ForecastBench + Polymarket/Manifold** — converts a
   descriptive pipeline into a *scored, calibrated forecaster*, which is what "world model" actually requires.
4. **Copy ViEWS' architecture** (probabilistic, subnational + country-month, uncertainty-quantified).

**Verdict on the founder's thesis:** the instinct that "the gap is more real-world data to ground on" is
directionally right, sharpened: the gap is not more GDELT, it is more *layers*, fused and calibrated. Everything
in layers 1–2, most of 4, and the calibration harness is **free and buildable now** — enough to stand up a real
multi-layer world model without any procurement.

## Sources
GDELT ([2.0 realtime](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/), [data](https://www.gdeltproject.org/data.html)),
POLECAT ([PLOVER/POLECAT paper](https://andrewhalterman.com/files/PLOVER_POLECAT_Halterman_Bagozzi_Beger_Schrodt_Scarborough.pdf),
[GDELT-vs-POLECAT comparison, MDPI 2026](https://www.mdpi.com/2306-5729/11/7/158)),
[ACLED API](https://acleddata.com/acled-api-documentation),
[UCDP GED v25.1](https://zenodo.org/records/17397479),
[V-Dem dataset](https://v-dem.net/data/the-v-dem-dataset/),
[World Bank WGI](https://www.worldbank.org/en/publication/worldwide-governance-indicators),
[WDI/Data360](https://data360.worldbank.org/en/dataset/WB_WDI),
[QoG datafinder](https://datafinder.qog.gu.se/dataset/wbgi),
[Fragile States Index](https://fragilestatesindex.org/indicators/),
[WVS Wave 7](https://datacatalog.ihsn.org/catalog/12312),
[Polymarket API](https://docs.polymarket.com/api-reference/introduction),
[ViEWS (PRIO)](https://www.prio.org/projects/1977) · [ViEWS Prediction Challenge, arXiv 2407.11045](https://arxiv.org/pdf/2407.11045) · [views_pipeline](https://github.com/prio-data/views_pipeline),
[ForecastBench, arXiv 2409.19839](https://ar5iv.labs.arxiv.org/html/2409.19839) · [forecastbench.org](https://www.forecastbench.org/),
[DARPA INCAS](https://www.darpa.mil/research/programs/influence-campaign-awareness-and-sensemaking),
LLM social-sim critique ([Too Human to Model, arXiv 2507.06310](https://arxiv.org/pdf/2507.06310) · [Static Sandboxes, arXiv 2510.13982](https://arxiv.org/pdf/2510.13982)).
