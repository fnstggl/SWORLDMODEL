# EXP-066 — can the LLM pick the right *rate* for a novel quantity? (the open measurement)

The last untested link in the compiler. Mechanism choice is robust (EXP-065: 15/15) and the engine + a
*correct* rate is calibrated (EXP-065: 80% coverage). This closes the loop: can the LLM supply the rate —
the per-unit-time **volatility** the diffusion needs — on its own? This is the most direct test of the
project's core bet, *"the inferences will be accurate enough."*

**Ground truth.** For 15 GSS attitude topics we measured the real year-to-year opinion **volatility** (σ,
pp/yr) from decades of data — they span **1.5–4.9 pp/yr** and contain a deliberate trap: the *noisiest*
attitudes are the reactive "government spending" items, **not** the famously fast-*drifting* moral issues
(whose year-to-year volatility is only moderate). So the benchmark separates a model that understands
**volatility** from one that confuses it with **cumulative drift**.

**Method.** An external model (Qwen-2.5-72B via HF) was asked, **blind** (plain-language topic, no data), in
one batched call, for each topic's typical annual opinion shift. Graded three ways.

---

## Results

### 1. Absolute scale — **excellent**
- geometric-mean ratio LLM/data = **0.96**, **100% of topics within 2×**.
- The LLM knows opinion moves **~3 pp/yr**. Its very first blind answer for opinion-in-general was "3",
  against a measured mean of ~2.9. It has the right internal prior for the *scale* of the rate.

### 2. Discrimination (which topics are more/less volatile) — **weak**
- Spearman rank correlation LLM vs data = **+0.39** — some signal, but it **cannot reliably rank** volatile
  vs stable. It falls into the trap: it rated `grass` (marijuana, big *drift*) the **most** volatile (4.0)
  though its year-to-year σ is only 2.9, and rated `gunlaw` (σ 1.8, one of the **most stable**) as 3.5. It
  partly conflates "changed a lot over decades" with "bounces a lot per year."

### 3. Downstream calibration — **the part that matters, and it's fine**
Coverage of the 80% forward interval when the rate comes from each source (nominal 0.80):

| rate source | coverage |
|---|---|
| the LLM's per-topic rates | **0.798** |
| the true data rates | 0.803 |
| a single global prior (2.9 everywhere) | 0.818 |

**Forecasts built from the LLM's rates are as calibrated (0.798) as forecasts built from the true rates
(0.803).** Because the LLM nails the *scale* and its errors stay within 2×, its weak per-topic
discrimination barely dents calibration — a single good global prior does about as well. For calibrated
forecasting, **the LLM's rate choice is good enough.**

### 4. Full autonomous spec compilation — **runs, but needs a validator**
Asked to emit a *complete* structural spec for "Will US CPI inflation be above 3% at year-end?", Qwen
produced a structurally sound spec — right mechanism (`generic_scm`), sensible current value (4.2%),
volatility, bounds, and outcome — but a **buggy equation**: its stated intent (per its own rationale) was
mean-reversion toward ~3%, yet it wrote `0.01*(100 - CPI) - 0.02*(CPI - 3)`, whose equilibrium is ~35%, so
inflation saturates the upper bound and the forecast degenerates to **P = 1.0**. *Right intent, wrong
formula.* Autonomous end-to-end spec authoring is error-prone and needs a **validation/repair loop**.

---

## What this means (the direct answer to "are the inferences good enough?")

| inference | verdict |
|---|---|
| **mechanism** | ✅ robust (15/15) |
| **rate — scale** | ✅ excellent (ratio 0.96, downstream coverage 0.798 ≈ 0.803 true) |
| **rate — per-topic ranking** | ⚠️ weak (Spearman 0.39) — but it **doesn't hurt calibration** |
| **full equation/spec authoring** | ❌ error-prone (unit/equilibrium bugs) — needs a spec validator |

**The bottom line.** For everything that drives calibrated forecasting — the mechanism and the *scale* of
the rate — the LLM's inferences are good enough, and this is now measured, not asserted. The failure is
narrow and concrete: the LLM can't finely rank which quantities are more volatile (and that barely
matters), and it makes numeric bugs when writing full equations unsupervised (and that matters, but is
fixable with a validation loop). The bet largely holds; the next build is a **spec validator/repair pass**
(equilibrium-in-bounds, no-saturation, dimensional sanity — simulate-and-check before trusting), which
would have caught the inflation bug automatically.

---

## Reproduce

`HF_TOKEN=… python -m experiments.exp066_llm_rate_choice` (token from env only, never stored). Committed
caches (`experiments/results/exp066/qwen_rates.json`, `qwen_fullspec.json`) make it rerun offline.
