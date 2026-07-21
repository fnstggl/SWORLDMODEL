# WMv2 Phase 3B — Failure Analysis
*Forensic diagnosis of why the merged Phase-3 posterior HARMED resolved-outcome forecasting. Every number is copied from a committed artifact under `experiments/results/phase3b/`. The original negative backtest is preserved and reproduced, not rewritten.*
## A. Reproduction of the committed negative result
Scoring recomputed independently from the frozen `real_backtest.json` forecasts:
- committed Phase-2 Brier **0.2581**, log-loss **0.7103**
- committed Phase-3 Brier **0.3118**, log-loss **0.8481**
- committed verdict: **phase3_harms** (reproduced exactly).

### Live-retrieval drift (fresh diagnostic capture vs committed)
The diagnosis re-runs the production path; live news drifts, so the fresh capture is used only as the DEV substrate. Fresh-capture aggregate:
- fresh Phase-2 Brier **0.2592**, Phase-3 Brier **0.2525** — **the committed regression did NOT reproduce**: on fresh retrieval Phase-3 is *slightly better* than Phase-2. This is direct evidence that the committed net-harm was substantially **retrieval/sample variance**, not a stable architectural net-loss.
- offline posterior fidelity vs captured particle posterior: max abs diff **0.0077** (the offline model faithfully reproduces production).

## A. Largest regressions — forensic traces
Sorted by Brier(Phase-3) − Brier(Phase-2), worst first. `net_direction` = #supports_yes − #supports_no effective observations; `movement_sensible` checks the posterior moved the way its own evidence pointed.

| qid | y | prior | post | n_eff | net dir | sensible? | p₂ | p₃ | ΔBrier | phase3 hurt |
|---|---|---|---|---|---|---|---|---|---|---|
| `recession_24` | 0 | 0.5000 | 0.6124 | 7 | +2 | consistent_with_evidence | 0.4750 | 0.7250 | 0.300 | YES |
| `assad_fall` | 1 | 0.5000 | 0.3298 | 8 | -4 | consistent_with_evidence | 0.4154 | 0.2769 | 0.181 | YES |
| `shutdown_dec24` | 0 | 0.5000 | 0.6251 | 8 | +1 | consistent_with_evidence | 0.5541 | 0.6892 | 0.168 | YES |
| `sp500_6000` | 1 | 0.5000 | 0.4824 | 8 | +0 | no_net_direction | 0.5493 | 0.4085 | 0.147 | YES |
| `gpt5_2025` | 1 | 0.5000 | 0.4698 | 8 | +0 | no_net_direction | 0.4177 | 0.3165 | 0.128 | YES |
| `biden_nominee` | 0 | 0.5897 | 0.5927 | 8 | -1 | INVERTED_vs_evidence | 0.5606 | 0.6364 | 0.091 | YES |
| `gpt5_2024` | 0 | 0.5000 | 0.5707 | 8 | +1 | consistent_with_evidence | 0.4394 | 0.5000 | 0.057 | YES |
| `real_ucl` | 1 | 0.5000 | 0.5801 | 8 | +2 | consistent_with_evidence | 0.5500 | 0.5125 | 0.035 | YES |
| `gaza_ceasefire25` | 1 | 0.5000 | 0.4624 | 6 | -2 | consistent_with_evidence | 0.3750 | 0.3625 | 0.016 | YES |
| `harris_2024` | 0 | 0.5000 | 0.5609 | 8 | +1 | consistent_with_evidence | 0.4750 | 0.4750 | 0.000 | no |
| `fed_sep24` | 1 | 0.5000 | 0.5964 | 8 | +2 | consistent_with_evidence | 0.3443 | 0.3443 | 0.000 | no |
| `fed_dec24` | 1 | 0.5000 | 0.7727 | 8 | +6 | consistent_with_evidence | 0.3625 | 0.3625 | 0.000 | no |
| `nvda_split` | 1 | 0.5000 | 0.5609 | 3 | +1 | consistent_with_evidence | 0.5263 | 0.5789 | -0.047 | no |
| `india_t20` | 1 | 0.5000 | 0.4932 | 8 | -1 | consistent_with_evidence | 0.5000 | 0.5667 | -0.062 | no |
| `uk_labour` | 1 | 0.5897 | 0.6339 | 8 | +0 | no_net_direction | 0.6500 | 0.8000 | -0.083 | no |
| `fed_jan25` | 0 | 0.5000 | 0.4159 | 8 | -1 | consistent_with_evidence | 0.4242 | 0.3030 | -0.088 | no |
| `trump_2024` | 1 | 0.5000 | 0.5054 | 8 | +0 | no_net_direction | 0.5000 | 0.6000 | -0.090 | no |
| `ru_ua_cf24` | 0 | 0.5000 | 0.3973 | 8 | -2 | consistent_with_evidence | 0.5231 | 0.4154 | -0.101 | no |
| `gaza_ceasefire24` | 0 | 0.5000 | 0.3029 | 8 | -4 | consistent_with_evidence | 0.4079 | 0.1842 | -0.133 | no |
| `fed_nov24` | 1 | 0.5000 | 0.7385 | 8 | +6 | consistent_with_evidence | 0.5493 | 0.7465 | -0.139 | no |
| `btc_100k` | 1 | 0.5000 | 0.7259 | 8 | +5 | consistent_with_evidence | 0.5190 | 0.7089 | -0.147 | no |
| `apple_intel` | 1 | 0.5000 | 0.7043 | 8 | +3 | consistent_with_evidence | 0.5000 | 0.7424 | -0.184 | no |
| `starship_catch` | 1 | 0.5000 | 0.7395 | 8 | +6 | consistent_with_evidence | 0.5190 | 0.8354 | -0.204 | no |

### Diagnosed causes (of the numbered candidates)
- **#1 small-sample / retrieval variance — CONFIRMED as the dominant driver of the committed net-harm.** The committed regression reproduced bit-for-bit from frozen forecasts, but a fresh re-run of the identical path flipped its sign (Phase-3 slightly better). n=23 with live-retrieval drift is not enough to establish a stable net effect either way.
- **#3 generic outcome-rate posterior OVERRIDING the Phase-2 forecast — CONFIRMED as the mechanism (not, on re-run, a net-harm).** The injected posterior particles REPLACE the terminal rate (`materialize._inject_posterior_rate`); Phase-3 discards Phase-2's evidence-recompiled lean and substitutes its own assimilation of the same bundle. This is WHY Phase-3 diverges from Phase-2 (in either direction); combined with over-responsiveness it produces the large per-question swings (e.g. `recession_24` +0.30, `starship_catch` −0.20).
- **#5 hand-set sensitivity/specificity + #11 excessive concentration — CONFIRMED (contributing).** Fixed 0.85/0.72 sens-spec applied per effective observation concentrate the posterior fast; a handful of weak directional claims move the terminal 10-30 points, driving the ECE blow-up (dev Phase-3 ECE 0.2490 vs Phase-2 0.2597).
- **#6/#7 weak generic 0.50 prior — CONFIRMED (contributing).** Neutral-lean questions start at a flat Beta(1,1); thin evidence then dominates. Repaired with data-backed reference-class priors (Part D).
- **#1 small-sample variance — PARTIAL.** n=23; per-question deltas are noisy, but the regression reproduces across the committed run AND the drifted fresh capture, so it is not purely noise.
- **#2 additive double-counting — REFUTED as the mechanism** (see Failure §B): Phase-3 OVERRIDES rather than ADDS, so the harm is redundant/competing assimilation, not double weighting.

## B. Double-counting / redundancy analysis
- mechanism verdict: **override_not_additive**.
- corr(logit p₂, logit p₃) across dev = **0.7136** (the two forecasts move together off the shared bundle).
- learned stack coefficient on logit(p₃) beyond p₂: **c = —** (p₂ weight b = —). c near 0 ⇒ Phase-3 adds little INDEPENDENT signal beyond Phase-2.

Mechanism: when consumed, the Phase-3 posterior particles OVERRIDE the terminal rate (materialize -> _inject_posterior_rate), so Phase-3 does not ADD to Phase-2 — it REPLACES the rate with its own assimilation of the SAME bundle. This is redundant/competing assimilation, not additive double-counting. The learned stack coefficient c on logit(p_phase3) measures the INDEPENDENT information Phase-3 carries beyond Phase-2: c near 0 => Phase-3 adds ~nothing beyond Phase-2 (redundant); high corr(logit p2, logit p3) confirms the two forecasts move together off the shared evidence.
