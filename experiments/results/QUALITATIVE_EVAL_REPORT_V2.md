# Qualitative actors, production phase — evaluation report v2 (honest)

**Scope of this phase:** default-on fail-loud wiring with a mandatory routing report; the
production causal audit (ON vs OFF through the real funnel); cluster-2.0 semantic scoring;
a 50-case, 17-domain frozen decision corpus (22 posture-contradicting, 13 from 2025); the
closed-loop trajectory benchmark (two runs); an external reference calibrator fit. Backend:
DeepSeek `deepseek-v4-flash` throughout. All artifacts committed alongside this report.
Claims ladder: implemented / mechanically verified / plausible / backtested / calibrated /
statistically better than baseline / still speculative.

## 1. THE headline finding: the 2025 slice is a contamination test, and it failed

Next-action accuracy split by case era:

| Arm | pre-2025 cases (n=37) | 2025 cases (n=13) |
|---|---|---|
| A numeric (uniform prior) | 9/37 (24%) | 4/13 (31%) |
| C stateless qualitative | 25/37 (**68%**) | 2/13 (**15%**) |
| D persistent qualitative | 21/37 (57%) | 1/13 (**8%**) |

The 2025 cases (Kimmel reinstatement, Franklin firing, CBS Late Show, UVA resignation,
Harvard rejection, tariff pause, Dončić trade, Gaza ceasefire, Wiz acceptance, …) are the
corpus's closest approximation to post-training-cutoff events for this backend. On them, both
qualitative arms performed **at or below the uniform baseline**; on pre-2025 cases — almost
certainly inside the model's weights — they scored 57–68%.

Honest reading: **most of the headline LLM-arm accuracy on famous historical cases is
elicited outcome knowledge, not forward prediction.** The architecture demonstrably elicits
and channels that knowledge far better than rating-and-blending does (§2), and the earlier
persistence ablation remains contamination-controlled — but genuine forward prediction, as
measured on the freshest slice available, is currently **around chance**. n=13 carries wide
error bars, some 2025 cases may still precede the backend's cutoff, and the 2025 evidence
lines carry the curators' flagged date approximations — none of which rescues the direction
of the signal. This is precisely why the evaluation plan demanded post-cutoff cases; building
a verified post-cutoff corpus (and re-running on future events as they resolve) is now the
single most important measurement task in the project.

## 2. Full-corpus results (n=50, four arms, zero case errors)

| Arm | Top-1 | Top-2 | Log loss | Brier | Conf−acc gap | Novel rate | Calls | Wall |
|---|---|---|---|---|---|---|---|---|
| A `numeric_policy` | 26.0% | 50.0% | 1.476 | 0.770 | 0.03 | 0 | 0 | ~0s |
| B `persona_blended_numeric_policy` | 18.0% | 46.0% | 1.901 | 0.850 | 0.16 | 0 | 50 | 377s |
| C `stateless_llm_policy` | **54.0%** | 58.0% | **1.178** | 0.797 | 0.28 | 0.30 | 301 | 3763s |
| D `persistent_qualitative_llm_policy` | 44.0% | **64.0%** | 1.280 | **0.788** | 0.16 | 0.43 | 300 | 5304s |

- The numeric arm's posterior is uniform over the candidate menu in every case (no fitted
  packs exist for these actors); log loss 1.476 ≈ ln(4.4). Its top-1 is argmax tie-luck.
- The persona blend (arm B, `cc17199`) again underperforms the uniform prior on every proper
  score at n=50 — the strongest evidence yet that rate-and-blend destroys the decision signal
  the same model expresses when asked to *choose*.
- **The 9-case pilot's D>C top-1 ordering reversed at n=50** (C 54% vs D 44%), while D leads
  top-2 (64% vs 58%), Brier, and confidence-calibration (gap 0.16 vs 0.28). Interpretation,
  consistent across both: on ONE-SHOT decisions, persistent hypothesis diversity spreads mass
  (novel rate 0.43 vs 0.30 — richer, more specific chosen actions that exact-match scoring
  only partially reclaims through cluster-2.0), costing argmax accuracy while producing
  broader, better-calibrated distributions. Persistence has no time to pay off inside a
  single decision; its designed value is trajectories and evolving state (§4, §5).
- Posture split: on the 22 decisions that CONTRADICTED the actor's public posture, both
  qualitative arms scored 8/22 (36%) vs the uniform prior's 4/22 — a real but modest gain on
  the hardest class; public-posture capture remains the dominant failure mode.

## 3. The production causal audit (ON vs OFF, matched worlds, real funnel)

61 qualitative decisions, zero fallbacks, `actor_policy_report` exact
(requested = actual = persistent_qualitative, both actors routed qualitatively).

- **The decision layer is load-bearing**: on matched worlds (identical coupling draws), the
  terminal trajectory distribution shifted decisively between arms — mean cooperative
  progress 0.364 (ON) vs 0.311 (OFF), max CDF gap 0.50.
- **The consequence coupling is the next bottleneck, now measured**: the binary
  answer (threshold 0.5) stayed 0.0 in BOTH arms because the sampled action→pathway step
  (~0.04/action prior) bounds few-round scenarios to small state movement. An excellent
  simulated decision followed by a crude fixed-magnitude consequence still produced an
  unchanged binary event — exactly the predicted failure class. Fix path: fit the coupling
  pack on scored trajectories; never hand-widen.

## 4. Closed-loop trajectory benchmark (3 real sequences, 11 steps, 6 branches, two runs)

| Metric | D persistent | C stateless |
|---|---|---|
| Step accuracy (run 1 / run 2) | 52.8% / 44.4% | 52.8% / 61.1% |
| Full-sequence accuracy | 0.0 (all runs) | 0.0 (all runs) |
| Reaction-prediction accuracy (scorable subset) | 43% / 55% | 36% / 60% |
| Hypothesis switches per (branch, actor) | 0 | — |

- **No measurable persistence advantage on trajectories at this scale** — the two runs
  bracket each other and run-to-run variance (±8–17pp) dominates. An honest
  inconclusive-to-negative result.
- **Full-sequence accuracy is zero everywhere**: no branch reproduced an entire real 3–4 step
  sequence. Error compounding is real and unsolved.
- Reaction anticipation scored well only on the Cuba sequence (75–82%) and at zero on the
  corporate/political sequences, with 56–74% of anticipations unscorable by the
  deterministic resolver — reaction measurement needs the LLM-mapper treatment next.
- Persistence itself is structurally verified (state revisions grow; a branch's actor never
  switches hypothesis).

## 5. What is now mechanically guaranteed (implemented + tested, 1218 tests green)

- **Default-on**: every run REQUESTS `hybrid_relevant_actor_policy`; `numeric_policy` exists
  only as an explicit benchmark/ablation request, the Tier-3 routine policy, or a loudly
  reported degradation. Silent bypass is structurally impossible: every result carries
  `actor_policy_report` (requested vs actual mode, per-actor routing, every fallback with its
  reason); construction failures degrade LOUDLY (warning + limitation + support-grade cap);
  Tier-1 LLM failures try configured fallback model families before the marked numeric
  fallback.
- **Universal human routes**: compiled worlds route consequential humans through the causal
  selector (authority/veto/principals/stances/reaction-is-the-question, reasons recorded,
  dynamic promotion); personal questions with supplied context
  (`user_context["individual"]`) run the same architecture directly from `simulate_world`,
  the target automatically Tier 1. Mechanical/non-human routes never spend cognition.
- **cluster-2.0**: identity normalization, ontology-anchor and conservative LLM-assisted
  novel→candidate mapping (maps only ONTO candidates), meaningful modifiers preserved
  (`accept[private]` ≠ `accept`), per-row method + human-auditable explanation.

## 6. Calibration

`fit_actor_calibration.py` fit a single reference-class temperature on arm D's counted
distributions (LOO grid): **t = 0.9**, a negligible correction (LOO NLL 2.606 vs 2.607 raw) —
counted branch frequencies were already temperature-neutral. The reliability table shows the
distributions remain overconfident at the top bins on this corpus. The fitted pack ships as
`experiments/actor_decision_calibration.json` (level: reference; fit provenance and caveats
inside); actor-/role-/domain-level calibration remains impossible until far more per-key
history exists, and everything it serves stays honestly labeled.

## 7. Cost and latency

Qualitative decision: 1 call, ~10–18s. 50-case arm (K=3×2): ~300 calls, 63–88 min at 5-way
parallelism. Whole phase (audit + two trajectory runs + 50-case four arms + demos):
~1,100 DeepSeek calls. Hybrid Tier-3 routing and per-run budgets are the production cost
controls.

## 8. Verdict on the ladder

- **Implemented, mechanically verified**: the full requested architecture, fail-loud
  default-on wiring, routing transparency, semantic clustering, causal audit, trajectory
  harness, 50-case corpus, calibration fit. 1218 tests green.
- **Backtested**: yes, at n=50 single decisions + 3 trajectories — with the era split
  exposing contamination rather than hiding it.
- **Statistically better than baseline**: at eliciting decisions on in-knowledge cases
  (C/D ≫ A/B on every proper score). **Not demonstrated for forward prediction** — the
  2025 slice sits at chance; trajectory persistence advantage not demonstrated.
- **Calibrated**: reference-level only, near-neutral fit, distributions still overconfident.
- **Still speculative**: true post-cutoff forward accuracy; consequence-coupling realism;
  sequence-level prediction; model-family diversity (HF fallback family is wired but no
  second key was available this run).

## 9. Priority order next (updated by these results)

1. **A verified post-cutoff corpus** — freeze questions on pending real decisions now, score
   as they resolve; nothing else can measure forward prediction.
2. **Consequence-coupling fitting** (the audit's measured bottleneck) on scored trajectories.
3. Public-posture-capture attack: adversarial "private collapse" hypothesis lenses (the 2025
   + contradicted-posture slices are where it binds).
4. LLM-mapper scoring for reaction anticipation (56–74% currently unscorable).
5. Second model family with a real key; measure cross-family divergence.
6. Production evidence pipeline feeding actor views in benchmarks (replacing curated bullets).
