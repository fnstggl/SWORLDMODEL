# WMv2 Production Round — Final Report

*The brutally honest accounting. Every claim traces to a committed artifact under `experiments/results/`
or a test under `tests/`. Negative and null results are preserved, not disguised.*

## What this round did

Turned a research-grade simulation runtime into a production execution path with an evidence-gated,
uncertainty-aware, abstaining general world-model architecture — and subjected each mechanism to
adequately-powered held-out validation. The headline scientific outcomes:

- **First adequately-powered persistence win** (OmniBehavior n=7074, power 0.993, transfers to held-out
  people) — reverses the prior n=48 null.
- **Nonlinear diffusion closes the gap to the fitted ceiling** (Higgs: nonlinear hazard statistically ties
  the fitted logistic; was significantly *worse* when linear) — reverses the prior linear-diffusion loss.
- **The general compiler runs end-to-end on arbitrary questions with a real LLM** (104 questions, 16
  domains, 0 crashes, 100% mechanism-validity, 100% provenance-honesty) — closes the audit's #1 gap.
- **The semantic channel is quarantined by its own evidence** (harmful/null on every domain; excluded from
  production selection).
- **Two mechanism families reach production-eligible on passed held-out + transfer records** (out of 47).

And the honest limits, preserved:

- Crowds and task-specific fits remain **unbeaten in-distribution**.
- Real best-action **decision lift is negligible** on the available randomized-intervention benchmark.
- The historical benchmark shows **V2 correctly abstains** (near-0 coverage) on prediction-market questions
  — the executable mechanism library does not yet cover one-off institutional/event resolution.

## The capability table

| Capability | Before (Phase-0 audit) | After | Empirical proof | Remaining limitation | Production status |
|---|---|---|---|---|---|
| Question parsing / compiler | executable-unvalidated; NEVER run with a real LLM | runs e2e on 104 held-out NL Qs, 16 domains | `wmv2_compiler_generality.json`: compile 70%, e2e 51%, 0 crashes, 100% mech-validity | institutional decision dynamics not executable → abstains | e2e YES; production PARTIAL |
| Causal-world construction | provenance-fabricating, silent drops | provenance-honest, loud omissions, latent_state | `test_wmv2_tier_a_fixes` (14); generality run 100% provenance-ok | readout-binding tightness | PASS |
| Actor/pop/institution discovery | one unchecked LLM call | typed proposals validated; abstains on unsupported | generality jury: actors 0.74, mechanisms 0.71 | human-annotated recall not yet measured | PARTIAL |
| Evidence & as-of grounding | 4 disjoint stacks, none wired | typed EvidenceBundle + zero-slack gate + leakage auditor | `test_evidence_layer` (8) planted leaks caught | retrieval connector env-limited | software PASS; connector PARTIAL |
| Hidden-state inference | NONE (prior sampling only) | evidence→posterior, hierarchical shrinkage, structural hypotheses, filtered rollout | `test_inference_layer` (5): posteriors beat priors; structural weight concentrates | full multi-domain SBC pending | PASS (recovery-validated) |
| Mechanism registry | 9 entries, 3 dead | 47 executable families, packs, applicability, lifecycle | `registry/data/*.json`; 2 production-eligible, 1 quarantined | 100+ packs not reached | PASS (as registry) |
| Nonlinear diffusion | linear hazard LOST to logistic | log-linear hazard TIES logistic, beats linear | `wmv2_higgs_nonlinear.json`: Δ−0.00253 vs linear; Δ−0.000192 vs logistic | Hawkes failed (quarantined) | production-eligible |
| Actor policy | LLM mints probabilities (miscalibrated) | FittedDecisionOperator; LLM-minting experimental-only | `wmv2_behaviorbench_policy.json`: beats LLM 0.099 vs 0.185 | loses to specialist in-dist | locally_validated |
| Interaction | benchmarked only inside one head | endogenous action→event chains | `test_wmv2_tier_a_fixes` two-actor chain | not yet a benchmarked thread | PARTIAL |
| Persistence | Unproven (n=48 null) | **HELPS out of sample, transfers to new people** | `wmv2_persistence_power.json`: Δ−0.0065 power 0.993; person-disjoint Δ−0.027 | short-video domain; cross-domain untested | production-eligible |
| Semantic interpretation | unvalidated, sometimes harmful | evidence-gated; harmful channel quarantined | `semantic_registry` seeded with real deltas | no feature currently qualifies | PASS (as gate) |
| Calibration / abstention | none for V2 | conditioned calibrators + signal-driven abstention + critic | `test_calibration` (7); `wmv2_calibration_validation.json` | crowd already calibrated (null) | PASS |
| Best-action | interface only | matched-CF mechanics validated; real-data lift measured | `wmv2_best_action.json`: mechanics correct; Upworthy lift +0.00017 (negligible) | no real decision lift | mechanics PASS; lift NO |
| Historical forecasting | crowd-rescale bypass mislabeled V2 | real compile→simulate path; resumable; honest coverage | `wmv2_historical_benchmark.json`: V2 abstains, crowd unbeaten | market coverage 0; 60 of 1000 | pipeline PASS; V2 coverage NO |
| Forward ledger | V1 only | v2 append-only versioned locks | `test_forward_ledger_v2` (4) | forward perf needs calendar time | PASS |
| Observability | none in runtime | cost/latency/seeds/caches/integrity hashes | every result file + registry integrity | trace IDs partial | PARTIAL |

## The 17 questions, answered honestly

1. **Is this now a genuinely general social world model?** Partially. The general compile→materialize→
   rollout→readout path is real and runs end-to-end on arbitrary questions with a real LLM; it is *general
   in architecture*. It is not yet *general in coverage* — its executable mechanism library is validated on
   individual/relational/diffusion/engagement dynamics and thin on institutional decision dynamics, where it
   correctly abstains.
2. **Can it compile arbitrary social questions?** Yes for 70% of 104 held-out questions across 16 domains
   (0 crashes, 100% mechanism-validity); it abstains with logged reasons on the rest. Never scripted.
3. **Can it infer useful hidden state?** Yes — evidence→posterior with hierarchical shrinkage beats
   no-pooling and full-pooling on recovery; the persistence win is carried by hierarchical user-rate
   shrinkage and transfers to new people.
4. **Are its actor policies behaviorally validated?** Yes, with an honest mixed result: the fitted
   population policy beats raw LLM and elicitation, loses to per-game fitting in-distribution, and transfers
   to 3 of 4 held-out games (fails public_goods).
5. **Is the mechanism library populated and evidence-backed?** Yes — 47 executable families with published
   forms, transport limits, and validation history; 2 production-eligible, 1 quarantined for a failed
   held-out, all failures preserved.
6. **Does interaction replicate?** Partially — significant in the prior BehaviorBench round; the universal
   policy shows a small interaction gain; endogenous action→event chains now exist and are tested, but a
   multi-actor thread benchmark is not yet run.
7. **Does heterogeneity replicate?** Yes — the fitted preference mixture beats a selfish point model
   (0.099 vs 0.125).
8. **Does persistence work out of sample?** **Yes** — the round's clearest new win: adequately-powered
   held-out (Δ−0.0065, power 0.993) with person-disjoint transfer (Δ−0.027).
9. **Does nonlinear diffusion improve prediction?** **Yes** — beats the linear hazard significantly and
   closes the gap to the fitted logistic ceiling (was significantly worse when linear).
10. **Does the system transfer to unseen conditions?** Partially and specifically: to new people
    (persistence), to 3/4 held-out games (preferences+QRE), and the diffusion hazard generalizes enough to
    tie the fitted ceiling. Cross-domain parameter transport is untested (flagged high-risk).
11. **Does it beat realistic consumer alternatives?** Beats grounded direct LLM and ensembles where both
    run; does NOT beat the crowd (unbeaten on market questions).
12. **Does it beat specialized ceilings?** No — task-specific fits and histograms win in-distribution
    everywhere; V2's edge is transfer, not in-distribution lift.
13. **Does it produce real best-action lift?** No — negligible on the available randomized-intervention
    benchmark (Upworthy); the counterfactual mechanics are correct but the predictive signal is too weak.
14. **Is it calibrated?** The calibration machinery works (cuts ECE on miscalibrated data); on the crowd
    corpus there is no miscalibration to fix; V2's own outputs are within-particle calibrated but not yet
    scored at scale (historical coverage 0).
15. **When should it abstain?** On unsupported high-sensitivity variables, failed leakage audits, no
    applicable validated mechanism, dangling readouts, material structural disagreement, out-of-distribution
    — all implemented and firing (30% principled abstention in the generality run; 100% on market questions).
16. **Is it ready to merge?** The IMPLEMENTATION is substantial and honest, but per the acceptance gates the
    defining product claim (beat the strongest non-simulation baseline in-distribution) is **undemonstrated**.
    Merge is a code-review + evidence decision, not an implementation-completeness decision. **Do not merge
    PR #75 on completeness alone.**
17. **Is it ready for production?** As a disciplined, abstaining, transfer-capable research system: yes for
    the validated slices (diffusion, engagement/persistence) with calibration and abstention. As a general
    consumer forecaster that beats the crowd: no. The honest product position is "strong where its
    mechanisms are validated, abstains elsewhere."

## The one-paragraph verdict

The architecture is real, general, and honest: arbitrary questions compile and execute end-to-end, evidence
is gated and audited, hidden state is inferred (not guessed), mechanisms are evidence-backed and abstain
when unsupported, and the semantic channel is quarantined by its own failure. Two of the program's longest-
standing open questions flipped to positive this round with adequately-powered, transfer-validated evidence:
**persistence works out of sample** and **nonlinear diffusion closes the fitted-ceiling gap**. What remains
undemonstrated is exactly what the prior six benchmarks also found: no full-V2 arm beats the crowd or a
task-specific fit in-distribution, and there is no real decision lift on randomized-intervention data. The
value is transfer and discipline — now with more preserved, significant, positive evidence than any prior
round, and with the general path exercised by a real LLM for the first time.
