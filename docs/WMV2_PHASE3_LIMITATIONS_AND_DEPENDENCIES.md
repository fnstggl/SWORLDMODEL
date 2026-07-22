# WMv2 Phase 3 — Limitations & Dependencies

*What Phase 3 does NOT yet do, what it depends on, and where it can silently mislead if trusted beyond its
evidence. Nothing here is hidden in a footnote: these are load-bearing caveats for anyone deciding whether to
rely on a posterior-conditioned forecast.*

---

## 1. The four status axes (never collapsed to "complete")

| capability | software-implemented | executes end-to-end | empirically validated | production-eligible |
|---|---|---|---|---|
| Particle posterior over outcome-rate | ✅ | ✅ (live + offline) | ✅ recovery/calibration on synthetic ground truth | ⚠️ **conditional** — validated on synthetic + demonstrated live; NOT calibrated against realized real-world outcomes yet |
| Structural posterior (likelihood, not heuristic) | ✅ | ✅ | ✅ 100% structure recovery (synthetic) | ⚠️ conditional — no real-world structural ground-truth benchmark |
| Dependence-corrected likelihood | ✅ | ✅ | ✅ largest calibration lift in the ablation | ✅ for the dedup it performs; ⚠️ depends on Phase-2 dependence groups being correct |
| Reference-class prior + transport inflation | ✅ | ✅ | ⚠️ mechanism tested; **no reference-class DATABASE wired** | ❌ not production until a real base-rate source feeds `reference_data` |
| Representation choice + anti-ornamental guard | ✅ | ✅ | ✅ representation ablation | ✅ as a selection tool; the *live* path uses the two fixed representations, not per-question selection |
| Posterior CONSUMED in execution | ✅ | ✅ (`rate_source=="posterior"`) | ✅ terminal moves deterministically | ✅ |
| ~20 claim-class observation models | ❌ (2 built) | n/a | n/a | ❌ documented extension |
| Correlated multi-latent joint state on general path | ❌ | n/a | n/a | ❌ `init_state.CorrelationRule` exists, not claim-driven |
| Learned latent representation | ❌ (candidate KIND only) | n/a | n/a | ❌ needs a trained encoder |

**Bottom line on production eligibility:** the posterior pipeline is *software-complete and executes
end-to-end on the universal path*, and is *empirically validated on synthetic ground truth*. It is **not yet
calibrated against realized real-world outcomes**, and the reference-class prior has no data source wired.
Ship it as an **exploratory/transfer-grade** posterior (which the support-grade axis already enforces), not as
a calibrated real-world probability.

---

## 2. Specific limitations

1. **No real-world outcome calibration.** Recovery/calibration are measured on a well-specified synthetic
   generator (the model's own likelihood). This proves the arithmetic is a correct Bayesian update and the
   uncertainty is honest *under the model*, but it does NOT prove the observation-model sensitivity/specificity
   constants match how real news evidence relates to real outcomes. A held-out real-world calibration study
   (forecast → later realized outcome) is the missing validation and is the top follow-up.
2. **Observation-model constants are fixed, not fitted.** `_STRENGTH_SENS_SPEC`, `_STRENGTH_DETECT_FALSE`,
   `_SOURCE_RELIABILITY`, `STRATEGIC_DISCOUNT`, `TRANSPORT_RETAINED` are hand-set broad tables (deliberately, to
   keep the LLM from minting them). They are defensible priors, not empirically calibrated rates. Fitting them
   to labeled evidence→outcome data would tighten calibration.
3. **Reference-class prior is a mechanism without a data source.** `phase3_priors` correctly widens a
   reference-class Beta by transport risk, but on the live path no connector supplies `reference_data`, so it
   falls back to the generic lean-Beta (honestly labeled). The transport-inflation math is validated; the base
   rates are not being drawn from real reference classes yet.
4. **Two representations on the live path.** The representation-choice module can select among five executable
   representations by held-out calibration, but the *production* pipeline instantiates the two chosen a priori
   (`continuous_probabilistic` rate, `discrete_structural` structure). Per-question representation selection is
   built and ablated but not yet wired into `simulate_with_posterior`.
5. **Single-pass assimilation, not a time-ordered filter.** The posterior assimilates dependence-collapsed
   observations in one pass. The windowed sequential variant (`inference_layer.run_filtered`, reweighting at
   each observation's `reported_at`) exists but is not the default general path. For questions where evidence
   ordering matters (a later report supersedes an earlier one), this is a simplification.
6. **Terminal effect vs Phase-2 recompile noise.** The clean causal proof that the posterior moves the
   terminal is the deterministic offline test (`test_posterior_moves_the_terminal_distribution`). The *live*
   `terminal_effect` (consumed vs ignored) also contains variance from the stochastic Phase-2 recompile, so it
   corroborates rather than cleanly isolates. Posterior *hash* reproducibility IS clean given fixed
   plan+bundle+tags.
7. **LLM tagging is the trust boundary.** Every number is model-free, but the qualitative tags
   (direction/strength/is_strategic/supported-hypothesis) come from the LLM. A systematically biased tagger
   biases the posterior. Reliability discounting and the strategic-statement discount mitigate, but do not
   eliminate, tagger error. Tags are validated to be enum-valued and tied to span-verified claims only.
8. **Structural hypotheses are compiler-proposed.** The competing structures come from the Phase-1 compiler
   (LLM). If the true structure is not among the proposed hypotheses, the structural posterior cannot recover
   it — it can only distribute mass over the proposed set (broad, not omniscient).

---

## 3. Dependencies

- **Phase 2 (`EvidenceBundleV2`)** — verified as-of claims, `dependence_group`, `source_type`, contradictions,
  visibility. Phase 3's dependence correction and reliability are only as good as these. A wrong dependence
  group over- or under-counts evidence.
- **Phase 1 compiler** — the plan's `outcome_lean` (prior), `structural_hypotheses` (+ priors), and the
  canonical `resolve_outcome` event the posterior is injected into.
- **`fallback.GenericOutcomeOperator`** — the tier-6/7 resolver that consumes the posterior rate. Domain
  mechanisms that resolve the outcome first are NOT overwritten (safety-net no-op), so on a fully
  domain-mechanised question the posterior rate may be superseded by a validated mechanism (by design).
- **DeepSeek V3 LLM** (semantic mapping) + **live Google News RSS / Wayback** (evidence) on the live path.
  Offline validation + the 27 unit tests need neither.
- **Python 3.11, dependency-free `swm` core.** No numpy/scipy; the particle filter, Beta sampling, grid
  posteriors, and calibration are hand-rolled and unit-tested.

---

## 4. Honest failure modes to watch in production
- A confident-looking posterior mean built from **many syndicated copies of one source** — mitigated by
  dependence correction, but only if Phase-2 grouped them.
- **Over-narrow intervals** when the observation-model reliability constants are too high for a noisy domain —
  currently broad by design; would need per-domain fitting.
- **Reference-class prior masquerading as evidence** — prevented: absent real data, the prior is labeled
  `generic_weakly_informative` with `transport_risk: high`, never `reference_class`.
- **A posterior that looks consumed but isn't** — guarded: the terminal `StateDelta` records `rate_source`, and
  a domain-mechanism no-op is labeled `already_resolved_by_domain_mechanism_noop`.
