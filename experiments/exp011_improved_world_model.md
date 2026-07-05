# EXP-011 — Improved world model (richer state, learned transitions, gating, hybrid)

Answers the iteration directive's Phase-10 questions. Numbers come from the same no-cheat HN
benchmark as `exp013_real_simulation_world_model.md` (1200 posts, temporal 840/360 split, LLM
features + predictions from an 8-agent swarm). Full slice table is in exp013; the verdicts:

- **Did the improved world model beat raw LLM + context anywhere?** Yes. On **high_context** posts
  the learned world model (GBDT 0.2560, simulation 0.2580) beats raw LLM + context (0.2852). On
  repeat_domain and Show HN the state models also lead.
- **Did the hybrid beat both?** Yes — **overall** the hybrid (0.3160) beats raw LLM (0.3200) and raw
  LLM + context (0.3229), and it wins ai_topic. The hybrid gate (`swm/worlds/hybrid.py`) trusts the
  world model when entity/domain history is deep and the LLM when it is cold.
- **Which slices favor the world model?** high_context, repeat_domain(≥5), Show HN, strong_domain
  (GBDT) — i.e. where repeated entities / evolving state exist. **Which favor the LLM?** cold_author,
  low_context, semantics_dominant — the core hypothesis, confirmed.
- **Which variables survived ablation?** The stateful ones: as-of author reputation + history depth
  and domain reputation carry the world-model tiers; the LLM-extracted `audience_fit`, `hn_native`,
  `technical_depth`, and category are the text features that lift the classifier above the old
  shallow model (old_classifier 0.3379 → learned_gbdt 0.3289).
- **Did richer state help?** Yes vs the old shallow model (old_classifier is the worst non-LLM tier);
  rolling entity history + depth is what makes the state-rich slices winnable.
- **Did learned transitions help?** Yes — the GBDT beats the logistic head (0.3289 vs 0.3379 overall;
  best-in-class on high_context 0.2560) by representing interactions the linear head cannot.
- **Did better action encoding help?** Yes — LLM-extracted structured features (fed to the world
  model, not as its probability) are what let the world-model tiers close on and beat the LLM in the
  state-rich slices.
- **Did state-sufficiency gating help?** Yes — the hybrid gate is what turns "wins on some slices,
  loses on others" into an overall win over the LLM. Standalone, neither the simulation nor the GBDT
  beats the LLM overall; gated together they do.
- **Is this now more than raw LLM + context?** **Overall, yes — via the hybrid**, and in the
  state-rich regime the world model beats the LLM standalone. On cold/semantics-dominant posts it is
  not; the honest system defers to the LLM there. So: more than raw LLM + context *as a gated system*,
  not as a standalone classifier.
- **What remains unproven?** The individual/repeat-entity claim on real behavior (blocked on private
  data); a precise multi-step degradation curve; whether the wins hold at larger n (n=360 test, ~30
  positives — the slice wins are directional, best treated as hypothesis-confirming not definitive).
- **What data is required next?** (1) private individual outcome logs (email/CRM) to validate the
  individual simulation and open the highest-value regime; (2) deeper per-author HN history for
  sharper state-rich slices; (3) per-reaction/exposure data to fit the simulation's segment dynamics
  directly rather than by final-outcome matching.

**Honest headline:** richer state + learned transitions + LLM-extracted features + gating turned the
EXP-009 tie into a hybrid that **beats raw LLM + context overall**, with the world model winning
outright where state is rich. It is a real improvement, earned on no-cheat held-out data — not a
claim of general victory.
