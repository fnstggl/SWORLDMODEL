# EXP-010 — Why the EXP-009 world model tied/lost to raw LLM (failure audit)

Harsh audit of the EXP-009 result (world model ≈ raw LLM on random HN). Companion to the deeper
architecture audit `exp012_simulation_architecture_audit.md`. No spin.

- **Did the world model have enough state depth?** No. State was a handful of EMA scalars (author
  quality/standing, domain reputation, topic salience). No rolling entity history, no sufficiency
  score, no context dynamics. Fixed here: `swm/state/entity_history.py`, `context_dynamics.py`.
- **Were entity histories too sparse?** Yes on a *random* HN sample — most posts are one-off authors
  (cold-start), so the individual-state signal had nothing to bite on. The benchmark confirms it:
  cold-start is the slice the world model loses; state-rich slices are where it wins.
- **Were transitions actually learned or mostly hand-coded?** Mostly hand-coded EMA updates + a
  logistic outcome head. No learned nonlinear transition. Fixed: `swm/transition/learned_transition.py`
  (pure-python gradient-boosted trees; beats logistic on interaction data 0.538 vs 0.674).
- **Did the state model reduce rich text to weak features?** Yes — it used `title_len`, `topic`,
  `is_show`. That threw away meaning. Fixed: LLM-extracted structured features (novelty, controversy,
  technical_depth, audience_fit, hn_native, source_credibility, category…) stored as ablatable
  variables and fed to the world model (not used as the LLM's probability).
- **Did raw LLM win because semantic title understanding dominated?** On random/cold posts, yes —
  the LLM's pretrained prior over HN titles is a strong substitute for explicit state. That is why
  it wins `semantics_dominant`/`low_context` and only ties overall.
- **Did retrieval hurt because context was thin/noisy?** On HN, as-of author/domain context slightly
  *hurt* the raw LLM (overconfidence) — there was little information gap to close. It is not that
  retrieval is broken; there was nothing to retrieve that the LLM didn't already price.
- **Which world-state variables were missing?** Rolling entity history + depth/sufficiency;
  community-segment structure; exposure/front-page/social-proof dynamics; novelty/fatigue; richer
  action semantics. All added (`entity_history`, `context_dynamics`, `swm/simulation/*`).
- **Which were present but weakly modeled?** Domain reputation and topic salience existed but as flat
  EMAs feeding a linear head — no interactions (deep-author × hot-topic). The GBDT + simulation now
  represent interactions.
- **Which transitions were fake/deterministic?** The one-step "rollout" applied deterministic scalar
  updates and (pre-fix) even used a state-ignoring `PriorHead` — no reactions, no cascade. Replaced
  by the multi-step simulation engine.
- **Which parts were a static feature model pretending to be a world model?** The whole aggregate/
  individual prediction path (`state -> feature vector -> logistic -> P`). It evolved state *between*
  items but each prediction was a classifier. That is the core finding, and the simulation rebuild
  (`exp012`/`exp013`) is the answer.

**Bottom line:** EXP-009's "tie" was really "a logistic over shallow as-of features ties the LLM on
random HN." The fixes — rich features, learned transitions, rolling entity state, and an actual
multi-step multi-actor simulation — are evaluated in `exp011_improved_world_model.md` and
`exp013_real_simulation_world_model.md`.
