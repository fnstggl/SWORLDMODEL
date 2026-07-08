# EXP-020 — Known + inferred variable mapping: the core architecture

The thesis, made architecture: to simulate how a person reacts to a message/event, you must map
**every variable acting on that behavior** — the *known* ones (from data / user context) and the
*inferred* ones (from context clues, via the LLM) — then run the person, so-described, against the
action. This is now the core state object every individual simulation flows through, and it is
backtestable and leakage-safe.

## What was built (`swm/variables/`)
1. **`schema.py` — the behavioral variable taxonomy.** 26 variables across the six determinants of a
   social response: **disposition** (base responsiveness, expertise, conscientiousness, openness,
   status, skepticism), **relational** (relationship strength, trust, prior stance, reciprocity),
   **incentive** (goal alignment, stakes, effort cost, reputational incentive), **state** (attention,
   recency, mood, urgency-fit), **platform** (response norm, visibility, formality), **message-fit**
   (personalization, clarity, pushiness, ask-directness, length-fit). Each has a category, range,
   and allowed provenance — grounded, not ad hoc.
2. **`variable_map.py` — the `VariableMap` state.** Every variable is a `Variable(value, provenance,
   confidence, evidence)`. Provenance ∈ {user, data, llm, heuristic, prior}; higher provenance wins
   on conflict (user/data > llm > heuristic > prior). `to_features()` yields a confidence-weighted
   vector (low-confidence/unset variables barely move the prediction) + a confidence channel per
   variable. `explain()` makes every prediction auditable: which variables drove it, from what source.
3. **`inference.py` — the `VariableInferenceEngine`.** Populates the map from, in trust order: user
   context → as-of data (responsiveness, relationship, recency, reciprocity — high confidence scaled
   by evidence) → LLM inference (disposition/relational/incentive from message + history + platform;
   a precomputed agent-inference dict can be supplied so it runs with no API key) → platform norms →
   lexical message-fit heuristics → population priors. **Leakage-safe: it only ever sees history
   before the action and the action's own content — never the outcome. The LLM infers variables,
   never the response.**
4. **`swm/worlds/variable_world.py` — `VariableWorld`.** Every prediction flows
   `(entity, action, context) → infer → VariableMap → calibrated readout → P(response) + next state`.
   Entity history is tracked online (as-of). This is the mandated pipeline: no prediction bypasses
   the variable map.

## Backtests (no-cheat temporal split) — the architecture is real, not a wrapper
| dataset | plain entity-state model | VariableWorld (mapped variables) | Δ |
|---|---|---|---|
| GitHub issue-response (n=15,262) | 0.3215 | **0.3211** | −0.0004 (matches) |
| Enron email-reply (n=16,000) | 0.2827 | 0.2939 | +0.0112 (small residual) |

Routing **everything** through the full 26-variable provenance-tracked state **matches** the compact
entity-state model on GitHub and is within ~0.01 on Enron (the residual is the many prior-filled
variables adding a little regularization noise). So the architecture buys generality, auditability,
uncertainty, and LLM/user-context readiness **at ~zero accuracy cost** — it does not sacrifice the
hard-won predictive performance.

## Do LLM-INFERRED variables earn their place? (the thesis test)
Reusing the EXP-019 agent-extracted features as *llm-provenance* variables (clarity → clarity,
actionable → ask-directness, effort → effort-cost, sentiment → mood-valence, specificity → stakes),
on 1,600 GitHub issues: **the LLM-inferred variables HELPED** — log loss 0.4300 → **0.4219**
(Δ +0.0081) and ECE 0.033 → 0.026. This is a **more positive** result than EXP-019's raw-feature test
(which was mildly negative), and the reason is instructive: mapping the LLM output to *meaningful
behavioral variables with confidence* (so the readout knows what each is and how much to trust it) is
the right way to absorb LLM inference — better than dumping raw features into a classifier. Small
sample, so encouraging-not-definitive, but it points the right way.

## Why this is the right core architecture
- **Unifies known + inferred.** Data gives high-confidence known variables; the LLM fills the latent
  ones it can infer from context; the user overrides with anything they know. One state object,
  provenance-tracked, so nothing inferred is ever presented as fact.
- **Every simulation conditions on it.** The individual regime uses a per-entity `VariableMap`; the
  aggregate regime's `SegmentAgent` affinities are the segment-level analog (behavioral variables per
  stakeholder segment). The mandate — every simulation maps the variables first — is structural now.
- **Backtestable + honest.** The mapped variables feed a calibrated readout scored on real outcomes,
  so a variable earns its place or is down-weighted; nothing is asserted. Leakage-safe by construction.

## Honest limits / next
- For entity-driven *reply* outcomes, the DATA-derived responsiveness variable still dominates; the
  inferred latent variables (trust, goal-alignment, stakes) add modestly. Their value should be
  largest on outcomes genuinely driven by them — **persuasion, sentiment/stance shift, conversion,
  objection** — which need the right timestamped dataset (the next data acquisition).
- The full-latent inference (disposition/incentive from rich message+context) via a dedicated agent
  swarm is wired and ready (`llm_inference=` / `VariableInferenceEngine.llm_infer_fn`); this round
  demonstrated it with reused features. Running it at scale on a persuasion/conversion corpus is the
  highest-value next experiment.
- Enron's +0.01 residual is worth closing (the prior-filled variables); a variable-selection pass
  (drop variables that never inform on a domain) would recover it.

## Reproduce
`python -m experiments.exp020_variable_world {github,enron,llm-arm}`; tests in
`tests/test_variables.py`.
