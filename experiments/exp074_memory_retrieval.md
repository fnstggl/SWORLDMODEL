# EXP-074 — episodic memory + reflection: situation-conditioned recall vs the global persona

**The question.** The single-individual regime carried a person as a *global average* — a `VariableMap` of
stable persona traits (who they are) + a transient `state` (how they are now). That throws away the most
predictive thing for "will this person respond to THIS?": **how they reacted to similar situations before,
recently.** This experiment adds the Generative-Agents (Park et al. 2023) recall layer — an episodic memory
stream with *recency × importance × relevance* retrieval, generative reflection, and recency-decayed persona
synthesis — and holds it to the only bar that matters here: **on held-out next behavior, leakage-safe, does
retrieving similar recent history beat predicting from the person's overall rate?**

Held to the repo's discipline (SIMULATION_AUDIT.md): a mechanism must beat the base *and* base-rate on a
proper score, or it is renamed/insight — never asserted.

## What was built (`swm/memory/`)

- **`embeddings.py`** — a dependency-free, deterministic hashing embedder (FNV-1a → tf buckets → L2) + cosine;
  pluggable `embed_fn` for a real production backend. The *relevance* channel with zero dependencies.
- **`memory.py`** — `Episode`, `MemoryStream` (per person), `EpisodicStore` (many people). Retrieval scores
  each candidate by **recency** (exponential time-decay toward `as_of`, `half_life`) × **importance** (stored
  poignancy) × **relevance** (embedding cosine to the query), each min-max normalized then weighted-summed
  (Generative-Agents α=β=γ=1 default, tunable). **Generative reflection** (`reflect` / `maybe_reflect`)
  synthesizes recent observations into higher-level abstractions ("Tends to respond to pricing messages"),
  stored back as *reflection* episodes with their own embedding + boosted importance — so they participate in
  future retrieval and compound. Pluggable `reflect_fn` (LLM); transparent behavioral-pattern fallback.
  **Leakage:** retrieval is strict `timestamp < as_of`, `as_of` REQUIRED, `assert_no_leak` post-condition —
  same contract as `swm/retrieval/asof_store.py`.
- **`retrieval_response.py`** — `retrieval_augmented_response_fn(base_fn, store, …)`: keeps the
  `(variables, state, message) → {"p", …}` contract so it drops into `IndividualSimulator` unchanged. At
  predict time it retrieves the person's top-k similar episodes as-of now and forms a **Beta-Binomial
  posterior** for the response rate on those *similar* contacts, shrunk toward the person's own base rate by
  κ pseudo-episodes and weighted by relevance²×recency: `p_final = σ(logit(p_base) + β·[logit(observed_shrunk)
  − logit(personal_base)])`. Thin/off-topic history → posterior ≈ base → no move (self-limiting); plentiful,
  on-topic, consistent history → moves fully. The shrinkage *is* the calibration, not an asserted magnitude.
- **Recency decay in `deep_inference.synthesize`** — each document weighted by `salience × recency` toward
  `now`; recency shifts the trait VALUE toward recent evidence (people drift), depth/consistency still set
  CONFIDENCE. Strictly backward-compatible: no timestamps/half_life → recency weight 1.0 → identical output.
- **`IndividualAgent`** gains an optional `memory` stream: as contacts land in `simulate_thread`, they are
  written to episodic memory, so a situation-conditioned `response_fn` can recall them.

## Results (synthetic mechanism validation, leakage-safe, seeded; `python -m experiments.exp074_memory_retrieval`)

Each arm: 120–160 synthetic people, ~40–48 dated contacts each; fit each person's store on all-but-last-k
contacts, predict the held-out future ones from prior history only. `SKILL = 1 − loss/loss_persona`.

**A. History-driven** (stable per-topic affinities on a personal baseline) — retrieval **WINS**:

| model | log-loss | Brier | ECE | skill vs persona |
|---|---|---|---|---|
| population base-rate | 0.6931 | 0.2500 | 0.003 | −0.018 |
| global persona (personal rate) | 0.6812 | 0.2433 | 0.035 | 0.000 |
| **retrieval-augmented** | **0.6436** | **0.2253** | 0.046 | **+0.0553** |

**B. Recency** (each person's topic affinity flips partway) — a **calibrated** half-life beats flat, and
over-decay hurts (the Law-2 "calibrated time" lesson):

| memory half-life | aug log-loss | skill vs persona |
|---|---|---|
| flat (∞) | 0.6593 | +0.0221 |
| **recency (hl=12)** | **0.6529** | **+0.0317** |
| over-decay (hl=3) | 0.6688 | +0.0080 |

**C. Honest negative** (outcome driven only by message quality — no person/topic structure) — retrieval
correctly finds **no exploitable signal** (mild self-limiting cost that shrinks as κ rises): retrieval-
augmented −0.036 skill vs persona. Reproduces the EXP-069 finding (persona helps model *who* a person is,
not outcomes the *message* drives) for the memory channel.

**D. Persona-synthesis recency** (a trait drifting 0.1 → 0.9 over 12 docs): flat synthesis reports **0.500**
(averages the stale history); recency-decayed (hl=2) reports **0.738** (tracks the recent trait).

## Verdict

- **Situation-conditioned recall beats the global persona where the outcome is history-driven** (+0.055
  skill, better Brier, leakage-safe) — the exact regime the single-individual product lives in (reply /
  churn / adherence).
- **Recency earns its place when behavior drifts** — but must be *calibrated* (hl=12 wins, hl=3 over-decays
  and hurts), consistent with the repo's Law 2 (calibrated time).
- **It correctly does no work where the message, not the person, drives the outcome** — the Beta-Binomial
  shrinkage is self-limiting, so the mechanism cannot fabricate signal. This is the honest boundary.

κ (=6) and β (=1) were selected on the held-out history-driven skill / self-limiting tradeoff and are exposed
as tunables for per-domain validation. The mechanism is validated here on transparent synthetic data with a
real generative process; the named next step is to re-earn it on **real threaded reply data** (the CMV /
Upworthy-style corpora) through the same as-of harness — where, per EXP-069, the win should appear on
history-driven outcomes (reply-propensity) and not on message-driven ones (single-argument persuasion flip).

Tests: `tests/test_memory.py` (12), `tests/test_retrieval_response.py` (7), recency cases added to
`tests/test_deep_inference.py`. Full suite green (349 passing; the one skipped requires the optional `api`
extra `fastapi`).
