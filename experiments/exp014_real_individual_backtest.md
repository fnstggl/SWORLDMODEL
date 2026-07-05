# EXP-014 — Real individual/entity-response backtest on public GitHub data

The individual world model was, until now, validated only on synthetic data. This is its first
backtest on **real behavior**, on a free public corpus — and it is the strongest confirmation yet of
the core hypothesis: **the world model's advantage grows exactly where repeated entities and
inferable latent state exist, and there it beats a raw LLM given the same information.**

## Setup (no-cheat, public data)
- **Source:** GitHub Archive (`data.gharchive.org`) — the public GitHub event stream. 12 contiguous
  hours (2024-06-03), all events. Settled outcomes (historical).
- **Task:** an issue is opened on a repo; does it get a **response** (a comment by someone other than
  the author) within **6 hours**? 15,262 labeled issues, base rate **13.9%**.
- **Entity = the repo/maintainer team** — a genuine repeat entity with an evolving responsiveness
  state. As-of correct: a repo's state for an issue is built only from its *earlier* issues in the
  stream; the 6h outcome window is fully elapsed.
- **Model:** the existing `IndividualWorld` / `IndividualTransition` (hierarchical partial pooling:
  repo ← global) with issue features (title, body length, labels, is-bug, author association) + the
  as-of repo response rate. Temporal 70/30 split.

## Result 1 — modeling the entity state cuts error, and the benefit scales with depth
| regime | log loss | ECE | uplift@20 |
|---|---|---|---|
| segment (global rate) | 0.3896 | 0.011 | 0.017 |
| + message features | 0.3365 | 0.012 | 0.147 |
| + person (repo state) | 0.3333 | 0.036 | 0.168 |
| **full individual** | **0.3100** | 0.026 | **0.208** |

Modeling the individual entity beats the segment baseline by **0.080 log loss (−20% relative)** on
real behavior. **By repo-history depth at prediction time:**

| slice | n | segment ll | full ll | Δ (state benefit) |
|---|---|---|---|---|
| cold repo (0 prior) | 2287 | 0.4261 | 0.3943 | +0.0318 |
| repeat repo (1–4) | 1370 | 0.3934 | 0.2889 | **+0.1045** |
| deep repo (5+) | 922 | 0.2937 | **0.1322** | **+0.1614** |

**The entity-state benefit increases monotonically with entity-history depth** — +0.03 cold →
+0.10 repeat → +0.16 deep. On deep-history repos, modeling the entity **more than halves** the log
loss (0.294 → 0.132). This is the hypothesis, measured on real data: repeated entities + inferable
latent state = where the world model wins.

## Result 2 — the world model beats a raw LLM given the SAME information
On 120 state-rich test issues (repo depth ≥ 1), a 5-agent swarm predicted response probability two
ways, and we compare to the world model on identical items:

| tier | log loss | Brier | ECE |
|---|---|---|---|
| raw LLM, message-only (the "individual guesser") | 0.3648 | 0.110 | 0.064 |
| raw LLM + repo context (**handed the same track record**) | 0.3043 | 0.091 | 0.102 |
| **world model (full individual)** | **0.2576** | **0.080** | **0.040** |

By depth: repeat(1–4) — world model **0.335** vs LLM+ctx 0.397; deep(5+) — world model **0.108** vs
LLM+ctx 0.126. **The world model beats the raw LLM even when the LLM is given the exact numeric repo
track record** — because it calibrates and pools that state precisely, where the LLM eyeballs it and
is overconfident (ECE 0.102 vs 0.040). This is the thesis in the user's words: *modeling the moving
parts with the relevant variables mapped beats an individual educated-guessing over the same
information* — confirmed, on real behavior, in the state-rich regime.

## Honest caveats
- "Response within 6h" is a specific, somewhat mechanical outcome (easier than "will this cold email
  get a reply"); it is real repeat-entity behavior but not the hardest individual-response task.
- The world model's edge is largest where a **precise numeric state** exists (repo rate); on cold
  repos the gap is small (+0.03) — the LLM's prior is competitive there, consistent with every prior
  experiment.
- The LLM comparison is n=120 (16 positives) — directional; the 15,262-issue depth-monotonicity is
  the statistically solid backbone.
- Baseline in Result 1 is the segment mean, not the LLM; Result 2 adds the LLM head-to-head.

## Why this matters
It converts the individual model from "validated on synthetic, blocked on real data" to **validated
on real public behavior, with the hypothesized depth-scaling and a head-to-head win over a raw LLM
given the same information.** The next unlock is the private-outcome regime (email/CRM reply,
conversion) where the entity state is even richer and the decision value is highest — the same
machinery applies.

## Reproduce
`python -m experiments.github_individual_harness fetch --date 2024-06-03 --h0 8 --hours 12 --window 6`
→ `run` → `score-llm`. Agent predictions + common set committed under
`experiments/results/exp014_gh_llm/`; the raw event dump stays under gitignored `data/`.
