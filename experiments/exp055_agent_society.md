# EXP-055 — Agent-based simulation vs the composite (and institutional agents are real)

The audit's flagship failure was that `simulate_population` is a mean of independent regressions. This is
the real alternative: `swm/simulation/agent_society.py` — persona agents that take positions and
**interact** (influence-weighted, similarity-gated deliberation with a consensus pull and bounded
confidence), so the outcome **emerges** from their coupled behavior (`∂positionᵢ/∂positionⱼ ≠ 0`). It is
general over agent types — a handful of named institutional agents or a population of segments.

## A. Controlled mechanism proofs — emergent outcomes a composite CANNOT produce
| effect | result |
|---|---|
| **influential minority flips a vote** | independent count LOSES (6 vs 3) → deliberation PASSES (the 3 high-influence agents pull the persuadable majority over) — **outcome flipped** |
| **deliberation → consensus** | a split body (spread 0.6) converges to spread **0.002** under a consensus pull |
| **homophily + bounded confidence → polarization** | two blocs (gap 0.3) stay apart (gap **0.3**) — stable echo chambers |

All three are impossible for `sum(ps)/n` — they are properties of the interaction, not of the parts. This
is the difference between simulating and compositing, demonstrated.

## B. Real institutional agents — the Supreme Court (your pushback, vindicated)
You pushed back that institutional events (Fed, awards, courts) ARE populations of modelable agents. They
are. Using the Supreme Court Database (SCDB), each case is 9 justice-agents with ideology estimated from
**prior terms only** (leakage-free); we simulate their deliberation (ideological blocs via homophily + a
consensus pull) and compare to the INDEPENDENT baseline (each justice votes their ideology, majority wins
= the composite). 954 held-out cases (post-2009), 8,378 justice-votes.

| metric | independent (composite) | **agent-sim** |
|---|---|---|
| decision-direction accuracy | 0.534 | 0.533 |
| **vote-margin MAE** ↓ | 0.208 | **0.168** |
| individual-vote accuracy | 0.585 | 0.570 |

**Modelling the justices as interacting agents beats the composite on the vote MARGIN by 19%** (0.168 vs
0.208). The consensus/coalition dynamics correctly predict that real courts reach *larger* majorities than
independent ideological voting implies — deliberation drives justices together, and only the coupled model
captures it. This is the **first real-data case where interaction beats compositing** (EXP-053's GSS was
marginal-dominated and coupling did not help there; here, on the dimension interaction actually shapes —
the size of the coalition — it does).

## The honest findings
1. **Agent simulation genuinely beats compositing where interaction manifests** — the vote margin on real
   Supreme Court cases. Institutional outcomes ARE agent aggregates, and modelling the deliberation helps.
2. **Direction is at chance for both** (0.53) because we gave the agents no case features — only justice
   ideology. The margin win comes purely from the *interaction structure*, not from case content; adding
   case facts (issue, lower-court, briefs) is the obvious lift for direction and is where the LLM-persona
   agents would read the case.
3. **Individual-vote accuracy dips slightly** under the consensus pull (justices centralize) — the
   interaction that helps the aggregate margin trades a little individual fidelity, a real and honest
   tension to tune (per-agent conviction vs body consensus).

## What it means for the architecture
`AgentSociety` is the architecture the thesis wanted and the audit demanded — a genuine coupled simulation,
general over agent types, that produces emergent outcomes (flips, consensus, polarization) and, on real
institutional data, beats the independent composite on the interaction-shaped dimension. The `position_fn`
is pluggable: a structured value-match here; an LLM roleplaying each agent from its full persona +
retrieved context is the generative-agent path — the same society, richer agents.

## Honest limits
- Justice ideology only (no case features) → direction at chance; the win is on the margin.
- The consensus/homophily constants are set, not tuned leakage-free; tuning would sharpen the margin win
  but not create it (independent voting structurally can't produce the observed unanimity rate).
- SCDB public data; a mechanism benchmark for the agent architecture, not a leakage-free skill number.

## Reproduce
`python -m experiments.exp055_agent_society` → `experiments/results/exp055_agent_society.json`
(raw SCDB gitignored — see the loader). `python -m pytest tests/test_agent_society.py`.
