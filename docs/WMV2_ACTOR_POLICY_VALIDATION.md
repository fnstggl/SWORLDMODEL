# WMv2 Actor-Policy Validation (Phase 4)

Code: `swm/world_model_v2/policy.py`, `registry/families/choice.py`, `registry/families/learning.py`,
`transitions.py` (FittedDecisionOperator). Artifact: `experiments/results/wmv2_behaviorbench_policy.json`.

## The contract (RC4 repair)

The LLM may NOT mint behavioral probabilities on the production path. `AgentDecisionOperator` now runs a
uniform reference policy by default (loudly flagged) and only accepts LLM-minted probabilities behind an
explicit experimental opt-in; `FittedDecisionOperator` (utility+QRE / anchored-logistic / hierarchical
rates) is the production decision path. Pinned by `tests/test_wmv2_tier_a_fixes.py`.

## The learned policy (BehaviorBench)

ONE population-preference model — a discrete Fehr-Schmidt (α,β) inequity-aversion mixture fitted JOINTLY
across games (partial pooling) — drives every two-player game through the SAME utility machinery; games
differ only in typed action space + payoff consequences. Choice noise is quantal response (McKelvey-Palfrey
logit) with payoff-scale-normalized precision that transfers across games. Cross-game interaction is
structural: proposer acceptance beliefs and investor return beliefs derive from the same fitted preference
population. Published packs (FS 1999, CHC 2004 τ, FGF 2001) provide the cold-start.

## Held-out results (same immutable splits as the prior round, seed 13)

| arm | mean W1_norm | reading |
|---|---|---|
| A0 uniform | 0.160 | floor |
| **A1 train histogram** | **0.038** | specialist ceiling (in-distribution) |
| A2 train KDE | 0.042 | |
| P_pub (published packs, $0, no local fit) | 0.148 | cold-start |
| **P_fit (universal FS+QRE+CH)** | **0.099** | beats raw LLM (0.185) + elicitation (0.123); LOSES to A1 |
| P_logo (leave-one-game-out transfer) | 0.144 | mixed (see below) |
| P_no_interaction (ablation) | 0.108 | interaction helps a little |
| P_selfish (α=β=0 ablation) | 0.125 | preferences matter |
| P_no_qre (λ→∞ ablation) | 0.151 | QRE noise matters |
| *prior hand-crafted V2 (quoted)* | *0.058* | |

## The honest verdict

- **Replicates the portfolio's central pattern**: raw LLM (0.185) < structured policy (0.099) < specialist
  fitted histogram (0.038). The universal, principled, published-form policy does NOT beat per-game
  fitting in-distribution.
- **Transfer is mixed** (leave-one-game-out, the regime where no on-distribution model exists): P_logo
  beats the pooled-histogram transfer baseline on dictator (Δ−0.067), ultimatum_proposer (Δ−0.034),
  trust_investor (Δ−0.058) — CIs exclude 0 — but **FAILS on public_goods** (+0.328; the FS→PG contribution
  mapping is too crude, KS 0.458). Preserved as a documented failure mode.
- Distributional calibration (PIT/KS) reported per game; ranges 0.09–0.46 (public_goods worst).

## Ablations (each refit at its own best config on train)

interaction: helps (P_fit 0.099 vs no_interaction 0.108). preferences: load-bearing (selfish 0.125).
QRE noise: load-bearing (no_qre 0.151). LLM semantic channel: NOT re-bought — quarantined by
`semantic_registry.py` (null-or-harmful on every prior domain).

## Learning families (executable, `learning.py`)

reinforcement (Q-learning), belief learning (fictitious play), EWA (Camerer-Ho), habit formation — all
executable transitions with published forms; unit-tested for correct dynamics (`test_policy_families.py`).
Not yet validated on a repeated-play behavioral dataset (the BehaviorBench games are one-shot).

## Four-status

- **software-implemented**: YES (policy layer, FittedDecisionOperator, 4 learning families).
- **executes-end-to-end**: YES (fitted policy runs through the world runtime; typed actions → StateDelta).
- **empirically-validated**: YES on BehaviorBench — with an HONEST mixed/negative result (beats LLM, loses
  to specialist, transfer mixed with one failure). Negative preserved.
- **production-eligible**: NO in-distribution (loses to specialist); the transfer wins (dictator/proposer/
  investor LOGO) are the trailhead but not yet a deployable net gain. `social_preference_population` sits
  at `locally_validated`, correctly not promoted.
