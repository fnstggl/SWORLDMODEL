# WMv2 Transfer-First Evaluation (Phase 14)

*The world model's central opportunity is NOT memorizing stationary distributions — a fitted per-task model
wins there. Its opportunity is transfer: new games, new conditions, new people, counterfactuals, where no
on-distribution training sample exists. Results are reported SEPARATELY by transfer type, never pooled.*

## 1. In-distribution (the specialist ceiling — V2 does not win here, and says so)

| benchmark | V2 arm | specialist ceiling | verdict |
|---|---|---|---|
| BehaviorBench | P_fit 0.099 W1 | train histogram A1 0.038 | specialist wins |
| Higgs | log-linear hazard | fitted exposure logistic H1 | **tied** (Δ−0.000192, CI straddles 0) |
| OmniBehavior | persistence 0.0965 Brier | — (no per-event specialist) | V2 best available |
| Upworthy CTR pick | surface model | oracle | negligible lift |

## 2. Cross-person transfer

| benchmark | result |
|---|---|
| **OmniBehavior person-disjoint** (14 users NEVER in train, n=216) | persistence mechanism Δ−0.027 [−0.032,−0.021] vs memoryless — **transfers to held-out people** |
| Enron person-disjoint (prior round) | fitted metadata transfers; structured cognition ns |

## 3. Cross-condition / cross-game transfer

| benchmark | result |
|---|---|
| **BehaviorBench leave-one-game-out** | P_logo beats pooled-histogram transfer on dictator (Δ−0.067), ultimatum_proposer (Δ−0.034), trust_investor (Δ−0.058) — CIs exclude 0; **FAILS on public_goods** (+0.328). Mixed, preserved. |
| BehaviorBench cross-game interaction (prior) | simulating the partner from another game's fitted latent: significant on 3 of 4 structural games |

## 4. Cross-domain transfer

| from → to | result |
|---|---|
| Higgs diffusion params → other cascades | UNVALIDATED (single cascade; transport risk high, logged) |
| lab economic games → field behavior | UNVALIDATED (logged) |
| Kuaishou engagement → deliberation | UNVALIDATED (logged) |

Cross-domain transfer is the least-tested axis; the registry flags every transported pack with a widening
factor and a transport note. No cross-domain transfer is claimed as validated.

## 5. Counterfactual / novel-condition

| benchmark | result |
|---|---|
| matched-counterfactual mechanics (synthetic, controlled) | correct best-action identification via paired CRN; P(best)/regret computed correctly |
| Upworthy randomized headline choice | negligible realized decision lift (surface model +0.00017 CTR vs oracle gap 0.00407) |

## The pattern, honestly

**Where V2 transfers, it is because mechanisms generalize across conditions in ways a fitted per-outcome
model cannot express:**
- persistence transfers to new PEOPLE (Δ−0.027, the round's clearest transfer win);
- FS-preference + QRE transfers to held-out GAMES on 3 of 4 (the games where inequity aversion is the right
  structure), and fails where it is not (public_goods);
- nonlinear diffusion hazard generalizes the exposure→activation relationship well enough to tie the
  fitted logistic in-distribution (a prerequisite for cross-cascade transfer, not yet tested).

**Where it does not**: raw in-distribution stationary prediction (specialist wins), cross-domain parameter
transport (untested), real randomized-intervention decision lift (negligible on the one benchmark
available).

## Reported separately (never one score)

in-distribution ≠ cross-person ≠ cross-game ≠ cross-domain ≠ counterfactual. The one-number summary would
be dishonest; the table above is the honest decomposition.
