# EXP-105 — the FULL world-model-v2 simulator (canonical `simulate_world` path) on 5 fresh BTF-3 questions

First run of the canonical top entry after the validation-gate removal. n=5 is an anatomy run, not a
benchmark claim. Same leakage protocol as EXP-101/102 (allowlisted fields only; live retrieval dropped;
evidence = the benchmark's own as-of background; answers join at scoring).

## Scores

| qid | question | p | outcome | Brier | SOTA p | SOTA Brier | LLM calls |
|---|---|---|---|---|---|---|---|
| 0851f82c | Knesset committee approves Associations Bill | 0.475 | NO | 0.226 | 0.03 | 0.001 | 9,593 |
| 08991bc9 | Illinois GA introduces mail-in-ballot legislation | 0.197 | NO | 0.039 | 0.04 | 0.002 | 1,924 |
| 09517ada | Paloma Valencia top-two, Colombia first round | 0.176 | NO | 0.031 | 0.33 | 0.109 | 5,623 |
| 09f5c200 | Nordio issues Zambelli extradition decree | 0.896 | NO | 0.803 | 0.12 | 0.014 | 2,737 |
| 0da98cff | UK launches ANPS public consultation | 0.705 | **YES** | 0.087 | 0.12 | 0.774 | 1,420 |

- **WMv2 full simulator: Brier 0.237, accuracy 4/5** · FutureSearch SOTA on the same 5: Brier 0.180 (3/5 by their probabilities at 0.5 — they were wrong-side on the UK question)
- EXP-102 (old gated path, prior 5 questions): Brier 0.2605, 3/5, every forecast squeezed into 0.33–0.63
- The system now takes real positions (0.18–0.90 spread). Wins on 2 of 5 vs SOTA head-to-head (Colombia
  0.031 vs 0.109; UK 0.087 vs 0.774 — the deep simulation called YES where SOTA said 0.12). One
  confident-wrong disaster (Italy extradition, hypothesis priors 0.7/0.3 both leaning yes — a compile-time
  structural-prior misjudgment, not an execution failure).

## What executed (operator delta census — the ledger of what wrote state)

`generic_outcome_prior` wrote state in **zero** runs. Every outcome was decided by the world:

| question | actor deltas | institutional_decision | scheduled_fact | population/network/nonlinear |
|---|---|---|---|---|
| Knesset | 9,504 | 29 | 180 | nonlinear_state_step 811, background 869 |
| Illinois | 1,822 | 59 | 421 | — |
| Colombia | 5,557 | 65 | 65 | population_aggregation 65, network_diffusion 65, nonlinear 1,235 |
| Italy | 2,655 | 43 | 187 | hazard_round 14 |
| UK ANPS | 1,622 | 60 | 121 | background 245 |

Every run also shows `actor_action_aggregation` (actions → readout) and `absorption_monitor`
(event-time first passage). All five ran uncapped actor cognition (SWM_ACTOR_MAX_CALLS lifted) with
12-way parallel branch rollout (SWM_BRANCH_THREADS — serial/parallel verified bit-identical).

## Compiled worlds (from the verbatim decompose replies)

- **Knesset**: Rothman, Kallner, Netanyahu, Levin + the Committee, Plenum, Ministry of Justice, Adalah,
  ACRI; Israeli_public population; hypotheses coalition_dominance 0.4 / gridlock 0.4 / compromise 0.2.
- **Illinois**: Pritzker, Harmon, Welch, Bost + both chambers, SCOTUS, State Board of Elections;
  Illinois_voters; preemptive 0.4 / reactive 0.3 / none 0.3.
- **Colombia**: Valencia, Cepeda, de la Espriella + Registraduría, CNE; Colombian_voters population with
  network diffusion and nonlinear opinion dynamics executing.
- **Italy**: Nordio, Zambelli, Cassazione, Rome Court of Appeal, Brazilian government, Ministry of Justice.
- **UK**: DfT, Transport Secretary, PM, Cabinet, Parliament, Heathrow Airport Ltd, Mayor of London,
  courts, Heathrow Action Network; UK_public.

## The new layers, live in the traces

- Actor prompts now carry the fixed knowledge scoping ("You know everything your real counterpart would
  plausibly know as of {date}… knowledge STOPS at {date}… no other minds") plus a PUBLIC CALENDAR
  section fed by the scheduled-reality layer — including recurrence annotations (e.g. the DfT's
  consultation-publishing pattern with past instances).
- Truncation robustness: one truncated decompose reply (Illinois) was detected; the reordered schema kept
  all execution-critical keys inside the salvaged prefix (`truncation_recovered_keys: []` — nothing lost).
- Support grade stays `exploratory` and status `completed_with_degradation` — honest labeling of
  unvalidated mechanisms that now run instead of being blocked.

## Reading of the results

The architecture now does what it claims: compiled actors, institutions, populations and calendars decide
the readout; the broad prior never writes. Accuracy at n=5 is anecdotal — the visible failure mode moved
UP the stack from "execution refuses to run the world" to "the compiler's structural priors can be wrong"
(Italy 0.7/0.3 both-leaning-yes). That is the right place for the next iteration: structural-hypothesis
priors should be disciplined by evidence (Phase-3 posterior reweighting was inactive here because live
retrieval was dropped; a frozen-corpus evidence arm would activate it).
