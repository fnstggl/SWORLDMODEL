# EXP-113 Full Forensic Audit

Read-only forensic audit of the completed EXP-113 five-question simulation-completion run. No
code was modified, no question rerun, no LLM/web call made. Every claim traces to a committed
trace file under `experiments/results/exp113_checkpoints/` and
`experiments/results/lean_v2_accuracy/<qid>-completion/`.

Companion documents:
- `docs/EXP113_TRACE_COMPLETENESS.md` — the call-by-call completeness verification.
- `experiments/results/lean_v2_accuracy/<qid>-completion/forensic_report.md` — the full A–J
  chronological reconstruction for each question.

## 1. Trace completeness (summary; full detail in the completeness doc)

For all five runs the four independent call counters agree exactly (harness `n_calls` =
`metrics.n_llm_calls` = `budget.calls` = `llm_calls.jsonl` rows: 33 / 41 / 21 / 30 / 18), call
ids are contiguous `0..N−1`, every row carries an exact non-empty prompt and reply, nothing is
truncated (largest reply 15 376 < 24 000-char cap), zero retries, zero failed calls, and
per-stage counts match the ledger. The traced `gateway.call` is provably the sole external LLM
path (the backend is invoked only at `gateway.py:61/67`; no lean_v2 module imports a provider
client). `actor_decision` rows reconcile as `unique_contexts + deliberations (+ challenger
contexts)`.

**One honest gap:** BoJ and Wale were re-run during development; on their final runs the
blueprint, blueprint-repair, and grounding calls were **persistent-cache hits**
(`blueprint.from_cache = true`, `hits_persistent = 4`), so those calls' exact *prompts* are not
in the final trace. The cached *responses* are recoverable from `experiments/results/exp113_cache/{boj,wale}/`.
The trace is therefore complete for every call *made during each final run*, but not a complete
prompt-level record of BoJ's and Wale's world compilation.

## 2. Per-question one-line outcome

| # | Question | Outcome | Prior | Sim-only | Resolved | Final | Final Brier | Simulation verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Banxico (unanimous 5-0?) | 1 | 0.833 | 0.099 | 1.00 | 0.099 | 0.812 | resolved fully, **wrong** (anti-consensus) |
| 2 | BoJ (raise rate?) | 1 | 0.875 | 0.036 | 1.00 | 0.036 | 0.929 | resolved fully, **wrong** (anti-consensus + roster) |
| 3 | visionOS (announce?) | 1 | 0.833 | 1.00 | 0.75 | 0.958 | 0.002 | **right** — beat prior and full-fidelity |
| 4 | Wale (elected PM?) | 1 | 0.167 | 0.130 | 1.00 | 0.130 | 0.757 | resolved fully, **wrong** (electorate = rivals) |
| 5 | Hormuz (≥ numeric transits?) | 0 | 0.500 | 1.00 | 0.667 | 0.833 | 0.694 | **wrong** — qualitative predicate, no numeric bridge |

The mechanical completion targets all pass (0/5 terminal `unknown_state` mass, mean resolved
0.883, mapping round-trips pass on all five, missing-mechanism only where proven unavoidable).
The simulation-only *accuracy* is right on 1/5. The four wrong questions fail for three distinct,
identifiable architectural reasons, below.

## 3. The three most important general architectural defects

### Defect 1 — False actor independence: the simulation has no deliberative convergence, so it is systematically **anti-consensus**

This is the dominant error; it drives Banxico, BoJ, and Wale (3 of the 4 wrong questions).

The engine assigns each actor a set of private-state variants and then **branches the world on
each actor's variant roughly independently** (conditional only on the shared world conditions).
Actors never move toward each other: there is no step where a member, seeing the room forming a
consensus, updates their own vote. The vote spreads prove it — in Banxico every one of the five
voters carries *both* a "cut" state and a "hold" state:

```
galia_borja_gomez:  {cut, hold}      jonathan_heath: {cut, hold}
irene_espinosa:     {hold, wait}     omar_mejia:     {cut, hold, wait}
victoria_rodriguez: {cut}
```

Because unanimity requires all five to land on the *same* option and the five draws are
near-independent, `P(all five identical)` collapses geometrically — the simulation puts only
**0.099** on unanimity where the counted base rate is **0.83** and reality was unanimous. The
same mechanism gives BoJ `P(≥3 of 5 raise) = 0.036` and Wale `P(≥3 of 5 for Wale) = 0.130`. Real
central-bank boards and governing coalitions *deliberate to consensus*; this simulation models
them as independent coin-flips, so it manufactures disagreement and is biased against the
unanimous / consensus / coalition-holds outcome on every multi-actor institutional vote.

**Mass affected:** essentially all resolved mass on Banxico (1.0), BoJ (1.0), Wale (1.0).
Correcting it could plausibly reverse all three toward their true (consensus) outcomes.

### Defect 2 — Institution roster collapse: the voting body is modeled with too few, wrong units

The blueprint compiler collapses large institutions into a handful of representative
actors/blocs, which misrepresents the voting arithmetic even after the terminal-law threshold
translation.

- **BoJ:** the thesis itself says "a simple majority of the **9 members** is required," yet the
  board is modeled as **5 units** — four named principals plus one bloc `other_members` (the
  remaining ~5 members) that casts **one** vote. Threshold 5-of-9 was translated to
  majority-of-5 (≥3), but the 5-member bloc's single split vote and the independence defect
  concentrate mass on "Maintain" → NO 0.964.
- **Wale (worse):** the **parliament that elects the PM** (50 seats) is modeled with 5 required
  participants that are **the rival candidates themselves** — `matthew_wale`, `john_agovaka`,
  `frederick_kologeto`, `jeremiah_manele`, plus one `opposition_coalition_mps` bloc. Three of the
  five "voters" are Wale's rivals, who naturally vote for themselves; the entire backbench that
  would actually deliver Wale's majority is compressed into a single bloc vote. So a coalition
  that in reality *had the numbers* appears to lose (Wale ≥3-of-5 almost never holds). The
  electorate has been replaced by the candidates.

**Mass affected:** all resolved mass on BoJ and Wale. A faithful roster (model the real voting
bloc sizes, or weight a bloc by the members it represents; never model the electorate as the
candidates) is necessary for either to be answerable correctly.

### Defect 3 — Terminal predicate does not encode the question's numeric bar; the numeric bridge is missing (Hormuz)

Hormuz asks whether daily tanker transits cross a specific numeric threshold. The blueprint's
terminal is a **qualitative event/disruption predicate**, not the numeric daily-transit count,
so an actor action that writes a "disruption occurred" / "some tankers transit" style flag
satisfies YES on 0.667 of the mass even though the question requires a specific daily count. The
missing-mechanism ladder tried to build a bounded numeric bridge and **correctly reported it
could not** — `failure_proof: "the resolution criterion carries no parseable numeric threshold
— a bounded numeric mechanism is not the right bridge for this terminal"` — leaving 0.333 as
honest `missing_mechanism` mass. The bridge that is absent: `actor decisions (blockade
effectiveness, escort capacity, insurance, routing) → a daily tanker-transit COUNT → compare to
the numeric threshold`. Because the terminal fired YES on a qualitative flag, the simulation
predicted 1.0 (disruption) where the outcome was 0 (NO). (Per-question detail and the exact
predicate are in the Hormuz forensic report.)

**Mass affected:** the 0.667 resolved-YES mass is spurious; the 0.333 was honestly unresolved.

## 4. What genuinely worked (the keepers)

- **State-completeness invariant + recovery ladder** — 0/5 questions had an empty actor-state
  set reach rollout; every actor received a weighted state set. This eliminated the original
  "unknown state → dead world" failure completely.
- **Terminal-writer canonicalization + synthetic round-trip** — every round-trip passes; the
  visionOS class of bug (a completed simulation discarded on a YES-label mismatch) is dead. On
  visionOS the resolved simulation (0.75 at P=1.0) was *kept* and improved the forecast.
- **Mass conservation** — weights sum to 1.0 within rounding across up to 1 458 nodes on every
  question; terminal groups reconcile exactly to the simulation-only probability.
- **Missing-mechanism honesty** — where no numeric bridge could be built (Hormuz 0.333,
  visionOS 0.25), the mass stayed labeled `missing_mechanism` with a recorded proof instead of
  silently vanishing or silently reverting to the prior.
- **Mass-weighted headline that never silently drops the simulation** — the fix that always
  emits both binary distribution keys means an all-NO resolved simulation is a valid mapped
  forecast; visionOS's headline correctly blended resolved simulation (0.75·1.0) with the prior
  for the unresolved 0.25 → 0.958.
- **Deadline-forced completion** — drove Banxico/BoJ/Wale to 100% resolved. (Caveat below: it
  also exposes Defect 1, and on Wale a non-trivial share of reopened decisions were deterministic
  forced votes — see §5.)
- **visionOS end-to-end** — small faithful roster + event-style terminal + coherent actor
  decisions produced the one correct, prior-beating forecast. This is the shape to generalize.

## 5. Caveat on forced completion (invented final choices)

The hard-deadline forced-vote path (`_force_terminal_vote`, which casts the actor's variant
`action_if_state` when a required participant is still waiting at the deadline) is legitimate for
*completing* a world, but it is a deterministic fallback, not a free actor choice. In the sampled
decision trace (capped at 200 rows) Wale shows a meaningful share of reopened decisions as
`wait`/`gather_information` that were then forced (≈66 of 200 sampled mandatory-terminal rows),
whereas visionOS's 12 mandatory-terminal rows were all genuine `act` votes. The per-question
reports quantify the forced-vote mass where the full manifests allow. This is disclosed, not
hidden: forced votes are grounded in the actor's own simulated state, but they are not
independent evidence and should be counted as such in any future accuracy analysis.

## 6. Final answers to the six audit questions

**1. Is the trace genuinely complete?** Yes for every external call *made during each of the
five final runs* — verified across four independent counters, contiguous ids, exact untruncated
prompts and replies, and a proven single external path. It is **not** a complete prompt-level
record of BoJ's and Wale's world compilation, whose blueprint/repair/grounding calls were
persistent-cache hits from earlier runs (responses recoverable from the cache dir, prompts not
in-trace).

**2. Which calls, if any, are missing?** No call made during any final run is missing. Absent
from the final traces (by cache hit, not by loss): BoJ's and Wale's `structural_generation`,
blueprint-repair, and `reference_class_grounding` prompts — 3 call types × 2 questions.

**3. First wrong assumption per question:**
- **Banxico** — modeling the five voters as near-independent state draws with no deliberative
  convergence, so unanimity is geometrically suppressed (0.099 vs base rate 0.83).
- **BoJ** — the 9-member board collapsed to 5 units (a 5-member bloc voting once) combined with
  the same independence defect, concentrating mass on "Maintain."
- **visionOS** — none material; the simulation was correct. (Minor: 0.25 mass left as honest
  missing_mechanism.)
- **Wale** — modeling the electing parliament as the rival candidates themselves (3 of 5 voters
  are Wale's opponents; the coalition backbench compressed to one bloc vote).
- **Hormuz** — the blueprint terminal encodes a qualitative disruption event, not the question's
  numeric daily-transit threshold, so a qualitative "disruption" flag fires YES.

**4. The three most important general architectural defects:** (1) false actor independence / no
deliberative convergence → systematic anti-consensus bias; (2) institution roster collapse →
wrong voting units and sizes (BoJ 9→5 bloc, Wale electorate = candidates); (3) terminal
predicates that do not encode the question's exact (numeric) resolution bar, with no bridge from
actor actions to the measured quantity.

**5. Which components genuinely worked:** the state-completeness invariant + recovery ladder;
terminal-writer canonicalization + synthetic round-trip; mass conservation; missing-mechanism
honesty (proof, no silent drop); the mass-weighted headline that never silently reverts to the
prior; and the full visionOS pipeline (the one correct, prior-beating forecast).

**6. The smallest next implementation required before any calibration:** a **deliberative
convergence step for co-deciding actors** — after the independent private-state leanings are
drawn, let members of the same institution facing the same decision update toward the emerging
institutional consensus (using the shared-condition / institutional-pressure signals that
already exist), instead of resolving the terminal as a near-independent product of member votes.
This single change directly attacks Defect 1, which dominates 3 of the 4 wrong questions, and
would move the simulation off its anti-consensus floor without touching the prior or any
combiner. It should be paired with a faithful-roster rule for Defect 2 (never model the
electorate as the candidates; weight a representative bloc by the members it stands for), but the
convergence step is the smallest single fix with the largest expected accuracy gain. Terminal
numeric-bar encoding (Defect 3) is a narrower, separate fix affecting one question.

Do **not** calibrate a prior↔simulation combiner until at least Defect 1 is fixed and the
simulation-only accuracy is re-measured — a good prior Brier must not be allowed to mask the
anti-consensus simulation.
