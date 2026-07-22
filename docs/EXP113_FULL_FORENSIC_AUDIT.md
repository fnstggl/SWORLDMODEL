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

Hormuz asks whether daily tanker transits cross a numeric bar (≥ 50 transits on one day, stated
four times in the resolution). The forensic reconstruction shows the blueprint's `yes_when`
*did* encode it (`daily_transit_count >= 50`, `threshold "50"`) — but two failures downstream
turned the executable terminal into a **qualitative OR of de-escalation events**, not a count
comparison:

1. **`terminal_canonicalization` over-reached.** The canonicalization step (a genuine keeper for
   boolean *event* terminals — it is what saved visionOS) rewrote three actions' write-keys all
   to the single boolean `__terminal_yes__`, collapsing the numeric predicate into an OR. So
   `us_lift_blockade` writing `__terminal_yes__ = "lifted"` (a precondition, not 50 crossings)
   satisfied YES on 0.333 of mass, and `tanker_operators_transit` wrote `__terminal_yes__ = ">=50"`
   as a **hard-coded string literal** — an asserted verdict, not a measured count — on another
   0.333. No action ever wrote an integer transit count.
2. **The threshold parser failed to extract "50"** from the resolution prose (the number is
   present but not behind an "at least / or more" trigger the regex recognises), so the
   missing-mechanism ladder returned `threshold: null` and `failure_proof: "the resolution
   criterion carries no parseable numeric threshold"`. Combined with evidence that carried no
   pre-as_of transit-count series, the ladder honestly built nothing and left 0.333 as
   `missing_mechanism`.

The bridge that is absent: `actor decisions (blockade effectiveness, escort capacity, insurance,
routing) → an integer daily tanker-transit COUNT → compare to 50`. Because the terminal fired YES
on a boolean flag, the simulation predicted 1.0 (disruption) where the outcome was 0 (NO). A
correct numeric mechanism would move the resolved mass heavily toward NO (the window opened at
~7 ships / 0 tankers with a persistent blockade in most worlds).

**Mass affected:** the 0.667 resolved-YES mass is spurious; the 0.333 was honestly unresolved.

## 4. What genuinely worked (the keepers)

- **State-completeness invariant + recovery ladder** — 0/5 questions had an empty actor-state
  set reach rollout; every actor received a weighted state set. This eliminated the original
  "unknown state → dead world" failure completely.
- **Terminal-writer canonicalization + synthetic round-trip** — every round-trip passes; the
  visionOS class of bug (a completed simulation discarded on a YES-label mismatch) is dead. On
  visionOS the resolved simulation (0.75 at P=1.0) was *kept* and improved the forecast.
  **Caveat (see Defect 3):** canonicalization is correct for boolean *event* terminals but must
  **skip numeric-threshold predicates** — on Hormuz it collapsed a `daily_transit_count >= 50`
  predicate into a boolean OR, which is a large part of that question's failure.
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

## 4b. Secondary implementation bugs (concrete, narrow, fixable — distinct from the three architectural defects)

The per-question reconstructions surfaced four specific bugs that amplified the defects above.
Each is small and independently fixable, and each moves a wrong forecast materially:

- **State-weighting inversion (BoJ).** The `ActorStatePosteriorEngine` attached the counted
  reference class "dissents-for-hike ≈ 0.75" to a *Maintain* (dovish) state for each of the three
  real April hawks — pinning a pro-hike counted rate onto an anti-hike state, inverting the
  dissenters. This is a class→state matching error (the counted class was bound to the
  semantically opposite hypothesis). It is a large part of why BoJ's hawks read dovish.
- **Forced-vote label-prefix drop (BoJ).** At the hard-deadline closure, Ueda's cast came back as
  `"vote:Raise to 1.0%"` — carrying the `vote:` menu prefix from the terminal-action set — which
  the terminal's option matcher never stripped, so his Raise was dropped and he contributed 0
  Raise across ~62% of nodes. A one-line normalization (strip the `vote:` prefix before matching)
  recovers it.
- **Forced-fallback inconsistency on "conflicted/unity" variants (Banxico).** The deterministic
  `_force_terminal_vote` uses the variant's `action_if_state`; for internally-conflicted /
  consensus-seeking variants that field is empty, so the fallback defaulted to the menu's lowest
  option — and it did so **inconsistently across actors** (irene → hold, victoria → hold, but
  omar's `pragmatic_unanimity_seeker` → not-hold, scored as a dissent). A consensus-seeking state
  was thus turned into a dissent over ~1/3 of mass.
- **Forced completion never helps the consensus/coalition outcome (Banxico, Wale).** Because the
  fallback casts a fixed per-state action, forcing only ever added *status-quo / self-interested*
  votes: on Wale **0.0 of the forced votes went to Wale** (they only suppressed him); on Banxico
  the forced unity variants split inconsistently. Forced votes are grounded but are not free
  actor evidence and structurally cannot produce the convergence the real institutions showed.

Quantified reversibility (from the per-question closed-form recomputations, all matching the sim
exactly): **Banxico** fix omar's unity vote → 0.099→0.198, full committee coordination → ~0.83;
**BoJ** weight the bloc as its 5 real seats → 0.036→0.500 (Brier 0.93→0.25), fix Ueda's dropped
Raise → 0.169, un-invert the hawks → toward 0.875; **Wale** fix the candidate self-abstention →
0.130→0.278, model the majority coalition as whipped/unified → ~0.6–0.8 (a Wale-win forecast);
**Hormuz** a correct integer transit mechanism → resolved mass moves heavily to NO. In every
wrong case a correction to the identified first-wrong-step reverses the forecast toward the true
outcome — the errors are architectural, not irreducible uncertainty.

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
  convergence, so unanimity is geometrically suppressed (0.099 = 24/243 vs base rate 0.83);
  amplified by the forced-fallback turning omar's consensus-seeking variant into a dissent.
- **BoJ** — the 9-member board collapsed to 5 units (a 5-member bloc voting once) plus the
  independence defect; amplified by two implementation bugs (the dissents-for-hike counted class
  bound to a Maintain state, inverting the hawks; and Ueda's `vote:`-prefixed Raise dropped by
  the terminal matcher).
- **visionOS** — none material; the simulation was correct. (Minor: 0.25 honest
  missing_mechanism because the boolean terminal has YES-writers but no positive NO-writer — the
  right fix is a deadline "no-announcement-recorded" writer, which the ladder correctly declined
  to replace with a numeric mechanism.)
- **Wale** — modeling the electing 50-seat parliament as the rival candidates themselves (3 of 5
  voters are Wale's opponents; the majority coalition compressed to one bloc vote); amplified by
  the candidate self-abstaining in 62.5% of his mass.
- **Hormuz** — a compound terminal failure: `terminal_canonicalization` collapsed the numeric
  `daily_transit_count >= 50` predicate into a boolean OR of de-escalation events, and the
  threshold parser failed to extract "50" from the prose, so no integer transit bridge was built
  and a qualitative flag fired YES.

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

Before (or alongside) that, the four secondary bugs in §4b are near-zero-cost quick wins that
each move a wrong forecast materially on their own — the bloc-as-real-seats weighting alone takes
BoJ from 0.036 to 0.500 (Brier 0.93→0.25), and stripping the `vote:` prefix and un-inverting the
counted-class→state binding are one-line fixes. They are worth landing first because they are
unambiguous correctness fixes, not modeling choices, and they de-noise the accuracy signal the
convergence work will be measured against.

Do **not** calibrate a prior↔simulation combiner until at least Defect 1 is fixed and the
simulation-only accuracy is re-measured — a good prior Brier must not be allowed to mask the
anti-consensus simulation.
