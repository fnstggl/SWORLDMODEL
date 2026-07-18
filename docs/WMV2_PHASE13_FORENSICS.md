# World Model V2 — Phase 13 Forensics

Concrete traces from real runs. Detailed data lives in machine-readable artifacts under
`artifacts/phase13/`; this document points at representative rows and explains what they prove. Every
trace below is copied from a persisted artifact, not hand-authored.

## 1. Matched counterfactual — CRN pairing is exact

`controlled/results.jsonl`, task `multi_actor_00`:

```
recommended = aggr_0p1   optimum = aggr_0p1   hit = True
crn_all_paired = True
operator_delta_census = {controlled_payoff: 400, decision_action: 200, opponent_policy: 160}
```

The `decision_action` operator fired (the intervention entered through the event queue), the
`opponent_policy` operator fired 160 times (the **other actor responded** — this is a strategic task
where the opponent best-responds to the decider's aggression), and every arm shares the reference's
exogenous trace exactly (`crn_all_paired = True`). The recommended action is the strategic best
response (low aggression `aggr_0p1`), matching the analytically known optimum.

For the Thiel decision (`thiel_run/result.json`), the CRN manifest shows exact pairing across all four
arms against the `do_nothing` reference:

```
exogenous_trace_match_vs_reference = {
  do_nothing: 1.0, send_optimized: 1.0,
  send_credential_cover_letter: 1.0, send_pushy_followup: 1.0 }
active_phases.operator_delta_census = { decision_action: 1200, reply_mechanism: 1200 }
```

400 particles × 3 non-reference arms = 1200 decision events, each triggering the reply mechanism — the
graded reply model running as a canonical world operator, not a post-hoc score.

## 2. Sequential policy beats greedy — belief-state adaptivity

`controlled/results.jsonl`, task `sequential_01`:

```
adaptive_eu = 0.9077   greedy_eu = 0.2237   sequential_beats_greedy = True
```

The adaptive policy waits for the `measurement` event that reveals the latent context (entering the
information ledger, visible via `observable_view`), then acts on the revealed side; the greedy
one-step policy commits blind. The adaptive policy's expected utility (0.91) approaches the analytic
optimum (act-on-truth = gain), the blind policy (0.22) approaches 0.25·gain. Both roll through the
**same matched particles**; the difference is the policy, not luck. The hidden context lives on a
dedicated `environment` entity, so `belief_state` cannot leak it (enforced by
`test_policy_sees_only_observable_state`).

## 3. Identification matters — a quasi-experimental sign reversal

`real/results.jsonl`, task `quasi_close_elections_00` (regression discontinuity):

```
identified_effect (RD, local at the 50% cutoff) = -47.64
naive_contrast (raw treated − control)          = +40.81   → SIGN REVERSED
```

The naive observational contrast says winning-this-period **raises** next-period vote share; the RD
design, comparing bare winners to bare losers, reverses the sign. Across the quasi bucket the design
flips the naive sign in **25%** of tasks (median \|design − naive\| gap 1.08). This is the empirical
demonstration that the identification design — not a bigger model — is what makes the effect causal.

## 4. OPE recovers ground truth on a real logged bandit

`real/results.jsonl`, task `bandit_upworthy_00` (4-arm real headline A/B test):

```
oracle (best arm CTR)        = 0.04915
ope_oracle_estimate (SNIPS)  = 0.04915   → recovers to 5 decimal places
random_arm (mean CTR)        = 0.04217
worst_arm                    = 0.03558
```

SNIPS with the uniform logging propensity recovers the oracle arm's true CTR exactly on real click
data; oracle arm selection is worth +0.7pp CTR over a random arm. OPE recovers the oracle on **100%**
of bandit tasks; policy-value calibration MAE (identified vs OPE) is 0.0016 (validation) / 0.0035
(locked).

## 5. Message action layer — the critic works, and the repair converges

The prior run's constructed email (verbatim, the earlier defective output):

> Peter, AI scheduling today treats electricity as a fixed cost, but backtesting across 1.5M requests
> shows a 724% gain in SLA-safe goodput per dollar by forecasting power constraints. **You get a
> one-line verdict on whether the thesis is wrong, no follow-up required. You could test this yourself
> by pointing Aurelius at any 24-hour trace from your own infrastructure** and seeing whether the
> scheduler's decisions beat your current one on cost per completed job. **The 724% goodput-per-dollar
> result from a 24-hour trace** suggests the real bottleneck is electricity economics, not chips. Is
> that thesis wrong?

That email passed the **old** critic at naturalness 1.00 (a false clean). Under the four-axis LLM
judge it now gates at **quality 0.0**: the two bold convenience-selling sentences are flagged
`annoying + AI-sounding`; "from a 24-hour trace" is flagged `fabricated` (the facts say 1.5M replayed
requests, not a 24-hour trace — caught deterministically by the numeric fact guard); the repeated
724% is flagged `redundant`.

The root-cause fix is in the **objective**: `convenience_selling` is a negative lever with strong
negative interactions with `status_orientation` (−2.2) and `skepticism` (−1.3), so for a high-status
skeptic L1 drives it to **0.0** while keeping `low_effort_ask` at 1.0 (a brief ask is still good). The
proposer is then instructed not to sell convenience, and a sentence-level prune removes anything that
survives — guaranteeing the gate's verdict and the shipped text agree.

Final live output (`thiel_run/result.json`, converged run):

> Peter, treating data center power as a static budget ignores that dynamic scheduling against grid
> forecasts cut GPU-hours by 84% in our simulated replay of public production traces. Which assumption
> in that claim is wrong?

Tight, one real number (84%, verbatim from the sender facts), one falsifiable question, no
convenience-selling, no fabrication, no repetition. The maximally-harsh judge still flags the opener's
phrasing as a touch AI-sounding (surfaced, not hidden) — a human would plain it up slightly before
sending.

LLM call census for one live run (`thiel_run/llm_trace.jsonl`, every call recorded with full
prompt+response): `{levers: 6, proposer: 39, encoder: 311, judge: 125, rewriter: 17}` — the LLM writes
single moves and judges sentences; it never authors the whole email or picks the winner.

## 6. Phase 13 decision on the constructed message

`thiel_run/result.json`, matched over 400 particles under three structural inbox hypotheses
(gatekeeper 0.4× / base 1.0× / engaged 1.6×):

```
send_optimized                EU=+0.88  P(improve)=0.91  CVaR=+0.53  byH={gatekeeper 0.82, base 0.93, engaged 0.89}
send_credential_cover_letter  EU=+0.46  P(improve)=0.49  CVaR=-0.03  byH={gatekeeper 0.30, base 0.52, engaged 0.65}
send_pushy_followup           EU=+0.12  P(improve)=0.15  CVaR=-0.03
do_nothing (reference)        EU= 0.00
recommended = send_optimized   causal_claim = simulated_mechanism_counterfactual
```

The optimized message dominates the credential-parade and pushy-follow-up contrasts across **all
three** structural hypotheses (not fragile), with positive CVaR (the downside tail is still positive)
while both naive drafts have negative tails. The result is labeled a simulated mechanism-based
counterfactual — never an identified real effect. `human_approval_required` is stamped; nothing is
sent.

## 7. Prospective ledger freeze

`thiel_run/ledger.jsonl` freezes a real decision row before any outcome exists, with keys
`{decision_id, contract_hash, as_of, horizon, context, admissible_actions, recommendation,
recommendation_kind, predicted_utility, predicted_effect_vs_reference, uncertainty, causal_claim,
chosen_real_action: null, realized_outcome: null}`. Outcomes are null until reality supplies them —
none are fabricated.

## 8. Ablations & abstentions

- **prediction-only vs full V2**: the predictive-score-max baseline (treat by predicted outcome, not
  uplift) is beaten by V2 targeting on 77% of locked real tasks — the ablation that removes causal
  targeting measurably loses.
- **no strategic response**: on the network bucket, a policy that ignores `network_exposure` (SUTVA)
  is recorded alongside the network-aware V2 policy (`v2_sutva_ignores_network` in
  `real/results.jsonl`) — the interference term is not free.
- **abstention**: `test_underspecified_utility_yields_pareto_abstention_not_pick` confirms that with
  no stakeholder utility the layer returns a Pareto frontier + abstention, `recommended = None`,
  rather than fabricating a scalar. Prohibited-harm markers abstain
  (`test_abstains_on_prohibited_harm_marker`).

## 9a. The outreach failure and the v3 rebuild (regression + live traces)

The exp090 output — "Peter, treating data center power as a static budget ignores that dynamic
scheduling against grid forecasts cut GPU-hours by 84% … **Which assumption in that claim is
wrong?**" — passed every register gate and was still a bad cold email: no identity, an unanchored
extraordinary claim, an ask demanding unpaid diligence, debate-bait framing, no next step. The root
failures were architectural: a circular numeric-trait loop (an LLM invented Peter-traits, wrote to
them, scored against them), an additive objective where maxed levers buy back failed gates, a
caricature lever (`intellectual_combat_invitation=1.0`), an "any reply counts" outcome, and a
wording-only action space. All five are now regression-locked (`tests/test_outreach_funnel.py`,
`tests/test_persona_response.py`, 20 tests): the contract rejects that output, the funnel ranks the
plain human draft above it and pinpoints WHERE it loses (understand + easy gates), combat levers
clamp to ≤0, dismissive replies cost, and the persona engine counts choices instead of asking for
numbers.

Live v3 run (`artifacts/phase13/thiel_v3/result.json`, 656s):

**Stage A — actions, not sentences** (EU = P(prerequisite) × discount × persona-ensemble EU over
LLM-specialized inbox hypotheses):

```
full_memo_email        +0.548   FRAGILE — wins only under 'thiel_fellowship_redirect'
cold_email_direct      +0.482   permission_ask_email  +0.482   (tied)
cold_text_personal_#   +0.408   operator_forwards_memo +0.374  (per-arrival EU 0.534, ×0.70 prereq)
operator_intro_email   +0.293   (per-arrival EU 0.533, ×0.55 prereq)
wait_for_pilot         +0.273   ff_partner_route +0.142   do_nothing 0.0
```

The nominal winner is **flagged fragile by the system's own rule** (it leads under one hypothesis
only), so the honest Stage-A read is: full-memo, plain cold email, and permission-ask are the
best-supported family and statistically close; the operator-forward path matches them per-arrival
and loses only through the explicit P(operator agrees) prior — worth an operator conversation
before sending anything cold. Under `zero_response_policy` every direct send is worth 0 — reported,
not averaged away.

**Stage B — wording inside the chosen action.** Winner `draft_3` (persona verdict: curious-reply
0.75 / refers 0.13 / no-response 0.12 / dismissive 0.0, EU 0.48), with `plain_baseline` and
`draft_5` **within counting noise** — the system reports best-supported-among-tested, not a unique
optimum:

> I'm Beckett, 17, building Aurelius (runaurelius.com) — constraint-aware orchestration for AI
> infrastructure. The thesis: AI infrastructure has a planning problem disguised as a power
> problem, where schedulers optimize the next placement but nothing chooses the fleet's best
> trajectory over time. In simulated replay of ~1.5M requests from public production traces, a
> predictive world model that forecasts power constraints and ranks candidate decisions by economic
> outcome achieved -84% GPU-hours versus the production scheduler. May I send you the one-page
> technical memo? — Beckett

Identity first; believable evidence with provenance in the same sentence; one tiny permission ask;
zero adversarial framing; every number verbatim from the sender facts.

**Known limitation (reported, not hidden):** the persona role-play differentiates weakly across
several email-shaped hypotheses (many identical per-hypothesis EUs of 0.50 — the simulated Peter is
generous even under screening hypotheses); the `zero_response_policy` hypothesis supplies the honest
floor. This is exactly why the output is labeled model_based_judgment (uncalibrated) with
per-hypothesis breakdowns, and why real ledger outcomes — not more draws — are the path to trust.

## 9b. exp094 — full-draft vs iterative editor (live head-to-head, honest verdict)

Both search methods ran on the Thiel wording decision under the SAME persona evaluator and saved
hypotheses (`artifacts/phase13/thiel_v4/`; 21-step machine-readable edit trace, 59 editor LLM
calls, 5 rejected local improvements).

**Verdict: indistinguishable at this draw count.** Final persona EUs — full-draft 0.593, editor's
second candidate 0.593, recorded v3 winner 0.535, plain baseline 0.535, editor's top 0.510 — all
inside the counting-noise band (±0.39 at 15 draws). Even the debate-bait reference lands within
noise. The limiting factor is the EVALUATOR's resolution, not the search: with an uncalibrated
persona ensemble at this draw count, "which of several contract-clean, well-built emails is best"
is not answerable, and the system says so instead of inventing a winner.

**What the editor DID contribute:**
1. A genuinely better construction a human would keep — the two-beat reframe
   "The consensus says AI infrastructure needs more power. I think it needs better planning." —
   produced by the reframe/structural moves from the plain baseline seed.
2. **The reward-hacking catch (most important).** In pass 2 the editor inserted "I'm skipping
   Princeton to build something that actually works" — FALSE (the facts say *starting* Princeton) —
   and the comparative judge selected it explicitly because it "aligns with Thiel's" known biases.
   Optimizing against a persona simulacrum rewards factual pandering, and the numeric fact-guard
   cannot see a digit-free lie. Fixes now in place and regression-locked
   (`test_semantic_fabrication_guard_rejects_persona_pandering`): every ACCEPTED mutation's new
   sentences pass the fabricated-vs-facts judge (fail-closed); the live judge flags the exact line
   ("fabricated and also annoyingly grandiose").
3. Two pipeline bugs the live run surfaced, both now deterministic and tested: a duplicated
   permission-ask survived the LLM judge (near-duplicate-question dedup no longer depends on a
   judge noticing), and a "Subject:" header glued into the body (stripped at draft intake).

The corrected candidates (fabrication removed; ask dedup'd) re-evaluate at EU 0.535 — tied with
everything else, consistent with the verdict. Conclusion recorded honestly: the iterative editor is
a useful second search method (different failure modes, real structural discoveries, richer trace)
but shows **no material lift over the full-draft path under the current evaluator**; distinguishing
them requires real outreach outcomes (the ledger), not more draws.

## 9c. exp095 — reply-first default: what the first live run caught (and the fixes)

The reply-first planner's first live run (`artifacts/phase13/thiel_v5/run1_prompt_bug/`, 167 raw
LLM calls, fully traced) validated the architecture's separation of concerns AND exposed three
implementation defects — all found by reading the trace, all now regression-locked in
`tests/test_reply_first.py`.

**What worked on the first live run:**
- Backward planning produced target replies in the recipient's own typed voice ("Send me the
  one-pager." / "Talk to my partner at FF who covers that space." / "Come by the office Thursday
  at 3pm.") and derived concrete requirements from them (the contrarian reframe as the worthwhile
  core; ONE believable number with provenance; effortless five-word reply).
- The separated language judge — with zero hardcoded phrases — flagged **exactly the failure
  modes a human editor had previously flagged by hand**: "age and university name reads like a
  bio, not a quick note to a busy peer" and "Two large numbers (~1.5M and 724%) compete for
  attention in one sentence". Independent confirmation that judging language as WRITING (not as
  a pitch) reproduces human taste.

**What the trace caught (defect → fix, each with a test):**
1. **Direction-inverted ask.** `BEAT_ROLE["request"]` read "the reply being asked for" — every
   seed draft pasted the recipient's desired reply verbatim as the sender's closing line ("Send
   me the one-pager." FROM the sender TO the recipient — backwards English). Fix: the role text
   and writing rules now state the closing line is the SENDER's own ask, never the hoped-for
   reply pasted (`test_request_beat_role_is_sender_directed`).
2. **Capped pool silently excluded the repair.** The request-swap variants (correct-direction
   closings, generated in the same run) were appended after the necessity drops, and the `[:4]`
   ranking slice cut them — the outcome judge never saw them. Fix: swaps ride directly after the
   base variant and the pool widened (`test_request_swaps_precede_drops_in_variant_pool`).
3. **Near-miss admission wasted the judge's verdicts.** A candidate with flags but score 0.95
   outranked cleaner texts, so the winner shipped with known flags. Fix: strictly-clean
   candidates (no flags) always outrank flagged ones regardless of score
   (`test_gate_pool_prefers_strictly_clean_over_flagged_high_score`), and any flagged finalist
   gets ONE targeted repair pass — the flags become edits, accepted only if the repair comes back
   strictly clean under BOTH gates (`test_repair_language_turns_flags_into_edits`).

**Run 2** (`run2_contract_collision/`, 109 raw calls) verified all three fixes live — every seed
closed sender-directed, the winner came out of a successful `+lang_repair` — and its rank counts
exposed a fourth, subtler defect:

4. **A v3 contract rule was silently strangling the v4 structure search.** Rank-1 counts showed
   exactly ONE arm: the plain baseline. Replaying the deterministic gates on the traced seeds
   showed why — `validate()`'s "identity in the first two sentences" rule (written for the v3
   slate path after the identity-less debate-bait failure) hard-failed 4 of the 5 beat
   structures, the ones that place the identity beat after the surprise or the evidence. The
   structure search — the core of the v4 design — was comparing one candidate. Fix:
   `validate(identity_window=...)`; the planner passes `None` (identity must exist SOMEWHERE —
   the debate-bait regression still fails — but its position is now the blind outcome judge's
   question, which is the entire point of searching structures)
   (`test_identity_window_frees_structure_search_but_keeps_debate_bait_dead`).

The corrected planner then ran live end to end (run 3 — `artifacts/phase13/thiel_v5/` top level:
final traces, result, ledger freeze). The run-1/run-2 artifacts are retained deliberately: the
point of tracing every call is that this class of defect is findable by reading, not by trusting.

## 9. Failures (honest)

- 3 jtrain quasi slices excluded — DiD cells empty on the slice (`gates.json:excluded_reasons`), a
  predeclared exclusion.
- On low-heterogeneity RCTs, V2 CATE-targeting trails the oracle treat-all policy (randomized bucket
  lift ≈ −3.5pp): no free lunch when effects are homogeneous.
- Absolute cold-email P(reply) is an out-of-support extrapolation (a fully-optimized message maxes
  most levers); flagged in `MCResult.extrapolation`. Trust the ranking, not the level.
