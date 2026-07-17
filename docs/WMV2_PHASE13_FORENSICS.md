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

## 9. Failures (honest)

- 3 jtrain quasi slices excluded — DiD cells empty on the slice (`gates.json:excluded_reasons`), a
  predeclared exclusion.
- On low-heterogeneity RCTs, V2 CATE-targeting trails the oracle treat-all policy (randomized bucket
  lift ≈ −3.5pp): no free lunch when effects are homogeneous.
- Absolute cold-email P(reply) is an out-of-support extrapolation (a fully-optimized message maxes
  most levers); flagged in `MCResult.extrapolation`. Trust the ranking, not the level.
