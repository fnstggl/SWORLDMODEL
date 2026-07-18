# Fixed-v1 semantic consequences — baseline demos + matched 4-mode evaluation

> **STATUS REFRAME.** Everything in this file evaluates
> `fixed_semantic_consequence_policy_v1` — the structured-consequence BASELINE with a
> developer-fixed ontology (closed OBJECT_TYPES, PROCESS_STAGES, primitive catalog, event
> names, institutional candidate menus). It decisively beats the legacy scalar coupling on
> the measured defect (§1), and it is NOT the production architecture: production is
> `generated_actor_mediated_world`, where each scenario GENERATES its own semantic types,
> events, processes, and outcome predicates, and every human response comes from that
> actor's own persistent LLM invocation (see `docs/GENERATED_WORLD.md` and
> `experiments/results/GENERATED_WORLD_REPORT.md`). These artifacts must not be read as
> validation of the generated architecture.

**Baseline claim under test.** An action executed by a simulated human changes its own world
branch by creating and modifying typed facts, objects, communications (with the exact
content), relationships, commitments, institutional states, and processes; other actors
generate downstream reactions from what actually reached them; numbers change only where the
world is genuinely numerical; the final answer is read from the evolved structured world.
`ACTION_PATHWAY_EFFECTS × pathway_step` survives only as the explicit
`legacy_scalar_pathway_consequences` benchmark arm.

Backend: DeepSeek `deepseek-v4-flash` (decisions AND consequence compilation — the untrusted
proposal path, validated op-by-op). Seeds fixed (SEED=11). All artifacts in
`experiments/results/semantic_*.json`. Claims ladder as before: implemented / mechanically
verified / demonstrated end-to-end / **not** yet a measured predictive improvement.

## 1. The matched 4-mode evaluation (the phase's headline)

One settlement scenario (two leaders, 2 mediated rounds each, 8 particles), identical worlds
and seeds in every arm; the contract reads the TYPED world (deal = an agreement signed OR the
negotiation process reaching provisional acceptance; the legacy arm's readout consumes the
same bar it always did). The arms differ ONLY in consequence mode × actor policy.

| Arm | Consequences × actors | P(deal) | bar mean | typed objects | real comms | scalar writes | ops applied | quarantined | wall |
|---|---|---|---|---|---|---|---|---|---|
| A | legacy scalar × persistent qualitative (pre-phase production) | **0.000** | 0.337 | 0 | 0 | 34 | 0 | 0 | 13m |
| B | semantic × numeric policy (actors off) | 0.125 | 0.465 | 20 | 0 | 0 | 44 | 1 | 0.3s |
| C | semantic × stateless qualitative | 0.250 | 0.528 | 310 | 243 | 0 | 731 | 232 | 43m |
| D | semantic × persistent qualitative (**new production default**) | 0.125 | 0.446 | 305 | 235 | 0 | 910 | 214 | 51m |

- **The defect is closed where it was measured.** The previous causal audit proved decisions
  shifted trajectories (max CDF gap 0.50) while the BINARY answer stayed frozen in both arms
  — the scalar coupling was the binding constraint. On the same scenario class, every
  semantic arm moved the binary answer (0.125–0.25) while the matched legacy arm reproduced
  the frozen 0.0. In arm D the deal branch is one whose negotiation process actually reached
  `provisional_acceptance` through staged typed transitions; its bar value (0.571) is the
  DERIVED projection of that stage, not an accumulated 0.04-step.
- **The causal chain is inspectable end-to-end** in the artifacts: the actor's decision text
  → the compiled program → proposal/communication objects carrying the actual words ("I will
  reject the current framework as too vague and propose a specific, binding guarantee
  mechanism before any substantive negotiation begins") → the recipient's own next decision
  on that text → process stage movements → the readout.
- **Do not over-read B vs C vs D.** At n=8 particles the spread 0.125–0.25 is one branch;
  this evaluation demonstrates the consequence ARCHITECTURE, not an actor-policy ranking
  (the 50-case benchmark remains the actor-policy evidence).
- **Cost is real**: qualitative arms spend ~2 LLM calls per decision (decide + compile) and
  reaction chains extend runs (43–51 min vs 13 min legacy). Budgets bound it loudly: 126 (D)
  / 81 (C) late decisions fell back numeric after the 240-call budget, all excluded from
  qualitative aggregation and counted in the reports.

## 2. Four end-to-end demos (DeepSeek, semantic default)

**demo1 — product launch** (6 particles): the founder's decision genuinely SPLITS the worlds —
`launched_publicly 0.5 / not_launched 0.5` — and each launched branch contains the product
launch process, the public statement with the actual announcement text, and the rival's own
subsequent decision taken with that text in their view (262 typed objects, 164 quarantined
untrusted ops recorded).

**demo2 — negotiation with real messages** (6 particles): every branch is a full exchange of
proposals and private communications with substantive content (551 deliveries across the
run); in this draw NO branch reached provisional acceptance → `no_deal 1.0` — an honest
negative sample of the same process the arms measured (arm C/D each found deal branches at
n=8). The typed record shows exactly why each branch stalled (ultimatum/counter-rejection
statuses on the proposal objects).

**demo3 — institutional decision** (4 particles): the CEO's requests enter real board
procedures (54 submissions across reaction chains, 483 quarantined ops — the noisiest run)
but NO branch produced a decided/approve outcome (`not_approved 1.0`): members' reaction
chains kept re-submitting and re-deliberating rather than holding approve as their standing
action at tally time. An honest negative: the fixed-v1 institutional loop closes structurally
(the offline smoke approves cleanly) but under long LLM reaction chains the vote-time
`current_action` convention is brittle — one of the reasons this mode is now the BASELINE
(see the generated actor-mediated architecture below), where institutional aggregation counts
explicit member decisions instead.

**demo4 — individual communication** (Priya, 3 hypotheses × 2 samples): the exact Sunday-
evening message reaches every sample; all six samples decide qualitatively (zero fallbacks,
zero legacy writes); replies are REAL communications (9 deliveries, 11 ops) whose content
preserves the hypothesis-specific stance ("I can do an hour but need this to be the last
Sunday fix unless we change the installer", "I'll explicitly frame this as a coaching
session"). Response distribution: reply_now 100% with modifier/target split; calibrated at
reference level by the committed pack.

## Mechanically guaranteed (1318 tests green, 36 new invariants)

- **Default-on**: `semantic_world_consequences` is the resolved mode for every runtime unless
  `SWM_CONSEQUENCES` explicitly requests the legacy benchmark or dual audit; both scalar
  writers **assert the mode** (invoking them under the semantic default raises), so a silent
  scalar fallback is structurally impossible; every result carries `consequence_report`
  (requested vs actual mode, ops applied by class, objects, deliveries, submissions,
  decisions opened, unsupported semantics, fallbacks with reasons, legacy writes = 0).
- **Closed primitive registry**: ~26 validated executors; numeric minting is quarantined at
  compile (forbidden keys AND forbidden state-target names) and again at execution; ops that
  fail at execution (malformed fields from the untrusted LLM) quarantine loudly — a branch
  can never crash or silently skip.
- **Communications carry the message**: `deliver_information` → private_communication object
  + InformationItem + `message_delivered`; the delivery operator exposes the EXACT text to
  the recipient and opens THEIR decision; the sender's anticipated reaction is stored as
  subjective actor-local state and never executes.
- **Institutions run real procedures**: submissions create typed submission + procedure
  objects; a sole right-holder's submission IS the decision (typed outcome, no fake vote);
  multi-holder submissions open the other members' decisions and schedule the
  `collective_vote` the vote operator consumes; the tally writes the typed outcome onto the
  submission and procedure; an empty tally decides nothing; declared outcome quantities are
  DERIVED projections of decided outcomes only.
- **Processes are stage machines** (product_launch, negotiation, acquisition, institutional
  procedure, regulatory review, adoption, generic): stages advance only along declared
  machines, terminal stages refuse further movement, history is recorded.
- **Numbers only where numerical**: resource moves are conservation-checked; belief/quantity
  scalar deltas from the old `possible_consequences` path never apply in semantic mode;
  `pathway_progress:*` is recomputed as a read-only projection of typed state (fixed point on
  unchanged state; only DECLARED bars).
- **Novel actions**: compile through the LLM path or fall back to deterministic
  ontology→primitive programs; unmodeled actions are marked `semantic_consequence_unmodeled`
  on the delta and the report; a novel action with an ontology anchor does NOT inherit the
  anchor's scalar effects.

## Honest limitations (what this phase does NOT claim)

- **No measured predictive improvement yet.** The demos and the matched evaluation prove the
  causal ARCHITECTURE (typed consequences flow to typed answers); they do not measure
  accuracy against frozen historical intermediate facts. That corpus work (consequence
  accuracy scored the way decision accuracy was scored at n=50) is the declared next
  measurement task, alongside the post-cutoff forward corpus from the previous report.
- **Downstream population/market quantities remain gated on fitted mechanisms.** Population
  responses are OPENED as typed events; adoption/market numbers only emerge where a fitted
  mechanism exists (none is fitted in this phase) — the report records them as opened, not
  resolved. No hand-set adoption numbers were added anywhere.
- **The deterministic compiler is bounded.** Off-ontology actions without LLM compilation
  become communications/observations rather than rich programs; that is the loud-fallback
  design, not a hidden capability claim.
- **LLM-compiled programs are only as specific as the model's proposal.** Validation
  guarantees safety (no minting, no authority violations, referential integrity), not
  richness; quarantine counts in the artifacts show how much was rejected.

## Reproduction

```
PYTHONPATH=. python experiments/semantic_consequences_demo.py smoke          # offline
DEEPSEEK_API_KEY=… PYTHONPATH=. python experiments/semantic_consequences_demo.py demo1
…demo2 / demo3 / demo4 / armA / armB / armC / armD / combine
```
