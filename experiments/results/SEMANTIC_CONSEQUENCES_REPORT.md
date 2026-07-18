# Semantic world consequences — demos + matched 4-mode evaluation (honest report)

**Phase claim under test.** After this phase, an action executed by a simulated human changes
its own world branch by creating and modifying typed facts, objects, communications (with the
exact content), relationships, commitments, institutional states, and processes; other actors
generate downstream reactions from what actually reached them; numbers change only where the
world is genuinely numerical; the final answer is read from the evolved structured world.
`ACTION_PATHWAY_EFFECTS × pathway_step` survives only as the explicit
`legacy_scalar_pathway_consequences` benchmark arm.

Backend: DeepSeek `deepseek-v4-flash` (decisions AND consequence compilation — the untrusted
proposal path, validated op-by-op). Seeds fixed (SEED=11). All artifacts in
`experiments/results/semantic_*.json`. Claims ladder as before: implemented / mechanically
verified / demonstrated end-to-end / **not** yet a measured predictive improvement.

## RESULTS_PLACEHOLDER

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
