# EXP-113 Trace Completeness Verification

Read-only audit of the committed EXP-113 trace files. No code changed, no question rerun, no
LLM/web calls. Every number below is read from the committed artifacts.

## Method

Per question, cross-checked four independent counters of external LLM calls:
- `n_calls` stored in `experiments/results/exp113_checkpoints/<qid>.json` (harness-side count of backend invocations);
- `metrics.n_llm_calls` (same checkpoint);
- `provenance.lean_v2.budget.calls` (the BudgetLedger, incremented once per successful gateway call);
- row count of `experiments/results/lean_v2_accuracy/<qid>-completion/llm_calls.jsonl` (written from `gateway.rows`).

Then verified call-id contiguity, non-empty exact prompt+reply on every row, the 24 000-char store cap, per-stage agreement between ledger and trace, and searched the runtime for any external call path that bypasses the traced gateway.

## Per-question result

| Question | harness n_calls | metrics n_llm | ledger calls | trace rows | ids 0..N−1 | all prompt+reply non-empty | truncated | retried | failed |
|---|---|---|---|---|---|---|---|---|---|
| Banxico | 33 | 33 | 33 | 33 | yes | yes | 0 | 0 | 0 |
| BoJ | 41 | 41 | 41 | 41 | yes | yes | 0 | 0 | 0 |
| visionOS | 21 | 21 | 21 | 21 | yes | yes | 0 | 0 | 0 |
| Wale | 30 | 30 | 30 | 30 | yes | yes | 0 | 0 | 0 |
| Hormuz | 18 | 18 | 18 | 18 | yes | yes | 0 | 0 | 0 |

All four counters agree exactly on every question. Call ids are contiguous `0..N−1`. Every row
carries a non-empty exact prompt and exact returned reply. The largest stored reply is 15 376
chars (Wale) — below the 24 000-char store cap, so **no prompt or reply is truncated**. Zero
rows are marked `retried`, and `budget.provider_retries = 0` on all five, so no call was retried
and no call failed through both attempts (either would desync the harness count from the gateway
rows; they are equal).

## Per-stage agreement (ledger vs trace) and count reconciliation

| Question | structural_generation | reference_class_grounding | state_generation | consequence_compile | actor_decision | schema_repair | per-stage match |
|---|---|---|---|---|---|---|---|
| Banxico | 1 | 1 | 6 | 0 | 25 | 0 | yes |
| BoJ | 0 | 0 | 4 | 0 | 37 | 0 | yes |
| visionOS | 2 | 1 | 1 | 1 | 16 | 0 | yes |
| Wale | 0 | 0 | 6 | 0 | 24 | 0 | yes |
| Hormuz | 1 | 1 | 1 | 5 | 10 | 0 | yes |

`actor_decision` rows reconcile against the structured manifests: they equal
`unique_decision_contexts + deliberations` (+ challenger contexts where a challenger ran):
- Banxico 25 = 18 unique + 7 deliberations;
- BoJ 37 = 21 unique + 16 deliberations;
- Wale 24 = 15 unique + 9 deliberations;
- Hormuz 10 = 7 unique + 3 deliberations;
- visionOS 16 = 7 unique + 3 deliberations + 6 challenger-fork contexts (challenger triggered).

`schema_format_repair` is 0 everywhere — the deterministic decision-normalization path (strip
phantom observation ids, containment-map votes) replaced every provider schema-repair call.

## Single-path guarantee (no untraced external call)

`gateway.call(stage, prompt)` is the only function that invokes the injected backend. The
backend is invoked at exactly two source lines — `gateway.py:61` (primary) and `gateway.py:67`
(the one bounded retry) — both inside `gateway.call`, which appends the traced row and
increments the ledger on success. Every other LLM-using site in the package routes through it:

```
blueprint.py:202,206   structural_generation / blueprint repair
grounding.py:264,267   reference_class_grounding
states.py:396          state_generation
state_completeness.py:353,435  state recovery + reversal search
mechanisms.py:130,271  observation extraction / regime mapping
consequences.py:233    novel consequence compile
engine.py:895,911,923  actor_decision (+ its staged repair)
deliberation.py:131    deliberation (stage "actor_decision")
```

No lean_v2 module imports a provider client (`deepseek`/`openai`/`anthropic`) or an HTTP client
directly; there is no side channel. `lean_routing.call_with_escalation` exists but is **not**
used by the gateway (the gateway only consults `policy.tier_for`). Therefore every external call
that happened during a run is present in that run's `llm_calls.jsonl`.

## Mass conservation (sanity, per question)

Node-audit weights sum to 1.0 within 6-decimal rounding across up to 1 458 nodes
(Banxico 0.999945, BoJ 0.999746, visionOS 0.999996, Wale 1.000242, Hormuz 1.000008), and the
terminal groups reconcile to each simulation-only probability (e.g. Banxico YES 0.0988 / NO
0.9012 → sim 0.0988; visionOS YES 0.75 / missing_mechanism 0.25 → conditional 1.0).

## The one honest completeness gap — persistent-cache hits on re-run questions

BoJ and Wale were re-run several times during development (to validate the terminal-law fixes).
The persistent compile cache is keyed by question/as_of/evidence/backend/version, so on the
**final** run their immutable compilation artifacts were served from cache
(`blueprint.from_cache = true`, `compile_cache.hits_persistent = 4`):

- **BoJ** final trace has `structural_generation = 0`, `reference_class_grounding = 0`, and no
  blueprint-repair row. Its blueprint, blueprint-repair, grounding, and consequence templates
  were persistent-cache hits.
- **Wale** final trace has `structural_generation = 0`, `reference_class_grounding = 0`, and no
  blueprint-repair row — same situation.
- Banxico, visionOS, Hormuz compiled fresh (`from_cache = false`), so their world-compilation
  and grounding prompts/replies **are** in their traces.

Consequence: for BoJ and Wale the exact **prompts** of the world-compilation, blueprint-repair,
and grounding calls are not present in the final run's `llm_calls.jsonl`. The cached **response
values** are recoverable from `experiments/results/exp113_cache/{boj,wale}/*.json`
(`blueprint_response`, `blueprint_repair_response`, `reference_class_grounding`), but the compile
cache stores only a dependency hash as the key, not the prompt, so the exact prompt text for
those specific calls cannot be reconstructed from the committed artifacts alone.

## Verdict

**The trace is complete for every external call made during each of the five final runs** —
counts agree across four independent counters, ids are contiguous, prompts and replies are exact
and untruncated, and the traced gateway is provably the sole external path.

**It is not a complete record of every call that produced BoJ's and Wale's world state**, because
those two questions' blueprint/repair/grounding calls were persistent-cache hits from earlier
runs and their prompts are not in the final trace (responses are recoverable from the cache dir).
Per the instruction to not claim full completeness unless every check passes, this gap is stated
explicitly rather than glossed: for the world-compilation section of the BoJ and Wale forensic
reports, the reconstruction uses the cached response artifacts and the parsed blueprint stored in
the checkpoint, and flags that the originating prompts are not in-trace.
