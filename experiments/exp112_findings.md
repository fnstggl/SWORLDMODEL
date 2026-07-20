# The `invalid_execution_plan` collapse — root cause, fix, verification

The frozen-5 audit (EXP-110) showed 4/5 questions collapse at ensemble compilation
(`execution_failed` / `invalid_execution_plan`, zero executable models). The grounded floor kept a forecast
flowing, but the general-purpose simulation was not actually starting. This fixes that.

## 1. Exactly why each candidate was rejected (EXP-112)

Ran BoJ and Knesset capturing `provenance.structural_ensemble_generation.candidates`. **Every candidate of
both questions (5/5 each) was rejected with the identical reason:**

> `nonexecutable_after_bounded_repair: CompilerExecutionError: qualitative actor policy requires an LLM
> backend and none was supplied; refusing the numeric-psychology substitution (§19)`

The candidates were **well-formed** — 2–5 named decisive actors, 2–4 decisive mechanisms each. They are not
impossible models.

## 2. Genuinely impossible, or over-strict? → OVER-STRICT

The compile-time executability critic `_executability_check` (ensemble_compiler.py) called
`operators_from_plan(plan, llm=None)` — **hardcoded `None`**, even though it receives a real backend. The
default-on qualitative actor policy (`production_actor_policy`) correctly refuses to construct without an
LLM backend (§19: never silently degrade named people to a numeric policy). That refusal is raised **before
the terminal outcome writer is instantiated**, so `operators_from_plan` aborts and the whole plan looks
non-executable — then one bounded repair (also `llm=None`) hits the same wall, and the candidate is
rejected. With every candidate rejected, the ensemble returns `invalid_execution_plan`.

The plan is perfectly executable **at rollout**, where the real backend is present
(`phase8_pipeline.py` calls `operators_from_plan(plan, llm=llm, allow_experimental=True)`). The check was
testing executability with no backend — an over-strict check, not an impossible plan. (This is also why the
collapse is *new*: PR #124/#125 made the qualitative actor policy default-on and strict; the old numeric
path never tripped this wall.)

## 3. The fix — defer runtime-backend operators, repair incomplete plans

`claude/wmv2-grounded-outside-view` (commit `1ff0c09`).

- **`operators_from_plan`** gains `defer_backend_operators` (compile-time check ONLY — never the real
  rollout). When set, an operator that needs a runtime LLM backend not yet bound (the actor policy) is
  recorded as a *deferred* rejection and skipped, instead of aborting instantiation. It is a runtime
  resource, not a plan defect, and it is **not an outcome writer** — so skipping it here cannot hide an
  unrunnable plan.
- **`_executability_check`** now (1) defers backend operators and (2) routes through
  `ensure_outcome_pathway` — the same rollout-viability invariant used at runtime — so a dropped/absent
  outcome writer is **repaired** (re-instantiate the writer, or synthesize the canonical `resolve_outcome`
  pathway) rather than rejected. It matches the rollout's `allow_experimental=True` instantiation. Only a
  genuinely unresolvable plan (an event-time contract that retained no absorbing channel) still fails,
  honestly named.

The real rollout path is unchanged (`defer_backend_operators=False`): it keeps the loud §19 refusal, never
a silent numeric swap.

## 4. Tests

`tests/test_ensemble_executability_defers_backend.py` — in STRICT mode (the suite's conftest sets
`SWM_ALLOW_NUMERIC_BASELINE=1`, which masks the production failure, so the tests `delenv` it):
- the rollout path still refuses the actor policy without a backend (no silent numeric);
- the check defers the actor policy and keeps the terminal outcome writer;
- the check **passes** for a well-formed actor-policy plan (was rejected before the fix);
- a plan that lost its outcome writer is repaired, not rejected;
- a backend-gated live test requiring each of the five frozen questions to produce ≥1 executable model.

129 materialize/ensemble tests still pass.

## 5. Live verification (EXP-110 rerun) — PENDING
Rerunning the frozen-5 full-actor pass on the fixed runtime. Expect each question to now produce ≥1
executable model (`n_models ≥ 1`, operator census populated, `fallback = N`) — the general-purpose
simulation actually starting. The rich-vs-lean comparison follows only after that is confirmed.
