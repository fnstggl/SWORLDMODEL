# EXP-067 — the spec validator + repair loop: a linter and test-run for LLM-compiled models

EXP-066 found the compiler's one real failure: the LLM gets the *structure* right but makes numeric bugs in
the *equations* (its inflation model reverted toward ~35% against a bound of 10, pinned to the bound, and
returned P = 1.0). This is the fix — a validator that **simulates the spec and inspects it**, plus a repair
loop that hands the concrete defects back to the LLM, exactly like a linter + test-run before trusting
generated code.

## `swm/api/spec_validator.py`

**`validate(spec)`** — static checks + a dynamic simulate-and-inspect pass:

| check | catches |
|---|---|
| **equilibrium out of bounds / saturates a bound** | the inflation bug — the level a variable is pulled toward lies outside its `[lo,hi]`, so it pins to a bound |
| degenerate / trivial outcome | the interval collapses to a point, or P(event) ∈ {0,1} with no spread |
| event threshold outside support | the event value lies outside the variable's range → P trivially 0/1 |
| value out of bounds · negative uncertainty · volatility too large | malformed variable declarations |
| bad / unparseable equation · undeclared variable | equations that reference things that don't exist or won't evaluate |

The **equilibrium check** is the load-bearing one: for a `generic_scm` variable it root-finds where the
drift is zero (the level the variable is pulled toward) and flags it if that's outside the declared bounds
— catching the bug from first principles, with no knowledge of the answer.

**`ValidatingCompiler`** wraps any compiler: compile → validate → if there are errors, hand the spec + the
concrete issues to the LLM to **repair** → re-validate, up to a few rounds → return a spec that passes or is
flagged unrepaired. Pluggable `repair_fn` (LLM or cached), same pattern as the rest of the system.

## The real bug, caught and repaired (live)

Qwen's actual inflation spec from EXP-066, run through the loop:

- **validator flagged:** `saturates_bound`, `degenerate_outcome`, `trivial_event`
- **buggy forecast:** P = 1.0, interval [10.0, 10.0]  *(the degenerate output)*
- **repair (live, Qwen via HF):** in **one round** the LLM rewrote the equation
  `0.01*(100-CPI) - 0.02*(CPI-3)` → **`-0.02*(CPI - 3)`** — proper mean-reversion toward 3%, inside bounds
- **repaired forecast:** P = 0.92, mean 3.96, interval **[2.28, 5.66]** — sane and non-degenerate

The validator caught the exact defect automatically, and the LLM fixed it when shown the specific issues.

## No false positives · every check fires

- **Clean specs pass:** the EXP-064 incumbent SCM, a committee, and a bracket each return **0 errors** — the
  validator doesn't cry wolf.
- **Battery:** deliberately-broken specs each trigger their check — `equilibrium_out_of_bounds`,
  `event_threshold_outside_support`, `value_out_of_bounds`, `volatility_too_large`.

## Tests — `tests/test_spec_validator.py` (7, all pass)

Catches the inflation bug; clean spec passes; each check fires; the `ValidatingCompiler` repairs to clean in
one round with a repair backend and reports `unclean` with none. Full suite: **261 passed**.

## What this closes

EXP-066 left one gap: autonomous equation authoring is error-prone. That gap is now closed operationally —
the compiler **self-corrects**: it simulates its own spec, catches the degeneracies a person would catch by
running it, and repairs them through the LLM. Combined with EXP-064/065/066, the pipeline is now
*question → compile → **validate & repair** → Monte-Carlo → calibrated distribution* — the generated model
is tested before it's trusted, the same discipline we'd apply to any generated code.
