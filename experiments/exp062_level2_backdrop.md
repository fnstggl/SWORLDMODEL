# EXP-062 — Level 2 + demographic backdrop: do stakeholders feel real public pressure?

EXP-055 modeled the Supreme Court as 9 interacting justice-agents and beat the independent composite on
vote-margin (MAE 0.208 → 0.168) — but *insulated* from the mass public. This adds the missing piece from
the framework: a coarse mean-field **public** behind the named stakeholders, felt in proportion to each
agent's accountability (`public_sensitivity`).

## What was built

- `PersonaAgent.public_sensitivity` and `AgentSociety.public_field` (both default to off — fully backward
  compatible; verified in tests). In the deliberation step, each stakeholder's target is pulled toward the
  public field by their sensitivity — the mass public now exerts pressure on the named agents.

## Scored test on real data

- **Stakeholders**: the real justices (SCDB), ideology estimated from **prior terms only** (leakage-free).
- **Backdrop**: a real **public-mood index** built from GSS attitudes in each Court term (liberal basket,
  as-of the term — no leakage).
- 954 real cases. We sweep `public_sensitivity` and report honestly.

| public_sensitivity | vote-margin MAE | direction acc |
|---|---|---|
| **0.0 (stakeholders only)** | **0.1683** | 0.533 |
| 0.1 | 0.1685 | 0.533 |
| 0.2 | 0.1721 | 0.531 |
| 0.35 | 0.1773 | 0.531 |
| 0.5 | 0.1870 | 0.537 |

**Verdict: the backdrop is neutral-to-harmful on SCOTUS.** The honest reading is twofold and informative:

1. **The justices' own records already price in their public responsiveness.** Ideology estimated from a
   justice's voting history *already* reflects however much they track public mood, so adding an explicit
   public field is largely redundant.
2. **A centrist mood pull is the wrong prior for a court that is often lopsided.** The mood index sits near
   0.5; pulling justices toward it compresses margins toward split decisions, but many cases are near-
   unanimous — so at higher sensitivity it actively *hurts*.

A thin-record split (cases with a newly-appointed justice, weak ideology estimate) did **not** rescue the
backdrop (MAE 0.168 → 0.172) — the interaction among the other justices already carries the case.

## Why this is a useful null, not a dead end

The mechanism is built, correct, and backward-compatible. The measurement says: **when you already have a
stakeholder's behavioral record, an explicit public backdrop adds nothing** — the record subsumes it. The
backdrop is expected to matter where that record is *absent*: a body whose members have no long public
voting history (a new commission, a corporate board, a one-off panel), or an outcome where public pressure
acts through a channel *not* in the members' past votes (an election-year legislature). Those are the cases
to point it at — not a court with decades of per-justice data.

## Tests

Covered in `tests/test_population_simulator.py`: the backdrop pulls stakeholders toward the public field
when `public_sensitivity > 0`, and has exactly zero effect when it's 0 (backward compatibility).
