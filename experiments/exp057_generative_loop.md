# EXP-057 — The full generative loop, assembled: one `simulate(question)` end-to-end

The capstone. `swm/api/generative_simulator.py` wires the whole thing into a single call: for an arbitrary
question it (1) **identifies the deciding agents** and maps each one's known + inferred variables from
context (LLM), (2) **instantiates** them as `PersonaAgent`s and assigns each an initial **position** by
having the LLM reason as/about that persona (the generative `position_fn`), (3) runs **`AgentSociety`
forward** so the agents deliberate and interact, and (4) reads the **emergent outcome** with a full audit
trail. Every LLM step is a pluggable backend (production = Anthropic API; here = structured / committed
judgments), the same pattern as `semantic_stance` and `intervention_selector`.

## A. Assembly correctness — the wiring is verifiably right (quantitative, no LLM, no contamination)
Running the assembled `GenerativeSimulator` on the Supreme Court with a **structured** `position_fn`
(justice ideology from prior terms — exactly EXP-055) reproduces the validated result **exactly**:

| | independent (composite) | generative loop |
|---|---|---|
| vote-margin MAE (954 cases) | 0.208 | **0.168** |

The assembled one-call loop **is** the validated agent simulation — the −19% margin win over the composite
is reproduced, independent of any LLM call. So the wiring is correct; swapping in an LLM `position_fn` is
the general path, not a different pipeline.

## B. LLM-persona demonstration — the loop runs end-to-end on an arbitrary question
A 9-member standards committee votes on a strict safety rule. The LLM identifies the agents (chair,
safety-lead, engineers, ops, juniors) with their variables, influence, openness, and conviction, and
assigns positions. The loop then deliberates:

- **Independent (naive count): p 0.477 → FAILS.** The persuadable majority leans slightly against.
- **Simulated (deliberation): p 0.781 → PASSES — an emergent flip.** Vote-share trajectory
  `0.22 → 0.44 → 0.89 → 1.0`: the high-influence, high-conviction chair (3.0) and safety-lead (2.2) pull the
  open, low-conviction engineers across (e.g. `eng_1` 0.38 → 0.77), and consensus forms.

The outcome the composite gets wrong, the coupled society gets — and every step is auditable (each agent's
persona, initial vs final position, influence, and the interaction trajectory). This is the difference the
whole project was chasing: not `sum(ps)/n`, but a society of grounded agents reasoning and interacting
until an outcome falls out.

## What this delivers
The general engine now exists as **one callable**: `GenerativeSimulator.simulate(question)` →
identify agents → map variables → LLM-persona positions → deliberation → emergent, auditable outcome —
general over institutions and populations, production-wired, and verifiably equal to the validated agent
simulation when the LLM step is replaced by a structured stand-in.

## Honest limits (the real work that remains)
- **The LLM steps are demonstrated, not skill-scored.** Assembly correctness is proven with a structured
  `position_fn`; the LLM `position_fn`/`identify_fn` are shown running end-to-end but a *leakage-free skill
  number* for the LLM-driven loop needs post-cutoff questions or the market-consistency/held-out controls
  used in EXP-047 — the same contamination discipline, now applied to a full simulation.
- **Interaction helps where interaction shapes the outcome** (institutional margins, cascades — EXP-055),
  and does not on marginal-dominated population opinion (EXP-053); the loop should route accordingly.
- **Retrieval is stubbed**: `identify_fn`/`position_fn` receive a `context` string; wiring real
  web/document retrieval to fill each agent's variables from "all accessible knowledge" is the next build.

## Reproduce
`python -m experiments.exp057_generative_loop` → `experiments/results/exp057_generative_loop.json`.
`python -m pytest tests/test_generative_simulator.py`.
