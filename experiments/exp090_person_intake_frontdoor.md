# The ask-the-user flow, wired into the live front door (`general_world_model`)

The dossier + grounded-inference machinery (EXP-086/089) is now reachable through the one call a user makes.
When a question turns on a **specific individual**, the front door assembles a dossier and — when the evidence
is too thin to infer honestly — **asks the user for their read on that person instead of fabricating a
disposition**. It's on by default and self-disables without an LLM key.

## The flow

`general_world_model(person=True)` wires a `PersonIntake` preflight that runs at the top of
`WorldModel.simulate`:

1. **Identify** (one LLM call): does this question depend mainly on one specific individual's disposition /
   relationship / state — as opposed to an institution, market, or population? If so, who, and which of their
   personal variables decide the outcome?
2. **Assemble a dossier** (`DossierAssembler`): user-supplied `context` first, then message history, then the
   web footprint.
3. **Route on evidence strength:**
   - **thin** (no user context, no lookup) → return `mode: "needs_user_context"` with the specific questions to
     ask — the simulation short-circuits rather than invent a person's mood/openness.
   - **sufficient** → fold the dossier into the context, infer the person's high-leverage variables through
     the three-pillar stack, and run the calibrated simulation on that grounded picture.

## Live behavior (DeepSeek)

| question | routed to | result |
|---|---|---|
| "Will Marcus agree to grab coffee if I text him?" | **ask** | identifies *Marcus*; returns the relationship/personality/history/topic questions |
| "Will US inflation be below 3% at year end?" | **proceed** | not a person (person = None) — normal grounded macro sim |
| "Will Marcus grab coffee?" + *"college roommate, close friend, we text weekly"* | **proceed** | dossier strength 0.65; 3 person-variables inferred from the context |

So a user asking about a specific person automatically gets the ask-the-user experience: the system requests
exactly what only they know, turns it into a dossier, and simulates the person from measured-quality inferred
variables — while general questions are untouched.

## Answering auto-runs the simulation (no separate step)

The ask isn't a dead end. `simulate(question, answers=<your answers>)` folds the answers into the context so
the **same call** proceeds through the dossier → grounded person-variables → calibrated run. Answering *is*
the trigger. `answers` accepts a plain string, a list, or a `{question: answer}` dict; the ask result carries
a `resume_hint` telling the caller exactly how to continue. Live round-trip: first call → `needs_user_context`
(4 questions); second call with the answers → the compiler runs on the user's context, `person = Marcus`,
forecast produced.

## What's wired

- `swm/api/person_intake.py` — `PersonIntake` (identify → dossier → ask/proceed) + `build_person_intake()`
  (live DeepSeek + web + the three-pillar `AnchoredExtractor`).
- `swm/api/world_model.py` — `WorldModel.person_intake` (opt-in) runs the preflight in `simulate`;
  `general_world_model(person=True)` wires it by default (self-disables with no LLM key).
- `tests/test_person_intake.py` — 5 deterministic tests (non-person proceeds, thin asks, context proceeds +
  grounds, short-circuit on ask, no-intake unchanged).
- Dossier gate tuned so one substantive piece of user context is enough to proceed; the gate fires when the
  user has told us nothing personal *and* the public footprint is thin.
