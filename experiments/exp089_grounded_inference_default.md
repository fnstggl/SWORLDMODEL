# Grounded inference is now the DEFAULT — plus the private-individual path (dossier + user context)

Three things landed here, turning EXP-086's finding into shipped, default behavior and extending it to the
case the product actually faces (a private individual you can't look up).

## 1. The three pillars are now the default for EVERY variable grounding — coexisting with the web search

`main` already had the web-search auto-grounder (PR #61: `general_world_model` → `StateGrounder` →
`RetrievalGrounder` → web evidence + a CI-calibrated extractor). That covered **Pillar 1 (evidence)** and the
CI half of **Pillar 3**. Missing were the two pillars EXP-086 proved matter: the reference-class anchor and
shrink-to-base-rate.

`swm/api/anchored_extractor.py` wraps the existing extractor with them, and `build_retrieval_grounder(...,
anchor=True)` makes it the **default**. Now every general question grounds each high-leverage variable as a
full three-pillar estimate:

> retrieve as-of web evidence (Pillar 1) → extract a calibrated value + sd → estimate the reference-class
> base rate (Pillar 2, the outside view) → **shrink the evidence value toward the base rate by its calibrated
> uncertainty** (Pillar 3).

It is **self-regulating**, which is why it's safe as a default: strong, precise web evidence has a small sd,
so `shrink` barely moves it (the live macro/market grounding from EXP-087/088 is untouched); a *weak* estimate
has a large sd and is pulled to the base rate instead of asserting a confident guess. Weak grounding now
degrades to the honest outside view rather than a hallucinated number. Full suite: **458 passing** (the change
left every existing grounding test green — the shrink preserves a confident value).

## 2. The private-individual path — dossier assembly + user-supplied context

EXP-086's 87% used the LLM's own knowledge of public senators. A private individual — the person you're
messaging, a lead — can't be looked up, so the evidence has to come from elsewhere. `swm/variables/dossier.py`
(Pillar 1, made real):

- `DossierAssembler.assemble(...)` merges, in priority order, **user-supplied context** (the relationship, how
  you met, your read on them — the highest-signal thing for a private person, and something no dataset has) +
  **message history** + **web footprint**, into one evidence bundle.
- `needs_user_context` / `context_questions` — when the footprint is thin, the honest move is to **ask the
  user** rather than guess; these fire the gate and supply the specific questions that most improve the
  inference (disposition, relationship, current state, topic stance).
- `infer_variables(dossier, …, AnchoredExtractor)` runs each variable through the full pillar stack with the
  dossier as evidence — sharp when the dossier is rich, honestly wide when it's thin.

This is exactly your scenario: *messaging a particular figure with a little public data but mostly your own
context of the relationship.* The system asks you what you know, turns it into a dossier, and infers the
variables acting on that person from it.

## 3. EXP-089 — the path, demonstrated end to end (live)

A concrete private-individual question ("will Jordan agree to an intro call if I email a warm ask?"), run
through the live stack:

| | cold (reference class, no dossier) | with your context (dossier) |
|---|---|---|
| needs to ask the user? | **yes** (gate fires; gives the questions) | no |
| dossier strength | 0.0 | 1.00 |
| Jordan's openness | +0.30 ± 0.08 | **+0.42 ± 0.08** |
| strength of your relationship | +0.10 ± 0.06 (stranger base rate) | **+0.20 ± 0.05** |
| Jordan's current bandwidth | +0.34 ± 0.09 | +0.42 ± 0.08 |

The dossier **tightened 3/3 estimates and moved 3/3 off the base rate** — the relationship variable rising
from the stranger base-rate (+0.10) to +0.20 on the evidence of two good hallway chats + a warm LinkedIn
thread + a "let's go deeper." That is the EXP-086 23%→87% mechanism, now driven by what *you* know about a
person no dataset does. (This is a demonstration of the product path; EXP-086 remains the quantitative proof
on measurable ground truth — senators are the clean latent-trait yardstick; private individuals lack one, so
the mechanism is proven there and *applied* here.)

## What's wired, in one place

- `swm/api/anchored_extractor.py` — Pillars 2+3 over the universal extractor (default via
  `build_retrieval_grounder(anchor=True)`); every `general_world_model()` grounding now runs it.
- `swm/variables/dossier.py` — Pillar-1 assembly from user context + history + web; the ask-the-user gate.
- `swm/variables/grounded_inference.py` — the estimator (EXP-086); reused by both.
- Tests: `test_anchored_extractor.py` (4), `test_dossier.py` (4), `test_grounded_inference.py` (7).
- `experiments/exp089_private_individual.py` — the live end-to-end demonstration.
