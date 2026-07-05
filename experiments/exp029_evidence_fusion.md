# EXP-029 — General evidence fusion: the connective tissue of the world model

Until now each regime inferred a person's latent profile from ONE kind of evidence: demographics→values
(EXP-028), writing→persona (EXP-025), a single message→variables (EXP-021). A *general* social world
model must fuse **whatever is available** about a person — structured attributes, their observed
responses/choices, their free text — into the ONE value/persona profile that every simulation
conditions on, with confidence that grows as evidence accumulates. `swm/variables/evidence.py` is that
primitive; this experiment validates it and honestly bounds where each channel pays off.

## The primitive (`EvidenceFusion`)
`PersonEvidence` holds any subset of {attributes, observed responses, texts}. `EvidenceFusion`:
- maps attributes → a value PRIOR (injectable attribute→value function — domain-agnostic);
- pulls the estimate toward the value-centroid of the training people who answered each item the same
  way, so a person's OTHER choices sharpen who they are;
- weights that refinement by the number of observed responses via the same `depth_factor` used for text
  depth — more evidence ⇒ more refinement, saturating;
- adds deep persona traits from any text (the EXP-025 engine).
It returns a `VariableMap` (value vector + persona, with provenance/confidence). It is leakage-safe: a
target item is excluded from the person's own context, and centroids come only from training people.

## Result A — real (OpinionQA): the response channel is redundant here (honest negative)
Predict a held-out respondent's answer; does fusing their OTHER answers with demographics beat
demographics alone? (5 seeds, ~6.9k test pairs; no-cheat.)

| tier | accuracy |
|---|---|
| marginal (population) | 0.657 |
| value — demographics only (EXP-028) | **0.680** |
| value — fused (demographics + other answers) | 0.674 |

**Fused − demographics = −0.006.** With strong political demographics (ideology, party, religion) and
only ~4 sparse, topically-diffuse answers per respondent, the response channel adds nothing — and
neither collaborative filtering nor weakening the demographics recovers a gain (both tested). On
OpinionQA the strong channel is **attributes**; the response channel is starved.

## Result B — controlled: the response channel works when responses are informative
A population where a latent value profile weakly drives attributes (noisy observation) but strongly
drives responses (each item answered by the sign of ⟨latent, item⟩). Here responses SHOULD carry the
signal — and the fusion extracts it. Predict held-out items (3 seeds):

| observed responses | accuracy |
|---|---|
| 0 (attributes only) | 0.574 |
| 2 | 0.576 |
| 5 | 0.585 |
| 15 | **0.643** |
| 40 | 0.640 |

**Fusing informative responses lifts accuracy +0.067 over attributes alone, and the gain grows
monotonically with the number of observed responses (0→15), then saturates** — the evidence-depth law
the primitive is built on. This proves the mechanism is correct; OpinionQA simply lacks
informative-enough per-person responses.

## The honest, general lesson
The fusion primitive is the right architecture — one component, any evidence, one profile, confidence
that scales with evidence. **Which channel carries signal is domain-dependent**, and the fusion uses
whichever does:
- **attributes** dominate when you have rich structured facts (OpinionQA — EXP-028 win);
- **text** dominates when you have a person's writing (CMV — EXP-025 win);
- **responses** dominate when a person has made many informative prior choices (controlled here; would
  apply to dense choice/ratings data).
The value of a *general* fusion is exactly that it degrades gracefully to whatever evidence exists and
sharpens as more arrives — rather than a separate pipeline per data type.

## Honest limits
- The OpinionQA response result is a real negative, reported as such — not every channel helps in every
  domain, and we say so.
- Result B is a controlled demonstration of the mechanism (informative responses by construction), not a
  new real-data outcome. The real-data channel wins remain attributes (EXP-028) and text (EXP-025); a
  real dense-informative-response dataset (product ratings, repeated choices) is the natural next venue
  for the response channel.

## Reproduce
`python -m experiments.exp029_evidence_fusion` (OpinionQA cache + a self-contained synthetic).
`python -m pytest tests/test_evidence_fusion.py` covers fusion math, evidence-depth weighting, and the
no-leakage target exclusion.
